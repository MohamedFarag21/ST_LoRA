# -*- coding: utf-8 -*-
"""
Pixel-OoD (pepper->tomato) with the fit@320 -> eval@NATIVE 1280x720 protocol.

Anomaly framing (identical to run_seed_ood.py / full_ft_tomato_pixel_ood):
  anomaly (1) = tomato foreground pixel (sem > 0)
  normal  (0) = background
  score       = per-pixel Shannon entropy of the CALIBRATED 2-class probs (nats).

Why streaming: the native eval is 4536*1280*720 ~= 4.18 B pixels. The old OoD code
(compute_pixel_ood_metrics) concatenates every pixel score into one array and calls
sklearn -> ~17 GB / method, infeasible. Binary entropy is bounded [0, ln2], so we
accumulate exact per-label float64 HISTOGRAMS over entropy and derive AUROC (Mann-
Whitney U), AUPR (step AP) and FPR95 from them with O(bins) memory. Verified against
sklearn on a bounded subsample (--validate_n).

Component metrics (sIoU/PPV/MeanF1) need per-frame connected components at a single
global best-threshold, which is not known until the histograms are complete -> a
SECOND native forward pass thresholds each frame and reuses the exact
compute_component_metrics() from ood_eval_comprehensive.py.

Two native forwards/seed (~1h45). Run via SLURM (ssl env). No direct python on login.
"""
import os
import sys
import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torchvision import transforms
from tqdm import tqdm

from transformers import Mask2FormerImageProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
ELORA = "/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/code/bup_20_trials/elora"
sys.path.insert(0, ELORA)
sys.path.insert(0, HERE)

from run_seed_calibration import (                                    # noqa: E402
    load_pepper_fullft, build_seg_probs, to_binary_pseudologits, predict_cache,
    ADE_MEAN, ADE_STD, PEPPER_NUM_LABELS, NUM_BIN,
)
from mask2former_lora_train_tomato import TomatoDataset               # noqa: E402
from calibrators import Calibrator, NAMES, pixel_entropy              # noqa: E402
from ood_eval_comprehensive import (                                  # noqa: E402
    compute_component_metrics, compute_fpr95)
from sklearn.metrics import roc_auc_score, average_precision_score    # noqa: E402

LN2 = math.log(2.0)
# Entropy histogram resolution over [0, ln2]. Real entropy is spiky: most background
# pixels sit in a narrow spike near 0, which is where the 95%-TPR threshold lands, so
# FPR95 quantization error is far worse than smooth synthetic scores suggest (at 16384
# bins: 2.5e-5 synthetic vs 4.5e-3 real -- the size of the TS-vs-uncal gap). AUROC/AUPR
# integrate over all bins and were unaffected (<1e-4). Histograms are tiny; buy bins.
NBINS = 1048576                    # 2^20 -> bin width 6.6e-7 nats
EPS = 1e-12


def component_metrics_fast(gt_mask, pred_mask, iou_thresh=0.25):
    """Exact vectorized equivalent of ood_eval_comprehensive.compute_component_metrics.

    The original scans the whole frame inside nested loops over components
    (O(n_comp * H*W)) — fine at 64x36, hopeless at native 1280x720 where a
    thresholded entropy map yields thousands of blobs. Here every pairwise
    component intersection comes from one contingency pass (O(H*W)).

    Identical adjusted-IoU definition:
        inter(g,p) = |g & p|
        other(g,p) = |p & (gt>0, gt!=g)| = sum_{g'>0,g'!=g} cont[g',p]
        union(g,p) = |g| + |p| - inter - other
    Verified against the original in validate_ood_math.py.
    """
    gt_lab, n_gt = ndimage.label(gt_mask)
    pred_lab, n_pred = ndimage.label(pred_mask)
    if n_gt == 0:
        return float("nan"), float("nan"), float("nan")

    g = gt_lab.ravel().astype(np.int64)
    p = pred_lab.ravel().astype(np.int64)
    gt_sizes = np.bincount(g, minlength=n_gt + 1).astype(np.float64)
    pred_sizes = np.bincount(p, minlength=n_pred + 1).astype(np.float64)

    # sparse contingency over co-occurring (gt_id, pred_id) pairs
    key = g * (n_pred + 1) + p
    uk, uc = np.unique(key, return_counts=True)
    gi = uk // (n_pred + 1)
    pi = uk % (n_pred + 1)
    m = (gi > 0) & (pi > 0)                     # overlapping pairs only
    gi, pi, inter = gi[m], pi[m], uc[m].astype(np.float64)

    # pred pixels covered by ANY gt component -> gives "other_gt" per pair
    pred_in_anygt = np.bincount(pi, weights=inter, minlength=n_pred + 1)
    other = pred_in_anygt[pi] - inter
    union = gt_sizes[gi] + pred_sizes[pi] - inter - other
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)

    best_gt = np.zeros(n_gt + 1, dtype=np.float64)
    np.maximum.at(best_gt, gi, iou)             # 0 for gt comps with no overlap
    best_pred = np.zeros(n_pred + 1, dtype=np.float64)
    np.maximum.at(best_pred, pi, iou)

    matched_gt = best_gt[1:] >= iou_thresh
    tp = int(matched_gt.sum())
    fn = int((~matched_gt).sum())
    fp = int((best_pred[1:] < iou_thresh).sum())
    sious = best_gt[1:][matched_gt]

    siou = float(sious.mean()) if sious.size else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * ppv * rec / (ppv + rec) if (ppv + rec) > 0 else 0.0
    return siou, ppv, f1


class OODHist:
    """Per-label float64 entropy histograms -> exact AUROC/AUPR/FPR95 (to bin res).

    `smax` is the upper bound of the score range, i.e. max entropy = ln(num_class):
    ln2 for the binary tomato study, ln(8) for the 8-class pepper study. Scores are
    binned over [0, smax]; getting this wrong silently clips or wastes resolution.
    """

    def __init__(self, nbins=NBINS, smax=LN2):
        self.nbins = nbins
        self.smax = smax
        self.h_anom = torch.zeros(nbins, dtype=torch.float64)
        self.h_norm = torch.zeros(nbins, dtype=torch.float64)

    @torch.no_grad()
    def update(self, entropy, label):
        # entropy (H,W) float on any device; label (H,W) uint8/long, 1=anomaly
        ent = entropy.reshape(-1)
        lab = label.reshape(-1)
        bi = torch.clamp((ent / self.smax * self.nbins).long(), 0, self.nbins - 1)
        self.h_anom += torch.bincount(bi[lab == 1], minlength=self.nbins).double().cpu()
        self.h_norm += torch.bincount(bi[lab == 0], minlength=self.nbins).double().cpu()

    # -- metrics derived purely from the two histograms --
    def metrics(self):
        a = self.h_anom.numpy(); n = self.h_norm.numpy()
        Na, Nn = a.sum(), n.sum()
        out = {"AUROC": float("nan"), "AUPR": float("nan"), "FPR95": float("nan"),
               "threshold": float("nan"), "n_anom": float(Na), "n_norm": float(Nn)}
        if Na == 0 or Nn == 0:
            return out
        # AUROC = P(score_anom > score_norm) + 0.5 P(tie)  (higher entropy = anomaly)
        cum_norm_below = np.cumsum(n) - n
        out["AUROC"] = float((a * (cum_norm_below + 0.5 * n)).sum() / (Na * Nn))
        # reverse-cumulative: TP/FP when predicting positive for bins >= b
        ca = np.cumsum(a[::-1])[::-1]         # TP(threshold = left edge of bin b)
        cn = np.cumsum(n[::-1])[::-1]         # FP
        tpr = ca / Na
        fpr = cn / Nn
        idxs = np.where(tpr >= 0.95)[0]       # largest b (highest thr) with TPR>=0.95
        out["FPR95"] = float(fpr[idxs.max()]) if idxs.size else 1.0
        # AUPR: step average precision over decreasing threshold (b: high->low)
        denom = ca + cn
        precision = np.divide(ca, denom, out=np.zeros_like(ca), where=denom > 0)
        recall = ca / Na
        r = recall[::-1]; p = precision[::-1]           # recall ascending 0->1
        r_prev = np.concatenate([[0.0], r[:-1]])
        out["AUPR"] = float(np.sum((r - r_prev) * p))
        out["threshold"] = self._best_threshold(a, n, ca, cn, Na)
        return out

    def _best_threshold(self, a, n, ca, cn, Na):
        """Replicate compute_pixel_ood_metrics: max pixel-F1 over thresholds in the
        50th-99th score percentile band (candidate thr = bin left edges)."""
        total = a + n
        N = total.sum()
        cum_below = (np.cumsum(total) - total) / N          # frac with score < edge[b]
        band = (cum_below >= 0.50) & (cum_below <= 0.99)
        if not band.any():
            band = np.ones_like(cum_below, dtype=bool)
        f1 = 2 * ca / (ca + cn + Na)                         # 2TP/(2TP+FP+FN)
        f1b = np.where(band, f1, -1.0)
        b = int(np.argmax(f1b))
        return float(b / self.nbins * self.smax)


def stream_native(model, val_ds, val_idx, proc, tfm, dev, fitted, methods, cb, desc):
    """One native forward pass; calls cb(idx, ents_dict{method:(H,W) dev tensor}, gt_np)."""
    with torch.no_grad():
        for idx in tqdm(val_idx, desc=desc):
            pil, sem_np, _ = val_ds[idx]
            W, H = pil.size
            pv = proc([tfm(pil)], return_tensors="pt")["pixel_values"].to(dev)
            seg8 = build_seg_probs(model(pixel_values=pv), H, W)
            plog = to_binary_pseudologits(seg8)                    # (1,2,H,W)
            img = tfm(pil)[None].to(dev)
            gt = (sem_np > 0).astype(np.uint8)                     # (H,W)
            ents = {}
            for m in methods:
                cl = plog if m == "Uncalibrated" else fitted[m].calibrate(
                    plog, img if m == "LTS" else None)
                ents[m] = pixel_entropy(cl.float())[0]             # (H,W) on dev
            cb(idx, ents, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pretrained", default="facebook/mask2former-swin-base-ade-semantic")
    ap.add_argument("--ckpt_root", default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/full_ft")
    ap.add_argument("--root_dir", default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/tomato_esra")
    ap.add_argument("--out_dir", default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/posthoc_calibration/fit320_ood_native")
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--fit_size", type=int, default=320)
    ap.add_argument("--n_fit", type=int, default=1000)
    ap.add_argument("--fit_seed", type=int, default=1234)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_images", type=int, default=None)       # smoke: limit val frames
    ap.add_argument("--validate_n", type=int, default=150)        # sklearn cross-check frames
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    seed_dir = os.path.join(args.out_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)
    dev = args.device

    ckpt = os.path.join(args.ckpt_root, f"seed_{args.seed}", f"{args.shot}.pt")
    print(f"[load] {ckpt}", flush=True)
    model = load_pepper_fullft(args.pretrained, ckpt, dev)

    train_ds = TomatoDataset(args.root_dir, split="train")
    val_ds = TomatoDataset(args.root_dir, split="val")
    rng = np.random.RandomState(args.fit_seed)
    fit_idx = sorted(rng.choice(len(train_ds), size=min(args.n_fit, len(train_ds)),
                                replace=False).tolist())

    # ── fit calibrators @320 (identical to the ECE study) ──
    fit_cache = predict_cache(model, train_ds, fit_idx, dev, args.fit_size)
    print(f"[fit cache] {tuple(fit_cache['logits'].shape)}", flush=True)
    fitted = {}
    for name in NAMES:
        tc = time.time()
        fitted[name] = Calibrator(name, num_class=NUM_BIN).fit(
            fit_cache, dev, epochs=args.epochs, batch_size=args.batch_size)
        print(f"[fit {name}] {time.time()-tc:.1f}s", flush=True)
    del fit_cache
    if dev == "cuda":
        torch.cuda.empty_cache()

    methods = ["Uncalibrated"] + NAMES
    proc = Mask2FormerImageProcessor(ignore_index=255, reduce_labels=False,
                                     do_resize=False, do_rescale=False,
                                     do_normalize=False, num_labels=PEPPER_NUM_LABELS)
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)])
    val_idx = list(range(len(val_ds)))
    if args.max_images:
        val_idx = val_idx[:args.max_images]
    vN = min(args.validate_n, len(val_idx))
    print(f"[stream] {len(val_idx)} val frames @ native (2 passes); validate_n={vN}", flush=True)

    # ── PASS 1: histograms + bounded sklearn cross-check (Uncalibrated only) ──
    hists = {m: OODHist() for m in methods}
    hist_v = OODHist()                         # restricted to first vN frames (Uncal)
    v_scores, v_labels = [], []

    def cb1(idx, ents, gt):
        gt_t = torch.from_numpy(gt).to(ents["Uncalibrated"].device)
        for m in methods:
            hists[m].update(ents[m], gt_t)
        if idx < vN:
            e = ents["Uncalibrated"]
            hist_v.update(e, gt_t)
            v_scores.append(e.reshape(-1).float().cpu().numpy())
            v_labels.append(gt.reshape(-1))

    tp1 = time.time()
    stream_native(model, val_ds, val_idx, proc, tfm, dev, fitted, methods, cb1,
                  "ood-pass1")
    print(f"[timing] pass1 loop {time.time()-tp1:.1f}s", flush=True)

    tm = time.time()
    results = {m: hists[m].metrics() for m in methods}
    print(f"[timing] hist metrics {time.time()-tm:.1f}s", flush=True)
    thr = {m: results[m]["threshold"] for m in methods}
    for m in methods:
        print(f"[pass1 {m}] AUROC={results[m]['AUROC']:.4f} AUPR={results[m]['AUPR']:.4f} "
              f"FPR95={results[m]['FPR95']:.4f} thr={thr[m]:.4f}", flush=True)

    # cross-check the histogram estimators against sklearn on the bounded subsample
    tv = time.time()
    vs = np.concatenate(v_scores).astype(np.float32)
    vl = np.concatenate(v_labels).astype(np.uint8)
    print(f"[timing] validate concat {time.time()-tv:.1f}s  ({vl.size/1e6:.1f} Mpx)", flush=True)
    tv = time.time(); _auroc = float(roc_auc_score(vl, vs))
    print(f"[timing] sklearn roc_auc_score {time.time()-tv:.1f}s", flush=True)
    tv = time.time(); _aupr = float(average_precision_score(vl, vs))
    print(f"[timing] sklearn average_precision {time.time()-tv:.1f}s", flush=True)
    tv = time.time(); _fpr95 = float(compute_fpr95(vl, vs))
    print(f"[timing] sklearn fpr95 {time.time()-tv:.1f}s", flush=True)
    sk = {"AUROC": _auroc, "AUPR": _aupr, "FPR95": _fpr95}
    hv = hist_v.metrics()
    print(f"[validate n={vN}] sklearn  AUROC={sk['AUROC']:.5f} AUPR={sk['AUPR']:.5f} "
          f"FPR95={sk['FPR95']:.5f}", flush=True)
    print(f"[validate n={vN}] histogram AUROC={hv['AUROC']:.5f} AUPR={hv['AUPR']:.5f} "
          f"FPR95={hv['FPR95']:.5f}", flush=True)
    validation = {"n_frames": vN, "n_pixels": int(vl.size),
                  "sklearn": {k: sk[k] for k in ("AUROC", "AUPR", "FPR95")},
                  "histogram": {k: hv[k] for k in ("AUROC", "AUPR", "FPR95")}}
    del v_scores, v_labels, vs, vl

    # ── PASS 2: component metrics (sIoU/PPV/MeanF1) at each method's best_thresh ──
    comp = {m: {"siou": [], "ppv": [], "f1": []} for m in methods}

    def cb2(idx, ents, gt):
        if gt.sum() == 0:
            return
        for m in methods:
            pred = (ents[m].cpu().numpy() >= thr[m]).astype(np.uint8)
            siou, ppv, f1 = component_metrics_fast(gt, pred)
            if not np.isnan(siou):
                comp[m]["siou"].append(siou)
                comp[m]["ppv"].append(ppv)
                comp[m]["f1"].append(f1)

    stream_native(model, val_ds, val_idx, proc, tfm, dev, fitted, methods, cb2,
                  "ood-pass2")

    for m in methods:
        c = comp[m]
        results[m]["sIoU"] = float(np.mean(c["siou"])) if c["siou"] else float("nan")
        results[m]["PPV"] = float(np.mean(c["ppv"])) if c["ppv"] else float("nan")
        results[m]["MeanF1"] = float(np.mean(c["f1"])) if c["f1"] else float("nan")
        print(f"[pass2 {m}] sIoU={results[m]['sIoU']:.4f} PPV={results[m]['PPV']:.4f} "
              f"MeanF1={results[m]['MeanF1']:.4f}", flush=True)

    out = {"seed": args.seed, "fit_size": args.fit_size, "shot": args.shot,
           "n_fit": len(fit_idx), "n_val": len(val_idx), "nbins": NBINS,
           "elapsed_s": time.time() - t0, "validation": validation,
           "eval_native": results}
    outp = os.path.join(seed_dir, "ood_stream_metrics.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
