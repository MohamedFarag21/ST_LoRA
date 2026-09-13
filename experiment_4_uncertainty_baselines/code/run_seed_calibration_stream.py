# -*- coding: utf-8 -*-
"""
Fit calibrators @320x180, EVALUATE @native 1280x720 on the FULL 4536-frame tomato
val set (no subsample), using STREAMING metrics so the 4.18 B-pixel native eval is
tractable and Arrow-free.

Why streaming: (a) hf_evaluate mean_iou builds a pyarrow table that overflows the
2 GB single-array limit on native maps; (b) a 1000-frame subsample gave ~3x too-low
ECE (unrepresentative). Streaming fixes both — full val, O(bins) memory.

Per val frame: one native forward -> binary pseudo-logits at native AND downsampled
to 320. Every calibrator (fit @320) is applied at BOTH resolutions and folded into
streaming accumulators. eval@320-full doubles as a correctness check: it must
reproduce run_seed_calibration.py's res320 numbers (validates the streaming ECE
matches torchmetrics MulticlassCalibrationError).

Run via SLURM (ssl env). No direct python on login nodes.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from transformers import Mask2FormerImageProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
ELORA = "/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/code/bup_20_trials/elora"
sys.path.insert(0, ELORA)
sys.path.insert(0, HERE)

# __main__-only siblings; guarded so StreamBinMetrics imports standalone as a metrics library.
try:                                                                  # noqa: E402
    from run_seed_calibration import (
        load_pepper_fullft, build_seg_probs, to_binary_pseudologits, predict_cache,
        ADE_MEAN, ADE_STD, PEPPER_NUM_LABELS, NUM_BIN,
    )
    from mask2former_lora_train_tomato import TomatoDataset
    from calibrators import Calibrator, NAMES
except ImportError:
    pass


class StreamBinMetrics:
    """Streaming top-label ECE (uniform bins, matches torchmetrics), adaptive ECE
    (equal-mass via fine histogram), mIoU (running confusion), accuracy.

    `num_classes` must match the logit depth: 2 for the binary tomato study, 8 for
    the multi-class pepper study. mIoU is dataset-pooled inter/union per class,
    macro-averaged over the classes PRESENT in gt-or-pred (empty classes are skipped
    rather than counted as 0, matching evaluate/mean_iou's nan-mean semantics)."""

    def __init__(self, n_ece=10, n_fine=4096, num_classes=2):
        self.n_ece, self.n_fine = n_ece, n_fine
        self.num_classes = num_classes
        z = lambda n: torch.zeros(n, dtype=torch.float64)
        self.e_cnt, self.e_conf, self.e_corr = z(n_ece), z(n_ece), z(n_ece)
        self.f_cnt, self.f_conf, self.f_corr = z(n_fine), z(n_fine), z(n_fine)
        self.inter, self.union = z(num_classes), z(num_classes)
        self.correct, self.total = 0.0, 0.0
        # proper scoring rules (float64, UNBINNED -> immune to the ECE scatter_add artifact)
        self.sq_sum, self.pgt_sum, self.nll_sum, self.npix = 0.0, 0.0, 0.0, 0

    @torch.no_grad()
    def update(self, probs, labels):
        # probs (B,C,H,W) float; labels (B,H,W) long
        valid = labels != 255
        conf, pred = probs.max(1)
        conf, pred, lab = conf[valid], pred[valid], labels[valid]
        corr = (pred == lab).double()
        self.correct += corr.sum().item()
        self.total += corr.numel()
        # Brier = mean_px[sum_c p_c^2 - 2 p_gt + 1]; NLL = mean_px[-log p_gt] (float64)
        pv = probs.permute(0, 2, 3, 1)[valid].double()          # (Nvalid, C)
        pgt = pv.gather(1, lab.long().unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        self.sq_sum += (pv * pv).sum().item()
        self.pgt_sum += pgt.sum().item()
        self.nll_sum += (-pgt.log()).sum().item()
        self.npix += pv.shape[0]
        for cnt, cf, cr, nb in ((self.e_cnt, self.e_conf, self.e_corr, self.n_ece),
                                (self.f_cnt, self.f_conf, self.f_corr, self.n_fine)):
            bi = torch.clamp((conf * nb).long(), 0, nb - 1)
            cnt += torch.bincount(bi, minlength=nb).double().cpu()
            cf  += torch.bincount(bi, weights=conf.double(), minlength=nb).cpu()
            cr  += torch.bincount(bi, weights=corr, minlength=nb).cpu()
        # per-class inter/union via bincount (exact, and O(C) GPU syncs -> 0):
        #   inter_c = |pred==c & lab==c|,  union_c = |pred==c| + |lab==c| - inter_c
        C = self.num_classes
        eq = pred == lab
        self.inter += torch.bincount(pred[eq], minlength=C).double().cpu()
        self.union += (torch.bincount(pred, minlength=C)
                       + torch.bincount(lab, minlength=C)
                       - torch.bincount(pred[eq], minlength=C)).double().cpu()

    def _binned_ce(self, cnt, conf, corr):
        N = cnt.sum()
        nz = cnt > 0
        acc = torch.zeros_like(cnt); cf = torch.zeros_like(cnt)
        acc[nz] = corr[nz] / cnt[nz]; cf[nz] = conf[nz] / cnt[nz]
        return ((cnt / N) * (acc - cf).abs()).sum().item()

    def _binned_max(self, cnt, conf, corr):
        """Max calibration error = max_bin |acc-conf| over NON-EMPTY bins (max-norm).
        MECE with equal-width bins (== torchmetrics norm='max' / MCE); MACE with the
        adaptive equal-mass bins."""
        nz = cnt > 0
        if not bool(nz.any()):
            return float("nan")
        acc = corr[nz] / cnt[nz]; cf = conf[nz] / cnt[nz]
        return float((acc - cf).abs().max())

    def _adaptive_bins(self, nbins=10):
        """Collapse the 4096-bin fine histogram into `nbins` equal-mass (adaptive) bins.
        Returns grouped (cnt, conf, corr) so both ACE (mean) and MACE (max) reuse them."""
        N = self.f_cnt.sum()
        if N == 0:
            return None
        cum = torch.cumsum(self.f_cnt, 0)
        edges = [(i + 1) * N / nbins for i in range(nbins)]
        gc, gf, gr = (torch.zeros(nbins, dtype=torch.float64) for _ in range(3))
        g = 0
        for k in range(self.n_fine):
            gc[g] += self.f_cnt[k]; gf[g] += self.f_conf[k]; gr[g] += self.f_corr[k]
            if cum[k] >= edges[g] and g < nbins - 1:
                g += 1
        return gc, gf, gr

    def _ace(self, nbins=10):
        b = self._adaptive_bins(nbins)
        return float("nan") if b is None else self._binned_ce(*b)

    def compute(self):
        # macro-mIoU over classes present in gt-or-pred. Skipping union==0 classes
        # (rather than scoring them 0) matches evaluate/mean_iou's nanmean and matters
        # for 8-class pepper, where rare subtypes can be absent from a small eval set.
        # For binary tomato both classes are always present pooled -> unchanged.
        present = self.union > 0
        miou = (float((self.inter[present] / self.union[present]).mean())
                if bool(present.any()) else float("nan"))
        npx = max(self.npix, 1)
        ab = self._adaptive_bins(10)
        return {"ECE": self._binned_ce(self.e_cnt, self.e_conf, self.e_corr),
                "ACE": self._ace(10),
                "MECE": self._binned_max(self.e_cnt, self.e_conf, self.e_corr),
                "MACE": (float("nan") if ab is None else self._binned_max(*ab)),
                "mIoU": miou,
                "n_classes_present": int(present.sum()),
                "acc": self.correct / max(self.total, 1),
                "Brier": (self.sq_sum - 2.0 * self.pgt_sum + self.npix) / npx,
                "NLL": self.nll_sum / npx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pretrained", default="facebook/mask2former-swin-base-ade-semantic")
    ap.add_argument("--ckpt_root", default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/full_ft")
    ap.add_argument("--root_dir", default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/tomato_esra")
    ap.add_argument("--out_dir", default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/posthoc_calibration/fit320_eval_native")
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--fit_size", type=int, default=320)
    ap.add_argument("--n_fit", type=int, default=1000)
    ap.add_argument("--fit_seed", type=int, default=1234)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_images", type=int, default=None)      # smoke: limit val frames
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

    # ── fit calibrators @ fit_size (reproduces the study's calibrators) ──
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

    # ── stream full val at native + 320 ──
    methods = ["Uncalibrated"] + NAMES
    acc_nat = {m: StreamBinMetrics() for m in methods}
    acc_320 = {m: StreamBinMetrics() for m in methods}
    proc = Mask2FormerImageProcessor(ignore_index=255, reduce_labels=False,
                                     do_resize=False, do_rescale=False,
                                     do_normalize=False, num_labels=PEPPER_NUM_LABELS)
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)])
    val_idx = list(range(len(val_ds)))
    if args.max_images:
        val_idx = val_idx[:args.max_images]
    print(f"[stream] {len(val_idx)} val frames @ native+320", flush=True)

    def score(plog, lab, img, store):
        for m in methods:
            if m == "Uncalibrated":
                cl = plog
            else:
                cl = fitted[m].calibrate(plog, img if m == "LTS" else None)
            store[m].update(F.softmax(cl.float(), dim=1), lab)

    with torch.no_grad():
        for idx in tqdm(val_idx, desc="stream-val"):
            pil_img, sem_np, _ = val_ds[idx]
            W, H = pil_img.size
            pv = proc([tfm(pil_img)], return_tensors="pt")["pixel_values"].to(dev)
            seg8 = build_seg_probs(model(pixel_values=pv), H, W)
            plog_n = to_binary_pseudologits(seg8)                    # (1,2,H,W)
            lab_n = torch.from_numpy((sem_np > 0).astype(np.int64))[None].to(dev)
            img_n = tfm(pil_img)[None].to(dev)
            score(plog_n, lab_n, img_n, acc_nat)
            # 320 versions
            h = args.fit_size; w = int(round(args.fit_size * W / H))
            plog_s = F.interpolate(plog_n, size=(h, w), mode="bilinear", align_corners=False)
            lab_s = F.interpolate(lab_n[None].float(), size=(h, w), mode="nearest").long()[0]
            img_s = F.interpolate(img_n, size=(h, w), mode="bilinear", align_corners=False)
            score(plog_s, lab_s, img_s, acc_320)

    res_nat = {m: acc_nat[m].compute() for m in methods}
    res_320 = {m: acc_320[m].compute() for m in methods}
    for m in methods:
        print(f"  [{m}] native ECE={res_nat[m]['ECE']:.4f} mIoU={res_nat[m]['mIoU']:.4f}"
              f"  | 320 ECE={res_320[m]['ECE']:.4f}", flush=True)

    out = {"seed": args.seed, "fit_size": args.fit_size, "shot": args.shot,
           "n_fit": len(fit_idx), "n_val": len(val_idx), "elapsed_s": time.time() - t0,
           "eval_native": res_nat, "eval_320_validation": res_320}
    outp = os.path.join(seed_dir, "stream_metrics.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
