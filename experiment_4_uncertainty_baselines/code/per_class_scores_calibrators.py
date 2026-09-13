# -*- coding: utf-8 -*-
"""
Per-class PROPER SCORES (Brier + NLL) for the 7 post-hoc calibration methods
(Uncalibrated + TS/Logistic/Dirichlet/LTS/Meta/Selective) on PEPPER test93 @NATIVE
(clean), fit@320 -> eval@native.

Proper-score analog of per_class_iou_calibrators.py. We prefer Brier/NLL over per-class
ECE: both are UNBINNED float64 proper scoring rules, so they are immune to the reliability-
binning instability that inflates ECE (see reference_torchmetrics_float32_ece and the
"ECE mis-ranked LoRA" finding). Definition — per-class scores are CONDITIONED ON THE
GROUND-TRUTH class c, i.e. computed over the pixels whose true label is c, using exactly
the p_gt-based sums of StreamBinMetrics (which for gt=c means p_gt = p_c):
    Brier_c = (sum_px[sum_k p_k^2] - 2*sum_px[p_c] + n_c) / n_c
    NLL_c   =  sum_px[-log p_c] / n_c                     (p_c clamped to 1e-12)
This mirrors per-class IoU ("how well are the pixels that truly belong to class c handled").

Writes <out_dir>/<Method>_seed<seed>.json:
  { "method","seed","M":1, "brier":{class:val|null}, "nll":{class:val|null},
    "npix":{class:int}, "n_present", "macro_brier","macro_nll", "elapsed_s" }
SLURM only (ssl env). No direct python.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")
sys.path.insert(0, f"{ROOT}/code/bup_20_trials/elora")

import pepper_common as pc                                            # noqa: E402
import pepper_stream_common as psc                                    # noqa: E402


class PerClassProper:
    """Per-class (conditioned on GT) Brier & NLL, float64 streaming. Same p_gt-based
    sums as StreamBinMetrics, but scatter-added into per-ground-truth-class buckets."""

    def __init__(self, num_classes):
        self.C = num_classes
        z = lambda: torch.zeros(num_classes, dtype=torch.float64)  # noqa: E731
        self.sq, self.pgt, self.nll, self.npix = z(), z(), z(), z()

    @torch.no_grad()
    def update(self, probs, labels):
        # probs (B,C,H,W) float; labels (B,H,W) long
        valid = labels != 255
        pv = probs.permute(0, 2, 3, 1)[valid].double()               # (N,C)
        lab = labels[valid].long()                                   # (N,)
        pgt = pv.gather(1, lab.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        sqp = (pv * pv).sum(1)                                        # (N,)
        dev = pv.device
        C = self.C
        idx = lab
        acc_sq = torch.zeros(C, dtype=torch.float64, device=dev).scatter_add_(0, idx, sqp)
        acc_pg = torch.zeros(C, dtype=torch.float64, device=dev).scatter_add_(0, idx, pgt)
        acc_nl = torch.zeros(C, dtype=torch.float64, device=dev).scatter_add_(
            0, idx, -pgt.log())
        self.sq += acc_sq.cpu()
        self.pgt += acc_pg.cpu()
        self.nll += acc_nl.cpu()
        self.npix += torch.bincount(idx.cpu(), minlength=C).double()

    def compute(self, names):
        brier, nll, npix = {}, {}, {}
        for c in range(self.C):
            n = self.npix[c].item()
            npix[names[c]] = int(n)
            if n > 0:
                brier[names[c]] = (self.sq[c].item() - 2.0 * self.pgt[c].item() + n) / n
                nll[names[c]] = self.nll[c].item() / n
            else:
                brier[names[c]] = None
                nll[names[c]] = None
        return brier, nll, npix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir",
                    default=f"{ROOT}/results/posthoc_calibration_pepper/per_class_scores")
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--fit_size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_images", type=int, default=None, help="smoke cap")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    t0 = time.time()
    dev = a.device
    num = pc.NUM
    names = [pc.ID2LABEL[c] for c in range(num)]
    os.makedirs(a.out_dir, exist_ok=True)

    ckpt = os.path.join(pc.MODELS_DIR, f"seed_{a.seed}", f"{a.shot}.pt")
    print(f"[load] {ckpt}", flush=True)
    model = pc.load_pepper_model(ckpt, dev)

    sp = pc.load_split()
    cal_ds = pc.COCOSegDataset(image_ids=sp["cal_ids"])
    test_ds = pc.COCOSegDataset(image_ids=sp["test_ids"])
    print(f"[split] cal={len(cal_ds)} test={len(test_ds)}", flush=True)

    # fit the 6 calibrators once on clean cal(30) @320 (needs grad -> outside no_grad)
    fitted, _ = psc.fit_calibrators_320(model, cal_ds, dev, a.fit_size,
                                        a.epochs, a.batch_size)
    proc = psc.native_processor()
    tfm = pc._tfm()
    methods = psc.METHODS                                            # Uncalibrated + 6

    idxs = list(range(len(test_ds)))
    if a.max_images:
        idxs = idxs[:a.max_images]

    accs = {m: PerClassProper(num) for m in methods}
    with torch.no_grad():
        for idx in tqdm(idxs, desc=f"test93@native s{a.seed}"):
            pil, sem_np, _ = test_ds[idx]
            W, H = pil.size
            pv = proc([tfm(pil)], return_tensors="pt")["pixel_values"].to(dev)
            seg = pc.build_seg_probs(model(pixel_values=pv), H, W)   # (1,NUM,H,W)
            plog = pc.to_pseudologits(seg)
            img = tfm(pil)[None].to(dev)                             # native, for LTS
            lab = torch.from_numpy(sem_np.astype(np.int64))[None].to(dev)
            for m in methods:
                cl = plog if m == "Uncalibrated" else fitted[m].calibrate(
                    plog, img if m == "LTS" else None)
                probs = F.softmax(cl.float(), dim=1)
                accs[m].update(probs, lab)

    for m in methods:
        brier, nll, npix = accs[m].compute(names)
        pres_b = [brier[names[c]] for c in range(num) if brier[names[c]] is not None]
        pres_n = [nll[names[c]] for c in range(num) if nll[names[c]] is not None]
        macro_b = sum(pres_b) / len(pres_b) if pres_b else None
        macro_n = sum(pres_n) / len(pres_n) if pres_n else None
        out = {"method": m, "seed": a.seed, "M": 1,
               "brier": brier, "nll": nll, "npix": npix,
               "n_present": len(pres_b),
               "macro_brier": macro_b, "macro_nll": macro_n,
               "elapsed_s": time.time() - t0}
        outp = os.path.join(a.out_dir, f"{m}_seed{a.seed}.json")
        json.dump(out, open(outp, "w"), indent=2)
        print(f"[{m:12s} s{a.seed}] macro-Brier={macro_b:.4f} macro-NLL={macro_n:.4f} "
              f"n_present={len(pres_b)}", flush=True)
    print(f"[done] {len(methods)} methods, seed {a.seed} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
