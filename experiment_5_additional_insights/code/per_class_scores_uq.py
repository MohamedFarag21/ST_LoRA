# -*- coding: utf-8 -*-
"""
Per-class PROPER SCORES (Brier + NLL) for each UQ method on bup20 pepper test93 @native
(clean). Proper-score companion to per_class_iou.py, and the UQ half of the per-class
Brier/NLL table (the calibrator half is per_class_scores_calibrators.py).

Same fresh-base loaders / run_inference as per_class_iou.py. Per-class scores are
CONDITIONED ON THE GROUND-TRUTH class c (over pixels whose true label is c), using the
p_gt-based float64 streaming sums:
    Brier_c = (sum[sum_k p_k^2] - 2*sum[p_c] + n_c) / n_c ,  NLL_c = sum[-log p_c]/n_c .
Both are unbinned proper scoring rules -> immune to ECE's reliability-binning instability.

Writes per_class_scores/<method>_seed<seed>.json:
  { "method","seed","M", "brier":{class:val|null}, "nll":{class:val|null},
    "npix":{class:int}, "n_present","macro_brier","macro_nll", "elapsed_s" }
SLURM only.
"""
import os, sys, json, time, argparse
from types import SimpleNamespace
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")

import calibration_shift_eval as cse                          # noqa: E402
import run_uq_shift_native_stream as ruq                      # noqa: E402


class PerClassProper:
    """Per-class (conditioned on GT) Brier & NLL, float64 streaming."""

    def __init__(self, num_classes):
        self.C = num_classes
        z = lambda: torch.zeros(num_classes, dtype=torch.float64)  # noqa: E731
        self.sq, self.pgt, self.nll, self.npix = z(), z(), z(), z()

    @torch.no_grad()
    def update(self, probs, labels):
        valid = labels != 255
        pv = probs.permute(0, 2, 3, 1)[valid].double()               # (N,C)
        lab = labels[valid].long()                                   # (N,)
        pgt = pv.gather(1, lab.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        sqp = (pv * pv).sum(1)
        dev = pv.device
        C = self.C
        acc_sq = torch.zeros(C, dtype=torch.float64, device=dev).scatter_add_(0, lab, sqp)
        acc_pg = torch.zeros(C, dtype=torch.float64, device=dev).scatter_add_(0, lab, pgt)
        acc_nl = torch.zeros(C, dtype=torch.float64, device=dev).scatter_add_(
            0, lab, -pgt.log())
        self.sq += acc_sq.cpu()
        self.pgt += acc_pg.cpu()
        self.nll += acc_nl.cpu()
        self.npix += torch.bincount(lab.cpu(), minlength=C).double()

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


def make_args(method, seed):
    return SimpleNamespace(
        method=method, seed=seed,
        coco_file=f"{ROOT}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json",
        root_dir=f"{ROOT}/data/bup_20",
        pretrained="facebook/mask2former-swin-base-ade-semantic",
        lora_dir=f"{ROOT}/results/lora_paper/hparam_sweep", config_name="final_model",
        shot_ids=[1, 2, 3, 4],
        fullft_dir=f"{ROOT}/results/lora_paper/full_ft", fullft_shot_ids=[1, 2, 3, 4, 5],
        mcdrop_dir=f"{ROOT}/results/lora_paper/mcdropout", dropout_p=0.25, T=10,
        ddu_dir=f"{ROOT}/results/lora_paper/ddu",
        batch_size=4, num_workers=8, height=1280, width=720)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fullft", "lora", "mcdropout", "ddu"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", default=f"{ROOT}/results/lora_paper/calibration_shift/per_class_scores")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num = cse.NUM_LABELS
    names = [cse.ID2LABEL[c] for c in range(num)]
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    args = make_args(a.method, a.seed)
    get_model, M = ruq.build_get_model(args, dev)
    print(f"[{a.method} s{a.seed}] M={M} clean test93 @native", flush=True)

    loader = cse.build_loader(args.coco_file, args.root_dir, None, None,
                              args.batch_size, args.num_workers)
    probs, gt = cse.run_inference(get_model, M, loader, dev, args.height, args.width)
    torch.cuda.empty_cache()

    acc = PerClassProper(num)
    acc.update(probs.float(), gt)
    brier, nll, npix = acc.compute(names)
    pres_b = [brier[names[c]] for c in range(num) if brier[names[c]] is not None]
    pres_n = [nll[names[c]] for c in range(num) if nll[names[c]] is not None]
    macro_b = sum(pres_b) / len(pres_b) if pres_b else None
    macro_n = sum(pres_n) / len(pres_n) if pres_n else None

    out = {"method": a.method, "seed": a.seed, "M": M,
           "brier": brier, "nll": nll, "npix": npix,
           "n_present": len(pres_b),
           "macro_brier": macro_b, "macro_nll": macro_n,
           "elapsed_s": time.time() - t0}
    outp = os.path.join(a.out_dir, f"{a.method}_seed{a.seed}.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[{a.method} s{a.seed}] macro-Brier={macro_b:.4f} macro-NLL={macro_n:.4f} "
          f"n_present={len(pres_b)}", flush=True)
    for c in range(num):
        b = brier[names[c]]
        n = nll[names[c]]
        print(f"    {names[c]:22s} Brier={'--' if b is None else f'{b:.4f}'}  "
              f"NLL={'--' if n is None else f'{n:.4f}'}", flush=True)
    print(f"[done] {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
