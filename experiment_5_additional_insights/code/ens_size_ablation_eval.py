# -*- coding: utf-8 -*-
"""
Ensemble-SIZE ablation eval — FRE vs ST-LoRA on bup20 pepper test93 @native (clean).

For a single (method, seed) we load the snapshot members shot_1..shot_maxk, and evaluate
the CUMULATIVE ensemble of the first k members for every k=1..maxk. Each member is run
exactly ONCE over test93; its per-frame softmax maps are added to a running sum, and after
member k the mean map (sum/k) is scored — so total cost is maxk forward passes, not maxk^2.

Metrics per k (float64, native 1280x720):
  * global : mIoU, ECE, acc, Brier, NLL   (reuse run_uq_shift_native_stream.metrics_from_probs)
  * per-class : IoU, Brier, NLL           (GT-conditioned; matches §11.5/§11.6/§11.7)

Member layout (this ablation's own tree, T_0=10/snapshot_every=10 runs):
  FRE     : <base>/fre/seed_<S>/model_shot_<k>.pt         (load_fullft_snapshot)
  ST-LoRA : <base>/stlora/seed_<S>/model_shot_<k>         (fresh base + PeftModel; final_model cfg)

Writes <out_dir>/<method>_seed<seed>_enssize.json:
  { method, seed, maxk, per_k: { "1": {...}, ..., "10": {...} } }
SLURM only (ssl env).
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)

import calibration_shift_eval as cse                                  # noqa: E402
from run_uq_shift_native_stream import metrics_from_probs            # noqa: E402
from peft import PeftModel                                            # noqa: E402

EPS = 1e-12


def seed_all(s):
    """Reproduce the training-time RNG state so ignore_mismatched_sizes re-inits the frozen
    class_predictor head IDENTICALLY. Training calls pl.seed_everything(seed) right before
    from_pretrained; the ST-LoRA adapter (modules_to_save=null) does NOT store that frozen
    random head, so it MUST be reconstructed from the seed at load time or the member is wrong
    (and unseeded => nondeterministic). No-op effect on FRE (full state_dict overwrites all)."""
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def member_paths(method, seed, base, maxk, start_shot=1):
    sd = os.path.join(base, "fre" if method == "fre" else "stlora", f"seed_{seed}")
    if method == "fre":
        return [os.path.join(sd, f"model_shot_{k}.pt") for k in range(start_shot, maxk + 1)]
    return [os.path.join(sd, f"model_shot_{k}") for k in range(start_shot, maxk + 1)]


def load_member(method, path, pretrained, dev):
    if method == "fre":
        return cse.load_fullft_snapshot(pretrained, path, dev)
    b = cse.load_base_model(pretrained, dev)                          # FRESH base per member
    return PeftModel.from_pretrained(b, path).eval().to(dev)


def per_class_metrics(probs, gt, num):
    """probs (N,C,H,W) float, gt (N,H,W) long -> per-class {iou,brier,nll} lists (None if absent)."""
    inter = np.zeros(num); union = np.zeros(num)
    sq = np.zeros(num); pgt = np.zeros(num); nll = np.zeros(num); npix = np.zeros(num)
    for i in range(probs.shape[0]):
        p = probs[i]                                                  # (C,H,W)
        g = gt[i]                                                     # (H,W)
        pred = p.argmax(0)
        valid = g != 255
        # per-class IoU (argmax)
        for c in range(num):
            gc = valid & (g == c)
            pc = valid & (pred == c)
            inter[c] += float((gc & pc).sum())
            union[c] += float((gc | pc).sum())
        # per-class proper scores (GT-conditioned) float64
        pv = p.permute(1, 2, 0)[valid].double()                      # (Nv,C)
        lab = g[valid].long()
        pg = pv.gather(1, lab.unsqueeze(1)).squeeze(1).clamp_min(EPS)
        sqp = (pv * pv).sum(1)
        idx = lab
        sq += torch.zeros(num, dtype=torch.float64).scatter_add_(0, idx, sqp.cpu()).numpy()
        pgt += torch.zeros(num, dtype=torch.float64).scatter_add_(0, idx, pg.cpu()).numpy()
        nll += torch.zeros(num, dtype=torch.float64).scatter_add_(0, idx, -pg.log().cpu()).numpy()
        npix += torch.bincount(idx.cpu(), minlength=num).double().numpy()
    iou = {}; brier = {}; nl = {}
    names = [cse.ID2LABEL[c] for c in range(num)]
    for c in range(num):
        iou[names[c]] = (inter[c] / union[c]) if union[c] > 0 else None
        if npix[c] > 0:
            brier[names[c]] = (sq[c] - 2.0 * pgt[c] + npix[c]) / npix[c]
            nl[names[c]] = nll[c] / npix[c]
        else:
            brier[names[c]] = None; nl[names[c]] = None
    present = [c for c in range(num) if union[c] > 0]
    macro_iou = float(np.mean([iou[names[c]] for c in present])) if present else None
    return {"iou": iou, "brier": brier, "nll": nl,
            "macro_iou": macro_iou}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fre", "stlora"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--maxk", type=int, default=10)
    ap.add_argument("--start_shot", type=int, default=1,
                    help="first snapshot index to include (drop earlier cycles as burn-in)")
    ap.add_argument("--per_member", action="store_true",
                    help="also score each snapshot in ISOLATION (single-member per-class IoU/cal)")
    ap.add_argument("--base", default=f"{ROOT}/results/lora_paper/ensemble_size_ablation")
    ap.add_argument("--out_dir",
                    default=f"{ROOT}/results/lora_paper/ensemble_size_ablation/eval")
    ap.add_argument("--pretrained", default="facebook/mask2former-swin-base-ade-semantic")
    ap.add_argument("--coco_file",
                    default=f"{ROOT}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    ap.add_argument("--root_dir", default=f"{ROOT}/data/bup_20")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--width", type=int, default=720)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num = cse.NUM_LABELS
    H, W = a.height, a.width
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    paths = member_paths(a.method, a.seed, a.base, a.maxk, a.start_shot)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing members: {missing}")
    print(f"=== ENS-SIZE ablation {a.method} seed {a.seed} shots {a.start_shot}..{a.maxk} "
          f"test93@native ===", flush=True)

    loader = cse.build_loader(a.coco_file, a.root_dir, None, None, a.batch_size, a.num_workers)
    N = len(loader.dataset)
    sum_probs = torch.zeros(N, num, H, W)                             # running member-prob sum
    this_probs = torch.zeros(N, num, H, W) if a.per_member else None  # single-member buffer
    gt = torch.zeros(N, H, W, dtype=torch.long)
    gt_done = False

    per_k = {}                                                        # cumulative ensemble @k
    per_member = {}                                                   # each snapshot in isolation
    for m, path in enumerate(paths, start=1):
        shot = a.start_shot + m - 1
        seed_all(a.seed)                                              # reproduce frozen head (ST-LoRA)
        model = load_member(a.method, path, a.pretrained, dev)
        idx = 0
        for batch in loader:
            pv = batch["pixel_values"].to(dev)
            seg = cse.build_seg_probs(model(pv), H, W)                # (B,C,H,W)
            B = seg.shape[0]
            sum_probs[idx:idx + B] += seg.cpu()
            if a.per_member:
                this_probs[idx:idx + B] = seg.cpu()
            if not gt_done:
                gt[idx:idx + B] = batch["seg_maps"]
            idx += B
        gt_done = True
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        ens = sum_probs / m                                          # mean over first m members
        g = metrics_from_probs(ens, gt, num)                        # ECE/ACE/mIoU/acc/Brier/NLL
        pcm = per_class_metrics(ens, gt, num)
        per_k[str(m)] = {
            "mIoU": g.get("mIoU"), "ECE": g.get("ECE"), "acc": g.get("acc"),
            "Brier": g.get("Brier"), "NLL": g.get("NLL"),
            "macro_iou_present": pcm["macro_iou"],
            "per_class_iou": pcm["iou"], "per_class_brier": pcm["brier"],
            "per_class_nll": pcm["nll"]}
        print(f"[cum k={m:2d} shot={shot:2d}] mIoU={g.get('mIoU'):.4f} "
              f"Brier={g.get('Brier'):.4f} NLL={g.get('NLL'):.4f} ECE={g.get('ECE'):.4f}",
              flush=True)

        if a.per_member:
            gm = metrics_from_probs(this_probs, gt, num)
            pmc = per_class_metrics(this_probs, gt, num)
            per_member[str(shot)] = {
                "mIoU": gm.get("mIoU"), "ECE": gm.get("ECE"), "acc": gm.get("acc"),
                "Brier": gm.get("Brier"), "NLL": gm.get("NLL"),
                "macro_iou_present": pmc["macro_iou"],
                "per_class_iou": pmc["iou"], "per_class_brier": pmc["brier"],
                "per_class_nll": pmc["nll"]}
            print(f"[single  shot={shot:2d}] mIoU={gm.get('mIoU'):.4f} "
                  f"Brier={gm.get('Brier'):.4f} NLL={gm.get('NLL'):.4f} ECE={gm.get('ECE'):.4f}",
                  flush=True)

    out = {"method": a.method, "seed": a.seed, "maxk": a.maxk, "start_shot": a.start_shot,
           "eval": "native_1280x720_test93", "num_class": num,
           "per_k": per_k, "elapsed_s": time.time() - t0}
    if a.per_member:
        out["per_member"] = per_member
    suffix = "" if a.start_shot == 1 else f"_from{a.start_shot}"
    if a.per_member:
        suffix += "_pm"
    outp = os.path.join(a.out_dir, f"{a.method}_seed{a.seed}_enssize{suffix}.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
