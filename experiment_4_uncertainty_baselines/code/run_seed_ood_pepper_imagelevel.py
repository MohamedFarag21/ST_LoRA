# -*- coding: utf-8 -*-
"""
PEPPER post-hoc calibrators — IMAGE-LEVEL OoD (per-seed).

Companion to run_seed_ood_pepper_stream.py (which is PIXEL-level). Here each frame
is reduced to ONE scalar: the mean per-pixel predictive entropy of the (calibrated)
8-class probs. Higher mean-entropy = more OoD. This is the image-level analog of the
UQ methods' NegEnt scorer, so all methods are compared on the SAME image-level score.

Detection task, per method (Uncalibrated + 6 calibrators):
  ID  = pepper `test` frames                         (label 0)
  OoD = growliflower (cauliflower, far-OoD)           (label 1)  -> AUROC/AUPR/FPR95
  OoD = tomato val   (near-OoD)                       (label 1)  -> AUROC/AUPR/FPR95

Tomato is randomly sub-sampled to --tomato_max (seed 0) to match the UQ image-level
runs (image_ood_extra.build_tomato_ood_loader uses the same RandomState(0) choice).

Writes seed_<S>/ood_imagelevel_metrics.json. Run via SLURM (ssl env).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")
sys.path.insert(0, f"{ROOT}/code/bup_20_trials/elora")

import pepper_common as pc                                            # noqa: E402
import pepper_stream_common as psc                                    # noqa: E402
from run_seed_ood_pepper import GrowliRaw                             # noqa: E402  (palette-safe)
from mask2former_lora_train_tomato import TomatoDataset               # noqa: E402
from ood_eval_comprehensive import compute_fpr95                      # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score    # noqa: E402


def collect_image_scores(model, ds, indices, proc, dev, fitted, methods, name):
    """Per-image MEAN entropy for every method over `indices` of `ds`."""
    scores = {m: [] for m in methods}

    def cb(i, ents, gt):
        for m in methods:
            scores[m].append(float(ents[m].mean()))   # (H,W) entropy -> scalar

    psc.stream_native(model, ds, indices, proc, dev, fitted, methods, cb,
                      f"{name}-imglvl", binary_anomaly=True)
    return {m: np.asarray(scores[m], dtype=np.float64) for m in methods}


def image_metrics(id_s, ood_s):
    """AUROC/AUPR/FPR95 with score = mean-entropy (higher = more OoD, label 1)."""
    lab = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    sc  = np.concatenate([id_s, ood_s])
    return {"AUROC": float(roc_auc_score(lab, sc)),
            "AUPR":  float(average_precision_score(lab, sc)),
            "FPR95": float(compute_fpr95(lab, sc))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir",
                    default=f"{ROOT}/results/posthoc_calibration_pepper/imagelevel_ood")
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--fit_size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--tomato_max", type=int, default=1200)
    ap.add_argument("--subsample_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    dev = args.device
    seed_dir = os.path.join(args.out_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)

    ckpt = os.path.join(pc.MODELS_DIR, f"seed_{args.seed}", f"{args.shot}.pt")
    print(f"[load] {ckpt}", flush=True)
    model = pc.load_pepper_model(ckpt, dev)

    sp = pc.load_split()
    cal_ds = pc.COCOSegDataset(image_ids=sp["cal_ids"])
    fitted, _ = psc.fit_calibrators_320(model, cal_ds, dev, args.fit_size,
                                        args.epochs, args.batch_size)
    proc = psc.native_processor()
    methods = psc.METHODS                                  # Uncalibrated + 6 calibrators

    # ── datasets ─────────────────────────────────────────────────────────────
    id_ds    = pc.COCOSegDataset(image_ids=sp["test_ids"])   # ID = pepper test
    growli   = GrowliRaw()                                   # far-OoD
    tomato   = TomatoDataset(pc.TOMATO_ROOT, split="val")    # near-OoD

    id_idx  = list(range(len(id_ds)))
    gr_idx  = list(range(len(growli)))
    n_tom   = len(tomato)
    if args.tomato_max and n_tom > args.tomato_max:
        rng = np.random.RandomState(args.subsample_seed)
        tom_idx = sorted(rng.choice(n_tom, size=args.tomato_max, replace=False).tolist())
    else:
        tom_idx = list(range(n_tom))
    print(f"[sets] ID(pepper test)={len(id_idx)}  growli={len(gr_idx)}  "
          f"tomato={len(tom_idx)}/{n_tom}", flush=True)

    id_sc  = collect_image_scores(model, id_ds,  id_idx,  proc, dev, fitted, methods, "pepperID")
    gr_sc  = collect_image_scores(model, growli, gr_idx,  proc, dev, fitted, methods, "growli")
    tom_sc = collect_image_scores(model, tomato, tom_idx, proc, dev, fitted, methods, "tomato")

    out_metrics = {}
    for m in methods:
        far  = image_metrics(id_sc[m], gr_sc[m])
        near = image_metrics(id_sc[m], tom_sc[m])
        out_metrics[m] = {"growliflower": far, "tomato": near,
                          "id_mean_entropy":  float(id_sc[m].mean()),
                          "growli_mean_entropy": float(gr_sc[m].mean()),
                          "tomato_mean_entropy": float(tom_sc[m].mean())}
        print(f"[{m:13s}] growli  AUROC={far['AUROC']:.4f} FPR95={far['FPR95']:.4f}   "
              f"tomato AUROC={near['AUROC']:.4f} FPR95={near['FPR95']:.4f}", flush=True)

    out = {"seed": args.seed, "shot": args.shot, "level": "image",
           "score": "mean per-pixel entropy (8-class, calibrated)",
           "id_dataset": "pepper_test", "n_id": len(id_idx),
           "n_growli": len(gr_idx), "n_tomato": len(tom_idx),
           "num_class": pc.NUM, "elapsed_s": time.time() - t0,
           "methods": out_metrics}
    outp = os.path.join(seed_dir, "ood_imagelevel_metrics.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
