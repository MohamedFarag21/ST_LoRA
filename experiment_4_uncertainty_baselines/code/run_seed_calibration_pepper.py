# -*- coding: utf-8 -*-
"""
Phase 1 — in-distribution 8-class post-hoc calibration of a pepper-trained
Mask2Former (one --seed = one from-scratch model_shot_5).

  1. Load pepper model_shot_5 (8 classes).
  2. Cache 8-class pseudo-logits at 64px for cal(30) [fit], val(33) & test(93) [eval].
  3. Fit each of the 6 calibrators on the cal cache; apply to each eval split.
  4. Report multi-class ECE / ACE / mIoU / acc for uncalibrated + 6 methods.

Run via SLURM (ssl env). No direct python.
"""
import os
import sys
import json
import time
import argparse
import torch

ROOT = "/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204"
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration_pepper")
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")

import pepper_common as pc
from calibrators import Calibrator, NAMES


def eval_all(fit_cache, eval_caches, device, epochs, batch_size):
    """Fit 6 calibrators on fit_cache; return {split: {method: metrics}}."""
    results = {s: {} for s in eval_caches}
    # uncalibrated
    for s, c in eval_caches.items():
        results[s]["Uncalibrated"] = pc.compute_metrics(c["logits"].float(), c["labels"])
        print(f"[Uncalibrated/{s}] {results[s]['Uncalibrated']}", flush=True)
    for name in NAMES:
        tc = time.time()
        cal = Calibrator(name, num_class=pc.NUM).fit(
            fit_cache, device, epochs=epochs, batch_size=batch_size)
        for s, c in eval_caches.items():
            outs = []
            N = c["logits"].shape[0]
            for i in range(0, N, batch_size):
                lo = c["logits"][i:i+batch_size].float().to(device)
                im = c["image"][i:i+batch_size].float().to(device) if name == "LTS" else None
                outs.append(cal.calibrate(lo, im).cpu())
            cal_logits = torch.cat(outs, 0)
            results[s][name] = pc.compute_metrics(cal_logits, c["labels"])
            if name == "Meta":
                results[s][name]["meta_threshold"] = cal.meta_threshold
        print(f"[{name}] " + " | ".join(f"{s}:{results[s][name]['ECE']:.4f}" for s in eval_caches)
              + f"  ({time.time()-tc:.1f}s)", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", default=pc.OUT_DIR)
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--cal_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--no_test", action="store_true", help="skip the 93-frame test eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    seed_dir = os.path.join(args.out_dir, f"seed_{args.seed}")
    cache_dir = os.path.join(seed_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    split = pc.load_split()
    ckpt = os.path.join(pc.MODELS_DIR, f"seed_{args.seed}", f"{args.shot}.pt")
    print(f"[load] {ckpt}", flush=True)
    model = pc.load_pepper_model(ckpt, args.device)

    cal_ds = pc.COCOSegDataset(image_ids=split["cal_ids"])
    val_ds = pc.COCOSegDataset(image_ids=split["val_ids"])
    test_ds = pc.COCOSegDataset(split="test")
    print(f"[data] cal={len(cal_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    def cache_of(ds):
        return pc.predict_cache(model, ds, list(range(len(ds))), args.device,
                                args.cal_size, args.max_images)

    fit_cache = cache_of(cal_ds)
    eval_caches = {"val": cache_of(val_ds)}
    if not args.no_test:
        eval_caches["test"] = cache_of(test_ds)
    torch.save(fit_cache, os.path.join(cache_dir, "cal.pt"))
    for s, c in eval_caches.items():
        torch.save(c, os.path.join(cache_dir, f"{s}.pt"))
    print(f"[cache] cal{tuple(fit_cache['logits'].shape)} "
          + " ".join(f"{s}{tuple(c['logits'].shape)}" for s, c in eval_caches.items()), flush=True)

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    results = eval_all(fit_cache, eval_caches, args.device, args.epochs, args.batch_size)

    out = {"seed": args.seed, "shot": args.shot, "cal_size": args.cal_size, "num_class": pc.NUM,
           "n_cal": len(cal_ds), "n_val": len(val_ds),
           "n_test": (0 if args.no_test else len(test_ds)),
           "elapsed_s": time.time() - t0, "results": results}
    outp = os.path.join(seed_dir, "metrics.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
