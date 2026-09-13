# -*- coding: utf-8 -*-
"""
Phase 2a — pepper 8-class post-hoc calibration UNDER DISTRIBUTION SHIFT.

For one --seed:
  * fit all 6 calibrators on the CLEAN cal(30) set (reuses seed_<S>/cache/cal.pt
    from phase 1 if present, else builds it);
  * for each corruption type x severity (7x5, reused from calibration_shift_eval.py),
    corrupt the full val(33) eval set, predict, apply each calibrator, record
    multi-class ECE / ACE / mIoU / acc.

Calibrators fit ONCE on clean data (standard "calibration under shift" protocol).
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
sys.path.insert(0, f"{ROOT}/code/bup_20_trials/elora")

import pepper_common as pc
from calibrators import Calibrator, NAMES
from calibration_shift_eval import CORRUPTIONS, CORRUPT_FNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", default=pc.OUT_DIR)
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--cal_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="one corruption x 1 severity only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    seed_dir = os.path.join(args.out_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)
    ckpt = os.path.join(pc.MODELS_DIR, f"seed_{args.seed}", f"{args.shot}.pt")
    model = pc.load_pepper_model(ckpt, args.device)

    split = pc.load_split()
    cal_ds = pc.COCOSegDataset(image_ids=split["cal_ids"])
    val_ds = pc.COCOSegDataset(image_ids=split["val_ids"])
    val_idx = list(range(len(val_ds)))

    # ── fit calibrators on the CLEAN cal(30) set (reuse cached cal.pt if present) ──
    cal_cache_p = os.path.join(seed_dir, "cache", "cal.pt")
    if os.path.exists(cal_cache_p):
        fit_cache = torch.load(cal_cache_p)
        print(f"[fit] reuse cached {cal_cache_p} {tuple(fit_cache['logits'].shape)}", flush=True)
    else:
        fit_cache = pc.predict_cache(model, cal_ds, list(range(len(cal_ds))),
                                     args.device, args.cal_size)

    calibrators = {}
    for name in NAMES:
        calibrators[name] = Calibrator(name, num_class=pc.NUM).fit(
            fit_cache, args.device, epochs=args.epochs, batch_size=args.batch_size)
    print(f"[fit] {len(calibrators)} calibrators fit ({time.time()-t0:.1f}s)", flush=True)

    def eval_cache(cache):
        out = {"Uncalibrated": pc.compute_metrics(cache["logits"].float(), cache["labels"])}
        for name in NAMES:
            outs = []
            N = cache["logits"].shape[0]
            for i in range(0, N, args.batch_size):
                lo = cache["logits"][i:i+args.batch_size].float().to(args.device)
                im = cache["image"][i:i+args.batch_size].float().to(args.device) if name == "LTS" else None
                outs.append(calibrators[name].calibrate(lo, im).cpu())
            out[name] = pc.compute_metrics(torch.cat(outs, 0), cache["labels"])
        return out

    results = {"clean": {}}
    clean_cache = pc.predict_cache(model, val_ds, val_idx, args.device, args.cal_size)
    results["clean"]["0"] = eval_cache(clean_cache)
    print(f"[clean] Uncal ECE={results['clean']['0']['Uncalibrated']['ECE']:.4f} "
          f"LTS ECE={results['clean']['0']['LTS']['ECE']:.4f}", flush=True)

    corr_items = list(CORRUPTIONS.items())
    if args.smoke:
        corr_items = corr_items[:1]
    for cname, params in corr_items:
        results[cname] = {}
        cfn = CORRUPT_FNS[cname]
        plist = params[:1] if args.smoke else params
        for lvl, (slabel, p) in enumerate(plist, start=1):
            cache = pc.predict_cache(model, val_ds, val_idx, args.device, args.cal_size,
                                     corrupt_fn=cfn, corrupt_param=p)
            results[cname][str(lvl)] = eval_cache(cache)
            print(f"[{cname} sev{lvl} ({p})] "
                  f"Uncal ECE={results[cname][str(lvl)]['Uncalibrated']['ECE']:.4f} "
                  f"LTS ECE={results[cname][str(lvl)]['LTS']['ECE']:.4f}", flush=True)

    out = {"seed": args.seed, "shot": args.shot, "n_val": len(val_idx),
           "cal_size": args.cal_size, "num_class": pc.NUM,
           "elapsed_s": time.time() - t0, "grid": results}
    outp = os.path.join(seed_dir, "shift_metrics.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
