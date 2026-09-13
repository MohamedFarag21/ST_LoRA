# -*- coding: utf-8 -*-
"""
PEPPER (in-domain, 8-class) post-hoc calibration UNDER DISTRIBUTION SHIFT,
fit@320 -> eval@NATIVE 1280x720 (the fit320_native layer).

This is the native-resolution counterpart of run_seed_shift_pepper.py (which fits AND
evaluates at 64px). Here:
  * calibrators are fit ONCE on the CLEAN cal(30) set cached at fit_size=320 (identical
    to the calibration/OoD fit320_native runners, via psc.fit_calibrators_320);
  * the full val(33) eval set is corrupted (7 types x 5 severities, reused from
    calibration_shift_eval.py) and streamed at NATIVE resolution, one forward/frame;
  * per corruption x severity x method we record float64 ECE / ACE / mIoU / acc through
    the same StreamBinMetrics used by the native calibration runner (torchmetrics'
    float32 bin accumulation corrupts ECE above ~16.7M px/bin; val33 native = 30.4M px,
    already over that line -- see reference_torchmetrics_float32_ece).

Corruption semantics mirror pepper_common.predict_cache EXACTLY: corrupt the PIL, and if
the corruption changed the geometry (rotation / translation) resize back to the original
native size so the model sees a native frame; the GT label is left in its original
orientation (this is the established shift protocol -- rotation is a shift that breaks
task alignment, driving accuracy down and testing calibration under high error).

Writes seed_<S>/shift_metrics_native.json. Run via SLURM (ssl env). No direct python.
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
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")
sys.path.insert(0, f"{ROOT}/code/bup_20_trials/elora")

import pepper_common as pc                                            # noqa: E402
import pepper_stream_common as psc                                    # noqa: E402
from run_seed_calibration_stream import StreamBinMetrics              # noqa: E402
from calibration_shift_eval import CORRUPTIONS, CORRUPT_FNS           # noqa: E402


@torch.no_grad()
def eval_corrupted(model, ds, indices, proc, tfm, dev, fitted, methods,
                   corrupt_fn, corrupt_param, desc):
    """Stream val(33) at native under one (corruption, severity); return per-method
    float64 metrics. corrupt_fn=None -> the clean pass."""
    accs = {m: StreamBinMetrics(num_classes=pc.NUM) for m in methods}
    for idx in tqdm(indices, desc=desc, leave=False):
        pil, sem_np, _ = ds[idx]
        W0, H0 = pil.size
        if corrupt_fn is not None:
            pil = corrupt_fn(pil, corrupt_param)
            if pil.size != (W0, H0):                       # rotation/translation
                pil = pil.resize((W0, H0), Image.BILINEAR)
        W, H = pil.size
        pv = proc([tfm(pil)], return_tensors="pt")["pixel_values"].to(dev)
        seg = pc.build_seg_probs(model(pixel_values=pv), H, W)        # (1,NUM,H,W)
        plog = pc.to_pseudologits(seg)                               # (1,NUM,H,W)
        img = tfm(pil)[None].to(dev)                                 # native, for LTS
        lab = torch.from_numpy(sem_np.astype(np.int64))[None].to(dev)  # (1,H,W)
        for m in methods:
            cl = plog if m == "Uncalibrated" else fitted[m].calibrate(
                plog, img if m == "LTS" else None)
            probs = F.softmax(cl.float(), dim=1)                     # (1,NUM,H,W)
            accs[m].update(probs, lab)
    return {m: accs[m].compute() for m in methods}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir",
                    default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native")
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--fit_size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_images", type=int, default=None,
                    help="smoke: cap val frames streamed per (corruption,severity)")
    ap.add_argument("--smoke", action="store_true",
                    help="one corruption x 1 severity only")
    ap.add_argument("--eval_split", default="val", choices=["val", "test"],
                    help="which held-out split to corrupt & evaluate (fit always on cal30)")
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
    eval_ids = sp["val_ids"] if args.eval_split == "val" else sp["test_ids"]
    val_ds = pc.COCOSegDataset(image_ids=eval_ids)      # name kept; may hold test ids
    print(f"[split] cal={len(cal_ds)} eval({args.eval_split})={len(val_ds)}", flush=True)

    # ── fit calibrators once on CLEAN cal(30) @320 (same as the other fit320_native runs)
    fitted, _ = psc.fit_calibrators_320(model, cal_ds, dev, args.fit_size,
                                        args.epochs, args.batch_size)
    proc = psc.native_processor()
    tfm = pc._tfm()
    methods = psc.METHODS

    val_idx = list(range(len(val_ds)))
    if args.max_images:
        val_idx = val_idx[:args.max_images]

    # ── clean baseline ──
    results = {"clean": {}}
    results["clean"]["0"] = eval_corrupted(model, val_ds, val_idx, proc, tfm, dev,
                                           fitted, methods, None, None, "clean@native")
    print(f"[clean] Uncal ECE={results['clean']['0']['Uncalibrated']['ECE']:.4f} "
          f"LTS ECE={results['clean']['0']['LTS']['ECE']:.4f} "
          f"Dirichlet ECE={results['clean']['0']['Dirichlet']['ECE']:.4f}", flush=True)

    # ── 7 x 5 corruption grid ──
    corr_items = list(CORRUPTIONS.items())
    if args.smoke:
        corr_items = corr_items[:1]
    for cname, params in corr_items:
        results[cname] = {}
        cfn = CORRUPT_FNS[cname]
        plist = params[:1] if args.smoke else params
        for lvl, (slabel, p) in enumerate(plist, start=1):
            results[cname][str(lvl)] = eval_corrupted(
                model, val_ds, val_idx, proc, tfm, dev, fitted, methods,
                cfn, p, f"{cname} sev{lvl}@native")
            r = results[cname][str(lvl)]
            print(f"[{cname} sev{lvl} ({p})] "
                  f"Uncal ECE={r['Uncalibrated']['ECE']:.4f} "
                  f"LTS ECE={r['LTS']['ECE']:.4f} "
                  f"Dir ECE={r['Dirichlet']['ECE']:.4f} "
                  f"Uncal mIoU={r['Uncalibrated']['mIoU']:.4f}", flush=True)

    out = {"seed": args.seed, "shot": args.shot, "fit_size": args.fit_size,
           "eval": f"native_1280x720_{args.eval_split}", "num_class": pc.NUM,
           "n_val": len(val_idx), "elapsed_s": time.time() - t0, "grid": results}
    suffix = "" if args.eval_split == "val" else f"_{args.eval_split}93"
    outp = os.path.join(seed_dir, f"shift_metrics_native{suffix}.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
