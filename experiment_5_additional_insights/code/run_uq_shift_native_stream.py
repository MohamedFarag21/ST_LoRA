# -*- coding: utf-8 -*-
"""
Re-measure the 4 UQ methods (FRE / ST-LoRA / MC-Dropout / DDU) under distribution shift
with the SAME protocol as the pepper post-hoc calibrators, for a fair 11-way comparison:

  * eval set  = test93 (the lora_paper TEST_IDS = 377-407,471-532; held-out for BOTH
    studies and byte-identical to this study's test split)
  * resolution = native 1280x720
  * ECE/ACE/mIoU/acc = float64 StreamBinMetrics (NOT torchmetrics float32, which inflates
    ECE up to 2-3x above 16.7M px/bin; test93 = 85.7M px — see reference_torchmetrics_float32_ece)
  * grid       = clean + 7 corruptions x 5 severities (reused from calibration_shift_eval)

Model loading + M-pass inference reuse calibration_shift_eval.py unchanged (FRE = 5-snapshot
avg, ST-LoRA = 4-adapter avg, MC-Dropout = T=10 samples, DDU = single pass). Only the metric
head changes: instead of stacking all probs and calling float32 torchmetrics, each frame's
mean-softmax map is streamed into the float64 accumulator.

Writes native_float64/<method>_seed<seed>_shift_native.json (schema matches the calibrator
runner: {"grid": {clean|<corruption>: {sev: {ECE,ACE,mIoU,acc}}}}). Run via SLURM (ssl).
"""
import os
import sys
import json
import time
import argparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")

import calibration_shift_eval as cse                                  # noqa: E402
from run_seed_calibration_stream import StreamBinMetrics              # noqa: E402
from peft import PeftModel                                            # noqa: E402


def metrics_from_probs(probs, gt_maps, num, eps=1e-12):
    """probs (N,C,H,W) float, gt (N,H,W) long -> float64 metric dict.

    ECE/ACE/mIoU/acc via StreamBinMetrics (binned). Brier + NLL are PROPER SCORING
    RULES accumulated per-pixel in float64 with NO binning, so they are immune to the
    torchmetrics float32/bin-count ECE artifact and give an independent read on which
    method is genuinely better-calibrated under shift.
      * multiclass Brier = mean_px sum_c (p_c - 1[y=c])^2 = mean_px (sum_c p_c^2 - 2 p_gt + 1)
      * NLL              = mean_px -log p_gt
    """
    acc = StreamBinMetrics(num_classes=num)
    sq_sum = pgt_sum = nll_sum = npix = 0.0
    for i in range(probs.shape[0]):
        acc.update(probs[i:i+1], gt_maps[i:i+1])
        p, g = probs[i], gt_maps[i]                 # (C,H,W), (H,W)
        valid = g != 255
        if bool(valid.any()):
            pv = p.permute(1, 2, 0)[valid].double()  # (Nv, C)
            pgt = pv.gather(1, g[valid].long().unsqueeze(1)).squeeze(1)  # (Nv,)
            sq_sum += float((pv * pv).sum())
            pgt_sum += float(pgt.sum())
            nll_sum += float((-(pgt.clamp_min(eps).log())).sum())
            npix += float(int(valid.sum()))
    out = acc.compute()
    if npix > 0:
        out["Brier"] = (sq_sum - 2.0 * pgt_sum + npix) / npix
        out["NLL"] = nll_sum / npix
    else:
        out["Brier"] = out["NLL"] = float("nan")
    return out


def build_get_model(args, device):
    """Return (get_model, M) for the requested method — mirrors calibration_shift_eval.main."""
    if args.method == "lora":
        paths = [os.path.join(args.lora_dir, args.config_name, f"seed_{args.seed}",
                              f"model_shot_{s}") for s in args.shot_ids]
        # FRESH base per member: PeftModel.from_pretrained silently re-wraps an already-
        # adapted base if the base is reused, corrupting the effective weights of members
        # 1..M-1 (verified: reused-base ECE 0.024 vs fresh-base 0.036 @ clean test93). Each
        # member must start from a clean base so the ensemble is 4 INDEPENDENT adapters.
        def _fresh_lora_member(m):
            b = cse.load_base_model(args.pretrained, device)
            return PeftModel.from_pretrained(b, paths[m]).eval().to(device)
        return _fresh_lora_member, len(paths)
    if args.method == "fullft":
        ckpts = [os.path.join(args.fullft_dir, f"seed_{args.seed}", f"model_shot_{s}.pt")
                 for s in args.fullft_shot_ids]
        return (lambda m: cse.load_fullft_snapshot(args.pretrained, ckpts[m], device)), len(ckpts)
    if args.method == "mcdropout":
        ckpt = os.path.join(args.mcdrop_dir, f"seed_{args.seed}", "model_final.pt")
        mc = cse.load_mc_model(args.pretrained, ckpt, args.dropout_p, device)
        return (lambda m: mc), args.T
    # ddu
    ckpt = os.path.join(args.ddu_dir, f"seed_{args.seed}", "model_final.pt")
    ddu = cse.load_ddu_model(args.pretrained, ckpt, device)
    return (lambda m: ddu), 1


def eval_cell(get_model, M, corrupt_fn, param, args, device, num):
    loader = cse.build_loader(args.coco_file, args.root_dir, corrupt_fn, param,
                              args.batch_size, args.num_workers)
    probs, gt = cse.run_inference(get_model, M, loader, device, args.height, args.width)
    torch.cuda.empty_cache()
    return metrics_from_probs(probs, gt, num)


def main():
    # own parser (calibration_shift_eval's parse_args is not parameterized for reuse)
    p = argparse.ArgumentParser()
    # replicate the cse args we need
    p.add_argument("--method", required=True, choices=["lora", "fullft", "mcdropout", "ddu"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--coco_file", default=f"{ROOT}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    p.add_argument("--root_dir", default=f"{ROOT}/data/bup_20")
    p.add_argument("--pretrained", default="facebook/mask2former-swin-base-ade-semantic")
    p.add_argument("--out_dir", default=f"{ROOT}/results/lora_paper/calibration_shift/native_float64")
    p.add_argument("--lora_dir", default=f"{ROOT}/results/lora_paper/hparam_sweep")
    p.add_argument("--config_name", default="final_model")
    p.add_argument("--shot_ids", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--fullft_dir", default=f"{ROOT}/results/lora_paper/full_ft")
    p.add_argument("--fullft_shot_ids", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--mcdrop_dir", default=f"{ROOT}/results/lora_paper/mcdropout")
    p.add_argument("--dropout_p", type=float, default=0.25)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--ddu_dir", default=f"{ROOT}/results/lora_paper/ddu")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--smoke", action="store_true", help="clean + blur s1 only")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num = cse.NUM_LABELS
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    print(f"=== UQ shift @native float64 : {args.method} seed {args.seed} "
          f"(test93, {num}-class) ===", flush=True)

    get_model, M = build_get_model(args, device)
    print(f"[model] {args.method} M={M}", flush=True)

    grid = {}
    # clean baseline
    grid["clean"] = {"0": eval_cell(get_model, M, None, None, args, device, num)}
    c0 = grid["clean"]["0"]
    print(f"[clean] ECE={c0['ECE']:.4f} Brier={c0['Brier']:.4f} NLL={c0['NLL']:.4f} "
          f"mIoU={c0['mIoU']:.4f}", flush=True)

    corr_items = list(cse.CORRUPTIONS.items())
    if args.smoke:
        corr_items = corr_items[:1]
    for cname, sevs in corr_items:
        grid[cname] = {}
        cfn = cse.CORRUPT_FNS[cname]
        plist = sevs[:1] if args.smoke else sevs
        for lvl, (slabel, param) in enumerate(plist, start=1):
            r = eval_cell(get_model, M, cfn, param, args, device, num)
            grid[cname][str(lvl)] = r
            print(f"[{cname} sev{lvl} ({param})] ECE={r['ECE']:.4f} Brier={r['Brier']:.4f} "
                  f"NLL={r['NLL']:.4f} mIoU={r['mIoU']:.4f}", flush=True)

    out = {"method": args.method, "seed": args.seed, "eval": "native_1280x720_test93",
           "ece_numeric": "float64_StreamBinMetrics", "M": M, "num_class": num,
           "elapsed_s": time.time() - t0, "grid": grid}
    outp = os.path.join(args.out_dir, f"{args.method}_seed{args.seed}_shift_native.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
