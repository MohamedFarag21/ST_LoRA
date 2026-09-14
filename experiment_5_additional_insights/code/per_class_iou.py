# -*- coding: utf-8 -*-
"""
Per-class IoU (intersection/union) for each UQ method on bup20 pepper test93 @native (clean).

The shift runner reports only macro-mIoU; this exposes the per-class breakdown (8 classes:
bg + 7 pepper subtypes). Reuses the SAME build_get_model as the fair recompute -> uses the
FIXED fresh-base LoRA loader, so LoRA is the clean 4-adapter ensemble. float64 accumulation.

Writes per_class_iou/<method>_seed<seed>.json:
  { "iou": {class_name: iou or null-if-absent}, "macro_miou_present": float,
    "inter": {...}, "union": {...}, "n_present": int }
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
from run_seed_calibration_stream import StreamBinMetrics      # noqa: E402


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
    ap.add_argument("--out_dir", default=f"{ROOT}/results/lora_paper/calibration_shift/per_class_iou")
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

    acc = StreamBinMetrics(num_classes=num)
    acc.update(probs.float(), gt)
    inter = acc.inter.tolist()
    union = acc.union.tolist()
    iou = {names[c]: (inter[c] / union[c] if union[c] > 0 else None) for c in range(num)}
    present = [iou[names[c]] for c in range(num) if iou[names[c]] is not None]
    macro = sum(present) / len(present) if present else None

    out = {"method": a.method, "seed": a.seed, "M": M,
           "iou": iou,
           "inter": {names[c]: inter[c] for c in range(num)},
           "union": {names[c]: union[c] for c in range(num)},
           "n_present": len(present), "macro_miou_present": macro,
           "elapsed_s": time.time() - t0}
    outp = os.path.join(a.out_dir, f"{a.method}_seed{a.seed}.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[{a.method} s{a.seed}] macro-mIoU(present)={macro:.4f}  n_present={len(present)}", flush=True)
    for c in range(num):
        v = iou[names[c]]
        print(f"    {names[c]:22s} IoU={'--' if v is None else f'{v:.4f}'}", flush=True)
    print(f"[done] {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
