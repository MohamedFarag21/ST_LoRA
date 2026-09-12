# -*- coding: utf-8 -*-
"""
Evaluate an EoMT **ST-LoRA snapshot ensemble** on bup20 sweet-pepper test93 (native 1280x720).
Direct analog of eomt_ensemble_evaluate_seeded.py (the EoMT-FRE deep ensemble), but the members
are LoRA-adapter snapshots {model_shot_1..4} (epochs 20/40/60/80) from eomt_lora_train_seeded.py,
loaded as PeftModel on a fresh base. Averages the per-pixel softmax probability maps over the M
members, then measures on the mean-ensemble probs:
  * overall mIoU (macro over PRESENT classes) + per-class IoU (all 8 classes)
  * calibration ECE / ACE (float64 binned) + Brier / NLL (float64 unbinned proper scores)

Everything except member loading is IDENTICAL to the FRE ensemble eval (same DataModule, same
post_process_semantic_segmentation(return_segmentation_scores=True) -> L1-normalised (C,H,W),
same StreamMetrics), so EoMT-FRE-ens and EoMT-ST-LoRA-ens are directly comparable.

⚠ Frozen-head determinism: the adapter (modules_to_save=['upscale_block']) does NOT store the
freshly-init, FROZEN class_predictor base. We MUST seed_everything(seed) BEFORE each fresh base
load so that random head reproduces the one the adapter was trained against
(see reference_stlora_frozen_head_seed_bug). PeftModel.from_pretrained then restores the LoRA
weights + the fully-trained upscale_block on top.

Runs in the ISOLATED `eomt` env (SLURM only). Writes seed_<S>/eval_metrics_<split>_ens<M>.json.
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eomt_full_train_seeded import PepperEomtDataModule, IGNORE_INDEX   # noqa: E402
from mask2former_full_train_seeded import ID2LABEL, NUM_LABELS          # noqa: E402
from eomt_evaluate_seeded import StreamMetrics                          # noqa: E402
from transformers import EomtForUniversalSegmentation                   # noqa: E402
from peft import PeftModel                                              # noqa: E402
from torchmetrics.segmentation import MeanIoU                           # noqa: E402


def _seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_member(ckpt, adapter_path, seed, device):
    _seed_all(seed)                                                     # frozen-head determinism
    base = EomtForUniversalSegmentation.from_pretrained(
        ckpt, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    print(f"  [member] {os.path.basename(adapter_path)} loaded (PEFT)", flush=True)
    return model.to(device).eval()


@torch.no_grad()
def main():
    R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
         "mibrahi2_hpc-my_research-1775524204")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shot_ids", nargs="+", type=int, default=[1, 2, 3, 4],
                    help="snapshot ids to ensemble (default 1 2 3 4 = FRE convention)")
    ap.add_argument("--base_save_dir", default=f"{R}/results/lora_paper/eomt_lora_noaug")
    ap.add_argument("--ckpt", default="tue-mps/ade20k_semantic_eomt_large_512")
    ap.add_argument("--coco_file", default=f"{R}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    ap.add_argument("--root_dir", default=f"{R}/data/bup_20")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()

    seed_dir = os.path.join(args.base_save_dir, f"seed_{args.seed}")
    out_dir = args.out_dir or seed_dir
    os.makedirs(out_dir, exist_ok=True)
    snaps = [os.path.join(seed_dir, f"model_shot_{s}") for s in args.shot_ids]
    for p in snaps:
        if not os.path.isdir(p):
            raise FileNotFoundError(p)
    M = len(snaps)
    print(f"=== EoMT ST-LoRA ENSEMBLE (M={M}, shots={args.shot_ids}) seed {args.seed} "
          f"{args.split} ===", flush=True)

    dm = PepperEomtDataModule(coco_file=args.coco_file, root_dir=args.root_dir,
                              ckpt=args.ckpt, batch_size=1, num_workers=4,
                              use_augmentation=False)
    dm.setup()
    loader = dm.val_dataloader() if args.split == "valid" else dm.test_dataloader()
    eps = 1e-12
    C = NUM_LABELS

    # Per-frame accumulation of summed member probs (CPU float32), GT captured once.
    sum_probs, gts = [], []
    for mi, snap in enumerate(snaps):
        model = load_member(args.ckpt, snap, args.seed, dev)
        fi = 0
        for batch in tqdm(loader, desc=f"member {mi+1}/{M}"):
            out = model(pixel_values=batch["pixel_values"].to(dev),
                        patch_offsets=batch.get("patch_offsets"))
            results = dm.processor.post_process_semantic_segmentation(
                out, target_sizes=batch["target_sizes"], return_segmentation_scores=True)
            for res, gt in zip(results, batch["original_segmentation_maps"]):
                scores = res.segmentation_scores.to(dev).clamp_min(0)      # (C,H,W) >=0
                probs = (scores + eps) / (scores.sum(0, keepdim=True) + C * eps)
                p_cpu = probs.float().cpu()
                if mi == 0:
                    sum_probs.append(p_cpu)
                    gts.append(gt.cpu())
                else:
                    sum_probs[fi] += p_cpu
                fi += 1
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    # Mean-ensemble probs -> metrics. mIoU_tm = torchmetrics.MeanIoU(8) = the TRAINING /
    # mask2former metric (per-image, all 8 classes); StreamMetrics mIoU kept for continuity
    # (dataset-pooled, present classes). The two aggregators differ; mIoU_tm is the headline.
    met = StreamMetrics(num_classes=NUM_LABELS)
    tm = MeanIoU(num_classes=NUM_LABELS, per_class=False).to(dev)
    for p_sum, gt in zip(sum_probs, gts):
        probs = (p_sum / M).to(dev)
        gt_d = gt.unsqueeze(0).to(dev)
        met.update(probs.unsqueeze(0), gt_d)
        tm.update(probs.argmax(0).unsqueeze(0), gt_d)

    m = met.compute()
    m["mIoU_stream_present"] = m["mIoU"]          # keep the old value under an explicit name
    m["mIoU"] = float(tm.compute().item())        # headline = training/mask2former metric
    m.update({"seed": args.seed, "method": "eomt_stlora", "M": M, "shot_ids": args.shot_ids,
              "split": args.split, "snapshots": snaps, "elapsed_s": time.time() - t0})
    outp = os.path.join(out_dir, f"eval_metrics_{args.split}_ens{M}.json")
    json.dump(m, open(outp, "w"), indent=2)
    print(f"\n[seed {args.seed} ENS M={M}] mIoU(tm)={m['mIoU']:.4f} "
          f"mIoU(stream)={m['mIoU_stream_present']:.4f} ECE={m['ECE']:.4f} "
          f"ACE={m['ACE']:.4f} Brier={m['Brier']:.4f} NLL={m['NLL']:.4f}", flush=True)
    print("  per-class IoU:", {k: (round(v, 3) if v is not None else None)
                               for k, v in m["per_class_IoU"].items()}, flush=True)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
