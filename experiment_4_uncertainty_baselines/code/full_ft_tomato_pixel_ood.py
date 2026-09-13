# -*- coding: utf-8 -*-
"""
Pixel-level OoD-style evaluation: full fine-tuning (full_ft) snapshot ensemble
on tomato val images.  Mirrors pepper_to_tomato_pixel_ood.py exactly but loads
.pt state-dict checkpoints via load_fullft_snapshot instead of LoRA adapters.

full_ft = Mask2Former initialised with ADE-20k weights, fully fine-tuned on
sweet pepper (no LoRA).  The ensemble is built from 4 shot snapshots per seed.
Run once per seed; aggregate with aggregate_pixel_ood_seeds.py for mean±std.

Anomaly framing (same as LoRA and ADE-base evals):
  anomaly (1) = any tomato foreground pixel  (sem_np > 0)
  normal  (0) = background pixel             (sem_np == 0)

Checkpoint layout:
  <full_ft_dir>/seed_<seed>/model_shot_<shot>.pt

Usage:
    python full_ft_tomato_pixel_ood.py --seed 42 --shot_ids 2 3 4 5
"""

import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ood_eval_comprehensive import (
    make_transform, load_fullft_snapshot,
    run_ensemble_ood, compute_pixel_ood_metrics,
)
from mask2former_lora_train_tomato import TomatoDataset


# ---------------------------------------------------------------------------
# Dataset / loader
# ---------------------------------------------------------------------------

class TomatoPixelDataset(Dataset):
    """Binary fg/bg: 0 = background, 1 = any tomato class (anomaly to detect)."""

    def __init__(self, root_dir, split, transform=None):
        self.ds = TomatoDataset(root_dir, split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil_img, sem_map, _ = self.ds[idx]
        image = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        mask = torch.from_numpy((sem_map > 0).astype(np.uint8))
        return image, mask


def build_tomato_loader(root_dir, split, H, W, batch_size, num_workers):
    ds = TomatoPixelDataset(root_dir, split, transform=make_transform(H, W))

    def collate(batch):
        images, masks = zip(*batch)
        masks_r = [torch.from_numpy(
                       np.array(Image.fromarray(m.numpy()).resize((W, H), Image.NEAREST)))
                   for m in masks]
        return torch.stack(images), torch.stack(masks_r)

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                      num_workers=num_workers, persistent_workers=(num_workers > 0))


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def plot_pixel_histograms(result, out_path, seed):
    COLOR_BG, COLOR_FG, ALPHA = "#1f77b4", "#ff7f0e", 0.60
    N = len(result["masks"])

    bg_ent, fg_ent, bg_mi, fg_mi = [], [], [], []
    for i in range(N):
        m  = result["masks"][i].numpy()
        e  = result["entropy"][i].numpy()
        mi = result["MI"][i].numpy()
        bg, fg = (m == 0), (m > 0)
        if bg.any(): bg_ent.append(e[bg]); bg_mi.append(mi[bg])
        if fg.any(): fg_ent.append(e[fg]); fg_mi.append(mi[fg])

    bg_ent = np.concatenate(bg_ent); fg_ent = np.concatenate(fg_ent)
    bg_mi  = np.concatenate(bg_mi);  fg_mi  = np.concatenate(fg_mi)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, bg, fg, title, xlabel in [
        (axes[0], bg_ent, fg_ent, "Entropy",                       "Entropy (bits)"),
        (axes[1], bg_mi,  fg_mi,  "Mutual Information (epistemic)", "MI (bits)"),
    ]:
        lo, hi = min(bg.min(), fg.min()), max(bg.max(), fg.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(bg, bins=bins, density=True, alpha=ALPHA, color=COLOR_BG,
                label=f"Background (μ={bg.mean():.3f})")
        ax.hist(fg, bins=bins, density=True, alpha=ALPHA, color=COLOR_FG,
                label=f"Tomato fruit (μ={fg.mean():.3f})")
        ax.axvline(bg.mean(), color=COLOR_BG, linestyle="--", linewidth=1.5)
        ax.axvline(fg.mean(), color=COLOR_FG, linestyle="--", linewidth=1.5)
        ax.set_title(f"Full-FT ensemble {title} — tomato pixels (seed={seed})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pixel-level OoD eval: full-FT ensemble on tomato images")
    parser.add_argument("--pretrained",  type=str,
                        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--root_dir",    type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/tomato_esra")
    parser.add_argument("--full_ft_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/full_ft")
    parser.add_argument("--split",       type=str, default="val", choices=["train", "val"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--shot_ids",    nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--out_prefix",  type=str, default="full_ft_tomato_pixelood",
                        help="Output filename prefix (before _seed<S>). Use a distinct "
                             "value for single-shot runs to avoid clobbering the ensemble.")
    parser.add_argument("--scorers",     nargs="+", type=str, default=["entropy", "MI"],
                        choices=["entropy", "MI"],
                        help="Which scorers to compute metrics for. A single-shot run "
                             "should use only 'entropy' (MI is ~0 with one member and "
                             "doubles the metric cost).")
    parser.add_argument("--out_dir",     type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/tomato_lora/pepper_transfer")
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu",         type=str, default="0")
    parser.add_argument("--height",      type=int, default=1280)
    parser.add_argument("--width",       type=int, default=720)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    H, W = args.height, args.width

    ckpt_paths = [
        os.path.join(args.full_ft_dir, f"seed_{args.seed}", f"model_shot_{s}.pt")
        for s in args.shot_ids
    ]
    missing = [p for p in ckpt_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(missing))
    M = len(ckpt_paths)

    def get_model(m):
        return load_fullft_snapshot(args.pretrained, ckpt_paths[m], device)

    print(f"Building tomato {args.split} pixel-level loader...")
    loader = build_tomato_loader(args.root_dir, args.split, H, W,
                                 args.batch_size, args.num_workers)
    print(f"Tomato {args.split} images: {len(loader.dataset)}")

    print(f"\nRunning {M}-snapshot full-FT ensemble on tomato images "
          f"(seed={args.seed}, shots={args.shot_ids})...")
    result = run_ensemble_ood(get_model, loader, device, H, W, M)

    prefix = os.path.join(args.out_dir, f"{args.out_prefix}_seed{args.seed}")

    print("\n── PIXEL-LEVEL ANALYSIS (tomato fruit = anomaly) ────────────────")
    metrics = {}
    for scorer in args.scorers:
        print(f"\n  Scorer: {scorer}")
        m = compute_pixel_ood_metrics(result, scorer=scorer)
        metrics[scorer] = m
        for k, v in m.items():
            if k != "threshold":
                print(f"    {k:<10}: {v:.4f}")

    plot_pixel_histograms(result, prefix + "_pixel_histograms.png", args.seed)

    out_json = {
        "method": "Full fine-tuning ensemble -> tomato (zero-shot transfer)",
        "seed": args.seed,
        "shot_ids": args.shot_ids,
        "split": args.split,
        "num_images": len(loader.dataset),
        "pixel_level": metrics,
        "ckpt_paths": ckpt_paths,
    }
    out_path = prefix + "_results.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[Saved] {out_path}")

    print(f"\n{'='*60}")
    print(f"  Full-FT -> tomato — Pixel-Level OoD Summary (seed={args.seed})")
    hdr = "  " + f"{'Metric':<12}" + "".join(f"{s:>10}" for s in args.scorers)
    print(hdr)
    print(f"  {'-'*(12 + 10*len(args.scorers))}")
    for k in ["AUROC", "AUPR", "FPR95", "sIoU", "PPV", "MeanF1"]:
        row = "  " + f"{k:<12}" + "".join(f"{metrics[s][k]:>10.4f}" for s in args.scorers)
        print(row)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
