# -*- coding: utf-8 -*-
"""
Pixel-level OoD-style evaluation: MC Dropout on tomato val images.
Mirrors full_ft_tomato_pixel_ood.py but uses a single model_final.pt per seed
with T stochastic forward passes (dropout kept active at test time).

MC Dropout = Mask2Former initialised with ADE weights, fully fine-tuned on
sweet pepper with a dropout layer inserted before the class predictor.
At test time dropout stays in .train() mode, giving T different stochastic
predictions from one model — analogous to a T-member ensemble.

Anomaly framing (same as all other evals):
  anomaly (1) = any tomato foreground pixel  (sem_np > 0)
  normal  (0) = background pixel             (sem_np == 0)

Checkpoint layout:
  <mcdropout_dir>/seed_<seed>/model_final.pt

Usage:
    python mcdropout_tomato_pixel_ood.py --seed 42 --T 10 --dropout_p 0.25
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
    make_transform, load_mc_model,
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
        ax.set_title(f"MC Dropout {title} — tomato pixels (seed={seed})")
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
        description="Pixel-level OoD eval: MC Dropout on tomato images")
    parser.add_argument("--pretrained",    type=str,
                        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--root_dir",      type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/tomato_esra")
    parser.add_argument("--mcdropout_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/mcdropout")
    parser.add_argument("--split",         type=str, default="val", choices=["train", "val"])
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--T",             type=int, default=10,
                        help="Number of stochastic MC forward passes")
    parser.add_argument("--dropout_p",     type=float, default=0.25)
    parser.add_argument("--out_dir",       type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/tomato_lora/pepper_transfer")
    parser.add_argument("--batch_size",    type=int, default=2)
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--gpu",           type=str, default="0")
    parser.add_argument("--height",        type=int, default=1280)
    parser.add_argument("--width",         type=int, default=720)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    H, W = args.height, args.width

    ckpt_path = os.path.join(args.mcdropout_dir, f"seed_{args.seed}", "model_final.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    print(f"Loading MC Dropout model (seed={args.seed}, dropout_p={args.dropout_p})...")
    mc_model = load_mc_model(args.pretrained, ckpt_path, args.dropout_p, device)
    M = args.T
    def get_model(m): return mc_model  # same model, T stochastic passes

    print(f"Building tomato {args.split} pixel-level loader...")
    loader = build_tomato_loader(args.root_dir, args.split, H, W,
                                 args.batch_size, args.num_workers)
    print(f"Tomato {args.split} images: {len(loader.dataset)}")

    print(f"\nRunning T={M} stochastic forward passes on tomato images "
          f"(seed={args.seed}, dropout_p={args.dropout_p})...")
    result = run_ensemble_ood(get_model, loader, device, H, W, M)

    prefix = os.path.join(args.out_dir, f"mcdropout_tomato_pixelood_seed{args.seed}")

    print("\n── PIXEL-LEVEL ANALYSIS (tomato fruit = anomaly) ────────────────")
    metrics = {}
    for scorer in ["entropy", "MI"]:
        print(f"\n  Scorer: {scorer}")
        m = compute_pixel_ood_metrics(result, scorer=scorer)
        metrics[scorer] = m
        for k, v in m.items():
            if k != "threshold":
                print(f"    {k:<10}: {v:.4f}")

    plot_pixel_histograms(result, prefix + "_pixel_histograms.png", args.seed)

    out_json = {
        "method": "MC Dropout -> tomato (zero-shot transfer)",
        "seed": args.seed,
        "T": args.T,
        "dropout_p": args.dropout_p,
        "split": args.split,
        "num_images": len(loader.dataset),
        "pixel_level": metrics,
        "ckpt_path": ckpt_path,
    }
    out_path = prefix + "_results.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[Saved] {out_path}")

    print(f"\n{'='*60}")
    print(f"  MC Dropout -> tomato — Pixel-Level OoD Summary (seed={args.seed})")
    print(f"  {'Metric':<12}  {'Entropy':>10}  {'MI':>10}")
    print(f"  {'-'*42}")
    for k in ["AUROC", "AUPR", "FPR95", "sIoU", "PPV", "MeanF1"]:
        print(f"  {k:<12}  {metrics['entropy'][k]:>10.4f}  {metrics['MI'][k]:>10.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
