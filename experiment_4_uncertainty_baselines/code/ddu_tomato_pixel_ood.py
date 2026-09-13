# -*- coding: utf-8 -*-
"""
Pixel-level OoD-style evaluation: DDU (Deep Deterministic Uncertainty) on tomato
val images.  Sibling to full_ft_tomato_pixel_ood.py / mcdropout_tomato_pixel_ood.py
/ pepper_to_tomato_pixel_ood.py, but for the DDU method.

DDU = single deterministic Mask2Former (spectral-norm) trained on sweet pepper +
per-class GMMs fit on ID (pepper) spatial / query features.  Anomaly is flagged by
low GMM log-likelihood (density) or high entropy.  Here the OoD source is the
tomato val set instead of GrowliFlower.

Anomaly framing (same as the other tomato evals):
  anomaly (1) = any tomato foreground pixel  (sem_np > 0)
  normal  (0) = background pixel             (sem_np == 0)

Scorers (matching the DDU GrowliFlower run in ood_eval_comprehensive_with_ddu.py):
  entropy          : predictive entropy            (high = OoD)
  density_spatial  : per-pixel GMM log-likelihood  (low = OoD -> negated)
  density_query    : per-image GMM log-likelihood  (low = OoD -> negated, broadcast)

Checkpoint / GMM layout (identical to the DDU GrowliFlower eval):
  <ddu_dir>/seed_<seed>/model_final.pt
  <ddu_dir>/seed_<seed>/gmm_{query,spatial}_<config>.pkl

Run once per seed; aggregate with aggregate_pixel_ood_seeds.py for mean+/-std.

Usage:
    python ddu_tomato_pixel_ood.py --seed 42
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

# DDU machinery lives in the "_with_ddu" variant of the comprehensive eval.
from ood_eval_comprehensive_with_ddu import (
    make_transform, load_ddu_model, load_gmms, load_norm_stats,
    run_ddu_ood, compute_pixel_ood_metrics,
    BEST_QUERY_CONFIG, BEST_SPATIAL_CONFIG,
)
from mask2former_lora_train_tomato import TomatoDataset


# ---------------------------------------------------------------------------
# Dataset / loader  (identical to full_ft_tomato_pixel_ood.py)
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
# Histogram  (entropy + spatial density, bg vs tomato-fruit)
# ---------------------------------------------------------------------------

def plot_pixel_histograms(result, out_path, seed):
    COLOR_BG, COLOR_FG, ALPHA = "#1f77b4", "#ff7f0e", 0.60
    N = len(result["masks"])

    bg_ent, fg_ent, bg_d, fg_d = [], [], [], []
    for i in range(N):
        m = result["masks"][i].numpy()
        e = result["entropy"][i].numpy()
        d = result["density_spat_map"][i].numpy()
        bg, fg = (m == 0), (m > 0)
        if bg.any(): bg_ent.append(e[bg]); bg_d.append(d[bg])
        if fg.any(): fg_ent.append(e[fg]); fg_d.append(d[fg])

    bg_ent = np.concatenate(bg_ent); fg_ent = np.concatenate(fg_ent)
    bg_d   = np.concatenate(bg_d);   fg_d   = np.concatenate(fg_d)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, bg, fg, title, xlabel in [
        (axes[0], bg_ent, fg_ent, "Entropy",          "Entropy (bits)"),
        (axes[1], bg_d,   fg_d,   "Spatial density",  "GMM log-likelihood"),
    ]:
        lo, hi = min(bg.min(), fg.min()), max(bg.max(), fg.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(bg, bins=bins, density=True, alpha=ALPHA, color=COLOR_BG,
                label=f"Background (mu={bg.mean():.3f})")
        ax.hist(fg, bins=bins, density=True, alpha=ALPHA, color=COLOR_FG,
                label=f"Tomato fruit (mu={fg.mean():.3f})")
        ax.axvline(bg.mean(), color=COLOR_BG, linestyle="--", linewidth=1.5)
        ax.axvline(fg.mean(), color=COLOR_FG, linestyle="--", linewidth=1.5)
        ax.set_title(f"DDU {title} — tomato pixels (seed={seed})")
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
        description="Pixel-level OoD eval: DDU on tomato images")
    parser.add_argument("--pretrained", type=str,
                        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--root_dir",   type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/tomato_esra")
    parser.add_argument("--ddu_dir",    type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/ddu")
    parser.add_argument("--query_config",   type=str, default=BEST_QUERY_CONFIG)
    parser.add_argument("--spatial_config", type=str, default=BEST_SPATIAL_CONFIG)
    parser.add_argument("--split",       type=str, default="val", choices=["train", "val"])
    parser.add_argument("--seed",        type=int, default=42)
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

    seed_dir  = os.path.join(args.ddu_dir, f"seed_{args.seed}")
    ckpt_path = os.path.join(seed_dir, "model_final.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing DDU checkpoint: {ckpt_path}")

    print(f"Loading DDU model + GMMs (seed={args.seed})...")
    model        = load_ddu_model(args.pretrained, ckpt_path, device)
    query_gmms   = load_gmms(seed_dir, args.query_config,   "query")
    spatial_gmms = load_gmms(seed_dir, args.spatial_config, "spatial")
    q_mean, q_std = load_norm_stats(seed_dir, args.query_config,   "query")
    s_mean, s_std = load_norm_stats(seed_dir, args.spatial_config, "spatial")
    if query_gmms is None or spatial_gmms is None:
        raise FileNotFoundError("Missing GMM(s) — check query/spatial config ids.")

    print(f"Building tomato {args.split} pixel-level loader...")
    loader = build_tomato_loader(args.root_dir, args.split, H, W,
                                 args.batch_size, args.num_workers)
    print(f"Tomato {args.split} images: {len(loader.dataset)}")

    print(f"\nRunning DDU inference on tomato images (seed={args.seed})...")
    result = run_ddu_ood(model, loader, device, H, W,
                         spatial_gmms, query_gmms,
                         s_mean, s_std, q_mean, q_std)

    prefix = os.path.join(args.out_dir, f"ddu_tomato_pixelood_seed{args.seed}")

    print("\n── PIXEL-LEVEL ANALYSIS (tomato fruit = anomaly) ────────────────")
    metrics = {}
    for scorer, key, negate in [
        ("entropy",         "entropy",            False),
        ("density_spatial", "density_spat_map",   True),
        ("density_query",   "density_query_map",  True),
    ]:
        print(f"\n  Scorer: {scorer}{' (negated)' if negate else ''}")
        m = compute_pixel_ood_metrics(result, scorer=key, negate=negate)
        metrics[scorer] = m
        for k, v in m.items():
            if k != "threshold":
                print(f"    {k:<10}: {v:.4f}")

    plot_pixel_histograms(result, prefix + "_pixel_histograms.png", args.seed)

    out_json = {
        "method": "DDU -> tomato (zero-shot transfer)",
        "seed": args.seed,
        "split": args.split,
        "num_images": len(loader.dataset),
        "query_config": args.query_config,
        "spatial_config": args.spatial_config,
        "pixel_level": metrics,
        "ckpt_path": ckpt_path,
    }
    out_path = prefix + "_results.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[Saved] {out_path}")

    print(f"\n{'='*74}")
    print(f"  DDU -> tomato — Pixel-Level OoD Summary (seed={args.seed})")
    print(f"  {'Metric':<12}  {'entropy':>10}  {'dens_spat':>10}  {'dens_query':>10}")
    print(f"  {'-'*50}")
    for k in ["AUROC", "AUPR", "FPR95", "sIoU", "PPV", "MeanF1"]:
        print(f"  {k:<12}  {metrics['entropy'][k]:>10.4f}  "
              f"{metrics['density_spatial'][k]:>10.4f}  {metrics['density_query'][k]:>10.4f}")
    print(f"{'='*74}")


if __name__ == "__main__":
    main()
