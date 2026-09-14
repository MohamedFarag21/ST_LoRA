# -*- coding: utf-8 -*-
"""
Spatial MSP Visualization — ID vs Far OoD
==========================================
Visualises one sweet pepper image (in-distribution) and one GrowliFlower-L
image (far OoD) side-by-side with their spatial Max Softmax Probability maps.

The MSP map shows per-pixel confidence: bright = model is confident,
dark = model is confused. OoD images should appear uniformly dark/cold.

Layout (2 rows × 2 columns):
  Row 1: Sweet pepper image  |  MSP heatmap
  Row 2: GrowliFlower image  |  MSP heatmap

Uses the same style constants as plot_robustness.py.

Usage:
    python visualise_msp.py --config_name base
    python visualise_msp.py --config_name base --id_idx 5 --ood_idx 10
"""

import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from PIL import Image
import skimage.draw

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# Style constants — mirrors plot_robustness.py
# ─────────────────────────────────────────────────────────────────────────────

DPI         = 600
LABEL_FONT  = 13
TICK_FONT   = 11
TICK_SIZE   = 2.5
LEGEND_FONT = 12
GRID_ALPHA  = 0.35
GRID_STYLE  = "--"

# MSP colormap: viridis (dark=low confidence, bright=high confidence)
MSP_CMAP    = "viridis"
MSP_VMIN    = 0.0
MSP_VMAX    = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Label map
# ─────────────────────────────────────────────────────────────────────────────

ID2LABEL_ORIG = {
    0: "bg", 11: "pepper_kp", 12: "pepper red", 13: "pepper yellow",
    14: "pepper green", 15: "pepper mixed", 17: "pepper mixed_red",
    18: "pepper mixed_yellow",
}
LABEL2ID   = {old: new for new, old in enumerate(sorted(ID2LABEL_ORIG.keys()))}
ID2LABEL   = {new: ID2LABEL_ORIG[old] for old, new in LABEL2ID.items()}
NUM_LABELS = len(ID2LABEL)
_REMAP_LUT = np.zeros(256, dtype=np.int64)
for old, new in LABEL2ID.items():
    _REMAP_LUT[old] = new

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sweet_pepper_image(coco_file, root_dir, idx=0):
    """Load one image from the sweet pepper test set. Returns PIL image."""
    TEST_IDS = set(list(range(377, 408)) + list(range(471, 533)))
    with open(coco_file) as f:
        data = json.load(f)
    images = [img for img in data["images"] if img["id"] in TEST_IDS]
    info   = images[idx % len(images)]
    rel    = info["path"].lstrip("/datasets/")
    return Image.open(os.path.join(root_dir, rel)).convert("RGB")


def load_growliflower_image(growliflower_dir, idx=0):
    """Load one image from GrowliFlower-L. Returns PIL image."""
    root  = Path(growliflower_dir)
    paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not paths:
        raise RuntimeError(f"No images found in {growliflower_dir}")
    return Image.open(paths[idx % len(paths)]).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ensemble(pretrained_name, adapter_paths, device):
    """Load base model + PEFT adapters. Returns list of models."""
    base = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
    )
    base.eval().to(device)
    for p in base.parameters():
        p.requires_grad = False

    models = []
    for ap in adapter_paths:
        m = PeftModel.from_pretrained(base, ap)
        m.eval().to(device)
        models.append(m)
    return models


# ─────────────────────────────────────────────────────────────────────────────
# MSP map computation
# ─────────────────────────────────────────────────────────────────────────────

def make_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])


@torch.no_grad()
def compute_score_maps(pil_img, models, device, H, W):
    """
    Run ensemble inference and return four spatial score maps.

    Returns:
        maps  : dict of (H, W) numpy arrays
        means : dict of image-level mean scalars
    """
    transform   = make_transform()
    pil_resized = pil_img.resize((W, H), Image.BILINEAR)
    tensor      = transform(pil_resized).unsqueeze(0).to(device)

    # Collect per-model seg maps → (M, 1, C, H, W)
    seg_list = []
    for model in models:
        outputs     = model(tensor)
        class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = outputs.masks_queries_logits.sigmoid()
        seg = torch.einsum("bqc,bqhw->bchw", class_probs, masks_probs)
        seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
        seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
        seg_list.append(seg[0])   # (C, H, W)

    seg_stack  = torch.stack(seg_list)   # (M, C, H, W)
    mean_probs = seg_stack.mean(dim=0)   # (C, H, W)

    # MSP
    msp_map = mean_probs.max(dim=0).values

    # FG-MSP
    fg      = mean_probs[1:]
    fg      = fg / (fg.sum(dim=0, keepdim=True) + 1e-6)
    fg_map  = fg.max(dim=0).values

    # Entropy of ensemble mean  (in bits)
    H_mean  = -(mean_probs * (mean_probs + 1e-12).log2()).sum(dim=0)  # (H,W)

    # Foreground-only MI — strip background before computing to avoid saturation
    fg_stack  = seg_stack[:, 1:, :, :]                                  # (M, C-1, H, W)
    fg_stack  = fg_stack / (fg_stack.sum(dim=1, keepdim=True) + 1e-6)
    fg_mean   = fg_stack.mean(dim=0)                                     # (C-1, H, W)
    H_fg_mean = -(fg_mean  * (fg_mean  + 1e-12).log2()).sum(dim=0)      # (H,W)
    H_fg_per  = -(fg_stack * (fg_stack + 1e-12).log2()).sum(dim=1)      # (M,H,W)
    MI_map    = H_fg_mean - H_fg_per.mean(dim=0)                        # (H,W)

    maps = {
        "msp":     msp_map.cpu().numpy(),
        "fg_msp":  fg_map.cpu().numpy(),
        "entropy": H_mean.cpu().numpy(),
        "mi":      MI_map.cpu().numpy(),
    }
    means = {k: float(v.mean()) for k, v in maps.items()}
    return maps, means


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def apply_style(ax, xlabel="", ylabel=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT)
    ax.tick_params(axis="both", labelsize=TICK_FONT, length=TICK_SIZE)


def plot_msp_comparison(
    id_image,  id_maps,  id_means,
    ood_image, ood_maps, ood_means,
    out_path,
):
    """
    2 rows × 5 columns:
      Col 0: raw RGB image
      Col 1: MSP map
      Col 2: FG-MSP map
      Col 3: Entropy map
      Col 4: MI (epistemic uncertainty) map
    """
    col_titles  = ["Image", "MSP", "FG-MSP", "Entropy", "Mutual Info"]
    map_keys    = ["msp", "fg_msp", "entropy", "mi"]
    row_labels  = ["Sweet Pepper (ID)", "GrowliFlower-L (OoD)"]
    row_images  = [id_image,  ood_image]
    row_maps    = [id_maps,   ood_maps]
    row_means   = [id_means,  ood_means]

    # Shared range for entropy and MI across both rows
    ent_max = max(id_maps["entropy"].max(), ood_maps["entropy"].max())
    mi_max  = max(id_maps["mi"].max(),      ood_maps["mi"].max())
    norms = {
        "msp":     Normalize(vmin=0.0, vmax=1.0),
        "fg_msp":  Normalize(vmin=0.0, vmax=1.0),
        "entropy": Normalize(vmin=0.0, vmax=ent_max),
        "mi":      Normalize(vmin=0.0, vmax=mi_max),
    }
    cmaps = {
        "msp":     "viridis",
        "fg_msp":  "viridis",
        "entropy": "viridis_r",   # bright = uncertain
        "mi":      "hot",         # bright = high epistemic uncertainty
    }
    score_labels = {
        "msp":     "MSP",
        "fg_msp":  "FG-MSP",
        "entropy": "Entropy",
        "mi":      "MI",
    }

    fig, axes = plt.subplots(
        2, 5,
        figsize=(17, 7),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1], "hspace": 0.12, "wspace": 0.05},
    )

    for row, (pil_img, maps, means, row_label) in enumerate(
        zip(row_images, row_maps, row_means, row_labels)
    ):
        axes[row][0].imshow(pil_img)
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        for spine in axes[row][0].spines.values():
            spine.set_visible(False)
        axes[row][0].set_ylabel(row_label, fontsize=LABEL_FONT, labelpad=8)

        for col, key in enumerate(map_keys, start=1):
            ax = axes[row][col]
            ax.imshow(maps[key], cmap=cmaps[key], norm=norms[key], aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(
                0.97, 0.03,
                f"{score_labels[key]} = {means[key]:.3f}",
                transform=ax.transAxes,
                fontsize=TICK_FONT, color="white", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, ec="none"),
            )

    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=LABEL_FONT, pad=6)

    # Horizontal colorbars below each map column
    col_lefts = [0.215, 0.395, 0.575, 0.755]
    for col_i, key in enumerate(map_keys):
        cbar_ax = fig.add_axes([col_lefts[col_i], 0.04, 0.155, 0.025])
        sm = cm.ScalarMappable(cmap=cmaps[key], norm=norms[key])
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(score_labels[key], fontsize=TICK_FONT)
        cbar.ax.tick_params(labelsize=TICK_FONT - 1, length=TICK_SIZE)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Full-dataset pixel-score collection for histograms
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_image_scores(pil_images, models, device, H, W):
    """
    Run ensemble over a list of PIL images.
    Returns IMAGE-LEVEL mean scores (one scalar per image) — avoids OOM
    from storing H×W pixel arrays for the full dataset.

    scores returned: msp, fg_msp, entropy
    MI excluded from histograms (≈0 for same-seed snapshot ensemble).
    """
    transform = make_transform()
    acc = {"msp": [], "fg_msp": [], "entropy": []}

    for pil_img in tqdm(pil_images, desc="  Collecting image scores", leave=False):
        pil_r  = pil_img.resize((W, H), Image.BILINEAR)
        tensor = transform(pil_r).unsqueeze(0).to(device)

        seg_list = []
        for model in models:
            outputs     = model(tensor)
            class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = outputs.masks_queries_logits.sigmoid()
            seg = torch.einsum("bqc,bqhw->bchw", class_probs, masks_probs)
            seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
            seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
            seg_list.append(seg[0])   # (C, H, W)

        seg_stack  = torch.stack(seg_list)   # (M, C, H, W)
        mean_probs = seg_stack.mean(dim=0)   # (C, H, W)

        # Image-level MSP: mean of per-pixel max prob
        acc["msp"].append(
            float(mean_probs.max(dim=0).values.mean().cpu()))

        # Image-level FG-MSP: foreground only
        fg = mean_probs[1:]
        fg = fg / (fg.sum(dim=0, keepdim=True) + 1e-6)
        acc["fg_msp"].append(
            float(fg.max(dim=0).values.mean().cpu()))

        # Image-level entropy: mean pixel entropy (bits)
        H_mean = -(mean_probs * (mean_probs + 1e-12).log2()).sum(dim=0)
        acc["entropy"].append(float(H_mean.mean().cpu()))

    return {k: np.array(v, dtype=np.float32) for k, v in acc.items()}


def load_all_sweet_pepper_images(coco_file, root_dir):
    """Return list of all PIL images from the sweet pepper test set."""
    TEST_IDS = set(list(range(377, 408)) + list(range(471, 533)))
    with open(coco_file) as f:
        data = json.load(f)
    images = [img for img in data["images"] if img["id"] in TEST_IDS]
    pils   = []
    for info in images:
        rel = info["path"].lstrip("/datasets/")
        pils.append(Image.open(os.path.join(root_dir, rel)).convert("RGB"))
    return pils


def load_all_growliflower_images(growliflower_dir):
    """Return list of all PIL images from GrowliFlower-L."""
    root  = Path(growliflower_dir)
    paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    return [Image.open(p).convert("RGB") for p in paths]


# ─────────────────────────────────────────────────────────────────────────────
# Histogram — ID vs OoD score distributions  (full dataset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_histograms(id_image_scores, ood_image_scores, out_path, n_bins=30):
    """
    1 row × 3 subplots — MSP, FG-MSP, Entropy.
    Each subplot shows overlapping histograms of IMAGE-LEVEL aggregated scores
    (one value per image, not per pixel) — no OOM risk.

    id_image_scores  : dict of (N_id_images,) arrays
    ood_image_scores : dict of (N_ood_images,) arrays
    """
    KEYS   = ["msp",   "fg_msp",  "entropy"]
    LABELS = ["MSP",   "FG-MSP",  "Entropy (bits)"]

    COLOR_ID  = "#1f77b4"
    COLOR_OOD = "#ff7f0e"
    ALPHA     = 0.60

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for ax, key, xlabel in zip(axes, KEYS, LABELS):
        id_vals  = id_image_scores[key]
        ood_vals = ood_image_scores[key]

        lo   = min(id_vals.min(), ood_vals.min())
        hi   = max(id_vals.max(), ood_vals.max())
        bins = np.linspace(lo, hi, n_bins + 1)

        ax.hist(id_vals,  bins=bins, density=True, alpha=ALPHA,
                color=COLOR_ID,  label="Sweet Pepper (ID)",    linewidth=0)
        ax.hist(ood_vals, bins=bins, density=True, alpha=ALPHA,
                color=COLOR_OOD, label="GrowliFlower-L (OoD)", linewidth=0)

        ax.axvline(id_vals.mean(),  color=COLOR_ID,  linewidth=1.6,
                   linestyle="--", alpha=0.9)
        ax.axvline(ood_vals.mean(), color=COLOR_OOD, linewidth=1.6,
                   linestyle="--", alpha=0.9)

        ax.set_xlabel(xlabel,    fontsize=LABEL_FONT)
        ax.set_ylabel("Density", fontsize=LABEL_FONT)
        ax.tick_params(axis="both", labelsize=TICK_FONT, length=TICK_SIZE)
        ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax is axes[0]:
            ax.legend(fontsize=LEGEND_FONT, framealpha=0.85)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Spatial MSP visualisation — ID vs OoD")

    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/hparam_sweep")
    parser.add_argument("--growliflower_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/figures")

    parser.add_argument("--config_name",  type=str, default="base")
    parser.add_argument("--config_idx",   type=int, default=None)
    parser.add_argument("--configs_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/code/bup_20_trials/elora/lora_hparam_configs.json")

    # Which seed to use for the visualisation (single seed is enough)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--shot_ids",  nargs="+", type=int, default=[1, 2, 3, 4])

    # Which sample to visualise
    parser.add_argument("--id_idx",    type=int, default=0,
        help="Index into sweet pepper test set")
    parser.add_argument("--ood_idx",   type=int, default=0,
        help="Index into GrowliFlower-L images")

    parser.add_argument("--gpu",       type=str, default="0")
    parser.add_argument("--height",    type=int, default=1280)
    parser.add_argument("--width",     type=int, default=720)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve config name
    if args.config_name is None and args.config_idx is not None:
        with open(args.configs_file) as f:
            all_configs = json.load(f)
        args.config_name = all_configs[args.config_idx]["name"]

    print(f"\n{'='*60}")
    print(f"  Spatial MSP Visualisation")
    print(f"  Config : {args.config_name}  |  Seed : {args.seed}")
    print(f"  ID  image idx : {args.id_idx}")
    print(f"  OoD image idx : {args.ood_idx}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # Load adapter ensemble for the chosen seed
    adapter_paths = [
        os.path.join(args.results_dir, args.config_name,
                     f"seed_{args.seed}", f"model_shot_{shot}")
        for shot in args.shot_ids
    ]
    missing = [p for p in adapter_paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(f"Missing adapters:\n" + "\n".join(missing))

    print(f"Loading ensemble ({len(adapter_paths)} adapters)...")
    models = load_ensemble(args.pretrained, adapter_paths, device)

    # Load images
    print(f"Loading ID  image (idx={args.id_idx})...")
    id_image  = load_sweet_pepper_image(args.coco_file, args.root_dir, args.id_idx)

    print(f"Loading OoD image (idx={args.ood_idx})...")
    ood_image = load_growliflower_image(args.growliflower_dir, args.ood_idx)

    # Compute score maps
    print("Computing score maps for ID image...")
    id_maps, id_means = compute_score_maps(
        id_image, models, device, args.height, args.width)

    print("Computing score maps for OoD image...")
    ood_maps, ood_means = compute_score_maps(
        ood_image, models, device, args.height, args.width)

    print(f"\n  {'Scorer':<10}  {'ID mean':>10}  {'OoD mean':>10}")
    for k in ["msp", "fg_msp", "entropy"]:
        print(f"  {k:<10}  {id_means[k]:>10.4f}  {ood_means[k]:>10.4f}")

    # Resize PIL images to match the map dimensions for display
    id_image_disp  = id_image.resize( (args.width, args.height), Image.BILINEAR)
    ood_image_disp = ood_image.resize((args.width, args.height), Image.BILINEAR)

    out_path = os.path.join(
        args.out_dir,
        f"msp_visualisation_{args.config_name}_seed{args.seed}"
        f"_id{args.id_idx}_ood{args.ood_idx}.png"
    )

    plot_msp_comparison(
        id_image_disp,  id_maps,  id_means,
        ood_image_disp, ood_maps, ood_means,
        out_path,
    )

    # ── Full-dataset histograms ───────────────────────────────────────────────
    print("\nLoading full datasets for histogram computation...")
    all_id_images  = load_all_sweet_pepper_images(args.coco_file, args.root_dir)
    all_ood_images = load_all_growliflower_images(args.growliflower_dir)
    print(f"  ID  images : {len(all_id_images)}")
    print(f"  OoD images : {len(all_ood_images)}")

    print("Collecting image scores over full ID dataset...")
    id_image_scores  = collect_image_scores(all_id_images,  models, device, args.height, args.width)

    print("Collecting image scores over full OoD dataset...")
    ood_image_scores = collect_image_scores(all_ood_images, models, device, args.height, args.width)

    hist_path = out_path.replace(".png", "_histograms.png")
    plot_score_histograms(id_image_scores, ood_image_scores, hist_path)


if __name__ == "__main__":
    main()