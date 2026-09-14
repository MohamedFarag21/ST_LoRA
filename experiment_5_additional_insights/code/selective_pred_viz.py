# -*- coding: utf-8 -*-
"""
Selective Prediction — Risk/Coverage Analysis & Visualisation
=============================================================
1. Samples background pixels to show high entropy but low risk
   (explains why AUGRC > baseline)
2. Computes risk/coverage curves separately for background and foreground
3. Plots both curves with baseline and optimal reference lines

Usage:
    python selective_pred_viz.py --config_name final_model --seed 42
"""

import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import skimage.draw
from tqdm import tqdm

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

DPI        = 600
LABEL_FONT = 13
TICK_FONT  = 11
TICK_SIZE  = 2.5
GRID_ALPHA = 0.35
GRID_STYLE = "--"

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255

ID2LABEL_ORIG = {
    0:  "bg", 11: "pepper_kp", 12: "pepper red", 13: "pepper yellow",
    14: "pepper green", 15: "pepper mixed", 17: "pepper mixed_red",
    18: "pepper mixed_yellow",
}
LABEL2ID   = {old: new for new, old in enumerate(sorted(ID2LABEL_ORIG.keys()))}
ID2LABEL   = {new: ID2LABEL_ORIG[old] for old, new in LABEL2ID.items()}
NUM_LABELS = len(ID2LABEL)
_REMAP_LUT = np.zeros(256, dtype=np.int64)
for old, new in LABEL2ID.items():
    _REMAP_LUT[old] = new


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SweetPepperTestDataset(Dataset):
    TEST_IDS = list(range(377, 408)) + list(range(471, 533))

    def __init__(self, coco_file, root_dir, transform=None):
        with open(coco_file) as f:
            data = json.load(f)
        self.root_dir  = root_dir
        self.transform = transform
        valid_ids      = set(self.TEST_IDS)
        self.images    = [img for img in data["images"] if img["id"] in valid_ids]
        self.ann_lookup: dict = {}
        for ann in data["annotations"]:
            self.ann_lookup.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info    = self.images[idx]
        H, W    = info["height"], info["width"]
        rel     = info["path"].lstrip("/datasets/")
        pil_img = Image.open(os.path.join(self.root_dir, rel)).convert("RGB")
        sem_map = np.zeros((H, W), dtype=np.uint8)
        for ann in self.ann_lookup.get(info["id"], []):
            for poly in ann.get("segmentation", []):
                pts = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(pts[:, 1], pts[:, 0], sem_map.shape)
                sem_map[rr, cc] = ann["category_id"]
        seg_map = _REMAP_LUT[sem_map.astype(np.int64)]
        image   = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        return image, torch.from_numpy(seg_map).long()


def build_dataloader(coco_file, root_dir, batch_size, num_workers):
    preprocessor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=False,
        do_resize=False, do_rescale=False, do_normalize=False,
        num_labels=NUM_LABELS,
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    ds = SweetPepperTestDataset(coco_file, root_dir, transform=transform)

    def collate(batch):
        images, seg_maps = zip(*batch)
        out = preprocessor(list(images), segmentation_maps=list(seg_maps),
                           return_tensors="pt")
        out["seg_maps"] = list(seg_maps)
        return out

    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(pretrained_name, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
    )
    return model.eval().to(device)


def load_adapter(base_model, adapter_path, device):
    return PeftModel.from_pretrained(base_model, adapter_path).eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Pixel collection — separate bg and fg
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_bg_fg_pixels(adapter_paths, base_model, dataloader, device, H, W):
    """
    Returns dicts with keys 'conf' and 'risk' for background and foreground pixels.
    Background  : gt == 0
    Foreground  : gt != 0 and gt != 255
    """
    M = len(adapter_paths)

    bg_conf_acc = None
    bg_risk_acc = None
    fg_conf_acc = None
    fg_risk_acc = None

    for m, ap in enumerate(adapter_paths):
        model = load_adapter(base_model, ap, device)

        bg_conf_m, bg_risk_m = [], []
        fg_conf_m, fg_risk_m = [], []

        for batch in tqdm(dataloader, desc=f"  Shot {m+1}/{M}", leave=False):
            pv  = batch["pixel_values"].to(device)
            out = model(pv)

            class_probs = out.class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = out.masks_queries_logits.sigmoid()
            seg = torch.einsum("bqc,bqhw->bchw", class_probs, masks_probs)
            seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
            seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)

            H_ent = -(seg * (seg + 1e-12).log2()).sum(dim=1)  # (B, H, W) full entropy
            preds = seg.argmax(dim=1)
            refs  = batch["seg_maps"]

            for i in range(seg.shape[0]):
                gt   = refs[i].to(device)
                ent  = H_ent[i]
                pred = preds[i]

                # Background pixels
                bg_mask = gt == 0
                if bg_mask.any():
                    bg_conf_m.append(ent[bg_mask].cpu())
                    bg_risk_m.append((pred[bg_mask] != gt[bg_mask]).float().cpu())

                # Foreground pixels (not bg, not ignore)
                fg_mask = (gt != 0) & (gt != 255)
                if fg_mask.any():
                    fg_conf_m.append(ent[fg_mask].cpu())
                    fg_risk_m.append((pred[fg_mask] != gt[fg_mask]).float().cpu())

        # Accumulate across snapshots
        bg_conf_m = torch.cat(bg_conf_m).numpy().astype(np.float64)
        bg_risk_m = torch.cat(bg_risk_m).numpy().astype(np.float64)
        fg_conf_m = torch.cat(fg_conf_m).numpy().astype(np.float64)
        fg_risk_m = torch.cat(fg_risk_m).numpy().astype(np.float64)

        if bg_conf_acc is None:
            bg_conf_acc, bg_risk_acc = bg_conf_m, bg_risk_m
            fg_conf_acc, fg_risk_acc = fg_conf_m, fg_risk_m
        else:
            bg_conf_acc += bg_conf_m; bg_risk_acc += bg_risk_m
            fg_conf_acc += fg_conf_m; fg_risk_acc += fg_risk_m

        del model; torch.cuda.empty_cache()

    bg = {"conf": (bg_conf_acc / M).astype(np.float32),
          "risk": (bg_risk_acc / M).astype(np.float32)}
    fg = {"conf": (fg_conf_acc / M).astype(np.float32),
          "risk": (fg_risk_acc / M).astype(np.float32)}
    return bg, fg


# ─────────────────────────────────────────────────────────────────────────────
# Risk/coverage curve
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_coverage_curve(conf, risk, n_points=200):
    """
    Sort pixels by entropy ascending (most confident first).
    At each coverage level compute mean risk of accepted pixels.
    Returns (coverages, risks, baseline).
    """
    order    = np.argsort(conf)              # ascending — low entropy first
    risk_sorted = risk[order]
    n        = len(risk_sorted)
    cumrisk  = np.cumsum(risk_sorted)

    # Sample n_points evenly spaced coverage levels
    indices  = np.linspace(0, n - 1, n_points, dtype=int)
    coverages = (indices + 1) / n
    risks_at_cov = cumrisk[indices] / (indices + 1)

    baseline = float(risk.mean())
    return coverages, risks_at_cov, baseline


def compute_aurc(conf, risk):
    n = len(conf)
    order = np.argsort(conf)
    risk_at_cov = np.cumsum(risk[order]) / np.arange(1, n + 1)
    return float(risk_at_cov.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_entropy_histograms(bg, fg, out_path, sample_size=1_000_000):
    """
    Show that background pixels have high entropy but low risk.
    Two subplots: entropy distributions and risk distributions.
    """
    rng = np.random.default_rng(42)

    # Sample for plotting
    bg_idx = rng.choice(len(bg["conf"]), size=min(sample_size, len(bg["conf"])), replace=False)
    fg_idx = rng.choice(len(fg["conf"]), size=min(sample_size, len(fg["conf"])), replace=False)

    bg_conf = bg["conf"][bg_idx]; bg_risk = bg["risk"][bg_idx]
    fg_conf = fg["conf"][fg_idx]; fg_risk = fg["risk"][fg_idx]

    COLOR_BG = "#1f77b4"
    COLOR_FG = "#ff7f0e"
    ALPHA    = 0.60

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Entropy distribution
    ax = axes[0]
    lo  = min(bg_conf.min(), fg_conf.min())
    hi  = max(bg_conf.max(), fg_conf.max())
    bins = np.linspace(lo, hi, 50)
    ax.hist(bg_conf, bins=bins, density=True, alpha=ALPHA,
            color=COLOR_BG, label="Background", linewidth=0)
    ax.hist(fg_conf, bins=bins, density=True, alpha=ALPHA,
            color=COLOR_FG, label="Foreground", linewidth=0)
    ax.axvline(bg_conf.mean(), color=COLOR_BG, linewidth=1.6, linestyle="--")
    ax.axvline(fg_conf.mean(), color=COLOR_FG, linewidth=1.6, linestyle="--")
    ax.set_xlabel("Entropy (bits)", fontsize=LABEL_FONT)
    ax.set_ylabel("Density",        fontsize=LABEL_FONT)
    ax.set_title("Entropy Distribution", fontsize=LABEL_FONT)
    ax.legend(fontsize=TICK_FONT)
    ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
    ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.97, 0.97,
            f"BG mean entropy: {bg_conf.mean():.3f}\n"
            f"FG mean entropy: {fg_conf.mean():.3f}\n"
            f"BG mean risk:    {bg_risk.mean():.4f}\n"
            f"FG mean risk:    {fg_risk.mean():.4f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"))

    # Risk bar chart — per-image mean and std (not per-pixel std which goes negative)
    ax = axes[1]
    # Compute per-image error rates for meaningful error bars
    # bg["risk"] and fg["risk"] are flat pixel arrays — we need per-image stats
    # Use bootstrap std as a practical approximation
    rng2 = np.random.default_rng(42)
    def bootstrap_std(arr, n_boot=200):
        means = [rng2.choice(arr, size=len(arr), replace=True).mean()
                 for _ in range(n_boot)]
        return np.std(means)

    bg_mean = bg_risk.mean()
    fg_mean = fg_risk.mean()
    bg_std  = bootstrap_std(bg_risk)
    fg_std  = bootstrap_std(fg_risk)

    labels  = ["Background", "Foreground"]
    means   = [bg_mean, fg_mean]
    stds    = [bg_std,  fg_std]
    colors  = [COLOR_BG, COLOR_FG]
    x = np.arange(2)
    bars = ax.bar(x, means, color=colors, alpha=0.75, width=0.4)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=5, linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=LABEL_FONT)
    ax.set_ylabel("Mean pixel error rate", fontsize=LABEL_FONT)
    ax.set_title("Risk by Region", fontsize=LABEL_FONT)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
    ax.grid(True, axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean + max(means)*0.02,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=TICK_FONT)

    fig.suptitle("Why AUGRC > Baseline: Background Has High Entropy but Low Risk",
                 fontsize=LABEL_FONT, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_risk_coverage_curves(bg, fg, out_path):
    """
    Plot risk/coverage curves for background and foreground separately.
    Shows baseline (random) and optimal (zero risk) reference lines.
    """
    bg_cov, bg_rc, bg_base = compute_risk_coverage_curve(bg["conf"], bg["risk"])
    fg_cov, fg_rc, fg_base = compute_risk_coverage_curve(fg["conf"], fg["risk"])

    bg_aurc = compute_aurc(bg["conf"], bg["risk"])
    fg_aurc = compute_aurc(fg["conf"], fg["risk"])

    COLOR_BG       = "#1f77b4"
    COLOR_FG       = "#ff7f0e"
    COLOR_BASELINE = "gray"
    COLOR_OPTIMAL  = "green"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cov, rc, base, aurc, color, label in [
        (axes[0], bg_cov, bg_rc, bg_base, bg_aurc, COLOR_BG, "Background"),
        (axes[1], fg_cov, fg_rc, fg_base, fg_aurc, COLOR_FG, "Foreground"),
    ]:
        ax.plot(cov, rc, color=color, linewidth=2.0,
                label=f"{label} (AURC={aurc:.4f})")
        ax.axhline(base, color=COLOR_BASELINE, linewidth=1.5, linestyle="--",
                   label=f"Baseline (random) = {base:.4f}")
        ax.axhline(0.0,  color=COLOR_OPTIMAL,  linewidth=1.0, linestyle=":",
                   label="Optimal (AURC=0)")

        # Mark 80% coverage point
        idx_80 = np.searchsorted(cov, 0.80)
        if idx_80 < len(rc):
            ax.scatter(cov[idx_80], rc[idx_80], color=color, s=60, zorder=5)
            ax.annotate(f"Risk@80%Cov\n= {rc[idx_80]:.4f}",
                        xy=(cov[idx_80], rc[idx_80]),
                        xytext=(cov[idx_80] - 0.25, rc[idx_80] + base * 0.3),
                        fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))

        ax.set_xlabel("Coverage", fontsize=LABEL_FONT)
        ax.set_ylabel("Risk (pixel error rate)", fontsize=LABEL_FONT)
        ax.set_title(f"{label} Pixels — Risk/Coverage Curve", fontsize=LABEL_FONT)
        ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
        ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=TICK_FONT, framealpha=0.85)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_combined_curves(bg, fg, out_path):
    """
    Single plot with both background and foreground curves overlaid.
    """
    bg_cov, bg_rc, bg_base = compute_risk_coverage_curve(bg["conf"], bg["risk"])
    fg_cov, fg_rc, fg_base = compute_risk_coverage_curve(fg["conf"], fg["risk"])

    bg_aurc = compute_aurc(bg["conf"], bg["risk"])
    fg_aurc = compute_aurc(fg["conf"], fg["risk"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(bg_cov, bg_rc, color="#1f77b4", linewidth=2.0,
            label=f"Background (AURC={bg_aurc:.4f}, base={bg_base:.4f})")
    ax.plot(fg_cov, fg_rc, color="#ff7f0e", linewidth=2.0,
            label=f"Foreground (AURC={fg_aurc:.4f}, base={fg_base:.4f})")
    ax.axhline(bg_base, color="#1f77b4", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axhline(fg_base, color="#ff7f0e", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.set_xlabel("Coverage", fontsize=LABEL_FONT)
    ax.set_ylabel("Risk (pixel error rate)", fontsize=LABEL_FONT)
    ax.set_title("Risk/Coverage Curve — Background vs Foreground", fontsize=LABEL_FONT)
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
    ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=TICK_FONT, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Selective Prediction Visualisation — BG vs FG")
    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/hparam_sweep")
    parser.add_argument("--out_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/figures")
    parser.add_argument("--config_name",  type=str, default="final_model")
    parser.add_argument("--configs_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/code/bup_20_trials/elora/lora_hparam_configs.json")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--shot_ids",  nargs="+",  type=int, default=[2, 3, 4, 5])
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu",         type=str, default="0")
    parser.add_argument("--height",      type=int, default=1280)
    parser.add_argument("--width",       type=int, default=720)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Selective Prediction Visualisation")
    print(f"  Config : {args.config_name}  |  Seed : {args.seed}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    adapter_paths = [
        os.path.join(args.results_dir, args.config_name,
                     f"seed_{args.seed}", f"model_shot_{shot}")
        for shot in args.shot_ids
    ]
    missing = [p for p in adapter_paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError("Missing adapters:\n" + "\n".join(missing))

    dataloader = build_dataloader(args.coco_file, args.root_dir,
                                  args.batch_size, args.num_workers)
    print(f"Test images: {len(dataloader.dataset)}")

    base_model = load_base_model(args.pretrained, device)

    print("\nCollecting background and foreground pixel scores...")
    bg, fg = collect_bg_fg_pixels(
        adapter_paths, base_model, dataloader, device, args.height, args.width)

    print(f"\nBackground pixels : {len(bg['conf']):,}  "
          f"| mean entropy: {bg['conf'].mean():.4f}  "
          f"| mean risk: {bg['risk'].mean():.4f}")
    print(f"Foreground pixels : {len(fg['conf']):,}  "
          f"| mean entropy: {fg['conf'].mean():.4f}  "
          f"| mean risk: {fg['risk'].mean():.4f}")

    prefix = os.path.join(args.out_dir,
                          f"sel_pred_{args.config_name}_seed{args.seed}")

    print("\nPlotting entropy/risk histograms...")
    plot_entropy_histograms(bg, fg, prefix + "_entropy_risk.png")

    print("Plotting separate risk/coverage curves...")
    plot_risk_coverage_curves(bg, fg, prefix + "_risk_coverage_separate.png")

    print("Plotting combined risk/coverage curve...")
    plot_combined_curves(bg, fg, prefix + "_risk_coverage_combined.png")

    # Print AURC summary
    print(f"\n{'='*55}")
    print(f"  AURC Summary  (seed {args.seed})")
    print(f"  {'Region':<15}  {'AURC':>10}  {'Baseline':>10}  {'Ratio':>8}")
    print(f"  {'─'*45}")
    for label, data in [("Background", bg), ("Foreground", fg)]:
        aurc = compute_aurc(data["conf"], data["risk"])
        base = float(data["risk"].mean())
        print(f"  {label:<15}  {aurc:>10.4f}  {base:>10.4f}  {aurc/base:>7.2f}x")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()