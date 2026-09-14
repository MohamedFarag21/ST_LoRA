# -*- coding: utf-8 -*-
"""
Selective Prediction — BG/FG Visualisation — MC Dropout
=========================================================
Mirrors selective_pred_viz.py but uses T stochastic passes.

Usage:
    python selective_pred_viz_mcdropout.py --seed 42
"""

import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
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

# ─────────────────────────────────────────────────────────────────────────────
# Style / Config
# ─────────────────────────────────────────────────────────────────────────────

DPI        = 600
LABEL_FONT = 13
TICK_FONT  = 11
TICK_SIZE  = 2.5
GRID_ALPHA = 0.35
GRID_STYLE = "--"

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255

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
# Model — MC Dropout
# ─────────────────────────────────────────────────────────────────────────────

def load_mc_model(pretrained_name, checkpoint_path, dropout_p, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
    )
    original = model.class_predictor
    model.class_predictor = nn.Sequential(nn.Dropout(p=dropout_p), original)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pixel collection — T passes
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_bg_fg_pixels(model, dataloader, device, H, W, T):
    bg_conf_acc = bg_risk_acc = fg_conf_acc = fg_risk_acc = None

    for t in range(T):
        bg_conf_t, bg_risk_t, fg_conf_t, fg_risk_t = [], [], [], []

        for batch in tqdm(dataloader, desc=f"  Pass {t+1}/{T}", leave=False):
            pv  = batch["pixel_values"].to(device)
            out = model(pv)
            class_probs = out.class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = out.masks_queries_logits.sigmoid()
            seg = torch.einsum("bqc,bqhw->bchw", class_probs, masks_probs)
            seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
            seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)

            H_ent = -(seg * (seg + 1e-12).log2()).sum(dim=1)
            preds = seg.argmax(dim=1)
            refs  = batch["seg_maps"]

            for i in range(seg.shape[0]):
                gt   = refs[i].to(device)
                ent  = H_ent[i]
                pred = preds[i]
                bg_mask = gt == 0
                fg_mask = (gt != 0) & (gt != 255)
                if bg_mask.any():
                    bg_conf_t.append(ent[bg_mask].cpu())
                    bg_risk_t.append((pred[bg_mask] != gt[bg_mask]).float().cpu())
                if fg_mask.any():
                    fg_conf_t.append(ent[fg_mask].cpu())
                    fg_risk_t.append((pred[fg_mask] != gt[fg_mask]).float().cpu())

        bg_c = torch.cat(bg_conf_t).numpy().astype(np.float64)
        bg_r = torch.cat(bg_risk_t).numpy().astype(np.float64)
        fg_c = torch.cat(fg_conf_t).numpy().astype(np.float64)
        fg_r = torch.cat(fg_risk_t).numpy().astype(np.float64)

        if bg_conf_acc is None:
            bg_conf_acc, bg_risk_acc = bg_c, bg_r
            fg_conf_acc, fg_risk_acc = fg_c, fg_r
        else:
            bg_conf_acc += bg_c; bg_risk_acc += bg_r
            fg_conf_acc += fg_c; fg_risk_acc += fg_r

    bg = {"conf": (bg_conf_acc / T).astype(np.float32),
          "risk": (bg_risk_acc / T).astype(np.float32)}
    fg = {"conf": (fg_conf_acc / T).astype(np.float32),
          "risk": (fg_risk_acc / T).astype(np.float32)}
    return bg, fg


# ─────────────────────────────────────────────────────────────────────────────
# Metrics and plots (identical to LoRA viz)
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_coverage_curve(conf, risk, n_points=200):
    order    = np.argsort(conf)
    n        = len(risk)
    cumrisk  = np.cumsum(risk[order])
    indices  = np.linspace(0, n - 1, n_points, dtype=int)
    return (indices + 1) / n, cumrisk[indices] / (indices + 1), float(risk.mean())


def compute_aurc(conf, risk):
    order = np.argsort(conf)
    return float((np.cumsum(risk[order]) / np.arange(1, len(risk) + 1)).mean())


def _plot_entropy_hist_and_risk(bg, fg, title_suffix, out_path, sample_size=1_000_000):
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(bg["conf"]), size=min(sample_size, len(bg["conf"])), replace=False)
    fg_idx = rng.choice(len(fg["conf"]), size=min(sample_size, len(fg["conf"])), replace=False)
    bg_conf = bg["conf"][bg_idx]; bg_risk = bg["risk"][bg_idx]
    fg_conf = fg["conf"][fg_idx]; fg_risk = fg["risk"][fg_idx]

    COLOR_BG = "#1f77b4"; COLOR_FG = "#ff7f0e"; ALPHA = 0.60
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    bins = np.linspace(min(bg_conf.min(), fg_conf.min()),
                       max(bg_conf.max(), fg_conf.max()), 50)
    ax.hist(bg_conf, bins=bins, density=True, alpha=ALPHA, color=COLOR_BG, label="Background", linewidth=0)
    ax.hist(fg_conf, bins=bins, density=True, alpha=ALPHA, color=COLOR_FG, label="Foreground",  linewidth=0)
    ax.axvline(bg_conf.mean(), color=COLOR_BG, linewidth=1.6, linestyle="--")
    ax.axvline(fg_conf.mean(), color=COLOR_FG, linewidth=1.6, linestyle="--")
    ax.set_xlabel("Entropy (bits)", fontsize=LABEL_FONT)
    ax.set_ylabel("Density", fontsize=LABEL_FONT)
    ax.set_title(f"Entropy Distribution — {title_suffix}", fontsize=LABEL_FONT)
    ax.legend(fontsize=TICK_FONT)
    ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
    ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.97, 0.97,
            f"BG mean entropy: {bg_conf.mean():.3f}\n"
            f"FG mean entropy: {fg_conf.mean():.3f}\n"
            f"BG mean risk:    {bg_risk.mean():.4f}\n"
            f"FG mean risk:    {fg_risk.mean():.4f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"))

    ax = axes[1]
    rng2 = np.random.default_rng(42)
    def bstd(arr):
        return np.std([rng2.choice(arr, size=len(arr), replace=True).mean() for _ in range(200)])
    means  = [bg_risk.mean(), fg_risk.mean()]
    stds   = [bstd(bg_risk), bstd(fg_risk)]
    x = np.arange(2)
    bars = ax.bar(x, means, color=[COLOR_BG, COLOR_FG], alpha=0.75, width=0.4)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(["Background", "Foreground"], fontsize=LABEL_FONT)
    ax.set_ylabel("Mean pixel error rate", fontsize=LABEL_FONT)
    ax.set_title(f"Risk by Region — {title_suffix}", fontsize=LABEL_FONT)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
    ax.grid(True, axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean + max(means)*0.02,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=TICK_FONT)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def _plot_risk_coverage(bg, fg, title_suffix, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, color, label in [
        (axes[0], bg, "#1f77b4", f"Background — {title_suffix}"),
        (axes[1], fg, "#ff7f0e", f"Foreground — {title_suffix}"),
    ]:
        cov, rc, base = compute_risk_coverage_curve(data["conf"], data["risk"])
        aurc = compute_aurc(data["conf"], data["risk"])
        ax.plot(cov, rc, color=color, linewidth=2.0, label=f"{label} (AURC={aurc:.4f})")
        ax.axhline(base, color="gray",  linewidth=1.5, linestyle="--", label=f"Baseline = {base:.4f}")
        ax.axhline(0.0,  color="green", linewidth=1.0, linestyle=":",  label="Optimal (AURC=0)")
        idx_80 = np.searchsorted(cov, 0.80)
        if idx_80 < len(rc):
            ax.scatter(cov[idx_80], rc[idx_80], color=color, s=60, zorder=5)
            ax.annotate(f"Risk@80%Cov\n= {rc[idx_80]:.4f}",
                        xy=(cov[idx_80], rc[idx_80]),
                        xytext=(cov[idx_80] - 0.25, rc[idx_80] + base * 0.3),
                        fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
        ax.set_xlabel("Coverage", fontsize=LABEL_FONT)
        ax.set_ylabel("Risk (pixel error rate)", fontsize=LABEL_FONT)
        ax.set_title(label, fontsize=LABEL_FONT)
        ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
        ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=TICK_FONT, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir",  type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/mcdropout")
    parser.add_argument("--out_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/figures")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--T",         type=int,   default=4)
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

    ckpt = os.path.join(args.results_dir, f"seed_{args.seed}", "model_final.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing: {ckpt}")

    dataloader = build_dataloader(args.coco_file, args.root_dir,
                                  args.batch_size, args.num_workers)
    model = load_mc_model(args.pretrained, ckpt, args.dropout_p, device)

    print(f"\n[MC Dropout] Collecting BG/FG pixels (seed={args.seed}, T={args.T})...")
    bg, fg = collect_bg_fg_pixels(model, dataloader, device,
                                   args.height, args.width, args.T)

    print(f"BG: {len(bg['conf']):,} px | entropy={bg['conf'].mean():.4f} | risk={bg['risk'].mean():.4f}")
    print(f"FG: {len(fg['conf']):,} px | entropy={fg['conf'].mean():.4f} | risk={fg['risk'].mean():.4f}")

    prefix = os.path.join(args.out_dir, f"mcdropout_sel_pred_seed{args.seed}")
    _plot_entropy_hist_and_risk(bg, fg, "MC Dropout", prefix + "_entropy_risk.png")
    _plot_risk_coverage(bg, fg, "MC Dropout", prefix + "_risk_coverage_separate.png")

    print(f"\n{'='*50}")
    for label, data in [("Background", bg), ("Foreground", fg)]:
        aurc = compute_aurc(data["conf"], data["risk"])
        base = float(data["risk"].mean())
        print(f"  {label:<15}  AURC={aurc:.4f}  Baseline={base:.4f}  ({aurc/base:.2f}x)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
