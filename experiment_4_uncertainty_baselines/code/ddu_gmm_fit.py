# -*- coding: utf-8 -*-
"""
DDU GMM Fitting — Full Grid Search
====================================
Extracts penultimate features from the training set (GPU, once per seed),
then fits a grid of GMM configurations (CPU, fast) and saves all variants.

Feature types:
  query   — transformer_decoder_last_hidden_state  (256-dim per query)
             Assignment: each query → GT class via mask overlap
  spatial — masks_queries_logits upsampled (100-dim per pixel)
             Assignment: each pixel → GT class via GT mask

Grid search axes:
  normalization  : none | zscore
  n_components   : 1 | 2
  covariance_type: full | tied
  reg_covar      : 1e-4 | 1e-3

Total combinations: 2 × 2 × 2 × 2 = 16 per feature type → 32 total

For each combination, BIC and AIC are recorded so the best configuration
can be selected without needing test data.

Output per seed (seed_<N>/):
  gmm_<feature>_<config_id>.pkl          — fitted GMMs
  gmm_<feature>_<config_id>_stats.npz   — normalization stats (if zscore)
  gmm_grid_results.json                  — BIC/AIC comparison table
  gmm_config.json                        — grid definition + metadata

Usage:
    python ddu_gmm_fit.py --seed 42
    python ddu_gmm_fit.py --seed 42 --feature_type query
"""

import os
import csv
import json
import pickle
import random
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.nn.utils.parametrizations import spectral_norm as sn_parametrize
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import skimage.draw
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255

ID2LABEL_ORIG = {
    0:  "bg",
    11: "pepper_kp",
    12: "pepper red",
    13: "pepper yellow",
    14: "pepper green",
    15: "pepper mixed",
    17: "pepper mixed_red",
    18: "pepper mixed_yellow",
}
LABEL2ID    = {old: new for new, old in enumerate(sorted(ID2LABEL_ORIG.keys()))}
ID2LABEL    = {new: ID2LABEL_ORIG[old] for old, new in LABEL2ID.items()}
NUM_LABELS  = len(ID2LABEL)
NUM_QUERIES = 100
QUERY_DIM   = 256

_REMAP_LUT = np.zeros(256, dtype=np.int64)
for old, new in LABEL2ID.items():
    _REMAP_LUT[old] = new

# ── Grid definition ───────────────────────────────────────────────────────────
GRID = {
    "normalization":   ["none", "zscore"],
    "n_components":    [1, 2],
    "covariance_type": ["full", "tied"],
    "reg_covar":       [1e-4, 1e-3],
}

def make_config_id(norm, n_comp, cov, reg):
    """Short deterministic string ID for a grid configuration."""
    reg_str = f"{reg:.0e}".replace("-0", "-")   # 1e-4 → 1e-4
    return f"norm={norm}_k={n_comp}_cov={cov}_reg={reg_str}"


def all_configs():
    """Generate all (config_id, config_dict) pairs from the grid."""
    for norm, k, cov, reg in itertools.product(
            GRID["normalization"],
            GRID["n_components"],
            GRID["covariance_type"],
            GRID["reg_covar"]):
        cid = make_config_id(norm, k, cov, reg)
        yield cid, {"normalization": norm, "n_components": k,
                    "covariance_type": cov, "reg_covar": reg}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class COCOSegDataset(Dataset):
    SPLIT_IDS = {
        "train": list(range(283, 345)) + list(range(408, 471)),
        "valid": list(range(345, 377)) + list(range(533, 564)),
        "test":  list(range(377, 408)) + list(range(471, 533)),
    }

    def __init__(self, coco_file, root_dir, split, transform=None):
        with open(coco_file) as f:
            data = json.load(f)
        self.root_dir  = root_dir
        self.transform = transform
        valid_ids      = set(self.SPLIT_IDS[split])
        self.images    = [img for img in data["images"] if img["id"] in valid_ids]
        self.ann_lookup: dict = {}
        for ann in data["annotations"]:
            self.ann_lookup.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info     = self.images[idx]
        H, W     = info["height"], info["width"]
        rel_path = info["path"].lstrip("/datasets/")
        pil_img  = Image.open(os.path.join(self.root_dir, rel_path)).convert("RGB")
        sem_map  = np.zeros((H, W), dtype=np.uint8)
        for ann in self.ann_lookup.get(info["id"], []):
            for poly in ann.get("segmentation", []):
                pts = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(pts[:, 1], pts[:, 0], sem_map.shape)
                sem_map[rr, cc] = ann["category_id"]
        seg_map = _REMAP_LUT[sem_map.astype(np.int64)]
        image   = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        return image, torch.from_numpy(seg_map).long()


def build_train_loader(coco_file, root_dir, batch_size, num_workers):
    preprocessor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=False,
        do_resize=False, do_rescale=False, do_normalize=False,
        num_labels=NUM_LABELS,
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    dataset = COCOSegDataset(coco_file, root_dir, "train", transform=transform)

    def collate_fn(batch):
        images, seg_maps = zip(*batch)
        out = preprocessor(list(images), segmentation_maps=list(seg_maps),
                           return_tensors="pt")
        out["seg_maps"] = torch.stack(list(seg_maps))
        return out

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def apply_spectral_norm(model: nn.Module) -> nn.Module:
    visited = set()
    def _recurse(module):
        for name, child in module.named_children():
            if id(child) in visited:
                continue
            visited.add(id(child))
            _recurse(child)
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                try:
                    sn_parametrize(child)
                except Exception:
                    pass
    _recurse(model)
    return model


def load_ddu_model(pretrained_name, checkpoint_path, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
    )
    model = apply_spectral_norm(model)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"[Loaded] {checkpoint_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (GPU — done once per seed)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_query_features(model, dataloader, device, H, W,
                            min_overlap: float = 0.1):
    """
    transformer_decoder_last_hidden_state → (100, 256) per image.
    Each query assigned to GT class with highest mask overlap.
    Returns dict {class_id: np.ndarray (N, 256)}.
    """
    features_by_class = {c: [] for c in range(NUM_LABELS)}

    for batch in tqdm(dataloader, desc="  [Query] GPU inference"):
        pv      = batch["pixel_values"].to(device)
        gt_maps = batch["seg_maps"]
        outputs = model(pv, output_hidden_states=True)

        query_feats = outputs.transformer_decoder_last_hidden_state  # (B, 100, 256)
        mask_probs  = outputs.masks_queries_logits.sigmoid()
        mask_probs  = F.interpolate(mask_probs, size=(H, W),
                                    mode="bilinear", align_corners=False)

        B = pv.shape[0]
        for b in range(B):
            gt = gt_maps[b].numpy()
            qf = query_feats[b].cpu().numpy()   # (100, 256)
            mp = mask_probs[b].cpu().numpy()     # (100, H, W)

            for q in range(NUM_QUERIES):
                best_class, best_overlap = -1, min_overlap
                for c in range(NUM_LABELS):
                    gt_c = (gt == c).astype(np.float32)
                    n_gt = gt_c.sum()
                    if n_gt == 0:
                        continue
                    overlap = float((mp[q] * gt_c).sum()) / n_gt
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_class   = c
                if best_class >= 0:
                    features_by_class[best_class].append(qf[q])

    for c in range(NUM_LABELS):
        features_by_class[c] = (np.stack(features_by_class[c], axis=0)
                                 if features_by_class[c]
                                 else np.zeros((0, QUERY_DIM), dtype=np.float32))
        print(f"  Class {c:2d} ({ID2LABEL[c]:<22}): {len(features_by_class[c]):>6} query vectors")

    return features_by_class


@torch.no_grad()
def extract_spatial_features(model, dataloader, device, H, W,
                              max_per_class_per_image: int = 300):
    """
    masks_queries_logits upsampled → (H, W, 100) per image.
    Each pixel assigned to GT class via GT mask. Subsampled per image.
    Returns dict {class_id: np.ndarray (N, 100)}.
    """
    features_by_class = {c: [] for c in range(NUM_LABELS)}

    for batch in tqdm(dataloader, desc="  [Spatial] GPU inference"):
        pv      = batch["pixel_values"].to(device)
        gt_maps = batch["seg_maps"]
        outputs = model(pv, output_hidden_states=True)

        spatial = outputs.masks_queries_logits.sigmoid()
        spatial = F.interpolate(spatial, size=(H, W),
                                mode="bilinear", align_corners=False)
        spatial = spatial.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 100)

        B = pv.shape[0]
        for b in range(B):
            gt   = gt_maps[b].numpy()
            feat = spatial[b]
            for c in range(NUM_LABELS):
                pixel_idx = np.where(gt.ravel() == c)[0]
                if len(pixel_idx) == 0:
                    continue
                if len(pixel_idx) > max_per_class_per_image:
                    pixel_idx = np.random.choice(
                        pixel_idx, max_per_class_per_image, replace=False)
                rows = pixel_idx // W
                cols = pixel_idx  % W
                features_by_class[c].append(feat[rows, cols])

    for c in range(NUM_LABELS):
        features_by_class[c] = (np.concatenate(features_by_class[c], axis=0)
                                 if features_by_class[c]
                                 else np.zeros((0, NUM_QUERIES), dtype=np.float32))
        print(f"  Class {c:2d} ({ID2LABEL[c]:<22}): {len(features_by_class[c]):>8} pixel vectors")

    return features_by_class


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def apply_normalization(features_by_class: dict, normalization: str):
    """
    normalization='none'   → return features unchanged, stats=None
    normalization='zscore' → global z-score (mean/std across all classes)
                             Returns normed dict + (mean, std) for test time.

    Global stats (not per-class) preserve cross-class distance structure.
    Per-class z-scoring would artificially equalise intra-class variance,
    which would hurt density-based OoD detection.
    """
    if normalization == "none":
        return features_by_class, None, None

    all_feats = np.concatenate(
        [f for f in features_by_class.values() if len(f) > 0], axis=0)
    mean = all_feats.mean(axis=0)
    std  = all_feats.std(axis=0) + 1e-8

    normed = {c: (f - mean) / std if len(f) > 0 else f
              for c, f in features_by_class.items()}
    return normed, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# GMM fitting (CPU — fast, run for each grid config)
# ─────────────────────────────────────────────────────────────────────────────

def fit_gmms(features_by_class: dict,
             n_components: int,
             covariance_type: str,
             reg_covar: float,
             min_samples: int = 10) -> tuple:
    """
    Fit one GaussianMixture per class.
    Returns (gmms dict, total_bic, total_aic).
    """
    gmms      = {}
    total_bic = 0.0
    total_aic = 0.0

    for c, feats in features_by_class.items():
        if len(feats) < min_samples:
            gmms[c] = None
            continue
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            max_iter=300,
            random_state=0,
        )
        gmm.fit(feats)
        gmms[c]    = gmm
        total_bic += gmm.bic(feats)
        total_aic += gmm.aic(feats)

    return gmms, total_bic, total_aic


# ─────────────────────────────────────────────────────────────────────────────
# Grid search runner
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(features_by_class: dict,
                    feature_label: str,
                    seed_dir: str) -> list:
    """
    Run all 16 grid configurations on the given features.
    Save each fitted GMM (+ normalization stats if needed).
    Returns list of result dicts for the summary table.
    """
    results = []
    n_configs = len(GRID["normalization"]) * len(GRID["n_components"]) * \
                len(GRID["covariance_type"]) * len(GRID["reg_covar"])

    print(f"\n  Running {n_configs} GMM configurations for [{feature_label}] features...")

    for i, (config_id, cfg) in enumerate(all_configs(), 1):
        norm     = cfg["normalization"]
        n_comp   = cfg["n_components"]
        cov      = cfg["covariance_type"]
        reg      = cfg["reg_covar"]

        # Apply normalization
        normed_feats, mean, std = apply_normalization(features_by_class, norm)

        # Fit GMMs
        t_start = time.time()
        gmms, bic, aic = fit_gmms(normed_feats, n_comp, cov, reg)
        elapsed = time.time() - t_start

        n_fitted = sum(1 for g in gmms.values() if g is not None)
        all_converged = all(g.converged_ for g in gmms.values() if g is not None)

        # Save GMM pickle
        pkl_path = os.path.join(seed_dir, f"gmm_{feature_label}_{config_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(gmms, f)

        # Save normalization stats (needed at test time for zscore configs)
        if norm == "zscore" and mean is not None:
            stats_path = os.path.join(
                seed_dir, f"gmm_{feature_label}_{config_id}_stats.npz")
            np.savez(stats_path, mean=mean, std=std)

        row = {
            "feature":        feature_label,
            "config_id":      config_id,
            "normalization":  norm,
            "n_components":   n_comp,
            "covariance_type": cov,
            "reg_covar":      reg,
            "n_fitted":       n_fitted,
            "all_converged":  all_converged,
            "total_BIC":      round(bic, 2),
            "total_AIC":      round(aic, 2),
            "fit_sec":        round(elapsed, 2),
        }
        results.append(row)

        print(f"  [{i:2d}/{n_configs}] {config_id}")
        print(f"           BIC={bic:>12.1f}  AIC={aic:>12.1f}  "
              f"converged={all_converged}  ({elapsed:.1f}s)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DDU GMM Grid Search — normalization × n_components × "
                    "covariance_type × reg_covar")
    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir",  type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/ddu")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--feature_type", type=str, default="both",
        choices=["query", "spatial", "both"])
    parser.add_argument("--min_overlap",  type=float, default=0.1,
        help="Min query-GT mask overlap for query assignment")
    parser.add_argument("--max_pixels_per_class", type=int, default=300,
        help="Max pixels per class per image (spatial mode)")
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu",         type=str, default="0")
    parser.add_argument("--height",      type=int, default=1280)
    parser.add_argument("--width",       type=int, default=720)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_dir        = os.path.join(args.results_dir, f"seed_{args.seed}")
    checkpoint_path = os.path.join(seed_dir, "model_final.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    n_total = (len(GRID["normalization"]) * len(GRID["n_components"]) *
               len(GRID["covariance_type"]) * len(GRID["reg_covar"]))

    print(f"\n{'='*65}")
    print(f"  DDU GMM Grid Search")
    print(f"  Seed         : {args.seed}")
    print(f"  Feature type : {args.feature_type}")
    print(f"  Grid size    : {n_total} configs per feature type")
    print(f"  Axes         : normalization={GRID['normalization']}")
    print(f"                 n_components={GRID['n_components']}")
    print(f"                 covariance_type={GRID['covariance_type']}")
    print(f"                 reg_covar={GRID['reg_covar']}")
    print(f"  Checkpoint   : {checkpoint_path}")
    print(f"{'='*65}\n")

    model  = load_ddu_model(args.pretrained, checkpoint_path, device)
    loader = build_train_loader(args.coco_file, args.root_dir,
                                args.batch_size, args.num_workers)
    print(f"Training images: {len(loader.dataset)}\n")

    t0      = time.time()
    all_results = []

    # ── Query features: GPU extraction → 16 CPU fits ─────────────────────────
    if args.feature_type in ("query", "both"):
        print("=" * 65)
        print("  QUERY features  (transformer_decoder_last_hidden_state, 256-dim)")
        print("=" * 65)
        query_feats = extract_query_features(
            model, loader, device, args.height, args.width,
            min_overlap=args.min_overlap,
        )
        all_results += run_grid_search(query_feats, "query", seed_dir)

    # ── Spatial features: GPU extraction → 16 CPU fits ───────────────────────
    if args.feature_type in ("spatial", "both"):
        print("\n" + "=" * 65)
        print("  SPATIAL features  (masks_queries_logits upsampled, 100-dim)")
        print("=" * 65)
        spatial_feats = extract_spatial_features(
            model, loader, device, args.height, args.width,
            max_per_class_per_image=args.max_pixels_per_class,
        )
        all_results += run_grid_search(spatial_feats, "spatial", seed_dir)

    total_elapsed = (time.time() - t0) / 60

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  GMM Grid Search Summary  (seed={args.seed})")
    print(f"{'='*90}")
    print(f"  {'Feature':<8} {'Config ID':<52} {'BIC':>14} {'AIC':>14}  Converged")
    print(f"  {'─'*84}")
    for r in sorted(all_results, key=lambda x: (x["feature"], x["total_BIC"])):
        print(f"  {r['feature']:<8} {r['config_id']:<52} "
              f"{r['total_BIC']:>14.1f} {r['total_AIC']:>14.1f}  "
              f"{'✓' if r['all_converged'] else '✗'}")

    # Best config per feature type
    print(f"\n  Best config by BIC:")
    for ft in set(r["feature"] for r in all_results):
        ft_results = [r for r in all_results if r["feature"] == ft]
        best = min(ft_results, key=lambda x: x["total_BIC"])
        print(f"    [{ft}] {best['config_id']}  BIC={best['total_BIC']:.1f}")
    print(f"{'='*90}")
    print(f"\nTotal time: {total_elapsed:.1f} min")

    # ── Save JSON + CSV results ───────────────────────────────────────────────
    json_path = os.path.join(seed_dir, "gmm_grid_results.json")
    with open(json_path, "w") as f:
        json.dump({"seed": args.seed, "results": all_results}, f, indent=2)
    print(f"[Saved] {json_path}")

    csv_path = os.path.join(seed_dir, "gmm_grid_results.csv")
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"[Saved] {csv_path}")

    # ── Save grid config metadata ─────────────────────────────────────────────
    config_meta = {
        "seed": args.seed,
        "feature_type": args.feature_type,
        "grid": GRID,
        "min_overlap": args.min_overlap,
        "max_pixels_per_class": args.max_pixels_per_class,
        "train_images": len(loader.dataset),
        "total_configs": len(all_results),
    }
    with open(os.path.join(seed_dir, "gmm_config.json"), "w") as f:
        json.dump(config_meta, f, indent=2)
    print(f"[Saved] gmm_config.json")


if __name__ == "__main__":
    main()
