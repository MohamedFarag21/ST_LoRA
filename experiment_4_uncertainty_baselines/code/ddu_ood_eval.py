# -*- coding: utf-8 -*-
"""
Far OoD Detection — Mask2Former + DDU
======================================
In-distribution  : Sweet pepper test set
Far OoD          : GrowliFlower-L images

Scorers
-------
Softmax-based (same as all other methods, single deterministic pass):
  NegEnt  — negative mean image entropy
  MSP     — mean max softmax probability

DDU-specific (epistemic, from GMM log-likelihood):
  GMM_query   — mean log-likelihood under per-class query GMM
  GMM_spatial — mean log-likelihood under per-class spatial GMM

For each GMM scorer, the best config by BIC from the grid search is used.
Classes with no GMM (pepper_kp, pepper_mixed) are skipped per pixel.

Usage:
    python ddu_ood_eval.py --seed 42
    python ddu_ood_eval.py --seed 42 --seeds 42 123 456 789 1337
"""

import os
import json
import pickle
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import skimage.draw
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn.utils.parametrizations import spectral_norm as sn_p

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from image_ood_extra import (metrics_all_positive, aggregate_imagelevel,
                             TomatoImageOoD, TOMATO_ROOT_DEFAULT)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ID2LABEL_ORIG = {
    0: "bg", 11: "pepper_kp", 12: "pepper red", 13: "pepper yellow",
    14: "pepper green", 15: "pepper mixed", 17: "pepper mixed_red",
    18: "pepper mixed_yellow",
}
LABEL2ID   = {old: new for new, old in enumerate(sorted(ID2LABEL_ORIG.keys()))}
ID2LABEL   = {new: ID2LABEL_ORIG[old] for old, new in LABEL2ID.items()}
NUM_LABELS = len(ID2LABEL)
NUM_QUERIES = 100

# gmm_spatial evaluates a per-class GMM over every pixel (~920k px/img at native
# res) — the timeout hotspot. The image-level score is a MEAN over pixels, so a
# random pixel subsample is a near-unbiased estimate. 0 = use all pixels.
SPATIAL_SUBSAMPLE = 0
_SPATIAL_RNG = np.random.RandomState(0)

_REMAP_LUT = np.zeros(256, dtype=np.int64)
for old, new in LABEL2ID.items():
    _REMAP_LUT[old] = new

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Best GMM configs from grid search (by BIC, seed=42)
BEST_QUERY_CONFIG   = "norm=none_k=2_cov=tied_reg=1e-4"
BEST_SPATIAL_CONFIG = "norm=none_k=2_cov=full_reg=1e-4"


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class PepperTestDataset(Dataset):
    TEST_IDS = list(range(377, 408)) + list(range(471, 533))

    def __init__(self, coco_file, root_dir, transform=None):
        with open(coco_file) as f:
            data = json.load(f)
        valid_ids      = set(self.TEST_IDS)
        self.images    = [img for img in data["images"] if img["id"] in valid_ids]
        self.root_dir  = root_dir
        self.transform = transform
        self.ann_lookup: dict = {}
        for ann in data["annotations"]:
            self.ann_lookup.setdefault(ann["image_id"], []).append(ann)

    def __len__(self): return len(self.images)

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


class GrowliFlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.files = []

        for split in ["Train", "Val", "Test"]:
            img_dir  = os.path.join(root_dir, "images", split)
            mask_dir = os.path.join(root_dir, "labels", split, "maskPlants")
            if not os.path.isdir(img_dir):
                continue
            for img_path in sorted(Path(img_dir).glob("*.jpg")):
                # Skip images with no plants (matching comprehensive script)
                if os.path.isdir(mask_dir):
                    noplants = os.path.join(
                        mask_dir, f"{img_path.stem}_Label_NoPlants_maskPlants.png")
                    if os.path.exists(noplants):
                        continue
                self.files.append(str(img_path))

        if not self.files:
            raise RuntimeError(
                f"No images found under {root_dir}/images/{{Train,Val,Test}}/\n"
                f"Expected GrowliFlower-L directory structure.")
        print(f"[GrowliFlower] Found {len(self.files)} images "
              f"(NoPlants images skipped)")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        pil_img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)


def make_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])


def build_id_loader(coco_file, root_dir, batch_size, num_workers):
    preprocessor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=False,
        do_resize=False, do_rescale=False, do_normalize=False,
        num_labels=NUM_LABELS)
    dataset = PepperTestDataset(coco_file, root_dir, make_transform())

    def collate(batch):
        imgs, segs = zip(*batch)
        out = preprocessor(list(imgs), segmentation_maps=list(segs), return_tensors="pt")
        out["seg_maps"] = torch.stack(list(segs))
        return out

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


def build_ood_loader(growliflower_dir, batch_size, num_workers):
    preprocessor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=False,
        do_resize=False, do_rescale=False, do_normalize=False,
        num_labels=NUM_LABELS)
    dataset = GrowliFlowerDataset(growliflower_dir, make_transform())

    def collate(batch):
        return preprocessor(list(batch), return_tensors="pt")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


def build_tomato_ood_loader_ddu(root_dir, split, batch_size, num_workers,
                                H, W, max_images):
    """Near-OoD tomato images -> DDU dict batches ({'pixel_values': ...}),
    matching build_ood_loader's collate. Resizes to (H, W), ADE-normalised;
    deterministic RandomState(0) sub-sample to `max_images` (matches the UQ runs)."""
    preprocessor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=False,
        do_resize=False, do_rescale=False, do_normalize=False,
        num_labels=NUM_LABELS)
    tfm = transforms.Compose([transforms.Resize((H, W)), make_transform()])
    dataset = TomatoImageOoD(root_dir, split, tfm,
                             max_images=max_images, subsample_seed=0)

    def collate(batch):
        return preprocessor(list(batch), return_tensors="pt")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ddu_model(pretrained_name, checkpoint_path, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    visited = set()
    def _sn(mod):
        for name, child in mod.named_children():
            if id(child) in visited: continue
            visited.add(id(child))
            _sn(child)
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                try: sn_p(child)
                except Exception: pass
    _sn(model)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval().to(device)
    for p in model.parameters(): p.requires_grad = False
    print(f"[Loaded] {checkpoint_path}")
    return model


def load_gmms(seed_dir, config_id, feature_type):
    """Load fitted GMM dict for (feature_type, config_id). Returns None if missing."""
    path = os.path.join(seed_dir, f"gmm_{feature_type}_{config_id}.pkl")
    if not os.path.exists(path):
        print(f"  [WARN] GMM not found: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_norm_stats(seed_dir, config_id, feature_type):
    """Load z-score stats if config uses normalization. Returns (None, None) otherwise."""
    path = os.path.join(seed_dir, f"gmm_{feature_type}_{config_id}_stats.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["mean"], data["std"]


# ─────────────────────────────────────────────────────────────────────────────
# Inference — single deterministic pass, collect softmax + features
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_scores(model, dataloader, device, H, W,
                   query_gmms, spatial_gmms,
                   query_norm_mean, query_norm_std,
                   spatial_norm_mean, spatial_norm_std,
                   is_ood=False):
    """
    Single forward pass per batch.
    Returns dict of per-image scorer arrays.
    """
    ent_list       = []
    gmm_query_list = []
    gmm_spat_list  = []

    for batch in tqdm(dataloader, desc="  Inference", leave=False):
        pv = batch["pixel_values"].to(device)
        outputs = model(pv, output_hidden_states=True)

        # ── Softmax probs ────────────────────────────────────────────────────
        cp  = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        mp  = outputs.masks_queries_logits.sigmoid()
        seg = torch.einsum("bqc,bqhw->bchw", cp, mp)
        seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
        seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)

        H_map = -(seg * torch.log2(seg + 1e-12)).sum(dim=1)    # (B, H, W)
        ent_list.append(H_map.mean(dim=(1, 2)).cpu().numpy())  # higher = more uncertain

        # ── GMM query scorer ─────────────────────────────────────────────────
        if query_gmms is not None:
            qf = outputs.transformer_decoder_last_hidden_state  # (B, 100, 256)
            B  = pv.shape[0]
            img_scores = np.zeros(B)
            for b in range(B):
                qf_b = qf[b].cpu().numpy()   # (100, 256)
                if query_norm_mean is not None:
                    qf_b = (qf_b - query_norm_mean) / (query_norm_std + 1e-8)
                ll = _gmm_score_queries(qf_b, query_gmms)
                img_scores[b] = ll
            gmm_query_list.append(img_scores)

        # ── GMM spatial scorer ───────────────────────────────────────────────
        # FAITHFUL reproduction of ood_eval_comprehensive_with_ddu.py (option B):
        #   ID  : score sf at FULL (H,W) under each pixel's PREDICTED class GMM (pixel-mean)
        #   OoD : score sf at 512×512 under MAX-over-class GMM (pixel-mean)
        # This asymmetric-resolution + asymmetric-reduction recipe is what produces the
        # reference density_spatial_AUROC≈0.94 (ood_v2). It is fragile by construction
        # (ood_v2_fixed's different choices gave ≈0.37) but matches the reference.
        if spatial_gmms is not None:
            if is_ood:
                Ho, Wo = min(H, 512), min(W, 512)
                sf = F.interpolate(outputs.masks_queries_logits.sigmoid(),
                                   size=(Ho, Wo), mode="bilinear", align_corners=False)
                sf = sf.permute(0, 2, 3, 1).cpu().numpy()          # (B, Ho, Wo, 100)
            else:
                sf = F.interpolate(outputs.masks_queries_logits.sigmoid(),
                                   size=(H, W), mode="bilinear", align_corners=False)
                sf = sf.permute(0, 2, 3, 1).cpu().numpy()          # (B, H, W, 100)
                seg_pred = seg.argmax(dim=1).cpu().numpy()          # (B, H, W) semantic argmax
            B  = pv.shape[0]
            img_scores = np.zeros(B)
            for b in range(B):
                sf_b = sf[b].reshape(-1, NUM_QUERIES)
                if is_ood:
                    if SPATIAL_SUBSAMPLE and sf_b.shape[0] > SPATIAL_SUBSAMPLE:
                        idx  = _SPATIAL_RNG.choice(sf_b.shape[0], SPATIAL_SUBSAMPLE, replace=False)
                        sf_b = sf_b[idx]
                    if spatial_norm_mean is not None:
                        sf_b = (sf_b - spatial_norm_mean) / (spatial_norm_std + 1e-8)
                    ll = _gmm_score_all_pixels(sf_b, spatial_gmms)         # max-over-class
                else:
                    seg_b = seg_pred[b].ravel()
                    if SPATIAL_SUBSAMPLE and sf_b.shape[0] > SPATIAL_SUBSAMPLE:
                        idx   = _SPATIAL_RNG.choice(sf_b.shape[0], SPATIAL_SUBSAMPLE, replace=False)
                        sf_b  = sf_b[idx]; seg_b = seg_b[idx]
                    if spatial_norm_mean is not None:
                        sf_b = (sf_b - spatial_norm_mean) / (spatial_norm_std + 1e-8)
                    ll = _gmm_score_pixels_by_class(sf_b, seg_b, spatial_gmms)  # predicted-class
                img_scores[b] = ll
            gmm_spat_list.append(img_scores)

    # For AUROC: higher score = more OoD.
    # Entropy: higher = more uncertain = more OoD ✓ — no negation needed.
    # GMM log-likelihood: higher = more ID (high density) → negate so OoD gets higher score.
    out = {
        "entropy": np.concatenate(ent_list),
    }
    if query_gmms is not None:
        out["gmm_query"]   = -np.concatenate(gmm_query_list)   # negate: low density = OoD
    if spatial_gmms is not None:
        out["gmm_spatial"] = -np.concatenate(gmm_spat_list)    # negate: low density = OoD
    return out


def _gmm_score_queries(qf_b, gmms):
    """
    Mean log-likelihood over all queries that have a valid GMM assignment.
    qf_b: (100, 256)
    """
    lls = []
    for q_vec in qf_b:
        # Score against all available class GMMs, take max (most likely class)
        best_ll = None
        for c, gmm in gmms.items():
            if gmm is None: continue
            try:
                ll = float(gmm.score(q_vec.reshape(1, -1)))
                if best_ll is None or ll > best_ll:
                    best_ll = ll
            except Exception:
                continue
        if best_ll is not None:
            lls.append(best_ll)
    return float(np.mean(lls)) if lls else 0.0


def _gmm_score_pixels_by_class(sf_b, pred_classes, gmms):
    """
    Per-pixel: score each pixel feature under ITS predicted-class GMM, then take the
    PIXEL-weighted mean — faithfully mirroring the validated original
    (ood_eval_comprehensive_with_ddu._gmm_score_spatial_pixel). Pixels whose predicted
    class has no GMM fall back to the best-class (max) score.
    NOTE: the previous version averaged per-CLASS means (class-weighted), which inverted
    the image-level AUROC vs the pixel-weighted reference (0.94 -> ~0.001).
    sf_b: (N_px, 100), pred_classes: (N_px,)
    """
    available = [(c, gmm) for c, gmm in gmms.items() if gmm is not None]
    if not available:
        return 0.0
    scores = np.full(len(sf_b), np.nan, dtype=np.float64)
    for c, gmm in available:
        mask = (pred_classes == c)
        if mask.any():
            scores[mask] = gmm.score_samples(sf_b[mask])
    nan_mask = np.isnan(scores)
    if nan_mask.any():
        all_ll = np.stack([gmm.score_samples(sf_b[nan_mask]) for _, gmm in available], axis=1)
        scores[nan_mask] = all_ll.max(axis=1)
    valid = ~np.isnan(scores)
    return float(scores[valid].mean()) if valid.any() else 0.0


def _gmm_score_all_pixels(sf_b, gmms):
    """
    OoD images have no GT/pred class labels — score each pixel under all
    class GMMs and take the max (best-fitting class). Mean over all pixels.
    """
    available = [gmm for gmm in gmms.values() if gmm is not None]
    if not available:
        return 0.0
    # Stack scores from all classes, take per-pixel max
    all_scores = np.stack([gmm.score_samples(sf_b) for gmm in available], axis=1)
    return float(all_scores.max(axis=1).mean())


# ─────────────────────────────────────────────────────────────────────────────
# OoD metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ood_metrics(id_scores, ood_scores):
    """AUROC/AUPR/FPR95. Higher score = more OoD (DDU convention, no negation)."""
    return metrics_all_positive(id_scores, ood_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DDU OoD Evaluation")
    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir",  type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--growliflower_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/growliflower_l")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/ddu")
    parser.add_argument("--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1337])
    parser.add_argument("--query_config",   type=str, default=BEST_QUERY_CONFIG)
    parser.add_argument("--spatial_config", type=str, default=BEST_SPATIAL_CONFIG)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu",         type=str, default="0")
    parser.add_argument("--height",      type=int, default=1280)
    parser.add_argument("--width",       type=int, default=720)
    parser.add_argument("--tomato_root",  type=str, default=TOMATO_ROOT_DEFAULT)
    parser.add_argument("--tomato_split", type=str, default="val")
    parser.add_argument("--tomato_max",   type=int, default=1200)
    parser.add_argument("--skip_spatial", action="store_true",
        help="Skip the all-pixel gmm_spatial scorer (~920k px/img GMM eval): the "
             "image-level timeout hotspot and weak in ood_v2_fixed (~0.37). Keeps "
             "entropy (headline) + gmm_query (DDU's strong density scorer).")
    parser.add_argument("--spatial_px_subsample", type=int, default=0,
        help="Random pixels/image for gmm_spatial (0=all). The image-level "
             "spatial score is a pixel MEAN, so subsampling is a near-unbiased "
             "estimate that makes the scorer tractable (e.g. 8192).")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global SPATIAL_SUBSAMPLE
    SPATIAL_SUBSAMPLE = args.spatial_px_subsample

    SCORERS = ["entropy", "gmm_query"] if args.skip_spatial \
              else ["entropy", "gmm_query", "gmm_spatial"]
    SOURCES = ["growliflower", "tomato"]      # far-OoD, near-OoD (image-level)

    id_loader  = build_id_loader( args.coco_file, args.root_dir,
                                   args.batch_size, args.num_workers)
    ood_loader = build_ood_loader(args.growliflower_dir,
                                   args.batch_size, args.num_workers)
    tom_loader = build_tomato_ood_loader_ddu(
        args.tomato_root, args.tomato_split, args.batch_size, args.num_workers,
        args.height, args.width, args.tomato_max)

    print(f"ID images     : {len(id_loader.dataset)}")
    print(f"OoD growli    : {len(ood_loader.dataset)}")
    print(f"OoD tomato    : {len(tom_loader.dataset)}\n")

    seed_results = []

    for seed in args.seeds:
        seed_dir   = os.path.join(args.results_dir, f"seed_{seed}")
        ckpt_path  = os.path.join(seed_dir, "model_final.pt")

        print(f"\n{'='*55}\n  Seed {seed}\n{'='*55}")

        model = load_ddu_model(args.pretrained, ckpt_path, device)

        # Load GMMs (spatial skipped for image-level speed unless requested)
        query_gmms = load_gmms(seed_dir, args.query_config,   "query")
        spat_gmms  = None if args.skip_spatial \
                     else load_gmms(seed_dir, args.spatial_config, "spatial")

        q_mean, q_std = load_norm_stats(seed_dir, args.query_config,   "query")
        s_mean, s_std = (None, None) if args.skip_spatial \
                        else load_norm_stats(seed_dir, args.spatial_config, "spatial")

        t0 = time.time()
        id_scores  = compute_scores(model, id_loader,  device, args.height, args.width,
                                    query_gmms, spat_gmms, q_mean, q_std, s_mean, s_std,
                                    is_ood=False)
        ood_grow   = compute_scores(model, ood_loader, device, args.height, args.width,
                                    query_gmms, spat_gmms, q_mean, q_std, s_mean, s_std,
                                    is_ood=True)
        ood_tom    = compute_scores(model, tom_loader, device, args.height, args.width,
                                    query_gmms, spat_gmms, q_mean, q_std, s_mean, s_std,
                                    is_ood=True)
        elapsed = (time.time() - t0) / 60
        ood_by_src = {"growliflower": ood_grow, "tomato": ood_tom}

        entry = {"seed": seed}
        for src in SOURCES:
            ood_scores = ood_by_src[src]
            print(f"\n  [{src}]  {'Scorer':<14} {'AUROC':>8}  {'FPR95':>8}")
            for name in SCORERS:
                if name not in id_scores: continue
                m = compute_ood_metrics(id_scores[name], ood_scores[name])
                for mk in ("AUROC", "AUPR", "FPR95"):
                    entry[f"{src}::{name}::{mk}"] = m[mk]
                print(f"          {name:<14} {m['AUROC']:>8.4f}  {m['FPR95']:>8.4f}")
        print(f"\n  Time: {elapsed:.1f} min")
        seed_results.append(entry)
        del model; torch.cuda.empty_cache()

    meta = {"method": "DDU", "seeds": args.seeds,
            "id_dataset": "sweet_pepper_test",
            "tomato_split": args.tomato_split, "tomato_max": args.tomato_max,
            "headline_scorer": "entropy (mean per-pixel entropy)"}
    summary = aggregate_imagelevel(seed_results, SCORERS, SOURCES, meta)

    out_json = os.path.join(args.results_dir, "ood_summary_imagelevel.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\n  DDU image-level OoD Summary ({len(seed_results)} seeds)\n{'='*70}")
    for src in SOURCES:
        print(f"  ── {src} ──")
        print(f"  {'Scorer':<14} {'AUROC':>16}  {'FPR95':>16}")
        for name in SCORERS:
            au = summary[src][name]["AUROC"]; fp = summary[src][name]["FPR95"]
            print(f"  {name:<14} {au['mean']:.4f}±{au['std']:.4f}  "
                  f"{fp['mean']:.4f}±{fp['std']:.4f}")
    print(f"\n[Saved] {out_json}")


if __name__ == "__main__":
    main()