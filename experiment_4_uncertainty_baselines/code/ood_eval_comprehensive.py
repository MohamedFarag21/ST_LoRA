# -*- coding: utf-8 -*-
"""
Comprehensive OoD Evaluation — LoRA / Full FT / MC Dropout
============================================================
Two-part evaluation using GrowliFlower-L with segmentation masks:

PART 1 — Image-level OoD
  - Entropy and MI histograms: GrowliFlower background vs foreground
  - Image-level AUROC: sweet pepper (ID) vs GrowliFlower (OoD)

PART 2 — Pixel-level OoD (anomaly segmentation)
  - Cauliflower pixels = anomaly (1), background pixels = normal (0)
  - Metrics: AUROC, AUPR, FPR95 (pixel), sIoU, PPV, MeanF1 (component)
  - Histogram comparison: sweet pepper bg/fg vs cauliflower bg/fg

Scorers: entropy and MI only (no MSP, no negation — higher = more OoD)

Usage:
    python ood_eval_comprehensive.py \\
        --growliflower_dir /path/to/growliflower \\
        --growliflower_mask_dir /path/to/growliflower/masks \\
        --method lora --seed 42
"""

import os
import json
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
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import skimage.draw
from tqdm import tqdm

from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve)

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from peft import PeftModel

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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

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
# Datasets
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


class GrowliFlowerDataset(Dataset):
    """
    GrowliFlower-L dataset with plant segmentation masks.

    Directory structure:
        root/images/{Train,Val,Test}/*.jpg
        root/labels/{Train,Val,Test}/maskPlants/*.png

    Label naming:
        {stem}_Label_maskPlants.png        → has plants  (use this)
        {stem}_Label_NoPlants_maskPlants.png → no plants (skip)

    Binary mask: 0 = background, >0 = cauliflower plant (anomaly).
    Uses all splits (Train/Val/Test) for OoD evaluation.
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.pairs     = []

        for split in ["Train", "Val", "Test"]:
            img_dir  = os.path.join(root_dir, "images", split)
            mask_dir = os.path.join(root_dir, "labels", split, "maskPlants")
            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
                continue

            for img_path in sorted(Path(img_dir).glob("*.jpg")):
                stem = img_path.stem
                mask_path = os.path.join(mask_dir, f"{stem}_Label_maskPlants.png")
                noplants  = os.path.join(mask_dir, f"{stem}_Label_NoPlants_maskPlants.png")

                # Skip images with no plants
                if os.path.exists(noplants):
                    continue
                if not os.path.exists(mask_path):
                    continue

                self.pairs.append((str(img_path), mask_path))

        if not self.pairs:
            raise RuntimeError(
                f"No valid image-mask pairs found in {root_dir}.\n"
                f"Expected structure: images/{{Train,Val,Test}}/*.jpg\n"
                f"                   labels/{{Train,Val,Test}}/maskPlants/*.png")
        print(f"[GrowliFlower-L] {len(self.pairs)} image-mask pairs "
              f"(skipped NoPlants images)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        pil_img  = Image.open(img_path).convert("RGB")
        pil_mask = Image.open(msk_path).convert("L")
        image    = self.transform(pil_img) if self.transform \
                   else transforms.ToTensor()(pil_img)

        raw = np.array(pil_mask)

        # Diagnostic: print unique values for the first image only
        if idx == 0:
            uniq = np.unique(raw)
            print(f"[GrowliFlower mask debug] unique pixel values in first mask: {uniq} "
                  f"(shape={raw.shape}, path={os.path.basename(msk_path)})")

        # GrowliFlower masks: determine plant pixels robustly.
        # If only two unique values exist, the smaller = background, larger = plant.
        # If three+ values, treat 0 = background, all others = plant.
        uniq_vals = np.unique(raw)
        if len(uniq_vals) == 2:
            plant_val = int(uniq_vals.max())
            binary = (raw == plant_val).astype(np.uint8)
        else:
            # 0 = background, >0 = plant (original assumption)
            binary = (raw > 0).astype(np.uint8)

        mask = torch.from_numpy(binary)
        return image, mask


def make_transform(H=None, W=None):
    ops = []
    if H is not None and W is not None:
        ops.append(transforms.Resize((H, W)))
    ops += [transforms.ToTensor(),
            transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)]
    return transforms.Compose(ops)


def build_id_loader(coco_file, root_dir, batch_size, num_workers):
    preprocessor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=False,
        do_resize=False, do_rescale=False, do_normalize=False,
        num_labels=NUM_LABELS,
    )
    ds = SweetPepperTestDataset(coco_file, root_dir, transform=make_transform())

    def collate(batch):
        images, seg_maps = zip(*batch)
        out = preprocessor(list(images), segmentation_maps=list(seg_maps),
                           return_tensors="pt")
        out["seg_maps"] = list(seg_maps)
        return out

    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


def build_ood_loader(root_dir, H, W, batch_size, num_workers):
    ds = GrowliFlowerDataset(root_dir, transform=make_transform(H, W))

    def collate(batch):
        # Filter out None items (safety)
        batch = [b for b in batch if b is not None]
        images, masks = zip(*batch)
        # Resize masks to (H, W) using nearest neighbour
        masks_r = [torch.from_numpy(
                       np.array(Image.fromarray(m.numpy())
                                .resize((W, H), Image.NEAREST)))
                   for m in masks]
        return torch.stack(images), torch.stack(masks_r)

    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(pretrained_name, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
    )
    return model.eval().to(device)


def load_lora_adapter(base_model, adapter_path, device):
    return PeftModel.from_pretrained(base_model, adapter_path).eval().to(device)


def load_fullft_snapshot(pretrained_name, ckpt_path, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval().to(device)


def load_mc_model(pretrained_name, ckpt_path, dropout_p, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
    )
    original = model.class_predictor
    model.class_predictor = nn.Sequential(nn.Dropout(p=dropout_p), original)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device)
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Seg probs builder
# ─────────────────────────────────────────────────────────────────────────────

def build_seg_probs(outputs, H, W):
    cp  = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
    mp  = outputs.masks_queries_logits.sigmoid()
    seg = torch.einsum("bqc,bqhw->bchw", cp, mp)
    seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
    return F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Inference — accumulate sum_probs and sum_H_fg across ensemble
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_ensemble_id(models_or_passes_fn, dataloader, device, H, W, M):
    """
    Returns per-image dicts with entropy and MI maps.
    models_or_passes_fn: callable(m) -> model for shot m (LoRA/FullFT)
                      or callable(t) -> same model for MC Dropout
    """
    N = len(dataloader.dataset)
    sum_probs = torch.zeros(N, NUM_LABELS, H, W)
    sum_H     = torch.zeros(N, H, W)
    gts       = [None] * N

    for m in range(M):
        model = models_or_passes_fn(m)
        idx   = 0
        for batch in tqdm(dataloader, desc=f"  ID pass {m+1}/{M}", leave=False):
            pv  = batch["pixel_values"].to(device)
            seg = build_seg_probs(model(pv), H, W)
            B   = seg.shape[0]
            H_t = -(seg * (seg + 1e-12).log2()).sum(dim=1)
            sum_probs[idx:idx+B] += seg.cpu()
            sum_H[idx:idx+B]     += H_t.cpu()
            if m == 0:
                for b, gt in enumerate(batch["seg_maps"]):
                    gts[idx+b] = gt
            idx += B
        if hasattr(model, 'base_model'):   # PEFT model — delete cleanly
            del model; torch.cuda.empty_cache()

    mean_probs = sum_probs / M
    mean_H     = sum_H     / M
    H_mean     = -(mean_probs * (mean_probs + 1e-12).log2()).sum(dim=1)
    MI         = H_mean - mean_H   # (N, H, W) — higher = more epistemic

    return {"mean_probs": mean_probs, "entropy": H_mean, "MI": MI, "gts": gts}


@torch.no_grad()
def run_ensemble_ood(models_or_passes_fn, dataloader, device, H, W, M):
    """
    Memory-efficient OoD inference.

    Problem: pre-allocating (N, C, H, W) for ~2000 GrowliFlower images
    requires ~65GB RAM and causes OOM.

    Solution: compute entropy and MI incrementally per batch without
    storing full prob tensors. We keep:
      - sum_mean_probs (N, C, H, W) as float16 — still large but ~32GB
      Actually still too large. Instead compute H(mean_p) in two passes:

    Pass strategy: accumulate sum_probs per-batch in float16 to disk via
    numpy memmap, then compute entropy in a final pass.

    Simpler: store only sum_probs as float16 numpy array (half the RAM).
    2000 × 9 × 1280 × 720 × 2 bytes = ~33GB — still too large.

    Best approach: single-pass MI approximation.
    Store only:
      - sum_H     : (N, H, W) float32 — mean of per-pass entropies ~7GB
      - sum_probs : (N, C, H, W) float16 — for H(mean_p) ~33GB

    Since sum_probs is too large, use a streaming approach:
    Compute entropy of mean_probs in M passes simultaneously by
    storing the running mean probs as float16 per image on CPU.
    For GrowliFlower (~2000 images) at 1280×720 this is unavoidable
    unless we subsample.

    Practical fix: process images at reduced resolution for OoD scoring
    then upsample the entropy/MI maps back to full resolution.
    OoD scoring does not require full resolution — we use 256×256 internally.
    """
    N = len(dataloader.dataset)

    # Reduced internal resolution for memory efficiency
    H_ood = min(H, 512)
    W_ood = min(W, 512)

    # float16 to halve memory: 2000 × 9 × 512 × 512 × 2 bytes ≈ 9.4GB
    sum_probs = np.zeros((N, NUM_LABELS, H_ood, W_ood), dtype=np.float32)
    sum_H     = np.zeros((N, H_ood, W_ood),             dtype=np.float32)
    masks_all = [None] * N

    for m in range(M):
        model = models_or_passes_fn(m)
        idx   = 0
        for images, masks in tqdm(dataloader, desc=f"  OoD pass {m+1}/{M}", leave=False):
            pv  = images.to(device)
            seg = build_seg_probs(model(pv), H_ood, W_ood)   # (B, C, H_ood, W_ood)
            B   = seg.shape[0]

            H_t = -(seg * (seg + 1e-12).log2()).sum(dim=1)   # (B, H_ood, W_ood)
            sum_probs[idx:idx+B] += seg.cpu().numpy()
            sum_H[idx:idx+B]     += H_t.cpu().numpy()

            if m == 0:
                for b in range(B):
                    masks_all[idx+b] = masks[b]
            idx += B

        if hasattr(model, 'base_model'):
            del model; torch.cuda.empty_cache()

    # Compute entropy of mean probs and MI
    mean_probs = sum_probs / M                                # (N, C, H_ood, W_ood)
    mean_H     = sum_H / M                                    # (N, H_ood, W_ood)

    # Entropy of mean (total uncertainty)
    H_mean = -(mean_probs * np.log2(mean_probs + 1e-12)).sum(axis=1)  # (N, H_ood, W_ood)
    MI     = H_mean - mean_H                                  # (N, H_ood, W_ood) epistemic

    # Upsample entropy/MI back to full resolution if needed
    if H_ood != H or W_ood != W:
        print(f"  [OoD] Upsampling entropy/MI from {H_ood}×{W_ood} to {H}×{W}...")
        H_mean_up = np.stack([
            np.array(Image.fromarray(H_mean[i]).resize((W, H), Image.BILINEAR))
            for i in range(N)
        ])
        MI_up = np.stack([
            np.array(Image.fromarray(MI[i]).resize((W, H), Image.BILINEAR))
            for i in range(N)
        ])
    else:
        H_mean_up = H_mean
        MI_up     = MI

    # Convert to tensors for compatibility with downstream functions
    return {
        "entropy": torch.from_numpy(H_mean_up),
        "MI":      torch.from_numpy(MI_up),
        "masks":   masks_all,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Image-level analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_image_level_scores(result, masks_key="gts", is_id=True):
    """
    Returns per-image entropy and MI scalars.
    For ID images: uses gt segmentation to separate bg/fg.
    For OoD images: uses cauliflower masks to separate bg/fg.
    """
    N   = result["entropy"].shape[0]
    ent = result["entropy"]   # (N, H, W)
    mi  = result["MI"]        # (N, H, W)
    masks = result[masks_key]

    img_ent, img_mi = [], []
    bg_ent, fg_ent  = [], []
    bg_mi,  fg_mi   = [], []

    for i in range(N):
        m   = masks[i].numpy() if hasattr(masks[i], "numpy") else np.array(masks[i])
        e   = ent[i].numpy()
        mi_ = mi[i].numpy()

        img_ent.append(float(e.mean()))
        img_mi.append(float(mi_.mean()))

        if is_id:
            bg = m == 0
            fg = (m != 0) & (m != 255)
        else:
            bg = m == 0
            fg = m > 0    # cauliflower (mask is 0=bg, >0=plant)

        if bg.any():
            bg_ent.append(float(e[bg].mean()))
            bg_mi.append(float(mi_[bg].mean()))
        if fg.any():
            fg_ent.append(float(e[fg].mean()))
            fg_mi.append(float(mi_[fg].mean()))

    return {
        "img_ent": np.array(img_ent),
        "img_mi":  np.array(img_mi),
        "bg_ent":  np.array(bg_ent),
        "fg_ent":  np.array(fg_ent),
        "bg_mi":   np.array(bg_mi),
        "fg_mi":   np.array(fg_mi),
    }


def compute_image_auroc(id_scores, ood_scores):
    """AUROC: ID=0, OoD=1. Higher score = more OoD."""
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return float("nan")
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Pixel-level OoD metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_fpr95(labels, scores):
    """FPR at 95% TPR."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def compute_component_metrics(gt_mask, pred_mask, iou_thresh=0.25):
    """
    Component-level metrics from SMIYC benchmark.
    gt_mask:   binary (H,W) — 1=anomaly ground truth
    pred_mask: binary (H,W) — 1=predicted anomaly

    Returns sIoU (mean adjusted IoU of TP components),
            PPV (component precision),
            F1 (component harmonic mean)
    """
    gt_labeled,   n_gt   = ndimage.label(gt_mask)
    pred_labeled, n_pred = ndimage.label(pred_mask)

    if n_gt == 0:
        return float("nan"), float("nan"), float("nan")

    tp, fp, fn = 0, 0, 0
    sious = []

    for gt_id in range(1, n_gt + 1):
        gt_comp = (gt_labeled == gt_id)
        # Find overlapping predicted components
        overlap_ids = np.unique(pred_labeled[gt_comp])
        overlap_ids = overlap_ids[overlap_ids > 0]

        best_iou = 0.0
        for pid in overlap_ids:
            pred_comp = (pred_labeled == pid)
            inter = (gt_comp & pred_comp).sum()
            # Adjusted IoU: exclude intersection with other gt components
            other_gt = (gt_labeled > 0) & (gt_labeled != gt_id)
            union = gt_comp.sum() + pred_comp.sum() - inter - (pred_comp & other_gt).sum()
            if union > 0:
                iou = inter / union
                best_iou = max(best_iou, iou)

        if best_iou >= iou_thresh:
            tp += 1
            sious.append(best_iou)
        else:
            fn += 1

    for pred_id in range(1, n_pred + 1):
        pred_comp = (pred_labeled == pred_id)
        overlap_ids = np.unique(gt_labeled[pred_comp])
        overlap_ids = overlap_ids[overlap_ids > 0]
        matched = False
        for gid in overlap_ids:
            gt_comp = (gt_labeled == gid)
            inter   = (gt_comp & pred_comp).sum()
            other_gt = (gt_labeled > 0) & (gt_labeled != gid)
            union    = gt_comp.sum() + pred_comp.sum() - inter - (pred_comp & other_gt).sum()
            if union > 0 and inter / union >= iou_thresh:
                matched = True
                break
        if not matched:
            fp += 1

    siou = float(np.mean(sious)) if sious else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * ppv * rec / (ppv + rec) if (ppv + rec) > 0 else 0.0
    return siou, ppv, f1


def compute_pixel_ood_metrics(ood_result, scorer="entropy", n_thresh=50):
    """
    Pixel-level OoD: cauliflower pixels (mask=1) are anomalies.
    Returns AUROC, AUPR, FPR95, sIoU, PPV, MeanF1.
    """
    all_scores, all_labels = [], []
    pred_masks, gt_masks   = [], []

    N = len(ood_result["masks"])
    scores_map = ood_result[scorer]   # (N, H, W)

    for i in range(N):
        m  = ood_result["masks"][i].numpy()
        s  = scores_map[i].numpy()

        # Only evaluate over pixels that have a label (mask is 0=bg, >0=plant)
        valid = (m >= 0)   # all pixels are valid (binary mask)
        all_scores.append(s[valid].ravel())
        all_labels.append((m[valid] > 0).astype(np.uint8))

    flat_scores = np.concatenate(all_scores).astype(np.float32)
    flat_labels = np.concatenate(all_labels).astype(np.uint8)

    # Diagnostic: print mask value distribution
    unique, counts = np.unique(flat_labels, return_counts=True)
    print(f"  [pixel OoD] label distribution: { {int(u): int(c) for u, c in zip(unique, counts)} }")

    if len(np.unique(flat_labels)) < 2:
        print(f"  [pixel OoD] WARNING: only one class in labels — skipping AUROC/AUPR/FPR95")
        return {
            "AUROC": float("nan"), "AUPR": float("nan"), "FPR95": float("nan"),
            "sIoU":  float("nan"), "PPV":  float("nan"), "MeanF1": float("nan"),
            "threshold": float("nan"),
        }

    auroc = float(roc_auc_score(flat_labels, flat_scores))
    aupr  = float(average_precision_score(flat_labels, flat_scores))
    fpr95 = compute_fpr95(flat_labels, flat_scores)

    # Component-level: find optimal threshold via F1 on pixel level
    thresholds = np.percentile(flat_scores, np.linspace(50, 99, n_thresh))
    best_f1, best_thresh = -1, thresholds[0]
    for t in thresholds:
        pred  = (flat_scores >= t).astype(np.uint8)
        tp = ((pred == 1) & (flat_labels == 1)).sum()
        fp = ((pred == 1) & (flat_labels == 0)).sum()
        fn = ((pred == 0) & (flat_labels == 1)).sum()
        f1 = 2*tp / (2*tp + fp + fn + 1e-6)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    sious, ppvs, f1s = [], [], []
    for i in range(N):
        m    = ood_result["masks"][i].numpy()
        s    = scores_map[i].numpy()
        pred = (s >= best_thresh).astype(np.uint8)
        gt   = (m > 0).astype(np.uint8)
        if gt.sum() == 0:
            continue
        siou, ppv, f1 = compute_component_metrics(gt, pred)
        if not np.isnan(siou):
            sious.append(siou); ppvs.append(ppv); f1s.append(f1)

    return {
        "AUROC":   auroc,
        "AUPR":    aupr,
        "FPR95":   fpr95,
        "sIoU":    float(np.mean(sious)) if sious else float("nan"),
        "PPV":     float(np.mean(ppvs))  if ppvs  else float("nan"),
        "MeanF1":  float(np.mean(f1s))   if f1s   else float("nan"),
        "threshold": float(best_thresh),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_entropy_mi_histograms(id_scores, ood_scores, out_path):
    """
    2 rows × 2 cols:
      Row 1: Entropy — BG vs FG histograms for ID (sweet pepper)
      Row 2: Entropy — BG vs FG histograms for OoD (GrowliFlower)
      Plus MI versions side by side
    """
    COLOR_BG = "#1f77b4"
    COLOR_FG = "#ff7f0e"
    ALPHA    = 0.60

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    titles = [
        ("Sweet Pepper (ID) — Entropy",    id_scores,  "entropy"),
        ("Sweet Pepper (ID) — MI",          id_scores,  "mi"),
        ("GrowliFlower-L (OoD) — Entropy", ood_scores, "entropy"),
        ("GrowliFlower-L (OoD) — MI",      ood_scores, "mi"),
    ]

    # Key suffix mapping: "entropy" -> "ent", "mi" -> "mi"
    _key = {"entropy": "ent", "mi": "mi"}

    for ax, (title, scores, metric) in zip(axes.ravel(), titles):
        k  = _key[metric]
        bg = scores[f"bg_{k}"]
        fg = scores[f"fg_{k}"]

        # Skip subplot if either array is empty
        if len(bg) == 0 or len(fg) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=LABEL_FONT)
            ax.set_title(title, fontsize=LABEL_FONT)
            continue

        lo  = min(bg.min(), fg.min())
        hi  = max(bg.max(), fg.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(bg, bins=bins, density=True, alpha=ALPHA, color=COLOR_BG,
                label=f"Background (μ={bg.mean():.3f})", linewidth=0)
        ax.hist(fg, bins=bins, density=True, alpha=ALPHA, color=COLOR_FG,
                label=f"{'Cauliflower' if 'OoD' in title else 'Pepper'} (μ={fg.mean():.3f})",
                linewidth=0)
        ax.axvline(bg.mean(), color=COLOR_BG, linewidth=1.6, linestyle="--")
        ax.axvline(fg.mean(), color=COLOR_FG, linewidth=1.6, linestyle="--")
        ax.set_title(title, fontsize=LABEL_FONT)
        ax.set_xlabel("Entropy (bits)" if metric == "entropy" else "Mutual Information (bits)",
                      fontsize=LABEL_FONT)
        ax.set_ylabel("Density", fontsize=LABEL_FONT)
        ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
        ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=TICK_FONT, framealpha=0.85)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_image_auroc_comparison(id_img, ood_img, method_name, out_path):
    """Histogram of image-level entropy: ID vs OoD with AUROC annotated."""
    COLOR_ID  = "#2196F3"
    COLOR_OOD = "#FF5722"
    ALPHA     = 0.60

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, scorer, xlabel in [
        (axes[0], "img_ent", "Mean image entropy (bits)"),
        (axes[1], "img_mi",  "Mean image MI (bits)"),
    ]:
        id_s  = id_img[scorer]
        ood_s = ood_img[scorer]
        auroc = compute_image_auroc(id_s, ood_s)

        lo   = min(id_s.min(), ood_s.min())
        hi   = max(id_s.max(), ood_s.max())
        bins = np.linspace(lo, hi, 35)

        ax.hist(id_s,  bins=bins, density=True, alpha=ALPHA, color=COLOR_ID,
                label=f"Sweet Pepper ID (n={len(id_s)})", linewidth=0)
        ax.hist(ood_s, bins=bins, density=True, alpha=ALPHA, color=COLOR_OOD,
                label=f"GrowliFlower OoD (n={len(ood_s)})", linewidth=0)
        ax.axvline(id_s.mean(),  color=COLOR_ID,  linewidth=1.6, linestyle="--")
        ax.axvline(ood_s.mean(), color=COLOR_OOD, linewidth=1.6, linestyle="--")
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT)
        ax.set_ylabel("Density", fontsize=LABEL_FONT)
        ax.set_title(f"AUROC = {auroc:.4f}", fontsize=LABEL_FONT)
        ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
        ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=TICK_FONT, framealpha=0.85)

    fig.suptitle(f"Image-Level OoD — {method_name}", fontsize=LABEL_FONT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive OoD Evaluation")
    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir",  type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--growliflower_dir", type=str, required=True,
        help="Root dir of GrowliFlower-L (contains images/ and labels/ subdirs)")
    parser.add_argument("--out_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/ood_v2")

    # Method selection
    parser.add_argument("--method", type=str, default="lora",
        choices=["lora", "fullft", "mcdropout"])
    parser.add_argument("--method_name", type=str, default=None,
        help="Display name for plots (auto-set if not provided)")

    # LoRA args
    parser.add_argument("--lora_results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/hparam_sweep")
    parser.add_argument("--config_name",  type=str, default="final_model")
    parser.add_argument("--shot_ids",     nargs="+", type=int, default=[ 2, 3, 4, 5])

    # Full FT args
    parser.add_argument("--fullft_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/full_ft")
    parser.add_argument("--fullft_shot_ids", nargs="+", type=int, default=[ 2, 3, 4, 5])

    # MC Dropout args
    parser.add_argument("--mcdrop_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/mcdropout")
    parser.add_argument("--dropout_p",  type=float, default=0.25)
    parser.add_argument("--T",          type=int,   default=10)

    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--batch_size",  type=int, default=2)
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

    H, W = args.height, args.width
    method_name = args.method_name or {
        "lora": "LoRA Snapshot", "fullft": "Full FT Snapshot",
        "mcdropout": "MC Dropout"}[args.method]

    print(f"\n{'='*65}")
    print(f"  Comprehensive OoD — {method_name}  (seed={args.seed})")
    print(f"{'='*65}\n")

    # ── Build ensemble function ───────────────────────────────────────────────
    if args.method == "lora":
        base = load_base_model(args.pretrained, device)
        adapter_paths = [
            os.path.join(args.lora_results_dir, args.config_name,
                         f"seed_{args.seed}", f"model_shot_{s}")
            for s in args.shot_ids
        ]
        M = len(adapter_paths)
        def get_model(m): return load_lora_adapter(base, adapter_paths[m], device)

    elif args.method == "fullft":
        ckpts = [os.path.join(args.fullft_dir, f"seed_{args.seed}",
                              f"model_shot_{s}.pt") for s in args.fullft_shot_ids]
        M = len(ckpts)
        def get_model(m): return load_fullft_snapshot(args.pretrained, ckpts[m], device)

    else:  # mcdropout
        ckpt  = os.path.join(args.mcdrop_dir, f"seed_{args.seed}", "model_final.pt")
        mc_model = load_mc_model(args.pretrained, ckpt, args.dropout_p, device)
        M = args.T
        def get_model(m): return mc_model

    # ── Dataloaders ───────────────────────────────────────────────────────────
    id_loader  = build_id_loader(args.coco_file, args.root_dir,
                                  args.batch_size, args.num_workers)
    ood_loader = build_ood_loader(args.growliflower_dir,
                                   H, W, args.batch_size, args.num_workers)
    print(f"ID  images: {len(id_loader.dataset)}")
    print(f"OoD images: {len(ood_loader.dataset)}\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running ensemble inference on ID (sweet pepper)...")
    id_result = run_ensemble_id(get_model, id_loader, device, H, W, M)

    print("Running ensemble inference on OoD (GrowliFlower)...")
    ood_result = run_ensemble_ood(get_model, ood_loader, device, H, W, M)

    prefix = os.path.join(args.out_dir, f"{args.method}_seed{args.seed}")

    # ── PART 1 — Image-level ──────────────────────────────────────────────────
    print("\n── PART 1: Image-Level OoD ──────────────────────────")
    id_scores  = compute_image_level_scores(id_result,  masks_key="gts",   is_id=True)
    ood_scores = compute_image_level_scores(ood_result, masks_key="masks", is_id=False)

    ent_auroc = compute_image_auroc(id_scores["img_ent"], ood_scores["img_ent"])
    mi_auroc  = compute_image_auroc(id_scores["img_mi"],  ood_scores["img_mi"])
    print(f"  Entropy AUROC (image-level): {ent_auroc:.4f}")
    print(f"  MI      AUROC (image-level): {mi_auroc:.4f}")

    plot_entropy_mi_histograms(id_scores, ood_scores,
                                prefix + "_part1_entropy_mi_histograms.png")
    plot_image_auroc_comparison(id_scores, ood_scores, method_name,
                                 prefix + "_part1_image_auroc.png")

    # ── PART 2 — Pixel-level ─────────────────────────────────────────────────
    print("\n── PART 2: Pixel-Level OoD ──────────────────────────")
    for scorer in ["entropy", "MI"]:
        print(f"\n  Scorer: {scorer}")
        metrics = compute_pixel_ood_metrics(ood_result, scorer=scorer)
        for k, v in metrics.items():
            if k != "threshold":
                print(f"    {k:<10}: {v:.4f}")

    # Save results JSON
    results = {
        "method":     method_name,
        "seed":       args.seed,
        "image_level": {
            "entropy_AUROC": ent_auroc,
            "MI_AUROC":      mi_auroc,
        },
        "pixel_level": {
            scorer: compute_pixel_ood_metrics(ood_result, scorer=scorer)
            for scorer in ["entropy", "MI"]
        },
    }
    out_json = prefix + "_ood_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_json}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  {method_name}  —  OoD Summary  (seed={args.seed})")
    print(f"  {'Metric':<25}  {'Entropy':>10}  {'MI':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Image AUROC':<25}  {ent_auroc:>10.4f}  {mi_auroc:>10.4f}")
    for scorer in ["entropy", "MI"]:
        m = compute_pixel_ood_metrics(ood_result, scorer=scorer)
        print(f"  {'─'*50}")
        print(f"  Pixel [{scorer}]")
        for k in ["AUROC", "AUPR", "FPR95", "sIoU", "PPV", "MeanF1"]:
            print(f"    {k:<22}  {m[k]:>10.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()