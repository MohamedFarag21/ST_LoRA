# -*- coding: utf-8 -*-
"""
Comprehensive OoD Evaluation — LoRA / Full FT / MC Dropout / DDU
=================================================================
Extends ood_eval_comprehensive.py with DDU support.

Existing methods (LoRA, Full FT, MC Dropout):
  Scorers: entropy + MI  (unchanged)

DDU:
  Scorers: entropy + GMM density
    - entropy          : aleatoric (softmax, same as other methods)
    - density_spatial  : epistemic (GMM log-likelihood on spatial features)
    - density_query    : epistemic (GMM log-likelihood on query features)
  High density = in-distribution.  Low density = OoD.
  Negated for AUROC so that OoD images get higher scores.

Usage:
    python ood_eval_comprehensive_with_ddu.py \\
        --growliflower_dir /path/to/growliflower --method lora  --seed 42
    python ood_eval_comprehensive_with_ddu.py \\
        --growliflower_dir /path/to/growliflower --method ddu   --seed 42
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
from torch.nn.utils.parametrizations import spectral_norm as sn_parametrize

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
LABEL2ID    = {old: new for new, old in enumerate(sorted(ID2LABEL_ORIG.keys()))}
ID2LABEL    = {new: ID2LABEL_ORIG[old] for old, new in LABEL2ID.items()}
NUM_LABELS  = len(ID2LABEL)
NUM_QUERIES = 100

_REMAP_LUT = np.zeros(256, dtype=np.int64)
for old, new in LABEL2ID.items():
    _REMAP_LUT[old] = new

# Best GMM configs from grid search (by BIC)
BEST_QUERY_CONFIG   = "norm=none_k=2_cov=tied_reg=1e-4"
BEST_SPATIAL_CONFIG = "norm=none_k=2_cov=full_reg=1e-4"


# ─────────────────────────────────────────────────────────────────────────────
# Datasets  (identical to ood_eval_comprehensive.py)
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
    def __init__(self, root_dir, transform=None, native_palette=False):
        # native_palette=False -> legacy behaviour (bit-identical to the older runs):
        #   read the mask via .convert("L") which remaps palette bg index 0 -> lum 60,
        #   so masks with 3+ palette values fall to the (raw>0) branch = ALL anomaly.
        # native_palette=True  -> CORRECT: read raw palette indices (bg=0), anomaly=(raw>0).
        self.transform      = transform
        self.native_palette = native_palette
        self.pairs     = []
        for split in ["Train", "Val", "Test"]:
            img_dir  = os.path.join(root_dir, "images", split)
            mask_dir = os.path.join(root_dir, "labels", split, "maskPlants")
            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
                continue
            for img_path in sorted(Path(img_dir).glob("*.jpg")):
                stem      = img_path.stem
                mask_path = os.path.join(mask_dir, f"{stem}_Label_maskPlants.png")
                noplants  = os.path.join(mask_dir, f"{stem}_Label_NoPlants_maskPlants.png")
                if os.path.exists(noplants): continue
                if not os.path.exists(mask_path): continue
                self.pairs.append((str(img_path), mask_path))
        if not self.pairs:
            raise RuntimeError(f"No valid image-mask pairs found in {root_dir}.")
        print(f"[GrowliFlower-L] {len(self.pairs)} image-mask pairs")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        pil_img  = Image.open(img_path).convert("RGB")
        image    = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        if self.native_palette:
            # native palette indices: 0 = background, >0 = cauliflower plant (anomaly)
            raw = np.asarray(Image.open(msk_path))
            if raw.ndim == 3:
                raw = raw[..., 0]
            binary = (raw > 0).astype(np.uint8)
        else:
            raw       = np.array(Image.open(msk_path).convert("L"))
            uniq_vals = np.unique(raw)
            if len(uniq_vals) == 2:
                binary = (raw == int(uniq_vals.max())).astype(np.uint8)
            else:
                binary = (raw > 0).astype(np.uint8)
        return image, torch.from_numpy(binary)


def make_transform(H=None, W=None):
    ops = []
    if H is not None and W is not None:
        ops.append(transforms.Resize((H, W)))
    ops += [transforms.ToTensor(), transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)]
    return transforms.Compose(ops)


def build_id_loader(coco_file, root_dir, batch_size, num_workers):
    preprocessor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=False,
        do_resize=False, do_rescale=False, do_normalize=False,
        num_labels=NUM_LABELS)
    ds = SweetPepperTestDataset(coco_file, root_dir, transform=make_transform())

    def collate(batch):
        images, seg_maps = zip(*batch)
        out = preprocessor(list(images), segmentation_maps=list(seg_maps), return_tensors="pt")
        out["seg_maps"] = list(seg_maps)
        return out

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                      num_workers=num_workers, persistent_workers=(num_workers > 0))


def build_ood_loader(root_dir, H, W, batch_size, num_workers, native_palette=False):
    ds = GrowliFlowerDataset(root_dir, transform=make_transform(H, W),
                             native_palette=native_palette)

    def collate(batch):
        batch  = [b for b in batch if b is not None]
        images, masks = zip(*batch)
        masks_r = [torch.from_numpy(
                       np.array(Image.fromarray(m.numpy()).resize((W, H), Image.NEAREST)))
                   for m in masks]
        return torch.stack(images), torch.stack(masks_r)

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                      num_workers=num_workers, persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — existing methods
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(pretrained_name, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    return model.eval().to(device)

def load_lora_adapter(base_model, adapter_path, device):
    return PeftModel.from_pretrained(base_model, adapter_path).eval().to(device)

def load_fullft_snapshot(pretrained_name, ckpt_path, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval().to(device)

def load_mc_model(pretrained_name, ckpt_path, dropout_p, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    model.class_predictor = nn.Sequential(nn.Dropout(p=dropout_p), model.class_predictor)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device)
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — DDU
# ─────────────────────────────────────────────────────────────────────────────

def load_ddu_model(pretrained_name, ckpt_path, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    visited = set()
    def _sn(mod):
        for name, child in mod.named_children():
            if id(child) in visited: continue
            visited.add(id(child))
            _sn(child)
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                try: sn_parametrize(child)
                except Exception: pass
    _sn(model)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval().to(device)


def load_gmms(seed_dir, config_id, feature_type):
    path = os.path.join(seed_dir, f"gmm_{feature_type}_{config_id}.pkl")
    if not os.path.exists(path):
        print(f"  [WARN] GMM not found: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_norm_stats(seed_dir, config_id, feature_type):
    path = os.path.join(seed_dir, f"gmm_{feature_type}_{config_id}_stats.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["mean"], data["std"]


# ─────────────────────────────────────────────────────────────────────────────
# Seg probs
# ─────────────────────────────────────────────────────────────────────────────

def build_seg_probs(outputs, H, W):
    cp  = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
    mp  = outputs.masks_queries_logits.sigmoid()
    seg = torch.einsum("bqc,bqhw->bchw", cp, mp)
    seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
    return F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Inference — existing methods (entropy + MI, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_ensemble_id(models_or_passes_fn, dataloader, device, H, W, M):
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
        if hasattr(model, "base_model"):
            del model; torch.cuda.empty_cache()

    mean_probs = sum_probs / M
    mean_H     = sum_H / M
    H_mean     = -(mean_probs * (mean_probs + 1e-12).log2()).sum(dim=1)
    MI         = H_mean - mean_H

    return {"mean_probs": mean_probs, "entropy": H_mean, "MI": MI, "gts": gts}


@torch.no_grad()
def run_ensemble_ood(models_or_passes_fn, dataloader, device, H, W, M):
    N     = len(dataloader.dataset)
    H_ood = min(H, 512)
    W_ood = min(W, 512)

    sum_probs = np.zeros((N, NUM_LABELS, H_ood, W_ood), dtype=np.float32)
    sum_H     = np.zeros((N, H_ood, W_ood),             dtype=np.float32)
    masks_all = [None] * N

    for m in range(M):
        model = models_or_passes_fn(m)
        idx   = 0
        for images, masks in tqdm(dataloader, desc=f"  OoD pass {m+1}/{M}", leave=False):
            pv  = images.to(device)
            seg = build_seg_probs(model(pv), H_ood, W_ood)
            B   = seg.shape[0]
            H_t = -(seg * (seg + 1e-12).log2()).sum(dim=1)
            sum_probs[idx:idx+B] += seg.cpu().numpy()
            sum_H[idx:idx+B]     += H_t.cpu().numpy()
            if m == 0:
                for b in range(B):
                    masks_all[idx+b] = masks[b]
            idx += B
        if hasattr(model, "base_model"):
            del model; torch.cuda.empty_cache()

    mean_probs = sum_probs / M
    mean_H     = sum_H / M
    H_mean = -(mean_probs * np.log2(mean_probs + 1e-12)).sum(axis=1)
    MI     = H_mean - mean_H

    if H_ood != H or W_ood != W:
        H_mean = np.stack([np.array(Image.fromarray(H_mean[i]).resize((W, H), Image.BILINEAR))
                           for i in range(N)])
        MI     = np.stack([np.array(Image.fromarray(MI[i]).resize((W, H), Image.BILINEAR))
                           for i in range(N)])

    return {"entropy": torch.from_numpy(H_mean), "MI": torch.from_numpy(MI),
            "masks": masks_all}


# ─────────────────────────────────────────────────────────────────────────────
# Inference — DDU (entropy + GMM density, single deterministic pass)
# ─────────────────────────────────────────────────────────────────────────────

def _gmm_score_spatial_pixel(sf_b, gmms, pred_classes=None):
    """
    Score each pixel under its GMM.
    sf_b        : (N_px, 100)  spatial features
    pred_classes: (N_px,) int  predicted class per pixel (None = use best class)
    Returns (N_px,) log-likelihood array.
    """
    available = [(c, gmm) for c, gmm in gmms.items() if gmm is not None]
    if not available:
        return np.zeros(len(sf_b), dtype=np.float32)

    if pred_classes is None:
        # OoD: no class labels — score under all GMMs, take max per pixel
        all_ll = np.stack([gmm.score_samples(sf_b) for _, gmm in available], axis=1)
        return all_ll.max(axis=1).astype(np.float32)
    else:
        scores = np.full(len(sf_b), fill_value=np.nan, dtype=np.float64)
        for c, gmm in available:
            mask = (pred_classes == c)
            if not mask.any(): continue
            scores[mask] = gmm.score_samples(sf_b[mask])
        # Fall back to best-class for pixels whose predicted class has no GMM
        nan_mask = np.isnan(scores)
        if nan_mask.any():
            all_ll = np.stack([gmm.score_samples(sf_b[nan_mask]) for _, gmm in available], axis=1)
            scores[nan_mask] = all_ll.max(axis=1)
        return scores.astype(np.float32)


def _gmm_score_query_image(qf_b, gmms):
    """
    Mean log-likelihood over all queries (max over classes per query).
    qf_b : (100, 256)
    """
    available = [(c, gmm) for c, gmm in gmms.items() if gmm is not None]
    if not available or len(qf_b) == 0:
        return 0.0
    all_ll = np.stack([gmm.score_samples(qf_b) for _, gmm in available], axis=1)
    return float(all_ll.max(axis=1).mean())


@torch.no_grad()
def run_ddu_id(model, dataloader, device, H, W,
               spatial_gmms, query_gmms,
               s_norm_mean, s_norm_std,
               q_norm_mean, q_norm_std):
    """
    Single pass DDU inference on ID (sweet pepper) data.
    Returns result dict with entropy, density maps, and GT maps.
    """
    N = len(dataloader.dataset)
    entropy_maps     = np.zeros((N, H, W), dtype=np.float32)
    density_spat_map = np.zeros((N, H, W), dtype=np.float32)  # per-pixel
    density_query_img = np.zeros(N, dtype=np.float32)          # per-image
    gts = [None] * N
    idx = 0

    for batch in tqdm(dataloader, desc="  [DDU] ID inference"):
        pv = batch["pixel_values"].to(device)
        outputs = model(pv, output_hidden_states=True)

        seg = build_seg_probs(outputs, H, W)
        B   = seg.shape[0]

        # Entropy
        H_ent = -(seg * torch.log2(seg + 1e-12)).sum(dim=1)
        entropy_maps[idx:idx+B] = H_ent.cpu().numpy()

        preds = seg.argmax(dim=1).cpu().numpy()   # (B, H, W)

        # Spatial features → per-pixel density
        if spatial_gmms is not None:
            sf = outputs.masks_queries_logits.sigmoid()
            sf = F.interpolate(sf, size=(H, W), mode="bilinear", align_corners=False)
            sf = sf.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 100)
            for b in range(B):
                sf_b = sf[b].reshape(-1, NUM_QUERIES)
                if s_norm_mean is not None:
                    sf_b = (sf_b - s_norm_mean) / (s_norm_std + 1e-8)
                pred_px = preds[b].ravel()
                density_spat_map[idx+b] = _gmm_score_spatial_pixel(
                    sf_b, spatial_gmms, pred_px).reshape(H, W)

        # Query features → per-image density
        if query_gmms is not None:
            qf = outputs.transformer_decoder_last_hidden_state  # (B, 100, 256)
            for b in range(B):
                qf_b = qf[b].cpu().numpy()
                if q_norm_mean is not None:
                    qf_b = (qf_b - q_norm_mean) / (q_norm_std + 1e-8)
                density_query_img[idx+b] = _gmm_score_query_image(qf_b, query_gmms)

        for b, gt in enumerate(batch["seg_maps"]):
            gts[idx+b] = gt
        idx += B

    # Broadcast query scalar to spatial map for consistency with OoD result
    density_query_map_id = np.stack([
        np.full((H, W), density_query_img[i], dtype=np.float32)
        for i in range(N)])

    return {
        "entropy":            torch.from_numpy(entropy_maps),
        "density_spat_map":   torch.from_numpy(density_spat_map),
        "density_query_img":  density_query_img,
        "density_query_map":  torch.from_numpy(density_query_map_id),
        "gts": gts,
    }


@torch.no_grad()
def run_ddu_ood(model, dataloader, device, H, W,
                spatial_gmms, query_gmms,
                s_norm_mean, s_norm_std,
                q_norm_mean, q_norm_std):
    """
    Single pass DDU inference on OoD (GrowliFlower) data.
    Memory-efficient: spatial features processed at 512×512 then upsampled.
    """
    N     = len(dataloader.dataset)
    H_ood = min(H, 512)
    W_ood = min(W, 512)

    entropy_maps     = np.zeros((N, H_ood, W_ood), dtype=np.float32)
    density_spat_map = np.zeros((N, H_ood, W_ood), dtype=np.float32)
    density_query_img = np.zeros(N, dtype=np.float32)
    masks_all = [None] * N
    idx = 0

    for images, masks in tqdm(dataloader, desc="  [DDU] OoD inference"):
        pv = images.to(device)
        outputs = model(pv, output_hidden_states=True)

        seg = build_seg_probs(outputs, H_ood, W_ood)
        B   = seg.shape[0]

        H_ent = -(seg * torch.log2(seg + 1e-12)).sum(dim=1)
        entropy_maps[idx:idx+B] = H_ent.cpu().numpy()

        # Spatial density — no GT class labels, use best-class GMM
        if spatial_gmms is not None:
            sf = outputs.masks_queries_logits.sigmoid()
            sf = F.interpolate(sf, size=(H_ood, W_ood), mode="bilinear", align_corners=False)
            sf = sf.permute(0, 2, 3, 1).cpu().numpy()  # (B, H_ood, W_ood, 100)
            for b in range(B):
                sf_b = sf[b].reshape(-1, NUM_QUERIES)
                if s_norm_mean is not None:
                    sf_b = (sf_b - s_norm_mean) / (s_norm_std + 1e-8)
                # OoD: no pred classes — score under best class
                density_spat_map[idx+b] = _gmm_score_spatial_pixel(
                    sf_b, spatial_gmms, pred_classes=None).reshape(H_ood, W_ood)

        # Query density
        if query_gmms is not None:
            qf = outputs.transformer_decoder_last_hidden_state
            for b in range(B):
                qf_b = qf[b].cpu().numpy()
                if q_norm_mean is not None:
                    qf_b = (qf_b - q_norm_mean) / (q_norm_std + 1e-8)
                density_query_img[idx+b] = _gmm_score_query_image(qf_b, query_gmms)

        for b in range(B):
            masks_all[idx+b] = masks[b]
        idx += B

    # Upsample to full resolution if needed
    if H_ood != H or W_ood != W:
        entropy_maps = np.stack([
            np.array(Image.fromarray(entropy_maps[i]).resize((W, H), Image.BILINEAR))
            for i in range(N)])
        density_spat_map = np.stack([
            np.array(Image.fromarray(density_spat_map[i]).resize((W, H), Image.BILINEAR))
            for i in range(N)])

    # Broadcast query scalar to spatial map (N, H, W) — uniform per image
    # This allows pixel-level metrics to be computed the same way as spatial density
    density_query_map = np.stack([
        np.full((H, W), density_query_img[i], dtype=np.float32)
        for i in range(N)])

    return {
        "entropy":            torch.from_numpy(entropy_maps),
        "density_spat_map":   torch.from_numpy(density_spat_map),   # (N,H,W) log-likelihood
        "density_query_img":  density_query_img,                     # (N,) log-likelihood
        "density_query_map":  torch.from_numpy(density_query_map),  # (N,H,W) broadcast
        "masks": masks_all,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image-level scoring — existing methods (entropy + MI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_image_level_scores(result, masks_key="gts", is_id=True):
    N   = result["entropy"].shape[0]
    ent = result["entropy"]
    mi  = result["MI"]
    masks = result[masks_key]

    img_ent, img_mi, bg_ent, fg_ent, bg_mi, fg_mi = [], [], [], [], [], []

    for i in range(N):
        m   = masks[i].numpy() if hasattr(masks[i], "numpy") else np.array(masks[i])
        e   = ent[i].numpy()
        mi_ = mi[i].numpy()

        img_ent.append(float(e.mean()))
        img_mi.append(float(mi_.mean()))

        bg = m == 0
        fg = (m != 0) & (m != 255) if is_id else (m > 0)

        if bg.any(): bg_ent.append(float(e[bg].mean())); bg_mi.append(float(mi_[bg].mean()))
        if fg.any(): fg_ent.append(float(e[fg].mean())); fg_mi.append(float(mi_[fg].mean()))

    return {"img_ent": np.array(img_ent), "img_mi": np.array(img_mi),
            "bg_ent": np.array(bg_ent),   "fg_ent": np.array(fg_ent),
            "bg_mi":  np.array(bg_mi),    "fg_mi":  np.array(fg_mi)}


# ─────────────────────────────────────────────────────────────────────────────
# Image-level scoring — DDU (entropy + density)
# ─────────────────────────────────────────────────────────────────────────────

def compute_image_level_scores_ddu(result, masks_key="gts", is_id=True):
    """
    Returns per-image scalars for entropy and GMM density.
    density scores are raw log-likelihood (higher = more ID).
    Negation for AUROC happens in compute_image_auroc_ddu.
    """
    N     = result["entropy"].shape[0]
    ent   = result["entropy"]             # (N, H, W)
    d_map = result["density_spat_map"]    # (N, H, W) log-likelihood
    d_q   = result["density_query_img"]   # (N,) log-likelihood
    masks = result[masks_key]

    img_ent, img_d_spat, img_d_query = [], [], []
    bg_ent, fg_ent = [], []
    bg_d,   fg_d   = [], []

    for i in range(N):
        m   = masks[i].numpy() if hasattr(masks[i], "numpy") else np.array(masks[i])
        e   = ent[i].numpy()
        d   = d_map[i].numpy()

        img_ent.append(float(e.mean()))
        img_d_spat.append(float(d.mean()))
        img_d_query.append(float(d_q[i]))

        bg = m == 0
        fg = (m != 0) & (m != 255) if is_id else (m > 0)

        if bg.any():
            bg_ent.append(float(e[bg].mean()))
            bg_d.append(float(d[bg].mean()))
        if fg.any():
            fg_ent.append(float(e[fg].mean()))
            fg_d.append(float(d[fg].mean()))

    return {
        "img_ent":     np.array(img_ent),
        "img_d_spat":  np.array(img_d_spat),   # spatial density (log-likelihood)
        "img_d_query": np.array(img_d_query),   # query density (log-likelihood)
        "bg_ent": np.array(bg_ent), "fg_ent": np.array(fg_ent),
        "bg_d":   np.array(bg_d),   "fg_d":   np.array(fg_d),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUROC helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_image_auroc(id_scores, ood_scores):
    """Higher score = more OoD."""
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return float("nan")
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_image_auroc_ddu(id_scores, ood_scores, is_density=False):
    """
    For entropy: higher = more OoD → pass directly.
    For density (log-likelihood): higher = more ID → negate before AUROC.
    """
    if is_density:
        return compute_image_auroc(-id_scores, -ood_scores)
    return compute_image_auroc(id_scores, ood_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-level OoD metrics  (shared, scorer-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fpr95(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95)
    return float(fpr[min(idx, len(fpr)-1)])


def compute_component_metrics(gt_mask, pred_mask, iou_thresh=0.25):
    gt_labeled,   n_gt   = ndimage.label(gt_mask)
    pred_labeled, n_pred = ndimage.label(pred_mask)
    if n_gt == 0:
        return float("nan"), float("nan"), float("nan")
    tp, fp, fn = 0, 0, 0
    sious = []
    for gt_id in range(1, n_gt + 1):
        gt_comp = (gt_labeled == gt_id)
        overlap_ids = np.unique(pred_labeled[gt_comp])
        overlap_ids = overlap_ids[overlap_ids > 0]
        best_iou = 0.0
        for pid in overlap_ids:
            pred_comp = (pred_labeled == pid)
            inter = (gt_comp & pred_comp).sum()
            other_gt = (gt_labeled > 0) & (gt_labeled != gt_id)
            union = gt_comp.sum() + pred_comp.sum() - inter - (pred_comp & other_gt).sum()
            if union > 0: best_iou = max(best_iou, inter / union)
        if best_iou >= iou_thresh: tp += 1; sious.append(best_iou)
        else: fn += 1
    for pred_id in range(1, n_pred + 1):
        pred_comp = (pred_labeled == pred_id)
        overlap_ids = np.unique(gt_labeled[pred_comp])
        overlap_ids = overlap_ids[overlap_ids > 0]
        matched = False
        for gid in overlap_ids:
            gt_comp  = (gt_labeled == gid)
            inter    = (gt_comp & pred_comp).sum()
            other_gt = (gt_labeled > 0) & (gt_labeled != gid)
            union    = gt_comp.sum() + pred_comp.sum() - inter - (pred_comp & other_gt).sum()
            if union > 0 and inter / union >= iou_thresh: matched = True; break
        if not matched: fp += 1
    siou = float(np.mean(sious)) if sious else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * ppv * rec / (ppv + rec) if (ppv + rec) > 0 else 0.0
    return siou, ppv, f1


def compute_pixel_ood_metrics(ood_result, scorer="entropy",
                               negate=False, n_thresh=50):
    """
    scorer: key in ood_result with shape (N, H, W).
    negate: set True for density scores (high density = ID → negate for OoD).
    """
    all_scores, all_labels = [], []
    N = len(ood_result["masks"])
    scores_map = ood_result[scorer]

    for i in range(N):
        m = ood_result["masks"][i].numpy()
        s = scores_map[i].numpy() if hasattr(scores_map[i], "numpy") else scores_map[i]
        if negate: s = -s
        all_scores.append(s.ravel())
        all_labels.append((m > 0).astype(np.uint8).ravel())

    flat_scores = np.concatenate(all_scores).astype(np.float32)
    flat_labels = np.concatenate(all_labels).astype(np.uint8)

    unique, counts = np.unique(flat_labels, return_counts=True)
    print(f"  [pixel OoD] label distribution: { {int(u): int(c) for u, c in zip(unique, counts)} }")

    if len(np.unique(flat_labels)) < 2:
        return {"AUROC": float("nan"), "AUPR": float("nan"), "FPR95": float("nan"),
                "sIoU": float("nan"), "PPV": float("nan"), "MeanF1": float("nan"),
                "threshold": float("nan")}

    auroc = float(roc_auc_score(flat_labels, flat_scores))
    aupr  = float(average_precision_score(flat_labels, flat_scores))
    fpr95 = compute_fpr95(flat_labels, flat_scores)

    thresholds = np.percentile(flat_scores, np.linspace(50, 99, n_thresh))
    best_f1, best_thresh = -1, thresholds[0]
    for t in thresholds:
        pred = (flat_scores >= t).astype(np.uint8)
        tp = ((pred==1)&(flat_labels==1)).sum()
        fp = ((pred==1)&(flat_labels==0)).sum()
        fn = ((pred==0)&(flat_labels==1)).sum()
        f1 = 2*tp/(2*tp+fp+fn+1e-6)
        if f1 > best_f1: best_f1, best_thresh = f1, t

    sious, ppvs, f1s = [], [], []
    for i in range(N):
        m  = ood_result["masks"][i].numpy()
        s  = scores_map[i].numpy() if hasattr(scores_map[i], "numpy") else scores_map[i]
        if negate: s = -s
        pred = (s >= best_thresh).astype(np.uint8)
        gt   = (m > 0).astype(np.uint8)
        if gt.sum() == 0: continue
        siou, ppv, f1 = compute_component_metrics(gt, pred)
        if not np.isnan(siou): sious.append(siou); ppvs.append(ppv); f1s.append(f1)

    return {"AUROC": auroc, "AUPR": aupr, "FPR95": fpr95,
            "sIoU":  float(np.mean(sious)) if sious else float("nan"),
            "PPV":   float(np.mean(ppvs))  if ppvs  else float("nan"),
            "MeanF1":float(np.mean(f1s))   if f1s   else float("nan"),
            "threshold": float(best_thresh)}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — existing methods (entropy + MI, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_entropy_mi_histograms(id_scores, ood_scores, out_path):
    COLOR_BG = "#1f77b4"; COLOR_FG = "#ff7f0e"; ALPHA = 0.60
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    titles = [("Sweet Pepper (ID) — Entropy",    id_scores,  "entropy"),
              ("Sweet Pepper (ID) — MI",          id_scores,  "mi"),
              ("GrowliFlower-L (OoD) — Entropy", ood_scores, "entropy"),
              ("GrowliFlower-L (OoD) — MI",      ood_scores, "mi")]
    _key = {"entropy": "ent", "mi": "mi"}
    for ax, (title, scores, metric) in zip(axes.ravel(), titles):
        k = _key[metric]
        bg, fg = scores[f"bg_{k}"], scores[f"fg_{k}"]
        if len(bg) == 0 or len(fg) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=LABEL_FONT)
            ax.set_title(title, fontsize=LABEL_FONT); continue
        bins = np.linspace(min(bg.min(), fg.min()), max(bg.max(), fg.max()), 40)
        ax.hist(bg, bins=bins, density=True, alpha=ALPHA, color=COLOR_BG,
                label=f"Background (μ={bg.mean():.3f})", linewidth=0)
        ax.hist(fg, bins=bins, density=True, alpha=ALPHA, color=COLOR_FG,
                label=f"{'Cauliflower' if 'OoD' in title else 'Pepper'} (μ={fg.mean():.3f})",
                linewidth=0)
        ax.axvline(bg.mean(), color=COLOR_BG, linewidth=1.6, linestyle="--")
        ax.axvline(fg.mean(), color=COLOR_FG, linewidth=1.6, linestyle="--")
        ax.set_title(title, fontsize=LABEL_FONT)
        ax.set_xlabel("Entropy (bits)" if metric=="entropy" else "MI (bits)", fontsize=LABEL_FONT)
        ax.set_ylabel("Density", fontsize=LABEL_FONT)
        ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
        ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=TICK_FONT, framealpha=0.85)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_image_auroc_comparison(id_img, ood_img, method_name, out_path):
    COLOR_ID = "#2196F3"; COLOR_OOD = "#FF5722"; ALPHA = 0.60
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, scorer, xlabel in [
        (axes[0], "img_ent", "Mean image entropy (bits)"),
        (axes[1], "img_mi",  "Mean image MI (bits)"),
    ]:
        id_s, ood_s = id_img[scorer], ood_img[scorer]
        auroc = compute_image_auroc(id_s, ood_s)
        bins  = np.linspace(min(id_s.min(), ood_s.min()), max(id_s.max(), ood_s.max()), 35)
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
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"[Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — DDU (entropy + density)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ddu_histograms(id_scores, ood_scores, out_path):
    """
    2 rows × 3 cols for DDU:
      Row 0: BG vs FG — Entropy | Spatial density | (blank — query is image-level only)
      Row 1: ID vs OoD — Entropy | Spatial density | Query density
    """
    COLOR_BG  = "#1f77b4"; COLOR_FG  = "#ff7f0e"
    COLOR_ID  = "#2196F3"; COLOR_OOD = "#FF5722"
    ALPHA = 0.60

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # ── Row 0: BG vs FG histograms ────────────────────────────────────────────
    for ax, col_label, bg_key, fg_key, xlabel in [
        (axes[0][0], "Entropy",         "bg_ent", "fg_ent", "Entropy (bits)"),
        (axes[0][1], "Spatial density", "bg_d",   "fg_d",   "GMM log-likelihood"),
    ]:
        bg_id  = id_scores[bg_key]
        fg_id  = id_scores[fg_key]
        bg_ood = ood_scores[bg_key]
        fg_ood = ood_scores[fg_key]

        all_vals = np.concatenate([bg_id, fg_id, bg_ood, fg_ood])
        bins = np.linspace(all_vals.min(), all_vals.max(), 40)

        ax.hist(bg_id,  bins=bins, density=True, alpha=ALPHA, color=COLOR_BG,
                label=f"ID bg (μ={bg_id.mean():.3f})", linewidth=0)
        ax.hist(fg_id,  bins=bins, density=True, alpha=0.5,   color=COLOR_FG,
                label=f"ID fg (μ={fg_id.mean():.3f})", linewidth=0)
        ax.hist(bg_ood, bins=bins, density=True, alpha=0.35,  color=COLOR_ID,
                label=f"OoD bg (μ={bg_ood.mean():.3f})", linewidth=0, hatch="//")
        ax.hist(fg_ood, bins=bins, density=True, alpha=0.35,  color=COLOR_OOD,
                label=f"OoD fg (μ={fg_ood.mean():.3f})", linewidth=0, hatch="//")
        ax.set_title(f"BG vs FG — {col_label}", fontsize=LABEL_FONT)
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT)
        ax.set_ylabel("Density", fontsize=LABEL_FONT)
        ax.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
        ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, framealpha=0.85)

    # (0,2) Query density — image-level only, no per-pixel BG/FG split possible
    ax02 = axes[0][2]
    id_q  = id_scores["img_d_query"]
    ood_q = ood_scores["img_d_query"]
    auroc_q = compute_image_auroc_ddu(id_q, ood_q, is_density=True)
    bins_q = np.linspace(min(id_q.min(), ood_q.min()), max(id_q.max(), ood_q.max()), 35)
    ax02.hist(id_q,  bins=bins_q, density=True, alpha=ALPHA, color=COLOR_ID,
              label=f"Sweet Pepper ID (n={len(id_q)})", linewidth=0)
    ax02.hist(ood_q, bins=bins_q, density=True, alpha=ALPHA, color=COLOR_OOD,
              label=f"GrowliFlower OoD (n={len(ood_q)})", linewidth=0)
    ax02.axvline(id_q.mean(),  color=COLOR_ID,  linewidth=1.6, linestyle="--")
    ax02.axvline(ood_q.mean(), color=COLOR_OOD, linewidth=1.6, linestyle="--")
    ax02.set_title(f"Query density — ID vs OoD  (AUROC={auroc_q:.4f})", fontsize=LABEL_FONT)
    ax02.set_xlabel("Query GMM log-likelihood", fontsize=LABEL_FONT)
    ax02.set_ylabel("Density", fontsize=LABEL_FONT)
    ax02.tick_params(labelsize=TICK_FONT, length=TICK_SIZE)
    ax02.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    ax02.spines["top"].set_visible(False); ax02.spines["right"].set_visible(False)
    ax02.legend(fontsize=TICK_FONT, framealpha=0.85)

    # ── Row 1: ID vs OoD image-level histograms ───────────────────────────────
    for ax, scorer_key, xlabel, is_density in [
        (axes[1][0], "img_ent",    "Mean image entropy (bits)",      False),
        (axes[1][1], "img_d_spat", "Mean image spatial log-likelihood", True),
        (axes[1][2], "img_d_query","Mean image query log-likelihood",   True),
    ]:
        id_s  = id_scores[scorer_key]
        ood_s = ood_scores[scorer_key]
        auroc = compute_image_auroc_ddu(id_s, ood_s, is_density=is_density)
        bins  = np.linspace(min(id_s.min(), ood_s.min()),
                            max(id_s.max(), ood_s.max()), 35)
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

    fig.suptitle("DDU — OoD Analysis: Entropy + Spatial Density + Query Density",
                 fontsize=LABEL_FONT)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"[Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive OoD Evaluation (all methods including DDU)")
    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir",  type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--growliflower_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/ood_v2")

    parser.add_argument("--method", type=str, default="lora",
        choices=["lora", "fullft", "mcdropout", "ddu"])
    parser.add_argument("--method_name", type=str, default=None)

    # LoRA
    parser.add_argument("--lora_results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/hparam_sweep")
    parser.add_argument("--config_name",  type=str, default="final_model")
    parser.add_argument("--shot_ids",     nargs="+", type=int, default=[2, 3, 4, 5])
    # Full FT
    parser.add_argument("--fullft_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/full_ft")
    parser.add_argument("--fullft_shot_ids", nargs="+", type=int, default=[2, 3, 4, 5])
    # MC Dropout
    parser.add_argument("--mcdrop_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/mcdropout")
    parser.add_argument("--dropout_p",  type=float, default=0.5)
    parser.add_argument("--T",          type=int,   default=4)
    # DDU
    parser.add_argument("--ddu_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/ddu")
    parser.add_argument("--query_config",   type=str, default=BEST_QUERY_CONFIG)
    parser.add_argument("--spatial_config", type=str, default=BEST_SPATIAL_CONFIG)

    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu",         type=str, default="0")
    parser.add_argument("--height",      type=int, default=1280)
    parser.add_argument("--width",       type=int, default=720)
    parser.add_argument("--native_palette", action="store_true",
        help="Read GrowliFlower masks as native palette indices (bg=0). "
             "Fixes the .convert('L') all-anomaly bug on 3+-value masks.")
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
        "mcdropout": "MC Dropout", "ddu": "DDU"}[args.method]

    print(f"\n{'='*65}")
    print(f"  Comprehensive OoD — {method_name}  (seed={args.seed})")
    print(f"{'='*65}\n")

    id_loader  = build_id_loader(args.coco_file, args.root_dir,
                                  args.batch_size, args.num_workers)
    ood_loader = build_ood_loader(args.growliflower_dir, H, W,
                                   args.batch_size, args.num_workers,
                                   native_palette=args.native_palette)
    print(f"ID  images: {len(id_loader.dataset)}")
    print(f"OoD images: {len(ood_loader.dataset)}\n")

    prefix = os.path.join(args.out_dir, f"{args.method}_seed{args.seed}")

    # ══ DDU branch ════════════════════════════════════════════════════════════
    if args.method == "ddu":
        seed_dir  = os.path.join(args.ddu_dir, f"seed_{args.seed}")
        ckpt_path = os.path.join(seed_dir, "model_final.pt")

        model        = load_ddu_model(args.pretrained, ckpt_path, device)
        query_gmms   = load_gmms(seed_dir, args.query_config,   "query")
        spatial_gmms = load_gmms(seed_dir, args.spatial_config, "spatial")
        q_mean, q_std = load_norm_stats(seed_dir, args.query_config,   "query")
        s_mean, s_std = load_norm_stats(seed_dir, args.spatial_config, "spatial")

        print("Running DDU inference on ID (sweet pepper)...")
        id_result = run_ddu_id(model, id_loader, device, H, W,
                               spatial_gmms, query_gmms,
                               s_mean, s_std, q_mean, q_std)

        print("Running DDU inference on OoD (GrowliFlower)...")
        ood_result = run_ddu_ood(model, ood_loader, device, H, W,
                                 spatial_gmms, query_gmms,
                                 s_mean, s_std, q_mean, q_std)

        id_scores  = compute_image_level_scores_ddu(id_result,  "gts",   is_id=True)
        ood_scores = compute_image_level_scores_ddu(ood_result, "masks", is_id=False)

        ent_auroc    = compute_image_auroc_ddu(id_scores["img_ent"],    ood_scores["img_ent"],    is_density=False)
        d_spat_auroc = compute_image_auroc_ddu(id_scores["img_d_spat"], ood_scores["img_d_spat"], is_density=True)
        d_qry_auroc  = compute_image_auroc_ddu(id_scores["img_d_query"],ood_scores["img_d_query"],is_density=True)

        print(f"\n── PART 1: Image-Level OoD ──────────────────────────")
        print(f"  Entropy AUROC         : {ent_auroc:.4f}")
        print(f"  Density spatial AUROC : {d_spat_auroc:.4f}")
        print(f"  Density query AUROC   : {d_qry_auroc:.4f}")

        plot_ddu_histograms(id_scores, ood_scores, prefix + "_part1_ddu_histograms.png")

        print(f"\n── PART 2: Pixel-Level OoD ──────────────────────────")
        px_ent  = compute_pixel_ood_metrics(ood_result, "entropy",          negate=False)
        px_dens = compute_pixel_ood_metrics(ood_result, "density_spat_map", negate=True)

        print(f"\n  Entropy scorer (pixel-level):")
        for k in ["AUROC","AUPR","FPR95","sIoU","PPV","MeanF1"]:
            print(f"    {k:<10}: {px_ent[k]:.4f}")
        print(f"\n  Spatial density scorer (pixel-level, negated):")
        for k in ["AUROC","AUPR","FPR95","sIoU","PPV","MeanF1"]:
            print(f"    {k:<10}: {px_dens[k]:.4f}")
        px_query = compute_pixel_ood_metrics(ood_result, "density_query_map", negate=True)

        print(f"\n  Query density scorer (broadcast uniform per image, negated):")
        for k in ["AUROC","AUPR","FPR95","sIoU","PPV","MeanF1"]:
            print(f"    {k:<10}: {px_query[k]:.4f}")

        results = {
            "method": method_name, "seed": args.seed,
            "image_level": {
                "entropy_AUROC":         ent_auroc,
                "density_spatial_AUROC": d_spat_auroc,
                "density_query_AUROC":   d_qry_auroc,
            },
            "pixel_level": {
                "entropy":         px_ent,
                "density_spatial": px_dens,
                "density_query":   px_query,
            },
        }

    # ══ Existing methods branch (unchanged) ═══════════════════════════════════
    else:
        if args.method == "lora":
            base = load_base_model(args.pretrained, device)
            adapter_paths = [os.path.join(args.lora_results_dir, args.config_name,
                             f"seed_{args.seed}", f"model_shot_{s}") for s in args.shot_ids]
            M = len(adapter_paths)
            def get_model(m): return load_lora_adapter(base, adapter_paths[m], device)

        elif args.method == "fullft":
            ckpts = [os.path.join(args.fullft_dir, f"seed_{args.seed}",
                     f"model_shot_{s}.pt") for s in args.fullft_shot_ids]
            M = len(ckpts)
            def get_model(m): return load_fullft_snapshot(args.pretrained, ckpts[m], device)

        else:  # mcdropout
            ckpt     = os.path.join(args.mcdrop_dir, f"seed_{args.seed}", "model_final.pt")
            mc_model = load_mc_model(args.pretrained, ckpt, args.dropout_p, device)
            M = args.T
            def get_model(m): return mc_model

        print(f"Running ensemble inference on ID (sweet pepper)...")
        id_result = run_ensemble_id(get_model, id_loader, device, H, W, M)

        print(f"Running ensemble inference on OoD (GrowliFlower)...")
        ood_result = run_ensemble_ood(get_model, ood_loader, device, H, W, M)

        print(f"\n── PART 1: Image-Level OoD ──────────────────────────")
        id_scores  = compute_image_level_scores(id_result,  masks_key="gts",   is_id=True)
        ood_scores = compute_image_level_scores(ood_result, masks_key="masks", is_id=False)

        ent_auroc = compute_image_auroc(id_scores["img_ent"], ood_scores["img_ent"])
        mi_auroc  = compute_image_auroc(id_scores["img_mi"],  ood_scores["img_mi"])
        print(f"  Entropy AUROC (image-level): {ent_auroc:.4f}")
        print(f"  MI      AUROC (image-level): {mi_auroc:.4f}")

        plot_entropy_mi_histograms(id_scores, ood_scores, prefix + "_part1_entropy_mi_histograms.png")
        plot_image_auroc_comparison(id_scores, ood_scores, method_name, prefix + "_part1_image_auroc.png")

        print(f"\n── PART 2: Pixel-Level OoD ──────────────────────────")
        px_ent = compute_pixel_ood_metrics(ood_result, scorer="entropy", negate=False)
        px_mi  = compute_pixel_ood_metrics(ood_result, scorer="MI",      negate=False)
        for scorer, metrics in [("entropy", px_ent), ("MI", px_mi)]:
            print(f"\n  Scorer: {scorer}")
            for k in ["AUROC","AUPR","FPR95","sIoU","PPV","MeanF1"]:
                print(f"    {k:<10}: {metrics[k]:.4f}")

        results = {
            "method": method_name, "seed": args.seed,
            "image_level": {"entropy_AUROC": ent_auroc, "MI_AUROC": mi_auroc},
            "pixel_level": {"entropy": px_ent, "MI": px_mi},
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = prefix + "_ood_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_json}")


if __name__ == "__main__":
    main()