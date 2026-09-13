# -*- coding: utf-8 -*-
"""
Shared helpers to extend the image-level OoD evaluators with
  (a) FPR@95 (canonical roc_curve convention, same as ade_base_tomato_pixel_ood.py),
  (b) a near-OoD source: whole TOMATO images (image-level), reusing TomatoDataset.

Scorer convention throughout: each per-image scorer returns a value where
HIGHER = more in-distribution (e.g. MSP, FG_MSP, NegEnt = -entropy, NegMI).
The detection score fed to the metrics is therefore -scorer (higher = more OoD),
identical to the existing compute_ood_metrics used across the UQ scripts.

Import-only module — no side effects.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255

TOMATO_ROOT_DEFAULT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/"
                       "mibrahi2_hpc-my_research-1783386603/"
                       "mibrahi2_hpc-my_research-1775524204/data/tomato_esra")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_fpr95(labels, scores):
    """FPR at 95% TPR. labels: 1=OoD(positive), 0=ID. scores: higher=more OoD.
    Canonical convention (matches ade_base_tomato_pixel_ood.compute_fpr95)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95)
    return float(fpr[min(idx, len(fpr) - 1)])


def metrics_all(id_scores, ood_scores):
    """AUROC/AUPR/FPR95 from per-image scorer arrays (higher scorer = more ID)."""
    id_scores  = np.asarray(id_scores,  dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    det = np.concatenate([-id_scores, -ood_scores])           # higher = more OoD
    lab = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    return {"AUROC": float(roc_auc_score(lab, det)),
            "AUPR":  float(average_precision_score(lab, det)),
            "FPR95": compute_fpr95(lab, det)}


def metrics_all_positive(id_scores, ood_scores):
    """AUROC/AUPR/FPR95 where the scorer is ALREADY oriented higher=more OoD
    (no negation) — e.g. DDU entropy / negated-GMM-loglik."""
    id_scores  = np.asarray(id_scores,  dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    sc  = np.concatenate([id_scores, ood_scores])
    lab = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    return {"AUROC": float(roc_auc_score(lab, sc)),
            "AUPR":  float(average_precision_score(lab, sc)),
            "FPR95": compute_fpr95(lab, sc)}


def aggregate_imagelevel(seed_results, scorers, sources, meta):
    """Build a nested summary from per-seed entries keyed 'src::scorer::metric'.
    metric in {AUROC, AUPR, FPR95}. Returns dict: meta + [src][scorer][metric]=agg."""
    def agg(vals):
        vals = [float(v) for v in vals]
        return {"mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals}
    summary = dict(meta)
    summary["scorers"] = list(scorers)
    summary["sources"] = list(sources)
    for src in sources:
        summary[src] = {}
        for sc in scorers:
            summary[src][sc] = {
                m: agg([r[f"{src}::{sc}::{m}"] for r in seed_results])
                for m in ("AUROC", "AUPR", "FPR95")}
    summary["per_seed"] = seed_results
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Near-OoD source: whole tomato images (image-level)
# ─────────────────────────────────────────────────────────────────────────────

def _ade_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])


class TomatoImageOoD(Dataset):
    """Whole tomato images as image-level OoD samples (returns image tensor only).

    Deterministically subsamples to `max_images` (seed-fixed) to bound the
    CPU score buffers, which scale as N*NUM_LABELS*H*W in the callers.
    """

    def __init__(self, root_dir, split, transform, max_images=None, subsample_seed=0):
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        from mask2former_lora_train_tomato import TomatoDataset  # noqa: E402
        self.ds = TomatoDataset(root_dir, split=split)
        self.transform = transform
        n = len(self.ds)
        self.index = np.arange(n)
        if max_images is not None and max_images <= 0:   # <=0 => use ALL images
            max_images = None
        if max_images is not None and n > max_images:
            rng = np.random.RandomState(subsample_seed)
            self.index = np.sort(rng.choice(n, size=max_images, replace=False))
        print(f"[Tomato-{split}] {n} images available; using {len(self.index)} "
              f"(image-level near-OoD)", flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        img, _sem, _key = self.ds[int(self.index[i])]
        return self.transform(img)


def build_tomato_ood_loader(root_dir, split, batch_size, num_workers,
                            H=1280, W=720, max_images=1200, subsample_seed=0):
    """Loader yielding tomato image tensors resized to (H, W) with ADE norm:
    drop-in compatible with the UQ scripts' compute_scores_ood(dataloader,...)."""
    transform = transforms.Compose([transforms.Resize((H, W)), _ade_transform()])
    ds = TomatoImageOoD(root_dir, split, transform,
                        max_images=max_images, subsample_seed=subsample_seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Far-OoD source: CLEAN GrowliFlower frames (plants-filtered, mask-free)
#
# CONTAMINATION FIX. The old FlatImageDataset rglob'd the whole growliflower_l
# tree with IMAGE_EXTENSIONS including ".png", so it scored 2198 real jpg frames
# PLUS 8792 mask PNG label files (maskPlants/Leaves/Stems/Void) as "OoD images"
# (= 10990). Those solid-palette mask images are trivially OoD and also blew the
# per-frame score buffer to ~360 GB (OoM). This helper reproduces exactly the
# canonical GrowliFlowerDataset.pairs set used by ood_v2_fixed / DDU / the
# calibrators: walk images/{Train,Val,Test}, keep only frames that have a real
# maskPlants label and skip the "NoPlants" frames  ->  ~1970 "frames with plants".
# ─────────────────────────────────────────────────────────────────────────────

def growli_with_plants_image_paths(root_dir):
    """Return the sorted list of ~1970 real growli jpg frames 'with plants',
    identical selection to GrowliFlowerDataset.pairs (mask-free image list)."""
    paths = []
    for split in ("Train", "Val", "Test"):
        img_dir  = os.path.join(root_dir, "images", split)
        mask_dir = os.path.join(root_dir, "labels", split, "maskPlants")
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            continue
        for img_path in sorted(Path(img_dir).glob("*.jpg")):
            stem     = img_path.stem
            mask     = os.path.join(mask_dir, f"{stem}_Label_maskPlants.png")
            noplants = os.path.join(mask_dir, f"{stem}_Label_NoPlants_maskPlants.png")
            if os.path.exists(noplants):
                continue
            if not os.path.exists(mask):
                continue
            paths.append(str(img_path))
    return paths


class GrowliImageOoD(Dataset):
    """Clean growli 'with plants' frames as image-level OoD (image tensor only)."""

    def __init__(self, root_dir, transform):
        self.paths = growli_with_plants_image_paths(root_dir)
        if not self.paths:
            raise RuntimeError(f"No growli 'with plants' frames under {root_dir}")
        self.transform = transform
        print(f"[GrowliFlower-L] {len(self.paths)} frames with plants "
              f"(mask-free, image-level far-OoD)", flush=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


def build_growli_ood_loader(root_dir, batch_size, num_workers, H=1280, W=720):
    """Loader yielding CLEAN growli image tensors (H, W, ADE norm): drop-in for
    the UQ scripts' compute_scores_ood(dataloader, ...). Replaces the buggy
    FlatImageDataset path."""
    transform = transforms.Compose([transforms.Resize((H, W)), _ade_transform()])
    ds = GrowliImageOoD(root_dir, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers,
                      persistent_workers=(num_workers > 0))
