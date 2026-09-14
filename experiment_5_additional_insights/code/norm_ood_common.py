# -*- coding: utf-8 -*-
"""Shared, env-agnostic helpers for the feature-norm OoD studies (EoMT / Mask2Former / SegFormer).
Pure numpy / PIL / sklearn — no model imports, so it loads in both the `eomt` and `ssl` envs."""
import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
     "mibrahi2_hpc-my_research-1775524204")
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255.0   # == ImageNet mean
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255.0   # == ImageNet std
PEP_TEST_IDS = set(list(range(377, 408)) + list(range(471, 533)))


def pepper_test_paths():
    coco = f"{R}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json"
    root = f"{R}/data/bup_20"
    d = json.load(open(coco))
    return sorted(os.path.join(root, im["path"].lstrip("/").replace("datasets/", "", 1))
                  for im in d["images"] if im["id"] in PEP_TEST_IDS)


def tomato_paths():
    return sorted(str(p) for p in Path(f"{R}/data/tomato_esra/rgb").rglob("*.png"))


def growli_paths():
    paths = []
    for split in ("Train", "Val", "Test"):
        img_dir = f"{R}/data/growliflower_l/images/{split}"
        mask_dir = f"{R}/data/growliflower_l/labels/{split}/maskPlants"
        if not os.path.isdir(img_dir):
            continue
        for ip in sorted(Path(img_dir).glob("*.jpg")):
            stem = ip.stem
            if os.path.exists(os.path.join(mask_dir, f"{stem}_Label_NoPlants_maskPlants.png")):
                continue
            if os.path.exists(os.path.join(mask_dir, f"{stem}_Label_maskPlants.png")):
                paths.append(str(ip))
    return paths


def subsample(paths, n, seed=0):
    if n <= 0 or len(paths) <= n:
        return paths
    rng = np.random.RandomState(seed)
    return [paths[i] for i in sorted(rng.choice(len(paths), n, replace=False))]


def preprocess(path, size):
    import torch
    arr = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR),
                     dtype=np.float32) / 255.0
    arr = (arr - ADE_MEAN) / ADE_STD
    return torch.from_numpy(arr).permute(2, 0, 1).float()


def auroc_fpr(id_s, ood_s):
    """Auto-oriented AUROC + FPR@95. Returns (auroc, fpr95, direction)."""
    s = np.concatenate([id_s, ood_s])
    y = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    a = roc_auc_score(y, s); direction = "higher=OoD"
    if a < 0.5:
        a, s, direction = 1 - a, -s, "lower=OoD"
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(tpr, 0.95)
    return float(a), float(fpr[min(idx, len(fpr) - 1)]), direction


DEMO = [(f"{R}/data/bup_20/CKA_sweet_pepper_2020_summer/20201001/row6/1601542323647353.tiff", "pepper-ID-1"),
        (f"{R}/data/bup_20/CKA_sweet_pepper_2020_summer/20201001/row6/1601542307516132.tiff", "pepper-ID-2"),
        (f"{R}/data/tomato_esra/rgb/145/1631266327479411364.png", "tomato-nearOoD-1"),
        (f"{R}/data/tomato_esra/rgb/145/1631266326879132986.png", "tomato-nearOoD-2"),
        (f"{R}/data/growliflower_l/images/Test/patch_2020_08_19_65708.jpg", "growli-farOoD-1"),
        (f"{R}/data/growliflower_l/images/Test/patch_2020_09_08_63881.jpg", "growli-farOoD-2")]
