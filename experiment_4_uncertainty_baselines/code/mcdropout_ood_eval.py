# -*- coding: utf-8 -*-
"""
Far OoD Detection — Mask2Former + MC Dropout
=============================================
In-distribution  : Sweet pepper test set
Far OoD          : GrowliFlower-L images

Method : T stochastic forward passes (dropout active) replace the
         snapshot ensemble used in the LoRA version.
         Scorers: MSP, FG-MSP, NegEnt, NegMI (same as LoRA OoD eval).

Usage:
    python mcdropout_ood_eval.py --seed 42 --growliflower_dir /path/to/growliflower
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

from PIL import Image
import skimage.draw
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from image_ood_extra import (metrics_all, aggregate_imagelevel,
                             build_tomato_ood_loader, build_growli_ood_loader,
                             TOMATO_ROOT_DEFAULT)

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


class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        root = Path(root_dir)
        self.paths = sorted([p for p in root.rglob("*")
                             if p.suffix.lower() in IMAGE_EXTENSIONS])
        if not self.paths:
            raise RuntimeError(f"No images found in {root_dir}")
        print(f"[GrowliFlower-L] Found {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pil_img = Image.open(self.paths[idx]).convert("RGB")
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
        num_labels=NUM_LABELS,
    )
    ds = SweetPepperTestDataset(coco_file, root_dir, transform=make_transform())

    def collate(batch):
        images, seg_maps = zip(*batch)
        return preprocessor(list(images), segmentation_maps=list(seg_maps),
                            return_tensors="pt")

    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


def build_ood_loader(growliflower_dir, batch_size, num_workers):
    # CONTAMINATION FIX: clean plants-filtered growli frames (mask-free, ~1970),
    # matching ood_v2_fixed / DDU / calibrators — NOT the buggy 10990-file rglob.
    return build_growli_ood_loader(growliflower_dir, batch_size, num_workers,
                                   H=1280, W=720)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — full model + dropout on class predictor
# ─────────────────────────────────────────────────────────────────────────────

def load_mc_model(pretrained_name, checkpoint_path, dropout_p, device):
    """Load trained MC Dropout model and enable stochastic inference."""
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
    )
    # Rebuild dropout architecture (must match training)
    original = model.class_predictor
    model.class_predictor = nn.Sequential(nn.Dropout(p=dropout_p), original)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Freeze all, keep dropout stochastic
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Seg prob builder
# ─────────────────────────────────────────────────────────────────────────────

def build_seg_probs(outputs, H, W):
    class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = outputs.masks_queries_logits.sigmoid()
    seg = torch.einsum("bqc,bqhw->bchw", class_probs, masks_probs)
    seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
    seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
    return seg


# ─────────────────────────────────────────────────────────────────────────────
# T stochastic passes — accumulate sum_probs and sum_H
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_scores_id(model, dataloader, device, H, W, T):
    N = len(dataloader.dataset)
    sum_probs = torch.zeros(N, NUM_LABELS, H, W, dtype=torch.float32)
    sum_H     = torch.zeros(N, H, W, dtype=torch.float32)

    for t in range(T):
        idx = 0
        for batch in tqdm(dataloader, desc=f"  ID pass {t+1}/{T}", leave=False):
            pv  = batch["pixel_values"].to(device)
            seg = build_seg_probs(model(pv), H, W)
            B   = seg.shape[0]
            H_t = -(seg * (seg + 1e-12).log()).sum(dim=1)   # full 9-class entropy
            sum_probs[idx:idx+B] += seg.cpu()
            sum_H[idx:idx+B]     += H_t.cpu()
            idx += B

    return _scores_from_sums(sum_probs, sum_H, T)


@torch.no_grad()
def compute_scores_ood(model, dataloader, device, H, W, T):
    N = len(dataloader.dataset)
    sum_probs = torch.zeros(N, NUM_LABELS, H, W, dtype=torch.float32)
    sum_H     = torch.zeros(N, H, W, dtype=torch.float32)

    for t in range(T):
        idx = 0
        for images in tqdm(dataloader, desc=f"  OoD pass {t+1}/{T}", leave=False):
            pv  = images.to(device)
            seg = build_seg_probs(model(pv), H, W)
            B   = seg.shape[0]
            H_t = -(seg * (seg + 1e-12).log()).sum(dim=1)   # full 9-class entropy
            sum_probs[idx:idx+B] += seg.cpu()
            sum_H[idx:idx+B]     += H_t.cpu()
            idx += B

    return _scores_from_sums(sum_probs, sum_H, T)


def _scores_from_sums(sum_probs, sum_H, T):
    # Memory-lean: mean in-place + per-class entropy loops (N×H×W accumulators).
    sum_probs /= T
    mp        = sum_probs
    mean_H    = sum_H / T          # mean of per-pass full-class entropies
    C         = mp.shape[1]

    msp    = mp.max(dim=1).values.mean(dim=(1, 2))
    fg_msp = mp[:, 1:].max(dim=1).values.mean(dim=(1, 2))

    H_mean = torch.zeros_like(mp[:, 0])
    for c in range(C):
        pc = mp[:, c]
        H_mean -= pc * (pc + 1e-12).log()
    neg_ent = -H_mean.mean(dim=(1, 2))

    # Full 9-class BALD mutual information = H(mean) - mean(H), matching the
    # original ood_eval_comprehensive_with_ddu.py (natural-log base; AUROC-invariant).
    neg_MI = -(H_mean - mean_H).mean(dim=(1, 2))

    return {
        "msp":     msp.numpy(),
        "fg_msp":  fg_msp.numpy(),
        "neg_ent": neg_ent.numpy(),
        "neg_mi":  neg_MI.numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# OoD metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ood_metrics(id_scores, ood_scores):
    return metrics_all(id_scores, ood_scores)   # AUROC/AUPR/FPR95


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Far OoD Detection — MC Dropout")
    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/mcdropout")
    parser.add_argument("--growliflower_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/mcdropout/ood")

    parser.add_argument("--seeds",      nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    parser.add_argument("--dropout_p",  type=float, default=0.25)
    parser.add_argument("--T",          type=int,   default=10,
        help="Number of stochastic forward passes per seed")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu",         type=str, default="0")
    parser.add_argument("--height",      type=int, default=1280)
    parser.add_argument("--width",       type=int, default=720)
    parser.add_argument("--tomato_root",  type=str, default=TOMATO_ROOT_DEFAULT)
    parser.add_argument("--tomato_split", type=str, default="val")
    parser.add_argument("--tomato_max",   type=int, default=1200)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "mcdropout_ood.json")

    print(f"\n{'='*65}")
    print(f"  Far OoD — MC Dropout")
    print(f"  Seeds : {args.seeds}  |  T passes : {args.T}")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")

    id_loader  = build_id_loader(args.coco_file, args.root_dir,
                                 args.batch_size, args.num_workers)
    ood_loader = build_ood_loader(args.growliflower_dir,
                                  args.batch_size, args.num_workers)
    tom_loader = build_tomato_ood_loader(
        args.tomato_root, args.tomato_split, args.batch_size, args.num_workers,
        H=args.height, W=args.width, max_images=args.tomato_max)
    print(f"  ID  samples : {len(id_loader.dataset)}")
    print(f"  OoD growli  : {len(ood_loader.dataset)}")
    print(f"  OoD tomato  : {len(tom_loader.dataset)}\n")

    SCORERS = {
        "MSP":    "msp",
        "FG_MSP": "fg_msp",
        "NegEnt": "neg_ent",
        "NegMI":  "neg_mi",
    }
    SOURCES = ["growliflower", "tomato"]      # far-OoD, near-OoD (image-level)

    seed_results = []
    t_start = time.time()

    for seed in args.seeds:
        print(f"── Seed {seed} ──────────────────────────────────────")
        ckpt = os.path.join(args.results_dir, f"seed_{seed}", "model_final.pt")
        if not os.path.exists(ckpt):
            print(f"  [SKIP] Missing checkpoint: {ckpt}")
            continue

        model = load_mc_model(args.pretrained, ckpt, args.dropout_p, device)

        H, W = args.height, args.width
        id_img   = compute_scores_id( model, id_loader,  device, H, W, args.T)
        ood_grow = compute_scores_ood(model, ood_loader, device, H, W, args.T)
        ood_tom  = compute_scores_ood(model, tom_loader, device, H, W, args.T)

        del model; torch.cuda.empty_cache()
        ood_by_src = {"growliflower": ood_grow, "tomato": ood_tom}

        seed_entry = {"seed": seed}
        for src in SOURCES:
            ood_img = ood_by_src[src]
            for name, key in SCORERS.items():
                m = compute_ood_metrics(id_img[key], ood_img[key])
                for mk in ("AUROC", "AUPR", "FPR95"):
                    seed_entry[f"{src}::{name}::{mk}"] = m[mk]
                seed_entry[f"{src}::{name}::id_mean"]  = float(id_img[key].mean())
                seed_entry[f"{src}::{name}::ood_mean"] = float(ood_img[key].mean())
                print(f"  [{src:<12}] {name:<8}  AUROC={m['AUROC']:.4f}  "
                      f"AUPR={m['AUPR']:.4f}  FPR95={m['FPR95']:.4f}")
        seed_results.append(seed_entry)

    elapsed = (time.time() - t_start) / 60

    meta = {
        "method": "MC Dropout", "T_passes": args.T,
        "seeds": args.seeds, "id_dataset": "sweet_pepper_test",
        "tomato_split": args.tomato_split, "tomato_max": args.tomato_max,
        "headline_scorer": "NegEnt (mean per-pixel entropy)",
        "elapsed_min": round(elapsed, 2),
    }
    summary = aggregate_imagelevel(seed_results, list(SCORERS.keys()), SOURCES, meta)

    img_path = os.path.join(args.out_dir, "mcdropout_ood_imagelevel.json")
    with open(img_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  MC Dropout image-level OoD ({len(seed_results)} seeds, T={args.T})")
    for src in SOURCES:
        print(f"  ── {src} ──")
        print(f"  {'Scorer':<8}  {'AUROC':>16}  {'FPR95':>16}")
        for name in SCORERS:
            au = summary[src][name]["AUROC"]; fp = summary[src][name]["FPR95"]
            print(f"  {name:<8}  {au['mean']:.4f}±{au['std']:.4f}  "
                  f"{fp['mean']:.4f}±{fp['std']:.4f}")
    print(f"{'='*70}\n[Saved] {img_path}")


if __name__ == "__main__":
    main()