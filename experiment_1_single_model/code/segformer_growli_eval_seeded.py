# -*- coding: utf-8 -*-
"""
SegFormer (B2 / B4) on GrowliFlower-L — SINGLE-MODEL evaluation, seeded.
========================================================================
Evaluates ONE snapshot (default the last, shot 5 — no ensembling) of an ST-LoRA or FRE
model trained by segformer_growli_train_seeded.py, on the GrowliFlower-L Test split.
Reports mIoU + all calibration metrics (ECE / MECE / ACE / MACE / Brier / NLL / acc).

Metric definitions are copied verbatim from segformer_lora_eval_seeded.py (float64 Brier/NLL,
torchmetrics ECE/MECE, torch_uncertainty ACE/MACE) so numbers are comparable across the study.

Usage:
    python segformer_growli_eval_seeded.py --method stlora --shot 5 \
        --pretrained nvidia/segformer-b2-finetuned-ade-512-512 \
        --seed 42 --root_dir /path/to/growliflower_l --results_dir OUT
    python segformer_growli_eval_seeded.py --method fre --shot 5 \
        --pretrained nvidia/segformer-b4-finetuned-ade-512-512 --seed 42 ...
"""

import os
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

from transformers import SegformerForSemanticSegmentation
from peft import PeftModel
from torchmetrics.classification import MulticlassCalibrationError
from torch_uncertainty.metrics.classification.adaptive_calibration_error import AdaptiveCalibrationError
from torchmetrics.segmentation import MeanIoU

# ─────────────────────────────────────────────────────────────────────────────
# Config — binary plant vs background (matches the trainer)
# ─────────────────────────────────────────────────────────────────────────────
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255
ID2LABEL   = {0: "bg", 1: "plant"}
LABEL2ID   = {"bg": 0, "plant": 1}
NUM_LABELS = 2
IGNORE_INDEX = 255
SPLITS = {"train": "Train", "valid": "Val", "test": "Test"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class GrowliFlowerEvalDataset(Dataset):
    def __init__(self, root_dir, split, resize=None):
        self.img_dir = os.path.join(root_dir, "images", SPLITS[split])
        self.lbl_dir = os.path.join(root_dir, "labels", SPLITS[split])
        self.resize = resize
        self.stems = sorted(os.path.splitext(f)[0]
                            for f in os.listdir(self.img_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)])

    def __len__(self): return len(self.stems)

    def _mask(self, masktype, stem):
        p = os.path.join(self.lbl_dir, masktype, f"{stem}_Label_{masktype}.png")
        return (np.array(Image.open(p)) > 0) if os.path.exists(p) else None

    def __getitem__(self, idx):
        stem = self.stems[idx]
        pil_img = Image.open(os.path.join(self.img_dir, f"{stem}.jpg")).convert("RGB")
        W, H = pil_img.size
        seg = np.zeros((H, W), dtype=np.uint8)
        plant = self._mask("maskPlants", stem)
        if plant is not None: seg[plant] = 1
        void = self._mask("maskVoid", stem)
        if void is not None: seg[void] = IGNORE_INDEX
        if self.resize is not None:
            rh, rw = self.resize
            pil_img = pil_img.resize((rw, rh), Image.BILINEAR)
            seg = np.array(Image.fromarray(seg).resize((rw, rh), Image.NEAREST))
        image = self.transform(pil_img)
        return image, torch.from_numpy(seg.astype(np.int64)).long()


def build_dataloader(root_dir, split, batch_size, num_workers=4, resize=None):
    ds = GrowliFlowerEvalDataset(root_dir, split, resize=resize)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_base_model(pretrained_name, device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, label2id=LABEL2ID,
        num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
    return model.eval().to(device)


def load_single_model(method, pretrained, seed_dir, shot, device):
    base = load_base_model(pretrained, device)
    if method == "stlora":
        adapter = os.path.join(seed_dir, f"model_shot_{shot}")
        if not os.path.isdir(adapter):
            raise FileNotFoundError(adapter)
        return PeftModel.from_pretrained(base, adapter).eval().to(device)
    ckpt = os.path.join(seed_dir, f"model_shot_{shot}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    base.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return base.eval().to(device)


@torch.no_grad()
def run_inference(model, dataloader, device):
    first, _ = next(iter(dataloader))
    H, W = first.shape[2], first.shape[3]
    N = len(dataloader.dataset)
    probs_all = torch.zeros(N, NUM_LABELS, H, W, dtype=torch.float32)
    gt_all = torch.zeros(N, H, W, dtype=torch.long)
    idx = 0
    for images, seg in tqdm(dataloader, desc="  inference"):
        out = model(pixel_values=images.to(device))
        logits = F.interpolate(out.logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = logits.softmax(dim=1)
        B = probs.shape[0]
        probs_all[idx:idx+B] = probs.cpu()
        gt_all[idx:idx+B] = seg
        idx += B
    return probs_all, gt_all


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — copied verbatim from segformer_lora_eval_seeded.py
# ─────────────────────────────────────────────────────────────────────────────
def _flatten(mean_probs, gt_maps):
    N, C, H, W = mean_probs.shape
    preds = mean_probs.permute(0, 2, 3, 1).reshape(-1, C).cpu()
    target = gt_maps.reshape(-1).cpu()
    return preds, target


def compute_brier_nll_acc(mean_probs, gt_maps, eps=1e-12):
    sq_sum = pgt_sum = nll_sum = 0.0
    correct = total = npix = 0
    C = mean_probs.shape[1]
    for i in range(mean_probs.shape[0]):
        p = mean_probs[i].permute(1, 2, 0).reshape(-1, C).double().cpu()
        lab = gt_maps[i].reshape(-1).long().cpu()
        m = lab != IGNORE_INDEX
        p, lab = p[m], lab[m]
        if lab.numel() == 0:
            continue
        pred = p.argmax(1)
        correct += int((pred == lab).sum()); total += int(lab.numel())
        pgt = p.gather(1, lab.unsqueeze(1)).squeeze(1).clamp_min(eps)
        sq_sum += float((p * p).sum())
        pgt_sum += float(pgt.sum())
        nll_sum += float((-pgt.log()).sum())
        npix += int(p.shape[0])
    npx = max(npix, 1)
    return {"Brier": (sq_sum - 2.0 * pgt_sum + npix) / npx,
            "NLL": nll_sum / npx,
            "acc": correct / max(total, 1)}


def compute_miou(mean_probs, gt_maps):
    metric = MeanIoU(num_classes=NUM_LABELS, per_class=False)
    metric.update(mean_probs.argmax(dim=1), gt_maps)
    return float(metric.compute())


def compute_ece(mean_probs, gt_maps, n_bins=10):
    preds, target = _flatten(mean_probs, gt_maps)
    return MulticlassCalibrationError(num_classes=NUM_LABELS, n_bins=n_bins, norm="l1")(preds, target).item()


def compute_mece(mean_probs, gt_maps, n_bins=10):
    preds, target = _flatten(mean_probs, gt_maps)
    return MulticlassCalibrationError(num_classes=NUM_LABELS, n_bins=n_bins, norm="max")(preds, target).item()


def compute_ace(mean_probs, gt_maps, n_bins=10):
    preds, target = _flatten(mean_probs, gt_maps)
    return AdaptiveCalibrationError(task="multiclass", num_bins=n_bins, norm="l1",
                                    num_classes=NUM_LABELS)(preds, target).item()


def compute_mace(mean_probs, gt_maps, n_bins=10):
    preds, target = _flatten(mean_probs, gt_maps)
    return AdaptiveCalibrationError(task="multiclass", num_bins=n_bins, norm="max",
                                    num_classes=NUM_LABELS)(preds, target).item()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SegFormer GrowliFlower-L single-model eval")
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--results_dir", type=str, required=True,
                   help="parent dir containing seed_<seed>/ (== training --base_save_dir)")
    p.add_argument("--pretrained", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    p.add_argument("--method", type=str, default="stlora", choices=["stlora", "fre"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shot", type=int, default=5, help="single snapshot to evaluate (last = 5)")
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--resize", nargs=2, type=int, default=None, metavar=("H", "W"))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--n_bins", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_dir = os.path.join(args.results_dir, f"seed_{args.seed}")
    resize = tuple(args.resize) if args.resize else None

    print(f"\n{'='*60}\n  SegFormer GrowliFlower-L eval — {args.method} shot {args.shot}\n"
          f"  backbone {args.pretrained}  seed {args.seed}  split {args.split}\n{'='*60}\n")

    dataloader = build_dataloader(args.root_dir, args.split, args.batch_size,
                                  args.num_workers, resize=resize)
    print(f"Dataset: {len(dataloader.dataset)} images")

    model = load_single_model(args.method, args.pretrained, seed_dir, args.shot, device)

    t0 = time.time()
    probs, gt = run_inference(model, dataloader, device)
    elapsed = (time.time() - t0) / 60

    mean_ent = float(-(probs * torch.log2(probs + 1e-12)).sum(dim=1).mean())
    bna = compute_brier_nll_acc(probs, gt)
    results = {
        "mIoU":  compute_miou(probs, gt),
        "ECE":   compute_ece(probs, gt, n_bins=args.n_bins),
        "MECE":  compute_mece(probs, gt, n_bins=args.n_bins),
        "ACE":   compute_ace(probs, gt, n_bins=args.n_bins),
        "MACE":  compute_mace(probs, gt, n_bins=args.n_bins),
        "Brier": bna["Brier"], "NLL": bna["NLL"], "acc": bna["acc"],
        "mean_entropy": mean_ent,
        "method": args.method, "seed": args.seed, "shot": args.shot, "split": args.split,
        "backbone": args.pretrained, "samples": len(dataloader.dataset),
        "inference_min": round(elapsed, 2),
    }
    print("\n" + "=" * 50)
    for k in ["mIoU", "ECE", "MECE", "ACE", "MACE", "Brier", "NLL", "acc", "mean_entropy"]:
        print(f"  {k:<14} {results[k]:.4f}")
    print("=" * 50 + "\n")

    out_path = os.path.join(seed_dir, f"eval_{args.split}_single_shot{args.shot}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
