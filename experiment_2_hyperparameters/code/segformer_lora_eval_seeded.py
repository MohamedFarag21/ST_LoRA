# -*- coding: utf-8 -*-
"""
SegFormer LoRA Snapshot Ensemble Evaluation — Seeded
=====================================================
Mirrors elora_evaluation_seeded.py exactly.
Evaluates one seed's snapshot ensemble and saves results to JSON.
Also saves a visualisation panel: original | ground truth | prediction.

Usage:
    python segformer_lora_eval_seeded.py --seed 42
    python segformer_lora_eval_seeded.py --seed 42 --config_name base_r16
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
import skimage.draw
from tqdm import tqdm

from transformers import SegformerForSemanticSegmentation
from peft import PeftModel
from torchmetrics.classification import MulticlassCalibrationError
from torch_uncertainty.metrics.classification.adaptive_calibration_error import AdaptiveCalibrationError
from torchmetrics.segmentation import MeanIoU

# ─────────────────────────────────────────────────────────────────────────────
# Config — identical to training script
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
LABEL2ID   = {old: new for new, old in enumerate(sorted(ID2LABEL_ORIG.keys()))}
ID2LABEL   = {new: ID2LABEL_ORIG[old] for old, new in LABEL2ID.items()}
NUM_LABELS = len(ID2LABEL)   # 8

_REMAP_LUT = np.zeros(256, dtype=np.int64)
for old, new in LABEL2ID.items():
    _REMAP_LUT[old] = new

SPLIT_IDS = {
    "train": list(range(283, 345)) + list(range(408, 471)),
    "valid": list(range(345, 377)) + list(range(533, 564)),
    "test":  list(range(377, 408)) + list(range(471, 533)),
}

# Class colours for visualisation (RGB 0-255)
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0  bg
    [0,   0,   255],   # 1  pepper_kp
    [199, 33,  28 ],   # 2  pepper red
    [255, 247, 0  ],   # 3  pepper yellow
    [0,   255, 0  ],   # 4  pepper green
    [225, 0,   255],   # 5  pepper mixed
    [255, 102, 0  ],   # 6  pepper mixed_red
    [209, 196, 21 ],   # 7  pepper mixed_yellow
], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — plain PyTorch (no HuggingFace preprocessor)
# ─────────────────────────────────────────────────────────────────────────────

class COCOSegDataset(Dataset):
    def __init__(self, coco_file, root_dir, split, transform=None):
        with open(coco_file) as f:
            data = json.load(f)
        self.root_dir  = root_dir
        self.transform = transform
        valid_ids      = set(SPLIT_IDS[split])
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
        image   = self.transform(pil_img) if self.transform \
                  else transforms.ToTensor()(pil_img)
        return image, torch.from_numpy(seg_map).long(), np.array(pil_img), seg_map


def build_dataloader(coco_file, root_dir, split, batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    dataset = COCOSegDataset(coco_file, root_dir, split, transform=transform)

    def collate_fn(batch):
        images, seg_maps, orig_imgs, orig_segs = zip(*batch)
        return (torch.stack(images), torch.stack(seg_maps),
                list(orig_imgs), list(orig_segs))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(pretrained_name, device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, label2id=LABEL2ID,
        num_labels=NUM_LABELS, ignore_mismatched_sizes=True,
    )
    return model.eval().to(device)


def load_adapter(base_model, adapter_path, device):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_ensemble_inference(base_model, adapter_paths, dataloader, device):
    N = len(dataloader.dataset)
    M = len(adapter_paths)
    C = NUM_LABELS

    # We need H, W — peek at first batch
    first_imgs, _, _, _ = next(iter(dataloader))
    H, W = first_imgs.shape[2], first_imgs.shape[3]

    ensemble_probs = torch.zeros(M, N, C, H, W, dtype=torch.float32)
    gt_maps        = torch.zeros(N, H, W, dtype=torch.long)
    orig_imgs_all  = [None] * N
    orig_segs_all  = [None] * N
    gt_collected   = False

    for m, adapter_path in enumerate(adapter_paths):
        print(f"\n[{m+1}/{M}] Loading adapter: {adapter_path}")
        model = load_adapter(base_model, adapter_path, device)
        idx   = 0
        for images, seg_maps, orig_imgs, orig_segs in tqdm(
                dataloader, desc=f"  Inference model {m+1}"):
            pv   = images.to(device)
            out  = model(pixel_values=pv)
            logits = F.interpolate(out.logits, size=(H, W),
                                   mode="bilinear", align_corners=False)
            probs  = logits.softmax(dim=1)   # (B, C, H, W)
            B = probs.shape[0]
            ensemble_probs[m, idx:idx+B] = probs.cpu()
            if not gt_collected:
                gt_maps[idx:idx+B]         = seg_maps
                for b in range(B):
                    orig_imgs_all[idx+b] = orig_imgs[b]
                    orig_segs_all[idx+b] = orig_segs[b]
            idx += B
        gt_collected = True
        del model; torch.cuda.empty_cache()

    return ensemble_probs, gt_maps, orig_imgs_all, orig_segs_all


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — identical to Mask2Former evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _flatten(mean_probs, gt_maps):
    N, C, H, W = mean_probs.shape
    preds  = mean_probs.permute(0, 2, 3, 1).reshape(-1, C).cpu()
    target = gt_maps.reshape(-1).cpu()
    return preds, target


def compute_brier_nll_acc(mean_probs, gt_maps, eps=1e-12):
    """Multiclass Brier / NLL / pixel-accuracy over valid (gt!=255) pixels, float64.
    Definitions mirror eomt_evaluate_seeded.StreamBinMetrics exactly for cross-arch parity."""
    sq_sum = pgt_sum = nll_sum = 0.0
    correct = total = npix = 0
    C = mean_probs.shape[1]
    for i in range(mean_probs.shape[0]):
        p = mean_probs[i].permute(1, 2, 0).reshape(-1, C).double().cpu()   # (P, C)
        lab = gt_maps[i].reshape(-1).long().cpu()                          # (P,)
        m = lab != 255
        p, lab = p[m], lab[m]
        if lab.numel() == 0:
            continue
        pred = p.argmax(1)
        correct += int((pred == lab).sum()); total += int(lab.numel())
        pgt = p.gather(1, lab.unsqueeze(1)).squeeze(1).clamp_min(eps)
        sq_sum  += float((p * p).sum())
        pgt_sum += float(pgt.sum())
        nll_sum += float((-pgt.log()).sum())
        npix    += int(p.shape[0])
    npx = max(npix, 1)
    return {"Brier": (sq_sum - 2.0 * pgt_sum + npix) / npx,
            "NLL":   nll_sum / npx,
            "acc":   correct / max(total, 1)}


def compute_miou(mean_probs, gt_maps):
    preds  = mean_probs.argmax(dim=1)   # (N, H, W)
    metric = MeanIoU(num_classes=NUM_LABELS, per_class=False)
    metric.update(preds, gt_maps)
    return float(metric.compute())


def compute_ece(mean_probs, gt_maps, n_bins=10):
    preds, target = _flatten(mean_probs, gt_maps)
    return MulticlassCalibrationError(
        num_classes=NUM_LABELS, n_bins=n_bins, norm="l1")(preds, target).item()


def compute_mece(mean_probs, gt_maps, n_bins=10):
    preds, target = _flatten(mean_probs, gt_maps)
    return MulticlassCalibrationError(
        num_classes=NUM_LABELS, n_bins=n_bins, norm="max")(preds, target).item()


def compute_ace(mean_probs, gt_maps, n_bins=10):
    preds, target = _flatten(mean_probs, gt_maps)
    return AdaptiveCalibrationError(
        task="multiclass", num_bins=n_bins, norm="l1",
        num_classes=NUM_LABELS)(preds, target).item()


def compute_mace(mean_probs, gt_maps, n_bins=10):
    preds, target = _flatten(mean_probs, gt_maps)
    return AdaptiveCalibrationError(
        task="multiclass", num_bins=n_bins, norm="max",
        num_classes=NUM_LABELS)(preds, target).item()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — original | ground truth | prediction
# ─────────────────────────────────────────────────────────────────────────────

def label_to_color(label_map):
    """Convert (H, W) int label map to (H, W, 3) uint8 RGB image."""
    h, w = label_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        color_img[label_map == cls_id] = color
    return color_img


def save_visualization(orig_img, gt_seg, pred_seg, out_path, seed, img_idx=0):
    """
    Three-panel figure per image:
      Left  — original RGB image
      Centre — ground truth segmentation (coloured)
      Right  — model prediction (coloured)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(label_to_color(gt_seg))
    axes[1].set_title("Ground Truth", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(label_to_color(pred_seg))
    axes[2].set_title("Model Prediction", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # Shared legend
    patches = [
        mpatches.Patch(color=CLASS_COLORS[i] / 255.0, label=ID2LABEL[i])
        for i in range(NUM_LABELS)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_LABELS,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f"SegFormer LoRA — seed={seed}  image={img_idx}",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Visualisation → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SegFormer LoRA Ensemble Evaluation — Seeded")
    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/"
                "CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/segformer",
        help="Parent dir of config_name/seed_X subdirs")
    parser.add_argument("--config_name", type=str, default="final_model")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--shot_ids", nargs="+", type=int, default=[2, 3, 4, 5],
        help="Snapshot IDs to include in ensemble (model_shot_N)")
    parser.add_argument("--split",       type=str, default="test",
        choices=["train", "valid", "test"])
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu",         type=str, default="0")
    parser.add_argument("--n_bins",      type=int, default=10)
    parser.add_argument("--vis_idx",     type=int, default=0,
        help="Dataset index of the image to visualise")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_dir = os.path.join(args.results_dir, args.config_name, f"seed_{args.seed}")
    adapter_paths = [
        os.path.join(seed_dir, f"model_shot_{shot}")
        for shot in args.shot_ids
    ]
    missing = [p for p in adapter_paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing adapters:\n" + "\n".join(missing))

    print(f"\n{'='*60}")
    print(f"  SegFormer LoRA Evaluation")
    print(f"  Config : {args.config_name}  Seed: {args.seed}")
    print(f"  Split  : {args.split}  |  Ensemble: {len(adapter_paths)} snapshots")
    for p in adapter_paths:
        print(f"    {p}")
    print(f"{'='*60}\n")

    dataloader = build_dataloader(
        args.coco_file, args.root_dir, args.split,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Dataset: {len(dataloader.dataset)} images")

    base_model = load_base_model(args.pretrained, device)

    t0 = time.time()
    ensemble_probs, gt_maps, orig_imgs, orig_segs = run_ensemble_inference(
        base_model, adapter_paths, dataloader, device)
    elapsed = (time.time() - t0) / 60
    print(f"\nInference time: {elapsed:.1f} min")

    mean_probs = ensemble_probs.mean(dim=0)   # (N, C, H, W)
    mean_ent   = float(
        -(mean_probs * torch.log2(mean_probs + 1e-12)).sum(dim=1).mean())

    print("\nComputing metrics...")
    miou = compute_miou(mean_probs, gt_maps)
    ece  = compute_ece( mean_probs, gt_maps, n_bins=args.n_bins)
    mece = compute_mece(mean_probs, gt_maps, n_bins=args.n_bins)
    ace  = compute_ace( mean_probs, gt_maps, n_bins=args.n_bins)
    mace = compute_mace(mean_probs, gt_maps, n_bins=args.n_bins)
    bna  = compute_brier_nll_acc(mean_probs, gt_maps)

    results = {
        "mIoU":            miou,
        "ECE":             ece,
        "MECE":            mece,
        "ACE":             ace,
        "MACE":            mace,
        "Brier":           bna["Brier"],
        "NLL":             bna["NLL"],
        "acc":             bna["acc"],
        "mean_entropy":    mean_ent,
        "seed":            args.seed,
        "config_name":     args.config_name,
        "split":           args.split,
        "ensemble_members":len(adapter_paths),
        "test_samples":    len(dataloader.dataset),
        "inference_min":   round(elapsed, 2),
        "adapter_paths":   adapter_paths,
    }

    print("\n" + "=" * 55)
    print(f"  Seed {args.seed} — {args.config_name} — Results")
    print("=" * 55)
    for k in ["mIoU", "ECE", "MECE", "ACE", "MACE", "Brier", "NLL", "acc", "mean_entropy"]:
        print(f"  {k:<20} {results[k]:.4f}")
    print("=" * 55 + "\n")

    # Save JSON
    out_path = os.path.join(seed_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] Results → {out_path}")

    # ── Visualisation — one image ─────────────────────────────────────────────
    vis_idx = min(args.vis_idx, len(dataloader.dataset) - 1)
    orig_img  = orig_imgs[vis_idx]                          # (H, W, 3) uint8
    gt_seg    = orig_segs[vis_idx]                          # (H, W) int
    pred_seg  = mean_probs[vis_idx].argmax(dim=0).numpy()  # (H, W) int

    vis_path = os.path.join(seed_dir, f"vis_{args.split}_img{vis_idx}.png")
    save_visualization(orig_img, gt_seg, pred_seg, vis_path,
                       seed=args.seed, img_idx=vis_idx)


if __name__ == "__main__":
    main()
