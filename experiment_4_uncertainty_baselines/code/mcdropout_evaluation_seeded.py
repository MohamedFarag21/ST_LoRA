"""
MC Dropout Evaluation — Mask2Former Sweet Pepper Segmentation
==============================================================
Loads the best trained model for one seed, enables MC Dropout at inference,
runs T stochastic forward passes, and computes uncertainty decomposition +
segmentation metrics.

During evaluation:
  - Whole model is frozen (no gradients)
  - Dropout layer stays in train() mode → stochastic predictions
  - T passes replace the snapshot ensemble used in LoRA evaluation

Usage:
    python mcdropout_evaluation_seeded.py --seed 42
    python mcdropout_evaluation_seeded.py --seed 42 --T 20
"""

import os
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import skimage.draw
from tqdm import tqdm

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from torchmetrics.classification import MulticlassCalibrationError
from torch_uncertainty.metrics.classification.adaptive_calibration_error import AdaptiveCalibrationError
import evaluate

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
LABEL2ID   = {old: new for new, old in enumerate(sorted(ID2LABEL_ORIG.keys()))}
ID2LABEL   = {new: ID2LABEL_ORIG[old] for old, new in LABEL2ID.items()}
NUM_LABELS = len(ID2LABEL)

_REMAP_LUT = np.zeros(256, dtype=np.int64)
for old, new in LABEL2ID.items():
    _REMAP_LUT[old] = new


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

        sem_map = np.zeros((H, W), dtype=np.uint8)
        for ann in self.ann_lookup.get(info["id"], []):
            for poly in ann.get("segmentation", []):
                pts = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(pts[:, 1], pts[:, 0], sem_map.shape)
                sem_map[rr, cc] = ann["category_id"]

        seg_map = _REMAP_LUT[sem_map.astype(np.int64)]
        image   = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        return image, torch.from_numpy(seg_map).long(), np.array(pil_img), seg_map


def build_dataloader(coco_file, root_dir, split, batch_size, num_workers):
    preprocessor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=False,
        do_resize=False, do_rescale=False, do_normalize=False,
        num_labels=NUM_LABELS,
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    dataset = COCOSegDataset(coco_file, root_dir, split, transform=transform)

    def collate_fn(batch):
        images, seg_maps, orig_imgs, orig_segs = zip(*batch)
        out = preprocessor(list(images), segmentation_maps=list(seg_maps),
                           return_tensors="pt")
        out["original_images"]            = list(orig_imgs)
        out["original_segmentation_maps"] = list(orig_segs)
        return out

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=num_workers,
                      persistent_workers=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(pretrained_name, checkpoint_path, dropout_p, device):
    """
    Load base Mask2Former, add dropout to class predictor,
    load trained weights, then freeze everything except dropout.
    """
    # 1. Load base architecture
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
    )

    # 2. Add dropout to class predictor (must match training architecture)
    original = model.class_predictor
    model.class_predictor = nn.Sequential(
        nn.Dropout(p=dropout_p),
        original,
    )

    # 3. Load trained weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"[Loaded] {checkpoint_path}")

    # 4. Move to device
    model = model.to(device)

    # 5. Freeze everything, then set dropout back to train mode
    #    — no gradients needed at eval, but dropout must stay stochastic
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    # Disable all gradients
    for param in model.parameters():
        param.requires_grad = False

    return model


# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout inference — T stochastic forward passes
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_mc_inference(model, dataloader, device, H, W, T):
    """
    Run T stochastic forward passes through the model.
    Accumulates sum_probs and sum_H incrementally (memory-efficient).

    Returns:
        sum_probs : (N, C, H, W) — accumulated softmax probs across T passes
        sum_H     : (N, H, W)    — accumulated per-pass pixel entropy
        gt_maps   : (N, H, W)    — ground truth segmentation maps
        T         : number of passes (for normalisation)
    """
    N = len(dataloader.dataset)
    C = NUM_LABELS

    sum_probs = torch.zeros(N, C, H, W, dtype=torch.float32)  # CPU
    sum_H     = torch.zeros(N, H, W,    dtype=torch.float32)  # CPU
    gt_maps   = torch.zeros(N, H, W,    dtype=torch.long)
    gt_collected = False

    for t in range(T):
        idx = 0
        for batch in tqdm(dataloader, desc=f"  Pass {t+1}/{T}", leave=False):
            pv  = batch["pixel_values"].to(device)
            out = model(pv)

            # Build soft seg map (B, C, H, W)
            class_probs = out.class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = out.masks_queries_logits.sigmoid()
            seg = torch.einsum("bqc,bqhw->bchw", class_probs, masks_probs)
            seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
            seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)

            B = seg.shape[0]
            H_t = -(seg * (seg + 1e-12).log()).sum(dim=1)  # (B, H, W)

            sum_probs[idx:idx+B] += seg.cpu()
            sum_H[idx:idx+B]     += H_t.cpu()

            if not gt_collected:
                for b, gt in enumerate(batch["original_segmentation_maps"]):
                    gt_maps[idx+b] = torch.from_numpy(gt).long()
            idx += B

        gt_collected = True

    return sum_probs, sum_H, gt_maps


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty decomposition (same formula as LoRA pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def compute_uncertainty(sum_probs, sum_H, T):
    mean_probs = sum_probs / T      # (N, C, H, W)
    mean_H     = sum_H     / T      # (N, H, W)  — mean aleatoric per pixel

    H_mean    = -(mean_probs * (mean_probs + 1e-12).log()).sum(dim=1)  # total
    aleatoric = mean_H
    epistemic = H_mean - aleatoric
    variance  = ((sum_probs / T) - mean_probs).pow(2).sum(dim=1)  # approx

    return {
        "mean_probs": mean_probs,
        "aleatoric":  aleatoric,
        "total":      H_mean,
        "epistemic":  epistemic,
        "variance":   variance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (identical to LoRA evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def _flatten(mean_probs, gt_maps):
    N, C, H, W = mean_probs.shape
    preds  = mean_probs.permute(0, 2, 3, 1).reshape(-1, C).cpu()
    target = gt_maps.reshape(-1).cpu()
    return preds, target


def compute_miou(mean_probs, gt_maps):
    preds  = mean_probs.argmax(dim=1)
    metric = evaluate.load("mean_iou")
    metric.add_batch(
        predictions=[preds[i].numpy() for i in range(preds.shape[0])],
        references=[gt_maps[i].numpy() for i in range(gt_maps.shape[0])],
    )
    return metric.compute(num_labels=NUM_LABELS, ignore_index=255)["mean_iou"]


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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="MC Dropout Evaluation — Mask2Former")

    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/mcdropout",
        help="Parent dir of seed_X subdirs")

    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--dropout_p",  type=float, default=0.25)
    parser.add_argument("--T",          type=int,   default=10,
        help="Number of stochastic forward passes (MC samples)")

    parser.add_argument("--split",       type=str, default="test",
        choices=["train", "valid", "test"])
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu",         type=str, default="0")
    parser.add_argument("--height",      type=int, default=1280)
    parser.add_argument("--width",       type=int, default=720)
    parser.add_argument("--n_bins",      type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint path — use model_final.pt saved at end of training
    seed_dir       = os.path.join(args.results_dir, f"seed_{args.seed}")
    checkpoint_path = os.path.join(seed_dir, "model_final.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Expected model_final.pt saved by training script."
        )

    print(f"\n{'='*60}")
    print(f"  MC Dropout Evaluation")
    print(f"  Seed       : {args.seed}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  T passes   : {args.T}")
    print(f"  Split      : {args.split}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    dataloader = build_dataloader(
        args.coco_file, args.root_dir, args.split,
        args.batch_size, args.num_workers,
    )
    print(f"Dataset size: {len(dataloader.dataset)}")

    model = load_model(args.pretrained, checkpoint_path, args.dropout_p, device)

    t0 = time.time()
    sum_probs, sum_H, gt_maps = run_mc_inference(
        model, dataloader, device, args.height, args.width, args.T
    )
    inference_time = (time.time() - t0) / 60
    print(f"\nInference time: {inference_time:.1f} min  ({args.T} passes)")

    print("\nComputing uncertainty maps...")
    uncertainty = compute_uncertainty(sum_probs, sum_H, args.T)
    mean_probs  = uncertainty["mean_probs"]

    print("Computing metrics...")
    miou = compute_miou(mean_probs, gt_maps)
    ece  = compute_ece( mean_probs, gt_maps, args.n_bins)
    mece = compute_mece(mean_probs, gt_maps, args.n_bins)
    ace  = compute_ace( mean_probs, gt_maps, args.n_bins)
    mace = compute_mace(mean_probs, gt_maps, args.n_bins)

    results = {
        "mIoU":               miou,
        "ECE":                ece,
        "MECE":               mece,
        "ACE":                ace,
        "MACE":               mace,
        "mean_aleatoric":     float(uncertainty["aleatoric"].mean()),
        "mean_epistemic":     float(uncertainty["epistemic"].mean()),
        "mean_total":         float(uncertainty["total"].mean()),
        "mean_pred_variance": float(uncertainty["variance"].mean()),
        "seed":               args.seed,
        "split":              args.split,
        "T_passes":           args.T,
        "test_samples":       len(dataloader.dataset),
        "inference_min":      round(inference_time, 2),
        "checkpoint":         checkpoint_path,
    }

    print("\n" + "=" * 55)
    print(f"  Seed {args.seed} — MC Dropout Evaluation Results")
    print("=" * 55)
    for k in ["mIoU", "ECE", "MECE", "ACE", "MACE",
              "mean_aleatoric", "mean_epistemic", "mean_total", "mean_pred_variance"]:
        print(f"  {k:<30} {results[k]:.4f}")
    print("=" * 55)

    out_path = os.path.join(seed_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
