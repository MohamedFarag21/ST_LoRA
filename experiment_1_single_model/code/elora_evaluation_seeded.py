# -*- coding: utf-8 -*-
"""
LoRA Snapshot Ensemble Evaluation — Seeded variant
===================================================
Evaluates one seed's ensemble and saves results to JSON for significance aggregation.

Usage:
    python elora_evaluation_seeded.py --seed 42
    python elora_evaluation_seeded.py --seed 42 --results_dir /path/to/significance_analysis
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
import skimage.draw
from tqdm import tqdm

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from peft import PeftModel
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

    def __init__(self, coco_file: str, root_dir: str, split: str, transform=None):
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


def build_dataloader(coco_file, root_dir, split, batch_size, num_workers=4):
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
        out = preprocessor(list(images), segmentation_maps=list(seg_maps), return_tensors="pt")
        out["original_images"]            = list(orig_imgs)
        out["original_segmentation_maps"] = list(orig_segs)
        return out

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(pretrained_name, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
    )
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_adapter(base_model, adapter_path, device):
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    return peft_model.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_ensemble_inference(base_model, adapter_paths, dataloader, device, H, W):
    N = len(dataloader.dataset)
    M = len(adapter_paths)
    C = NUM_LABELS

    ensemble_probs = torch.zeros(M, N, C, H, W, dtype=torch.float32)
    gt_maps        = torch.zeros(N, H, W, dtype=torch.long)
    gt_collected   = False

    for m, adapter_path in enumerate(adapter_paths):
        print(f"\n[{m+1}/{M}] Loading adapter: {adapter_path}")
        model = load_adapter(base_model, adapter_path, device)

        sample_idx = 0
        for batch in tqdm(dataloader, desc=f"  Inference model {m+1}"):
            pixel_values = batch["pixel_values"].to(device)
            outputs      = model(pixel_values)

            class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = outputs.masks_queries_logits.sigmoid()
            seg = torch.einsum("bqc,bqhw->bchw", class_probs, masks_probs)
            seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
            seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)

            B = seg.shape[0]
            ensemble_probs[m, sample_idx:sample_idx + B] = seg.cpu()

            if not gt_collected:
                for b, gt in enumerate(batch["original_segmentation_maps"]):
                    gt_maps[sample_idx + b] = torch.from_numpy(gt).long()
            sample_idx += B

        gt_collected = True
        del model
        torch.cuda.empty_cache()

    return ensemble_probs, gt_maps


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def compute_uncertainty(ensemble_probs, weights=None):
    M  = ensemble_probs.shape[0]
    if weights is None:
        weights = torch.full((M,), 1.0 / M)
    w = weights.view(M, 1, 1, 1, 1)

    mean_probs        = (w * ensemble_probs).sum(dim=0)
    per_model_entropy = -(ensemble_probs * torch.log2(ensemble_probs + 1e-12)).sum(dim=2)
    aleatoric         = (w.squeeze(2) * per_model_entropy).sum(dim=0)
    total             = -(mean_probs * torch.log2(mean_probs + 1e-12)).sum(dim=1)

    # Foreground-only epistemic uncertainty (MI) — avoids background saturation
    # Strip class 0 (background), renormalise over foreground, then compute MI
    fg_stack  = ensemble_probs[:, :, 1:, :, :]              # (M, N, C-1, H, W)
    fg_stack  = fg_stack / (fg_stack.sum(dim=2, keepdim=True) + 1e-6)
    fg_mean   = (w * fg_stack).sum(dim=0)                   # (N, C-1, H, W)
    H_fg_mean = -(fg_mean  * torch.log2(fg_mean  + 1e-12)).sum(dim=1)  # (N,H,W)
    H_fg_per  = -(fg_stack * torch.log2(fg_stack + 1e-12)).sum(dim=2)  # (M,N,H,W)
    w_ep      = weights.view(M, 1, 1, 1)
    epistemic = H_fg_mean - (w_ep * H_fg_per).sum(dim=0)   # (N, H, W)

    variance  = (w * (ensemble_probs - mean_probs.unsqueeze(0)) ** 2).sum(dim=0).sum(dim=1)

    return {
        "mean_probs":          mean_probs,
        "aleatoric":           aleatoric,
        "total":               total,
        "epistemic":           epistemic,
        "predictive_variance": variance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_to_pixels(mean_probs, gt_maps):
    """(N, C, H, W) → (N*H*W, C)  and  (N, H, W) → (N*H*W,)."""
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
    preds  = mean_probs.argmax(dim=1)
    metric = evaluate.load("mean_iou")
    metric.add_batch(
        predictions=[preds[i].numpy() for i in range(preds.shape[0])],
        references=[gt_maps[i].numpy() for i in range(gt_maps.shape[0])],
    )
    return metric.compute(num_labels=NUM_LABELS, ignore_index=255)["mean_iou"]


def compute_ece(mean_probs, gt_maps, n_bins=10):
    _, C, _, _ = mean_probs.shape
    preds, target = _flatten_to_pixels(mean_probs, gt_maps)
    return MulticlassCalibrationError(num_classes=C, n_bins=n_bins, norm="l1")(preds, target).item()


def compute_mece(mean_probs, gt_maps, n_bins=10):
    _, C, _, _ = mean_probs.shape
    preds, target = _flatten_to_pixels(mean_probs, gt_maps)
    return MulticlassCalibrationError(num_classes=C, n_bins=n_bins, norm="max")(preds, target).item()


def compute_ace(mean_probs, gt_maps, n_bins=10):
    _, C, _, _ = mean_probs.shape
    preds, target = _flatten_to_pixels(mean_probs, gt_maps)
    return AdaptiveCalibrationError(task="multiclass", num_bins=n_bins, norm="l1", num_classes=C)(preds, target).item()


def compute_mace(mean_probs, gt_maps, n_bins=10):
    _, C, _, _ = mean_probs.shape
    preds, target = _flatten_to_pixels(mean_probs, gt_maps)
    return AdaptiveCalibrationError(task="multiclass", num_bins=n_bins, norm="max", num_classes=C)(preds, target).item()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Ensemble Evaluation — Seeded")
    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")

    # Directory containing seed_X subdirectories produced by training script
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/significance_analysis",
        help="Parent dir of seed_X subdirs. Adapters are auto-detected from here.")

    # Which seed to evaluate
    parser.add_argument("--seed", type=int, default=42)

    # Explicit adapter override (optional — if omitted, auto-detected from results_dir/seed_X)
    parser.add_argument("--adapters", nargs="+", type=str, default=None,
        help="Explicit adapter paths. If omitted, auto-detected from results_dir/seed_<seed>/model_shot_*")

    # Snapshot selection: which shots to include in the ensemble
    parser.add_argument("--shot_ids", nargs="+", type=int, default=[4, 6, 8, 10],
        help="model_shot IDs to include in ensemble (default: 4 6 8 10)")

    parser.add_argument("--split",       type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu",         type=str, default="0")
    parser.add_argument("--height",      type=int, default=1280)
    parser.add_argument("--width",       type=int, default=720)
    parser.add_argument("--n_bins",      type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    # Deterministic eval
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Resolve adapter paths ─────────────────────────────────────────────────
    if args.adapters is not None:
        adapter_paths = args.adapters
    else:
        seed_dir = os.path.join(args.results_dir, f"seed_{args.seed}")
        adapter_paths = [
            os.path.join(seed_dir, f"model_shot_{shot}")
            for shot in args.shot_ids
        ]
        missing = [p for p in adapter_paths if not os.path.isdir(p)]
        if missing:
            raise FileNotFoundError(
                f"[Seed {args.seed}] Missing adapter directories:\n" + "\n".join(missing)
            )

    print(f"\n{'='*60}")
    print(f"  Seed  : {args.seed}")
    print(f"  Device: {device}  (GPU {args.gpu})")
    print(f"  Split : {args.split}  |  Ensemble members: {len(adapter_paths)}")
    for p in adapter_paths:
        print(f"    {p}")
    print(f"{'='*60}\n")

    dataloader = build_dataloader(
        args.coco_file, args.root_dir, args.split,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Dataset size: {len(dataloader.dataset)} | Batches: {len(dataloader)}")

    base_model = load_base_model(args.pretrained, device)

    t0 = time.time()
    ensemble_probs, gt_maps = run_ensemble_inference(
        base_model, adapter_paths, dataloader, device,
        H=args.height, W=args.width,
    )
    inference_time = (time.time() - t0) / 60
    print(f"\nInference time: {inference_time:.1f} min")

    print("\nComputing uncertainty maps...")
    uncertainty = compute_uncertainty(ensemble_probs)
    mean_probs  = uncertainty["mean_probs"]

    print("\nComputing metrics...")
    miou = compute_miou(mean_probs, gt_maps)
    ece  = compute_ece( mean_probs, gt_maps, n_bins=args.n_bins)
    mece = compute_mece(mean_probs, gt_maps, n_bins=args.n_bins)
    ace  = compute_ace( mean_probs, gt_maps, n_bins=args.n_bins)
    mace = compute_mace(mean_probs, gt_maps, n_bins=args.n_bins)
    bna  = compute_brier_nll_acc(mean_probs, gt_maps)

    results = {
        # Primary metrics (used for mean ± std aggregation)
        "mIoU":               miou,
        "ECE":                ece,
        "MECE":               mece,
        "ACE":                ace,
        "MACE":               mace,
        "Brier":              bna["Brier"],
        "NLL":                bna["NLL"],
        "acc":                bna["acc"],
        "mean_aleatoric":     float(uncertainty["aleatoric"].mean()),
        "mean_epistemic":     float(uncertainty["epistemic"].mean()),
        "mean_total":         float(uncertainty["total"].mean()),
        "mean_pred_variance": float(uncertainty["predictive_variance"].mean()),
        # Metadata
        "seed":               args.seed,
        "split":              args.split,
        "ensemble_members":   len(adapter_paths),
        "test_samples":       len(dataloader.dataset),
        "inference_min":      round(inference_time, 2),
        "adapter_paths":      adapter_paths,
    }

    # Pretty-print
    print("\n" + "=" * 55)
    print(f"  Seed {args.seed} — Evaluation Results")
    print("=" * 55)
    metric_keys = ["mIoU", "ECE", "MECE", "ACE", "MACE", "Brier", "NLL", "acc",
                   "mean_aleatoric", "mean_epistemic", "mean_total", "mean_pred_variance"]
    for k in metric_keys:
        print(f"  {k:<30} {results[k]:.4f}")
    print("=" * 55 + "\n")

    # Save JSON alongside the seed directory so the aggregation script can find it
    out_dir  = os.path.join(args.results_dir, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] Results → {out_path}")

    return results


if __name__ == "__main__":
    main()