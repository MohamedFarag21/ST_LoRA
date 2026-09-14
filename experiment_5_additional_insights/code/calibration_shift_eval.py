# -*- coding: utf-8 -*-
"""
Calibration Under Distribution Shift — All Four Methods
=========================================================
Evaluates ECE, ACE, and mIoU for each method under 7 corruption types
× 5 severity levels designed to exceed the training augmentation range.

Corruption severity relative to training augmentation:
  blur       : σ ∈ {1.5, 3, 5, 7, 10}       training: σ ∈ [0.5, 1.5]
  noise      : std ∈ {0.04, 0.08, 0.12, 0.18, 0.25}  training: 0.02
  brightness : factor ∈ {0.6, 0.4, 0.25, 0.10, 0.05} training: [0.7, 1.3]
  contrast   : factor ∈ {0.6, 0.4, 0.25, 0.10, 0.05} training: [0.7, 1.3]
  saturation : factor ∈ {0.7, 0.5, 0.3, 0.1, 0.0}   training: [0.9, 1.1]
  translation: shift ∈ {20, 30, 40, 50, 60}% width   training: ±10%
  rotation   : {vflip, 90°, 180°, 270°, 90°+vflip}   training: hflip only

All four methods produce mean softmax probabilities — ECE/ACE computed
identically across methods for fair comparison.

Usage:
    python calibration_shift_eval.py --method lora   --seed 42
    python calibration_shift_eval.py --method fullft --seed 42
    python calibration_shift_eval.py --method mcdropout --seed 42
    python calibration_shift_eval.py --method ddu    --seed 42
"""

import os
import json
import time
import random
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance
import skimage.draw
from tqdm import tqdm

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from peft import PeftModel
from torchmetrics.classification import MulticlassCalibrationError
from torch_uncertainty.metrics.classification.adaptive_calibration_error import AdaptiveCalibrationError
import evaluate as hf_evaluate

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

# ── Corruption grid ───────────────────────────────────────────────────────────
# Each entry: (severity_label, severity_param)
CORRUPTIONS = {
    "blur": [
        ("s1", 1.5), ("s2", 3.0), ("s3", 5.0), ("s4", 7.0), ("s5", 10.0)
    ],
    "noise": [
        ("s1", 0.04), ("s2", 0.08), ("s3", 0.12), ("s4", 0.18), ("s5", 0.25)
    ],
    "brightness": [
        ("s1", 0.6), ("s2", 0.4), ("s3", 0.25), ("s4", 0.10), ("s5", 0.05)
    ],
    "contrast": [
        ("s1", 0.6), ("s2", 0.4), ("s3", 0.25), ("s4", 0.10), ("s5", 0.05)
    ],
    "saturation": [
        ("s1", 0.8), ("s2", 0.6), ("s3", 0.4), ("s4", 0.15), ("s5", 0.0)
    ],
    "translation": [
        ("s1", 0.20), ("s2", 0.30), ("s3", 0.40), ("s4", 0.50), ("s5", 0.60)
    ],
    "rotation": [
        ("s1", "vflip"), ("s2", "rot90"), ("s3", "rot180"),
        ("s4", "rot270"), ("s5", "rot90_vflip"),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Corruption functions (applied to PIL images before normalization)
# ─────────────────────────────────────────────────────────────────────────────

def corrupt_blur(pil_img: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur with given sigma — beyond training σ ∈ [0.5, 1.5]."""
    import torchvision.transforms.v2.functional as TF
    k = int(2 * round(2 * sigma) + 1)
    k = k if k % 2 == 1 else k + 1
    return TF.gaussian_blur(pil_img, kernel_size=k, sigma=sigma)


def corrupt_noise(pil_img: Image.Image, std: float) -> Image.Image:
    """Additive Gaussian noise — beyond training std=0.02."""
    img_t = transforms.ToTensor()(pil_img)
    img_t = (img_t + torch.randn_like(img_t) * std).clamp(0, 1)
    return transforms.ToPILImage()(img_t)


def corrupt_brightness(pil_img: Image.Image, factor: float) -> Image.Image:
    """PIL brightness factor: 0=black, 1=original — below training min 0.7."""
    return ImageEnhance.Brightness(pil_img).enhance(factor)


def corrupt_contrast(pil_img: Image.Image, factor: float) -> Image.Image:
    """PIL contrast factor: 0=grey, 1=original — below training min 0.7."""
    return ImageEnhance.Contrast(pil_img).enhance(factor)


def corrupt_saturation(pil_img: Image.Image, factor: float) -> Image.Image:
    """PIL saturation factor: 0=greyscale, 1=original — outside training [0.9, 1.1]."""
    return ImageEnhance.Color(pil_img).enhance(factor)


def corrupt_translation(pil_img: Image.Image, frac: float) -> Image.Image:
    """Horizontal shift by frac × width — beyond training ±10%."""
    W, H  = pil_img.size
    shift = int(frac * W)
    affine = (1, 0, shift, 0, 1, 0)   # shift right by `shift` pixels
    return pil_img.transform(
        pil_img.size, Image.AFFINE, affine,
        resample=Image.BILINEAR, fillcolor=0,
    )


def corrupt_rotation(pil_img: Image.Image, mode: str) -> Image.Image:
    """Geometric transforms beyond horizontal flip used in training."""
    if mode == "vflip":
        return pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    elif mode == "rot90":
        return pil_img.transpose(Image.ROTATE_90)
    elif mode == "rot180":
        return pil_img.transpose(Image.ROTATE_180)
    elif mode == "rot270":
        return pil_img.transpose(Image.ROTATE_270)
    elif mode == "rot90_vflip":
        return pil_img.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
    return pil_img


CORRUPT_FNS = {
    "blur":        corrupt_blur,
    "noise":       corrupt_noise,
    "brightness":  corrupt_brightness,
    "contrast":    corrupt_contrast,
    "saturation":  corrupt_saturation,
    "translation": corrupt_translation,
    "rotation":    corrupt_rotation,
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset with on-the-fly corruption
# ─────────────────────────────────────────────────────────────────────────────

class CorruptedTestDataset(Dataset):
    TEST_IDS = list(range(377, 408)) + list(range(471, 533))

    def __init__(self, coco_file, root_dir, corrupt_fn=None, corrupt_param=None):
        with open(coco_file) as f:
            data = json.load(f)
        valid_ids      = set(self.TEST_IDS)
        self.images    = [img for img in data["images"] if img["id"] in valid_ids]
        self.root_dir  = root_dir
        self.corrupt_fn    = corrupt_fn
        self.corrupt_param = corrupt_param
        self.ann_lookup: dict = {}
        for ann in data["annotations"]:
            self.ann_lookup.setdefault(ann["image_id"], []).append(ann)

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info    = self.images[idx]
        H, W    = info["height"], info["width"]
        rel     = info["path"].lstrip("/datasets/")
        pil_img = Image.open(os.path.join(self.root_dir, rel)).convert("RGB")

        # Apply corruption to raw PIL image before normalization
        if self.corrupt_fn is not None:
            pil_img = self.corrupt_fn(pil_img, self.corrupt_param)
            # Rotations can swap H/W — always resize back to original dims
            if pil_img.size != (W, H):
                pil_img = pil_img.resize((W, H), Image.BILINEAR)

        sem_map = np.zeros((H, W), dtype=np.uint8)
        for ann in self.ann_lookup.get(info["id"], []):
            for poly in ann.get("segmentation", []):
                pts = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(pts[:, 1], pts[:, 0], sem_map.shape)
                sem_map[rr, cc] = ann["category_id"]

        seg_map = _REMAP_LUT[sem_map.astype(np.int64)]
        image   = self.normalize(pil_img)
        return image, torch.from_numpy(seg_map).long()


def build_loader(coco_file, root_dir, corrupt_fn, corrupt_param, batch_size, num_workers):
    preprocessor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=False,
        do_resize=False, do_rescale=False, do_normalize=False,
        num_labels=NUM_LABELS,
    )
    dataset = CorruptedTestDataset(coco_file, root_dir, corrupt_fn, corrupt_param)

    def collate_fn(batch):
        images, seg_maps = zip(*batch)
        out = preprocessor(list(images), segmentation_maps=list(seg_maps),
                           return_tensors="pt")
        out["seg_maps"] = torch.stack(list(seg_maps))
        return out

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(pretrained, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    return model.eval().to(device)


def load_lora_adapter(base, adapter_path, device):
    return PeftModel.from_pretrained(base, adapter_path).eval().to(device)


def load_fullft_snapshot(pretrained, ckpt_path, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval().to(device)


def load_mc_model(pretrained, ckpt_path, dropout_p, device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    orig = model.class_predictor
    model.class_predictor = nn.Sequential(nn.Dropout(p=dropout_p), orig)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device)
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()
    return model


def load_ddu_model(pretrained, ckpt_path, device):
    from torch.nn.utils.parametrizations import spectral_norm as sn_p
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    visited = set()
    def _apply_sn(mod):
        for name, child in mod.named_children():
            if id(child) in visited: continue
            visited.add(id(child))
            _apply_sn(child)
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                try: sn_p(child)
                except Exception: pass
    _apply_sn(model)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Inference — produces mean softmax probs (N, C, H, W)
# ─────────────────────────────────────────────────────────────────────────────

def build_seg_probs(outputs, H, W):
    cp  = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
    mp  = outputs.masks_queries_logits.sigmoid()
    seg = torch.einsum("bqc,bqhw->bchw", cp, mp)
    seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
    return F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)


@torch.no_grad()
def run_inference(get_model_fn, M, dataloader, device, H, W):
    """
    Run M forward passes (ensemble members or MC samples), return
    mean softmax probs (N, C, H, W) and GT maps (N, H, W).
    """
    N = len(dataloader.dataset)
    sum_probs = torch.zeros(N, NUM_LABELS, H, W)
    gt_maps   = torch.zeros(N, H, W, dtype=torch.long)
    gt_done   = False

    for m in range(M):
        model = get_model_fn(m)
        idx   = 0
        for batch in dataloader:
            pv  = batch["pixel_values"].to(device)
            seg = build_seg_probs(model(pv), H, W)
            B   = seg.shape[0]
            sum_probs[idx:idx+B] += seg.cpu()
            if not gt_done:
                gt_maps[idx:idx+B] = batch["seg_maps"]
            idx += B
        gt_done = True
        if hasattr(model, "base_model"):
            del model; torch.cuda.empty_cache()

    return sum_probs / M, gt_maps


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(probs, gt_maps, n_bins=10):
    """Returns dict with mIoU, ECE, ACE."""
    preds_hard = probs.argmax(dim=1)
    metric     = hf_evaluate.load("mean_iou")
    metric.add_batch(
        predictions=[preds_hard[i].numpy() for i in range(len(preds_hard))],
        references=[gt_maps[i].numpy()     for i in range(len(gt_maps))],
    )
    miou = metric.compute(num_labels=NUM_LABELS, ignore_index=255)["mean_iou"]

    preds_flat  = probs.permute(0, 2, 3, 1).reshape(-1, NUM_LABELS).cpu()
    target_flat = gt_maps.reshape(-1).cpu()

    ece = MulticlassCalibrationError(
        num_classes=NUM_LABELS, n_bins=n_bins, norm="l1")(
        preds_flat, target_flat).item()
    ace = AdaptiveCalibrationError(
        task="multiclass", num_bins=n_bins, norm="l1",
        num_classes=NUM_LABELS)(
        preds_flat, target_flat).item()

    return {"mIoU": round(miou, 4), "ECE": round(ece, 4), "ACE": round(ace, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibration Under Distribution Shift — All Methods")
    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir",  type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--pretrained", type=str,
        default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--out_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/calibration_shift")

    parser.add_argument("--method", type=str, required=True,
        choices=["lora", "fullft", "mcdropout", "ddu"])
    parser.add_argument("--seed", type=int, default=42)

    # LoRA
    parser.add_argument("--lora_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/hparam_sweep")
    parser.add_argument("--config_name", type=str, default="final_model")
    parser.add_argument("--shot_ids",    nargs="+", type=int, default=[1, 2, 3, 4])
    # Full FT
    parser.add_argument("--fullft_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/full_ft")
    parser.add_argument("--fullft_shot_ids", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    # MC Dropout
    parser.add_argument("--mcdrop_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/mcdropout")
    parser.add_argument("--dropout_p",  type=float, default=0.25)
    parser.add_argument("--T",          type=int,   default=10)
    # DDU
    parser.add_argument("--ddu_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/ddu")

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
    os.makedirs(args.out_dir, exist_ok=True)

    H, W = args.height, args.width
    method_names = {
        "lora": "LoRA Snapshot", "fullft": "Full FT Snapshot",
        "mcdropout": "MC Dropout", "ddu": "DDU"}
    method_name = method_names[args.method]

    print(f"\n{'='*65}")
    print(f"  Calibration Under Distribution Shift")
    print(f"  Method : {method_name}  (seed={args.seed})")
    print(f"  Grid   : {len(CORRUPTIONS)} corruption types × 5 severity levels")
    print(f"{'='*65}\n")

    # ── Build model loading function ──────────────────────────────────────────
    if args.method == "lora":
        base = load_base_model(args.pretrained, device)
        adapter_paths = [
            os.path.join(args.lora_dir, args.config_name,
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

    elif args.method == "mcdropout":
        ckpt     = os.path.join(args.mcdrop_dir, f"seed_{args.seed}", "model_final.pt")
        mc_model = load_mc_model(args.pretrained, ckpt, args.dropout_p, device)
        M = args.T
        def get_model(m): return mc_model

    else:  # ddu
        ckpt      = os.path.join(args.ddu_dir, f"seed_{args.seed}", "model_final.pt")
        ddu_model = load_ddu_model(args.pretrained, ckpt, device)
        M = 1
        def get_model(m): return ddu_model

    # ── Run grid ──────────────────────────────────────────────────────────────
    results = {}   # corruption → severity_label → metrics dict
    rows    = []   # for CSV

    n_total = sum(len(v) for v in CORRUPTIONS.values())
    done    = 0

    for corruption, severities in CORRUPTIONS.items():
        results[corruption] = {}
        corrupt_fn = CORRUPT_FNS[corruption]

        for sev_label, sev_param in severities:
            done += 1
            print(f"[{done:2d}/{n_total}] {corruption:12s} {sev_label}  (param={sev_param})")

            loader = build_loader(
                args.coco_file, args.root_dir,
                corrupt_fn, sev_param,
                args.batch_size, args.num_workers,
            )

            t0 = time.time()
            probs, gt_maps = run_inference(get_model, M, loader, device, H, W)
            metrics = compute_metrics(probs, gt_maps, args.n_bins)
            elapsed = time.time() - t0

            results[corruption][sev_label] = metrics
            print(f"           mIoU={metrics['mIoU']:.4f}  "
                  f"ECE={metrics['ECE']:.4f}  ACE={metrics['ACE']:.4f}  "
                  f"({elapsed:.0f}s)")

            rows.append({
                "method":      method_name,
                "seed":        args.seed,
                "corruption":  corruption,
                "severity":    sev_label,
                "sev_param":   str(sev_param),
                "mIoU":        metrics["mIoU"],
                "ECE":         metrics["ECE"],
                "ACE":         metrics["ACE"],
            })

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {method_name} — Calibration Shift Summary")
    print(f"{'='*65}")
    print(f"  {'Corruption':<14} {'S1':>7} {'S2':>7} {'S3':>7} {'S4':>7} {'S5':>7}")
    print(f"  {'─'*50}")
    for metric in ["mIoU", "ECE", "ACE"]:
        print(f"\n  [{metric}]")
        for corruption, sevs in results.items():
            vals = "  ".join(f"{sevs[s][metric]:>7.4f}"
                             for s in ["s1", "s2", "s3", "s4", "s5"])
            print(f"  {corruption:<14} {vals}")
    print(f"{'='*65}")

    # ── Save results ──────────────────────────────────────────────────────────
    prefix = os.path.join(args.out_dir, f"{args.method}_seed{args.seed}")

    out_json = prefix + "_calibration_shift.json"
    with open(out_json, "w") as f:
        json.dump({
            "method": method_name, "seed": args.seed,
            "corruptions": CORRUPTIONS,
            "results": results,
        }, f, indent=2)
    print(f"\n[Saved] {out_json}")

    out_csv = prefix + "_calibration_shift.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {out_csv}")


if __name__ == "__main__":
    main()
