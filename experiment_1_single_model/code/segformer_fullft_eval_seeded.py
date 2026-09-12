# -*- coding: utf-8 -*-
"""
SegFormer Full FT Snapshot Ensemble Evaluation
===============================================
Loads model_shot_N.pt state_dicts, runs ensemble inference.
Mirrors segformer_lora_eval_seeded.py but loads plain state_dicts.

Usage:
    python segformer_fullft_eval_seeded.py --seed 42
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
from torchmetrics.classification import MulticlassCalibrationError
from torch_uncertainty.metrics.classification.adaptive_calibration_error import AdaptiveCalibrationError
from torchmetrics.segmentation import MeanIoU

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255

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

SPLIT_IDS = {
    "train": list(range(283, 345)) + list(range(408, 471)),
    "valid": list(range(345, 377)) + list(range(533, 564)),
    "test":  list(range(377, 408)) + list(range(471, 533)),
}

CLASS_COLORS = np.array([
    [0,0,0],[0,0,255],[199,33,28],[255,247,0],
    [0,255,0],[225,0,255],[255,102,0],[209,196,21]], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class COCOSegDataset(Dataset):
    def __init__(self, coco_file, root_dir, split, transform=None):
        with open(coco_file) as f:
            data = json.load(f)
        self.root_dir = root_dir; self.transform = transform
        valid_ids     = set(SPLIT_IDS[split])
        self.images   = [img for img in data["images"] if img["id"] in valid_ids]
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
        transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)])
    dataset = COCOSegDataset(coco_file, root_dir, split, transform)

    def collate(batch):
        images, seg_maps, orig_imgs, orig_segs = zip(*batch)
        return (torch.stack(images), torch.stack(seg_maps),
                list(orig_imgs), list(orig_segs))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — state_dict (no PEFT)
# ─────────────────────────────────────────────────────────────────────────────

def load_snapshot(pretrained_name, ckpt_path, device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_name, id2label=ID2LABEL, label2id=LABEL2ID,
        num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_ensemble_inference(pretrained_name, ckpt_paths, dataloader, device):
    N = len(dataloader.dataset)
    M = len(ckpt_paths)
    first_imgs, _, _, _ = next(iter(dataloader))
    H, W = first_imgs.shape[2], first_imgs.shape[3]

    ensemble_probs = torch.zeros(M, N, NUM_LABELS, H, W)
    gt_maps        = torch.zeros(N, H, W, dtype=torch.long)
    orig_imgs_all  = [None] * N; orig_segs_all = [None] * N
    gt_collected   = False

    for m, ckpt_path in enumerate(ckpt_paths):
        print(f"\n[{m+1}/{M}] Loading: {ckpt_path}")
        model = load_snapshot(pretrained_name, ckpt_path, device)
        idx   = 0
        for images, seg_maps, orig_imgs, orig_segs in tqdm(
                dataloader, desc=f"  Inference {m+1}"):
            pv   = images.to(device)
            out  = model(pixel_values=pv)
            probs = F.interpolate(out.logits, size=(H,W),
                                  mode="bilinear", align_corners=False).softmax(dim=1)
            B = probs.shape[0]
            ensemble_probs[m, idx:idx+B] = probs.cpu()
            if not gt_collected:
                gt_maps[idx:idx+B] = seg_maps
                for b in range(B):
                    orig_imgs_all[idx+b] = orig_imgs[b]
                    orig_segs_all[idx+b] = orig_segs[b]
            idx += B
        gt_collected = True
        del model; torch.cuda.empty_cache()

    return ensemble_probs, gt_maps, orig_imgs_all, orig_segs_all


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _flatten(mean_probs, gt_maps):
    N, C, H, W = mean_probs.shape
    return (mean_probs.permute(0,2,3,1).reshape(-1,C).cpu(),
            gt_maps.reshape(-1).cpu())

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
    metric = MeanIoU(num_classes=NUM_LABELS, per_class=False)
    metric.update(preds, gt_maps); return float(metric.compute())

def compute_ece(mp, gt, n=10):
    p,t = _flatten(mp, gt)
    return MulticlassCalibrationError(num_classes=NUM_LABELS, n_bins=n, norm="l1")(p,t).item()

def compute_mece(mp, gt, n=10):
    p,t = _flatten(mp, gt)
    return MulticlassCalibrationError(num_classes=NUM_LABELS, n_bins=n, norm="max")(p,t).item()

def compute_ace(mp, gt, n=10):
    p,t = _flatten(mp, gt)
    return AdaptiveCalibrationError(task="multiclass", num_bins=n, norm="l1",
                                    num_classes=NUM_LABELS)(p,t).item()

def compute_mace(mp, gt, n=10):
    p,t = _flatten(mp, gt)
    return AdaptiveCalibrationError(task="multiclass", num_bins=n, norm="max",
                                    num_classes=NUM_LABELS)(p,t).item()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def label_to_color(lm):
    out = np.zeros((*lm.shape, 3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS): out[lm == c] = col
    return out


def save_visualization(orig_img, gt_seg, pred_seg, out_path, seed, img_idx=0):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig_img);           axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[1].imshow(label_to_color(gt_seg));   axes[1].set_title("Ground Truth",    fontsize=13, fontweight="bold")
    axes[2].imshow(label_to_color(pred_seg)); axes[2].set_title("Model Prediction", fontsize=13, fontweight="bold")
    for ax in axes: ax.axis("off")
    patches = [mpatches.Patch(color=CLASS_COLORS[i]/255., label=ID2LABEL[i])
               for i in range(NUM_LABELS)]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_LABELS,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"SegFormer Full FT — seed={seed}  image={img_idx}",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Saved] Visualisation → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    p.add_argument("--root_dir",  type=str, default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    p.add_argument("--pretrained", type=str,
        default="nvidia/segformer-b2-finetuned-ade-512-512")
    p.add_argument("--results_dir", type=str,
        default="   /lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/segformer_fullft")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--shot_ids", nargs="+", type=int, default=[2,3,4,5])
    p.add_argument("--split",       type=str, default="test")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu",         type=str, default="0")
    p.add_argument("--n_bins",      type=int, default=10)
    p.add_argument("--vis_idx",     type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_dir  = os.path.join(args.results_dir, f"seed_{args.seed}")
    ckpt_paths = [os.path.join(seed_dir, f"model_shot_{s}.pt") for s in args.shot_ids]
    missing   = [p for p in ckpt_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))

    print(f"\n{'='*60}")
    print(f"  SegFormer Full FT Eval  seed={args.seed}  split={args.split}")
    print(f"  Ensemble: {len(ckpt_paths)} snapshots")
    print(f"{'='*60}\n")

    dataloader = build_dataloader(args.coco_file, args.root_dir, args.split,
                                   args.batch_size, args.num_workers)
    print(f"Dataset: {len(dataloader.dataset)} images")

    t0 = time.time()
    ensemble_probs, gt_maps, orig_imgs, orig_segs = run_ensemble_inference(
        args.pretrained, ckpt_paths, dataloader, device)
    elapsed = (time.time()-t0)/60
    print(f"Inference time: {elapsed:.1f} min")

    mean_probs = ensemble_probs.mean(dim=0)
    mean_ent   = float(-(mean_probs*torch.log2(mean_probs+1e-12)).sum(dim=1).mean())

    print("Computing metrics...")
    miou = compute_miou(mean_probs, gt_maps)
    ece  = compute_ece(mean_probs, gt_maps, args.n_bins)
    mece = compute_mece(mean_probs, gt_maps, args.n_bins)
    ace  = compute_ace(mean_probs, gt_maps, args.n_bins)
    mace = compute_mace(mean_probs, gt_maps, args.n_bins)
    bna  = compute_brier_nll_acc(mean_probs, gt_maps)

    results = {"mIoU": miou, "ECE": ece, "MECE": mece, "ACE": ace, "MACE": mace,
               "Brier": bna["Brier"], "NLL": bna["NLL"], "acc": bna["acc"],
               "mean_entropy": mean_ent, "seed": args.seed, "split": args.split,
               "ensemble_members": len(ckpt_paths),
               "test_samples": len(dataloader.dataset),
               "inference_min": round(elapsed, 2)}

    print("\n" + "="*55)
    for k in ["mIoU","ECE","MECE","ACE","MACE","Brier","NLL","acc","mean_entropy"]:
        print(f"  {k:<20} {results[k]:.4f}")
    print("="*55)

    out_path = os.path.join(seed_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {out_path}")

    vis_idx  = min(args.vis_idx, len(dataloader.dataset)-1)
    pred_seg = mean_probs[vis_idx].argmax(dim=0).numpy()
    save_visualization(orig_imgs[vis_idx], orig_segs[vis_idx], pred_seg,
                       os.path.join(seed_dir, f"vis_{args.split}_img{vis_idx}.png"),
                       seed=args.seed, img_idx=vis_idx)


if __name__ == "__main__":
    main()
