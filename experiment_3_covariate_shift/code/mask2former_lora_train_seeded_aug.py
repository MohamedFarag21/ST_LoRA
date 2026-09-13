# -*- coding: utf-8 -*-
"""
Mask2Former + LoRA (DoRA) Fine-tuning for Sweet Pepper Segmentation
Seeded variant — supports --seed CLI arg for significance analysis.

Usage:
    python mask2former_lora_train_seeded.py --seed 42
    python mask2former_lora_train_seeded.py --seed 42 --gpus 4
"""

import os
import time
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import json
from PIL import Image
import skimage.draw
from torchvision import transforms
import torchvision.transforms.v2.functional as TF
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF

from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)
from peft import LoraConfig, get_peft_model
from torchmetrics.segmentation import MeanIoU

# ─────────────────────────────────────────────────────────────────────────────
# Global Config
# NOTE: pl.seed_everything is called inside main() using args.seed
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
LABEL2ID = {old: new for new, old in enumerate(sorted(ID2LABEL_ORIG.keys()))}
ID2LABEL = {new: ID2LABEL_ORIG[old] for old, new in LABEL2ID.items()}
_REMAP_LUT = np.zeros(256, dtype=np.int64)
for old, new in LABEL2ID.items():
    _REMAP_LUT[old] = new

NUM_LABELS = len(ID2LABEL)  # 8


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class COCODataset(Dataset):
    SPLIT_IDS = {
        "train": list(range(283, 345)) + list(range(408, 471)),
        "valid": list(range(345, 377)) + list(range(533, 564)),
        "test":  list(range(377, 408)) + list(range(471, 533)),
    }

    def __init__(self, coco_file: str, root_dir: str, split: str = None):
        with open(coco_file) as f:
            data = json.load(f)
        self.root_dir = root_dir
        images = data["images"]
        if split and split in self.SPLIT_IDS:
            valid_ids = set(self.SPLIT_IDS[split])
            images = [img for img in images if img["id"] in valid_ids]
        self.images = images
        self.ann_lookup: dict = {}
        for ann in data["annotations"]:
            self.ann_lookup.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info     = self.images[idx]
        image_id = info["id"]
        H, W     = info["height"], info["width"]
        rel_path = info["path"].lstrip("/datasets/")
        img      = Image.open(os.path.join(self.root_dir, rel_path)).convert("RGB")

        sem_map = np.zeros((H, W), dtype=np.uint8)
        for ann in self.ann_lookup.get(image_id, []):
            for poly in ann.get("segmentation", []):
                pts = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(pts[:, 1], pts[:, 0], sem_map.shape)
                sem_map[rr, cc] = ann["category_id"]

        return img, sem_map, image_id


# ─────────────────────────────────────────────────────────────────────────────
# Joint augmentation
# ─────────────────────────────────────────────────────────────────────────────

class JointAugmentation:
    """
    Geometric (applied to BOTH image and mask):
      - Horizontal flip   : robot traversal direction
      - Translation       : lateral wobble of the robot platform

    Photometric (applied to IMAGE only — labels unaffected):
      - Color jitter      : brightness, contrast, saturation
      - Gaussian blur     : motion blur / focus drift
      - Gaussian noise    : sensor noise
    """

    def __init__(
        self,
        flip_p:          float = 0.5,
        translate_p:     float = 0.5,
        translate_range: float = 0.10,
        brightness:      float = 0.3,
        contrast:        float = 0.3,
        saturation:      float = 0.1,
        blur_p:          float = 0.3,
        blur_sigma:      tuple = (0.5, 1.5),
        noise_p:         float = 0.2,
        noise_std:       float = 0.02,
        local_blur_p:    float = 0.3,    # non-uniform depth-of-field blur
        erase_p:         float = 0.3,    # random erasing (image only)
        erase_scale:     tuple = (0.02, 0.15),  # patch area as fraction of image
        erase_ratio:     tuple = (0.3, 3.3),    # patch aspect ratio range
        erase_n:         int   = 3,      # max number of erased patches per image
    ):
        self.flip_p          = flip_p
        self.translate_p     = translate_p
        self.translate_range = translate_range
        self.color_jitter    = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation
        )
        self.blur_p      = blur_p
        self.blur_sigma  = blur_sigma
        self.noise_p     = noise_p
        self.noise_std   = noise_std
        self.local_blur_p  = local_blur_p
        self.erase_p       = erase_p
        self.erase_scale   = erase_scale
        self.erase_ratio   = erase_ratio
        self.erase_n       = erase_n

    def __call__(self, pil_img: Image.Image, seg_np: np.ndarray):
        W, H = pil_img.size

        # ── Horizontal flip ───────────────────────────────────────────────────
        if torch.rand(1).item() < self.flip_p:
            pil_img = TF.horizontal_flip(pil_img)
            seg_np  = np.fliplr(seg_np).copy()

        # ── Translation (left / right) ────────────────────────────────────────
        # Image : BILINEAR interpolation (smooth boundary)
        # Mask  : NEAREST interpolation (no label blending — integer labels must
        #         remain integers; bilinear would create fractional label values)
        # Both use the same shift value so they stay perfectly aligned.
        if torch.rand(1).item() < self.translate_p:
            max_px = int(self.translate_range * W)
            shift  = torch.randint(-max_px, max_px + 1, (1,)).item()
            if shift != 0:
                # PIL affine transform: (a,b,c,d,e,f) maps output←input
                # For a pure horizontal shift of `shift` pixels:
                #   output_x = input_x - shift  (shift>0 → content moves right)
                affine = (1, 0, -shift, 0, 1, 0)

                pil_img = pil_img.transform(
                    pil_img.size, Image.AFFINE, affine,
                    resample=Image.BILINEAR, fillcolor=0,
                )
                # Mask: integer mode, NEAREST to avoid label blending
                seg_pil = Image.fromarray(seg_np.astype(np.int32), mode="I")
                seg_pil = seg_pil.transform(
                    seg_pil.size, Image.AFFINE, affine,
                    resample=Image.NEAREST, fillcolor=0,   # 0 = background
                )
                seg_np = np.array(seg_pil, dtype=np.int64)

        # ── Colour jitter (image only) ────────────────────────────────────────
        pil_img = self.color_jitter(pil_img)

        # ── Gaussian blur (image only) ────────────────────────────────────────
        if torch.rand(1).item() < self.blur_p:
            sigma = self.blur_sigma[0] + torch.rand(1).item() * (
                    self.blur_sigma[1] - self.blur_sigma[0])
            k = int(2 * round(2 * sigma) + 1)
            k = k if k % 2 == 1 else k + 1
            pil_img = TF.gaussian_blur(pil_img, kernel_size=k, sigma=sigma)

        # ── Gaussian noise (image only) ───────────────────────────────────────
        if torch.rand(1).item() < self.noise_p:
            img_t   = transforms.ToTensor()(pil_img)
            img_t   = (img_t + torch.randn_like(img_t) * self.noise_std).clamp(0, 1)
            pil_img = transforms.ToPILImage()(img_t)

        # ── Local non-uniform blur (image only) ───────────────────────────────
        # Simulates depth-of-field: centre is sharper, edges are softer.
        # A Gaussian weight mask blends the sharp original with a blurred version,
        # so blur strength increases smoothly toward image edges.
        if torch.rand(1).item() < self.local_blur_p:
            sigma_edge = self.blur_sigma[0] + torch.rand(1).item() * (
                         self.blur_sigma[1] - self.blur_sigma[0])
            k = int(2 * round(2 * sigma_edge) + 1)
            k = k if k % 2 == 1 else k + 1

            img_arr    = np.array(pil_img).astype(np.float32)
            blurred    = np.array(TF.gaussian_blur(pil_img, kernel_size=k,
                                                    sigma=sigma_edge)).astype(np.float32)

            # Build a spatial weight map: 1 at centre (keep sharp), 0 at edges (use blurred)
            h_arr, w_arr = img_arr.shape[:2]
            ys = np.linspace(-1, 1, h_arr)
            xs = np.linspace(-1, 1, w_arr)
            xx, yy  = np.meshgrid(xs, ys)
            # Gaussian fall-off from centre; sigma_spatial controls how quickly
            # blur increases toward the edges (0.7 keeps ~50% area sharp)
            sigma_s = 0.7
            weight  = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_s ** 2))
            weight  = weight[:, :, np.newaxis]   # (H, W, 1) for broadcasting

            mixed   = weight * img_arr + (1 - weight) * blurred
            pil_img = Image.fromarray(mixed.clip(0, 255).astype(np.uint8))

        # ── Random erasing (image only — mask labels unchanged) ───────────────
        # Simulates occlusion by leaves, support wires, or other plant canopy.
        # Multiple small patches are erased per image; the mask is NOT modified
        # so the model must learn to handle hidden/occluded pepper instances.
        if torch.rand(1).item() < self.erase_p:
            img_t   = transforms.ToTensor()(pil_img)   # (3, H, W)
            _, H_t, W_t = img_t.shape
            n_patches = torch.randint(1, self.erase_n + 1, (1,)).item()
            for _ in range(n_patches):
                # Sample patch area and aspect ratio
                area      = H_t * W_t
                patch_area = area * (
                    self.erase_scale[0] + torch.rand(1).item() *
                    (self.erase_scale[1] - self.erase_scale[0])
                )
                log_ratio  = torch.rand(1).item() * (
                    np.log(self.erase_ratio[1]) - np.log(self.erase_ratio[0])
                ) + np.log(self.erase_ratio[0])
                ratio      = np.exp(log_ratio)
                ph = int(round(np.sqrt(patch_area / ratio)))
                pw = int(round(np.sqrt(patch_area * ratio)))
                ph = min(ph, H_t)
                pw = min(pw, W_t)
                y0 = torch.randint(0, H_t - ph + 1, (1,)).item()
                x0 = torch.randint(0, W_t - pw + 1, (1,)).item()
                # Fill with channel-wise mean (neutral, avoids colour bias)
                fill = img_t.mean(dim=(1, 2), keepdim=True).expand(3, ph, pw)
                img_t[:, y0:y0 + ph, x0:x0 + pw] = fill
            pil_img = transforms.ToPILImage()(img_t.clamp(0, 1))

        return pil_img, seg_np



class SegmentationDataset(Dataset):
    """
    Wraps COCODataset with:
      - JointAugmentation  (geometric + photometric, train only)
      - CutMix             (p=0.5, train only — requires use_cutmix=True)
    """

    def __init__(self, base_dataset: COCODataset, image_transform=None,
                 joint_augmentation=None, use_cutmix: bool = False,
                 cutmix_p: float = 0.5):
        self.base               = base_dataset
        self.image_transform    = image_transform
        self.joint_augmentation = joint_augmentation
        self.use_cutmix         = use_cutmix
        self.cutmix_p           = cutmix_p

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _cutmix(img_a, seg_a, img_b, seg_b):
        """
        Paste a random rectangle from (img_b, seg_b) into (img_a, seg_a).

        The rectangle size is sampled from Beta(1,1) ~ Uniform[0,1] so the
        cut covers between 0% and 100% of the image area, with expected 25%
        (since both width and height are sampled independently from √Uniform).

        Image boundary: filled with bilinear-blended pixels from img_b.
        Mask  boundary: hard copy from seg_b — NO label interpolation.

        Returns: (mixed_img_arr, mixed_seg_arr)
        """
        H, W = img_a.shape[:2]

        # Sample cut size (lam → fraction of area kept from A)
        lam = np.random.beta(1.0, 1.0)
        cut_w = int(W * np.sqrt(1.0 - lam))
        cut_h = int(H * np.sqrt(1.0 - lam))

        # Random centre
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        mixed_img = img_a.copy()
        mixed_seg = seg_a.copy()

        mixed_img[y1:y2, x1:x2] = img_b[y1:y2, x1:x2]
        mixed_seg[y1:y2, x1:x2] = seg_b[y1:y2, x1:x2]   # NEAREST — no blending

        return mixed_img, mixed_seg

    def __getitem__(self, idx):
        pil_img, sem_np, _ = self.base[idx]

        # ── Joint augmentation ────────────────────────────────────────────────
        if self.joint_augmentation is not None:
            pil_img, sem_np = self.joint_augmentation(pil_img, sem_np)

        # ── CutMix ────────────────────────────────────────────────────────────
        if self.use_cutmix and torch.rand(1).item() < self.cutmix_p:
            idx_b   = torch.randint(len(self.base), (1,)).item()
            pil_b, sem_b, _ = self.base[idx_b]

            # Apply the same joint augmentation to the second image too
            if self.joint_augmentation is not None:
                pil_b, sem_b = self.joint_augmentation(pil_b, sem_b)

            img_a = np.array(pil_img)
            img_b = np.array(pil_b)

            # Resize B to match A if needed (should be same size in this dataset)
            if img_b.shape[:2] != img_a.shape[:2]:
                pil_b = pil_b.resize((pil_img.width, pil_img.height), Image.BILINEAR)
                sem_b = np.array(
                    Image.fromarray(sem_b.astype(np.int32), mode="I").resize(
                        (pil_img.width, pil_img.height), Image.NEAREST
                    ), dtype=np.int64
                )
                img_b = np.array(pil_b)

            mixed_img, sem_np = self._cutmix(img_a, sem_np, img_b, sem_b)
            pil_img = Image.fromarray(mixed_img.astype(np.uint8))

        image   = self.image_transform(pil_img) if self.image_transform \
                  else transforms.ToTensor()(pil_img)
        seg_map = torch.from_numpy(_REMAP_LUT[sem_np.astype(np.int64)]).long()
        return image, seg_map, np.array(pil_img), seg_map.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Data Module
# ─────────────────────────────────────────────────────────────────────────────

class SweetPepperDataModule(pl.LightningDataModule):
    def __init__(self, coco_file: str, root_dir: str, batch_size: int = 4,
                 num_workers: int = 4, use_augmentation: bool = True):
        super().__init__()
        self.coco_file        = coco_file
        self.root_dir         = root_dir
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.use_augmentation = use_augmentation

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ])
        self.preprocessor = Mask2FormerImageProcessor(
            ignore_index=255, reduce_labels=False,
            do_resize=False, do_rescale=False, do_normalize=False,
            num_labels=NUM_LABELS,
        )
        # Augmentation applied only to training set
        self.joint_aug = JointAugmentation() if use_augmentation else None

    def setup(self, stage=None):
        self.train_ds = SegmentationDataset(
            COCODataset(self.coco_file, self.root_dir, split="train"),
            image_transform=self.train_transform,
            joint_augmentation=self.joint_aug,
            use_cutmix=self.use_augmentation,   # CutMix on iff augmentation on
            cutmix_p=0.5,
        )
        self.val_ds = SegmentationDataset(
            COCODataset(self.coco_file, self.root_dir, split="valid"),
            image_transform=self.test_transform,
        )
        self.test_ds = SegmentationDataset(
            COCODataset(self.coco_file, self.root_dir, split="test"),
            image_transform=self.test_transform,
        )

    def _collate(self, batch):
        images, seg_maps, orig_imgs, orig_segs = zip(*batch)
        processed = self.preprocessor(
            list(images), segmentation_maps=list(seg_maps), return_tensors="pt",
        )
        processed["original_images"]            = list(orig_imgs)
        processed["original_segmentation_maps"] = list(orig_segs)
        return processed

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self._collate, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self._collate, num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self._collate, num_workers=self.num_workers,
                          persistent_workers=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerLoRAModule(pl.LightningModule):
    """Mask2Former fine-tuned with LoRA/DoRA. Uses torchmetrics.MeanIoU (DDP-safe)."""

    def __init__(
        self,
        pretrained_name: str = "facebook/mask2former-swin-base-ade-semantic",
        lora_r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        use_dora: bool = False,
        lora_bias: str = "none",
        target_modules: list = None,
        fullft_modules: list = None,
        lr: float = 2e-4,
        lr_min: float = 2e-6,
        cosine_T0: int = 20,
        constant_lr: bool = False,
        save_dir: str = "./checkpoints",
    ):
        super().__init__()
        self.save_hyperparameters()

        base_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
        )

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "class_predictor", "dense"]

        print(f"\n[LoRA] Target modules : {target_modules}")
        print(f"[LoRA] Bias           : {lora_bias}")

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout, bias=lora_bias, use_dora=use_dora,
        )
        self.model = get_peft_model(base_model, lora_cfg)
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora" in name
        # structural ablation: fully fine-tune named containers (e.g. the transformer decoder)
        # on top of the LoRA-adapted encoder — unfreeze their params after the LoRA freeze pass.
        if fullft_modules:
            n_unfrozen = 0
            for name, param in self.model.named_parameters():
                if any(fm in name for fm in fullft_modules):
                    param.requires_grad = True
                    n_unfrozen += 1
            print(f"[fullft] unfroze {n_unfrozen} params in {fullft_modules}")
        self._print_trainable_params()

        self.preprocessor = Mask2FormerImageProcessor(
            ignore_index=255, reduce_labels=False,
            do_resize=False, do_rescale=False, do_normalize=False,
            num_labels=NUM_LABELS,
        )

        # DDP-safe metric: torchmetrics handles distributed sync automatically
        self.val_miou  = MeanIoU(num_classes=NUM_LABELS, per_class=False)
        self.test_miou = MeanIoU(num_classes=NUM_LABELS, per_class=False)

    def _print_trainable_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"\n[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)\n")

    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        return self.model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[l.to(self.device) for l in batch["mask_labels"]],
            class_labels=[l.to(self.device) for l in batch["class_labels"]],
        )
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[l.to(self.device) for l in batch["mask_labels"]],
            class_labels=[l.to(self.device) for l in batch["class_labels"]],
        )
        self.log("val/loss", outputs.loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        orig_imgs = batch["original_images"]
        target_sz = [(img.shape[0], img.shape[1]) for img in orig_imgs]
        preds = self.preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sz)
        refs  = batch["original_segmentation_maps"]

        for pred, ref in zip(preds, refs):
            self.val_miou.update(pred.unsqueeze(0).to(self.device),
                                 torch.from_numpy(ref).unsqueeze(0).to(self.device))
        return outputs.loss

    def on_validation_epoch_end(self):
        miou = self.val_miou.compute()
        self.log("val/mean_iou", miou, prog_bar=True, sync_dist=True)
        self.val_miou.reset()
        if self.trainer.is_global_zero:
            print(f"\n[Epoch {self.current_epoch}] Val mIoU: {miou.item():.4f}")

    def test_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[l.to(self.device) for l in batch["mask_labels"]],
            class_labels=[l.to(self.device) for l in batch["class_labels"]],
        )
        orig_imgs = batch["original_images"]
        target_sz = [(img.shape[0], img.shape[1]) for img in orig_imgs]
        preds = self.preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sz)
        refs  = batch["original_segmentation_maps"]

        for pred, ref in zip(preds, refs):
            self.test_miou.update(pred.unsqueeze(0).to(self.device),
                                  torch.from_numpy(ref).unsqueeze(0).to(self.device))

    def on_test_epoch_end(self):
        miou = self.test_miou.compute()
        self.log("test/mean_iou", miou, rank_zero_only=True)
        self.test_miou.reset()
        if self.trainer.is_global_zero:
            print(f"\n[Test] Mean IoU: {miou.item():.4f}")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.hparams.lr,
        )
        if self.hparams.constant_lr:
            # Ablation arm: single flat LR at self.hparams.lr, no cosine warm-restarts.
            print(f"[optim] CONSTANT LR (no scheduler) @ lr={self.hparams.lr}")
            return {"optimizer": opt}
        sched = CosineAnnealingWarmRestarts(
            opt, T_0=self.hparams.cosine_T0, T_mult=1, eta_min=self.hparams.lr_min,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1}}


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Ensemble Callback
# ─────────────────────────────────────────────────────────────────────────────

class SnapshotEnsembleCallback(pl.Callback):
    def __init__(self, save_dir: str, save_every: int = 20):
        self.save_dir   = save_dir
        self.save_every = save_every

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.save_every == 0:
            shot = epoch // self.save_every
            path = os.path.join(self.save_dir, f"model_shot_{shot}")
            pl_module.model.save_pretrained(path)
            print(f"\n[Snapshot] Saved adapter snapshot {shot} → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Mask2Former LoRA Training (Seeded)")

    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    parser.add_argument("--base_save_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/significance_analysis",
        help="Parent directory. Seed-specific subdir is created automatically.")
    parser.add_argument("--log_dir",   type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/significance_analysis/logs")

    parser.add_argument("--pretrained",    type=str, default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--lora_r",        type=int,   default=8)
    parser.add_argument("--lora_alpha",    type=int,   default=8)
    parser.add_argument("--lora_dropout",  type=float, default=0.1)
    parser.add_argument("--lora_bias",     type=str,   default="none",
        choices=["none", "all", "lora_only"],
        help="Which biases to train: none, lora_only, or all")
    parser.add_argument("--no_dora",       action="store_true")
    parser.add_argument("--target_modules", nargs="+", type=str,
        default=["dense", "q_proj", "k_proj", "class_predictor"],
        help="LoRA target module names, e.g. --target_modules q_proj k_proj value dense")
    parser.add_argument("--fullft_modules", nargs="+", type=str, default=None,
        help="module-name substrings to FULLY fine-tune on top of LoRA (e.g. transformer_module.decoder)")

    parser.add_argument("--epochs",          type=int,   default=110)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--lr_min",          type=float, default=2e-6)
    parser.add_argument("--cosine_T0",       type=int,   default=20)
    parser.add_argument("--constant_lr",     action="store_true",
        help="Ablation: single flat LR (=--lr), NO cosine scheduler / warm-restarts")
    parser.add_argument("--snapshot_every",  type=int,   default=20)

    parser.add_argument("--gpus",        type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--strategy",    type=str, default=None)
    parser.add_argument("--precision",   type=str, default="32")

    # ── Significance analysis ─────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed. Run once per seed for significance analysis.")
    parser.add_argument("--no_augmentation", action="store_true",
        help="Disable training augmentation (for ablation/reproducibility)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Seed everything inside main() — affects DataLoader workers too
    pl.seed_everything(args.seed, workers=True)

    # Each seed gets its own isolated output directory
    save_dir = os.path.join(args.base_save_dir, f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Seed: {args.seed}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*60}\n")

    devices     = args.gpus if torch.cuda.is_available() else 1
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    if args.strategy:
        strategy = args.strategy
    elif devices > 1:
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"

    data_module = SweetPepperDataModule(
        coco_file=args.coco_file, root_dir=args.root_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_augmentation=not args.no_augmentation,
    )

    model = Mask2FormerLoRAModule(
        pretrained_name=args.pretrained,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, use_dora=not args.no_dora,
        lora_bias=args.lora_bias,
        target_modules=args.target_modules,
        fullft_modules=args.fullft_modules,
        lr=args.lr, lr_min=args.lr_min, cosine_T0=args.cosine_T0,
        constant_lr=args.constant_lr,
        save_dir=save_dir,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "best"),
            filename="best-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss", mode="min", save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "best_iou"),
            filename="best-iou-{epoch:03d}-{val/mean_iou:.4f}",
            monitor="val/mean_iou", mode="max", save_top_k=3,
        ),
        SnapshotEnsembleCallback(save_dir=save_dir, save_every=args.snapshot_every),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = TensorBoardLogger(
        save_dir=args.log_dir, name=f"mask2former_lora_seed{args.seed}"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=devices, accelerator=accelerator, strategy=strategy,
        precision=args.precision,
        callbacks=callbacks, logger=logger,
        log_every_n_steps=10, enable_progress_bar=True,
        sync_batchnorm=(devices > 1),
    )

    start = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f"\nTraining complete in {(time.time()-start)/60:.1f} min")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()