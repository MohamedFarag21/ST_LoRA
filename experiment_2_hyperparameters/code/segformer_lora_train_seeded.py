# -*- coding: utf-8 -*-
"""
SegFormer + LoRA Snapshot Ensemble — Sweet Pepper Segmentation
===============================================================
Mirrors mask2former_lora_train_seeded.py exactly. Fully self-contained.

Key differences vs Mask2Former:
  - Plain PyTorch dataloader — no HuggingFace preprocessor needed
  - Loss: CrossEntropyLoss on upsampled logits (SegFormer outputs H/4 × W/4)
  - LoRA target modules: query, key, value, dense (MixTransformer encoder)
  - decode_head fully trained via modules_to_save

Snapshot schedule:
  CosineAnnealingWarmRestarts T0=20, T_mult=1
  Save adapter at each LR trough (every snapshot_every epochs)
  5 snapshots from 110 epochs (shots 0-4)

Usage:
    python segformer_lora_train_seeded.py --seed 42
    python segformer_lora_train_seeded.py --seed 42 --config_name base_r16
"""

import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from PIL import Image
import skimage.draw
from torchvision import transforms
import torchvision.transforms.v2.functional as TF

from transformers import SegformerForSemanticSegmentation
from peft import LoraConfig, get_peft_model
from torchmetrics.segmentation import MeanIoU

# ─────────────────────────────────────────────────────────────────────────────
# Label config — identical to Mask2Former pipeline
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

DEFAULT_TARGET_MODULES  = ["query", "key", "value", "dense"]
DEFAULT_MODULES_TO_SAVE = ["decode_head"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — identical to Mask2Former pipeline
# ─────────────────────────────────────────────────────────────────────────────

class COCODataset(Dataset):
    def __init__(self, coco_file, root_dir, split=None):
        with open(coco_file) as f:
            data = json.load(f)
        self.root_dir = root_dir
        images = data["images"]
        if split and split in SPLIT_IDS:
            valid_ids = set(SPLIT_IDS[split])
            images    = [img for img in images if img["id"] in valid_ids]
        self.images = images
        self.ann_lookup: dict = {}
        for ann in data["annotations"]:
            self.ann_lookup.setdefault(ann["image_id"], []).append(ann)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        H, W = info["height"], info["width"]
        rel  = info["path"].lstrip("/datasets/")
        img  = Image.open(os.path.join(self.root_dir, rel)).convert("RGB")
        sem_map = np.zeros((H, W), dtype=np.uint8)
        for ann in self.ann_lookup.get(info["id"], []):
            for poly in ann.get("segmentation", []):
                pts = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(pts[:, 1], pts[:, 0], sem_map.shape)
                sem_map[rr, cc] = ann["category_id"]
        return img, sem_map, info["id"]


# ─────────────────────────────────────────────────────────────────────────────
# Joint augmentation — identical to Mask2Former pipeline
# ─────────────────────────────────────────────────────────────────────────────

class JointAugmentation:
    def __init__(
        self,
        flip_p: float = 0.5,
        translate_p: float = 0.5, translate_range: float = 0.10,
        brightness: float = 0.3, contrast: float = 0.3, saturation: float = 0.1,
        blur_p: float = 0.3, blur_sigma: tuple = (0.5, 1.5),
        noise_p: float = 0.2, noise_std: float = 0.02,
        local_blur_p: float = 0.3,
        erase_p: float = 0.3, erase_scale: tuple = (0.02, 0.15),
        erase_ratio: tuple = (0.3, 3.3), erase_n: int = 3,
    ):
        self.flip_p          = flip_p
        self.translate_p     = translate_p
        self.translate_range = translate_range
        self.color_jitter    = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation)
        self.blur_p       = blur_p
        self.blur_sigma   = blur_sigma
        self.noise_p      = noise_p
        self.noise_std    = noise_std
        self.local_blur_p = local_blur_p
        self.erase_p      = erase_p
        self.erase_scale  = erase_scale
        self.erase_ratio  = erase_ratio
        self.erase_n      = erase_n

    def __call__(self, pil_img, seg_np):
        W, H = pil_img.size
        if torch.rand(1).item() < self.flip_p:
            pil_img = TF.horizontal_flip(pil_img)
            seg_np  = np.fliplr(seg_np).copy()
        if torch.rand(1).item() < self.translate_p:
            max_px = int(self.translate_range * W)
            shift  = torch.randint(-max_px, max_px + 1, (1,)).item()
            if shift != 0:
                affine  = (1, 0, -shift, 0, 1, 0)
                pil_img = pil_img.transform(pil_img.size, Image.AFFINE, affine,
                                             resample=Image.BILINEAR, fillcolor=0)
                seg_pil = Image.fromarray(seg_np.astype(np.int32), mode="I")
                seg_pil = seg_pil.transform(seg_pil.size, Image.AFFINE, affine,
                                             resample=Image.NEAREST, fillcolor=0)
                seg_np  = np.array(seg_pil, dtype=np.int64)
        pil_img = self.color_jitter(pil_img)
        if torch.rand(1).item() < self.blur_p:
            sigma = self.blur_sigma[0] + torch.rand(1).item() * (
                    self.blur_sigma[1] - self.blur_sigma[0])
            k = int(2 * round(2 * sigma) + 1); k = k if k % 2 == 1 else k + 1
            pil_img = TF.gaussian_blur(pil_img, kernel_size=k, sigma=sigma)
        if torch.rand(1).item() < self.noise_p:
            img_t   = transforms.ToTensor()(pil_img)
            img_t   = (img_t + torch.randn_like(img_t) * self.noise_std).clamp(0, 1)
            pil_img = transforms.ToPILImage()(img_t)
        if torch.rand(1).item() < self.local_blur_p:
            sigma_e = self.blur_sigma[0] + torch.rand(1).item() * (
                      self.blur_sigma[1] - self.blur_sigma[0])
            k = int(2 * round(2 * sigma_e) + 1); k = k if k % 2 == 1 else k + 1
            img_arr = np.array(pil_img).astype(np.float32)
            blurred = np.array(TF.gaussian_blur(pil_img, kernel_size=k,
                                                 sigma=sigma_e)).astype(np.float32)
            h_a, w_a = img_arr.shape[:2]
            ys = np.linspace(-1, 1, h_a); xs = np.linspace(-1, 1, w_a)
            xx, yy  = np.meshgrid(xs, ys)
            weight  = np.exp(-(xx ** 2 + yy ** 2) / (2 * 0.7 ** 2))[:, :, np.newaxis]
            mixed   = weight * img_arr + (1 - weight) * blurred
            pil_img = Image.fromarray(mixed.clip(0, 255).astype(np.uint8))
        if torch.rand(1).item() < self.erase_p:
            img_t = transforms.ToTensor()(pil_img)
            _, H_t, W_t = img_t.shape
            n_patches = torch.randint(1, self.erase_n + 1, (1,)).item()
            for _ in range(n_patches):
                area       = H_t * W_t
                patch_area = area * (self.erase_scale[0] + torch.rand(1).item() *
                             (self.erase_scale[1] - self.erase_scale[0]))
                log_ratio  = torch.rand(1).item() * (
                    np.log(self.erase_ratio[1]) - np.log(self.erase_ratio[0])
                ) + np.log(self.erase_ratio[0])
                ratio = np.exp(log_ratio)
                ph = min(int(round(np.sqrt(patch_area / ratio))), H_t)
                pw = min(int(round(np.sqrt(patch_area * ratio))), W_t)
                y0 = torch.randint(0, H_t - ph + 1, (1,)).item()
                x0 = torch.randint(0, W_t - pw + 1, (1,)).item()
                fill = img_t.mean(dim=(1, 2), keepdim=True).expand(3, ph, pw)
                img_t[:, y0:y0 + ph, x0:x0 + pw] = fill
            pil_img = transforms.ToPILImage()(img_t.clamp(0, 1))
        return pil_img, seg_np


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation dataset wrapper — identical to Mask2Former pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    def __init__(self, base_dataset, image_transform=None,
                 joint_augmentation=None, use_cutmix=False, cutmix_p=0.5):
        self.base            = base_dataset
        self.image_transform = image_transform
        self.joint_aug       = joint_augmentation
        self.use_cutmix      = use_cutmix
        self.cutmix_p        = cutmix_p

    def __len__(self): return len(self.base)

    @staticmethod
    def _cutmix(img_a, seg_a, img_b, seg_b):
        H, W   = img_a.shape[:2]
        lam    = np.random.beta(1.0, 1.0)
        cut_w  = int(W * np.sqrt(1.0 - lam))
        cut_h  = int(H * np.sqrt(1.0 - lam))
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1 = max(cx - cut_w // 2, 0); y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W); y2 = min(cy + cut_h // 2, H)
        mixed_img = img_a.copy(); mixed_seg = seg_a.copy()
        mixed_img[y1:y2, x1:x2] = img_b[y1:y2, x1:x2]
        mixed_seg[y1:y2, x1:x2] = seg_b[y1:y2, x1:x2]
        return mixed_img, mixed_seg

    def __getitem__(self, idx):
        pil_img, sem_np, _ = self.base[idx]
        if self.joint_aug is not None:
            pil_img, sem_np = self.joint_aug(pil_img, sem_np)
        if self.use_cutmix and torch.rand(1).item() < self.cutmix_p:
            idx_b = torch.randint(len(self.base), (1,)).item()
            pil_b, sem_b, _ = self.base[idx_b]
            if self.joint_aug is not None:
                pil_b, sem_b = self.joint_aug(pil_b, sem_b)
            img_a, img_b = np.array(pil_img), np.array(pil_b)
            if img_b.shape[:2] != img_a.shape[:2]:
                pil_b = pil_b.resize((pil_img.width, pil_img.height), Image.BILINEAR)
                sem_b = np.array(Image.fromarray(sem_b.astype(np.int32), mode="I")
                                 .resize((pil_img.width, pil_img.height), Image.NEAREST),
                                 dtype=np.int64)
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
    def __init__(self, coco_file, root_dir, batch_size=4, num_workers=4,
                 resize=None, use_augmentation=True):
        super().__init__()
        self.coco_file        = coco_file
        self.root_dir         = root_dir
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.resize           = resize
        self.use_augmentation = use_augmentation
        resize_op = [transforms.Resize(resize)] if resize else []
        self.transform = transforms.Compose(
            resize_op + [transforms.ToTensor(),
                         transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)])
        self.joint_aug = JointAugmentation() if use_augmentation else None

    def setup(self, stage=None):
        self.train_ds = SegmentationDataset(
            COCODataset(self.coco_file, self.root_dir, split="train"),
            image_transform=self.transform,
            joint_augmentation=self.joint_aug,
            use_cutmix=self.use_augmentation, cutmix_p=0.5,
        )
        self.val_ds = SegmentationDataset(
            COCODataset(self.coco_file, self.root_dir, split="valid"),
            image_transform=self.transform,
        )
        self.test_ds = SegmentationDataset(
            COCODataset(self.coco_file, self.root_dir, split="test"),
            image_transform=self.transform,
        )

    @staticmethod
    def _collate(batch):
        # SegmentationDataset returns (image, seg_map, orig_img_arr, orig_seg_arr)
        images, seg_maps, _, _ = zip(*batch)
        return torch.stack(images), torch.stack(seg_maps)

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

class SegFormerLoRAModule(pl.LightningModule):
    """
    SegFormer fine-tuned with LoRA snapshot ensemble.
    Loss   : CrossEntropyLoss on upsampled logits (ignore_index=255)
    Metric : torchmetrics MeanIoU (DDP-safe, same as Mask2Former)
    """

    def __init__(
        self,
        pretrained_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_dora: bool = False,
        lora_bias: str = "none",
        target_modules: list = None,
        modules_to_save: list = None,
        fullft_modules: list = None,
        lr: float = 2e-4,
        lr_min: float = 2e-6,
        cosine_T0: int = 20,
        save_dir: str = "./checkpoints",
    ):
        super().__init__()
        self.save_hyperparameters()

        base_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            id2label=ID2LABEL, label2id=LABEL2ID,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
        )

        if target_modules  is None: target_modules  = DEFAULT_TARGET_MODULES
        if modules_to_save is None: modules_to_save = DEFAULT_MODULES_TO_SAVE

        print(f"\n[LoRA] Target modules  : {target_modules}")
        print(f"[LoRA] Modules to save : {modules_to_save}")
        print(f"[LoRA] r={lora_r}  alpha={lora_alpha}  bias={lora_bias}  dora={use_dora}")

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            lora_dropout=lora_dropout,
            bias=lora_bias, use_dora=use_dora,
        )
        self.model = get_peft_model(base_model, lora_cfg)
        if fullft_modules:                       # decoder full-FT on top of encoder LoRA
            n = 0
            for name, param in self.model.named_parameters():
                if any(fm in name for fm in fullft_modules):
                    param.requires_grad = True; n += 1
            print(f"[fullft] unfroze {n} params in {fullft_modules}")
        self._print_trainable_params()

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.val_miou  = MeanIoU(num_classes=NUM_LABELS, per_class=False)
        self.test_miou = MeanIoU(num_classes=NUM_LABELS, per_class=False)

    def _print_trainable_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)\n")

    def _upsample(self, logits, hw):
        return F.interpolate(logits, size=hw, mode="bilinear", align_corners=False)

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def training_step(self, batch, batch_idx):
        images, seg_maps = batch
        outputs = self(images)
        logits  = self._upsample(outputs.logits, images.shape[2:])
        loss    = self.criterion(logits, seg_maps.long())
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, seg_maps = batch
        outputs = self(images)
        logits  = self._upsample(outputs.logits, images.shape[2:])
        loss    = self.criterion(logits, seg_maps.long())
        preds   = logits.argmax(dim=1)
        self.log("val/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.val_miou.update(preds.to(self.device), seg_maps.to(self.device))
        return loss

    def on_validation_epoch_end(self):
        miou = self.val_miou.compute()
        self.log("val/mean_iou", miou, prog_bar=True, sync_dist=True)
        self.val_miou.reset()
        if self.trainer.is_global_zero:
            print(f"\n[Epoch {self.current_epoch}] Val mIoU: {miou.item():.4f}")

    def test_step(self, batch, batch_idx):
        images, seg_maps = batch
        outputs = self(images)
        logits  = self._upsample(outputs.logits, images.shape[2:])
        preds   = logits.argmax(dim=1)
        self.test_miou.update(preds.to(self.device), seg_maps.to(self.device))

    def on_test_epoch_end(self):
        miou = self.test_miou.compute()
        self.log("test/mean_iou", miou, rank_zero_only=True)
        self.test_miou.reset()
        if self.trainer.is_global_zero:
            print(f"\n[Test] Mean IoU: {miou.item():.4f}")
            results = {"mIoU": float(miou.item())}
            out_path = os.path.join(self.hparams.save_dir, "eval_test.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[Saved] {out_path}")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.hparams.lr,
        )
        sched = CosineAnnealingWarmRestarts(
            opt, T_0=self.hparams.cosine_T0, T_mult=1,
            eta_min=self.hparams.lr_min,
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched,
                                 "interval": "epoch", "frequency": 1}}


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Ensemble Callback — identical to Mask2Former
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
            if trainer.is_global_zero:
                print(f"\n[Snapshot] Saved adapter shot {shot} → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SegFormer LoRA Snapshot Ensemble Training (Seeded)")

    parser.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    parser.add_argument("--root_dir",  type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/")
    parser.add_argument("--base_save_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/segformer")
    parser.add_argument("--log_dir",   type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/segformer/logs")

    parser.add_argument("--pretrained", type=str,
        default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--resize", nargs=2, type=int, default=None,
        metavar=("H", "W"))

    # LoRA — overridden by --config_name if provided
    parser.add_argument("--config_name",  type=str, default=None,
        help="Named config from segformer_lora_configs.json")
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_bias",    type=str,   default="none",
        choices=["none", "all", "lora_only"])
    parser.add_argument("--no_dora",      action="store_true")
    parser.add_argument("--target_modules",  nargs="+", type=str,
        default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--modules_to_save", nargs="+", type=str,
        default=DEFAULT_MODULES_TO_SAVE)
    parser.add_argument("--fullft_modules", nargs="+", type=str, default=None,
        help="module-name substrings to fully fine-tune on top of LoRA (e.g. decode_head)")

    parser.add_argument("--epochs",          type=int,   default=110)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--lr_min",          type=float, default=2e-6)
    parser.add_argument("--cosine_T0",       type=int,   default=20)
    parser.add_argument("--snapshot_every",  type=int,   default=20)
    parser.add_argument("--no_augmentation", action="store_true")

    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--gpus",        type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--strategy",    type=str, default=None)
    parser.add_argument("--precision",   type=str, default="bf16-mixed")

    return parser.parse_args()


def load_named_config(config_name, script_dir):
    config_file = os.path.join(script_dir, "segformer_lora_configs.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file) as f:
        configs = {c["name"]: c for c in json.load(f)}
    if config_name not in configs:
        raise ValueError(f"Config '{config_name}' not in {list(configs.keys())}")
    return configs[config_name]


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    use_dora = not args.no_dora
    if args.config_name:
        cfg = load_named_config(
            args.config_name, os.path.dirname(os.path.abspath(__file__)))
        args.lora_r          = cfg["lora_r"]
        args.lora_alpha      = cfg["lora_alpha"]
        args.lora_dropout    = cfg["lora_dropout"]
        args.lora_bias       = cfg.get("lora_bias", "none")
        args.target_modules  = cfg.get("target_modules", DEFAULT_TARGET_MODULES)
        args.modules_to_save = cfg.get("modules_to_save", DEFAULT_MODULES_TO_SAVE)
        use_dora             = cfg.get("use_dora", False)
        print(f"[Config] Loaded '{args.config_name}'")

    config_label = args.config_name or f"r{args.lora_r}"
    save_dir = os.path.join(args.base_save_dir, config_label, f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SegFormer LoRA — config={config_label}  seed={args.seed}")
    print(f"  Save dir : {save_dir}")
    print(f"{'='*60}\n")

    resize      = tuple(args.resize) if args.resize else None
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
        resize=resize, use_augmentation=not args.no_augmentation,
    )

    model = SegFormerLoRAModule(
        pretrained_name=args.pretrained,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, use_dora=use_dora,
        lora_bias=args.lora_bias,
        target_modules=args.target_modules,
        modules_to_save=args.modules_to_save,
        fullft_modules=args.fullft_modules,
        lr=args.lr, lr_min=args.lr_min, cosine_T0=args.cosine_T0,
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
        save_dir=args.log_dir,
        name=f"segformer_lora_{config_label}_seed{args.seed}",
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