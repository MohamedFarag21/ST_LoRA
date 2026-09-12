# -*- coding: utf-8 -*-
"""
SegFormer Full Fine-tuning — Snapshot Ensemble
===============================================
All parameters trainable. Identical cosine annealing schedule and snapshot
saving to mask2former_full_train_seeded.py.
Snapshots saved as model_shot_N.pt (state_dict, no PEFT).

Usage:
    python segformer_fullft_train_seeded.py --seed 42
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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
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
# Joint augmentation — identical to all other scripts
# ─────────────────────────────────────────────────────────────────────────────

class JointAugmentation:
    def __init__(self, flip_p=0.5, translate_p=0.5, translate_range=0.10,
                 brightness=0.3, contrast=0.3, saturation=0.1,
                 blur_p=0.3, blur_sigma=(0.5, 1.5),
                 noise_p=0.2, noise_std=0.02,
                 local_blur_p=0.3,
                 erase_p=0.3, erase_scale=(0.02, 0.15),
                 erase_ratio=(0.3, 3.3), erase_n=3):
        self.flip_p = flip_p; self.translate_p = translate_p
        self.translate_range = translate_range
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation)
        self.blur_p = blur_p; self.blur_sigma = blur_sigma
        self.noise_p = noise_p; self.noise_std = noise_std
        self.local_blur_p = local_blur_p
        self.erase_p = erase_p; self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio; self.erase_n = erase_n

    def __call__(self, pil_img, seg_np):
        W, H = pil_img.size
        if torch.rand(1).item() < self.flip_p:
            pil_img = TF.horizontal_flip(pil_img)
            seg_np  = np.fliplr(seg_np).copy()
        if torch.rand(1).item() < self.translate_p:
            max_px = int(self.translate_range * W)
            shift  = torch.randint(-max_px, max_px + 1, (1,)).item()
            if shift != 0:
                a = (1, 0, -shift, 0, 1, 0)
                pil_img = pil_img.transform(pil_img.size, Image.AFFINE, a,
                                             resample=Image.BILINEAR, fillcolor=0)
                sp = Image.fromarray(seg_np.astype(np.int32), mode="I")
                sp = sp.transform(sp.size, Image.AFFINE, a,
                                  resample=Image.NEAREST, fillcolor=0)
                seg_np = np.array(sp, dtype=np.int64)
        pil_img = self.color_jitter(pil_img)
        if torch.rand(1).item() < self.blur_p:
            s = self.blur_sigma[0] + torch.rand(1).item() * (self.blur_sigma[1] - self.blur_sigma[0])
            k = int(2 * round(2 * s) + 1); k = k if k % 2 == 1 else k + 1
            pil_img = TF.gaussian_blur(pil_img, kernel_size=k, sigma=s)
        if torch.rand(1).item() < self.noise_p:
            t = transforms.ToTensor()(pil_img)
            pil_img = transforms.ToPILImage()((t + torch.randn_like(t) * self.noise_std).clamp(0, 1))
        if torch.rand(1).item() < self.local_blur_p:
            s = self.blur_sigma[0] + torch.rand(1).item() * (self.blur_sigma[1] - self.blur_sigma[0])
            k = int(2 * round(2 * s) + 1); k = k if k % 2 == 1 else k + 1
            arr = np.array(pil_img).astype(np.float32)
            blr = np.array(TF.gaussian_blur(pil_img, kernel_size=k, sigma=s)).astype(np.float32)
            ha, wa = arr.shape[:2]
            xx, yy = np.meshgrid(np.linspace(-1, 1, wa), np.linspace(-1, 1, ha))
            w = np.exp(-(xx**2 + yy**2) / (2*0.7**2))[:, :, np.newaxis]
            pil_img = Image.fromarray((w*arr + (1-w)*blr).clip(0,255).astype(np.uint8))
        if torch.rand(1).item() < self.erase_p:
            t = transforms.ToTensor()(pil_img)
            _, Ht, Wt = t.shape
            for _ in range(torch.randint(1, self.erase_n+1, (1,)).item()):
                pa = Ht*Wt*(self.erase_scale[0]+torch.rand(1).item()*(self.erase_scale[1]-self.erase_scale[0]))
                lr = np.exp(torch.rand(1).item()*(np.log(self.erase_ratio[1])-np.log(self.erase_ratio[0]))+np.log(self.erase_ratio[0]))
                ph = min(int(round(np.sqrt(pa/lr))), Ht)
                pw = min(int(round(np.sqrt(pa*lr))), Wt)
                y0 = torch.randint(0, Ht-ph+1, (1,)).item()
                x0 = torch.randint(0, Wt-pw+1, (1,)).item()
                t[:, y0:y0+ph, x0:x0+pw] = t.mean(dim=(1,2), keepdim=True).expand(3, ph, pw)
            pil_img = transforms.ToPILImage()(t.clamp(0, 1))
        return pil_img, seg_np


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    def __init__(self, base_dataset, image_transform=None,
                 joint_augmentation=None, use_cutmix=False, cutmix_p=0.5):
        self.base = base_dataset; self.image_transform = image_transform
        self.joint_aug = joint_augmentation
        self.use_cutmix = use_cutmix; self.cutmix_p = cutmix_p

    def __len__(self): return len(self.base)

    @staticmethod
    def _cutmix(ia, sa, ib, sb):
        H, W = ia.shape[:2]; lam = np.random.beta(1., 1.)
        cw = int(W*np.sqrt(1.-lam)); ch = int(H*np.sqrt(1.-lam))
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1=max(cx-cw//2,0); y1=max(cy-ch//2,0)
        x2=min(cx+cw//2,W); y2=min(cy+ch//2,H)
        mi=ia.copy(); ms=sa.copy()
        mi[y1:y2,x1:x2]=ib[y1:y2,x1:x2]; ms[y1:y2,x1:x2]=sb[y1:y2,x1:x2]
        return mi, ms

    def __getitem__(self, idx):
        pil_img, sem_np, _ = self.base[idx]
        if self.joint_aug: pil_img, sem_np = self.joint_aug(pil_img, sem_np)
        if self.use_cutmix and torch.rand(1).item() < self.cutmix_p:
            ib, sb, _ = self.base[torch.randint(len(self.base),(1,)).item()]
            if self.joint_aug: ib, sb = self.joint_aug(ib, sb)
            ia, ib = np.array(pil_img), np.array(ib)
            if ib.shape[:2] != ia.shape[:2]:
                ib = np.array(Image.fromarray(ib.astype(np.uint8)).resize(
                    (pil_img.width, pil_img.height), Image.BILINEAR))
                sb = np.array(Image.fromarray(sb.astype(np.int32), mode="I").resize(
                    (pil_img.width, pil_img.height), Image.NEAREST), dtype=np.int64)
            mi, sem_np = self._cutmix(ia, sem_np, ib, sb)
            pil_img = Image.fromarray(mi.astype(np.uint8))
        image   = self.image_transform(pil_img) if self.image_transform \
                  else transforms.ToTensor()(pil_img)
        seg_map = torch.from_numpy(_REMAP_LUT[sem_np.astype(np.int64)]).long()
        return image, seg_map, np.array(pil_img), seg_map.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Data Module
# ─────────────────────────────────────────────────────────────────────────────

class SweetPepperDataModule(pl.LightningDataModule):
    def __init__(self, coco_file, root_dir, batch_size=4,
                 num_workers=4, use_augmentation=True):
        super().__init__()
        self.coco_file = coco_file; self.root_dir = root_dir
        self.batch_size = batch_size; self.num_workers = num_workers
        self.use_augmentation = use_augmentation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)])
        self.joint_aug = JointAugmentation() if use_augmentation else None

    def setup(self, stage=None):
        self.train_ds = SegmentationDataset(
            COCODataset(self.coco_file, self.root_dir, "train"),
            self.transform, self.joint_aug, use_cutmix=self.use_augmentation)
        self.val_ds  = SegmentationDataset(
            COCODataset(self.coco_file, self.root_dir, "valid"), self.transform)
        self.test_ds = SegmentationDataset(
            COCODataset(self.coco_file, self.root_dir, "test"),  self.transform)

    @staticmethod
    def _collate(batch):
        images, seg_maps, _, _ = zip(*batch)
        return torch.stack(images), torch.stack(seg_maps)

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, shuffle=True,
                          collate_fn=self._collate, num_workers=self.num_workers,
                          persistent_workers=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds,  self.batch_size, shuffle=False,
                          collate_fn=self._collate, num_workers=self.num_workers,
                          persistent_workers=True)
    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size, shuffle=False,
                          collate_fn=self._collate, num_workers=self.num_workers,
                          persistent_workers=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class SegFormerFullFTModule(pl.LightningModule):
    """SegFormer full fine-tuning — all parameters trainable, snapshot ensemble."""

    def __init__(self, pretrained_name="nvidia/segformer-b2-finetuned-ade-512-512",
                 lr=2e-4, lr_min=2e-6, cosine_T0=20, save_dir="./checkpoints"):
        super().__init__()
        self.save_hyperparameters()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name, id2label=ID2LABEL, label2id=LABEL2ID,
            num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"\n[Full FT] Trainable parameters: {total:,}\n")
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.val_miou  = MeanIoU(num_classes=NUM_LABELS, per_class=False)
        self.test_miou = MeanIoU(num_classes=NUM_LABELS, per_class=False)

    def _upsample(self, logits, hw):
        return F.interpolate(logits, size=hw, mode="bilinear", align_corners=False)

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def training_step(self, batch, batch_idx):
        images, seg_maps = batch
        logits = self._upsample(self(images).logits, images.shape[2:])
        loss   = self.criterion(logits, seg_maps.long())
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, seg_maps = batch
        logits = self._upsample(self(images).logits, images.shape[2:])
        loss   = self.criterion(logits, seg_maps.long())
        preds  = logits.argmax(dim=1)
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
        logits = self._upsample(self(images).logits, images.shape[2:])
        preds  = logits.argmax(dim=1)
        self.test_miou.update(preds.to(self.device), seg_maps.to(self.device))

    def on_test_epoch_end(self):
        miou = self.test_miou.compute()
        self.log("test/mean_iou", miou, rank_zero_only=True)
        self.test_miou.reset()
        if self.trainer.is_global_zero:
            print(f"\n[Test] Mean IoU: {miou.item():.4f}")
            results = {"mIoU": float(miou.item())}
            with open(os.path.join(self.hparams.save_dir, "eval_test.json"), "w") as f:
                json.dump(results, f, indent=2)

    def configure_optimizers(self):
        opt   = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        sched = CosineAnnealingWarmRestarts(
            opt, T_0=self.hparams.cosine_T0, T_mult=1, eta_min=self.hparams.lr_min)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot callback — saves state_dict at each LR trough
# ─────────────────────────────────────────────────────────────────────────────

class SnapshotCallback(pl.Callback):
    def __init__(self, save_dir, save_every=20):
        self.save_dir = save_dir; self.save_every = save_every

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch > 0 and epoch % self.save_every == 0:
            shot = epoch // self.save_every
            path = os.path.join(pl_module.hparams.save_dir, f"model_shot_{shot}.pt")
            torch.save(pl_module.model.state_dict(), path)
            if trainer.is_global_zero:
                print(f"\n[Snapshot] Saved shot {shot} → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_file", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    p.add_argument("--root_dir",  type=str, default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/bup_20")
    p.add_argument("--base_save_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/segformer_fullft")
    p.add_argument("--log_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/segformer_fullft/logs")
    p.add_argument("--pretrained", type=str,
        default="nvidia/segformer-b2-finetuned-ade-512-512")
    p.add_argument("--epochs",          type=int,   default=110)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--lr_min",          type=float, default=2e-6)
    p.add_argument("--cosine_T0",       type=int,   default=20)
    p.add_argument("--snapshot_every",  type=int,   default=20)
    p.add_argument("--gpus",            type=int,   default=1)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--strategy",        type=str,   default=None)
    p.add_argument("--precision",       type=str,   default="bf16-mixed")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--no_augmentation", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    save_dir = os.path.join(args.base_save_dir, f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SegFormer Full FT Snapshot Ensemble  seed={args.seed}")
    print(f"  Epochs: {args.epochs}  Snapshot every: {args.snapshot_every}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*60}\n")

    devices     = args.gpus if torch.cuda.is_available() else 1
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    strategy    = args.strategy or ("ddp_find_unused_parameters_true"
                                    if devices > 1 else "auto")

    data_module = SweetPepperDataModule(
        args.coco_file, args.root_dir, args.batch_size, args.num_workers,
        use_augmentation=not args.no_augmentation)
    model = SegFormerFullFTModule(
        args.pretrained, args.lr, args.lr_min, args.cosine_T0, save_dir)

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(save_dir, "best"),
                        filename="best-{epoch:03d}-{val/loss:.4f}",
                        monitor="val/loss", mode="min", save_top_k=1),
        ModelCheckpoint(dirpath=os.path.join(save_dir, "best_iou"),
                        filename="best-iou-{epoch:03d}-{val/mean_iou:.4f}",
                        monitor="val/mean_iou", mode="max", save_top_k=3),
        SnapshotCallback(save_dir, args.snapshot_every),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs, devices=devices, accelerator=accelerator,
        strategy=strategy, precision=args.precision,
        callbacks=callbacks,
        logger=TensorBoardLogger(args.log_dir, name=f"fullft_seed{args.seed}"),
        log_every_n_steps=10, enable_progress_bar=True,
        sync_batchnorm=(devices > 1))

    t0 = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f"\nTraining done in {(time.time()-t0)/60:.1f} min")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
