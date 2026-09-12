# -*- coding: utf-8 -*-
"""
SegFormer (B2 / B4) on GrowliFlower-L — binary plant-vs-background, seeded.
===========================================================================
Experiment 1 (single model, no ensembling): ST-LoRA (rank 8) vs full fine-tune (FRE),
NO augmentation. One script covers both methods (--method) and both backbones (--pretrained).

Mirrors segformer_lora_train_seeded.py's recipe exactly (CosineAnnealingWarmRestarts T0=20,
snapshot every 20 epochs, CrossEntropy on upsampled logits, decode_head fully trained under
LoRA via modules_to_save). Only the dataset differs.

GrowliFlower-L layout (semantic, binary here):
    images/{Train,Val,Test}/<stem>.jpg
    labels/{Train,Val,Test}/maskPlants/<stem>_Label_maskPlants.png   -> class 1 (plant) where raw>0
    labels/{Train,Val,Test}/maskVoid/<stem>_Label_maskVoid.png       -> ignore (255) where raw>0
Masks are 8-bit palette PNGs; we read the RAW palette index (no RGB convert) and threshold >0
(the GrowliFlower native-palette convention). Everything else is background (0).

Usage (single GPU, no augmentation):
    python segformer_growli_train_seeded.py --method stlora --lora_r 8 --lora_alpha 8 \
        --pretrained nvidia/segformer-b2-finetuned-ade-512-512 --no_augmentation \
        --seed 42 --root_dir /path/to/growliflower_l --base_save_dir OUT
    python segformer_growli_train_seeded.py --method fre \
        --pretrained nvidia/segformer-b4-finetuned-ade-512-512 --no_augmentation --seed 42 ...
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
from torchvision import transforms
import torchvision.transforms.v2.functional as TF

from transformers import SegformerForSemanticSegmentation
from peft import LoraConfig, get_peft_model
from torchmetrics.segmentation import MeanIoU

# ─────────────────────────────────────────────────────────────────────────────
# Label config — binary plant vs background
# ─────────────────────────────────────────────────────────────────────────────
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255

ID2LABEL   = {0: "bg", 1: "plant"}
LABEL2ID   = {"bg": 0, "plant": 1}
NUM_LABELS = 2
IGNORE_INDEX = 255

SPLITS = {"train": "Train", "valid": "Val", "test": "Test"}

DEFAULT_TARGET_MODULES  = ["query", "key", "value", "dense"]   # MixTransformer encoder
DEFAULT_MODULES_TO_SAVE = ["decode_head"]                       # head fully trained under LoRA


# ─────────────────────────────────────────────────────────────────────────────
# GrowliFlower-L dataset (binary)
# ─────────────────────────────────────────────────────────────────────────────
class GrowliFlowerDataset(Dataset):
    """Reads images/<Split> + labels/<Split>/maskPlants|maskVoid. Returns (image_tensor, seg_long)."""

    def __init__(self, root_dir, split, resize=None, augment=False):
        self.root = root_dir
        self.split_dir = SPLITS[split]
        self.img_dir = os.path.join(root_dir, "images", self.split_dir)
        self.lbl_dir = os.path.join(root_dir, "labels", self.split_dir)
        self.resize = resize          # (H, W) or None
        self.augment = augment
        self.stems = sorted(os.path.splitext(f)[0]
                            for f in os.listdir(self.img_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.normalize = transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)

    def __len__(self):
        return len(self.stems)

    def _read_mask(self, masktype, stem):
        p = os.path.join(self.lbl_dir, masktype, f"{stem}_Label_{masktype}.png")
        if not os.path.exists(p):
            return None
        # raw palette index (do NOT convert to RGB) -> foreground is raw>0
        return np.array(Image.open(p)) > 0

    def _compose_seg(self, stem, hw):
        seg = np.zeros(hw, dtype=np.uint8)                 # 0 = background
        plant = self._read_mask("maskPlants", stem)
        if plant is not None:
            seg[plant] = 1                                 # 1 = plant
        void = self._read_mask("maskVoid", stem)
        if void is not None:
            seg[void] = IGNORE_INDEX                       # 255 = ignore (overrides)
        return seg

    def __getitem__(self, idx):
        stem = self.stems[idx]
        pil_img = Image.open(os.path.join(self.img_dir, f"{stem}.jpg")).convert("RGB")
        W, H = pil_img.size
        seg = self._compose_seg(stem, (H, W))

        if self.resize is not None:
            rh, rw = self.resize
            pil_img = pil_img.resize((rw, rh), Image.BILINEAR)
            seg = np.array(Image.fromarray(seg).resize((rw, rh), Image.NEAREST))

        if self.augment and torch.rand(1).item() < 0.5:   # simple hflip (no-aug leaves this off)
            pil_img = TF.horizontal_flip(pil_img)
            seg = np.fliplr(seg).copy()

        image = self.normalize(transforms.ToTensor()(pil_img))
        seg_t = torch.from_numpy(seg.astype(np.int64)).long()
        return image, seg_t


class GrowliFlowerDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=4, num_workers=4, resize=None, use_augmentation=False):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.use_augmentation = use_augmentation

    def setup(self, stage=None):
        self.train_ds = GrowliFlowerDataset(self.root_dir, "train", self.resize,
                                            augment=self.use_augmentation)
        self.val_ds   = GrowliFlowerDataset(self.root_dir, "valid", self.resize, augment=False)
        self.test_ds  = GrowliFlowerDataset(self.root_dir, "test",  self.resize, augment=False)

    def _dl(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def train_dataloader(self): return self._dl(self.train_ds, True)
    def val_dataloader(self):   return self._dl(self.val_ds, False)
    def test_dataloader(self):  return self._dl(self.test_ds, False)


# ─────────────────────────────────────────────────────────────────────────────
# Lightning module (ST-LoRA or FRE)
# ─────────────────────────────────────────────────────────────────────────────
class SegFormerGrowliModule(pl.LightningModule):
    def __init__(self, method="stlora",
                 pretrained_name="nvidia/segformer-b2-finetuned-ade-512-512",
                 lora_r=8, lora_alpha=8, lora_dropout=0.1, use_dora=False, lora_bias="none",
                 target_modules=None, modules_to_save=None,
                 lr=2e-3, lr_min=2e-5, cosine_T0=20, save_dir="./checkpoints"):
        super().__init__()
        self.save_hyperparameters()

        base = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name, id2label=ID2LABEL, label2id=LABEL2ID,
            num_labels=NUM_LABELS, ignore_mismatched_sizes=True)

        if method == "stlora":
            tm = target_modules or DEFAULT_TARGET_MODULES
            ms = modules_to_save or DEFAULT_MODULES_TO_SAVE
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=tm,
                             modules_to_save=ms, lora_dropout=lora_dropout,
                             bias=lora_bias, use_dora=use_dora)
            self.model = get_peft_model(base, cfg)
            print(f"[ST-LoRA] r={lora_r} alpha={lora_alpha} targets={tm} save={ms}")
        elif method == "fre":
            self.model = base                              # full fine-tune (all params trainable)
            print("[FRE] full fine-tune (all parameters trainable)")
        else:
            raise ValueError(method)

        tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.model.parameters())
        print(f"[params] trainable {tr:,} / {tot:,} ({100*tr/tot:.2f}%)")

        self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.val_miou  = MeanIoU(num_classes=NUM_LABELS, per_class=False)
        self.test_miou = MeanIoU(num_classes=NUM_LABELS, per_class=False)

    def _up(self, logits, hw):
        return F.interpolate(logits, size=hw, mode="bilinear", align_corners=False)

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def training_step(self, batch, _):
        images, seg = batch
        logits = self._up(self(images).logits, images.shape[2:])
        loss = self.criterion(logits, seg.long())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        images, seg = batch
        logits = self._up(self(images).logits, images.shape[2:])
        loss = self.criterion(logits, seg.long())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_miou.update(logits.argmax(1).to(self.device), seg.to(self.device))
        return loss

    def on_validation_epoch_end(self):
        self.log("val/mean_iou", self.val_miou.compute(), prog_bar=True, sync_dist=True)
        self.val_miou.reset()

    def test_step(self, batch, _):
        images, seg = batch
        logits = self._up(self(images).logits, images.shape[2:])
        self.test_miou.update(logits.argmax(1).to(self.device), seg.to(self.device))

    def on_test_epoch_end(self):
        miou = float(self.test_miou.compute())
        self.test_miou.reset()
        if self.trainer.is_global_zero:
            print(f"[Test] mIoU: {miou:.4f}")
            with open(os.path.join(self.hparams.save_dir, "eval_test.json"), "w") as f:
                json.dump({"mIoU": miou}, f, indent=2)

    def configure_optimizers(self):
        opt = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                               lr=self.hparams.lr)
        sched = CosineAnnealingWarmRestarts(opt, T_0=self.hparams.cosine_T0, T_mult=1,
                                            eta_min=self.hparams.lr_min)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1}}


class SnapshotEnsembleCallback(pl.Callback):
    """Save a snapshot at each LR trough: LoRA -> adapter dir; FRE -> full state_dict .pt."""
    def __init__(self, save_dir, save_every, method):
        self.save_dir = save_dir
        self.save_every = save_every
        self.method = method

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.save_every != 0:
            return
        shot = epoch // self.save_every
        if self.method == "stlora":
            path = os.path.join(self.save_dir, f"model_shot_{shot}")
            pl_module.model.save_pretrained(path)
        else:
            path = os.path.join(self.save_dir, f"model_shot_{shot}.pt")
            torch.save(pl_module.model.state_dict(), path)
        if trainer.is_global_zero:
            print(f"[Snapshot] shot {shot} -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SegFormer B2/B4 on GrowliFlower-L (binary), seeded")
    p.add_argument("--root_dir", type=str, required=True,
                   help="GrowliFlower-L root (contains images/ and labels/)")
    p.add_argument("--base_save_dir", type=str, required=True)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--pretrained", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    p.add_argument("--method", type=str, default="stlora", choices=["stlora", "fre"])
    p.add_argument("--resize", nargs=2, type=int, default=None, metavar=("H", "W"),
                   help="optional resize; GrowliFlower patches are 448x368 and uniform, so default None")
    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    p.add_argument("--no_dora", action="store_true")
    p.add_argument("--target_modules", nargs="+", type=str, default=DEFAULT_TARGET_MODULES)
    p.add_argument("--modules_to_save", nargs="+", type=str, default=DEFAULT_MODULES_TO_SAVE)
    # schedule / optim
    p.add_argument("--epochs", type=int, default=110)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr_min", type=float, default=2e-5)
    p.add_argument("--cosine_T0", type=int, default=20)
    p.add_argument("--snapshot_every", type=int, default=20)
    p.add_argument("--no_augmentation", action="store_true")
    # runtime
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--strategy", type=str, default=None)
    p.add_argument("--precision", type=str, default="bf16-mixed")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    save_dir = os.path.join(args.base_save_dir, f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    log_dir = args.log_dir or os.path.join(args.base_save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    tag = f"segformer_growli_{args.method}_seed{args.seed}"
    print(f"\n{'='*60}\n  {tag}\n  backbone : {args.pretrained}\n  save_dir : {save_dir}\n{'='*60}\n")

    resize = tuple(args.resize) if args.resize else None
    devices = args.gpus if torch.cuda.is_available() else 1
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    strategy = args.strategy if args.strategy else ("ddp_find_unused_parameters_true" if devices > 1 else "auto")

    dm = GrowliFlowerDataModule(root_dir=args.root_dir, batch_size=args.batch_size,
                                num_workers=args.num_workers, resize=resize,
                                use_augmentation=not args.no_augmentation)

    model = SegFormerGrowliModule(
        method=args.method, pretrained_name=args.pretrained,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        use_dora=not args.no_dora, lora_bias=args.lora_bias,
        target_modules=args.target_modules, modules_to_save=args.modules_to_save,
        lr=args.lr, lr_min=args.lr_min, cosine_T0=args.cosine_T0, save_dir=save_dir)

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(save_dir, "best_iou"),
                        filename="best-iou-{epoch:03d}-{val/mean_iou:.4f}",
                        monitor="val/mean_iou", mode="max", save_top_k=1),
        SnapshotEnsembleCallback(save_dir=save_dir, save_every=args.snapshot_every, method=args.method),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(save_dir=log_dir, name=tag)

    trainer = pl.Trainer(
        max_epochs=args.epochs, devices=devices, accelerator=accelerator, strategy=strategy,
        precision=args.precision, callbacks=callbacks, logger=logger,
        log_every_n_steps=10, enable_progress_bar=True, sync_batchnorm=(devices > 1))

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    print(f"\nTraining complete in {(time.time()-t0)/60:.1f} min")
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
