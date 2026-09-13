# -*- coding: utf-8 -*-
"""
Mask2Former + LoRA (DoRA) Fine-tuning for Tomato Ripeness Segmentation
Mirrors mask2former_lora_train_seeded.py (sweet pepper) but reads the tomato_esra
dataset format: per-sequence rgb/<seq>/<ts>.png + annotations/<seq>/<ts>.pkl,
where each pkl is a dict of instance_id -> {bbox, area, semantic_label, instance_mask}.

Dataset layout (see train_val_eval_splits.yaml):
    'train' / 'val'  — sequences with pkl annotations (used for supervised training/validation)
    'eval'           — sequences with rgb only, no annotations (inference-only, no ground truth)

Usage:
    python mask2former_lora_train_tomato.py --seed 42
    python mask2former_lora_train_tomato.py --seed 42 --gpus 4
"""

import os
import time
import pickle
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import yaml
from PIL import Image
from torchvision import transforms

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

ID2LABEL = {
    0: "bg",
    1: "green",
    2: "orange",
    3: "mixed_orange",
    4: "red",
    5: "mixed_red",
}
LABEL2ID   = {name: idx for idx, name in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)  # 6


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TomatoDataset(Dataset):
    """Raw loader: rgb/<seq>/<ts>.png + annotations/<seq>/<ts>.pkl -> (PIL image, semantic map).

    'train' and 'val' sequences have annotations; 'eval' sequences are rgb-only
    (no ground-truth masks) and are intended for inference, not loss/metric computation.
    """

    def __init__(self, root_dir: str, split: str, splits_yaml: str = "train_val_eval_splits.yaml"):
        self.root_dir = root_dir
        self.labeled  = split != "eval"

        with open(os.path.join(root_dir, splits_yaml)) as f:
            splits = yaml.safe_load(f)
        seqs = splits[split]

        self.samples = []  # list of (seq, frame_ts)
        for seq in seqs:
            if self.labeled:
                ann_dir = os.path.join(root_dir, "annotations", seq)
                frame_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith(".pkl"))
            else:
                rgb_dir = os.path.join(root_dir, "rgb", seq)
                frame_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(rgb_dir)
                                   if f.endswith(".png") and not f.startswith("."))
            self.samples.extend((seq, ts) for ts in frame_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, ts = self.samples[idx]
        rgb_path = os.path.join(self.root_dir, "rgb", seq, f"{ts}.png")
        img = Image.open(rgb_path).convert("RGB")
        W, H = img.size

        sem_map = np.zeros((H, W), dtype=np.uint8)
        if self.labeled:
            ann_path = os.path.join(self.root_dir, "annotations", seq, f"{ts}.pkl")
            with open(ann_path, "rb") as f:
                instances = pickle.load(f)
            for inst in instances.values():
                sem_map[inst["instance_mask"]] = LABEL2ID.get(inst["semantic_label"], 0)

        return img, sem_map, f"{seq}/{ts}"


class SegmentationDataset(Dataset):
    def __init__(self, base_dataset: TomatoDataset, image_transform=None):
        self.base = base_dataset
        self.image_transform = image_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        pil_img, sem_np, _ = self.base[idx]
        image = self.image_transform(pil_img) if self.image_transform else transforms.ToTensor()(pil_img)
        seg_map = torch.from_numpy(sem_np.astype(np.int64))
        return image, seg_map, np.array(pil_img), seg_map.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Data Module
# ─────────────────────────────────────────────────────────────────────────────

class TomatoDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int = 4, num_workers: int = 4):
        super().__init__()
        self.root_dir    = root_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers

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

    def setup(self, stage=None):
        self.train_ds = SegmentationDataset(
            TomatoDataset(self.root_dir, split="train"),
            image_transform=self.train_transform,
        )
        self.val_ds = SegmentationDataset(
            TomatoDataset(self.root_dir, split="val"),
            image_transform=self.test_transform,
        )
        # 'eval' sequences have no ground-truth masks — inference only, see predict_dataloader().
        self.predict_ds = SegmentationDataset(
            TomatoDataset(self.root_dir, split="eval"),
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

    def _collate_predict(self, batch):
        images, _, orig_imgs, _ = zip(*batch)
        processed = self.preprocessor(list(images), return_tensors="pt")
        processed["original_images"] = list(orig_imgs)
        return processed

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self._collate, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self._collate, num_workers=self.num_workers,
                          persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self._collate_predict, num_workers=self.num_workers,
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
        target_modules: list = None,
        lr: float = 2e-4,
        lr_min: float = 2e-6,
        cosine_T0: int = 20,
        save_dir: str = "./checkpoints",
    ):
        super().__init__()
        self.save_hyperparameters()

        base_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_name, id2label=ID2LABEL, ignore_mismatched_sizes=True,
        )

        if target_modules is None:
            target_modules = ["dense", "q_proj", "k_proj", "class_predictor"]

        print(f"\n[LoRA] Target modules: {target_modules}")

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout, bias="none", use_dora=use_dora,
        )
        self.model = get_peft_model(base_model, lora_cfg)
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora" in name
        self._print_trainable_params()

        self.preprocessor = Mask2FormerImageProcessor(
            ignore_index=255, reduce_labels=False,
            do_resize=False, do_rescale=False, do_normalize=False,
            num_labels=NUM_LABELS,
        )

        self.val_miou = MeanIoU(num_classes=NUM_LABELS, per_class=False, input_format="index")

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

    def predict_step(self, batch, batch_idx):
        outputs = self(pixel_values=batch["pixel_values"])
        orig_imgs = batch["original_images"]
        target_sz = [(img.shape[0], img.shape[1]) for img in orig_imgs]
        preds = self.preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sz)
        return {"predictions": preds, "original_images": orig_imgs}

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.hparams.lr,
        )
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
    parser = argparse.ArgumentParser(description="Mask2Former LoRA Training — Tomato")

    parser.add_argument("--root_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/data/tomato_esra")
    parser.add_argument("--base_save_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/tomato_lora",
        help="Parent directory. Seed-specific subdir is created automatically.")
    parser.add_argument("--log_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/tomato_lora/logs")

    parser.add_argument("--pretrained",    type=str, default="facebook/mask2former-swin-base-ade-semantic")
    parser.add_argument("--lora_r",        type=int,   default=8)
    parser.add_argument("--lora_alpha",    type=int,   default=8)
    parser.add_argument("--lora_dropout",  type=float, default=0.1)
    parser.add_argument("--no_dora",       action="store_true")
    parser.add_argument("--target_modules", nargs="+", type=str,
        default=["dense", "q_proj", "k_proj", "class_predictor"],
        help="LoRA target module names, e.g. --target_modules q_proj k_proj value dense")

    parser.add_argument("--epochs",          type=int,   default=110)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--lr_min",          type=float, default=2e-6)
    parser.add_argument("--cosine_T0",       type=int,   default=20)
    parser.add_argument("--snapshot_every",  type=int,   default=20)

    parser.add_argument("--gpus",        type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--strategy",    type=str, default=None)
    parser.add_argument("--precision",   type=str, default="32")

    parser.add_argument("--seed", type=int, default=42,
        help="Random seed. Run once per seed for significance analysis.")

    return parser.parse_args()


def main():
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)

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

    data_module = TomatoDataModule(
        root_dir=args.root_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = Mask2FormerLoRAModule(
        pretrained_name=args.pretrained,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, use_dora=not args.no_dora,
        target_modules=args.target_modules,
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
        save_dir=args.log_dir, name=f"mask2former_lora_tomato_seed{args.seed}"
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


if __name__ == "__main__":
    main()
