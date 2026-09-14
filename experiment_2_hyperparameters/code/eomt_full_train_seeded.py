# -*- coding: utf-8 -*-
"""
EoMT (Encoder-only Mask Transformer) full fine-tuning on bup20 sweet-pepper, 8-class.
Sibling of mask2former_full_train_seeded.py — SAME data/splits/metric/snapshot scheme,
only the model + processor + a few EoMT-specific data rules change. Runs in the ISOLATED
`eomt` conda env (transformers>=5.14); ssl is untouched.

Everything below was VERIFIED against the real 5.14 API (see eomt_shape_smoke.py logs):

  * Model/loss contract is IDENTICAL to Mask2Former: encoder-only is an ARCHITECTURE
    statement, not a target one. Training still feeds `mask_labels` (per-target binary
    masks) + `class_labels` (their class ids); `model(...).loss` is the same Hungarian
    set-prediction loss. No decoder does NOT mean no mask/class targets.

  * EomtImageProcessor(ignore_index=255): fixed 512x512 input, DINOv2/ImageNet norm
    (mean .485/.456/.406 — NOT the ADE stats used for Swin-Mask2Former). It HARD-WIRES
    the ADE convention `label 0 -> ignore, real classes 1-indexed` (image_processing_eomt
    :63, no reduce_labels toggle). Pepper is 0=REAL bg / 255=ignore, so we feed a REMAP
    `where(seg==255, 0, seg+1)`: 255->0(ignore), classes 0..7 -> 1..8. The processor's
    internal `-1` then round-trips predictions back to 0..7 with bg preserved as class 0.

  * Non-square 1280x720 frames are TILED into overlapping 512x512 patches, so the
    processor emits PER-PATCH pixel_values/mask_labels/class_labels + patch_offsets
    (B images -> N>=B patches). Training is per-patch; eval stitches patches back to the
    native map via post_process_semantic_segmentation(outputs, target_sizes) which reads
    outputs.patch_offsets (the model echoes patch_offsets passed to forward).

  * Backbone (DINOv2 ViT: embeddings/layernorm/layers.0..23) gets a smaller LR than the
    freshly-initialised heads (query/upscale_block/mask_head/class_predictor).

SLURM only (never run python directly). Env:
  module purge; module load CUDA/12.6.0; module load Miniforge3; source activate eomt
  export HF_HOME=<lustre cache>; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"""
import os
import sys
import time
import argparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from transformers import EomtForUniversalSegmentation, EomtImageProcessor
from torchmetrics.segmentation import MeanIoU

# Reuse the EXACT dataset / augmentation / label map / snapshot logic from the
# Mask2Former Full-FT trainer (importing runs only module-level defs, not main()).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mask2former_full_train_seeded import (                       # noqa: E402
    COCODataset, JointAugmentation, SnapshotCallback,
    ID2LABEL, NUM_LABELS, _REMAP_LUT,
)

IGNORE_INDEX = 255
DEFAULT_CKPT = "tue-mps/ade20k_semantic_eomt_large_512"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: raw uint8 image + 0..7 label map. NO normalization here — the EoMT
# processor does resize(512^2)+rescale+normalize+tiling+target-encoding in collate.
# ─────────────────────────────────────────────────────────────────────────────
class EomtPepperDataset(Dataset):
    def __init__(self, base: COCODataset, joint_augmentation=None,
                 use_cutmix: bool = False, cutmix_p: float = 0.5):
        self.base = base
        self.joint_aug = joint_augmentation
        self.use_cutmix = use_cutmix
        self.cutmix_p = cutmix_p

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _cutmix(img_a, seg_a, img_b, seg_b):
        # box-CutMix IDENTICAL to mask2former_full_train_seeded.SegmentationDataset._cutmix
        H, W = img_a.shape[:2]
        lam = np.random.beta(1.0, 1.0)
        cut_w = int(W * np.sqrt(1.0 - lam))
        cut_h = int(H * np.sqrt(1.0 - lam))
        cx = np.random.randint(W); cy = np.random.randint(H)
        x1 = max(cx - cut_w // 2, 0); y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W); y2 = min(cy + cut_h // 2, H)
        mixed_img = img_a.copy(); mixed_seg = seg_a.copy()
        mixed_img[y1:y2, x1:x2] = img_b[y1:y2, x1:x2]
        mixed_seg[y1:y2, x1:x2] = seg_b[y1:y2, x1:x2]
        return mixed_img, mixed_seg

    def __getitem__(self, idx):
        pil_img, sem_np, _ = self.base[idx]          # PIL, uint8 ORIG-id map, image_id
        if self.joint_aug is not None:
            pil_img, sem_np = self.joint_aug(pil_img, sem_np)

        # CutMix on the ORIG-id map (before remap), matching the Mask2Former recipe
        if self.use_cutmix and torch.rand(1).item() < self.cutmix_p:
            idx_b = torch.randint(len(self.base), (1,)).item()
            pil_b, sem_b, _ = self.base[idx_b]
            if self.joint_aug is not None:
                pil_b, sem_b = self.joint_aug(pil_b, sem_b)
            img_a = np.array(pil_img.convert("RGB"))
            img_b = np.array(pil_b.convert("RGB"))
            sem_b = np.asarray(sem_b, dtype=np.int64)
            if img_b.shape[:2] != img_a.shape[:2]:
                pil_b = pil_b.resize((pil_img.width, pil_img.height), Image.BILINEAR)
                sem_b = np.array(
                    Image.fromarray(sem_b.astype(np.int32), mode="I").resize(
                        (pil_img.width, pil_img.height), Image.NEAREST), dtype=np.int64)
                img_b = np.array(pil_b.convert("RGB"))
            mixed_img, sem_np = self._cutmix(
                img_a, np.asarray(sem_np, dtype=np.int64), img_b, sem_b)
            pil_img = Image.fromarray(mixed_img.astype(np.uint8))

        img = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)         # (H,W,3)
        seg = _REMAP_LUT[np.asarray(sem_np, dtype=np.int64)]             # 0..7 (bg=0)
        return img, seg.astype(np.int64)


class PepperEomtDataModule(pl.LightningDataModule):
    def __init__(self, coco_file, root_dir, ckpt=DEFAULT_CKPT,
                 batch_size=2, num_workers=4, use_augmentation=True):
        super().__init__()
        self.coco_file = coco_file
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_aug = use_augmentation
        self.processor = EomtImageProcessor.from_pretrained(ckpt, ignore_index=IGNORE_INDEX)
        self.joint_aug = JointAugmentation() if use_augmentation else None

    def setup(self, stage=None):
        self.train_ds = EomtPepperDataset(
            COCODataset(self.coco_file, self.root_dir, "train"), self.joint_aug,
            use_cutmix=self.use_aug, cutmix_p=0.5)
        self.val_ds = EomtPepperDataset(
            COCODataset(self.coco_file, self.root_dir, "valid"), None)
        self.test_ds = EomtPepperDataset(
            COCODataset(self.coco_file, self.root_dir, "test"), None)

    def _collate(self, batch):
        imgs, segs = zip(*batch)
        # pepper (0=bg real, 255=ignore) -> EoMT convention (0=ignore, classes 1..8)
        eomt_maps = [np.where(s == IGNORE_INDEX, 0, s + 1) for s in segs]
        proc = self.processor(images=list(imgs), segmentation_maps=eomt_maps,
                              return_tensors="pt")
        # native 0..7 GT + native (H=1280,W=720) sizes for eval / patch stitching
        proc["original_segmentation_maps"] = [torch.from_numpy(s) for s in segs]
        proc["target_sizes"] = [tuple(s.shape) for s in segs]
        return proc

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=self._collate, num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)

    def test_dataloader(self):
        return self._loader(self.test_ds, False)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone-vs-head param groups: pretrained DINOv2 ViT trains slower than fresh heads.
# ─────────────────────────────────────────────────────────────────────────────
def split_param_groups(model, base_lr, backbone_lr_mult):
    bb_prefixes = ("embeddings", "layernorm", "layers.")
    bb, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if n.startswith(bb_prefixes) else head).append(p)
    groups = [{"params": head, "lr": base_lr},
              {"params": bb, "lr": base_lr * backbone_lr_mult}]
    return groups, len(head), len(bb)


# ─────────────────────────────────────────────────────────────────────────────
# Lightning module
# ─────────────────────────────────────────────────────────────────────────────
class EomtFullModule(pl.LightningModule):
    def __init__(self, ckpt=DEFAULT_CKPT, lr=1e-4, lr_min=1e-6, cosine_T0=20,
                 backbone_lr_mult=0.1, weight_decay=0.05, grad_checkpointing=False,
                 save_dir="./checkpoints"):
        super().__init__()
        self.save_hyperparameters()
        self.model = EomtForUniversalSegmentation.from_pretrained(
            ckpt, id2label=ID2LABEL, ignore_mismatched_sizes=True)
        if grad_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.processor = EomtImageProcessor.from_pretrained(ckpt, ignore_index=IGNORE_INDEX)
        self.val_miou = MeanIoU(num_classes=NUM_LABELS, per_class=False)
        self.test_miou = MeanIoU(num_classes=NUM_LABELS, per_class=False)
        n = sum(p.numel() for p in self.model.parameters())
        print(f"\n[EoMT Full-FT] params: {n/1e6:.1f}M | num_labels={self.model.config.num_labels}\n")

    def _mask_class(self, batch):
        return ([m.to(self.device) for m in batch["mask_labels"]],
                [c.to(self.device) for c in batch["class_labels"]])

    def training_step(self, batch, batch_idx):
        ml, cl = self._mask_class(batch)
        out = self.model(pixel_values=batch["pixel_values"], mask_labels=ml, class_labels=cl)
        self.log("train/loss", out.loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return out.loss

    def _eval_step(self, batch, miou, tag):
        ml, cl = self._mask_class(batch)
        # pass patch_offsets so outputs carry them -> post_process stitches tiles per image
        out = self.model(pixel_values=batch["pixel_values"], mask_labels=ml,
                         class_labels=cl, patch_offsets=batch.get("patch_offsets"))
        self.log(f"{tag}/loss", out.loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        preds = self.processor.post_process_semantic_segmentation(
            out, target_sizes=batch["target_sizes"])                 # list[(H,W) class-id]
        for pred, ref in zip(preds, batch["original_segmentation_maps"]):
            miou.update(pred.unsqueeze(0).to(self.device),
                        ref.unsqueeze(0).to(self.device))
        return out.loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, self.val_miou, "val")

    def on_validation_epoch_end(self):
        m = self.val_miou.compute()
        self.log("val/mean_iou", m, prog_bar=True, sync_dist=True)
        self.val_miou.reset()
        if self.trainer.is_global_zero:
            print(f"\n[Epoch {self.current_epoch}] Val mIoU: {m.item():.4f}")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, self.test_miou, "test")

    def on_test_epoch_end(self):
        m = self.test_miou.compute()
        self.log("test/mean_iou", m, rank_zero_only=True)
        self.test_miou.reset()
        if self.trainer.is_global_zero:
            print(f"\n[Test] Mean IoU: {m.item():.4f}")

    def configure_optimizers(self):
        groups, nh, nb = split_param_groups(
            self.model, self.hparams.lr, self.hparams.backbone_lr_mult)
        if self.trainer.is_global_zero:
            print(f"[optim] head={nh} tensors @lr={self.hparams.lr} | "
                  f"backbone={nb} tensors @lr={self.hparams.lr*self.hparams.backbone_lr_mult}")
        opt = torch.optim.AdamW(groups, lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = CosineAnnealingWarmRestarts(
            opt, T_0=self.hparams.cosine_T0, T_mult=1, eta_min=self.hparams.lr_min)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1}}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
         "mibrahi2_hpc-my_research-1775524204")
    p = argparse.ArgumentParser()
    p.add_argument("--coco_file", default=f"{R}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    p.add_argument("--root_dir", default=f"{R}/data/bup_20")
    p.add_argument("--base_save_dir", default=f"{R}/results/lora_paper/eomt_full")
    p.add_argument("--log_dir", default=f"{R}/results/lora_paper/eomt_full/logs")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--epochs", type=int, default=110)
    p.add_argument("--batch_size", type=int, default=2)      # x tiles => effective larger
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--cosine_T0", type=int, default=20)
    p.add_argument("--snapshot_every", type=int, default=20)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="bf16-mixed")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_augmentation", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: few train/val batches to validate the full pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    save_dir = os.path.join(args.base_save_dir, f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"\n{'='*60}\n  EoMT Full Fine-tuning (Snapshot Ensemble)\n"
          f"  ckpt={args.ckpt}\n  seed={args.seed} epochs={args.epochs} "
          f"bs={args.batch_size} precision={args.precision}\n  save={save_dir}\n{'='*60}\n")

    devices = args.gpus if torch.cuda.is_available() else 1
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    strategy = "ddp_find_unused_parameters_true" if devices > 1 else "auto"

    dm = PepperEomtDataModule(
        coco_file=args.coco_file, root_dir=args.root_dir, ckpt=args.ckpt,
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_augmentation=not args.no_augmentation)
    model = EomtFullModule(
        ckpt=args.ckpt, lr=args.lr, lr_min=args.lr_min, cosine_T0=args.cosine_T0,
        backbone_lr_mult=args.backbone_lr_mult, weight_decay=args.weight_decay,
        grad_checkpointing=args.grad_checkpointing, save_dir=save_dir)

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(save_dir, "best_iou"),
                        filename="best-iou-{epoch:03d}-{val/mean_iou:.4f}",
                        monitor="val/mean_iou", mode="max", save_top_k=1),
        SnapshotCallback(save_dir=save_dir, save_every=args.snapshot_every),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(save_dir=args.log_dir, name=f"eomt_full_seed{args.seed}")

    extra = dict(limit_train_batches=8, limit_val_batches=4) if args.smoke else {}
    trainer = pl.Trainer(
        max_epochs=(2 if args.smoke else args.epochs),
        devices=devices, accelerator=accelerator, strategy=strategy,
        precision=args.precision, callbacks=callbacks, logger=logger,
        log_every_n_steps=5, enable_progress_bar=True,
        sync_batchnorm=(devices > 1), **extra)

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    print(f"\nTraining complete in {(time.time()-t0)/60:.1f} min")
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
