# -*- coding: utf-8 -*-
"""
EoMT (Encoder-only Mask Transformer) + LoRA (ST-LoRA) fine-tuning on bup20 sweet-pepper, 8-class.
Sibling of eomt_full_train_seeded.py (the FRE analog). REUSES that file's data pipeline,
label map, dataset and snapshot cadence verbatim; only the trainable parametrisation changes:
full fine-tuning -> LoRA adapters + a few fully-trained heads.

TRAINING CONFIG IS MATCHED TO eomt_full_train_seeded.py (FRE) so the two are directly comparable:
same ckpt/data/splits/metric, epochs=110, batch_size=2, lr=1e-4, backbone_lr_mult=0.1,
weight_decay=0.05, cosine_T0=20, snapshot_every=20, precision=bf16-mixed, AdamW, NO augmentation.

LoRA parametrisation (rank 8, alpha 8 -> scale 1.0, dropout 0.1, standard LoRA / no DoRA),
verified against the real EoMT module tree (results/lora_paper/eomt_structure/*):

  four module groups (user spec) ─────────────────────────────────────────────────────────
  * layers  (24x EomtLayer, DINOv2 backbone):
        attention -> q_proj, k_proj, v_proj, out_proj   (Linear 1024x1024, 24 each)
        mlp       -> fc1, fc2                            (Linear, 24 each)
  * mask_head (EomtMaskHead): fc1, fc2, fc3             (Linear 1024x1024)
  * class_predictor (Linear 1024 -> num_labels+1, freshly-init, frozen base + trained LoRA delta)
  * upscale_block (EomtScaleBlock): FULLY TRAINED (user choice), NOT LoRA — it contains conv1
        which is a ConvTranspose2d that PEFT 0.15.2 LoRA does not support, and conv2 which is a
        depthwise Conv2d. modules_to_save keeps the whole block trainable, covering both.

  IMPORTANT suffix collision (by design, matches user intent): the bare suffixes `fc1`/`fc2`
  match BOTH `layers.N.mlp.fc1/2` (24 each) AND `mask_head.fc1/2` (1 each) — 25 modules each.
  So `fc1,fc2` alone already LoRA-adapts the mask head's first two linears; `fc3` is added
  explicitly to complete mask-head coverage. This is intended; the dry-run job asserts it.

  class_predictor caveat (see [[reference_stlora_frozen_head_seed_bug]]): loaded with
  id2label=ID2LABEL + ignore_mismatched_sizes so its base is freshly RANDOM-initialised and then
  FROZEN; only the LoRA delta trains. pl.seed_everything(seed) is called BEFORE model construction
  so that random init is deterministic per seed.

SLURM only (never run python directly). Env (ISOLATED eomt, transformers>=5.14, peft 0.15.2):
  module purge; module load CUDA/12.6.0; module load Miniforge3; source activate eomt
  export HF_HOME=<lustre cache>; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"""
import os
import sys
import time
import argparse

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from transformers import EomtForUniversalSegmentation, EomtImageProcessor
from peft import LoraConfig, get_peft_model
from torchmetrics.segmentation import MeanIoU

# Reuse the EXACT data pipeline / label map / dataset from the EoMT Full-FT (FRE) trainer.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eomt_full_train_seeded import (                                  # noqa: E402
    PepperEomtDataModule, IGNORE_INDEX, DEFAULT_CKPT,
)
from mask2former_full_train_seeded import ID2LABEL, NUM_LABELS        # noqa: E402

# LoRA target module name-suffixes (PEFT matches a module whose dotted name endswith `.suffix`
# or equals it). fc1/fc2 intentionally hit layers.*.mlp AND mask_head; fc3 completes mask_head.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "out_proj",   # layers.*.attention
    "fc1", "fc2",                               # layers.*.mlp  +  mask_head
    "fc3",                                      # mask_head only
    "class_predictor",                          # top-level head
]
# Fully fine-tuned (no LoRA) — contains conv1 (ConvTranspose2d, unsupported by LoRA) + conv2.
DEFAULT_MODULES_TO_SAVE = ["upscale_block"]


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer param groups — MIRRORS eomt_full.split_param_groups but over PEFT names.
# Backbone adapters (inside `layers.`) train at backbone_lr_mult x base_lr; every other
# trainable tensor (mask_head / class_predictor LoRA + fully-trained upscale_block) at base_lr.
# ─────────────────────────────────────────────────────────────────────────────
def split_lora_param_groups(model, base_lr, backbone_lr_mult):
    bb, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if ".layers." in n else head).append(p)
    groups = [{"params": head, "lr": base_lr},
              {"params": bb, "lr": base_lr * backbone_lr_mult}]
    return groups, len(head), len(bb)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter snapshot ensemble — saves the PEFT adapter (+ modules_to_save) every N epochs,
# same cadence as the FRE SnapshotCallback (shots at epoch 0,20,...,100 for 110 epochs).
# ─────────────────────────────────────────────────────────────────────────────
class EomtLoRASnapshotCallback(pl.Callback):
    def __init__(self, save_dir, save_every=20):
        self.save_dir = save_dir
        self.save_every = save_every

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if epoch % self.save_every == 0 and trainer.is_global_zero:
            shot = epoch // self.save_every
            path = os.path.join(self.save_dir, f"model_shot_{shot}")
            pl_module.model.save_pretrained(path)
            print(f"\n[Snapshot] Saved adapter snapshot {shot} -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Lightning module
# ─────────────────────────────────────────────────────────────────────────────
class EomtLoRAModule(pl.LightningModule):
    def __init__(self, ckpt=DEFAULT_CKPT, lora_r=8, lora_alpha=8, lora_dropout=0.1,
                 use_dora=False, target_modules=None, modules_to_save=None,
                 lr=1e-4, lr_min=1e-6, cosine_T0=20, backbone_lr_mult=0.1,
                 weight_decay=0.05, grad_checkpointing=False, save_dir="./checkpoints"):
        super().__init__()
        self.save_hyperparameters()

        target_modules = target_modules or DEFAULT_TARGET_MODULES
        modules_to_save = modules_to_save or DEFAULT_MODULES_TO_SAVE

        base = EomtForUniversalSegmentation.from_pretrained(
            ckpt, id2label=ID2LABEL, ignore_mismatched_sizes=True)
        if grad_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()

        print(f"\n[LoRA] r={lora_r} alpha={lora_alpha} dropout={lora_dropout} "
              f"dora={use_dora}\n[LoRA] target_modules={target_modules}"
              f"\n[LoRA] modules_to_save (fully trained)={modules_to_save}")
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
            lora_dropout=lora_dropout, bias="none", use_dora=use_dora,
            modules_to_save=modules_to_save)
        # get_peft_model sets requires_grad correctly: LoRA params + modules_to_save.default
        # trainable, everything else frozen. We trust that (NOT the `"lora" in name` override,
        # which would wrongly re-freeze the fully-trained upscale_block).
        self.model = get_peft_model(base, lora_cfg)

        self.processor = EomtImageProcessor.from_pretrained(ckpt, ignore_index=IGNORE_INDEX)
        self.val_miou = MeanIoU(num_classes=NUM_LABELS, per_class=False)
        self.test_miou = MeanIoU(num_classes=NUM_LABELS, per_class=False)
        self._print_trainable()

    def _print_trainable(self):
        tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.model.parameters())
        print(f"\n[LoRA] Trainable: {tr:,} / {tot:,} ({100*tr/tot:.3f}%)\n")

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
        out = self.model(pixel_values=batch["pixel_values"], mask_labels=ml,
                         class_labels=cl, patch_offsets=batch.get("patch_offsets"))
        self.log(f"{tag}/loss", out.loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        preds = self.processor.post_process_semantic_segmentation(
            out, target_sizes=batch["target_sizes"])
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
        # MIRRORS mask2former_lora_train_seeded.py EXACTLY: single flat Adam over ALL
        # trainable params (LoRA adapters + fully-trained heads) at ONE learning rate.
        # No param-group split / backbone_lr_mult (that was a FRE full-FT recipe artifact
        # that starved the zero-init encoder adapters at 1e-5) and no weight decay.
        trainables = [p for p in self.model.parameters() if p.requires_grad]
        if self.trainer.is_global_zero:
            n = sum(p.numel() for p in trainables)
            print(f"[optim] flat Adam over {len(trainables)} trainable tensors "
                  f"({n:,} params) @ single lr={self.hparams.lr}")
        opt = torch.optim.Adam(trainables, lr=self.hparams.lr)
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
    p.add_argument("--base_save_dir", default=f"{R}/results/lora_paper/eomt_lora")
    p.add_argument("--log_dir", default=f"{R}/results/lora_paper/eomt_lora/logs")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    # ── LoRA hyperparams ──────────────────────────────────────────────────────
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--use_dora", action="store_true", help="off by default -> standard LoRA")
    p.add_argument("--target_modules", nargs="+", default=None)
    p.add_argument("--modules_to_save", nargs="+", default=None)
    # ── training config (MATCHED to eomt_full FRE) ────────────────────────────
    p.add_argument("--epochs", type=int, default=110)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)          # mask2former ST-LoRA default
    p.add_argument("--lr_min", type=float, default=2e-6)      # mask2former ST-LoRA default
    p.add_argument("--backbone_lr_mult", type=float, default=1.0,
                   help="DEPRECATED/unused — optimizer is now a single flat LR (mask2former parity)")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="unused — mask2former ST-LoRA reference uses plain Adam, no weight decay")
    p.add_argument("--cosine_T0", type=int, default=20)
    p.add_argument("--snapshot_every", type=int, default=20)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="bf16-mixed")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_augmentation", action="store_true", default=True,
                   help="EoMT ST-LoRA uses the NO-AUG case by default (matches the FRE no-aug run)")
    p.add_argument("--augmentation", dest="no_augmentation", action="store_false",
                   help="opt back into JointAugmentation+CutMix (parity variant)")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    # Seed BEFORE model construction so the freshly-init (then frozen) class_predictor base
    # is deterministic per seed — see [[reference_stlora_frozen_head_seed_bug]].
    pl.seed_everything(args.seed, workers=True)

    save_dir = os.path.join(args.base_save_dir, f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"\n{'='*60}\n  EoMT + LoRA (ST-LoRA) — matched to EoMT FRE config\n"
          f"  ckpt={args.ckpt}\n  seed={args.seed} epochs={args.epochs} bs={args.batch_size} "
          f"lr={args.lr} precision={args.precision} aug={not args.no_augmentation}\n"
          f"  save={save_dir}\n{'='*60}\n")

    devices = args.gpus if torch.cuda.is_available() else 1
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    strategy = "ddp_find_unused_parameters_true" if devices > 1 else "auto"

    dm = PepperEomtDataModule(
        coco_file=args.coco_file, root_dir=args.root_dir, ckpt=args.ckpt,
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_augmentation=not args.no_augmentation)
    model = EomtLoRAModule(
        ckpt=args.ckpt, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, use_dora=args.use_dora,
        target_modules=args.target_modules, modules_to_save=args.modules_to_save,
        lr=args.lr, lr_min=args.lr_min, cosine_T0=args.cosine_T0,
        backbone_lr_mult=args.backbone_lr_mult, weight_decay=args.weight_decay,
        grad_checkpointing=args.grad_checkpointing, save_dir=save_dir)

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(save_dir, "best_iou"),
                        filename="best-iou-{epoch:03d}-{val/mean_iou:.4f}",
                        monitor="val/mean_iou", mode="max", save_top_k=1),
        EomtLoRASnapshotCallback(save_dir=save_dir, save_every=args.snapshot_every),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(save_dir=args.log_dir, name=f"eomt_lora_seed{args.seed}")

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
