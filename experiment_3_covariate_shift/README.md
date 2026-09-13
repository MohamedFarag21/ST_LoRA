# Experiment 3 — Performance under covariate shift

Two complementary analyses. Both report **mIoU + ECE** (mean ± std over 5 seeds) on the
**last-four-snapshot ST-LoRA ensemble** (shots 2,3,4,5). Photometric corruptions alter pixel
statistics only (image); **geometric corruptions are applied to image *and* mask** (nearest-neighbour;
revealed regions → image 0, mask ignore 255).

## 3a — Augmentation importance / ensemble diversity (SegFormer-B2 / GrowliFlower-L)

Compares **four ensembles**: ST-LoRA and FRE, each **homogeneous** (LR schedule only, no augmentation)
vs **heterogeneous** (LR + augmentation). The heterogeneous training augmentation is
**random rotation ~U(0°, 25°) + horizontal flip (p=0.5), applied to image and mask** (NN on mask).

**Principle:** training augmentation is **rotation + horizontal-flip only**, so evaluation uses the
*other* shift types (we never test on what we trained on). The four evaluation corruptions
(continuous, 20 levels each) → **80 = 4 × 20**:
brightness [1.0,2.0], contrast [1.0,2.0], Gaussian noise σ [0,0.285], zoom [1.0×,1.475×].
> `shift_corruptions.GROWLI_ACTIVE` is set to these four (rotation is implemented but excluded from
> the test grid because it is a training augmentation; hflip likewise is training-only).

## 3b — Component sensitivity under shift (Mask2Former-Swin/B / BUP20)

Re-evaluates the **Experiment-2** ST-LoRA hyperparameter configs (rank, α, dropout, target modules,
structural variants) under shift — i.e. how robust each component choice is. These models are
**trained with NO augmentation** (exactly the Experiment-2 recipe) and then tested under covariate
shift. By default it **reuses the Experiment-2 checkpoints** (no retraining); a train-from-scratch
job is provided as an option.

Shift grid (3 discrete levels each; hflip single → 7×3 − 2 = **19** conditions):
brightness/contrast {0.50,0.65,0.80}, Gaussian noise σ {0.05,0.10,0.20}, Gaussian blur σ {1,2,3},
zoom {1.10,1.25,1.50}, translation {5,10,15}% width, horizontal flip (binary).

## Contents

```
experiment_3_covariate_shift/
├── env.sh
├── configs/m2f.json              # the Exp-2 catalog (3b indexes into it)
├── code/
│   ├── shift_corruptions.py           # corruption functions + the two shift grids
│   ├── shift_eval_segformer_growli.py # 3a: SegFormer/GrowliFlower ensemble under shift
│   ├── shift_eval_m2f_bup20.py        # 3b: M2F/BUP20 config under shift
│   ├── segformer_growli_train_seeded.py  # 3a trainer (hetero aug = rotation+hflip)
│   ├── mask2former_lora_train_seeded_aug.py + prepare_run/resolve_targets  # 3b train-from-scratch
│   └── calibration_shift_eval.py, run_seed_calibration_stream.py, segformer_growli_eval_seeded.py
└── jobs/
    ├── segformer_growli_train.sh      # 3a: train the 4 models (METHOD × VARIANT)
    ├── shift_eval_segformer_growli.sh # 3a: shift evaluation
    ├── shift_eval_m2f_bup20.sh        # 3b: shift evaluation (reads EXP2_M2F_DIR)
    └── train_m2f_bup20.sh             # 3b: OPTIONAL train-from-scratch
```

## Setup

```bash
conda env create -f ../environment.yml && conda activate st_lora
$EDITOR env.sh    # set BUP20_DIR, GROWLI_DIR, and EXP2_M2F_DIR (path to results/exp2/m2f)
source env.sh
```

## Run — SLURM

```bash
source env.sh
# --- 3a: train the four SegFormer/GrowliFlower models, then shift-eval ---
for M in stlora fre; do for V in homogeneous heterogeneous; do
  METHOD=$M VARIANT=$V sbatch jobs/segformer_growli_train.sh          # (homogeneous == the Exp-1 GrowliFlower arm)
done; done
for M in stlora fre; do for V in homogeneous heterogeneous; do
  METHOD=$M VARIANT=$V sbatch jobs/shift_eval_segformer_growli.sh
done; done

# --- 3b: shift-eval the Exp-2 M2F configs (reuse checkpoints via EXP2_M2F_DIR) ---
for S in 42 123 456 789 1337; do SEED=$S sbatch --array=0-34 jobs/shift_eval_m2f_bup20.sh; done
# (optional) retrain the M2F configs from scratch first:
# for S in 42 123 456 789 1337; do SEED=$S sbatch --array=0-34 jobs/train_m2f_bup20.sh; done
#   then set EXP2_M2F_DIR=$ST_LORA_ROOT/results/exp3/m2f_bup20 and run the shift-eval.
```

## Run — no SLURM

```bash
source env.sh; conda activate "$ENV_MAIN"
# 3a example (one model, one seed):
PYTHONPATH=code python code/shift_eval_segformer_growli.py --method stlora --variant heterogeneous \
  --seed 42 --results_dir results/exp3/growli/segformer_b2_stlora_heterogeneous --root_dir "$GROWLI_DIR"
# 3b example (one config, one seed):
PYTHONPATH=code python code/shift_eval_m2f_bup20.py --index 0 --seed 42 \
  --results_dir "$EXP2_M2F_DIR" --coco_file "$BUP20_COCO" --root_dir "$BUP20_DIR"
```

## Notes

- **Ensemble** = mean softmax of the last four snapshots (shots 2,3,4,5), consistent across the study.
- **ECE/mIoU** use the float64 streaming metric (`run_seed_calibration_stream.StreamBinMetrics`).
- **Geometric shifts corrupt the mask too** (NN); photometric shifts leave the mask untouched.
- 3a homogeneous models are identical to the Experiment-1 GrowliFlower arm — you can reuse those
  trained checkpoints instead of retraining (point `--results_dir` at them).
- Seeds `{42,123,456,789,1337}`; report mean ± std.
