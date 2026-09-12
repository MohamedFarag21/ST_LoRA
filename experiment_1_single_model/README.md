# Experiment 1 — Single-model ST-LoRA (r8) vs FRE, no augmentation

**Question.** For a *single* fine-tuned model (no ensembling — the last snapshot of the training
trajectory), how does **ST-LoRA (rank 8)** compare to **full fine-tuning (FRE)** when trained
**without augmentation**, across three segmentation backbones?

**Architectures:** Mask2Former (Swin-B), SegFormer (MiT), EoMT (ViT-L).
**Datasets:** BUP20 (sweet pepper, 8 classes) for M2F/SegFormer/EoMT; plus **GrowliFlower-L**
(cauliflower, binary plant-vs-background) for a SegFormer-B2/B4 arm. **Seeds:** `{42, 123, 456, 789, 1337}`.

Dataset links:
- **BUP20** — https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/b5d18108-53e1-46d1-873d-4230a72dfad7
- **GrowliFlower-L** — https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/cb328232-31f5-4b84-a929-8e1ee551d66a

**What we report**
- **Accuracy:** mIoU.
- **Calibration (all):** ECE, MECE, ACE, MACE, Brier, NLL (float64, cross-arch parity).
- **Reliability diagrams** (per-bin accuracy vs confidence).
- **Efficiency / computational complexity:** number of fine-tuned (trainable) parameters,
  **wall-clock** training time, **GPU energy** (kWh), and **CO₂** (gCO₂eq) — measured with
  CodeCarbon, with a per-epoch energy trace and an nvidia-smi power cross-check
  (grid factor 380.95 gCO₂/kWh).

---

## Contents

```
experiment_1_single_model/
├── env.sh          # set ST_LORA_ROOT + BUP20_DIR + GROWLI_DIR here (used by both run modes)
├── code/           # all Python (flat; scripts import each other by name)
└── jobs/           # SLURM launchers (portable: ST_LORA_ROOT, relative logs, no personal info)
```

### Files by role

| Role | Mask2Former | SegFormer | EoMT |
|---|---|---|---|
| Train ST-LoRA r8 | `mask2former_lora_train_seeded_aug.py` | `segformer_lora_train_seeded.py` | `eomt_lora_train_seeded.py` |
| Train FRE | `mask2former_full_train_seeded.py` | `segformer_fullft_train_seeded.py` | `eomt_full_train_seeded.py` |
| Evaluate (mIoU + calibration) | `elora_evaluation_seeded.py` / `fullft_evaluation_seeded.py` | `segformer_lora_eval_seeded.py` / `segformer_fullft_eval_seeded.py` | `eomt_lora_ensemble_evaluate_seeded.py` (+ `eomt_evaluate_seeded.py`) |
| Efficiency (params/time/energy/CO₂) | `co2_compare_train.py` | `segformer_co2_compare_train.py` | `eomt_co2_compare_train.py` |
| Efficiency plots / stats | `plot_co2_energy_compare.py`, `co2_paired_significance.py`, `co2_crossarch_significance.py` | `plot_co2_energy_compare_segformer.py`, `co2_paired_significance_segformer.py` | `plot_co2_energy_compare_eomt.py` |

Matching SLURM launchers in `jobs/`: `*_train*.sh`, `eval_noaug.sh`, `eomt_stlora_single_eval.sh`,
`co2_train.sh` / `segformer_co2_train.sh` / `eomt_co2_train_5seeds.sh`, `plot_co2_energy.sh` /
`segformer_co2_plot.sh`, `co2_crossarch_sig.sh`, `agg_co2_eomt_3arm.sh`.

### SegFormer-B2 / B4 on GrowliFlower-L (binary plant vs background)

Same single-model, no-aug, ST-LoRA-r8-vs-FRE comparison, on a second dataset with two SegFormer
backbones. One trainer + one eval cover both backbones and both methods (env vars `BACKBONE=b2|b4`,
`METHOD=stlora|fre`):

| Role | Script | Job |
|---|---|---|
| Train (B2/B4, ST-LoRA r8 or FRE) | `segformer_growli_train_seeded.py` | `segformer_growli_train.sh` |
| Evaluate single model (mIoU + all calibration) | `segformer_growli_eval_seeded.py` | `segformer_growli_eval.sh` |

Backbones: `nvidia/segformer-b2-finetuned-ade-512-512`, `nvidia/segformer-b4-finetuned-ade-512-512`.
Labels: `maskPlants` raw>0 → plant (1), `maskVoid` raw>0 → ignore (255), else background (0).

---

## Setup

```bash
# 1) create the environments (once), from the repo root
conda env create -f ../environment.yml        # -> st_lora   (Mask2Former + SegFormer)
conda env create -f ../environment_eomt.yml    # -> st_lora_eomt (EoMT)
# CO2 runs additionally need codecarbon in the EoMT env clone (see env.sh: ENV_EOMT_CC)

# 2) point env.sh at your data + choose where outputs go
$EDITOR env.sh          # set BUP20_DIR, GROWLI_DIR (and ST_LORA_ROOT if not using the default)
source env.sh
```
`BUP20_DIR` should contain the COCO-style annotation json (`BUP20_COCO`) and the images.
`GROWLI_DIR` should contain `images/{Train,Val,Test}` and `labels/{Train,Val,Test}/{maskPlants,maskVoid,…}`.
Download links are at the top of this file.

---

## Run — SLURM cluster

Each launcher is a 5-seed array (`--array=0-4`). Adapt the `#SBATCH --partition/--account/--gres`
headers to your cluster, then:

```bash
source env.sh

# --- train (ST-LoRA r8, no-aug) ---
sbatch jobs/m2f_lora_r8_noaug_train.sh
sbatch jobs/segformer_lora_r8_noaug_train.sh
sbatch jobs/eomt_lora_flatlr_train_5seeds.sh        # EoMT (uses the eomt env)
# --- train (FRE, no-aug) ---
sbatch jobs/m2f_fre_noaug_train.sh
sbatch jobs/segformer_fre_noaug_train.sh
sbatch jobs/eomt_noaug_train_5seeds.sh

# --- evaluate the SINGLE model (last snapshot = shot 5): mIoU + ECE/MECE/ACE/MACE/Brier/NLL ---
SET=m2flora sbatch jobs/eval_noaug.sh
SET=m2ffre  sbatch jobs/eval_noaug.sh
SET=seglora sbatch jobs/eval_noaug.sh
SET=segfre  sbatch jobs/eval_noaug.sh
sbatch jobs/eomt_stlora_single_eval.sh

# --- efficiency (trainable params, wall-clock, GPU energy, CO2) ---
METHOD=stlora sbatch jobs/co2_train.sh ;  METHOD=fullft sbatch jobs/co2_train.sh
METHOD=stlora sbatch jobs/segformer_co2_train.sh ;  METHOD=fullft sbatch jobs/segformer_co2_train.sh
METHOD=stlora sbatch jobs/eomt_co2_train_5seeds.sh ; METHOD=fullft sbatch jobs/eomt_co2_train_5seeds.sh
# then aggregate / plot / test
sbatch jobs/plot_co2_energy.sh ; sbatch jobs/segformer_co2_plot.sh
sbatch jobs/agg_co2_eomt_3arm.sh ; sbatch jobs/co2_crossarch_sig.sh

# --- SegFormer-B2/B4 on GrowliFlower-L (binary): train then single-model eval ---
for BB in b2 b4; do for M in stlora fre; do
  BACKBONE=$BB METHOD=$M sbatch jobs/segformer_growli_train.sh
done; done
# after training finishes:
for BB in b2 b4; do for M in stlora fre; do
  BACKBONE=$BB METHOD=$M sbatch jobs/segformer_growli_eval.sh
done; done
```

---

## Run — no SLURM (plain workstation, single GPU)

Same scripts, called directly. `PYTHONPATH=code` lets the scripts find each other; pass the data
path explicitly. Loop over seeds yourself. Example for **Mask2Former ST-LoRA r8 (no-aug)**:

```bash
source env.sh
conda activate "$ENV_MAIN"

for S in 42 123 456 789 1337; do
  PYTHONPATH=code python code/mask2former_lora_train_seeded_aug.py \
    --seed $S --no_augmentation --lora_r 8 --lora_alpha 8 --lora_dropout 0.1 --no_dora \
    --target_modules k_proj v k q_proj dense class_predictor \
    --epochs 110 --batch_size 4 --lr 2e-4 --lr_min 2e-6 --cosine_T0 20 --snapshot_every 20 \
    --gpus 1 --num_workers 8 --precision 32 \
    --coco_file "$BUP20_COCO" --root_dir "$BUP20_DIR" \
    --base_save_dir "$ST_LORA_ROOT/results/m2f_lora_r8_noaug" \
    --log_dir "$ST_LORA_ROOT/results/m2f_lora_r8_noaug/logs"
done

# evaluate the single model (shot 5)
for S in 42 123 456 789 1337; do
  PYTHONPATH=code python code/elora_evaluation_seeded.py \
    --results_dir "$ST_LORA_ROOT/results/m2f_lora_r8_noaug" --seed $S --shot_ids 5 --split test \
    --coco_file "$BUP20_COCO" --root_dir "$BUP20_DIR"
done
```

The other arms follow the same pattern — copy the exact flags from the matching `jobs/*.sh`
(they differ only in script name and a few hyperparameters: SegFormer uses `--config_name
final_model --target_modules query key value dense --lr 2e-3 --precision bf16-mixed`; EoMT uses
`ENV_EOMT`, `--batch_size 2 --precision bf16-mixed`). FRE arms drop the LoRA flags and use
`--precision bf16-mixed`.

> **EoMT note:** activate `st_lora_eomt` (`conda activate "$ENV_EOMT"`) for any `eomt_*` script —
> it needs transformers 5.14.1.

---

## Reproducibility notes

- **Single model = last snapshot (shot 5).** Ensembling is *not* part of Experiment 1.
- **ST-LoRA frozen-head seeding:** Mask2Former/EoMT reinitialize the classifier head as a LoRA
  target with `modules_to_save=null`; the eval scripts seed the RNG before each base-model load so
  results are deterministic (±0 vs ±0.2 mIoU otherwise).
- **Calibration is float64** and defined identically across the three architectures.
- **Efficiency** is measured on the CO₂ arm (`*_co2_compare_train.py`) which logs trainable-param
  count, wall-clock, GPU energy (kWh) and CO₂ (gCO₂eq) via CodeCarbon.

### One piece still to wire (reliability diagrams)
The eval scripts emit the scalar calibration metrics (ECE/MECE/ACE/MACE/Brier/NLL) but not yet the
per-bin arrays a reliability diagram needs. Adding a `--dump_bins` flag to the `*_evaluation_seeded`
scripts (or a small single-model reliability plotter) is the remaining step for the reliability
figure; the underlying binning already exists in the ECE computation.
