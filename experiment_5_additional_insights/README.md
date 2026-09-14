# Experiment 5 — Additional insights

Four supporting analyses for Mask2Former on BUP20 (ST-LoRA vs FRE unless noted).

## 1. Ensemble-member diversity vs the training recipe

How the **learning-rate schedule** and **augmentation** shape member diversity. Three variants,
for both ST-LoRA and FRE:

| Variant | LR schedule | Augmentation |
|---|---|---|
| constant-LR | constant (`--constant_lr`, no cosine restarts) | none (`--no_augmentation`) |
| cosine | cosine warm-restart | none |
| cosine + aug | cosine warm-restart | on |

- Train: `mask2former_lora_train_seeded_aug.py` (ST-LoRA) / `mask2former_full_train_seeded.py` (FRE)
  with the flags above. Score diversity: `per_class_diversity_ensemble.py`, aggregate with
  `aggregate_per_class_diversity.py`.
- Jobs: `m2f_div_stlora_train.sh`, `m2f_div_fullft_train.sh`, `m2f_div_score.sh`.

## 2. Qualitative results (success / failure)

- **In-distribution:** `stlora_success_failure_panels.py` — e.g. a class the model predicts well
  vs a harder one (green vs mixed pepper). Job: `stlora_success_failure_panels.sh`.
- **Pixel-level OoD:** `stlora_ood_success_failure_panels.py` — a GrowliFlower-L case (model
  struggles) and a tomato/BUTom21 case (model behaves well). Job: `stlora_ood_panels.sh`.
- Uncertainty maps: `uncertainty_decomp_maps.py` (`uncertainty_decomp.sh`), `visualise_msp.py`,
  `selective_pred_viz*.py`. `run_all_viz.sh` runs the panels together.

## 3. Per-class performance (BUP20)

Per-class **mIoU** (`per_class_iou.py`) and per-class **calibration / proper scores**
(`per_class_scores_uq.py`). Jobs: `per_class_iou.sh`, `per_class_scores_uq.sh`.

## 4. Ensemble-size ablation (with vs without the first three snapshots)

Metric vs number of ensemble members, evaluated two ways: **all snapshots** and **dropping the
first three** (the early, less-converged snapshots).

- `ens_size_ablation_eval.py` + `plot_ens_size_ablation.py`.
- Jobs: `ens_ablation_T20_train_stlora.sh`, `ens_ablation_T20_train_fre.sh`,
  `ens_ablation_T20_eval.sh` (all snapshots), `ens_ablation_T20_eval_from4.sh` (drops the first
  three → members from snapshot 4 on), `plot_ens_size_ablation_T20.sh`.

## Setup & run

```bash
conda env create -f ../environment.yml && conda activate st_lora
$EDITOR env.sh          # set BUP20_DIR (+ GROWLI_DIR, BUTOM21_DIR for the OoD panels)
source env.sh
# examples:
sbatch jobs/m2f_div_stlora_train.sh ; sbatch jobs/m2f_div_fullft_train.sh ; sbatch jobs/m2f_div_score.sh   # (1)
sbatch jobs/stlora_success_failure_panels.sh ; sbatch jobs/stlora_ood_panels.sh                              # (2)
sbatch jobs/per_class_iou.sh ; sbatch jobs/per_class_scores_uq.sh                                            # (3)
sbatch jobs/ens_ablation_T20_eval.sh ; sbatch jobs/ens_ablation_T20_eval_from4.sh ; sbatch jobs/plot_ens_size_ablation_T20.sh  # (4)
```

Each script is a plain `python code/<script>.py …` and runs without SLURM too (copy the invocation
from the matching `jobs/*.sh`). Seeds `{42,123,456,789,1337}`.
