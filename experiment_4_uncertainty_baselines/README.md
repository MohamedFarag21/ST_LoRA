# Experiment 4 — Uncertainty baselines: calibration, shift, and OoD

Compares **ST-LoRA** and **full fine-tuning (FRE)** — both trained with full augmentation — against
**MC-Dropout**, **DDU**, and **post-hoc calibration methods** (Temperature Scaling, Vector/Logistic
Scaling, Dirichlet, LTS, Meta-Cal). Four parts:

| Part | What | Dataset(s) | Metrics |
|---|---|---|---|
| **a** | In-distribution performance | BUP20 | mIoU + calibration (ECE/ACE/MECE/MACE/Brier/NLL) |
| **b** | Under covariate shift | BUP20 | mIoU + calibration + shift-degradation plots |
| **c** | Image- & pixel-level OoD detection | GrowliFlower-L, BUTom21 (tomato) | AUROC / FPR95 (image & pixel) |
| **d** | FRE vs ST-LoRA significance on a/b/c | — | bootstrap CI + paired-t + Wilcoxon |

Models: ST-LoRA = the deployed `final_model` (rank 32, full augmentation); FRE = `full_ft` (full
augmentation); 5 seeds `{42,123,456,789,1337}`, report mean ± std. The post-hoc calibrators are **fit at 320×180 and
evaluated at native resolution** without recalibration.

## Files by part

**Shared:** `pepper_common.py`, `pepper_stream_common.py`, `run_seed_calibration_stream.py`
(StreamBinMetrics), `calibrators.py` + `selectivecal_src/`, `calibration_shift_eval.py`.

**(a) In-distribution.**
`run_seed_calibration_pepper_stream.py --fit_size 320` (calibration methods) · `elora_evaluation_seeded.py`
(ST-LoRA) · `fullft_evaluation_seeded.py` (FRE) · `mcdropout_evaluation_seeded.py` + `mask2former_mcdropout_train_seeded.py`
· `mask2former_ddu_train_seeded.py` + `ddu_gmm_fit.py` (DDU). Aggregate: `aggregate_fit320_native.py`,
`aggregate_pepper.py`, `per_class_iou_calibrators.py`, `per_class_scores_calibrators.py`.
Jobs: `fit320_native.sh`, `run_calibration.sh`, `aggregate_fit320_native.sh`, `train_mcdropout_seeds.sh`,
`eval_mcdropout_seeds.sh`, `train_ddu_seeds.sh`, `ddu_gmm_fit.sh`, `train_full_seeds.sh`, `eval_full_seeds.sh`.

**(b) Covariate shift.**
`run_seed_shift_pepper_stream.py`, `calibration_shift_eval.py` (methods: lora / fullft / mcdropout / ddu).
Aggregate + plot: `aggregate_calibration_shift.py`, `aggregate_plot_shift_native.py`, `plot_calibration_shift.py`,
`plot_uq_brier_nll_shift.py`, `plot_shift_11_brier_nll.py`, `merge_plot_shift_11.py`.
Jobs: `run_shift_native.sh`, `run_uq_shift_allmethods.sh`, `calibration_shift_eval.sh`,
`aggregate_shift_native.sh`, `aggregate_calibration_shift.sh`, `plot_uq_brier_nll.sh`, `plot_shift_11_bnll.sh`, `merge_shift_11.sh`.

**(c) OoD (image + pixel).**
`run_seed_ood_pepper_imagelevel.py` (image), `run_seed_ood_pepper_stream.py` + `ood_eval_comprehensive_with_ddu.py`
(pixel, all methods incl. DDU), `*_tomato_pixel_ood.py` (FRE / MC-Dropout / DDU / ST-LoRA on tomato),
`pepper_to_tomato_pixel_ood.py`. Aggregate: `aggregate_imagelevel_ood_table.py`,
`aggregate_tomato_pixel_ood_all_methods.py`, `aggregate_pixel_ood_seeds.py`.
Jobs: `run_ood.sh`, `ood_eval_comprehensive_with_ddu.sh`, `ood_eval_comprehensive.sh`, `aggregate_ood_v2_fixed.sh`.

**(d) Significance (FRE vs ST-LoRA).**
`miou_paired_significance.py` (a), `calib_indomain_shift_significance_noholm.py` + `shift_paired_significance.py` +
`calib_shift_crosstab_significance.py` (a/b calibration), `imagelevel_ood_paired_significance.py` +
`ood_paired_significance_full.py` + `ood_full_crosstab_significance.py` (c).
Jobs of the same names.

## Setup

```bash
conda env create -f ../environment.yml && conda activate st_lora
$EDITOR env.sh     # set BUP20_DIR, GROWLI_DIR, BUTOM21_DIR
source env.sh      # exports ST_LORA_ROOT (SLURM inherits it)
```

## Run — SLURM

Jobs read code from `${ST_LORA_ROOT}/code` and are 5-seed arrays where applicable. After
`source env.sh`:

```bash
# (a) in-distribution
sbatch jobs/fit320_native.sh                 # post-hoc calibrators, fit@320 -> native
sbatch jobs/train_mcdropout_seeds.sh && sbatch jobs/eval_mcdropout_seeds.sh
sbatch jobs/train_ddu_seeds.sh && sbatch jobs/ddu_gmm_fit.sh
sbatch jobs/train_full_seeds.sh && sbatch jobs/eval_full_seeds.sh
sbatch jobs/aggregate_fit320_native.sh

# (b) covariate shift
sbatch jobs/run_shift_native.sh ; sbatch jobs/run_uq_shift_allmethods.sh
sbatch jobs/aggregate_calibration_shift.sh ; sbatch jobs/plot_uq_brier_nll.sh

# (c) OoD (GrowliFlower-L + BUTom21)
sbatch jobs/run_ood.sh ; sbatch jobs/ood_eval_comprehensive_with_ddu.sh
sbatch jobs/aggregate_ood_v2_fixed.sh

# (d) FRE vs ST-LoRA significance
sbatch jobs/miou_paired_significance.sh
sbatch jobs/calib_indomain_shift_significance_noholm.sh
sbatch jobs/imagelevel_ood_paired_significance.sh ; sbatch jobs/ood_paired_significance_full.sh
```

Each script is a plain `python code/<script>.py …` and runs without SLURM too — copy the invocation
from the matching `jobs/*.sh`.

## Notes

- **Significance:** FRE vs ST-LoRA reported with bootstrap CI + paired t-test + Wilcoxon.
- The image/pixel-level OoD uses the native-palette segmentation masks for GrowliFlower-L and the
  tomato (BUTom21) frames.
- Seeds `{42,123,456,789,1337}`.
