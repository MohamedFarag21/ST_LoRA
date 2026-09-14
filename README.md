# ST-LoRA

**Single-Trajectory LoRA Ensemble for Uncertainty-Aware Agricultural Segmentation**

ST-LoRA fine-tunes large semantic-segmentation backbones parameter-efficiently and turns a *single*
cosine-restart training run into a **snapshot ensemble** — the periodic snapshots along one training
trajectory are ensembled at inference for better accuracy **and calibrated, uncertainty-aware
predictions**, at a fraction of the trainable parameters of full fine-tuning (FRE). We study it
across three architectures — **Mask2Former** (Swin-B), **SegFormer** (MiT), and **EoMT** (ViT-L) —
on agricultural datasets.

## Walkthrough

https://github.com/user-attachments/assets/a5b016d2-6d04-46b7-8ce3-e05ca809ef18

## What's in this repository

The study is released as self-contained experiments — each folder has its own `README`, `code/`,
`jobs/` (SLURM launchers, also runnable without SLURM), and an `env.sh` for paths.

| Folder | Experiment |
|---|---|
| [`experiment_1_single_model/`](experiment_1_single_model/) | **Single model** (last snapshot, no ensembling): ST-LoRA (rank 8) vs full fine-tuning (FRE), no augmentation — mIoU, all calibration metrics, and efficiency (trainable params, wall-clock, GPU energy, CO₂). Includes a SegFormer-B2/B4 arm on GrowliFlower-L. |
| [`experiment_2_hyperparameters/`](experiment_2_hyperparameters/) | **ST-LoRA hyperparameter ablation** (evaluated as the last-four-snapshot ensemble): rank, scaling factor α, dropout, and target modules — plus structural variants (encoder-only, encoder-LoRA + decoder full-FT, encoder+decoder) and an FFN-only vs FFN+attention efficiency comparison. |
| [`experiment_3_covariate_shift/`](experiment_3_covariate_shift/) | **Performance under covariate shift** (mIoU + ECE, last-four-snapshot ensemble): (3a) SegFormer-B2 / GrowliFlower-L augmentation importance — ST-LoRA & FRE, homogeneous (no aug) vs heterogeneous (rotation + hflip) ensembles; (3b) Mask2Former / BUP20 hyperparameter component sensitivity under shift. |
| [`experiment_4_uncertainty_baselines/`](experiment_4_uncertainty_baselines/) | **Uncertainty baselines**: ST-LoRA & FRE vs MC-Dropout, DDU, and post-hoc calibration methods — in-distribution, under covariate shift, and image/pixel-level OoD detection (GrowliFlower-L, BUTom21), with FRE-vs-ST-LoRA significance testing. |
| [`experiment_5_additional_insights/`](experiment_5_additional_insights/) | **Additional insights**: ensemble-member diversity vs LR schedule & augmentation, qualitative success/failure panels, per-class mIoU + calibration, and an ensemble-size ablation. |

More experiments will be added here.

## Architectures & datasets

**Architectures** (pretrained backbones pulled from the HuggingFace Hub):
- Mask2Former — `facebook/mask2former-swin-base-ade-semantic`
- SegFormer — `nvidia/segformer-b2-finetuned-ade-512-512` (and `-b4-…`)
- EoMT — `tue-mps/ade20k_semantic_eomt_large_512`

**Datasets:**
- **BUP20** (sweet pepper, 8 classes) — https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/b5d18108-53e1-46d1-873d-4230a72dfad7
- **GrowliFlower-L** (cauliflower; binary plant-vs-background here) — https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/cb328232-31f5-4b84-a929-8e1ee551d66a

## Environments

Two conda environments, identical except for the `transformers` version:

| File | conda name | transformers | Use for |
|---|---|---|---|
| [`environment.yml`](environment.yml) | `st_lora` | 4.52.2 | Mask2Former + SegFormer |
| [`environment_eomt.yml`](environment_eomt.yml) | `st_lora_eomt` | 5.14.1 | EoMT |

```bash
conda env create -f environment.yml
conda env create -f environment_eomt.yml      # only if you run EoMT
```
Core stack: Python 3.11, torch 2.7.0 (CUDA 12.6), peft 0.15.2, pytorch-lightning 2.5.1,
torchmetrics 1.7.1, torch_uncertainty 0.5.0. A full pip freeze is in
[`requirements_ssl_full.txt`](requirements_ssl_full.txt); a curated list in
[`requirements.txt`](requirements.txt).

## Quick start

```bash
# 1) create the environment(s)
conda env create -f environment.yml && conda activate st_lora
# 2) pick an experiment, set data paths, run
cd experiment_1_single_model
$EDITOR env.sh          # set BUP20_DIR (+ GROWLI_DIR for the GrowliFlower arm)
source env.sh
sbatch jobs/m2f_lora_r8_noaug_train.sh          # SLURM ...
# ... or run the equivalent `python` command directly (no scheduler) — see the experiment README.
```
Every experiment runs on a SLURM cluster (`sbatch jobs/*.sh`) **or** on a plain workstation with a
single GPU (the launchers are portable and each README gives the direct `python` commands). All
experiments use the 5-seed set `{42, 123, 456, 789, 1337}` and report mean ± std.

## Citation

The paper is under review; BibTeX will be added on acceptance.

## License

See [LICENSE](LICENSE).
