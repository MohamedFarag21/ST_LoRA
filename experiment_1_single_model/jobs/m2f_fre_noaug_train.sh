#!/bin/bash
#SBATCH --job-name=m2f_fre_noaug
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --array=0-4

# Exp2 no-aug: Mask2Former FRE (full-FT), NO augmentation, cosine warm-restart (default),
# 5 seeds. Single GPU (avoid the multi-GPU congestion). Matches m2f_div FRE config but aug OFF.
ROOT="${ST_LORA_ROOT:?export ST_LORA_ROOT to the repo root (see env.sh)}"
SD="$(cd "$(dirname "$0")/../code" && pwd)"
OUT="${ROOT}/results/lora_paper/m2f_fre_noaug"
SEEDS=(42 123 456 789 1337); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
mkdir -p "${OUT}/logs"

module purge; module load CUDA/12.6.0 Miniforge3; source activate ssl
echo "=== M2F FRE no-aug seed=${SEED} ==="; nvidia-smi --query-gpu=name --format=csv,noheader
python -m py_compile "${SD}/mask2former_full_train_seeded.py" || { echo "py_compile FAILED"; exit 1; }
python "${SD}/mask2former_full_train_seeded.py" \
    --seed ${SEED} --base_save_dir "${OUT}" --log_dir "${OUT}/logs" \
    --epochs 110 --batch_size 4 --lr 2e-4 --lr_min 2e-8 \
    --cosine_T0 20 --snapshot_every 20 --gpus 1 --num_workers 8 \
    --precision "bf16-mixed" --no_augmentation
rc=$?; echo "Finished M2F FRE no-aug seed=${SEED} at $(date) exit=${rc}"; exit ${rc}
