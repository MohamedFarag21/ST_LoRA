#!/bin/bash
#SBATCH --job-name=m2fdiv_full
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00:00
#SBATCH --array=0-4

# Mask2Former FULL-FT diversity arm. Matches the deployed full_ft (train_full_seeds.sh)
# EXACTLY (4xGPU bs4 bf16-mixed lr2e-4/lr_min2e-8 cosine_T0 20 snap 20 110ep) except the
# ONE ablated factor selected by $ARM:
#   ARM=constlr : full aug + single flat LR (no cosine)   [--constant_lr]
#   ARM=noaug   : NO aug + cosine schedule                [--no_augmentation]
ARM="${ARM:?set ARM=constlr or ARM=noaug}"
ROOT="${ST_LORA_ROOT}"
SD="${ROOT}/code"
OUT="${ROOT}/results/lora_paper/m2f_div/fullft_${ARM}"
mkdir -p "${OUT}/logs" "${ROOT}/results/lora_paper/m2f_div/logs"
SEEDS=(42 123 456 789 1337); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

case "$ARM" in
  constlr) EXTRA="--constant_lr" ;;
  noaug)   EXTRA="--no_augmentation" ;;
  *) echo "bad ARM=$ARM"; exit 2 ;;
esac

module purge; module load CUDA/12.6.0 Miniforge3; source activate ssl
echo "=== M2F full-FT diversity arm=${ARM} seed=${SEED} EXTRA='${EXTRA}' ==="
nvidia-smi --query-gpu=name --format=csv,noheader
python -m py_compile "${SD}/mask2former_full_train_seeded.py" || { echo "py_compile FAILED"; exit 1; }

python "${SD}/mask2former_full_train_seeded.py" \
    --seed ${SEED} --base_save_dir "${OUT}" --log_dir "${OUT}/logs" \
    --epochs 110 --batch_size 4 --lr 2e-4 --lr_min 2e-8 \
    --cosine_T0 20 --snapshot_every 20 --gpus 4 --num_workers 8 \
    --precision "bf16-mixed" ${EXTRA}
rc=$?; echo "Finished arm=${ARM} seed=${SEED} at $(date) exit=${rc}"; exit ${rc}
