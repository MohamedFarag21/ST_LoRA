#!/bin/bash
#SBATCH --job-name=ddu_train
#SBATCH --account=ag_igg_roscher
#SBATCH --partition=sgpu_long
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
module load CUDA/12.6.0 Miniforge3
source activate ssl

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR=${ST_LORA_ROOT}/code
RESULTS_DIR=${ST_LORA_ROOT}/results/lora_paper/ddu

mkdir -p ${RESULTS_DIR}/logs

# ─────────────────────────────────────────────────────────────────────────────
# Seed mapping  (array index → seed)
# ─────────────────────────────────────────────────────────────────────────────
SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "================================================================"
echo "  Job  : ddu_train  (${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
echo "  Node : $(hostname)"
echo "  Seed : ${SEED}"
echo "  Start: $(date)"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# A100 tuning
# ─────────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
python "${SCRIPT_DIR}/mask2former_ddu_train_seeded.py" \
    --seed             ${SEED}              \
    --base_save_dir    ${RESULTS_DIR}       \
    --log_dir          ${RESULTS_DIR}/logs  \
    --epochs           110                  \
    --batch_size       4                    \
    --lr               2e-4                 \
    --lr_min           2e-8                 \
    --cosine_T0        20                   \
    --gpus             4                    \
    --num_workers      8                    \
    --precision        "bf16-mixed"

echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
