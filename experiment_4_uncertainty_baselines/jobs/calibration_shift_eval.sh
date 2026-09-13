#!/bin/bash
#SBATCH --job-name=calib_shift
#SBATCH --account=ag_igg_roscher
#SBATCH --partition=sgpu_long
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# Array layout: 4 methods × 5 seeds = 20 jobs
#   task 0-4:   lora       seeds [42,123,456,789,1337]
#   task 5-9:   fullft     seeds [42,123,456,789,1337]
#   task 10-14: mcdropout  seeds [42,123,456,789,1337]
#   task 15-19: ddu        seeds [42,123,456,789,1337]
# ─────────────────────────────────────────────────────────────────────────────

module load CUDA/12.6.0 Miniforge3
source activate ssl

SCRIPT_DIR=${ST_LORA_ROOT}/code
OUT_DIR=${ST_LORA_ROOT}/results/lora_paper/calibration_shift

mkdir -p ${OUT_DIR}/logs

METHODS=(lora lora lora lora lora \
         fullft fullft fullft fullft fullft \
         mcdropout mcdropout mcdropout mcdropout mcdropout \
         ddu ddu ddu ddu ddu)
SEEDS=(42 123 456 789 1337 \
       42 123 456 789 1337 \
       42 123 456 789 1337 \
       42 123 456 789 1337)

METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "================================================================"
echo "  Job    : calib_shift  (${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
echo "  Node   : $(hostname)"
echo "  Method : ${METHOD}"
echo "  Seed   : ${SEED}"
echo "  Start  : $(date)"
echo "================================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "${SCRIPT_DIR}/calibration_shift_eval.py" \
    --method          ${METHOD}  \
    --seed            ${SEED}    \
    --out_dir         ${OUT_DIR} \
    --batch_size      4          \
    --num_workers     4          \
    --gpu             0          \
    --height          1280       \
    --width           720        \
    --n_bins          10         \
    --shot_ids        2 3 4 5    \
    --fullft_shot_ids 2 3 4 5  \
    --dropout_p       0.5       \
    --T               4

echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
