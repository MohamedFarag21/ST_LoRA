#!/bin/bash
#SBATCH --job-name=ddu_gmm
#SBATCH --account=ag_igg_roscher
#SBATCH --partition=sgpu_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

module load CUDA/12.6.0 Miniforge3
source activate ssl

SCRIPT_DIR=${ST_LORA_ROOT}/code
RESULTS_DIR=${ST_LORA_ROOT}/results/lora_paper/ddu

SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "================================================================"
echo "  Job  : ddu_gmm  (${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
echo "  Node : $(hostname)"
echo "  Seed : ${SEED}"
echo "  Start: $(date)"
echo "================================================================"

python "${SCRIPT_DIR}/ddu_gmm_fit.py" \
    --seed                  ${SEED}        \
    --results_dir           ${RESULTS_DIR} \
    --feature_type          both           \
    --min_overlap           0.1            \
    --max_pixels_per_class  300            \
    --batch_size            2              \
    --num_workers           4              \
    --gpu                   0              \
    --height                1280           \
    --width                 720

echo "================================================================"
echo "  Done: $(date)"
echo "================================================================"
