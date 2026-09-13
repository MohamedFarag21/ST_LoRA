#!/bin/bash
#SBATCH --job-name=m2f_full_eval
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-4        # 5 seeds
# NOTE: Submit with --dependency=afterok:<TRAIN_JOB_ID>

SCRIPT_DIR="${ST_LORA_ROOT}/code"
RESULTS_DIR="${ST_LORA_ROOT}/results/lora_paper/full_ft"

SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "================================================================"
echo "  Job : $SLURM_JOB_NAME ($SLURM_JOB_ID)  task $SLURM_ARRAY_TASK_ID"
echo "  Node: $SLURMD_NODENAME  |  Seed: $SEED"
echo "  Start: $(date)"
echo "================================================================"

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl

mkdir -p "${RESULTS_DIR}/logs"

python "${SCRIPT_DIR}/fullft_evaluation_seeded.py" \
    --seed         ${SEED}          \
    --results_dir  ${RESULTS_DIR}   \
    --shot_ids     2 3 4 5        \
    --split        test             \
    --batch_size   4                \
    --num_workers  8                \
    --gpu          0                \
    --height       1280             \
    --width        720              \
    --n_bins       10

EXIT_CODE=$?
echo "Finished at $(date) with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
