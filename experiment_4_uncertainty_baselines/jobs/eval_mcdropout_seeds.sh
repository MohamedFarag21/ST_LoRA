#!/bin/bash
#SBATCH --job-name=m2f_mcdrop_eval
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-4        # 5 seeds, one job per seed
# NOTE: Submit with --dependency=afterok:<TRAIN_JOB_ID>

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="${ST_LORA_ROOT}/code"
RESULTS_DIR="${ST_LORA_ROOT}/results/lora_paper/mcdropout"

# ─────────────────────────────────────────────────────────────────────────────
# Seed mapping: array task ID → seed
# ─────────────────────────────────────────────────────────────────────────────
SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "================================================================"
echo "  Job name   : $SLURM_JOB_NAME"
echo "  Job ID     : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "  Node       : $SLURMD_NODENAME"
echo "  Partition  : $SLURM_JOB_PARTITION"
echo "  Seed       : $SEED"
echo "  Start time : $(date)"
echo "================================================================"

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl

echo "Python  : $(which python)"
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo ""

echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

mkdir -p "${RESULTS_DIR}/logs"

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo "--- MC Dropout eval: seed=${SEED} ---"

python "${SCRIPT_DIR}/mcdropout_evaluation_seeded.py" \
    --seed         ${SEED}          \
    --results_dir  ${RESULTS_DIR}   \
    --dropout_p    0.5             \
    --T            4               \
    --split        test             \
    --batch_size   4                \
    --num_workers  8                \
    --gpu          0                \
    --height       1280             \
    --width        720              \
    --n_bins       10

EXIT_CODE=$?
echo ""
echo "Evaluation finished at $(date) with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
