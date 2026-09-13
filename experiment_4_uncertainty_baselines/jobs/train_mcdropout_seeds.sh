#!/bin/bash
#SBATCH --job-name=m2f_mcdrop_train
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0-5:00:00
#SBATCH --array=0-4        # 5 seeds

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
echo "  Job name       : $SLURM_JOB_NAME"
echo "  Job ID         : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "  Node           : $SLURMD_NODENAME"
echo "  Partition      : $SLURM_JOB_PARTITION"
echo "  Seed           : $SEED"
echo "  Start time     : $(date)"
echo "================================================================"

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl

echo "Python  : $(which python)"
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA    : $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/seed_${SEED}"

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
echo "--- Training: seed=${SEED} ---"

python "${SCRIPT_DIR}/mask2former_mcdropout_train_seeded.py" \
    --seed             ${SEED}            \
    --base_save_dir    ${RESULTS_DIR}     \
    --log_dir          ${RESULTS_DIR}/logs \
    --dropout_p        0.5               \
    --epochs           110                \
    --batch_size       4                  \
    --lr               2e-4               \
    --lr_min           2e-8               \
    --cosine_T0        20                 \
    --gpus             4                  \
    --num_workers      8                  \
    --precision        "bf16-mixed"

EXIT_CODE=$?
echo ""
echo "Training finished at $(date) with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
