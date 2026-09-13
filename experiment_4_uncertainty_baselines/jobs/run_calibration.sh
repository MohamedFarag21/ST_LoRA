#!/bin/bash
#SBATCH --job-name=pep_cal
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-4

SCRIPT_DIR="${ST_LORA_ROOT}/code"
SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MAX_ARG=""; if [ -n "${MAX_IMAGES}" ]; then MAX_ARG="--max_images ${MAX_IMAGES}"; fi

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl

echo "=== pepper phase-1 calibration seed=$SEED  MAX_IMAGES=${MAX_IMAGES:-<full>} ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
cd "${SCRIPT_DIR}"
python "${SCRIPT_DIR}/run_seed_calibration_pepper.py" \
    --seed ${SEED} --cal_size 64 --epochs 40 --batch_size 20 ${MAX_ARG}
echo "Finished at $(date) with exit code $?"
