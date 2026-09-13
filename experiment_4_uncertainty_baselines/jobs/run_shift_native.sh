#!/bin/bash
#SBATCH --job-name=pep_shift_native
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-4%2

# PEPPER post-hoc calibration UNDER DISTRIBUTION SHIFT, fit@320 -> eval @NATIVE.
# Fits the 6 calibrators once on clean cal(30)@320, then streams val(33) at native
# under the 7x5 corruption grid, recording float64 ECE/ACE/mIoU/acc per method.
# Env hooks:  SMOKE=1 -> 1 corruption x 1 severity (+clean);  MAX_IMAGES=N -> cap frames.

SD="${ST_LORA_ROOT}/code"
OUT_DIR="${ST_LORA_ROOT}/results/posthoc_calibration_pepper/fit320_native"

SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl

echo "=== seed $SEED : pepper shift fit@320 -> eval @native ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
mkdir -p "${OUT_DIR}/logs"

python -m py_compile "${SD}/run_seed_shift_pepper_stream.py" || { echo "py_compile FAILED"; exit 1; }

python "${SD}/run_seed_shift_pepper_stream.py" \
    --seed ${SEED} --out_dir ${OUT_DIR} --fit_size 320 \
    --epochs 40 --batch_size 20 --eval_split ${EVAL_SPLIT:-val} \
    ${SMOKE:+--smoke} ${MAX_IMAGES:+--max_images ${MAX_IMAGES}}
RC=$?

echo "Finished at $(date) with exit code ${RC}"
exit ${RC}
