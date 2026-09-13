#!/bin/bash
#SBATCH --job-name=uq_shift_bnll
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=0-4%3

# Re-measure ALL 4 UQ methods (ddu/fullft/mcdropout/lora) over the full shift grid on
# test93 @native with float64 StreamBinMetrics ECE PLUS proper scoring rules Brier + NLL
# (no binning -> immune to the float32 ECE artifact). Fixed fresh-base LoRA loader.
# One array task per seed loops the 4 methods -> only 3 GPUs in flight at once.
# Overwrites the ECE-only native_float64/<method>_seed<seed>_shift_native.json in place
# (deterministic forwards -> ECE/mIoU identical, now with Brier/NLL added).

SD="${ST_LORA_ROOT}/code"
OUT="${ST_LORA_ROOT}/results/lora_paper/calibration_shift/native_float64"

SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl

echo "=== UQ shift @native float64 + Brier/NLL : seed=$SEED ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
mkdir -p "${OUT}/logs"

python -m py_compile "${SD}/run_uq_shift_native_stream.py" || { echo "py_compile FAILED"; exit 1; }

for M in ddu fullft mcdropout lora; do
  echo "########## method=${M} seed=${SEED} ##########"
  python "${SD}/run_uq_shift_native_stream.py" \
      --method ${M} --seed ${SEED} --out_dir ${OUT} \
      --batch_size 4 --num_workers 8 ${SMOKE:+--smoke} || { echo "FAILED ${M} seed ${SEED}"; exit 1; }
done
echo "Finished all methods for seed ${SEED} at $(date)"
