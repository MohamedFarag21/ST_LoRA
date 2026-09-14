#!/bin/bash
#SBATCH --job-name=per_class_iou
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-4%3

# Per-class IoU on bup20 pepper test93 @native (clean) for all 4 UQ methods.
# Uses the FIXED fresh-base LoRA loader (via run_uq_shift_native_stream.build_get_model).

SD="${ST_LORA_ROOT}/code"
OUT="${ST_LORA_ROOT}/results/lora_paper/calibration_shift/per_class_iou"
SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl
mkdir -p "${OUT}/logs"

python -m py_compile "${SD}/per_class_iou.py" || { echo "py_compile FAILED"; exit 1; }
for M in fullft lora mcdropout ddu; do
  echo "=== ${M} seed ${SEED} ==="
  python "${SD}/per_class_iou.py" --method ${M} --seed ${SEED} --out_dir ${OUT} || { echo "FAILED ${M}"; exit 1; }
done
echo "Finished at $(date)"
