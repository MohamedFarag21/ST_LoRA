#!/bin/bash
#SBATCH --job-name=pep320_native
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=0-4

# PEPPER (in-domain, 8-class): fit calibrators @320x180 -> evaluate at NATIVE without
# recalibrating. Runs BOTH halves per seed:
#   1) calibration  (ECE/ACE/mIoU/acc) on val33 + test93   [float64 StreamBinMetrics]
#   2) pixel-OoD    (AUROC/AUPR/FPR95/sIoU/PPV/MeanF1) on FULL tomato(4536) +
#      growliflower(1970)                                  [float64 OODHist, smax=ln8]
# MAX_IMAGES (env) runs a smoke subset. STAGE (env) = ece|ood|both (default both).

SD="${ST_LORA_ROOT}/code"
OUT_DIR="${ST_LORA_ROOT}/results/posthoc_calibration_pepper/fit320_native"

SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
STAGE=${STAGE:-both}

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl

echo "=== seed $SEED : pepper fit@320 -> eval @native (stage=$STAGE) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
mkdir -p "${OUT_DIR}/logs"

python -m py_compile "${SD}/run_seed_calibration_pepper_stream.py" \
                     "${SD}/run_seed_ood_pepper_stream.py" || { echo "py_compile FAILED"; exit 1; }

if [ "$STAGE" = "ece" ] || [ "$STAGE" = "both" ]; then
  echo "--- stage 1: calibration @native ---"
  python "${SD}/run_seed_calibration_pepper_stream.py" \
      --seed ${SEED} --out_dir ${OUT_DIR} --fit_size 320 \
      --epochs 40 --batch_size 20 \
      ${MAX_IMAGES:+--max_images ${MAX_IMAGES}} || exit 1
fi

if [ "$STAGE" = "ood" ] || [ "$STAGE" = "both" ]; then
  echo "--- stage 2: pixel-OoD @native ---"
  python "${SD}/run_seed_ood_pepper_stream.py" \
      --seed ${SEED} --out_dir ${OUT_DIR} --fit_size 320 \
      --epochs 40 --batch_size 20 \
      ${MAX_IMAGES:+--max_images ${MAX_IMAGES}} || exit 1
fi

echo "Finished at $(date) with exit code $?"
