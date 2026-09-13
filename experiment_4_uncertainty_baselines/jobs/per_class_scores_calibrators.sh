#!/bin/bash
#SBATCH --job-name=pep_pcscore
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --array=0-4

# Per-class PROPER SCORES (Brier + NLL) for the 7 post-hoc calibration methods on PEPPER
# test93 @native (clean), fit@320 -> eval@native. Proper-score analog of the per_class_iou
# calibrator job; drops into the unified 11-method per-class Brier/NLL table.
# Env override: MAX_IMAGES (smoke cap), OUT_DIR.

SD="${ST_LORA_ROOT}/code"
OUT_DIR=${OUT_DIR:-${ST_LORA_ROOT}/results/posthoc_calibration_pepper/per_class_scores}

SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl
mkdir -p "${OUT_DIR}/logs"

echo "=== seed $SEED : pepper calibrator per-class Brier/NLL @native (out=$OUT_DIR) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python -m py_compile "${SD}/per_class_scores_calibrators.py" || { echo "py_compile FAILED"; exit 1; }
python -c "import torch, transformers, skimage.draw" || { echo "PREFLIGHT IMPORT FAILED"; exit 1; }

python "${SD}/per_class_scores_calibrators.py" \
    --seed ${SEED} --out_dir "${OUT_DIR}" --fit_size 320 --epochs 40 --batch_size 20 \
    ${MAX_IMAGES:+--max_images ${MAX_IMAGES}}
rc=$?
echo "Finished seed ${SEED} at $(date) exit=${rc}"
exit ${rc}
