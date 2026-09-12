#!/bin/bash
#SBATCH --job-name=segf_growli_ev
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --account=ag_igg_roscher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --array=0-4

# SINGLE-model eval (last snapshot, shot 5) on GrowliFlower-L Test: mIoU + all calibration metrics.
#     BACKBONE=b2|b4   METHOD=stlora|fre   sbatch jobs/segformer_growli_eval.sh
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
CODE="$(cd "$(dirname "$0")/../code" && pwd)"

BACKBONE="${BACKBONE:?set BACKBONE=b2 or b4}"
METHOD="${METHOD:?set METHOD=stlora or fre}"
case "$BACKBONE" in
  b2) PRE="nvidia/segformer-b2-finetuned-ade-512-512" ;;
  b4) PRE="nvidia/segformer-b4-finetuned-ade-512-512" ;;
  *) echo "bad BACKBONE=$BACKBONE"; exit 2 ;;
esac
SHOT="${SHOT:-5}"
SEEDS=(42 123 456 789 1337); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RES="${ST_LORA_ROOT}/results/growli/segformer_${BACKBONE}_${METHOD}"

module purge; module load CUDA/12.6.0 Miniforge3; source activate "$ENV_MAIN"
echo "=== EVAL SegFormer-$BACKBONE $METHOD GrowliFlower seed=$SEED shot=$SHOT ==="
nvidia-smi --query-gpu=name --format=csv,noheader
python -m py_compile "$CODE/segformer_growli_eval_seeded.py" || { echo "py_compile FAILED"; exit 1; }

python "$CODE/segformer_growli_eval_seeded.py" \
    --method "$METHOD" --pretrained "$PRE" --shot "$SHOT" \
    --root_dir "$GROWLI_DIR" --results_dir "$RES" \
    --seed "$SEED" --split test --batch_size 4 --num_workers 8
echo "Finished eval SegFormer-$BACKBONE $METHOD seed=$SEED at $(date) exit=$?"
