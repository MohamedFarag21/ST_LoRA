#!/bin/bash
#SBATCH --job-name=exp3a_shift
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --account=ag_igg_roscher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-4

# Exp 3a — evaluate a trained SegFormer-B2/GrowliFlower model under the 20-level shift grid
# (last-4-snapshot ensemble; mIoU + ECE per condition).
#   METHOD=stlora|fre  VARIANT=homogeneous|heterogeneous  sbatch jobs/shift_eval_segformer_growli.sh
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
CODE="$(cd "$(dirname "$0")/../code" && pwd)"
METHOD="${METHOD:?set METHOD=stlora or fre}"
VARIANT="${VARIANT:?set VARIANT=homogeneous or heterogeneous}"
SEEDS=(42 123 456 789 1337); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RES="$ST_LORA_ROOT/results/exp3/growli/segformer_b2_${METHOD}_${VARIANT}"

module purge; module load CUDA/12.6.0 Miniforge3; source activate "$ENV_MAIN"
echo "=== EXP3a shift-eval $METHOD $VARIANT seed=$SEED ==="
python -m py_compile "$CODE/shift_eval_segformer_growli.py"
python "$CODE/shift_eval_segformer_growli.py" \
    --method "$METHOD" --variant "$VARIANT" --seed "$SEED" \
    --results_dir "$RES" --root_dir "$GROWLI_DIR" --shots 2 3 4 5 \
    --batch_size 8 --num_workers 8
echo "Finished 3a shift-eval $METHOD $VARIANT seed=$SEED $(date)"
