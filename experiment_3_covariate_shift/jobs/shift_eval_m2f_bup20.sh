#!/bin/bash
#SBATCH --job-name=exp3b_shift
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --account=ag_igg_roscher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=0-34

# Exp 3b — Mask2Former/BUP20 component sensitivity under shift. One array task = one Exp-2 catalog
# config (index into configs/m2f.json; 35 configs -> --array=0-34), for a fixed SEED. Evaluates the
# config's last-4-snapshot ensemble under the 19 BUP20 shift conditions (mIoU + ECE).
# Reads trained checkpoints from EXP2_M2F_DIR (set in env.sh; Exp-2 output or Exp-3 retrain output).
#   SEED=42 sbatch --array=0-34 jobs/shift_eval_m2f_bup20.sh   (loop SEED over the 5 seeds)
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
CODE="$(cd "$(dirname "$0")/../code" && pwd)"
SEED="${SEED:-42}"
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array}"

module purge; module load CUDA/12.6.0 Miniforge3; source activate "$ENV_MAIN"
echo "=== EXP3b shift-eval m2f cfg-idx=$IDX seed=$SEED (reading $EXP2_M2F_DIR) ==="
python -m py_compile "$CODE/shift_eval_m2f_bup20.py"
python "$CODE/shift_eval_m2f_bup20.py" \
    --index "$IDX" --seed "$SEED" \
    --results_dir "$EXP2_M2F_DIR" \
    --catalog "$(cd "$(dirname "$0")/.." && pwd)/configs/m2f.json" \
    --coco_file "$BUP20_COCO" --root_dir "$BUP20_DIR" \
    --shots 2 3 4 5 --batch_size 2 --num_workers 8
echo "Finished 3b shift-eval idx=$IDX seed=$SEED $(date)"
