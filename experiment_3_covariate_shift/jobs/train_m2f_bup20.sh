#!/bin/bash
#SBATCH --job-name=exp3b_train
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --account=ag_igg_roscher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --array=0-34

# Exp 3b — OPTIONAL: train the Mask2Former/BUP20 ablation configs from scratch (identical recipe to
# Experiment 2: NO augmentation, r/alpha/dropout/targets per configs/m2f.json). Only needed if you
# don't reuse the Experiment-2 checkpoints. Output -> results/exp3/m2f_bup20/<config>; then point
# EXP2_M2F_DIR at it for jobs/shift_eval_m2f_bup20.sh.
#   SEED=42 sbatch --array=0-34 jobs/train_m2f_bup20.sh   (loop SEED over the 5 seeds)
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
CODE="$(cd "$(dirname "$0")/../code" && pwd)"
SEED="${SEED:-42}"
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array}"

module purge; module load CUDA/12.6.0 Miniforge3; source activate "$ENV_MAIN"
eval "$(python "$CODE/prepare_run.py" --arch m2f --index "$IDX" \
        --catalog_dir "$(cd "$(dirname "$0")/.." && pwd)/configs")"
echo "[config] $NAME r=$LORA_R a=$LORA_ALPHA do=$LORA_DROPOUT mode=$DECODER_MODE"
OUT="$ST_LORA_ROOT/results/exp3/m2f_bup20/$NAME"; mkdir -p "$OUT/logs"
FF_FLAG=""; [ -n "${FULLFT_MODULES:-}" ] && FF_FLAG="--fullft_modules $FULLFT_MODULES"

python -m py_compile "$CODE/mask2former_lora_train_seeded_aug.py"
python "$CODE/mask2former_lora_train_seeded_aug.py" \
    --seed "$SEED" --no_augmentation --epochs 110 --cosine_T0 20 --snapshot_every 20 \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --target_modules $TARGET_MODULES $FF_FLAG \
    --base_save_dir "$OUT" --log_dir "$OUT/logs" \
    --lr 2e-4 --lr_min 2e-6 --precision 32 --no_dora --gpus 1 --num_workers 8 --batch_size 4
echo "Finished 3b train idx=$IDX ($NAME) seed=$SEED $(date)"
