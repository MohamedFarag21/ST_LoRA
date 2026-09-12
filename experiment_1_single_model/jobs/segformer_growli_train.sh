#!/bin/bash
#SBATCH --job-name=segf_growli_tr
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --account=ag_igg_roscher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --array=0-4

# SegFormer B2/B4 on GrowliFlower-L (binary plant vs bg), NO augmentation, 5 seeds.
# Experiment 1: single model, ST-LoRA r8 vs FRE.  Select arm with env vars:
#     BACKBONE=b2|b4   METHOD=stlora|fre   sbatch jobs/segformer_growli_train.sh
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
# method-specific LR (matches the validated SegFormer recipe): LoRA 2e-3, full-FT 2e-4
if [ "$METHOD" = "stlora" ]; then
  LR=2e-3; LR_MIN=2e-5; LORA="--lora_r 8 --lora_alpha 8 --lora_dropout 0.1 --no_dora"
else
  LR=2e-4; LR_MIN=2e-6; LORA=""
fi

SEEDS=(42 123 456 789 1337); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT="${ST_LORA_ROOT}/results/growli/segformer_${BACKBONE}_${METHOD}"
mkdir -p "$OUT/logs"

module purge; module load CUDA/12.6.0 Miniforge3; source activate "$ENV_MAIN"
echo "=== SegFormer-$BACKBONE $METHOD GrowliFlower seed=$SEED : $(python -c 'import torch;print(torch.__version__)') ==="
nvidia-smi --query-gpu=name --format=csv,noheader
python -m py_compile "$CODE/segformer_growli_train_seeded.py" || { echo "py_compile FAILED"; exit 1; }

python "$CODE/segformer_growli_train_seeded.py" \
    --method "$METHOD" --pretrained "$PRE" $LORA \
    --root_dir "$GROWLI_DIR" --base_save_dir "$OUT" --log_dir "$OUT/logs" \
    --epochs 110 --batch_size 4 --lr "$LR" --lr_min "$LR_MIN" \
    --cosine_T0 20 --snapshot_every 20 --no_augmentation \
    --seed "$SEED" --gpus 1 --num_workers 8 --precision bf16-mixed
echo "Finished SegFormer-$BACKBONE $METHOD seed=$SEED at $(date) exit=$?"
