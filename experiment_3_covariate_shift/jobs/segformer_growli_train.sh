#!/bin/bash
#SBATCH --job-name=exp3_growli_tr
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

# Exp 3a — SegFormer-B2 on GrowliFlower-L, the four ensemble models (5 seeds each):
#   VARIANT=homogeneous   -> LR schedule only (--no_augmentation)
#   VARIANT=heterogeneous -> LR + augmentation (random rotation ~U(0,25deg) + hflip 0.5, image+mask)
#   METHOD=stlora|fre
# e.g.  METHOD=stlora VARIANT=heterogeneous sbatch jobs/segformer_growli_train.sh
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
CODE="$(cd "$(dirname "$0")/../code" && pwd)"
METHOD="${METHOD:?set METHOD=stlora or fre}"
VARIANT="${VARIANT:?set VARIANT=homogeneous or heterogeneous}"
PRE="nvidia/segformer-b2-finetuned-ade-512-512"
case "$VARIANT" in
  homogeneous)   AUG="--no_augmentation" ;;
  heterogeneous) AUG="" ;;                 # augmentation ON (rotation + hflip)
  *) echo "bad VARIANT=$VARIANT"; exit 2 ;;
esac
if [ "$METHOD" = "stlora" ]; then LR=2e-3; LRMIN=2e-5; LORA="--lora_r 8 --lora_alpha 8 --lora_dropout 0.1 --no_dora"
else LR=2e-4; LRMIN=2e-6; LORA=""; fi

SEEDS=(42 123 456 789 1337); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT="$ST_LORA_ROOT/results/exp3/growli/segformer_b2_${METHOD}_${VARIANT}"
mkdir -p "$OUT/logs"
module purge; module load CUDA/12.6.0 Miniforge3; source activate "$ENV_MAIN"
echo "=== SegFormer-B2 $METHOD $VARIANT GrowliFlower seed=$SEED ==="
python -m py_compile "$CODE/segformer_growli_train_seeded.py"
python "$CODE/segformer_growli_train_seeded.py" \
    --method "$METHOD" --pretrained "$PRE" $LORA $AUG \
    --root_dir "$GROWLI_DIR" --base_save_dir "$OUT" --log_dir "$OUT/logs" \
    --epochs 110 --batch_size 4 --lr "$LR" --lr_min "$LRMIN" \
    --cosine_T0 20 --snapshot_every 20 --seed "$SEED" --gpus 1 --num_workers 8 --precision bf16-mixed
echo "Finished $METHOD $VARIANT seed=$SEED $(date)"
