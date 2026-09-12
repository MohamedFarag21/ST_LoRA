#!/bin/bash
#SBATCH --job-name=exp2_ablate
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

# Experiment 2 — ST-LoRA hyperparameter ablation on BUP20, NO augmentation.
# One array task = one CATALOG CONFIG (index) for a fixed SEED. Submit per arch & seed, e.g.:
#     ARCH=m2f       SEED=42 sbatch --array=0-34 jobs/run_ablation.sh
#     ARCH=segformer SEED=42 sbatch --array=0-26 jobs/run_ablation.sh
#     ARCH=eomt      SEED=42 sbatch --array=0-26 jobs/run_ablation.sh
# (array upper bound = num configs-1; see configs/<arch>.json. Loop SEED over {42,123,456,789,1337}.)
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
CODE="$(cd "$(dirname "$0")/../code" && pwd)"
ARCH="${ARCH:?set ARCH=m2f|segformer|eomt}"
SEED="${SEED:-42}"
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array}"

case "$ARCH" in
  m2f)       ENV="$ENV_MAIN"; TRAIN=mask2former_lora_train_seeded_aug.py;  PRETRAINED="facebook/mask2former-swin-base-ade-semantic"; LR=2e-4; LRMIN=2e-6; PREC=32;          BS=4; EVAL=elora_evaluation_seeded.py ;;
  segformer) ENV="$ENV_MAIN"; TRAIN=segformer_lora_train_seeded.py;        PRETRAINED="nvidia/segformer-b2-finetuned-ade-512-512";   LR=2e-3; LRMIN=2e-5; PREC=bf16-mixed; BS=4; EVAL=segformer_lora_eval_seeded.py ;;
  eomt)      ENV="$ENV_EOMT"; TRAIN=eomt_lora_train_seeded.py;             PRETRAINED="tue-mps/ade20k_semantic_eomt_large_512";     LR=2e-4; LRMIN=2e-6; PREC=bf16-mixed; BS=2; EVAL=eomt_lora_ensemble_evaluate_seeded.py ;;
  *) echo "bad ARCH=$ARCH"; exit 2 ;;
esac

module purge; module load CUDA/12.6.0 Miniforge3; source activate "$ENV"
echo "=== EXP2 arch=$ARCH idx=$IDX seed=$SEED : $(python -c 'import torch;print(torch.__version__)') ==="

# Resolve collision-safe flags for this catalog entry (loads the model once).
eval "$(python "$CODE/prepare_run.py" --arch "$ARCH" --index "$IDX")"
echo "[config] $NAME  r=$LORA_R a=$LORA_ALPHA do=$LORA_DROPOUT mode=$DECODER_MODE fam=$FAMILY"

OUT="$ST_LORA_ROOT/results/exp2/$ARCH/$NAME"
mkdir -p "$OUT/logs"
MTS_FLAG=""; [ -n "${MODULES_TO_SAVE:-}" ] && MTS_FLAG="--modules_to_save $MODULES_TO_SAVE"
FF_FLAG="";  [ -n "${FULLFT_MODULES:-}" ] && FF_FLAG="--fullft_modules $FULLFT_MODULES"

COMMON="--seed $SEED --no_augmentation --epochs 110 --cosine_T0 20 --snapshot_every 20 \
        --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT \
        --target_modules $TARGET_MODULES $MTS_FLAG $FF_FLAG \
        --base_save_dir $OUT --log_dir $OUT/logs --gpus 1 --num_workers 8 --batch_size $BS"

python -m py_compile "$CODE/$TRAIN"
if [ "$ARCH" = "eomt" ]; then
    python "$CODE/$TRAIN" $COMMON --lr $LR --lr_min $LRMIN
else
    python "$CODE/$TRAIN" $COMMON --lr $LR --lr_min $LRMIN --precision $PREC --no_dora
fi

# ---- single-model eval (last snapshot, shot 5): mIoU + all calibration metrics ----
echo "=== EVAL $ARCH $NAME seed=$SEED shot 5 ==="
case "$ARCH" in
  m2f)       python "$CODE/$EVAL" --results_dir "$OUT" --seed "$SEED" --shot_ids 5 --split test \
                    --coco_file "$BUP20_COCO" --root_dir "$BUP20_DIR" ;;
  segformer) python "$CODE/$EVAL" --results_dir "$OUT" --config_name "r$LORA_R" --seed "$SEED" \
                    --shot_ids 5 --split test --pretrained "$PRETRAINED" \
                    --coco_file "$BUP20_COCO" --root_dir "$BUP20_DIR" ;;
  eomt)      python "$CODE/$EVAL" --base_save_dir "$OUT" --seed "$SEED" --shot_ids 5 \
                    --out_dir "$OUT/seed_$SEED" --split test ;;
esac
echo "=== done arch=$ARCH idx=$IDX ($NAME) seed=$SEED $(date) ==="
