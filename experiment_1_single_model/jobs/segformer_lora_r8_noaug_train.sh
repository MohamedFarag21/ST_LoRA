#!/bin/bash
#SBATCH --job-name=segf_lora8_noaug
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-4

# Exp2 no-aug: SegFormer ST-LoRA rank 8, NO augmentation, cosine warm-restart (default), 5 seeds.
# Same as deployed SegFormer LoRA (final_model, r8, lr 2e-3) but aug OFF.
ROOT="${ST_LORA_ROOT:?export ST_LORA_ROOT to the repo root (see env.sh)}"
SD="$(cd "$(dirname "$0")/../code" && pwd)"
OUT="${ROOT}/results/lora_paper/segformer_lora_r8_noaug"
SEEDS=(42 123 456 789 1337); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
mkdir -p "${OUT}/final_model/seed_${SEED}" "${OUT}/logs"

module purge; module load CUDA/12.6.0 Miniforge3; source activate ssl
echo "=== SegFormer ST-LoRA r8 no-aug seed=${SEED} ==="; nvidia-smi --query-gpu=name --format=csv,noheader
python -m py_compile "${SD}/segformer_lora_train_seeded.py" || { echo "py_compile FAILED"; exit 1; }
python "${SD}/segformer_lora_train_seeded.py" \
    --seed ${SEED} --config_name final_model \
    --base_save_dir "${OUT}" --log_dir "${OUT}/logs" \
    --epochs 110 --batch_size 4 \
    --lora_r 8 --lora_alpha 8 --lora_dropout 0.1 \
    --lr 2e-3 --lr_min 2e-5 --cosine_T0 20 --snapshot_every 20 \
    --target_modules query key value dense \
    --gpus 1 --num_workers 8 --precision bf16-mixed --no_augmentation
rc=$?; echo "Finished SegFormer ST-LoRA r8 no-aug seed=${SEED} at $(date) exit=${rc}"; exit ${rc}
