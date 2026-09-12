#!/bin/bash
#SBATCH --job-name=eomtlora_flat
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --array=0-4%2

# EoMT ST-LoRA — CORRECTED optimizer to MATCH mask2former_lora_train_seeded.py EXACTLY:
# single flat Adam over all trainable params @ ONE lr (2e-4), lr_min 2e-6, no param-group
# split / no backbone_lr_mult (the old 0.1 starved the zero-init encoder adapters at 1e-5),
# no weight decay. r8/a8, no-aug, 110ep cosine T0=20, snapshot_every 20. -> eomt_lora_noaug_flatlr.
ROOT="${ST_LORA_ROOT:?export ST_LORA_ROOT to the repo root (see env.sh)}"
SD="$(cd "$(dirname "$0")/../code" && pwd)"
OUT="${ROOT}/results/lora_paper/eomt_lora_noaug_flatlr"

SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate eomt
export HF_HOME="${ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${OUT}/logs"

echo "=== EoMT+LoRA FLAT-LR (mask2former parity) seed ${SEED} : env=$CONDA_PREFIX ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m py_compile "${SD}/eomt_lora_train_seeded.py" || { echo "py_compile FAILED"; exit 1; }

python "${SD}/eomt_lora_train_seeded.py" \
    --seed ${SEED} --epochs 110 --snapshot_every 20 --no_augmentation \
    --lora_r 8 --lora_alpha 8 --lora_dropout 0.1 \
    --batch_size 2 --num_workers 8 --precision bf16-mixed \
    --lr 2e-4 --lr_min 2e-6 --cosine_T0 20 \
    --base_save_dir "${OUT}" \
    --log_dir "${OUT}/logs"
rc=$?
echo "Finished seed ${SEED} at $(date) exit=${rc}"
exit ${rc}
