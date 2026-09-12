#!/bin/bash
#SBATCH --job-name=eomtna_tr
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

# EoMT 5-seed full fine-tune with NO augmentation (--no_augmentation => JointAug OFF and
# CutMix OFF): the SCHEDULE-ONLY control arm. Same 110-ep cosine schedule / snapshot_every
# 20 as the other two arms, so member diversity differences isolate augmentation's effect
# vs the LR schedule alone. Writes to a NEW dir (eomt_full_noaug). Isolated eomt env.
SD="$(cd "$(dirname "$0")/../code" && pwd)"
ROOT="${ST_LORA_ROOT:?export ST_LORA_ROOT to the repo root (see env.sh)}"
OUT="${ROOT}/results/lora_paper/eomt_full_noaug"

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

echo "=== EoMT NO-AUG (schedule-only) train seed ${SEED} : env=$CONDA_PREFIX ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m py_compile "${SD}/eomt_full_train_seeded.py" || { echo "py_compile FAILED"; exit 1; }
# Preflight: guard against transient env failures (e.g. pytorch_lightning not importable
# on a bad node) so a broken env fails the job instead of silently satisfying afterok.
python -c "import pytorch_lightning, torch, transformers" || { echo "PREFLIGHT IMPORT FAILED"; exit 1; }

python "${SD}/eomt_full_train_seeded.py" \
    --seed ${SEED} --epochs 110 --snapshot_every 20 --no_augmentation \
    --batch_size 2 --num_workers 8 --precision bf16-mixed \
    --base_save_dir "${OUT}" \
    --log_dir "${OUT}/logs"
rc=$?
echo "Finished seed ${SEED} at $(date) exit=${rc}"
exit ${rc}
