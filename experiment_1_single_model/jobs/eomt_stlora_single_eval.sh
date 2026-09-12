#!/bin/bash
#SBATCH --job-name=eomt_stl_single
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --array=0-4

# Generate the ST-LoRA SINGLE-model eval (M=1, shot 5 = peak, matches FRE single=shot_5)
# for the flat-LR no-aug r8 EoMT, so Exp2 has a like-for-like single-model comparison.
ROOT="${ST_LORA_ROOT:?export ST_LORA_ROOT to the repo root (see env.sh)}"
E="$(cd "$(dirname "$0")/../code" && pwd)"
BASE="${ROOT}/results/lora_paper/eomt_lora_noaug_flatlr"
SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
mkdir -p "${BASE}/logs"

module purge; module load CUDA/12.6.0 Miniforge3; source activate eomt
echo "=== EoMT ST-LoRA single (shot 5) seed ${SEED} : torch $(python -c 'import torch;print(torch.__version__)') ==="
nvidia-smi --query-gpu=name --format=csv,noheader
python "${E}/eomt_lora_ensemble_evaluate_seeded.py" \
    --seed "${SEED}" --shot_ids 5 \
    --base_save_dir "${BASE}" --out_dir "${BASE}/seed_${SEED}" --split test
rc=$?; echo "Finished seed ${SEED} single at $(date) exit=${rc}"; exit ${rc}
