#!/bin/bash
#SBATCH --job-name=stlora_panels
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:40:00

# ST-LoRA (final_model) success vs failure qualitative panels, seed 123, deployed shots 1-4.
SD="${ST_LORA_ROOT}/code"
OUT="${ST_LORA_ROOT}/results/lora_paper/qualitative_stlora"
SEED=${SEED:-123}

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl
mkdir -p "${OUT}/logs"

python -m py_compile "${SD}/stlora_success_failure_panels.py" || { echo "py_compile FAILED"; exit 1; }
python -c "import torch, transformers, peft, skimage.draw, matplotlib" || { echo "PREFLIGHT IMPORT FAILED"; exit 1; }

python "${SD}/stlora_success_failure_panels.py" --seed ${SEED} --shot_ids 1 2 3 4 --out_dir "${OUT}"
rc=$?
echo "Finished ST-LoRA panels seed ${SEED} at $(date) exit=${rc}"
exit ${rc}
