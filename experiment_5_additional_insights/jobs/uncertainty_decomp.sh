#!/bin/bash
#SBATCH --job-name=unc_decomp
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00

# Spatial uncertainty decomposition (total / aleatoric / epistemic) for 2 bup20 test frames,
# FRE vs ST-LoRA full-aug snapshot ensembles (M=4). Read-only inference in ssl env.
SD="${ST_LORA_ROOT}/code"
OUT="${ST_LORA_ROOT}/results/lora_paper/uncertainty_decomp"

module purge
module load CUDA/12.6.0 Miniforge3
source activate ssl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SD}:${PYTHONPATH}"
mkdir -p "${OUT}/logs"

echo "=== uncertainty decomposition (FRE vs ST-LoRA, M=4, seed 42) : env=$CONDA_PREFIX ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m py_compile "${SD}/uncertainty_decomp_maps.py" || { echo "py_compile FAILED"; exit 1; }
python -c "import calibration_shift_eval as c; print('cse OK; NUM_LABELS', c.NUM_LABELS)" \
    && echo "import cse OK" || { echo "cse import FAILED"; exit 1; }

python "${SD}/uncertainty_decomp_maps.py" --seed 42 --shot_ids 1 2 3 4
rc=$?
echo "Finished unc_decomp at $(date) exit=${rc}"
exit ${rc}
