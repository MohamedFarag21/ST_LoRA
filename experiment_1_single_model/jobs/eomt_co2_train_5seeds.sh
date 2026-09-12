#!/bin/bash
#SBATCH --job-name=eomt_co2_tr
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --array=0-9%3

# EoMT CO2/energy: FRE full-FT vs ST-LoRA r8, 40 epochs, NO-aug, 5 seeds, CodeCarbon-instrumented.
# Matches the Mask2Former CO2 protocol (40-ep cost proxy). 10 tasks = 2 methods x 5 seeds.
#   task 0-4 : fre     seeds {42,123,456,789,1337}
#   task 5-9 : stlora  seeds {42,123,456,789,1337}
ROOT="${ST_LORA_ROOT:?export ST_LORA_ROOT to the repo root (see env.sh)}"
SD="$(cd "$(dirname "$0")/../code" && pwd)"
OUT="${ROOT}/results/lora_paper/co2_eomt"
mkdir -p "${OUT}/logs"

METHODS=(fre fre fre fre fre  stlora stlora stlora stlora stlora)
SEEDS=(42 123 456 789 1337  42 123 456 789 1337)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module purge; module load CUDA/12.6.0 Miniforge3
source activate "${ROOT}/envs/eomt_cc"
export HF_HOME="${ROOT}/hf_cache"; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
echo "=== EoMT CO2 ${METHOD} seed ${SEED} : env=$CONDA_PREFIX ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m py_compile "${SD}/eomt_co2_compare_train.py" || { echo "py_compile FAILED"; exit 1; }

python "${SD}/eomt_co2_compare_train.py" \
    --method ${METHOD} --seed ${SEED} --lora_r 8 \
    --epochs 40 --cosine_T0 20 --snapshot_every 20 \
    --batch_size 2 --num_workers 8 --sample_secs 3 \
    --save_root "${OUT}/ckpt_${METHOD}_seed${SEED}" \
    --out_dir   "${OUT}"
rc=$?
echo "Finished ${METHOD} seed ${SEED} at $(date) exit=${rc}"
exit ${rc}
