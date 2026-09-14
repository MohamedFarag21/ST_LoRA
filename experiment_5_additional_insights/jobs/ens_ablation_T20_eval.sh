#!/bin/bash
#SBATCH --job-name=ensT20_eval
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-2

# Ensemble-SIZE ablation eval (VARIANT T20, deployed cadence): cumulative ensembles
# k=1..10 for FRE + ST-LoRA on test93 @native.
SD="${ST_LORA_ROOT}/code"
BASE="${ST_LORA_ROOT}/results/lora_paper/ensemble_size_ablation_T20"
OUT="${BASE}/eval"
SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl
mkdir -p "${OUT}/logs"

python -m py_compile "${SD}/ens_size_ablation_eval.py" || { echo "py_compile FAILED"; exit 1; }
python -c "import torch, transformers, peft, skimage.draw, evaluate" || { echo "PREFLIGHT IMPORT FAILED"; exit 1; }

rc=0
for M in fre stlora; do
  echo "=== ${M} seed ${SEED} : k=1..10 (T20) ==="
  python "${SD}/ens_size_ablation_eval.py" --method ${M} --seed ${SEED} --maxk 10 \
      --base "${BASE}" --out_dir "${OUT}" \
      || { echo "FAILED ${M}"; rc=1; break; }
done
echo "Finished T20 eval seed ${SEED} at $(date) exit=${rc}"
exit ${rc}
