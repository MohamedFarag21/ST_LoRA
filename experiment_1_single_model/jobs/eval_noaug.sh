#!/bin/bash
#SBATCH --job-name=eval_noaug
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-4

# Eval the no-aug Exp2 runs: SINGLE (shot 5) + ENS4 (shots 2 3 4 5) -> mIoU/ECE/ACE(/MECE/MACE).
# SET selects which of the 4 sets. Renames eval_test.json -> eval_test_single.json / _ens4.json.
SET="${SET:?set SET=m2ffre|m2flora|segfre|seglora}"
ROOT="${ST_LORA_ROOT:?export ST_LORA_ROOT to the repo root (see env.sh)}"
E="$(cd "$(dirname "$0")/../code" && pwd)"
SEEDS=(42 123 456 789 1337); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
mkdir -p "${ROOT}/results/lora_paper/eval_noaug_logs"

case "$SET" in
  m2ffre)  SCRIPT=fullft_evaluation_seeded.py;      RES="${ROOT}/results/lora_paper/m2f_fre_noaug";        EXTRA="" ;;
  m2flora) SCRIPT=elora_evaluation_seeded.py;       RES="${ROOT}/results/lora_paper/m2f_lora_r8_noaug";    EXTRA="" ;;
  segfre)  SCRIPT=segformer_fullft_eval_seeded.py;  RES="${ROOT}/results/lora_paper/segformer_fullft_noaug"; EXTRA="" ;;
  seglora) SCRIPT=segformer_lora_eval_seeded.py;    RES="${ROOT}/results/lora_paper/segformer_lora_r8_noaug"; EXTRA="--config_name final_model" ;;
  segffn)  SCRIPT=segformer_lora_eval_seeded.py;    RES="${ROOT}/results/lora_paper/segformer_lora_ffn_r8_noaug"; EXTRA="--config_name final_model_ffn" ;;
  *) echo "bad SET=$SET"; exit 2 ;;
esac

module purge; module load CUDA/12.6.0 Miniforge3; source activate ssl
echo "=== EVAL ${SET} seed ${SEED} : $(python -c 'import torch;print(torch.__version__)') ==="
nvidia-smi --query-gpu=name --format=csv,noheader
python -m py_compile "${E}/${SCRIPT}" || { echo "py_compile FAILED"; exit 1; }

rename_eval () {  # $1 = suffix (single|ens4)
  EV=$(find "${RES}" -path "*seed_${SEED}*/eval_test.json" 2>/dev/null | head -1)
  if [ -n "$EV" ]; then mv "$EV" "$(dirname "$EV")/eval_test_$1.json"; echo "wrote $(dirname "$EV")/eval_test_$1.json"; else echo "MISSING eval_test.json for $1"; fi
}

echo "--- SINGLE (shot 5) ---"
python "${E}/${SCRIPT}" --results_dir "${RES}" --seed ${SEED} --shot_ids 5 --split test ${EXTRA}
rename_eval single

echo "--- ENS4 (shots 2 3 4 5) ---"
python "${E}/${SCRIPT}" --results_dir "${RES}" --seed ${SEED} --shot_ids 2 3 4 5 --split test ${EXTRA}
rename_eval ens4

echo "Finished ${SET} seed ${SEED} at $(date)"
