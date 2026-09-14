#!/bin/bash
#SBATCH --job-name=m2fdiv_score
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --array=0-19

# Score ensemble-member diversity for the 4 m2f_div arms x 5 seeds (M=4, shots 1-4).
ROOT="${ST_LORA_ROOT}"
E="${ROOT}/code"
DIV="${ROOT}/results/lora_paper/m2f_div"
SEEDS=(42 123 456 789 1337)
ARMS=(fullft_noaug stlora_noaug fullft_constlr stlora_constlr)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID/5))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID%5))]}
OUT="${DIV}/diversity_scores/${ARM}"
mkdir -p "${OUT}" "${DIV}/diversity_scores/logs"

module purge; module load CUDA/12.6.0 Miniforge3; source activate ssl
echo "=== DIVERSITY arm=${ARM} seed=${SEED} : $(python -c 'import torch;print(torch.__version__)') ==="
python -m py_compile "${E}/per_class_diversity_ensemble.py" || { echo "py_compile FAILED"; exit 1; }

case "$ARM" in
  fullft_noaug|fullft_constlr)
     python "${E}/per_class_diversity_ensemble.py" --method fullft --seed ${SEED} \
        --shot_ids 1 2 3 4 --fullft_dir "${DIV}/${ARM}" --out_dir "${OUT}" ;;
  stlora_noaug|stlora_constlr)
     python "${E}/per_class_diversity_ensemble.py" --method lora --seed ${SEED} \
        --shot_ids 1 2 3 4 --lora_dir "${DIV}" --config_name "${ARM}" --out_dir "${OUT}" ;;
  *) echo "bad ARM=$ARM"; exit 2 ;;
esac
echo "arm=${ARM} seed=${SEED} exit=$? at $(date)"
