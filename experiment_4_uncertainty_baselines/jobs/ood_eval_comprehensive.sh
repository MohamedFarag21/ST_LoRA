#!/bin/bash
#SBATCH --job-name=m2f_ood_comp
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --array=0-2   # 0=lora, 1=fullft, 2=mcdropout

SCRIPT_DIR="${ST_LORA_ROOT}/code"
LORA_DIR="${ST_LORA_ROOT}/results/lora_paper/hparam_sweep"
FULLFT_DIR="${ST_LORA_ROOT}/results/lora_paper/full_ft"
MCDROP_DIR="${ST_LORA_ROOT}/results/lora_paper/mcdropout"
GROWLI_DIR="${ST_LORA_ROOT}/data/growliflower_l"
OUT_DIR="${ST_LORA_ROOT}/results/lora_paper/ood_v2"

METHODS=("lora" "fullft" "mcdropout")
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "================================================================"
echo "  Job : $SLURM_JOB_NAME ($SLURM_JOB_ID)  task $SLURM_ARRAY_TASK_ID"
echo "  Node: $SLURMD_NODENAME  |  Method: $METHOD"
echo "  Start: $(date)"
echo "================================================================"

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl

mkdir -p "${OUT_DIR}/logs"

python "${SCRIPT_DIR}/ood_eval_comprehensive.py" \
    --method               ${METHOD}             \
    --growliflower_dir     ${GROWLI_DIR}          \
    --out_dir              ${OUT_DIR}             \
    --lora_results_dir     ${LORA_DIR}            \
    --config_name          final_model            \
    --shot_ids             2 3 4 5                \
    --fullft_dir           ${FULLFT_DIR}          \
    --fullft_shot_ids      2 3 4 5              \
    --mcdrop_dir           ${MCDROP_DIR}          \
    --dropout_p            0.5                  \
    --T                    4                     \
    --seed                 1337                     \
    --batch_size           2                      \
    --num_workers          8                      \
    --gpu                  0                      \
    --height               1280                   \
    --width                720

EXIT_CODE=$?
echo "Finished at $(date) with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}