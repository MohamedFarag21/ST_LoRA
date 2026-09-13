#!/bin/bash
#SBATCH --job-name=ddu_agg
#SBATCH --account=ag_igg_roscher
#SBATCH --partition=sgpu_short
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=0:15:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

module load Miniforge3
source activate ssl

SCRIPT_DIR=${ST_LORA_ROOT}/code
RESULTS_DIR=${ST_LORA_ROOT}/results/lora_paper/ddu

echo "Aggregating DDU results..."

python "${SCRIPT_DIR}/aggregate_ddu.py" \
    --results_dir ${RESULTS_DIR} \
    --split       test

echo "Done."
