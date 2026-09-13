#!/bin/bash
#SBATCH --job-name=calib_agg
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
RESULTS_DIR=${ST_LORA_ROOT}/results/lora_paper/calibration_shift

echo "Aggregating calibration shift results..."

python "${SCRIPT_DIR}/aggregate_calibration_shift.py" \
    --results_dir ${RESULTS_DIR}

echo "Done."
