#!/bin/bash
#SBATCH --job-name=calib_plot
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

echo "Plotting calibration shift results..."

python "${SCRIPT_DIR}/plot_calibration_shift.py" \
    --summary_json ${RESULTS_DIR}/calibration_shift_summary.json \
    --out_dir      ${RESULTS_DIR}/figures

echo "Done."
