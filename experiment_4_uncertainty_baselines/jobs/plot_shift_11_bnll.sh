#!/bin/bash
#SBATCH --job-name=shift11_bnll
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00

ROOT="${ST_LORA_ROOT}"
CODE="${ROOT}/code"

module purge
module load Miniforge3
source activate ssl
mkdir -p "${ROOT}/results/posthoc_calibration_pepper/fit320_native/logs"

# also refresh the 4-method proper-scores figure/summary for good measure
python "${CODE}/plot_shift_11_brier_nll.py"
python "${CODE}/plot_uq_brier_nll_shift.py"
echo "Finished at $(date) exit=$?"
