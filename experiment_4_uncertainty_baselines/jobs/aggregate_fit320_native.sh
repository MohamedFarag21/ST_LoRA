#!/bin/bash
#SBATCH --job-name=agg_pepnat
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00

# Synthetic correctness proofs for the binary->8-class generalization of
# StreamBinMetrics (mIoU) and OODHist (entropy bound). Includes REGRESSION checks
# that the binary tomato defaults are bit-identical. CPU-only, no dataset reads.

SD="${ST_LORA_ROOT}/code"
mkdir -p "${ST_LORA_ROOT}/results/posthoc_calibration_pepper/fit320_native/logs"
module purge
module load Miniforge3
source activate ssl
python -m py_compile "${SD}/aggregate_fit320_native.py" || { echo "py_compile FAILED"; exit 1; }
python -u "${SD}/aggregate_fit320_native.py"
echo "done $(date) exit=$?"
