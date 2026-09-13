#!/bin/bash
#SBATCH --job-name=pep_shift_merge11
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00

SD="${ST_LORA_ROOT}/code"

module purge
module load Miniforge3
source activate ssl

python -m py_compile "${SD}/merge_plot_shift_11.py" || { echo "py_compile FAILED"; exit 1; }
python "${SD}/merge_plot_shift_11.py"
echo "Finished at $(date) with exit code $?"
