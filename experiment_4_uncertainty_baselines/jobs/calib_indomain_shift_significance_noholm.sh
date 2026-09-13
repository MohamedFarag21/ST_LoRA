#!/bin/bash
#SBATCH --job-name=calib_sig_noholm
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
ROOT="${ST_LORA_ROOT}"
E="${ROOT}/code"
module purge; module load Miniforge3; source activate ssl
python -m py_compile "${E}/calib_indomain_shift_significance_noholm.py" || { echo "py_compile FAILED"; exit 1; }
python "${E}/calib_indomain_shift_significance_noholm.py"
echo "done exit=$?"
