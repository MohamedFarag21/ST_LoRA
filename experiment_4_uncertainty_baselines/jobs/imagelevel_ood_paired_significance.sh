#!/bin/bash
#SBATCH --job-name=imgood_sig
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
python -m py_compile "${E}/imagelevel_ood_paired_significance.py" || { echo "py_compile FAILED"; exit 1; }
python "${E}/imagelevel_ood_paired_significance.py"
echo "done exit=$?"
