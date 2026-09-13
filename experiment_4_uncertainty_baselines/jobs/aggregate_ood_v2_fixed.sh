#!/bin/bash
#SBATCH --job-name=ood_agg
#SBATCH --partition=intelsr_short
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

module purge
module load Miniforge3
source activate ssl

python ${ST_LORA_ROOT}/code/aggregate_ood_v2_fixed.py
