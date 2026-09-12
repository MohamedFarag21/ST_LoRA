#!/bin/bash
#SBATCH --job-name=aggco2eomt
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00

ROOT="${ST_LORA_ROOT:?export ST_LORA_ROOT to the repo root (see env.sh)}"
module purge; module load CUDA/12.6.0 Miniforge3; source activate ssl
python "${ROOT}/results/lora_paper/co2_eomt/agg_co2_eomt_3arm.py"
echo "done exit=$?"
