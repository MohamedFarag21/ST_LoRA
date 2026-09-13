#!/bin/bash
#SBATCH --job-name=miou_paired_sig
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
ROOT="${ST_LORA_ROOT}"
SD="${ROOT}/code"
module purge; module load CUDA/12.6.0 Miniforge3; source activate ssl_cc
python "${SD}/miou_paired_significance.py"
echo "done exit=$?"
