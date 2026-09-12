#!/bin/bash
#SBATCH --job-name=sf_co2_plot
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:12:00
ROOT="${ST_LORA_ROOT:?export ST_LORA_ROOT to the repo root (see env.sh)}"
E="$(cd "$(dirname "$0")/../code" && pwd)"
module purge; module load CUDA/12.6.0 Miniforge3; source activate ssl
echo "########## paired significance ##########"
python "${E}/co2_paired_significance_segformer.py" | tee "${ROOT}/results/lora_paper/segformer_co2/paired_significance.txt"
echo "########## 6-panel figure ##########"
python "${E}/plot_co2_energy_compare_segformer.py"
echo "exit=$?"
