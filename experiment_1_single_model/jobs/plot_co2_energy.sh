#!/bin/bash
#SBATCH --job-name=plot_co2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00

# Render the FRE vs ST-LoRA energy/CO2 comparison figure (6 panels). Read-only on results;
# ssl_cc env (has matplotlib) — non-mutating.
SD="$(cd "$(dirname "$0")/../code" && pwd)"
module purge
module load Miniforge3
source activate ssl_cc
python -m py_compile "${SD}/plot_co2_energy_compare.py" || { echo "py_compile FAILED"; exit 1; }
python "${SD}/plot_co2_energy_compare.py"
echo "Finished plot_co2 at $(date) exit=$?"
