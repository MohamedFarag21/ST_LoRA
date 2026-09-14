#!/bin/bash
#SBATCH --job-name=ensT20_plot
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=intelsr_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00

SD="${ST_LORA_ROOT}/code"
E="${ST_LORA_ROOT}/results/lora_paper/ensemble_size_ablation_T20/eval"

module purge
module load Miniforge3
source activate ssl

python -m py_compile "${SD}/plot_ens_size_ablation.py" || { echo "py_compile FAILED"; exit 1; }
python "${SD}/plot_ens_size_ablation.py" "${E}" "T0=20 (deployed cadence), snapshot/20ep"
rc=$?
echo "Finished T20 plot at $(date) exit=${rc}"
ls -l "${E}/ens_size_ablation_curves.png"
exit ${rc}
