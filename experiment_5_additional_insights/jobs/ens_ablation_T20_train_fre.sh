#!/bin/bash
#SBATCH --job-name=ensT20_fre
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=7:00:00
#SBATCH --array=0-2

# Ensemble-size ablation (VARIANT T20) — FRE (Full-FT snapshot) arm.
# DEPLOYED cadence: T_0=20, snapshot every 20 epochs. epochs=210 -> snapshots at
# {20,40,...,200} = 10 members (Lightning runs epochs 0..209). Members are as-converged as
# the deployed 20-epoch-cycle models (~2x compute of the 110-ep T10 variant).
SD="${ST_LORA_ROOT}/code"
BASE="${ST_LORA_ROOT}/results/lora_paper/ensemble_size_ablation_T20/fre"
SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
EPOCHS=${EPOCHS:-210}
SNAP=${SNAP:-20}
T0=${T0:-20}

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl
mkdir -p "${BASE}/logs"

echo "=== FRE ens-ablation-T20 seed $SEED : epochs=$EPOCHS T0=$T0 snap_every=$SNAP ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python -m py_compile "${SD}/mask2former_full_train_seeded.py" || { echo "py_compile FAILED"; exit 1; }

python "${SD}/mask2former_full_train_seeded.py" \
    --seed ${SEED} \
    --base_save_dir "${BASE}" \
    --log_dir "${BASE}/tb_logs" \
    --epochs ${EPOCHS} --cosine_T0 ${T0} --snapshot_every ${SNAP}
rc=$?
echo "Finished FRE-T20 seed ${SEED} at $(date) exit=${rc}"
echo "snapshots:"; ls -1 "${BASE}/seed_${SEED}"/model_shot_*.pt 2>/dev/null
exit ${rc}
