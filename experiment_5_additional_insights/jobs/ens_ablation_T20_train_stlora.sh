#!/bin/bash
#SBATCH --job-name=ensT20_stlora
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

# Ensemble-size ablation (VARIANT T20) — ST-LoRA arm, DEPLOYED final_model LoRA config
# (r=32, alpha=32, dropout=0.1, no_dora, targets [k_proj,v,k,q_proj,dense,class_predictor]).
# DEPLOYED cadence: T_0=20, snapshot every 20 epochs. epochs=210 -> shots {0,20,...,200}:
# usable members shots 1..10 (shot_0 = epoch-0 init, ignored).
SD="${ST_LORA_ROOT}/code"
BASE="${ST_LORA_ROOT}/results/lora_paper/ensemble_size_ablation_T20/stlora"
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

echo "=== ST-LoRA ens-ablation-T20 seed $SEED : epochs=$EPOCHS T0=$T0 snap_every=$SNAP (final_model cfg) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python -m py_compile "${SD}/mask2former_lora_train_seeded.py" || { echo "py_compile FAILED"; exit 1; }

python "${SD}/mask2former_lora_train_seeded.py" \
    --seed ${SEED} \
    --base_save_dir "${BASE}" \
    --log_dir "${BASE}/tb_logs" \
    --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 --no_dora \
    --target_modules k_proj v k q_proj dense class_predictor \
    --epochs ${EPOCHS} --cosine_T0 ${T0} --snapshot_every ${SNAP}
rc=$?
echo "Finished ST-LoRA-T20 seed ${SEED} at $(date) exit=${rc}"
echo "snapshots:"; ls -1d "${BASE}/seed_${SEED}"/model_shot_* 2>/dev/null
exit ${rc}
