#!/bin/bash
#SBATCH --job-name=co2_train
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-1

# CO2 comparison: FRE (full-FT) vs ST-LoRA, MATCHED aug+CutMix pipeline, same epochs/cadence/
# batch/1-GPU. Validated setup (smoke): cgroup isolates to 1 GPU; codecarbon offline (DEU) agrees
# with nvidia-smi power integral within ~2%. Each run also logs nvidia-smi power as ground truth.
SD="$(cd "$(dirname "$0")/../code" && pwd)"
OUT="${ST_LORA_ROOT}/results/lora_paper/co2"
METHODS=(fre stlora)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
EPOCHS=${EPOCHS:-40}
SEED=${SEED:-42}

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl_cc
mkdir -p "${OUT}/logs" "${OUT}/${M}"

echo "########## CO2 run: method=${M} epochs=${EPOCHS} seed=${SEED} ##########"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"; nvidia-smi -L

# ground-truth GPU power logger (2s cadence) on our (isolated) GPU
nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits -lms 2000 \
    > "${OUT}/${M}/power_${M}_seed${SEED}.csv" 2>/dev/null &
NVPID=$!

python -m py_compile "${SD}/co2_compare_train.py" || { kill $NVPID 2>/dev/null; echo "py_compile FAILED"; exit 1; }
python "${SD}/co2_compare_train.py" \
    --method ${M} --epochs ${EPOCHS} --seed ${SEED} \
    --cosine_T0 20 --snapshot_every 20 --batch_size 4 --num_workers 4 \
    --save_root "${OUT}/${M}/ckpt" --out_dir "${OUT}" \
    --measure_secs 15 --tracking_mode machine --log_level error
RC=$?

kill $NVPID 2>/dev/null; wait $NVPID 2>/dev/null

echo "########## ground-truth GPU energy (nvidia-smi integral) ##########"
awk -F',' 'NF>=2{p=$2+0; sum+=p; n++} END{ if(n){ dt=2.0; wh=sum*dt/3600.0;
  printf "method='"${M}"'  samples=%d  avg_power=%.1f W  GPU_energy=%.4f Wh = %.5f kWh\n", n, sum/n, wh, wh/1000.0 } }' \
    "${OUT}/${M}/power_${M}_seed${SEED}.csv"
echo "Finished co2 ${M} at $(date) exit=${RC}"
exit ${RC}
