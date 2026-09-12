#!/bin/bash
#SBATCH --job-name=sf_co2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --array=0-9

# SegFormer CO2/energy: FRE vs ST-LoRA, MATCHED aug+CutMix, 40 ep / 1 GPU, 5 seeds.
# Mirrors the M2F (§11.10) and EoMT CO2 studies. ssl_cc env. Per-epoch trace + nvidia-smi ground truth.
SD="$(cd "$(dirname "$0")/../code" && pwd)"
OUT="${ST_LORA_ROOT}/results/lora_paper/segformer_co2"
METHODS=(fre fre fre fre fre stlora stlora stlora stlora stlora)
SEEDS=(42 123 456 789 1337 42 123 456 789 1337)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}; SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
EPOCHS=${EPOCHS:-40}

module purge; module load CUDA/12.6.0 Miniforge3; source activate ssl_cc
mkdir -p "${OUT}/logs" "${OUT}/${M}"
echo "########## SegFormer CO2: method=${M} epochs=${EPOCHS} seed=${SEED} ##########"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"; nvidia-smi -L

nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits -lms 2000 \
    > "${OUT}/${M}/power_${M}_seed${SEED}.csv" 2>/dev/null &
NVPID=$!
python -m py_compile "${SD}/segformer_co2_compare_train.py" || { kill $NVPID 2>/dev/null; echo "py_compile FAILED"; exit 1; }
python "${SD}/segformer_co2_compare_train.py" \
    --method ${M} --epochs ${EPOCHS} --seed ${SEED} \
    --cosine_T0 20 --snapshot_every 20 --batch_size 4 --num_workers 4 \
    --save_root "${OUT}/${M}/ckpt_seed${SEED}" --out_dir "${OUT}" \
    --measure_secs 15 --tracking_mode machine --log_level error \
    --per_epoch_trace --sample_secs 2.0
RC=$?
kill $NVPID 2>/dev/null; wait $NVPID 2>/dev/null
awk -F',' 'NF>=2{p=$2+0; sum+=p; n++} END{ if(n){ dt=2.0; wh=sum*dt/3600.0;
  printf "GROUND-TRUTH method='"${M}"' seed='"${SEED}"' samples=%d avg=%.1fW GPU_energy=%.5f kWh\n", n, sum/n, wh/1000.0 } }' \
    "${OUT}/${M}/power_${M}_seed${SEED}.csv"
echo "Finished ${M} seed ${SEED} at $(date) exit=${RC}"; exit ${RC}
