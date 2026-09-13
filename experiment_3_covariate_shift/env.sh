# Experiment 3 — set paths once, then `source env.sh`. Works with or without SLURM.
export ST_LORA_ROOT="${ST_LORA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export CODE_DIR="${ST_LORA_ROOT}/code"

# 3b (Mask2Former component sensitivity) uses BUP20; 3a (SegFormer ensemble diversity) uses GrowliFlower-L.
# BUP20        — https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/b5d18108-53e1-46d1-873d-4230a72dfad7
# GrowliFlower-L — https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/cb328232-31f5-4b84-a929-8e1ee551d66a
export BUP20_DIR="${BUP20_DIR:-/path/to/BUP20}"
export BUP20_COCO="${BUP20_COCO:-${BUP20_DIR}/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json}"
export GROWLI_DIR="${GROWLI_DIR:-/path/to/growliflower_l}"

# 3b shift-eval reads trained Mask2Former checkpoints from here. Point it at the Experiment-2
# output (results/exp2/m2f, the default), OR at the Exp-3 train-from-scratch output
# (results/exp3/m2f_bup20) if you retrain with jobs/train_m2f_bup20.sh.
export EXP2_M2F_DIR="${EXP2_M2F_DIR:-/path/to/results/exp2/m2f}"

export HF_HOME="${HF_HOME:-${ST_LORA_ROOT}/hf_cache}"
export ENV_MAIN="${ENV_MAIN:-st_lora}"       # Mask2Former + SegFormer

echo "[env] ST_LORA_ROOT=$ST_LORA_ROOT  BUP20_DIR=$BUP20_DIR  GROWLI_DIR=$GROWLI_DIR"
