# Experiment 5 — set paths once, then `source env.sh` before sbatch (SLURM inherits the env).
export ST_LORA_ROOT="${ST_LORA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export CODE_DIR="${ST_LORA_ROOT}/code"

# BUP20 for diversity / per-class / ensemble-size; GrowliFlower-L + BUTom21 for the OoD panels.
export BUP20_DIR="${BUP20_DIR:-/path/to/BUP20}"
export BUP20_COCO="${BUP20_COCO:-${BUP20_DIR}/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json}"
export GROWLI_DIR="${GROWLI_DIR:-/path/to/growliflower_l}"
export BUTOM21_DIR="${BUTOM21_DIR:-/path/to/butom21}"   # tomato — https://arxiv.org/abs/2607.14934

export HF_HOME="${HF_HOME:-${ST_LORA_ROOT}/hf_cache}"
export ENV_MAIN="${ENV_MAIN:-st_lora}"

echo "[env] ST_LORA_ROOT=$ST_LORA_ROOT  BUP20_DIR=$BUP20_DIR  GROWLI_DIR=$GROWLI_DIR  BUTOM21_DIR=$BUTOM21_DIR"
