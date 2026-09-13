# Experiment 4 — set paths once, then `source env.sh` before sbatch (SLURM inherits the env).
export ST_LORA_ROOT="${ST_LORA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export CODE_DIR="${ST_LORA_ROOT}/code"

# Datasets. In-distribution + shift use BUP20; OoD uses GrowliFlower-L and BUTom21 (tomato).
# BUP20        — https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/b5d18108-53e1-46d1-873d-4230a72dfad7
# GrowliFlower-L — https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/cb328232-31f5-4b84-a929-8e1ee551d66a
# BUTom21 (tomato) — https://arxiv.org/abs/2607.14934
export BUP20_DIR="${BUP20_DIR:-/path/to/BUP20}"
export BUP20_COCO="${BUP20_COCO:-${BUP20_DIR}/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json}"
export GROWLI_DIR="${GROWLI_DIR:-/path/to/growliflower_l}"
export BUTOM21_DIR="${BUTOM21_DIR:-/path/to/butom21}"

export HF_HOME="${HF_HOME:-${ST_LORA_ROOT}/hf_cache}"
# Conda envs: ENV_MAIN for training/eval; the paired-significance scripts run in ENV_STATS.
export ENV_MAIN="${ENV_MAIN:-st_lora}"
export ENV_STATS="${ENV_STATS:-st_lora}"

echo "[env] ST_LORA_ROOT=$ST_LORA_ROOT  BUP20_DIR=$BUP20_DIR  GROWLI_DIR=$GROWLI_DIR  BUTOM21_DIR=$BUTOM21_DIR"
