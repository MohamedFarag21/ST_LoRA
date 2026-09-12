# Experiment 1 — one place to set paths. `source env.sh` before sbatch / local runs.
# Works the same on a SLURM cluster and a plain workstation.

# Repo root (where outputs go). Default: the folder that contains this experiment.
export ST_LORA_ROOT="${ST_LORA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Code dir for this experiment (all Python lives here, flat).
export CODE_DIR="${ST_LORA_ROOT}/code"

# ---- datasets (EDIT THESE) ----
# BUP20 (sweet pepper, 8-class) — https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/b5d18108-53e1-46d1-873d-4230a72dfad7
export BUP20_DIR="${BUP20_DIR:-/path/to/BUP20}"
# COCO-style annotation json inside BUP20 (adjust to your download's layout):
export BUP20_COCO="${BUP20_COCO:-${BUP20_DIR}/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json}"

# GrowliFlower-L (cauliflower; binary plant-vs-bg here) — https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/cb328232-31f5-4b84-a929-8e1ee551d66a
# expects images/{Train,Val,Test} and labels/{Train,Val,Test}/{maskPlants,maskVoid,...}
export GROWLI_DIR="${GROWLI_DIR:-/path/to/growliflower_l}"

# HuggingFace cache for the pretrained backbones (downloaded on first run).
export HF_HOME="${HF_HOME:-${ST_LORA_ROOT}/hf_cache}"

# Conda env names (create from ../environment.yml and ../environment_eomt.yml).
#   Mask2Former + SegFormer -> $ENV_MAIN ;  EoMT -> $ENV_EOMT ;  CO2/EoMT -> $ENV_EOMT_CC
export ENV_MAIN="${ENV_MAIN:-st_lora}"
export ENV_EOMT="${ENV_EOMT:-st_lora_eomt}"
export ENV_EOMT_CC="${ENV_EOMT_CC:-st_lora_eomt_cc}"   # eomt env + codecarbon (CO2 only)

echo "[env] ST_LORA_ROOT=$ST_LORA_ROOT  CODE_DIR=$CODE_DIR  BUP20_DIR=$BUP20_DIR  GROWLI_DIR=$GROWLI_DIR"
