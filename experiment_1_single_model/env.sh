# Experiment 1 — one place to set paths. `source env.sh` before sbatch / local runs.
# Works the same on a SLURM cluster and a plain workstation.

# Repo root (where outputs go). Default: the folder that contains this experiment.
export ST_LORA_ROOT="${ST_LORA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Code dir for this experiment (all Python lives here, flat).
export CODE_DIR="${ST_LORA_ROOT}/code"

# ---- datasets (EDIT THESE) ----
# Experiment 1 uses BUP20 (sweet pepper) only.
export BUP20_DIR="${BUP20_DIR:-/path/to/BUP20}"
# COCO-style annotation json inside BUP20 (adjust to your download's layout):
export BUP20_COCO="${BUP20_COCO:-${BUP20_DIR}/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json}"

# HuggingFace cache for the pretrained backbones (downloaded on first run).
export HF_HOME="${HF_HOME:-${ST_LORA_ROOT}/hf_cache}"

# Conda env names (create from ../environment.yml and ../environment_eomt.yml).
#   Mask2Former + SegFormer -> $ENV_MAIN ;  EoMT -> $ENV_EOMT ;  CO2/EoMT -> $ENV_EOMT_CC
export ENV_MAIN="${ENV_MAIN:-st_lora}"
export ENV_EOMT="${ENV_EOMT:-st_lora_eomt}"
export ENV_EOMT_CC="${ENV_EOMT_CC:-st_lora_eomt_cc}"   # eomt env + codecarbon (CO2 only)

echo "[env] ST_LORA_ROOT=$ST_LORA_ROOT  CODE_DIR=$CODE_DIR  BUP20_DIR=$BUP20_DIR"
