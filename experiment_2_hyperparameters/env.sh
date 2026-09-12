# Experiment 2 — set paths once, then `source env.sh`. Works with or without SLURM.
export ST_LORA_ROOT="${ST_LORA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export CODE_DIR="${ST_LORA_ROOT}/code"

# Experiment 2 ablations run on BUP20 (sweet pepper) only.
# https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/b5d18108-53e1-46d1-873d-4230a72dfad7
export BUP20_DIR="${BUP20_DIR:-/path/to/BUP20}"
export BUP20_COCO="${BUP20_COCO:-${BUP20_DIR}/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json}"

export HF_HOME="${HF_HOME:-${ST_LORA_ROOT}/hf_cache}"

# Conda envs (from ../environment.yml / ../environment_eomt.yml)
export ENV_MAIN="${ENV_MAIN:-st_lora}"        # Mask2Former + SegFormer
export ENV_EOMT="${ENV_EOMT:-st_lora_eomt}"    # EoMT (transformers 5.14.1)

echo "[env] ST_LORA_ROOT=$ST_LORA_ROOT  BUP20_DIR=$BUP20_DIR"
