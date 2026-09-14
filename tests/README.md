# Tests

Fast, GPU-light checks that every experiment's code is well-formed and its dependencies resolve —
no training, no datasets, no result files required.

For each of the five experiments the suite runs:
1. **`py_compile`** of every `code/*.py` (syntax).
2. **`bash -n`** of every `jobs/*.sh` (shell syntax).
3. **Import check** of every module (dependency wiring), env-aware: `eomt_*.py` modules are imported
   with the EoMT env (transformers 5.14), the rest with the main env (transformers 4.52).

Then **functional unit tests** (`functional_tests.py`): the Experiment-3 corruption functions applied
to a synthetic image+mask (labels stay in {0,1,255}; grid counts 19 for BUP20 and 80 for GrowliFlower),
and the Experiment-2 config-catalog counts (35 / 27 / 27).

Import results are classified: **FAIL** = missing dependency / broken import (real bug); **warn** = a
script that runs at import and needs data/results absent from a fresh checkout (not a code bug). The
suite fails only on FAILs.

## Run

```bash
# create the two envs first (repo root): environment.yml -> st_lora, environment_eomt.yml -> st_lora_eomt
conda activate st_lora
PY_MAIN=$(which python) PY_EOMT=/path/to/st_lora_eomt/bin/python bash tests/run_all_tests.sh
```

On SLURM the script is copied to a spool dir, so pass the repo root via `ST_LORA_REPO`:
```bash
sbatch --export=ALL,ST_LORA_REPO=$PWD,PY_MAIN=/path/st_lora/python,PY_EOMT=/path/st_lora_eomt/python \
       tests/run_all_tests.sh
```
Exit code = number of failing groups (0 = all passed).
