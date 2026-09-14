#!/bin/bash
#SBATCH --job-name=st_lora_tests
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=mlgpu_short
#SBATCH --account=ag_igg_roscher
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00

# Test suite for all ST-LoRA experiments. For each experiment it checks:
#   1. py_compile of every code/*.py            (syntax)
#   2. bash -n of every jobs/*.sh               (shell syntax)
#   3. import of every module                   (dependency wiring; env-aware)
# then runs lightweight functional unit tests (corruptions, config catalog).
#
# Imports are classified: FAIL = missing dependency / broken import; warn = ran at import and needs
# data/results not present in a fresh checkout (not a code bug). The suite fails only on FAILs.
#
# Envs: set PY_MAIN (Mask2Former/SegFormer, transformers 4.52) and PY_EOMT (EoMT, transformers 5.14).
#   Locally:  PY_MAIN=$(which python) PY_EOMT=/path/to/eomt/python bash tests/run_all_tests.sh
#   SLURM:    edit the two lines below, then sbatch tests/run_all_tests.sh
set -uo pipefail
# Locate the repo. Under `bash tests/run_all_tests.sh` the script path works; under sbatch the
# script is copied to a spool dir, so set ST_LORA_REPO to the repo root when submitting.
REPO="${ST_LORA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TESTS="$REPO/tests"
[ -d "$REPO/experiment_1_single_model" ] || { echo "ERROR: repo not found at '$REPO' — set ST_LORA_REPO"; exit 2; }
: "${PY_MAIN:=python}"
: "${PY_EOMT:=$PY_MAIN}"
module load CUDA/12.6.0 2>/dev/null || true

total_fail=0
declare -a summary

for EXP in experiment_1_single_model experiment_2_hyperparameters experiment_3_covariate_shift \
           experiment_4_uncertainty_baselines experiment_5_additional_insights; do
  D="$REPO/$EXP"; [ -d "$D/code" ] || { echo "SKIP $EXP (missing)"; continue; }
  echo; echo "############################## $EXP ##############################"
  efail=0

  echo "--- (1) py_compile ---"
  "$PY_MAIN" -m py_compile "$D"/code/*.py 2>/tmp/pc.err && echo "  compile OK" || { echo "  COMPILE FAIL"; sed 's/^/    /' /tmp/pc.err | tail -5; efail=$((efail+1)); }

  echo "--- (2) bash -n jobs ---"
  jf=0; for j in "$D"/jobs/*.sh; do bash -n "$j" || { echo "  FAIL $(basename "$j")"; jf=1; }; done
  [ $jf = 0 ] && echo "  all jobs OK" || efail=$((efail+1))

  echo "--- (3) import check ---"
  EOMT_FILES=(); MAIN_FILES=()
  for f in "$D"/code/*.py; do
    case "$(basename "$f")" in eomt_*) EOMT_FILES+=("$f");; *) MAIN_FILES+=("$f");; esac
  done
  PYTHONPATH="$D/code" "$PY_MAIN" "$TESTS/import_check.py" "${MAIN_FILES[@]}" || efail=$((efail+1))
  if [ ${#EOMT_FILES[@]} -gt 0 ]; then
    echo "  (eomt modules, EoMT env)"
    PYTHONPATH="$D/code" "$PY_EOMT" "$TESTS/import_check.py" "${EOMT_FILES[@]}" || efail=$((efail+1))
  fi

  total_fail=$((total_fail+efail))
  summary+=("$EXP: $([ $efail = 0 ] && echo PASS || echo "FAIL ($efail)")")
done

echo; echo "############################## functional unit tests ##############################"
if "$PY_MAIN" "$TESTS/functional_tests.py"; then fstat=PASS; else fstat=FAIL; total_fail=$((total_fail+1)); fi
summary+=("functional_tests: $fstat")

echo; echo "================================ SUMMARY ================================"
for s in "${summary[@]}"; do echo "  $s"; done
echo "  TOTAL failing groups: $total_fail"
[ $total_fail = 0 ] && echo "  ALL TESTS PASSED" || echo "  SOME TESTS FAILED"
exit $total_fail
