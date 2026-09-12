#!/bin/bash
#SBATCH --job-name=exp2_eff
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=sgpu_short
#SBATCH --account=ag_igg_roscher
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Efficiency: adapting FFN ONLY vs FFN+attention. Trainable-parameter count (the cheap,
# deterministic efficiency proxy) via the dry-run asserter, for all three architectures.
# For wall-clock / GPU-energy / CO2, run the Experiment-1 co2_compare_train.py scripts with the
# same two target sets (see this experiment's README).
set -uo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
CODE="$(cd "$(dirname "$0")/../code" && pwd)"
module load CUDA/12.6.0 2>/dev/null || true
SSL=/home/mibrahi2_hpc/.conda/envs/ssl/bin/python
EOMT=/home/mibrahi2_hpc/.conda/envs/eomt/bin/python
R="$CODE/resolve_targets.py"

echo "############ M2F: FFN-only vs FFN+attn (r=8) ############"
echo "-- FFN only (enc_ffn + dec_ffn):"     ; "$SSL"  "$R" --arch m2f --roles enc_ffn dec_ffn --assert >/dev/null
echo "-- FFN + attn (baseline):"            ; "$SSL"  "$R" --arch m2f --roles enc_attn enc_ffn dec_attn dec_ffn --assert >/dev/null
echo "############ SegFormer: FFN-only vs FFN+attn ############"
echo "-- FFN only (enc_ffn + dec_ffn):"     ; "$SSL"  "$R" --arch segformer --roles enc_ffn dec_ffn --assert >/dev/null
echo "-- FFN + attn (baseline):"            ; "$SSL"  "$R" --arch segformer --roles enc_attn enc_ffn dec_ffn --assert >/dev/null
echo "############ EoMT: FFN-only vs FFN+attn ############"
echo "-- FFN only (enc_ffn + mask_ffn):"    ; "$EOMT" "$R" --arch eomt --roles enc_ffn mask_ffn --assert >/dev/null
echo "-- FFN + attn (baseline):"            ; "$EOMT" "$R" --arch eomt --roles enc_attn enc_ffn mask_ffn --assert >/dev/null
echo "done $(date)"
