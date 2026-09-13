#!/bin/bash
#SBATCH --job-name=m2f_mcdrop_agg
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=0:15:00
# NOTE: Submit with --dependency=afterok:<EVAL_JOB_ID>

echo "================================================================"
echo "  MC Dropout Aggregation"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURMD_NODENAME"
echo "  Start  : $(date)"
echo "================================================================"

module purge
module load CUDA/12.6.0
module load Miniforge3
source activate ssl

RESULTS_DIR="${ST_LORA_ROOT}/results/lora_paper/mcdropout"
SEEDS=(42 123 456 789 1337)

python3 - <<EOF
import os, json
import numpy as np

results_dir = "${RESULTS_DIR}"
seeds       = [42, 123, 456, 789, 1337]
split       = "test"

METRICS = {
    "mIoU":               "mIoU ↑",
    "ECE":                "ECE ↓",
    "MECE":               "MECE ↓",
    "ACE":                "ACE ↓",
    "MACE":               "MACE ↓",
    "mean_aleatoric":     "Aleatoric ↓",
    "mean_epistemic":     "Epistemic ↓",
    "mean_total":         "Total Unc. ↓",
    "mean_pred_variance": "Pred. Var. ↓",
}

records = []
for seed in seeds:
    path = os.path.join(results_dir, f"seed_{seed}", f"eval_{split}.json")
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        continue
    with open(path) as f:
        records.append(json.load(f))

if not records:
    print("[ERROR] No results found.")
    exit(1)

print(f"\nLoaded {len(records)} seed results from {results_dir}")
print(f"T passes per seed: {records[0].get('T_passes', '?')}\n")

summary = {"method": "MC Dropout", "split": split, "n_seeds": len(records)}
agg_rows = []

print("=" * 58)
print(f"  MC Dropout — {split.upper()} split  ({len(records)} seeds)")
print("=" * 58)
print(f"  {'Metric':<30}  {'Mean':>10}  {'Std':>10}")
print(f"  {'─'*50}")

for key, label in METRICS.items():
    vals = [r[key] for r in records if key in r]
    if not vals:
        continue
    arr  = np.array(vals)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    summary[key] = {"mean": mean, "std": std, "values": vals}
    print(f"  {label:<30}  {mean:>10.4f}  {std:>10.4f}")
    agg_rows.append({"metric": key, "label": label, "mean": mean, "std": std})

print("=" * 58)

summary["per_seed"] = records

# Save summary JSON
out_json = os.path.join(results_dir, f"summary_{split}.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[Saved] JSON → {out_json}")

# Save CSV
import csv
out_csv = os.path.join(results_dir, f"summary_{split}.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "label", "mean", "std"]
                    + [f"seed_{s}" for s in seeds[:len(records)]])
    for key, label in METRICS.items():
        if key not in summary or not isinstance(summary[key], dict):
            continue
        row = [key, label,
               f"{summary[key]['mean']:.6f}",
               f"{summary[key]['std']:.6f}"]
        row += [f"{v:.6f}" for v in summary[key]["values"]]
        writer.writerow(row)
print(f"[Saved] CSV  → {out_csv}")
EOF

echo ""
echo "Aggregation complete at $(date)"
