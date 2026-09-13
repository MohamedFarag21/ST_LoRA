# -*- coding: utf-8 -*-
"""
Calibration Shift Aggregation — mean ± std across 5 seeds per method
======================================================================
Reads per-seed CSV files, computes mean ± std across seeds, and produces
a single summary CSV and JSON with the full comparison table.

Usage:
    python aggregate_calibration_shift.py
"""

import os
import json
import csv
import argparse
import numpy as np
from collections import defaultdict

METHODS  = ["lora", "fullft", "mcdropout", "ddu"]
SEEDS    = [42, 123, 456, 789, 1337]
METRICS  = ["mIoU", "ECE", "ACE"]
SEV_LABELS = ["s1", "s2", "s3", "s4", "s5"]

METHOD_NAMES = {
    "lora":      "LoRA Snapshot",
    "fullft":    "Full FT Snapshot",
    "mcdropout": "MC Dropout",
    "ddu":       "DDU",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/calibration_shift")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load all per-seed CSVs ────────────────────────────────────────────────
    # Structure: data[method][corruption][severity][metric] = list of values
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for method in METHODS:
        for seed in SEEDS:
            csv_path = os.path.join(
                args.results_dir,
                f"{method}_seed{seed}_calibration_shift.csv")
            if not os.path.exists(csv_path):
                print(f"  [MISSING] {csv_path}")
                continue
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    corr = row["corruption"]
                    sev  = row["severity"]
                    for m in METRICS:
                        data[method][corr][sev][m].append(float(row[m]))

    corruptions = list(next(iter(data.values())).keys()) if data else []
    print(f"Loaded data for methods: {[m for m in METHODS if m in data]}")
    print(f"Corruptions: {corruptions}\n")

    # ── Compute mean ± std ────────────────────────────────────────────────────
    summary = {}
    for method in METHODS:
        if method not in data:
            continue
        summary[method] = {}
        for corr in corruptions:
            summary[method][corr] = {}
            for sev in SEV_LABELS:
                summary[method][corr][sev] = {}
                for metric in METRICS:
                    vals = data[method][corr][sev][metric]
                    if not vals:
                        summary[method][corr][sev][metric] = {"mean": None, "std": None}
                        continue
                    arr = np.array(vals)
                    summary[method][corr][sev][metric] = {
                        "mean":   round(float(arr.mean()), 4),
                        "std":    round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
                        "values": vals,
                    }

    # ── Print summary table ───────────────────────────────────────────────────
    for metric in METRICS:
        print(f"\n{'='*80}")
        print(f"  {metric} — mean ± std across {len(SEEDS)} seeds")
        print(f"{'='*80}")
        header = f"  {'Method':<18} {'Corruption':<14} " + \
                 "  ".join(f"{'S'+str(i+1):>12}" for i in range(5))
        print(header)
        print(f"  {'─'*75}")
        for method in METHODS:
            if method not in summary:
                continue
            for ci, corr in enumerate(corruptions):
                method_label = METHOD_NAMES[method] if ci == 0 else ""
                vals = []
                for sev in SEV_LABELS:
                    entry = summary[method][corr][sev].get(metric, {})
                    if entry.get("mean") is not None:
                        vals.append(f"{entry['mean']:.3f}±{entry['std']:.3f}")
                    else:
                        vals.append("  n/a  ")
                print(f"  {method_label:<18} {corr:<14} " +
                      "  ".join(f"{v:>12}" for v in vals))
            print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = os.path.join(args.results_dir, "calibration_shift_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {out_json}")

    # ── Save flat CSV ─────────────────────────────────────────────────────────
    out_csv = os.path.join(args.results_dir, "calibration_shift_summary.csv")
    rows = []
    for method in METHODS:
        if method not in summary:
            continue
        for corr in corruptions:
            for sev in SEV_LABELS:
                row = {"method": METHOD_NAMES[method], "corruption": corr, "severity": sev}
                for metric in METRICS:
                    entry = summary[method][corr][sev].get(metric, {})
                    row[f"{metric}_mean"] = entry.get("mean", "")
                    row[f"{metric}_std"]  = entry.get("std",  "")
                rows.append(row)

    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"[Saved] {out_csv}")


if __name__ == "__main__":
    main()
