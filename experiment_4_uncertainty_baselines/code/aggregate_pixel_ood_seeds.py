# -*- coding: utf-8 -*-
"""
Aggregate pepper_to_tomato_pixel_ood.py per-seed JSON results across all
seeds into mean +/- std per metric, for both the entropy and MI scorers.

Usage:
    python aggregate_pixel_ood_seeds.py --results_dir <dir> --seeds 42 123 456 789 1337
"""

import os
import json
import argparse

import numpy as np

METRICS = ["AUROC", "AUPR", "FPR95", "sIoU", "PPV", "MeanF1"]


def main():
    parser = argparse.ArgumentParser(description="Aggregate pepper->tomato pixel-OoD results across seeds")
    parser.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/tomato_lora/pepper_transfer")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    per_seed = {}
    missing = []
    for seed in args.seeds:
        path = os.path.join(args.results_dir, f"pepper_to_tomato_pixelood_seed{seed}_results.json")
        if not os.path.isfile(path):
            missing.append(path)
            continue
        with open(path) as f:
            per_seed[seed] = json.load(f)

    if missing:
        print("Missing result files (not yet finished):")
        for p in missing:
            print(f"  {p}")

    found_seeds = sorted(per_seed.keys())
    print(f"\nAggregating over {len(found_seeds)}/{len(args.seeds)} seeds: {found_seeds}\n")

    summary = {"seeds_used": found_seeds, "scorers": {}}
    for scorer in ["entropy", "MI"]:
        vals = {m: [] for m in METRICS}
        for seed in found_seeds:
            m = per_seed[seed]["pixel_level"][scorer]
            for k in METRICS:
                vals[k].append(m[k])

        print(f"Scorer: {scorer}")
        print(f"  {'Metric':<10} {'mean':>8} {'std':>8}   per-seed values")
        scorer_summary = {}
        for k in METRICS:
            arr = np.array(vals[k])
            mean, std = arr.mean(), arr.std()
            scorer_summary[k] = {"mean": float(mean), "std": float(std),
                                 "per_seed": dict(zip(found_seeds, arr.tolist()))}
            vals_str = ", ".join(f"{v:.4f}" for v in arr)
            print(f"  {k:<10} {mean:>8.4f} {std:>8.4f}   [{vals_str}]")
        print()
        summary["scorers"][scorer] = scorer_summary

    out_path = args.out or os.path.join(args.results_dir, "pepper_to_tomato_pixelood_seed_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
