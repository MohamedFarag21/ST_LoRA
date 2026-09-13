# -*- coding: utf-8 -*-
"""Aggregate pepper phase-1 (in-distribution) metrics across seeds.

Reads seed_<S>/metrics.json (results[split][method] = ECE/ACE/mIoU/acc) and writes
summary_<split>.{md,csv} + ece_barplot.png for each eval split (val, test).
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [42, 123, 456, 789, 1337]
METHODS = ["Uncalibrated", "TS", "Logistic", "Dirichlet", "LTS", "Meta", "Selective"]
PRETTY = {"Uncalibrated": "Uncalibrated", "TS": "Temperature", "Logistic": "Logistic",
          "Dirichlet": "Dirichlet", "LTS": "Local Temp (LTS)", "Meta": "Meta-Cal",
          "Selective": "Selective Scaling"}
KEYS = ["ECE", "ACE", "mIoU", "acc"]


def ms(vals):
    v = np.array([x for x in vals if x == x], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def aggregate_split(out_dir, seeds, split):
    data = {m: {k: [] for k in KEYS} for m in METHODS}
    found = []
    for s in seeds:
        p = os.path.join(out_dir, f"seed_{s}", "metrics.json")
        if not os.path.exists(p):
            continue
        res = json.load(open(p)).get("results", {}).get(split)
        if res is None:
            continue
        found.append(s)
        for m in METHODS:
            if m in res:
                for k in KEYS:
                    data[m][k].append(res[m][k])
    if not found:
        print(f"[{split}] no results")
        return None
    lines = [f"# Pepper in-distribution calibration ({split}), {len(found)} seeds {found}\n",
             "Mean ± std. ECE/ACE lower better; mIoU/acc higher better (8-class).\n",
             "| Method | ECE | ACE | mIoU | Acc |", "|---|---|---|---|---|"]
    csv = ["method,ECE_mean,ECE_std,ACE_mean,ACE_std,mIoU_mean,mIoU_std,acc_mean,acc_std"]
    ece_means, ece_stds = [], []
    for m in METHODS:
        cells, csvc = [], [PRETTY[m]]
        for k in KEYS:
            mu, sd = ms(data[m][k])
            cells.append(f"{mu:.4f} ± {sd:.4f}")
            csvc += [f"{mu:.5f}", f"{sd:.5f}"]
        lines.append(f"| {PRETTY[m]} | " + " | ".join(cells) + " |")
        csv.append(",".join(csvc))
        mu, sd = ms(data[m]["ECE"]); ece_means.append(mu); ece_stds.append(sd)
    open(os.path.join(out_dir, f"summary_{split}.md"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(out_dir, f"summary_{split}.csv"), "w").write("\n".join(csv) + "\n")
    print("\n".join(lines) + "\n")

    # ECE barplot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(METHODS))
    ax.bar(x, ece_means, yerr=ece_stds, capsize=4, color="#0e7c86", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in METHODS], rotation=30, ha="right")
    ax.set_ylabel("ECE ↓"); ax.set_title(f"Pepper 8-class ECE — {split} ({len(found)} seeds)")
    ax.grid(axis="y", ls="--", alpha=0.35)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"ece_barplot_{split}.png"), dpi=150)
    plt.close(fig)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/posthoc_calibration_pepper")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    for split in ["val", "test"]:
        print("=" * 60 + f"\n{split.upper()}\n" + "=" * 60)
        aggregate_split(args.out_dir, args.seeds, split)


if __name__ == "__main__":
    main()
