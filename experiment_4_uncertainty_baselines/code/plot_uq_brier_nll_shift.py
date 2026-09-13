# -*- coding: utf-8 -*-
"""
Proper-scoring-rule view of the 4 UQ methods under distribution shift.

ECE bins the reliability diagram and (a) suffers the float32 scatter_add artifact and
(b) penalises a *sharper* predictive distribution even when it is more accurate. Brier &
NLL are proper scoring rules computed in float64 with NO binning -> immune to both.

This figure shows the ranking FLIP for ST-LoRA:
  - under ECE, ST-LoRA looks WORST under shift,
  - under Brier AND NLL, ST-LoRA is the BEST under shift (and best shift mIoU).

Reads:  lora_paper/calibration_shift/native_float64/<m>_seed*_shift_native.json  (5 seeds)
Writes (into out_dir):
  uq_proper_scores_shift.png   3 panels (ECE | Brier | NLL) mean-over-grid vs severity
  uq_proper_scores_summary.md  clean + shift-averaged table with the ranking per metric
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
SEEDS = [42, 123, 456, 789, 1337]
CORRUPTIONS = ["blur", "noise", "brightness", "contrast",
               "saturation", "translation", "rotation"]
SEVS = ["1", "2", "3", "4", "5"]
METRICS = [("ECE", "ECE ↓  (binned — artifact-prone)"),
           ("Brier", "Brier ↓  (proper, float64)"),
           ("NLL", "NLL ↓  (proper, float64)")]

# palette consistent with shift_fair figure
SERIES = [
    ("lora",      "ST-LoRA",        "#4a3aa7", "*", 2.6),
    ("fullft",    "FRE (Full-FT)",  "#e34948", "h", 2.0),
    ("mcdropout", "MC-Dropout",     "#17a2b8", "<", 2.0),
    ("ddu",       "DDU",            "#8c564b", ">", 2.0),
]


def load(uq_dir):
    """data[method][seed]['grid'] -> nested dict."""
    d = {}
    for m, *_ in SERIES:
        d[m] = {}
        for s in SEEDS:
            p = os.path.join(uq_dir, f"{m}_seed{s}_shift_native.json")
            if os.path.exists(p):
                d[m][s] = json.load(open(p))["grid"]
    return d


def clean_val(data, method, metric):
    vals = [data[method][s]["clean"]["0"][metric] for s in data[method]]
    return float(np.mean(vals)), float(np.std(vals))


def sev_curve(data, method, metric):
    """mean over 7 corruptions AND seeds at each severity -> (means[5], stds[5])."""
    means, stds = [], []
    for sev in SEVS:
        # per-seed mean over the 7 corruptions at this severity
        per_seed = []
        for s in data[method]:
            g = data[method][s]
            vals = [g[c][sev][metric] for c in CORRUPTIONS if c in g and sev in g[c]]
            if vals:
                per_seed.append(np.mean(vals))
        means.append(float(np.mean(per_seed)) if per_seed else np.nan)
        stds.append(float(np.std(per_seed)) if per_seed else np.nan)
    return np.array(means), np.array(stds)


def shift_avg(data, method, metric):
    """single scalar: per-seed mean over the whole 7x5 grid, then mean/std over seeds."""
    per_seed = []
    for s in data[method]:
        g = data[method][s]
        vals = [g[c][sev][metric] for c in CORRUPTIONS for sev in SEVS
                if c in g and sev in g[c]]
        per_seed.append(np.mean(vals))
    return float(np.mean(per_seed)), float(np.std(per_seed))


def plot(data, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    for ax, (metric, ylab) in zip(axes, METRICS):
        for m, disp, col, mk, lw in SERIES:
            means, stds = sev_curve(data, m, metric)
            ax.plot(range(1, 6), means, color=col, marker=mk, ms=7, lw=lw, label=disp,
                    zorder=3)
            ax.fill_between(range(1, 6), means - stds, means + stds, color=col,
                            alpha=0.12, zorder=1)
        ax.set_yscale("log")
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(["S1", "S2", "S3", "S4", "S5"])
        ax.set_xlabel("Corruption severity (mean over 7)", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_title("ECE ranks ST-LoRA WORST", fontsize=12, color="#4a3aa7")
    axes[1].set_title("Brier ranks ST-LoRA BEST", fontsize=12, color="#4a3aa7")
    axes[2].set_title("NLL ranks ST-LoRA BEST", fontsize=12, color="#4a3aa7")
    axes[0].legend(fontsize=9.5, framealpha=0.9, loc="upper left")
    fig.suptitle("Pepper UQ under shift: binned ECE vs proper scoring rules (test93, "
                 "native, float64, 5 seeds)\nThe ST-LoRA “worst-calibrated” verdict is "
                 "a binning artifact — proper scores rank it best.", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def write_table(data, out_path):
    L = ["# UQ methods under shift — proper scoring rules (test93, native, float64, 5 seeds)",
         "", "Brier & NLL are float64, unbinned -> immune to the ECE scatter_add artifact.",
         "", "## Clean (severity 0)", "",
         "| Method | ECE | Brier | NLL | mIoU |", "|---|---|---|---|---|"]
    for m, disp, *_ in SERIES:
        c = {k: clean_val(data, m, k) for k in ("ECE", "Brier", "NLL", "mIoU")}
        L.append(f"| {disp} | {c['ECE'][0]:.4f}±{c['ECE'][1]:.4f} | "
                 f"{c['Brier'][0]:.4f}±{c['Brier'][1]:.4f} | "
                 f"{c['NLL'][0]:.4f}±{c['NLL'][1]:.4f} | "
                 f"{c['mIoU'][0]:.4f}±{c['mIoU'][1]:.4f} |")
    L += ["", "## Shift-averaged (7 corruptions x 5 severities = 35 cells)", "",
          "| Method | ECE | Brier | NLL | mIoU |", "|---|---|---|---|---|"]
    for m, disp, *_ in SERIES:
        sv = {k: shift_avg(data, m, k) for k in ("ECE", "Brier", "NLL", "mIoU")}
        L.append(f"| {disp} | {sv['ECE'][0]:.4f}±{sv['ECE'][1]:.4f} | "
                 f"{sv['Brier'][0]:.4f}±{sv['Brier'][1]:.4f} | "
                 f"{sv['NLL'][0]:.4f}±{sv['NLL'][1]:.4f} | "
                 f"{sv['mIoU'][0]:.4f}±{sv['mIoU'][1]:.4f} |")
    L += ["", "## Shift ranking (lower = better)"]
    for k in ("ECE", "Brier", "NLL"):
        order = sorted([m for m, *_ in SERIES], key=lambda m: shift_avg(data, m, k)[0])
        disp = {m: d for m, d, *_ in SERIES}
        L.append(f"- **{k}**: " + " < ".join(
            f"{disp[m]} ({shift_avg(data, m, k)[0]:.4f})" for m in order))
    open(out_path, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"[saved] {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uq_dir",
                    default=f"{ROOT}/results/lora_paper/calibration_shift/native_float64")
    ap.add_argument("--out_dir",
                    default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native")
    args = ap.parse_args()
    data = load(args.uq_dir)
    for m, *_ in SERIES:
        print(f"[load] {m}: {sorted(data[m].keys())}", flush=True)
    write_table(data, os.path.join(args.out_dir, "uq_proper_scores_summary.md"))
    plot(data, os.path.join(args.out_dir, "uq_proper_scores_shift.png"))


if __name__ == "__main__":
    main()
