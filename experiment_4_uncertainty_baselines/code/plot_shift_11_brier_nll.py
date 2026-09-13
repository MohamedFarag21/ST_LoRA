# -*- coding: utf-8 -*-
"""
11-method proper-scoring-rule view under distribution shift (test93, native, float64).
Brier & NLL are unbinned -> immune to the ECE scatter_add artifact. For each metric:
  * <metric>_grid.png : 7 corruption panels x 5 severities, all 11 methods (+legend panel)
  * <metric>_mean.png : mean over the 7 corruptions vs severity (headline)
Reads the SAME files as aggregate_plot_shift_fair.py (calibrators now carry Brier/NLL
after the StreamBinMetrics extension + re-sweep).
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
SEEDS = [42, 123, 456, 789, 1337]
CORRUPTIONS = ["blur", "noise", "brightness", "contrast",
               "saturation", "translation", "rotation"]
SEVS = ["1", "2", "3", "4", "5"]

SERIES = [
    ("Uncalibrated", "Uncalibrated",      "#9a9a94", ":",  "o"),
    ("TS",           "Temp. Scaling",     "#2a78d6", "-",  "s"),
    ("Logistic",     "Logistic",          "#eb6834", "-",  "^"),
    ("Dirichlet",    "Dirichlet",         "#1baf7a", "-",  "D"),
    ("LTS",          "Local Temp (LTS)",  "#eda100", "-",  "v"),
    ("Meta",         "Meta-Cal",          "#e87ba4", "-",  "P"),
    ("Selective",    "Selective Scaling", "#008300", "-",  "X"),
    ("lora",         "ST-LoRA",           "#4a3aa7", "--", "*"),
    ("fullft",       "FRE (Full-FT)",     "#e34948", "--", "h"),
    ("mcdropout",    "MC-Dropout",        "#17a2b8", "--", "<"),
    ("ddu",          "DDU",               "#8c564b", "--", ">"),
]
CAL_KEYS = {"Uncalibrated", "TS", "Logistic", "Dirichlet", "LTS", "Meta", "Selective"}


def load_all(cal_dir, uq_dir):
    data = {s: {} for s in SEEDS}
    for s in SEEDS:
        p = os.path.join(cal_dir, f"seed_{s}", "shift_metrics_native_test93.json")
        if os.path.exists(p):
            grid = json.load(open(p))["grid"]
            for m in CAL_KEYS:
                data[s][m] = {c: {sev: grid[c][sev][m] for sev in grid.get(c, {})}
                              for c in CORRUPTIONS if c in grid}
    for m in ("lora", "fullft", "mcdropout", "ddu"):
        for s in SEEDS:
            p = os.path.join(uq_dir, f"{m}_seed{s}_shift_native.json")
            if os.path.exists(p):
                grid = json.load(open(p))["grid"]
                data[s][m] = {c: {sev: grid[c][sev] for sev in grid.get(c, {})}
                              for c in CORRUPTIONS if c in grid}
    return data


def present(data, key):
    return any(key in data[s] for s in SEEDS)


def sev_mean(data, method, corr, metric):
    out = []
    for sev in SEVS:
        vals = [data[s][method][corr][sev][metric]
                for s in SEEDS
                if method in data[s] and corr in data[s].get(method, {})
                and sev in data[s][method][corr]
                and data[s][method][corr][sev].get(metric) is not None]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def corr_sev_mean(data, method, metric):
    per_sev = []
    for i, sev in enumerate(SEVS):
        vals = [sev_mean(data, method, c, metric)[i] for c in CORRUPTIONS]
        vals = [v for v in vals if not np.isnan(v)]
        per_sev.append(float(np.mean(vals)) if vals else float("nan"))
    return per_sev


def plot_grid(data, metric, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), squeeze=False)
    for i, corr in enumerate(CORRUPTIONS):
        ax = axes[i // 4][i % 4]
        for key, disp, col, ls, mk in SERIES:
            if not present(data, key):
                continue
            ys = sev_mean(data, key, corr, metric)
            lw = 1.3 if key in ("Uncalibrated", "TS", "Logistic") else 2.0
            ax.plot(range(1, 6), ys, color=col, ls=ls, marker=mk, ms=5, lw=lw,
                    zorder=2 if key == "Uncalibrated" else 3)
        ax.set_yscale("log")
        ax.set_title(corr.capitalize(), fontsize=13)
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(["S1", "S2", "S3", "S4", "S5"], fontsize=9)
        ax.set_ylabel(f"{metric} ↓ (log)", fontsize=10); ax.set_xlabel("Severity", fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lg = axes[1][3]; lg.axis("off")
    handles = [Line2D([0], [0], color=c, ls=ls, marker=mk, ms=7, lw=2, label=d)
               for k, d, c, ls, mk in SERIES if present(data, k)]
    lg.legend(handles=handles, loc="center", fontsize=11, framealpha=0.9,
              title="Method  (solid = post-hoc calibrator, dashed = UQ)", title_fontsize=11)
    fig.suptitle(f"Pepper {metric} under distribution shift — 11 methods "
                 f"(test93, native, float64, 5-seed mean, log-y)", fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def plot_mean(data, metric, out_path):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for key, disp, col, ls, mk in SERIES:
        if not present(data, key):
            continue
        ys = corr_sev_mean(data, key, metric)
        lw = 1.4 if key in ("Uncalibrated", "TS", "Logistic") else 2.2
        ax.plot(range(1, 6), ys, color=col, ls=ls, marker=mk, ms=6, lw=lw, label=disp,
                zorder=2 if key == "Uncalibrated" else 3)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(["S1", "S2", "S3", "S4", "S5"], fontsize=11)
    ax.set_xlabel("Corruption severity (mean over 7 corruptions)", fontsize=12)
    ax.set_ylabel(f"{metric} ↓  (log scale)", fontsize=12)
    ax.set_title(f"Pepper {metric} under shift — 11 methods, fair protocol\n"
                 "(test93, native, float64)", fontsize=13)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, ncol=2, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal_dir", default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native")
    ap.add_argument("--uq_dir", default=f"{ROOT}/results/lora_paper/calibration_shift/native_float64")
    ap.add_argument("--out_dir", default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native")
    args = ap.parse_args()
    data = load_all(args.cal_dir, args.uq_dir)
    print(f"[load] present: {[k for k,*_ in SERIES if present(data,k)]}", flush=True)
    for metric in ("Brier", "NLL"):
        plot_grid(data, metric, os.path.join(args.out_dir, f"shift_11_{metric.lower()}_grid.png"))
        plot_mean(data, metric, os.path.join(args.out_dir, f"shift_11_{metric.lower()}_mean.png"))


if __name__ == "__main__":
    main()
