# -*- coding: utf-8 -*-
"""
Aggregate the 5-seed fit@320 -> eval@native SHIFT run and plot it.

Reads  fit320_native/seed_*/shift_metrics_native.json
Writes fit320_native/
  shift_summary_native.md          Clean + Mean-corr ECE, per-corruption ECE table
  shift_summary_native.csv         long form (method,corruption,severity,metric,mean,std)
  summary_shift_native.json        {method:{corruption:{s1..s5:{metric:{mean,std}}}}}
  calibration_shift_native_all.png 3 rows (mIoU/ECE/ACE) x 7 corruptions, 7 method lines
  calibration_shift_native_ece_mean.png  headline: mean-over-corruptions ECE vs severity

Post-hoc calibrators (7 methods): Uncalibrated + TS/Logistic/Dirichlet/LTS/Meta/Selective.
Fit@320 on clean cal(30); evaluated at native 1280x720 under the 7x5 grid.
Style follows the lora_paper calibration_shift figure. Run via SLURM (ssl). No direct python.
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")

SEEDS = [42, 123, 456, 789, 1337]
METHODS = ["Uncalibrated", "TS", "Logistic", "Dirichlet", "LTS", "Meta", "Selective"]
METHOD_DISPLAY = {"Uncalibrated": "Uncalibrated", "TS": "Temp. Scaling",
                  "Logistic": "Logistic", "Dirichlet": "Dirichlet",
                  "LTS": "Local Temp (LTS)", "Meta": "Meta-Cal",
                  "Selective": "Selective Scaling"}
# Uncalibrated = neutral dashed baseline; 6 calibrators take validated categorical slots 1-6.
METHOD_COLOR = {"Uncalibrated": "#898781", "TS": "#2a78d6", "Logistic": "#eb6834",
                "Dirichlet": "#1baf7a", "LTS": "#eda100", "Meta": "#e87ba4",
                "Selective": "#008300"}
METHOD_MARKER = {"Uncalibrated": "o", "TS": "s", "Logistic": "^", "Dirichlet": "D",
                 "LTS": "v", "Meta": "P", "Selective": "X"}
METHOD_LS = {"Uncalibrated": "--"}          # rest solid

CORRUPTIONS = ["blur", "noise", "brightness", "contrast",
               "saturation", "translation", "rotation"]
CORR_DISPLAY = {c: c.capitalize() for c in CORRUPTIONS}
METRIC_ROWS = ["mIoU", "ECE", "ACE"]
METRIC_YLABEL = {"mIoU": "mIoU ↑", "ECE": "ECE ↓", "ACE": "ACE ↓"}
SEVS = ["1", "2", "3", "4", "5"]
SEV_X = ["S1", "S2", "S3", "S4", "S5"]

# style
DPI, LABEL_FONT, TICK_FONT, LEGEND_FONT = 300, 12, 10, 11
LINE_WIDTH, MARKER_SIZE, GRID_ALPHA = 1.7, 5, 0.35


def load_runs(res_dir):
    runs = {}
    for s in SEEDS:
        p = os.path.join(res_dir, f"seed_{s}", "shift_metrics_native.json")
        if os.path.exists(p):
            runs[s] = json.load(open(p))
    return runs


def cell(runs, corr, sev_key, method, metric):
    """List of per-seed values for grid[corr][sev_key][method][metric]."""
    vals = []
    for r in runs.values():
        try:
            v = r["grid"][corr][sev_key][method][metric]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        except (KeyError, TypeError):
            pass
    return vals


def build_summary(runs):
    """summary[method][corruption][s1..s5][metric] = {mean,std,n}."""
    summary = {}
    for m in METHODS:
        summary[m] = {}
        for c in CORRUPTIONS:
            summary[m][c] = {}
            for i, sev in enumerate(SEVS, start=1):
                d = {}
                for metric in ("mIoU", "ECE", "ACE", "acc"):
                    vals = cell(runs, c, sev, m, metric)
                    if vals:
                        d[metric] = {"mean": float(np.mean(vals)),
                                     "std": float(np.std(vals)), "n": len(vals)}
                summary[m][c][f"s{i}"] = d
    return summary


def clean_stats(runs, method, metric):
    vals = cell(runs, "clean", "0", method, metric)
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)


def means_over_sev(summary, method, corruption, metric):
    return [summary[method][corruption][f"s{i}"].get(metric, {}).get("mean")
            for i in range(1, 6)]


# ── figure 1: full grid (reference style) ────────────────────────────────────
def plot_grid(summary, out_path):
    nr, nc = len(METRIC_ROWS), len(CORRUPTIONS)
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc + 1.2, 3.1 * nr + 0.4),
                             squeeze=False)
    xs = list(range(5))
    for ri, metric in enumerate(METRIC_ROWS):
        for ci, corr in enumerate(CORRUPTIONS):
            ax = axes[ri][ci]
            for m in METHODS:
                ys = [v if v is not None else float("nan")
                      for v in means_over_sev(summary, m, corr, metric)]
                if all(np.isnan(ys)):
                    continue
                ax.plot(xs, ys, color=METHOD_COLOR[m], lw=LINE_WIDTH,
                        ls=METHOD_LS.get(m, "-"), marker=METHOD_MARKER[m],
                        ms=MARKER_SIZE, label=METHOD_DISPLAY[m],
                        alpha=0.95, zorder=2 if m == "Uncalibrated" else 3)
            ax.set_xticks(xs); ax.set_xticklabels(SEV_X, fontsize=TICK_FONT)
            if ri == 0:
                ax.set_title(CORR_DISPLAY[corr], fontsize=LABEL_FONT, pad=4)
            if ci == 0:
                ax.set_ylabel(METRIC_YLABEL[metric], fontsize=LABEL_FONT)
            if ri == nr - 1:
                ax.set_xlabel("Severity", fontsize=LABEL_FONT)
            ax.tick_params(labelsize=TICK_FONT, length=2.5)
            ax.grid(True, ls="--", alpha=GRID_ALPHA)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS),
               fontsize=LEGEND_FONT, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Pepper post-hoc calibration under distribution shift  "
                 "(fit@320 → eval@native 1280×720, 5 seeds)",
                 fontsize=LABEL_FONT + 2, y=1.00)
    fig.tight_layout(h_pad=1.8, w_pad=1.0, rect=[0, 0.05, 1, 0.99])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


# ── figure 2: headline — mean ECE over corruptions vs severity ───────────────
def plot_ece_mean(summary, runs, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = list(range(6))                      # 0 = clean, 1..5 = severities
    for m in METHODS:
        ys, es = [], []
        cm, cs = clean_stats(runs, m, "ECE")
        ys.append(cm); es.append(cs)
        for i in range(1, 6):
            per = [summary[m][c][f"s{i}"].get("ECE", {}).get("mean") for c in CORRUPTIONS]
            per = [v for v in per if v is not None]
            ys.append(float(np.mean(per)) if per else float("nan"))
            es.append(float(np.std(per)) if per else 0.0)
        ax.plot(xs, ys, color=METHOD_COLOR[m], lw=2.0, ls=METHOD_LS.get(m, "-"),
                marker=METHOD_MARKER[m], ms=6, label=METHOD_DISPLAY[m],
                zorder=2 if m == "Uncalibrated" else 3)
    ax.set_xticks(xs); ax.set_xticklabels(["Clean", "S1", "S2", "S3", "S4", "S5"],
                                          fontsize=TICK_FONT)
    ax.set_xlabel("Corruption severity (mean over 7 corruptions)", fontsize=LABEL_FONT)
    ax.set_ylabel("ECE ↓", fontsize=LABEL_FONT)
    ax.set_title("Pepper calibration under shift @native — mean ECE vs severity",
                 fontsize=LABEL_FONT + 1)
    ax.grid(True, ls="--", alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def write_tables(summary, runs, res_dir):
    # markdown: clean + mean-corr ECE, then per-corruption mean ECE
    lines = ["# Pepper post-hoc calibration under distribution shift — "
             "fit@320 → eval@native (5 seeds)", "",
             f"Seeds present: {sorted(runs)}. Eval = native 1280×720, val(33) "
             "corrupted over the 7×5 grid; calibrators fit once on clean cal(30)@320.",
             "ECE lower = better. 'Clean' = uncorrupted val(33); 'Mean-corr' = mean over "
             "all 7×5 cells.", "",
             "| Method | Clean ECE | Mean-corr ECE |", "|---|---|---|"]
    for m in METHODS:
        cm, cs = clean_stats(runs, m, "ECE")
        allv = [summary[m][c][f"s{i}"].get("ECE", {}).get("mean")
                for c in CORRUPTIONS for i in range(1, 6)]
        allv = [v for v in allv if v is not None]
        mc = float(np.mean(allv)) if allv else float("nan")
        lines.append(f"| {METHOD_DISPLAY[m]} | {cm:.4f} ± {cs:.4f} | {mc:.4f} |")
    lines += ["", "## Mean ECE per corruption (mean over 5 severities + seeds)", "",
              "| Method | " + " | ".join(CORR_DISPLAY[c] for c in CORRUPTIONS) + " |",
              "|---|" + "---|" * len(CORRUPTIONS)]
    for m in METHODS:
        cells = []
        for c in CORRUPTIONS:
            vv = [summary[m][c][f"s{i}"].get("ECE", {}).get("mean") for i in range(1, 6)]
            vv = [v for v in vv if v is not None]
            cells.append(f"{np.mean(vv):.4f}" if vv else "—")
        lines.append(f"| {METHOD_DISPLAY[m]} | " + " | ".join(cells) + " |")
    open(os.path.join(res_dir, "shift_summary_native.md"), "w").write("\n".join(lines))

    csv = ["method,corruption,severity,metric,mean,std,n"]
    for m in METHODS:
        cm, cs = clean_stats(runs, m, "ECE")
        csv.append(f"{m},clean,s0,ECE,{cm:.6f},{cs:.6f},{len(cell(runs,'clean','0',m,'ECE'))}")
        for c in CORRUPTIONS:
            for i in range(1, 6):
                for metric in ("mIoU", "ECE", "ACE", "acc"):
                    d = summary[m][c][f"s{i}"].get(metric)
                    if d:
                        csv.append(f"{m},{c},s{i},{metric},{d['mean']:.6f},{d['std']:.6f},{d['n']}")
    open(os.path.join(res_dir, "shift_summary_native.csv"), "w").write("\n".join(csv))
    json.dump(summary, open(os.path.join(res_dir, "summary_shift_native.json"), "w"),
              indent=2)
    print("[saved] shift_summary_native.md / .csv / summary_shift_native.json", flush=True)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir",
                    default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native")
    args = ap.parse_args()
    runs = load_runs(args.res_dir)
    if not runs:
        sys.exit("no shift_metrics_native.json found")
    print(f"[load] seeds {sorted(runs)}", flush=True)
    summary = build_summary(runs)
    write_tables(summary, runs, args.res_dir)
    plot_grid(summary, os.path.join(args.res_dir, "calibration_shift_native_all.png"))
    plot_ece_mean(summary, runs,
                  os.path.join(args.res_dir, "calibration_shift_native_ece_mean.png"))


if __name__ == "__main__":
    main()
