# -*- coding: utf-8 -*-
"""
Overlay the 7 post-hoc calibrators (this study) with the 4 UQ methods from the
lora_paper study -> 11 lines per corruption, matching the reference figure.

*** MEASUREMENT CAVEAT (baked into the figure) ***
The two families are NOT measured identically:
  - post-hoc calibrators (SOLID lines): eval on val(33), float64 StreamBinMetrics ECE;
  - UQ methods FRE/ST-LoRA/MC-Dropout/DDU (DASHED lines): eval on test(93), torchmetrics
    float32 ECE (inflated up to 2-3x above 16.7M px/bin; test93 = 85.7M px is over that
    line -- see reference_torchmetrics_float32_ece).
Both are native 1280x720, same 8-class pepper map, same 7x5 grid, same 5 seeds.
=> mIoU is comparable; ECE/ACE differences ACROSS families are not reliable. This overlay
   was requested as a quick view; the rigorous version re-runs all 11 on test93+float64.

Reads  fit320_native/summary_shift_native.json          (7 calibrators)
       lora_paper/calibration_shift/calibration_shift_summary.json  (4 UQ methods)
Writes fit320_native/calibration_shift_native_11methods.png   (mIoU/ECE/ACE x 7 corr)
       fit320_native/calibration_shift_native_11methods_ece.png (ECE-only vs severity)
Run via SLURM (ssl). No direct python.
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")

# (display, color, linestyle, marker, family)  — order = legend order
SERIES = [
    # post-hoc calibrators — SOLID (Uncalibrated dotted, recessive)
    ("Uncalibrated",     "Uncalibrated",       "#898781", ":",  "o", "posthoc"),
    ("TS",               "Temp. Scaling",      "#2a78d6", "-",  "s", "posthoc"),
    ("Logistic",         "Logistic",           "#eb6834", "-",  "^", "posthoc"),
    ("Dirichlet",        "Dirichlet",          "#1baf7a", "-",  "D", "posthoc"),
    ("LTS",              "Local Temp (LTS)",   "#eda100", "-",  "v", "posthoc"),
    ("Meta",             "Meta-Cal",           "#e87ba4", "-",  "P", "posthoc"),
    ("Selective",        "Selective Scaling",  "#008300", "-",  "X", "posthoc"),
    # UQ methods (lora_paper) — DASHED, distinct hues
    ("lora",             "ST-LoRA",            "#4a3aa7", "--", "*", "uq"),
    ("fullft",           "FRE (Full-FT)",      "#e34948", "--", "h", "uq"),
    ("mcdropout",        "MC-Dropout",         "#17becf", "--", "<", "uq"),
    ("ddu",              "DDU",                "#8c564b", "--", ">", "uq"),
]
CORRUPTIONS = ["blur", "noise", "brightness", "contrast",
               "saturation", "translation", "rotation"]
CORR_DISPLAY = {c: c.capitalize() for c in CORRUPTIONS}
METRIC_ROWS = ["mIoU", "ECE", "ACE"]
METRIC_YLABEL = {"mIoU": "mIoU ↑", "ECE": "ECE ↓", "ACE": "ACE ↓"}
SEVS = ["s1", "s2", "s3", "s4", "s5"]
SEV_X = ["S1", "S2", "S3", "S4", "S5"]
CAVEAT = ("Solid = post-hoc calibrators (val33, float64 ECE).   "
          "Dashed = UQ methods (test93, float32 ECE — inflated up to 2–3× above 16.7M px/bin).   "
          "mIoU comparable; ECE/ACE NOT comparable across families.")


def sev_series(summary, key, corr, metric):
    out = []
    for s in SEVS:
        try:
            out.append(summary[key][corr][s][metric]["mean"])
        except (KeyError, TypeError):
            out.append(None)
    return out


def plot_grid(cal, uq, out_path):
    src = {"posthoc": cal, "uq": uq}
    nr, nc = len(METRIC_ROWS), len(CORRUPTIONS)
    fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc + 1.3, 3.2 * nr + 0.6),
                             squeeze=False)
    xs = list(range(5))
    for ri, metric in enumerate(METRIC_ROWS):
        for ci, corr in enumerate(CORRUPTIONS):
            ax = axes[ri][ci]
            for key, disp, col, ls, mk, fam in SERIES:
                ys = [v if v is not None else float("nan")
                      for v in sev_series(src[fam], key, corr, metric)]
                if all(np.isnan(ys)):
                    continue
                ax.plot(xs, ys, color=col, lw=1.6, ls=ls, marker=mk, ms=5,
                        alpha=0.95, zorder=2 if key == "Uncalibrated" else 3)
            ax.set_xticks(xs); ax.set_xticklabels(SEV_X, fontsize=9)
            if ri == 0:
                ax.set_title(CORR_DISPLAY[corr], fontsize=12, pad=4)
            if ci == 0:
                ax.set_ylabel(METRIC_YLABEL[metric], fontsize=12)
            if ri == nr - 1:
                ax.set_xlabel("Severity", fontsize=11)
            ax.tick_params(labelsize=9, length=2.5)
            ax.grid(True, ls="--", alpha=0.35)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    handles = [Line2D([0], [0], color=c, ls=ls, marker=mk, ms=6, lw=1.8, label=d)
               for _, d, c, ls, mk, _ in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Pepper: post-hoc calibration vs UQ methods under distribution shift "
                 "(native 1280×720, 5 seeds)", fontsize=14, y=1.00)
    fig.text(0.5, 0.045, CAVEAT, ha="center", va="bottom", fontsize=9,
             style="italic", color="#52514e")
    fig.tight_layout(h_pad=1.8, w_pad=1.0, rect=[0, 0.075, 1, 0.99])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def plot_ece_grid(cal, uq, out_path):
    """ECE-only, one panel per corruption (4x2), 11 lines — the headline view."""
    src = {"posthoc": cal, "uq": uq}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4), squeeze=False)
    xs = list(range(5))
    for i, corr in enumerate(CORRUPTIONS):
        ax = axes[i // 4][i % 4]
        for key, disp, col, ls, mk, fam in SERIES:
            ys = [v if v is not None else float("nan")
                  for v in sev_series(src[fam], key, corr, "ECE")]
            if all(np.isnan(ys)):
                continue
            ax.plot(xs, ys, color=col, lw=1.8, ls=ls, marker=mk, ms=6,
                    zorder=2 if key == "Uncalibrated" else 3)
        ax.set_title(CORR_DISPLAY[corr], fontsize=13)
        ax.set_xticks(xs); ax.set_xticklabels(SEV_X, fontsize=10)
        ax.set_ylabel("ECE ↓", fontsize=11); ax.set_xlabel("Severity", fontsize=11)
        ax.grid(True, ls="--", alpha=0.35)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # 8th cell -> legend
    lg = axes[1][3]
    lg.axis("off")
    handles = [Line2D([0], [0], color=c, ls=ls, marker=mk, ms=7, lw=2, label=d)
               for _, d, c, ls, mk, _ in SERIES]
    lg.legend(handles=handles, loc="center", fontsize=11, framealpha=0.9,
              title="Method", title_fontsize=12)
    fig.suptitle("Pepper ECE under distribution shift — 7 calibrators (solid) + "
                 "4 UQ methods (dashed)", fontsize=15, y=1.00)
    fig.text(0.5, 0.005, CAVEAT, ha="center", va="bottom", fontsize=9.5,
             style="italic", color="#52514e")
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal_json",
                    default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native/summary_shift_native.json")
    ap.add_argument("--uq_json",
                    default=f"{ROOT}/results/lora_paper/calibration_shift/calibration_shift_summary.json")
    ap.add_argument("--out_dir",
                    default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native")
    args = ap.parse_args()
    cal = json.load(open(args.cal_json))
    uq = json.load(open(args.uq_json))
    print(f"[load] calibrators={list(cal)} uq={list(uq)}", flush=True)
    plot_grid(cal, uq, os.path.join(args.out_dir, "calibration_shift_native_11methods.png"))
    plot_ece_grid(cal, uq, os.path.join(args.out_dir, "calibration_shift_native_11methods_ece.png"))


if __name__ == "__main__":
    main()
