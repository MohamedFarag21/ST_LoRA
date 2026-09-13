# -*- coding: utf-8 -*-
"""
FAIR 11-method comparison of calibration under distribution shift.
ALL methods on the SAME protocol: test93, native 1280x720, float64 StreamBinMetrics ECE.
=> ECE/ACE are now directly comparable across every method (no float32/val33 caveat).

Reads:
  calibrators (7): posthoc_calibration_pepper/fit320_native/seed_*/shift_metrics_native_test93.json
  UQ methods  (4): lora_paper/calibration_shift/native_float64/<m>_seed*_shift_native.json
  LoRA-scale (opt, 2 lines): lora_paper/calibration_shift/lora_scale/seed_*_scale_shift.json
                             -> "LoRA-scale*" (tuned) and (folded into ST-LoRA family)

Writes (into fit320_native/):
  shift_fair_summary.md / .csv
  shift_fair_ece_mean.png    headline: mean ECE over 7 corruptions vs severity (log-y)
  shift_fair_ece_grid.png    ECE per corruption (7 panels + legend), log-y
Enhanced design: log-y (tames the fragile blow-up), validated family-coded palette
(solid=post-hoc calibrators, dashed=UQ methods), Uncalibrated recessive.
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
from matplotlib.lines import Line2D

ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
SEEDS = [42, 123, 456, 789, 1337]
CORRUPTIONS = ["blur", "noise", "brightness", "contrast",
               "saturation", "translation", "rotation"]
CORR_DISPLAY = {c: c.capitalize() for c in CORRUPTIONS}
SEVS = ["1", "2", "3", "4", "5"]
SEV_X = ["S1", "S2", "S3", "S4", "S5"]

# (key, display, color, linestyle, marker, family)
SERIES = [
    ("Uncalibrated", "Uncalibrated",      "#9a9a94", ":",  "o", "cal"),
    ("TS",           "Temp. Scaling",     "#2a78d6", "-",  "s", "cal"),
    ("Logistic",     "Logistic",          "#eb6834", "-",  "^", "cal"),
    ("Dirichlet",    "Dirichlet",         "#1baf7a", "-",  "D", "cal"),
    ("LTS",          "Local Temp (LTS)",  "#eda100", "-",  "v", "cal"),
    ("Meta",         "Meta-Cal",          "#e87ba4", "-",  "P", "cal"),
    ("Selective",    "Selective Scaling", "#008300", "-",  "X", "cal"),
    ("lora",         "ST-LoRA",           "#4a3aa7", "--", "*", "uq"),
    ("fullft",       "FRE (Full-FT)",     "#e34948", "--", "h", "uq"),
    ("mcdropout",    "MC-Dropout",        "#17a2b8", "--", "<", "uq"),
    ("ddu",          "DDU",               "#8c564b", "--", ">", "uq"),
]
CAL_KEYS = {"Uncalibrated", "TS", "Logistic", "Dirichlet", "LTS", "Meta", "Selective"}


def load_all(cal_dir, uq_dir, scale_dir=None):
    """Return per-seed dict: data[seed][method][corr][sev][metric] = value."""
    data = {s: {} for s in SEEDS}
    # calibrators
    for s in SEEDS:
        p = os.path.join(cal_dir, f"seed_{s}", "shift_metrics_native_test93.json")
        if not os.path.exists(p):
            continue
        grid = json.load(open(p))["grid"]
        for m in CAL_KEYS:
            data[s][m] = {c: {sev: grid[c][sev][m] for sev in grid.get(c, {})}
                          for c in CORRUPTIONS if c in grid}
    # UQ methods
    for m in ("lora", "fullft", "mcdropout", "ddu"):
        for s in SEEDS:
            p = os.path.join(uq_dir, f"{m}_seed{s}_shift_native.json")
            if not os.path.exists(p):
                continue
            grid = json.load(open(p))["grid"]
            data[s][m] = {c: {sev: grid[c][sev] for sev in grid.get(c, {})}
                          for c in CORRUPTIONS if c in grid}
    # optional LoRA-scale (tuned) -> "lora_scale"
    if scale_dir:
        for s in SEEDS:
            p = os.path.join(scale_dir, f"seed_{s}_scale_shift.json")
            if not os.path.exists(p):
                continue
            g = json.load(open(p))["variants"]["tuned"]["grid"]
            data[s]["lora_scale"] = {c: {sev: g[c][sev] for sev in g.get(c, {})}
                                     for c in CORRUPTIONS if c in g}
    return data


def sev_mean(data, method, corr, metric):
    """mean over seeds at each severity S1..S5 -> list of 5 (nan if absent)."""
    out = []
    for sev in SEVS:
        vals = [data[s][method][corr][sev][metric]
                for s in SEEDS
                if method in data[s] and corr in data[s][method]
                and sev in data[s][method][corr]
                and data[s][method][corr][sev].get(metric) is not None]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def corr_sev_mean(data, method, metric):
    """mean over 7 corruptions AND seeds at each severity -> list of 5."""
    per_sev = []
    for i, sev in enumerate(SEVS):
        vals = []
        for c in CORRUPTIONS:
            m = sev_mean(data, method, c, metric)[i]
            if not np.isnan(m):
                vals.append(m)
        per_sev.append(float(np.mean(vals)) if vals else float("nan"))
    return per_sev


def present(data, key):
    return any(key in data[s] for s in SEEDS)


def plot_headline(data, out_path, has_scale):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for key, disp, col, ls, mk, fam in SERIES:
        if not present(data, key):
            continue
        ys = corr_sev_mean(data, key, "ECE")
        lw = 1.4 if key in ("Uncalibrated", "TS", "Logistic") else 2.2
        ax.plot(range(1, 6), ys, color=col, ls=ls, marker=mk, ms=6, lw=lw,
                label=disp, alpha=0.9 if key == "Uncalibrated" else 1.0,
                zorder=2 if key == "Uncalibrated" else 3)
    if has_scale and present(data, "lora_scale"):
        ys = corr_sev_mean(data, "lora_scale", "ECE")
        ax.plot(range(1, 6), ys, color="#111111", ls="-.", marker="*", ms=9, lw=2.6,
                label="LoRA-scale* (tuned)", zorder=5)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(SEV_X, fontsize=11)
    ax.set_xlabel("Corruption severity (mean over 7 corruptions)", fontsize=12)
    ax.set_ylabel("ECE ↓  (log scale)", fontsize=12)
    ax.set_title("Pepper calibration under shift — 11 methods, fair protocol\n"
                 "(test93, native, float64)", fontsize=13)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, ncol=2, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def plot_grid(data, out_path, has_scale):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), squeeze=False)
    extra = [("lora_scale", "LoRA-scale* (tuned)", "#111111", "-.", "*")] if \
        (has_scale and present(data, "lora_scale")) else []
    for i, corr in enumerate(CORRUPTIONS):
        ax = axes[i // 4][i % 4]
        for key, disp, col, ls, mk, fam in SERIES:
            if not present(data, key):
                continue
            ys = sev_mean(data, key, corr, "ECE")
            lw = 1.3 if key in ("Uncalibrated", "TS", "Logistic") else 2.0
            ax.plot(range(1, 6), ys, color=col, ls=ls, marker=mk, ms=5, lw=lw,
                    zorder=2 if key == "Uncalibrated" else 3)
        for key, disp, col, ls, mk in extra:
            ax.plot(range(1, 6), sev_mean(data, key, corr, "ECE"),
                    color=col, ls=ls, marker=mk, ms=7, lw=2.4, zorder=5)
        ax.set_yscale("log")
        ax.set_title(CORR_DISPLAY[corr], fontsize=13)
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(SEV_X, fontsize=9)
        ax.set_ylabel("ECE ↓ (log)", fontsize=10); ax.set_xlabel("Severity", fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lg = axes[1][3]; lg.axis("off")
    handles = [Line2D([0], [0], color=c, ls=ls, marker=mk, ms=7, lw=2, label=d)
               for k, d, c, ls, mk, f in SERIES if present(data, k)]
    handles += [Line2D([0], [0], color=c, ls=ls, marker=mk, ms=8, lw=2.4, label=d)
                for k, d, c, ls, mk in extra]
    lg.legend(handles=handles, loc="center", fontsize=11, framealpha=0.9,
              title="Method  (solid = post-hoc calibrator, dashed = UQ)", title_fontsize=11)
    fig.suptitle("Pepper ECE under distribution shift — fair 11-method comparison "
                 "(test93, native, float64, log-y)", fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)


def write_tables(data, out_dir, has_scale):
    keys = [(k, d) for k, d, *_ in SERIES if present(data, k)]
    if has_scale and present(data, "lora_scale"):
        keys.append(("lora_scale", "LoRA-scale* (tuned)"))
    lines = ["# Fair 11-method calibration under shift — test93, native, float64", "",
             "All methods share ONE protocol (test93 / native / float64 StreamBinMetrics),",
             "so ECE/ACE are directly comparable. 5 seeds.", "",
             "| Method | Clean ECE | Mean-corr ECE | Mean-corr mIoU |", "|---|---|---|---|"]
    # clean from calibrator/uq files: severity key "0" under grid.clean — reload quickly
    for key, disp in keys:
        allece = [v for c in CORRUPTIONS for v in sev_mean(data, key, c, "ECE")
                  if not np.isnan(v)]
        allmiou = [v for c in CORRUPTIONS for v in sev_mean(data, key, c, "mIoU")
                   if not np.isnan(v)]
        mc = np.mean(allece) if allece else float("nan")
        mm = np.mean(allmiou) if allmiou else float("nan")
        lines.append(f"| {disp} | — | {mc:.4f} | {mm:.4f} |")
    lines += ["", "## Mean ECE per corruption (mean over 5 severities + seeds)", "",
              "| Method | " + " | ".join(CORR_DISPLAY[c] for c in CORRUPTIONS) + " |",
              "|---|" + "---|" * len(CORRUPTIONS)]
    for key, disp in keys:
        cells = []
        for c in CORRUPTIONS:
            v = [x for x in sev_mean(data, key, c, "ECE") if not np.isnan(x)]
            cells.append(f"{np.mean(v):.4f}" if v else "—")
        lines.append(f"| {disp} | " + " | ".join(cells) + " |")
    open(os.path.join(out_dir, "shift_fair_summary.md"), "w").write("\n".join(lines))
    csv = ["method,corruption,severity,metric,mean"]
    for key, disp in keys:
        for c in CORRUPTIONS:
            for i, sev in enumerate(SEVS):
                for metric in ("ECE", "ACE", "mIoU"):
                    v = sev_mean(data, key, c, metric)[i]
                    if not np.isnan(v):
                        csv.append(f"{key},{c},s{sev},{metric},{v:.6f}")
    open(os.path.join(out_dir, "shift_fair_summary.csv"), "w").write("\n".join(csv))
    print("\n".join(lines))
    print("[saved] shift_fair_summary.md/.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal_dir", default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native")
    ap.add_argument("--uq_dir", default=f"{ROOT}/results/lora_paper/calibration_shift/native_float64")
    ap.add_argument("--scale_dir", default=f"{ROOT}/results/lora_paper/calibration_shift/lora_scale")
    ap.add_argument("--out_dir", default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native")
    args = ap.parse_args()
    has_scale = os.path.isdir(args.scale_dir) and bool(
        glob.glob(os.path.join(args.scale_dir, "seed_*_scale_shift.json")))
    data = load_all(args.cal_dir, args.uq_dir, args.scale_dir if has_scale else None)
    present_keys = [k for k, *_ in SERIES if present(data, k)]
    print(f"[load] methods present: {present_keys}  lora_scale={has_scale}", flush=True)
    write_tables(data, args.out_dir, has_scale)
    plot_headline(data, os.path.join(args.out_dir, "shift_fair_ece_mean.png"), has_scale)
    plot_grid(data, os.path.join(args.out_dir, "shift_fair_ece_grid.png"), has_scale)


if __name__ == "__main__":
    main()
