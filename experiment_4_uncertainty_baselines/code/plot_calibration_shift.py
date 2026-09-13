# -*- coding: utf-8 -*-
"""
Plot Calibration Under Distribution Shift
==========================================
Reads the aggregated summary JSON and produces one publication-quality figure:
  - 3 rows : mIoU (row 1), ECE (row 2), ACE (row 3)
  - 7 cols : one per corruption type
  - 4 lines per subplot : one per method
  - x-axis : severity S1 → S5 (beyond training augmentation range)
  - Only mean across 5 seeds is plotted (no std band)

Style matches plot_robustness.py exactly.

Usage:
    python plot_calibration_shift.py
    python plot_calibration_shift.py --summary_json /path/to/summary.json
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Style — matches plot_robustness.py exactly
# ─────────────────────────────────────────────────────────────────────────────

DPI         = 600
LABEL_FONT  = 13
TICK_FONT   = 11
TICK_SIZE   = 2.5
LEGEND_FONT = 12
LINE_WIDTH  = 1.8
MARKER_SIZE = 6
GRID_ALPHA  = 0.35
GRID_STYLE  = "--"

# One colour per method — fixed mapping
METHOD_COLORS = {
    "lora":      "#1f77b4",   # blue
    "fullft":    "#ff7f0e",   # orange
    "mcdropout": "#2ca02c",   # green
    "ddu":       "#d62728",   # red
}
METHOD_DISPLAY = {
    "lora":      "ST-LoRA",
    "fullft":    "FRE",
    "mcdropout": "MC Dropout",
    "ddu":       "DDU",
}
METHODS_ORDER = ["lora", "fullft", "mcdropout", "ddu"]

CORRUPTION_ORDER = [
    "blur", "noise", "brightness", "contrast",
    "saturation", "translation", "rotation",
]
CORRUPTION_DISPLAY = {
    "blur":        "Blur",
    "noise":       "Noise",
    "brightness":  "Brightness",
    "contrast":    "Contrast",
    "saturation":  "Saturation",
    "translation": "Translation",
    "rotation":    "Rotation",
}

METRIC_ROWS   = ["mIoU", "ECE", "ACE"]
METRIC_YLABEL = {
    "mIoU": "mIoU ↑",
    "ECE":  "ECE ↓",
    "ACE":  "ACE ↓",
}

SEV_LABELS = ["s1", "s2", "s3", "s4", "s5"]
SEV_XTICKS = ["S1", "S2", "S3", "S4", "S5"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_style(ax, xlabel="Severity", ylabel="", title=""):
    if title:
        ax.set_title(title, fontsize=LABEL_FONT, pad=5)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT)
    ax.tick_params(axis="both", labelsize=TICK_FONT, length=TICK_SIZE)
    ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_means(summary, method, corruption, metric):
    """Extract ordered mean values S1→S5 for (method, corruption, metric)."""
    means = []
    for sev in SEV_LABELS:
        entry = summary.get(method, {}).get(corruption, {}).get(sev, {}).get(metric, {})
        means.append(entry.get("mean") if entry else None)
    return means


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_shift(summary, out_path):
    n_rows = len(METRIC_ROWS)
    n_cols = len(CORRUPTION_ORDER)

    # Match per-group figure proportions from plot_robustness.py
    subplot_w = 18 / 7     # ~2.57 in per subplot (same as FIG_W/n_corruptions)
    subplot_h = 3.5        # same as FIG_H
    fig_w = subplot_w * n_cols + 1.5   # +1.5 for y-labels
    fig_h = subplot_h * n_rows + 0.6 * (n_rows - 1)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    xs = list(range(len(SEV_LABELS)))

    for row_idx, metric in enumerate(METRIC_ROWS):
        for col_idx, corruption in enumerate(CORRUPTION_ORDER):
            ax = axes[row_idx][col_idx]

            for method in METHODS_ORDER:
                if method not in summary:
                    continue
                means = get_means(summary, method, corruption, metric)
                if all(v is None for v in means):
                    continue
                # Replace None with nan so line breaks rather than crashing
                ys = [v if v is not None else float("nan") for v in means]

                ax.plot(
                    xs, ys,
                    color=METHOD_COLORS[method],
                    linewidth=LINE_WIDTH,
                    linestyle="-",
                    marker="o",
                    markersize=MARKER_SIZE,
                    label=METHOD_DISPLAY[method],
                )

            ax.set_xticks(xs)
            ax.set_xticklabels(
                SEV_XTICKS,
                fontsize=TICK_FONT,
            )

            # Column title: corruption name on top row only
            col_title = CORRUPTION_DISPLAY.get(corruption, corruption) \
                        if row_idx == 0 else ""

            # Y-label: metric name on left column only
            ylabel = METRIC_YLABEL[metric] if col_idx == 0 else ""

            # X-label: "Severity" on bottom row only
            xlabel = "Severity" if row_idx == n_rows - 1 else ""

            apply_style(ax, xlabel=xlabel, ylabel=ylabel, title=col_title)

    # ── Single shared legend below the figure ────────────────────────────────
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(METHODS_ORDER),
        fontsize=LEGEND_FONT,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout(h_pad=2.0, w_pad=1.2, rect=[0, 0.04, 1, 1])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot calibration under distribution shift")
    parser.add_argument("--summary_json", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/lora_paper/calibration_shift/calibration_shift_summary.json")
    parser.add_argument("--out_dir", type=str, default=None,
        help="Output directory. Default: same dir as summary_json")
    return parser.parse_args()


def main():
    args    = parse_args()
    out_dir = args.out_dir or os.path.dirname(args.summary_json)

    if not os.path.exists(args.summary_json):
        raise FileNotFoundError(f"Summary JSON not found: {args.summary_json}\n"
                                f"Run aggregate_calibration_shift.py first.")

    with open(args.summary_json) as f:
        summary = json.load(f)

    print(f"Loaded summary for methods: {list(summary.keys())}")
    print(f"Corruptions found: {list(next(iter(summary.values())).keys())}\n")

    out_path = os.path.join(out_dir, "calibration_shift_all_methods.png")
    plot_calibration_shift(summary, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
