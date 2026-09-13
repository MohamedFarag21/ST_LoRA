# -*- coding: utf-8 -*-
"""
Cross-method aggregation of the tomato pixel-level OoD evals (tomato fruit =
anomaly).  Combines all methods that were run with the identical
TomatoPixelDataset harness:

  std-LoRA   : pepper_to_tomato_pixelood_seed<S>_results.json   (entropy, MI)
  Full-FT    : full_ft_tomato_pixelood_seed<S>_results.json     (entropy, MI)
  MC-Dropout : mcdropout_tomato_pixelood_seed<S>_results.json   (entropy, MI)
  DDU        : ddu_tomato_pixelood_seed<S>_results.json         (entropy,
                                                                 density_spatial,
                                                                 density_query)
  ADE-base   : ade_base_tomato_pixelood_results.json            (entropy)  [single run]

Produces:
  * a primary ENTROPY table (mean +/- std over seeds) with one row per method,
  * per-method extra-scorer tables (MI for the ensemble methods; the two GMM
    density scorers for DDU),
  * a JSON dump with per-seed values.

All methods share the pkl-derived tomato mask (bg=0, fruit>0) and are NOT
affected by the GrowliFlower palette bug.

Usage:
    python aggregate_tomato_pixel_ood_all_methods.py [--results_dir ...] [--seeds ...]
"""

import os
import json
import argparse

import numpy as np

METRICS = ["AUROC", "AUPR", "FPR95", "sIoU", "PPV", "MeanF1"]

# label -> (filename prefix, is_seeded)
METHODS = [
    ("std-LoRA",   "pepper_to_tomato_pixelood_seed", True),
    ("Full-FT",    "full_ft_tomato_pixelood_seed",   True),
    ("MC-Dropout", "mcdropout_tomato_pixelood_seed", True),
    ("DDU",        "ddu_tomato_pixelood_seed",       True),
    ("ADE-base",   "ade_base_tomato_pixelood",       False),  # single, no seed suffix
]


def load_method(results_dir, prefix, is_seeded, seeds):
    """Return {seed: pixel_level_dict}. Non-seeded methods key on 'single'."""
    out = {}
    if is_seeded:
        for s in seeds:
            p = os.path.join(results_dir, f"{prefix}{s}_results.json")
            if os.path.isfile(p):
                with open(p) as f:
                    out[s] = json.load(f)["pixel_level"]
    else:
        p = os.path.join(results_dir, f"{prefix}_results.json")
        if os.path.isfile(p):
            with open(p) as f:
                out["single"] = json.load(f)["pixel_level"]
    return out


def agg_scorer(per_seed, scorer):
    """mean/std over seeds for one scorer; returns (stats_dict, per_seed_values) or None."""
    rows = {sk: pv[scorer] for sk, pv in per_seed.items() if scorer in pv}
    if not rows:
        return None
    stats, per = {}, {}
    for k in METRICS:
        arr = np.array([rows[sk][k] for sk in rows if k in rows[sk]], dtype=float)
        if arr.size == 0:
            continue
        stats[k] = {"mean": float(arr.mean()), "std": float(arr.std())}
        per[k] = {str(sk): float(rows[sk][k]) for sk in rows if k in rows[sk]}
    return {"stats": stats, "per_seed": per, "n": len(rows)}


def fmt(stats, k):
    if k not in stats:
        return "--"
    return f"{stats[k]['mean']:.4f} +/- {stats[k]['std']:.4f}"


def main():
    ap = argparse.ArgumentParser(description="Aggregate tomato pixel-OoD across all methods")
    ap.add_argument("--results_dir", type=str,
        default="/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204/results/tomato_lora/pepper_transfer")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    ap.add_argument("--out_md",  type=str, default=None)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    rd = args.results_dir
    loaded = {label: load_method(rd, pre, seeded, args.seeds)
              for label, pre, seeded in METHODS}

    summary = {}
    lines = []
    lines.append("# Tomato pixel-level OoD — all methods (tomato fruit = anomaly)")
    lines.append("")
    lines.append("Mean +/- std over seeds. Anomaly = any tomato foreground pixel (`sem>0`);")
    lines.append("pkl-derived masks, unaffected by the GrowliFlower palette bug.")
    lines.append("")

    # ---- availability ----
    lines.append("## Seeds available")
    for label, _, _ in METHODS:
        ks = sorted(str(k) for k in loaded[label].keys())
        lines.append(f"- **{label}**: {len(ks)} run(s) {ks}")
    lines.append("")

    # ---- primary entropy table ----
    lines.append("## Primary scorer: entropy")
    lines.append("")
    lines.append("| method | n | " + " | ".join(METRICS) + " |")
    lines.append("|--------|---|" + "|".join(["-------"] * len(METRICS)) + "|")
    for label, _, _ in METHODS:
        a = agg_scorer(loaded[label], "entropy")
        summary.setdefault(label, {})["entropy"] = a
        if a is None:
            lines.append(f"| {label} | 0 | " + " | ".join(["--"] * len(METRICS)) + " |")
            continue
        cells = " | ".join(fmt(a["stats"], k) for k in METRICS)
        lines.append(f"| {label} | {a['n']} | {cells} |")
    lines.append("")

    # ---- extra scorers ----
    EXTRA = {
        "MI": ["std-LoRA", "Full-FT", "MC-Dropout"],
        "density_spatial": ["DDU"],
        "density_query": ["DDU"],
    }
    for scorer, labels in EXTRA.items():
        lines.append(f"## Scorer: {scorer}")
        lines.append("")
        lines.append("| method | n | " + " | ".join(METRICS) + " |")
        lines.append("|--------|---|" + "|".join(["-------"] * len(METRICS)) + "|")
        for label in labels:
            a = agg_scorer(loaded[label], scorer)
            summary.setdefault(label, {})[scorer] = a
            if a is None:
                lines.append(f"| {label} | 0 | " + " | ".join(["--"] * len(METRICS)) + " |")
                continue
            cells = " | ".join(fmt(a["stats"], k) for k in METRICS)
            lines.append(f"| {label} | {a['n']} | {cells} |")
        lines.append("")

    out_md = args.out_md or os.path.join(rd, "tomato_pixel_ood_all_methods_summary.md")
    out_json = args.out_json or os.path.join(rd, "tomato_pixel_ood_all_methods_summary.json")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    with open(out_json, "w") as f:
        json.dump({"seeds_requested": args.seeds, "methods": summary}, f, indent=2)

    print("\n".join(lines))
    print(f"\n[Saved] {out_md}")
    print(f"[Saved] {out_json}")


if __name__ == "__main__":
    main()
