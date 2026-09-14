# -*- coding: utf-8 -*-
"""
Plot the ensemble-SIZE ablation (1..10 members) for FRE vs ST-LoRA on bup20 test93 @native.
Reads eval/{fre,stlora}_seed<S>_enssize.json (per_k metrics), aggregates mean±std over seeds,
renders mIoU / NLL / ECE vs #members with shaded std bands. SLURM only (ssl env).
"""
import os
import sys
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
E = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/lora_paper/ensemble_size_ablation/eval"
CADENCE = sys.argv[2] if len(sys.argv) > 2 else "T0=10, snapshot/10ep"
METHODS = [("fre", "FRE (Full-FT)", "#d62728"), ("stlora", "ST-LoRA", "#1f77b4")]
PANELS = [("mIoU", "mIoU ↑", False), ("NLL", "NLL ↓", True), ("ECE", "ECE ↓", True)]


def load(method):
    ks, per = None, {}
    for f in sorted(glob.glob(os.path.join(E, f"{method}_seed*_enssize.json"))):
        d = json.load(open(f))
        s = d["seed"]
        per[s] = d["per_k"]
        ks = sorted(int(k) for k in d["per_k"].keys())
    return ks, per


def agg(per, ks, metric):
    mean, std = [], []
    for k in ks:
        vals = [per[s][str(k)][metric] for s in per if per[s].get(str(k)) is not None]
        vals = [v for v in vals if v is not None]
        m = float(np.mean(vals)); sd = float(np.std(vals))
        mean.append(m); std.append(sd)
    return np.array(mean), np.array(std)


def main():
    data = {m: load(m) for m, _, _ in METHODS}
    seeds = sorted(next(iter(data.values()))[1].keys())
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 4.6))
    for ax, (metric, ylabel, logy) in zip(axes, PANELS):
        for m, label, color in METHODS:
            ks, per = data[m]
            mean, std = agg(per, ks, metric)
            ax.plot(ks, mean, "-o", color=color, label=label, lw=2, ms=5)
            ax.fill_between(ks, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_xlabel("ensemble members (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(range(1, 11))
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=9)
    fig.suptitle(f"Ensemble-size ablation ({CADENCE}) — FRE vs ST-LoRA, "
                 f"bup20 test93 @native, {len(seeds)} seeds", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(E, "ens_size_ablation_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
