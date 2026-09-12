# -*- coding: utf-8 -*-
"""
EXPERIMENT 1 — statistical significance of ST-LoRA vs FRE training COST, across the
three architectures {Mask2Former, SegFormer, EoMT}. Paired design (same 5 seeds run
under both methods).

Reports TWO bases, both from the SAME per-epoch-traced runs for every architecture
(byte-identical CO2 harness: tracker wraps whole main(); a Lightning callback marks the
train window [epoch_start, val_start) and flags epoch 0 as warmup):

  HEADLINE — whole training run (CodeCarbon totals over main(), incl. validation +
             warmup + sanity): CO2 / total energy / GPU energy / wall-clock / GPU power.
  SUPPORTING — training loop ONLY, warmup epoch trimmed (summed from the per-epoch CSV,
             validation excluded, epoch 0 dropped): GPU energy + wall-clock.

For every architecture × metric we pair by seed, d_i = FRE_i - ST-LoRA_i, and report:
  * FRE / ST-LoRA mean±sd, %Δ = 100*(ST-LoRA-FRE)/FRE  (negative => ST-LoRA lower)
  * paired t-test p, Wilcoxon p (n=5 floor 0.0625), Cohen's dz = mean(d)/sd(d)
  * 95% bootstrap CI of %Δ (20k resamples)
Sig (directional): paired-t p<0.05 AND %Δ CI excludes 0. ST-LoRA↓ = significantly lower
(greener); ST-LoRA↑ = significantly higher (worse). No Holm correction (per request).

Outputs (results/lora_paper/stats_crossarch/):
  co2_crossarch_significance.md / .tex   master table (headline + supporting)
  co2_crossarch_forest.png / .pdf        forest plot of %Δ ± 95% CI (line at 0)
  co2_crossarch_significance.json
CPU only (matplotlib Agg); lower is better for every metric here.
"""
import os
import csv
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as ss

R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
     "mibrahi2_hpc-my_research-1775524204")
SEEDS = [42, 123, 456, 789, 1337]
# Mask2Former -> co2_pe_v2 (fresh per-epoch-traced re-run; both bases from same runs)
ARCHS = [("Mask2Former", "co2_pe_v2", "#1f77b4"),
         ("SegFormer", "segformer_co2", "#2ca02c"),
         ("EoMT", "co2_eomt", "#d62728")]
WHOLE_METRICS = [("emissions_kgCO2e", "CO$_2$ (kg)"),
                 ("energy_consumed_kWh", "Total energy (kWh)"),
                 ("gpu_energy_kWh", "GPU energy (kWh)"),
                 ("wall_clock_s", "Wall-clock (s)"),
                 ("gpu_power_W", "GPU power (W)")]
TRAIN_METRICS = [("train_gpu_energy_kWh", "GPU energy (train-only)"),
                 ("train_wall_s", "Wall-clock (train-only)")]
np.random.seed(0)


def load_whole(arch_dir, method, seed, key):
    p = f"{R}/results/lora_paper/{arch_dir}/summary_{method}_seed{seed}.json"
    return json.load(open(p))[key]


def load_train_only(arch_dir, method, seed):
    """Sum per-epoch TRAIN-window quantities over epochs>0 (warmup trimmed, val excluded).
    Returns {'train_gpu_energy_kWh':..., 'train_wall_s':...} or None if CSV missing."""
    p = f"{R}/results/lora_paper/{arch_dir}/perepoch_{method}_seed{seed}.csv"
    if not os.path.exists(p):
        return None
    e = w = 0.0
    n = 0
    with open(p) as f:
        for row in csv.DictReader(f):
            if int(row["epoch"]) == 0:            # trim warmup
                continue
            if row.get("train_gpu_energy_kWh"):
                e += float(row["train_gpu_energy_kWh"]); n += 1
            if row.get("train_dur_s"):
                w += float(row["train_dur_s"])
    if n == 0:
        return None
    return {"train_gpu_energy_kWh": e, "train_wall_s": w}


def boot_pct_ci(fre, stl, n_boot=20000):
    n = len(fre)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.random.randint(0, n, n)
        out[b] = 100.0 * (stl[idx].mean() - fre[idx].mean()) / fre[idx].mean()
    return np.percentile(out, [2.5, 97.5])


def paired_row(arch, color, mlabel, key, basis, fre, stl):
    d = fre - stl
    pct = 100.0 * (stl.mean() - fre.mean()) / fre.mean()
    t, p_t = ss.ttest_rel(fre, stl)
    try:
        _, p_w = ss.wilcoxon(fre, stl)
    except Exception:
        p_w = float("nan")
    lo, hi = boot_pct_ci(fre, stl)
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0
    sig = bool((p_t < 0.05) and ((lo > 0) or (hi < 0)))
    return dict(arch=arch, color=color, metric=mlabel, key=key, basis=basis,
                fre_m=fre.mean(), fre_s=fre.std(ddof=1), stl_m=stl.mean(), stl_s=stl.std(ddof=1),
                pct=pct, ci_lo=lo, ci_hi=hi, p_t=float(p_t), p_w=float(p_w), dz=dz, sig=sig,
                verdict=("—" if not sig else ("ST-LoRA↓" if pct < 0 else "ST-LoRA↑")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=f"{R}/results/lora_paper/stats_crossarch")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    whole_rows, train_rows = [], []
    for arch, adir, color in ARCHS:
        # headline (whole-run)
        for key, mlabel in WHOLE_METRICS:
            fre = np.array([load_whole(adir, "fre", s, key) for s in SEEDS], float)
            stl = np.array([load_whole(adir, "stlora", s, key) for s in SEEDS], float)
            whole_rows.append(paired_row(arch, color, mlabel, key, "whole", fre, stl))
        # supporting (train-only, warmup-trimmed) — only if per-epoch CSVs present
        tf = [load_train_only(adir, "fre", s) for s in SEEDS]
        tst = [load_train_only(adir, "stlora", s) for s in SEEDS]
        if all(x is not None for x in tf + tst):
            for key, mlabel in TRAIN_METRICS:
                fre = np.array([x[key] for x in tf], float)
                stl = np.array([x[key] for x in tst], float)
                train_rows.append(paired_row(arch, color, mlabel, key, "train", fre, stl))
        else:
            print(f"[warn] {arch}: per-epoch CSVs incomplete — train-only rows skipped "
                  f"(re-run with --per_epoch_trace)")
    rows = whole_rows + train_rows

    # ── markdown ─────────────────────────────────────────────────────────────
    def md_section(title, note, rws):
        out = [f"## {title}", "", note, "",
               "| Arch | Metric | FRE | ST-LoRA | %Δ | 95% CI | t p | Wilcoxon p | dz | Sig |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rws:
            out.append(f"| {r['arch']} | {r['metric']} | {r['fre_m']:.4g}±{r['fre_s']:.2g} "
                       f"| {r['stl_m']:.4g}±{r['stl_s']:.2g} | {r['pct']:+.1f}% "
                       f"| [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] | {r['p_t']:.3f} "
                       f"| {r['p_w']:.3f} | {r['dz']:+.2f} | {r['verdict']} |")
        out.append("")
        return out
    L = ["# Exp 1 — ST-LoRA vs FRE training cost, cross-architecture significance", "",
         f"Paired, n={len(SEEDS)} seeds {SEEDS}. %Δ = 100·(ST-LoRA−FRE)/FRE (negative ⇒ ST-LoRA "
         "lower). Lower is better for every metric. Sig = paired-t p<0.05 AND 95% bootstrap CI "
         "excludes 0; ST-LoRA↓ = significantly greener, ST-LoRA↑ = significantly worse. "
         "(Wilcoxon two-sided p floors at 0.0625 at n=5.)", ""]
    L += md_section("Headline — whole training run (incl. validation + warmup)",
                    "CodeCarbon totals over the entire `main()`.", whole_rows)
    if train_rows:
        L += md_section("Supporting — training loop only, warmup epoch trimmed",
                        "Summed per-epoch train windows (validation excluded, epoch 0 dropped); "
                        "GPU energy from the pynvml sampler.", train_rows)
    open(os.path.join(a.out_dir, "co2_crossarch_significance.md"), "w").write("\n".join(L) + "\n")

    # ── LaTeX booktabs (single table, section rows) ──────────────────────────
    T = [r"% Requires \usepackage{booktabs}",
         r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{5pt}",
         r"\caption{Experiment~1: paired significance of ST-LoRA vs FRE training cost across "
         r"architectures ($n{=}5$ seeds). $\%\Delta=100(\text{ST-LoRA}-\text{FRE})/\text{FRE}$; "
         r"negative favours ST-LoRA. Sig: paired-$t$~$p<0.05$ and 95\% bootstrap CI excludes 0 "
         r"($\downarrow$ greener, $\uparrow$ worse). Lower is better throughout.}",
         r"\label{tab:exp1_co2_sig}",
         r"\begin{tabular}{@{}ll rr r c ccc c@{}}", r"\toprule",
         r"Arch & Metric & FRE & ST-LoRA & $\%\Delta$ & 95\% CI & $p_t$ & $p_{\text{Wilc}}$ "
         r"& $d_z$ & Sig \\"]

    def tex_rows(title, rws):
        block = [r"\midrule", r"\multicolumn{10}{@{}l}{\emph{" + title + r"}}\\", r"\midrule"]
        cur = None
        for r in rws:
            arch_cell = r["arch"] if r["arch"] != cur else ""
            if r["arch"] != cur and cur is not None:
                block.append(r"\addlinespace[1pt]")
            cur = r["arch"]
            sig = {"—": "--", "ST-LoRA↓": r"$\downarrow$", "ST-LoRA↑": r"$\uparrow$"}[r["verdict"]]
            block.append(f"{arch_cell} & {r['metric']} & {r['fre_m']:.4g} & {r['stl_m']:.4g} & "
                         f"${r['pct']:+.1f}\\%$ & $[{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}]$ & "
                         f"{r['p_t']:.3f} & {r['p_w']:.3f} & ${r['dz']:+.2f}$ & {sig} \\\\")
        return block
    T += tex_rows("Headline --- whole training run (incl.\\ validation + warmup)", whole_rows)
    if train_rows:
        T += tex_rows("Supporting --- training loop only, warmup trimmed", train_rows)
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(a.out_dir, "co2_crossarch_significance.tex"), "w").write("\n".join(T) + "\n")

    # ── forest plot ──────────────────────────────────────────────────────────
    metric_order = [m[1] for m in WHOLE_METRICS] + [m[1] for m in TRAIN_METRICS]
    order = []
    for mlabel in metric_order:
        for arch, adir, color in ARCHS:
            r = next((x for x in rows if x["metric"] == mlabel and x["arch"] == arch), None)
            if r is not None:
                order.append(r)
    # ── horizontal (landscape) forest: categories along x, %Δ on y ────────────
    n_arch = len(ARCHS)
    xpos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(0.52 * len(order) + 3.0, 7.6))  # landscape
    seen = set()
    for x, r in zip(xpos, order):
        lab = r["arch"] if r["arch"] not in seen else None
        seen.add(r["arch"])
        ax.errorbar(x, r["pct"], yerr=[[r["pct"] - r["ci_lo"]], [r["ci_hi"] - r["pct"]]],
                    fmt="o", color=r["color"], ecolor=r["color"], elinewidth=2, capsize=4,
                    markersize=9 if r["sig"] else 8,
                    markerfacecolor=r["color"] if r["sig"] else "white",
                    markeredgecolor=r["color"], markeredgewidth=1.6, label=lab)
        ax.text(x, r["ci_hi"] + 0.5, f"{r['pct']:+.1f}%", rotation=90,
                ha="center", va="bottom", fontsize=6.8, color="#333")
    ylo = min(r["ci_lo"] for r in order); yhi = max(r["ci_hi"] for r in order)
    ax.set_ylim(ylo - 2.0, yhi + 8.0)           # headroom for the rotated %Δ labels
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.axhline(0, ls="--", color="grey", lw=1.2)
    # one centered tick per metric group (arch is encoded by colour + legend)
    group_centers = [i * n_arch + (n_arch - 1) / 2 for i in range(len(metric_order))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(metric_order, rotation=30, ha="right", fontsize=8)
    for i in range(1, len(metric_order)):       # faint dividers between groups
        ax.axvline(i * n_arch - 0.5, color="#e5e5e5", lw=0.8)
    # divider between whole-run (left) and train-only (right) blocks
    if train_rows:
        xdiv = len(order) - len(TRAIN_METRICS) * n_arch - 0.5
        ax.axvline(xdiv, color="#888", lw=1.6, ls="-")
        ax.text(xdiv + 0.15, ax.get_ylim()[1], " train-only →",
                fontsize=7.5, color="#555", va="top", ha="left")
    ax.set_ylabel("%Δ  ST-LoRA vs FRE  (negative ⇒ ST-LoRA lower / greener)")
    ax.set_title("Exp 1 · ST-LoRA vs FRE training cost\n"
                 "paired %Δ, 95% bootstrap CI · filled = significant (p<0.05 & CI≠0)\n"
                 "whole-run (left) · train-only warmup-trimmed (right)", fontsize=9.5)
    ax.legend(title="Architecture", loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(a.out_dir, "co2_crossarch_forest.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    json.dump({"seeds": SEEDS, "rows": [{k: v for k, v in r.items() if k != "color"}
                                        for r in rows]},
              open(os.path.join(a.out_dir, "co2_crossarch_significance.json"), "w"), indent=2)
    print("\n".join(L))
    print("[fig] wrote", out)


if __name__ == "__main__":
    main()
