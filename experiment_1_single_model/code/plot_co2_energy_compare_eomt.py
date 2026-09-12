# -*- coding: utf-8 -*-
"""
EoMT energy / CO2 comparison figure — FRE full-FT vs ST-LoRA r8. NO-aug 40-epoch runs.
Same 5-panel layout as plot_co2_energy_compare.py (Mask2Former), reading the EoMT CO2
outputs written by eomt_co2_compare_train.py.
  A Total CO2 (kg) ; B Energy GPU/CPU/RAM ; C Wall-clock ; D power×time ; F per-epoch GPU energy
Reads results/lora_paper/co2_eomt/{summary,perepoch,powertrace}_*.{json,csv}.
Writes results/lora_paper/co2_eomt/co2_energy_compare_eomt.{png,pdf}. Non-GPU; matplotlib Agg.
"""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
     "mibrahi2_hpc-my_research-1775524204")
CO2 = f"{R}/results/lora_paper/co2_eomt"
PE = CO2                      # EoMT wrapper writes perepoch/powertrace into the same dir
SEEDS = [42, 123, 456, 789, 1337]
METHODS = [("fre", "FRE", "#d62728"), ("stlora", "ST-LoRA r8", "#1f77b4")]
TRACE_SEED = 42


def load_summary(method, seed):
    with open(f"{CO2}/summary_{method}_seed{seed}.json") as f:
        return json.load(f)


def agg(method, key):
    vals = [load_summary(method, s)[key] for s in SEEDS]
    return np.mean(vals), np.std(vals), vals


def load_perepoch(method):
    ep, e_wh, warm = [], [], []
    with open(f"{PE}/perepoch_{method}_seed{TRACE_SEED}.csv") as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"]))
            e_wh.append(float(row["train_gpu_energy_kWh"]) * 1000.0)   # kWh -> Wh
            warm.append(int(row["is_warmup"]))
    return np.array(ep), np.array(e_wh), np.array(warm)


def pct_drop(fre_mean, st_mean):
    return 100.0 * (fre_mean - st_mean) / fre_mean


def bar_pair(ax, means, stds, colors, ylabel, title, unit_fmt="{:.3f}"):
    x = np.arange(2)
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, width=0.6,
                  edgecolor="black", linewidth=0.6, error_kw=dict(lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in METHODS])
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, ls="--")
    for b, mn, sd in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width()/2, mn + sd, unit_fmt.format(mn),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    d = pct_drop(means[0], means[1])
    ax.text(0.5, 0.94, f"ST-LoRA {d:+.1f}%", transform=ax.transAxes, ha="center",
            va="top", fontsize=9, color="#333",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f4f4f4", ec="#bbb"))
    ax.set_ylim(0, max(means[i]+stds[i] for i in range(2)) * 1.22)


def main():
    colors = [m[2] for m in METHODS]

    co2_m = [agg(m[0], "emissions_kgCO2e") for m in METHODS]
    tot_m = [agg(m[0], "energy_consumed_kWh") for m in METHODS]
    gpu_m = [agg(m[0], "gpu_energy_kWh") for m in METHODS]
    cpu_m = [agg(m[0], "cpu_energy_kWh") for m in METHODS]
    ram_m = [agg(m[0], "ram_energy_kWh") for m in METHODS]
    wall_m = [agg(m[0], "wall_clock_s") for m in METHODS]
    gpuP_m = [agg(m[0], "gpu_power_W") for m in METHODS]

    fig, axd = plt.subplot_mosaic(
        [["A", "A", "B", "B", "C", "C"],
         [".", "D", "D", "F", "F", "."]],
        figsize=(15.5, 9.2))
    axA, axB, axC, axD, axF = axd["A"], axd["B"], axd["C"], axd["D"], axd["F"]
    fig.suptitle("EoMT training energy & CO$_2$ — FRE vs ST-LoRA r8  "
                 "(no-aug, 40 epochs, 1×A100, 5 seeds; German grid 380.95 gCO$_2$/kWh)",
                 fontsize=13.5, fontweight="bold")

    bar_pair(axA, [c[0] for c in co2_m], [c[1] for c in co2_m], colors,
             "CO$_2$ emissions (kg CO$_2$e)", "A · Total CO$_2$ emissions", "{:.4f}")

    x = np.arange(2)
    gpu = np.array([g[0] for g in gpu_m]); cpu = np.array([c[0] for c in cpu_m])
    ram = np.array([r[0] for r in ram_m]); tot_sd = np.array([t[1] for t in tot_m])
    axB.bar(x, gpu, width=0.6, color=colors, edgecolor="black", lw=0.6, label="GPU")
    axB.bar(x, cpu, bottom=gpu, width=0.6, color="#888", edgecolor="black", lw=0.6, label="CPU")
    axB.bar(x, ram, bottom=gpu+cpu, width=0.6, color="#ccc", edgecolor="black", lw=0.6, label="RAM")
    axB.errorbar(x, gpu+cpu+ram, yerr=tot_sd, fmt="none", ecolor="black", capsize=6, lw=1.2)
    for xi, g, c, r in zip(x, gpu, cpu, ram):
        axB.text(xi, g/2, f"GPU\n{g:.3f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        axB.text(xi, g+c+r+tot_sd[0]*0.4+0.004, f"tot {g+c+r:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels([m[1] for m in METHODS])
    axB.set_ylabel("Energy (kWh)"); axB.set_title("B · Energy breakdown (GPU/CPU/RAM)", fontweight="bold")
    axB.grid(axis="y", alpha=0.3, ls="--")
    axB.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axB.text(0.5, 0.94, f"GPU energy ST-LoRA {pct_drop(gpu[0],gpu[1]):+.1f}%",
             transform=axB.transAxes, ha="center", va="top", fontsize=9, color="#333",
             bbox=dict(boxstyle="round,pad=0.25", fc="#f4f4f4", ec="#bbb"))
    axB.set_ylim(0, (gpu+cpu+ram).max()*1.25)

    bar_pair(axC, [w[0]/60 for w in wall_m], [w[1]/60 for w in wall_m], colors,
             "Wall-clock (min)", "C · Wall-clock time", "{:.1f}")

    axD2 = axD.twinx()
    xw = np.arange(2); w = 0.34
    pw = [g[0] for g in gpuP_m]; pw_sd = [g[1] for g in gpuP_m]
    wm = [w0[0]/60 for w0 in wall_m]; wm_sd = [w0[1]/60 for w0 in wall_m]
    axD.bar(xw - w/2, pw, w, yerr=pw_sd, capsize=4, color=colors, edgecolor="black", lw=0.6, alpha=0.95)
    axD2.bar(xw + w/2, wm, w, yerr=wm_sd, capsize=4, color=colors, edgecolor="black", lw=0.6, alpha=0.45, hatch="//")
    axD.set_xticks(xw); axD.set_xticklabels([m[1] for m in METHODS])
    axD.set_ylabel("Avg GPU power (W)  [solid]"); axD2.set_ylabel("Wall-clock (min)  [hatched]")
    axD.set_title("D · energy = power × time  (power noisy; time is the lever)", fontweight="bold")
    axD.set_ylim(0, max(pw)*1.18); axD2.set_ylim(0, max(wm)*1.25)
    for xi, p, sd in zip(xw - w/2, pw, pw_sd):
        axD.text(xi, p + sd, f"{p:.0f}±{sd:.0f}W", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    for xi, t in zip(xw + w/2, wm):
        axD2.text(xi, t, f"{t:.1f}m", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for key, lab, col in METHODS:
        ep, e_wh, warm = load_perepoch(key)
        axF.plot(ep, e_wh, "-o", color=col, lw=1.6, ms=3.5, label=lab)
        wi = np.where(warm == 1)[0]
        if len(wi):
            axF.scatter(ep[wi], e_wh[wi], s=90, facecolors="none", edgecolors=col, lw=1.8, zorder=5)
    axF.set_xlabel("Epoch"); axF.set_ylabel("Train GPU energy (Wh / epoch)")
    axF.set_title(f"F · Per-epoch GPU energy (seed {TRACE_SEED})", fontweight="bold")
    axF.grid(alpha=0.3, ls="--"); axF.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = f"{CO2}/co2_energy_compare_eomt.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("[fig] wrote", out)
    print(f"  CO2  kg : FRE {co2_m[0][0]:.4f}±{co2_m[0][1]:.4f} | ST {co2_m[1][0]:.4f}±{co2_m[1][1]:.4f}"
          f" ({pct_drop(co2_m[0][0],co2_m[1][0]):+.1f}%)")
    print(f"  GPU kWh : FRE {gpu_m[0][0]:.4f} | ST {gpu_m[1][0]:.4f} ({pct_drop(gpu_m[0][0],gpu_m[1][0]):+.1f}%)")
    print(f"  wall min: FRE {wall_m[0][0]/60:.1f} | ST {wall_m[1][0]/60:.1f} ({pct_drop(wall_m[0][0],wall_m[1][0]):+.1f}%)")
    print(f"  GPU W   : FRE {gpuP_m[0][0]:.1f} | ST {gpuP_m[1][0]:.1f}")


if __name__ == "__main__":
    main()
