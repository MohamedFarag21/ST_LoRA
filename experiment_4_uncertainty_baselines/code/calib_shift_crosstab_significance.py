# -*- coding: utf-8 -*-
"""
Calibration + performance-under-shift significance CROSS-TAB, FRE vs ST-LoRA
(Mask2Former, 5 seeds), rendered in the SAME style as ood_full_crosstab.md.

Source: authoritative native float64 grid
  results/lora_paper/calibration_shift/native_float64/{fullft,lora}_seed{S}_shift_native.json
  (native 1280x720 test93, StreamBinMetrics float64, M=4 snapshot ensemble).

Settings:
  IN-DOMAIN (clean, severity 0)
  UNDER SHIFT — OVERALL (mean over all 35 corrupted cells)
  UNDER SHIFT — BY SEVERITY s1..s5 (mean over the 7 corruptions at each severity)
Metrics: mIoU (higher better); ECE, ACE, Brier, NLL (lower better).

STATS (standing rule): paired t + Wilcoxon + 95% bootstrap CI (20k) + Cohen's dz.
NO Holm / NO BH.  d = ST-LoRA - FRE  (same convention as the OoD cross-tab).
Significant iff (paired-t p<0.05 AND CI excludes 0). Wilcoxon n=5 floor 0.0625 (never gates).
Writes results/lora_paper/stats_methods/calib_shift_crosstab.md
"""
import json
import numpy as np
from scipy import stats as ss

np.random.seed(0)
R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
     "mibrahi2_hpc-my_research-1775524204")
N = f"{R}/results/lora_paper/calibration_shift/native_float64"
OUT = f"{R}/results/lora_paper/stats_methods/calib_shift_crosstab.md"
SEEDS = [42, 123, 456, 789, 1337]
CORR = ["blur", "noise", "brightness", "contrast", "saturation", "rotation", "translation"]
SEVS = ["1", "2", "3", "4", "5"]
HI, LO = "higher", "lower"
METRICS = [("mIoU", HI), ("ECE", LO), ("ACE", LO), ("Brier", LO), ("NLL", LO)]


def load(tag):
    return {s: json.load(open(f"{N}/{tag}_seed{s}_shift_native.json"))["grid"] for s in SEEDS}


FRE = load("fullft")
STL = load("lora")


def clean_val(grid, m):
    return float(grid["clean"]["0"][m])


def mean_metric(grid, m, sev=None):
    svs = [sev] if sev else SEVS
    return float(np.mean([float(grid[c][sv][m]) for c in CORR for sv in svs]))


def boot_ci(x, n_boot=20000, alpha=0.05):
    x = np.asarray(x, float); n = len(x)
    b = np.array([x[np.random.randint(0, n, n)].mean() for _ in range(n_boot)])
    return np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def stat(fre, stl, better):
    fre = np.asarray(fre, float); stl = np.asarray(stl, float); d = stl - fre  # STL - FRE
    t, p = ss.ttest_rel(stl, fre)
    try:
        _, pw = ss.wilcoxon(stl, fre)
    except Exception:
        pw = float("nan")
    lo, hi = boot_ci(d); sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else float("nan")
    excl0 = (lo > 0) or (hi < 0)
    if better == HI:
        fav = "ST-LoRA" if d.mean() > 0 else "FRE"
    else:
        fav = "ST-LoRA" if d.mean() < 0 else "FRE"
    return dict(mF=fre.mean(), sF=fre.std(ddof=1), mS=stl.mean(), sS=stl.std(ddof=1),
                d=d.mean(), t=t, p=p, pw=pw, lo=lo, hi=hi, dz=dz, excl0=excl0,
                fav=fav, sig=(p < 0.05 and excl0))


# (setting label, getter(grid, metric)->scalar)
SETTINGS = [("In-domain (clean)", lambda g, m: clean_val(g, m)),
            ("Shift — overall (35 cells)", lambda g, m: mean_metric(g, m))]
for sv in SEVS:
    SETTINGS.append((f"Shift — severity {sv}", (lambda sv_: (lambda g, m: mean_metric(g, m, sv_)))(sv)))


def main():
    L = []
    L.append("# FRE vs ST-LoRA — calibration & performance-under-shift cross-tab (M2F, 5 seeds)")
    L.append("")
    L.append("Same style as `ood_full_crosstab.md`. Paired by seed {42,123,456,789,1337}; "
             "**d = ST-LoRA − FRE**. Stats = paired-t + Wilcoxon + 20k bootstrap CI + Cohen's *d_z*. "
             "**No Holm / no BH.** Significant iff (p<0.05 **and** CI excludes 0). Wilcoxon n=5 "
             "floor = 0.0625 (never gates). ↑ higher better, ↓ lower better. Native float64 grid, M=4. "
             "Source: `calibration_shift/native_float64/`.")
    L.append("")
    L.append("`Winner` is blank (tie) unless significant. `d` and CI are on the ST-LoRA−FRE scale, "
             "so for ↓-metrics a **negative d = ST-LoRA better**.")

    hdr = ("| Setting | Metric | FRE (mean±sd) | ST-LoRA (mean±sd) | d (S−F) | p | "
           "Wilcoxon p | d_z | 95% CI | excl0 | **Winner** |")
    sep = "|" + "---|" * 11
    L.append(""); L.append(hdr); L.append(sep)

    print("=" * 118)
    print("FRE vs ST-LoRA — CALIBRATION & SHIFT CROSS-TAB;  d = ST-LoRA - FRE;  NO Holm/BH")
    print("=" * 118)
    print(f"{'setting':28s} {'metric':6s} {'FRE':>8s} {'STL':>8s} {'d(S-F)':>9s} "
          f"{'p':>8s} {'Wp':>8s} {'dz':>7s} {'CI_lo':>8s} {'CI_hi':>8s}  winner")
    for slab, getter in SETTINGS:
        for metric, better in METRICS:
            fre = [getter(FRE[s], metric) for s in SEEDS]
            stl = [getter(STL[s], metric) for s in SEEDS]
            r = stat(fre, stl, better)
            arrow = "↑" if better == HI else "↓"
            win = f"**{r['fav']}**" if r["sig"] else ""
            L.append(f"| {slab} | {metric} {arrow} | {r['mF']:.4f}±{r['sF']:.4f} | "
                     f"{r['mS']:.4f}±{r['sS']:.4f} | {r['d']:+.4f} | {r['p']:.4f} | {r['pw']:.4f} | "
                     f"{r['dz']:+.2f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {str(r['excl0'])} | {win} |")
            print(f"{slab:28s} {metric:6s} {r['mF']:8.4f} {r['mS']:8.4f} {r['d']:+9.4f} "
                  f"{r['p']:8.4f} {r['pw']:8.4f} {r['dz']:+7.2f} {r['lo']:+8.4f} {r['hi']:+8.4f}  "
                  f"{r['fav'] if r['sig'] else 'tie'}")
        L.append("| | | | | | | | | | | |")  # visual group separator between settings

    L.append("")
    L.append("**Reading it:** mIoU is *performance*; ECE/ACE/Brier/NLL are *calibration* "
             "(ECE/ACE = binning error, Brier/NLL = proper scores). In-domain and shift-overall "
             "share the same 5 seeds and metric family as the OoD cross-tab.")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
