# -*- coding: utf-8 -*-
"""
Paired significance test for the HEADLINE accuracy claim: EoMT-FRE vs corrected
EoMT-ST-LoRA (flat-LR), M=4 snapshot ensemble, bup20 test93. PAIRED by seed
(both methods use seeds {42,123,456,789,1337}); d = FRE - ST-LoRA.

Metrics: mIoU (higher better -> positive d = FRE advantage), and the calibration
scores ECE/Brier/NLL (lower better -> positive d = ST-LoRA advantage). Reports
per-seed values, mean+/-SD, paired t-test, Wilcoxon, 95% percentile bootstrap CI
of the mean paired difference, and Cohen's dz. Same protocol as the CO2 test.
"""
import os
import json
import numpy as np

np.random.seed(0)
R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
     "mibrahi2_hpc-my_research-1775524204")
FRE_DIR = f"{R}/results/lora_paper/eomt_full_noaug"
STL_DIR = f"{R}/results/lora_paper/eomt_lora_noaug_flatlr"
SEEDS = [42, 123, 456, 789, 1337]
# key, label, "higher"/"lower" is better
METRICS = [
    ("mIoU", "mIoU", "higher"),
    ("ECE", "ECE", "lower"),
    ("Brier", "Brier", "lower"),
    ("NLL", "NLL", "lower"),
]

from scipy import stats as ss


def load(dirp, seed, key):
    with open(f"{dirp}/seed_{seed}/eval_metrics_test_ens4.json") as f:
        return json.load(f)[key]


def boot_ci(d, n_boot=20000, alpha=0.05):
    d = np.asarray(d, float); n = len(d)
    means = np.array([d[np.random.randint(0, n, n)].mean() for _ in range(n_boot)])
    return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])


def main():
    print("=" * 82)
    print("PAIRED SIGNIFICANCE  —  EoMT-FRE vs corrected ST-LoRA (flat-LR), M=4 ens, test93")
    print(f"n = {len(SEEDS)} seeds {SEEDS};  paired by seed;  d = FRE - ST-LoRA")
    print("=" * 82)
    for key, label, better in METRICS:
        fre = np.array([load(FRE_DIR, s, key) for s in SEEDS], float)
        stl = np.array([load(STL_DIR, s, key) for s in SEEDS], float)
        d = fre - stl
        adv = "FRE" if better == "higher" else "ST-LoRA"  # who benefits from positive d
        print(f"\n### {label}  ({better} is better; positive d favors {adv})")
        for i, s in enumerate(SEEDS):
            print(f"    seed {s:>4}: FRE={fre[i]:.5f}  ST-LoRA={stl[i]:.5f}  d={d[i]:+.5f}")
        print(f"    FRE      mean+/-SD = {fre.mean():.5f} +/- {fre.std(ddof=1):.5f}")
        print(f"    ST-LoRA  mean+/-SD = {stl.mean():.5f} +/- {stl.std(ddof=1):.5f}")
        pct = 100.0 * d.mean() / fre.mean()
        print(f"    mean paired diff = {d.mean():+.5f}  ({pct:+.2f}% of FRE)  SD(d)={d.std(ddof=1):.5f}")
        t, p_t = ss.ttest_rel(fre, stl)
        print(f"    paired t-test : t({len(SEEDS)-1}) = {t:.3f},  p = {p_t:.4f}")
        try:
            w, p_w = ss.wilcoxon(fre, stl)
            print(f"    Wilcoxon      : W = {w:.1f},  p = {p_w:.4f}  (n=5 -> min two-sided p = 0.0625)")
        except Exception as e:
            print(f"    Wilcoxon      : n/a ({e})")
        lo, hi = boot_ci(d)
        excl0 = (lo > 0) or (hi < 0)
        print(f"    95% bootstrap CI of mean diff = [{lo:+.5f}, {hi:+.5f}]  "
              f"-> {'excludes 0' if excl0 else 'INCLUDES 0'}")
        dz = d.mean()/d.std(ddof=1) if d.std(ddof=1) > 0 else float('nan')
        print(f"    Cohen's dz = {dz:.3f}")
        survives = (p_t < 0.05) and excl0
        who = adv if d.mean() != 0 else "neither"
        print(f"    >>> difference significant at alpha=0.05? {'YES ('+who+' better)' if survives else 'NO -> report as comparable/soften'}")
    print("\n" + "=" * 82)
    print("n=5 -> Wilcoxon can reach p=0.0625 at best (still not <0.05); lead with paired t + dz + CI.")
    print("=" * 82)


if __name__ == "__main__":
    main()
