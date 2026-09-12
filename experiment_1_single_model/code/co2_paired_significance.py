# -*- coding: utf-8 -*-
"""
Reviewer request: paired significance tests + bootstrap CIs for FRE-vs-ST-LoRA
efficiency claims. The CO2/energy runs are a PAIRED design: the SAME 3 seeds
{42,123,456} were run under both methods, so we pair by seed and test the
per-seed difference d_i = FRE_i - ST-LoRA_i.

For each metric we report:
  * per-seed paired values + differences
  * mean +/- SD for each method, and mean paired difference (+ %)
  * paired t-test (scipy.stats.ttest_rel)            -> t, two-sided p, df
  * Wilcoxon signed-rank (exact)                      -> W, p  (n=3 caveat)
  * 95% bootstrap CI of the mean paired difference (percentile, 20k resamples)
  * Cohen's dz = mean(d)/SD(d)  (paired effect size)
A claim of advantage "survives" only if the test rejects at alpha=0.05 AND the
bootstrap CI excludes 0. With n=3 seeds power is very low -> we expect to SOFTEN.
"""
import os
import json
import numpy as np

np.random.seed(0)
R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
     "mibrahi2_hpc-my_research-1775524204")
CO2 = f"{R}/results/lora_paper/co2"
SEEDS = [42, 123, 456, 789, 1337]
# metric key -> (label, unit, direction) ; direction "lower_is_better"
METRICS = [
    ("emissions_kgCO2e", "CO2 emissions", "kg CO2e"),
    ("energy_consumed_kWh", "Total energy", "kWh"),
    ("gpu_energy_kWh", "GPU energy", "kWh"),
    ("wall_clock_s", "Wall-clock", "s"),
]

try:
    from scipy import stats as ss
    HAVE_SCIPY = True
except Exception as e:
    HAVE_SCIPY = False
    print(f"[warn] scipy unavailable ({e}); t/Wilcoxon p-values skipped")


def load(method, seed, key):
    with open(f"{CO2}/summary_{method}_seed{seed}.json") as f:
        return json.load(f)[key]


def boot_ci(d, n_boot=20000, alpha=0.05):
    d = np.asarray(d, float)
    n = len(d)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.random.randint(0, n, n)
        means[b] = d[idx].mean()
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi


def main():
    print("=" * 78)
    print("PAIRED SIGNIFICANCE TESTS  —  FRE vs ST-LoRA  (n = %d seeds: %s)"
          % (len(SEEDS), SEEDS))
    print("Pairing by seed; difference d = FRE - ST-LoRA  (positive => ST-LoRA lower)")
    print("=" * 78)

    for key, label, unit in METRICS:
        fre = np.array([load("fre", s, key) for s in SEEDS], float)
        stl = np.array([load("stlora", s, key) for s in SEEDS], float)
        d = fre - stl
        print(f"\n### {label}  ({unit})  [lower is better]")
        for i, s in enumerate(SEEDS):
            print(f"    seed {s:>4}: FRE={fre[i]:.6g}  ST-LoRA={stl[i]:.6g}  d={d[i]:+.6g}")
        print(f"    FRE      mean+/-SD = {fre.mean():.6g} +/- {fre.std(ddof=1):.3g}")
        print(f"    ST-LoRA  mean+/-SD = {stl.mean():.6g} +/- {stl.std(ddof=1):.3g}")
        pct = 100.0 * d.mean() / fre.mean()
        print(f"    mean paired diff = {d.mean():+.6g}  ({pct:+.1f}% vs FRE)   "
              f"SD(d)={d.std(ddof=1):.3g}")

        # paired t-test + Wilcoxon
        if HAVE_SCIPY:
            t, p_t = ss.ttest_rel(fre, stl)
            print(f"    paired t-test : t({len(SEEDS)-1}) = {t:.3f},  p = {p_t:.4f}  (two-sided)")
            try:
                w, p_w = ss.wilcoxon(fre, stl)  # exact for small n
                print(f"    Wilcoxon      : W = {w:.1f},  p = {p_w:.4f}  "
                      f"(n=3 -> min possible two-sided p = 0.25)")
            except Exception as e:
                print(f"    Wilcoxon      : n/a ({e})")
        # bootstrap CI
        lo, hi = boot_ci(d)
        excl0 = (lo > 0) or (hi < 0)
        print(f"    95% bootstrap CI of mean diff = [{lo:+.6g}, {hi:+.6g}]  "
              f"-> {'excludes 0' if excl0 else 'INCLUDES 0'}")
        # effect size
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
        print(f"    Cohen's dz = {dz:.3f}")

        # verdict
        if HAVE_SCIPY:
            survives = (p_t < 0.05) and excl0
            print(f"    >>> claim of ST-LoRA advantage SURVIVES at alpha=0.05? "
                  f"{'YES' if survives else 'NO -> SOFTEN'}")

    print("\n" + "=" * 78)
    print("NOTE: n=3 seeds gives very low power; Wilcoxon cannot reach p<0.05 at n=3.")
    print("Report effect sizes + CIs and soften any claim whose CI includes 0.")
    print("=" * 78)


if __name__ == "__main__":
    main()
