# -*- coding: utf-8 -*-
"""
Paired FRE-vs-ST-LoRA significance for CALIBRATION, Mask2Former, on the
authoritative float64 grid:
  results/lora_paper/calibration_shift/native_float64/{fullft,lora}_seed{S}_shift_native.json
(native 1280x720 test93, StreamBinMetrics float64, M=4 snapshot ensemble).

TWO calibration settings, sharing the same 5 seeds and the same metric family:
  (1) IN-DOMAIN (clean, severity 0)  -- the "calibration" axis, never tested before.
  (2) UNDER COVARIATE SHIFT           -- 7 corruptions x 5 severities:
        (A) OVERALL  : per seed, mean metric over ALL 35 corrupted cells -> 5 tests.
        (B) BY SEV   : per seed, mean over the 7 corruptions at each severity.

Metrics: mIoU (higher better), ECE/ACE/Brier/NLL (lower better).
Paired by seed {42,123,456,789,1337}; d = FRE - ST-LoRA.

STATS PER STANDING RULE -- NO Holm, NO BH. Exactly:
  paired t-test  +  Wilcoxon signed-rank  +  95% percentile bootstrap CI (20k)  +  Cohen's dz.
Verdict = "significant" iff (paired-t p<0.05 AND bootstrap CI excludes 0). Wilcoxon reported
for completeness (its floor at n=5 is p=0.0625, so it can never cross 0.05 -- stated, not used
as a gate).
"""
import json
import numpy as np
from scipy import stats as ss

np.random.seed(0)
R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
     "mibrahi2_hpc-my_research-1775524204")
N = f"{R}/results/lora_paper/calibration_shift/native_float64"
SEEDS = [42, 123, 456, 789, 1337]
CORR = ["blur", "noise", "brightness", "contrast", "saturation", "rotation", "translation"]
SEVS = ["1", "2", "3", "4", "5"]
METRICS = [("mIoU", "higher"), ("ECE", "lower"), ("ACE", "lower"),
           ("Brier", "lower"), ("NLL", "lower")]


def load(tag):
    out = {}
    for s in SEEDS:
        with open(f"{N}/{tag}_seed{s}_shift_native.json") as f:
            out[s] = json.load(f)["grid"]
    return out


FRE = load("fullft")
STL = load("lora")


def clean_val(grid, metric):
    return float(grid["clean"]["0"][metric])


def cell(grid, corr, sev, metric):
    return float(grid[corr][sev][metric])


def mean_metric(grid, metric, sev=None):
    """mean over the 7 corruptions; over all severities 1-5 if sev is None."""
    svs = [sev] if sev else SEVS
    return float(np.mean([cell(grid, c, sv, metric) for c in CORR for sv in svs]))


def boot_ci(d, n_boot=20000, alpha=0.05):
    d = np.asarray(d, float); n = len(d)
    m = np.array([d[np.random.randint(0, n, n)].mean() for _ in range(n_boot)])
    return np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def paired(fre_vals, stl_vals, better):
    """FRE vs ST-LoRA paired test; d = FRE - ST-LoRA. Returns a dict of stats."""
    fre = np.asarray(fre_vals, float); stl = np.asarray(stl_vals, float); d = fre - stl
    t, p = ss.ttest_rel(fre, stl)
    # Wilcoxon (guard the all-zero / n<... degeneracy)
    try:
        w, pw = ss.wilcoxon(fre, stl)
    except Exception:
        w, pw = float("nan"), float("nan")
    lo, hi = boot_ci(d)
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else float("nan")
    excl0 = (lo > 0) or (hi < 0)
    if better == "higher":
        fav = "FRE" if d.mean() > 0 else "ST-LoRA"
    else:
        fav = "ST-LoRA" if d.mean() > 0 else "FRE"
    sig = (p < 0.05) and excl0
    return dict(fre=fre, stl=stl, d=d, meanF=fre.mean(), meanS=stl.mean(),
                dmean=d.mean(), dsd=sd, t=t, p=p, w=w, pw=pw, lo=lo, hi=hi,
                dz=dz, excl0=excl0, fav=fav, sig=sig)


def verdict(r):
    if r["sig"]:
        return f"SIGNIFICANT -> {r['fav']} better"
    return "n.s. -> comparable"


BAR = "=" * 98


def block_full(title, getter):
    """Print a full per-seed block for each metric using getter(grid, metric)->scalar."""
    print("\n" + BAR)
    print(title)
    print(BAR)
    for metric, better in METRICS:
        fv = [getter(FRE[s], metric) for s in SEEDS]
        sv = [getter(STL[s], metric) for s in SEEDS]
        r = paired(fv, sv, better)
        adv = "FRE" if better == "higher" else "ST-LoRA"
        print(f"\n### {metric}  ({better} is better; positive d = FRE-ST-LoRA favours {adv})")
        for i, s in enumerate(SEEDS):
            print(f"    seed {s:>4}: FRE={r['fre'][i]:.4f}  ST-LoRA={r['stl'][i]:.4f}  d={r['d'][i]:+.4f}")
        print(f"    FRE     mean±SD = {r['meanF']:.4f} ± {r['fre'].std(ddof=1):.4f}")
        print(f"    ST-LoRA mean±SD = {r['meanS']:.4f} ± {r['stl'].std(ddof=1):.4f}")
        print(f"    mean d (F-S) = {r['dmean']:+.4f}   SD(d) = {r['dsd']:.4f}")
        print(f"    paired t-test   : t(4) = {r['t']:+.3f},  p = {r['p']:.4f}")
        print(f"    Wilcoxon        : W = {r['w']:.1f},  p = {r['pw']:.4f}  (n=5 floor 0.0625)")
        print(f"    95% bootstrap CI of mean d = [{r['lo']:+.4f}, {r['hi']:+.4f}] -> "
              f"{'excludes 0' if r['excl0'] else 'INCLUDES 0'}")
        print(f"    Cohen's dz = {r['dz']:+.3f}")
        print(f"    >>> {verdict(r)}")


def main():
    print(BAR)
    print("PAIRED FRE vs ST-LoRA CALIBRATION SIGNIFICANCE (M2F, native float64, M=4)")
    print(f"n = {len(SEEDS)} seeds {SEEDS};  paired by seed;  d = FRE - ST-LoRA")
    print("STATS: paired t + Wilcoxon + 95% bootstrap CI (20k) + Cohen's dz.  NO Holm / NO BH.")
    print("Verdict SIGNIFICANT iff (paired-t p<0.05 AND bootstrap CI excludes 0).")

    # (1) IN-DOMAIN clean
    block_full("(1) IN-DOMAIN  (clean, severity 0)  -- test93", clean_val)

    # (2A) shift OVERALL
    block_full("(2A) UNDER SHIFT -- OVERALL (mean over all 35 corrupted cells)",
               lambda g, m: mean_metric(g, m))

    # (2B) shift BY SEVERITY -- compact table
    print("\n" + BAR)
    print("(2B) UNDER SHIFT -- BY SEVERITY (mean over 7 corruptions at each severity)")
    print(BAR)
    print(f"\n{'metric':6s} {'sev':4s} {'FRE':>8s} {'STLoRA':>8s} {'d(F-S)':>9s} "
          f"{'t':>7s} {'p':>8s} {'Wp':>8s} {'dz':>7s} {'CI_lo':>8s} {'CI_hi':>8s} {'excl0':>6s}  verdict")
    for metric, better in METRICS:
        for sev in SEVS:
            fv = [mean_metric(FRE[s], metric, sev) for s in SEEDS]
            sv = [mean_metric(STL[s], metric, sev) for s in SEEDS]
            r = paired(fv, sv, better)
            tag = f"{r['fav']}" if r["sig"] else "comparable"
            print(f"{metric:6s} s{sev:3s} {r['meanF']:8.4f} {r['meanS']:8.4f} {r['dmean']:+9.4f} "
                  f"{r['t']:+7.2f} {r['p']:8.4f} {r['pw']:8.4f} {r['dz']:+7.2f} "
                  f"{r['lo']:+8.4f} {r['hi']:+8.4f} {str(r['excl0']):>6s}  {tag}")

    print("\n" + BAR)
    print("No multiplicity correction applied (per standing rule). Wilcoxon shown for")
    print("completeness only; its n=5 floor is p=0.0625 so it never gates. Lead with")
    print("paired-t + bootstrap CI + dz.")
    print(BAR)


if __name__ == "__main__":
    main()
