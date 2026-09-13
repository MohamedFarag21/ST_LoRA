# -*- coding: utf-8 -*-
"""
Paired FRE-vs-ST-LoRA significance under COVARIATE SHIFT, on the AUTHORITATIVE
float64 grid: results/lora_paper/calibration_shift/native_float64/{fullft,lora}_
seed{S}_shift_native.json (native 1280x720 test93, StreamBinMetrics float64, M=4).
Grid: 7 corruptions {blur,noise,brightness,contrast,saturation,rotation,translation}
x 5 severities, plus a clean reference. Metrics: mIoU (higher better), ECE/ACE/
Brier/NLL (lower better) -- the same 5 used for the UQ-method & calibrator tables.

Paired by seed {42,123,456,789,1337}; d = FRE - ST-LoRA. Two granularities:
  (A) OVERALL  : per seed, mean metric over ALL 35 corrupted cells -> 5 tests.
  (B) BY SEV   : per seed, mean over the 7 corruptions at each severity -> 25 tests.
Each: paired t-test + 95% bootstrap CI + Cohen's dz; BH/Holm adjusted within family.
Clean (severity 0) reported as an un-tested reference.
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


def cell(grid, corr, sev, metric):
    return float(grid[corr][sev][metric])


def mean_metric(grid, metric, sev=None):
    """mean over the 7 corruptions; over all severities 1-5 if sev is None."""
    svs = [sev] if sev else SEVS
    vals = [cell(grid, c, sv, metric) for c in CORR for sv in svs]
    return float(np.mean(vals))


def clean_val(grid, metric):
    return float(grid["clean"]["0"][metric])


def boot_ci(d, n_boot=20000, alpha=0.05):
    d = np.asarray(d, float); n = len(d)
    m = np.array([d[np.random.randint(0, n, n)].mean() for _ in range(n_boot)])
    return np.percentile(m, [100*alpha/2, 100*(1-alpha/2)])


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 1.0
    for r in range(m-1, -1, -1):
        i = o[r]; run = min(run, p[i]*m/(r+1)); adj[i] = min(run, 1.0)
    return adj


def one_test(fre_vals, stl_vals, better):
    fre = np.array(fre_vals); stl = np.array(stl_vals); d = fre - stl
    t, p = ss.ttest_rel(fre, stl)
    lo, hi = boot_ci(d)
    dz = d.mean()/d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
    excl0 = (lo > 0) or (hi < 0)
    if better == "higher":
        fav = "FRE" if d.mean() > 0 else "ST-LoRA"
    else:
        fav = "ST-LoRA" if d.mean() > 0 else "FRE"
    return dict(meanF=fre.mean(), meanS=stl.mean(), d=d.mean(), t=t, p=p,
                lo=lo, hi=hi, dz=dz, excl0=excl0, fav=fav)


print("=" * 96)
print("PAIRED FRE vs ST-LoRA UNDER COVARIATE SHIFT (native float64 grid) -- 7 corruptions x 5 sev")
print(f"n={len(SEEDS)} seeds {SEEDS}; d = FRE - ST-LoRA; metrics: mIoU, ECE, ACE, Brier, NLL")
print("=" * 96)

# clean reference
print("\n--- CLEAN reference (severity 0, NOT tested) ---")
for metric, _ in METRICS:
    cf = np.mean([clean_val(FRE[s], metric) for s in SEEDS])
    cs = np.mean([clean_val(STL[s], metric) for s in SEEDS])
    print(f"    {metric:6s}: FRE={cf:.4f}  ST-LoRA={cs:.4f}")

# (A) overall
print("\n########## (A) OVERALL shift-averaged (mean over 35 corrupted cells), family m=5 ##########")
ovr = []
for metric, better in METRICS:
    fv = [mean_metric(FRE[s], metric) for s in SEEDS]
    sv = [mean_metric(STL[s], metric) for s in SEEDS]
    ovr.append((metric, better, one_test(fv, sv, better)))
pv = [r[2]["p"] for r in ovr]; QB = bh(pv); QH = holm(pv)
print(f"\n{'metric':6s} {'FRE':>8s} {'STLoRA':>8s} {'d(F-S)':>9s} {'p':>8s} {'BHq':>7s} {'Holm':>7s} {'dz':>7s} {'CIexcl0':>8s}  verdict")
for (metric, better, r), b, h in zip(ovr, QB, QH):
    surv = "SURV(Holm)" if (r["excl0"] and h < 0.05) else ("BH" if (r["excl0"] and b < 0.05) else ("raw" if (r["excl0"] and r["p"] < 0.05) else "ns"))
    tag = f"{r['fav']} ({surv})" if surv != "ns" else "comparable"
    print(f"{metric:6s} {r['meanF']:8.4f} {r['meanS']:8.4f} {r['d']:+9.4f} {r['p']:8.4f} {b:7.4f} {h:7.4f} {r['dz']:+7.2f} {str(r['excl0']):>8s}  {tag}")

# (B) by severity
print("\n\n########## (B) BY SEVERITY (mean over 7 corruptions), family m=25 ##########")
recs = []
for metric, better in METRICS:
    for sev in SEVS:
        fv = [mean_metric(FRE[s], metric, sev) for s in SEEDS]
        sv = [mean_metric(STL[s], metric, sev) for s in SEEDS]
        recs.append([metric, sev, better, one_test(fv, sv, better)])
pv = [x[3]["p"] for x in recs]; QB = bh(pv); QH = holm(pv)
for x, b, h in zip(recs, QB, QH):
    x[3]["bh"] = b; x[3]["holm"] = h
print(f"\n{'metric':6s} {'sev':4s} {'FRE':>8s} {'STLoRA':>8s} {'d(F-S)':>9s} {'p':>8s} {'BHq':>7s} {'dz':>7s}  verdict")
for metric, sev, better, r in recs:
    surv = "SURV" if (r["excl0"] and r["holm"] < 0.05) else ("BH" if (r["excl0"] and r["bh"] < 0.05) else ("raw" if (r["excl0"] and r["p"] < 0.05) else "ns"))
    tag = f"{r['fav']}({surv})" if surv != "ns" else "comparable"
    print(f"{metric:6s} s{sev:3s} {r['meanF']:8.4f} {r['meanS']:8.4f} {r['d']:+9.4f} {r['p']:8.4f} {r['bh']:7.4f} {r['dz']:+7.2f}  {tag}")

print("\n" + "=" * 96)
print("BH/Holm adjusted WITHIN each family (A: m=5, B: m=25). Wilcoxon floor at n=5 is 0.0625.")
print("=" * 96)
