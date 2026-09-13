# -*- coding: utf-8 -*-
"""
Paired FRE-vs-ST-LoRA IMAGE-LEVEL OoD significance (Mask2Former, 5 seeds).
Headline image score = mean per-pixel predictive entropy per image (NegEnt scorer).
Sources have per-seed `values` arrays (seed order [42,123,456,789,1337]):
  FRE = results/lora_paper/full_ft/ood/fullft_ood_imagelevel.json
  STL = results/lora_paper/ood/final_model_ood_imagelevel.json
Both cover NEAR-OoD tomato (val, 1200 subsample seed0) AND FAR-OoD GrowliFlower.
Metrics: AUROC, AUPR (higher better), FPR95 (lower better).

Paired by seed; d = ST-LoRA - FRE.
STATS (standing rule): paired t + Wilcoxon + 95% bootstrap CI (20k) + Cohen's dz.
NO Holm / NO BH. Significant iff (paired-t p<0.05 AND bootstrap CI excludes 0).
Wilcoxon's n=5 floor is p=0.0625 -> reported, never gates.
"""
import json
import numpy as np
from scipy import stats as ss

np.random.seed(0)
R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
     "mibrahi2_hpc-my_research-1775524204")
LP = f"{R}/results/lora_paper"
FRE_P = f"{LP}/full_ft/ood/fullft_ood_imagelevel.json"
STL_P = f"{LP}/ood/final_model_ood_imagelevel.json"
SEEDS = [42, 123, 456, 789, 1337]
SCORER = "NegEnt"
METRICS = [("AUROC", "higher"), ("AUPR", "higher"), ("FPR95", "lower")]
SOURCES = [("tomato", "NEAR-OoD tomato"), ("growliflower", "FAR-OoD GrowliFlower")]


def vals(d, src, metric):
    return list(d[src][SCORER][metric]["values"])


def boot_ci(x, n_boot=20000, alpha=0.05):
    x = np.asarray(x, float); n = len(x)
    m = np.array([x[np.random.randint(0, n, n)].mean() for _ in range(n_boot)])
    return np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def paired(fre, stl, better):
    fre = np.asarray(fre, float); stl = np.asarray(stl, float); d = stl - fre
    t, p = ss.ttest_rel(stl, fre)
    try:
        w, pw = ss.wilcoxon(stl, fre)
    except Exception:
        w, pw = float("nan"), float("nan")
    lo, hi = boot_ci(d); sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else float("nan")
    excl0 = (lo > 0) or (hi < 0)
    if better == "higher":
        fav = "ST-LoRA" if d.mean() > 0 else "FRE"
    else:
        fav = "ST-LoRA" if d.mean() < 0 else "FRE"
    return dict(fre=fre, stl=stl, d=d, mF=fre.mean(), mS=stl.mean(), dm=d.mean(),
                sdF=fre.std(ddof=1), sdS=stl.std(ddof=1), t=t, p=p, pw=pw,
                lo=lo, hi=hi, dz=dz, excl0=excl0, fav=fav, sig=(p < 0.05 and excl0))


BAR = "=" * 100


def main():
    FRE = json.load(open(FRE_P)); STL = json.load(open(STL_P))
    print(BAR)
    print("PAIRED FRE vs ST-LoRA — IMAGE-LEVEL OoD (M2F, headline = mean per-pixel entropy)")
    print(f"n={len(SEEDS)} seeds {SEEDS}; paired by seed; d = ST-LoRA - FRE")
    print("paired-t + Wilcoxon + 20k bootstrap CI + Cohen's dz.  NO Holm / NO BH.")
    print("Significant iff (p<0.05 AND CI excludes 0).  ↑ higher better, ↓ lower better.")
    for src, slab in SOURCES:
        print("\n" + BAR)
        print(f"{slab}  (image-level)")
        print(BAR)
        for met, better in METRICS:
            r = paired(vals(FRE, src, met), vals(STL, src, met), better)
            arrow = "↑" if better == "higher" else "↓"
            print(f"\n### {met} {arrow}  ({better} better; d = STL-FRE)")
            for i, s in enumerate(SEEDS):
                print(f"    seed {s:>4}: FRE={r['fre'][i]:.4f}  ST-LoRA={r['stl'][i]:.4f}  d={r['d'][i]:+.4f}")
            print(f"    FRE     mean±SD = {r['mF']:.4f} ± {r['sdF']:.4f}")
            print(f"    ST-LoRA mean±SD = {r['mS']:.4f} ± {r['sdS']:.4f}")
            print(f"    mean d = {r['dm']:+.4f}")
            print(f"    paired t-test   : t(4) = {r['t']:+.3f},  p = {r['p']:.4f}")
            print(f"    Wilcoxon        : p = {r['pw']:.4f}  (n=5 floor 0.0625)")
            print(f"    95% bootstrap CI of mean d = [{r['lo']:+.4f}, {r['hi']:+.4f}] -> "
                  f"{'excludes 0' if r['excl0'] else 'INCLUDES 0'}")
            print(f"    Cohen's dz = {r['dz']:+.3f}")
            print(f"    >>> {'SIGNIFICANT -> '+r['fav']+' better' if r['sig'] else 'n.s. -> comparable'}")
    print("\n" + BAR)
    print("No multiplicity correction (standing rule). Wilcoxon floor at n=5 = 0.0625.")
    print(BAR)


if __name__ == "__main__":
    main()
