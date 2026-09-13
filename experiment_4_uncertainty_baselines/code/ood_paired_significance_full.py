# -*- coding: utf-8 -*-
"""
COMPREHENSIVE paired FRE-vs-ST-LoRA OoD significance (Mask2Former, 5 seeds), covering
EVERY stored OoD metric -- not just entropy-AUROC/FPR95. Supersedes the two-metric
`ood_paired_significance.py` on the pixel side.

Scorers  : entropy, MI (mutual information / BALD)
Metrics  : AUROC, AUPR, sIoU, PPV, MeanF1  (higher better);  FPR95 (lower better)
Sources  :
  * Near-OoD TOMATO      -- results/tomato_lora/pepper_transfer/
        FRE = full_ft_tomato_pixelood_seed{S}_results.json
        STL = pepper_to_tomato_pixelood_seed{S}_results.json
    -> PIXEL-LEVEL ONLY. These evals never produced an image_level block, so there is
       NO image-level tomato significance to compute here (would need an eval re-run).
  * Far-OoD GROWLIFLOWER -- results/lora_paper/ood_v2_fixed/
        FRE = fullft_seed{S}_ood_results.json
        STL = lora_seed{S}_ood_results.json
    -> has BOTH pixel_level.{scorer}.{metric} AND image_level.{scorer}_AUROC.

Paired by seed {42,123,456,789,1337}; d = ST-LoRA - FRE.
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
PT = f"{R}/results/tomato_lora/pepper_transfer"
GF = f"{R}/results/lora_paper/ood_v2_fixed"
SEEDS = [42, 123, 456, 789, 1337]
SCORERS = ["entropy", "MI"]
PIX_METRICS = [("AUROC", "higher"), ("AUPR", "higher"), ("FPR95", "lower"),
               ("sIoU", "higher"), ("PPV", "higher"), ("MeanF1", "higher")]


def jload(p):
    with open(p) as f:
        return json.load(f)


def boot_ci(d, n_boot=20000, alpha=0.05):
    d = np.asarray(d, float); n = len(d)
    m = np.array([d[np.random.randint(0, n, n)].mean() for _ in range(n_boot)])
    return np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def paired(fre, stl, better):
    fre = np.asarray(fre, float); stl = np.asarray(stl, float); d = stl - fre  # STL - FRE
    t, p = ss.ttest_rel(stl, fre)
    try:
        w, pw = ss.wilcoxon(stl, fre)
    except Exception:
        w, pw = float("nan"), float("nan")
    lo, hi = boot_ci(d)
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else float("nan")
    excl0 = (lo > 0) or (hi < 0)
    # who is better: higher-better -> +d favours STL; lower-better -> -d favours STL
    if better == "higher":
        fav = "ST-LoRA" if d.mean() > 0 else "FRE"
    else:
        fav = "ST-LoRA" if d.mean() < 0 else "FRE"
    sig = (p < 0.05) and excl0
    return dict(fre=fre, stl=stl, d=d, mF=fre.mean(), mS=stl.mean(), dm=d.mean(),
                t=t, p=p, pw=pw, lo=lo, hi=hi, dz=dz, excl0=excl0, fav=fav, sig=sig)


BAR = "=" * 104


def table(title, rows):
    """rows: list of (rowlabel, better, fre_vec, stl_vec)."""
    print("\n" + BAR)
    print(title)
    print(BAR)
    print(f"\n{'metric':16s} {'FRE':>8s} {'STLoRA':>8s} {'d(S-F)':>9s} {'t':>7s} "
          f"{'p':>8s} {'Wp':>8s} {'dz':>7s} {'CI_lo':>8s} {'CI_hi':>8s} {'excl0':>6s}  verdict")
    for lab, better, fv, sv in rows:
        r = paired(fv, sv, better)
        tag = f"{r['fav']} better" if r["sig"] else "comparable"
        arrow = "↑" if better == "higher" else "↓"
        print(f"{lab+' '+arrow:16s} {r['mF']:8.4f} {r['mS']:8.4f} {r['dm']:+9.4f} {r['t']:+7.2f} "
              f"{r['p']:8.4f} {r['pw']:8.4f} {r['dz']:+7.2f} {r['lo']:+8.4f} {r['hi']:+8.4f} "
              f"{str(r['excl0']):>6s}  {tag}")


def main():
    print(BAR)
    print("COMPREHENSIVE FRE vs ST-LoRA OoD SIGNIFICANCE (M2F, all stored metrics)")
    print(f"n={len(SEEDS)} seeds {SEEDS}; paired by seed; d = ST-LoRA - FRE")
    print("paired-t + Wilcoxon + 20k bootstrap CI + Cohen's dz.  NO Holm / NO BH.")
    print("Significant iff (p<0.05 AND CI excludes 0).  ↑ = higher better, ↓ = lower better.")

    # ---- NEAR-OoD TOMATO (pixel only) ----
    fre_j = {s: jload(f"{PT}/full_ft_tomato_pixelood_seed{s}_results.json") for s in SEEDS}
    stl_j = {s: jload(f"{PT}/pepper_to_tomato_pixelood_seed{s}_results.json") for s in SEEDS}
    for sc in SCORERS:
        rows = []
        for met, better in PIX_METRICS:
            fv = [fre_j[s]["pixel_level"][sc][met] for s in SEEDS]
            sv = [stl_j[s]["pixel_level"][sc][met] for s in SEEDS]
            rows.append((met, better, fv, sv))
        table(f"NEAR-OoD TOMATO — PIXEL-level, scorer = {sc}", rows)
    print("\n[note] Image-level tomato: NOT AVAILABLE — the tomato OoD evals stored no "
          "image_level block. Would require re-running the eval with per-image aggregation.")

    # ---- FAR-OoD GROWLIFLOWER (pixel + image) ----
    freg = {s: jload(f"{GF}/fullft_seed{s}_ood_results.json") for s in SEEDS}
    stlg = {s: jload(f"{GF}/lora_seed{s}_ood_results.json") for s in SEEDS}
    for sc in SCORERS:
        rows = []
        for met, better in PIX_METRICS:
            fv = [freg[s]["pixel_level"][sc][met] for s in SEEDS]
            sv = [stlg[s]["pixel_level"][sc][met] for s in SEEDS]
            rows.append((met, better, fv, sv))
        table(f"FAR-OoD GROWLIFLOWER — PIXEL-level, scorer = {sc}", rows)
    # image-level growli (both scorers)
    rows = []
    for sc in SCORERS:
        key = f"{sc}_AUROC"
        fv = [freg[s]["image_level"][key] for s in SEEDS]
        sv = [stlg[s]["image_level"][key] for s in SEEDS]
        rows.append((f"{sc}_AUROC", "higher", fv, sv))
    table("FAR-OoD GROWLIFLOWER — IMAGE-level AUROC", rows)

    print("\n" + BAR)
    print("No multiplicity correction (standing rule). Wilcoxon floor at n=5 = 0.0625.")
    print(BAR)


if __name__ == "__main__":
    main()
