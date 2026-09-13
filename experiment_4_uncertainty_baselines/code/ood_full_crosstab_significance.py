# -*- coding: utf-8 -*-
"""
COMPREHENSIVE OoD significance CROSS-TAB, FRE vs ST-LoRA (Mask2Former, 5 seeds).
One unified paired table over the full grid:
    LEVEL   {image, pixel}
  x DATASET {tomato (near-OoD), GrowliFlower (far-OoD)}
  x SCORER  {entropy, MI}
  x METRIC  image: AUROC/AUPR/FPR95 ; pixel: AUROC/AUPR/FPR95/sIoU/PPV/MeanF1

Per-seed sources (seed order [42,123,456,789,1337]):
  IMAGE, entropy : full_ft/ood/fullft_ood_imagelevel.json  vs  ood/final_model_ood_imagelevel.json
                   -> d[src]['NegEnt'][metric]['values']  (tomato + growliflower; AUROC/AUPR/FPR95)
  IMAGE, MI      : ood_v2_fixed/{fullft,lora}_seed{S}_ood_results.json
                   -> d['image_level']['MI_AUROC']  (GrowliFlower ONLY, AUROC ONLY; tomato image-MI was never
                      computed, and image AUPR/FPR95 exist for the entropy headline only -> those cells = n/a)
  PIXEL, tomato  : tomato_lora/pepper_transfer/{full_ft_tomato,pepper_to_tomato}_pixelood_seed{S}_results.json
                   -> d['pixel_level'][scorer][metric]
  PIXEL, growli  : ood_v2_fixed/{fullft,lora}_seed{S}_ood_results.json
                   -> d['pixel_level'][scorer][metric]

STATS (standing rule): paired t + Wilcoxon + 95% bootstrap CI (20k) + Cohen's dz.
NO Holm / NO BH. d = ST-LoRA - FRE. Significant iff (paired-t p<0.05 AND CI excludes 0).
Writes results/lora_paper/stats_methods/ood_full_crosstab.md
"""
import json
import numpy as np
from scipy import stats as ss

np.random.seed(0)
R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
     "mibrahi2_hpc-my_research-1775524204")
LP = f"{R}/results/lora_paper"
PT = f"{R}/results/tomato_lora/pepper_transfer"
GF = f"{LP}/ood_v2_fixed"
OUT = f"{LP}/stats_methods/ood_full_crosstab.md"
SEEDS = [42, 123, 456, 789, 1337]
HI, LO = "higher", "lower"
IMG_METRICS = [("AUROC", HI), ("AUPR", HI), ("FPR95", LO)]
PIX_METRICS = [("AUROC", HI), ("AUPR", HI), ("FPR95", LO), ("sIoU", HI), ("PPV", HI), ("MeanF1", HI)]


def jload(p):
    with open(p) as f:
        return json.load(f)


# ---- cached source reads ----
IMG_FRE = jload(f"{LP}/full_ft/ood/fullft_ood_imagelevel.json")
IMG_STL = jload(f"{LP}/ood/final_model_ood_imagelevel.json")
GF_FRE = {s: jload(f"{GF}/fullft_seed{s}_ood_results.json") for s in SEEDS}
GF_STL = {s: jload(f"{GF}/lora_seed{s}_ood_results.json") for s in SEEDS}
PT_FRE = {s: jload(f"{PT}/full_ft_tomato_pixelood_seed{s}_results.json") for s in SEEDS}
PT_STL = {s: jload(f"{PT}/pepper_to_tomato_pixelood_seed{s}_results.json") for s in SEEDS}
SRCKEY = {"tomato": "tomato", "growli": "growliflower"}


def get_vecs(level, dataset, scorer, metric):
    """Return (fre_vec, stl_vec) or (None,None) if that combination doesn't exist."""
    if level == "image":
        if scorer == "entropy":
            k = SRCKEY[dataset]
            return (IMG_FRE[k]["NegEnt"][metric]["values"],
                    IMG_STL[k]["NegEnt"][metric]["values"])
        else:  # image + MI: only growli AUROC exists
            if dataset != "growli" or metric != "AUROC":
                return (None, None)
            return ([GF_FRE[s]["image_level"]["MI_AUROC"] for s in SEEDS],
                    [GF_STL[s]["image_level"]["MI_AUROC"] for s in SEEDS])
    else:  # pixel
        if dataset == "tomato":
            F, S = PT_FRE, PT_STL
        else:
            F, S = GF_FRE, GF_STL
        return ([F[s]["pixel_level"][scorer][metric] for s in SEEDS],
                [S[s]["pixel_level"][scorer][metric] for s in SEEDS])


def boot_ci(x, n_boot=20000, alpha=0.05):
    x = np.asarray(x, float); n = len(x)
    m = np.array([x[np.random.randint(0, n, n)].mean() for _ in range(n_boot)])
    return np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def stats(fre, stl, better):
    fre = np.asarray(fre, float); stl = np.asarray(stl, float); d = stl - fre
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
    sig = (p < 0.05) and excl0
    return dict(mF=fre.mean(), sF=fre.std(ddof=1), mS=stl.mean(), sS=stl.std(ddof=1),
                d=d.mean(), t=t, p=p, pw=pw, lo=lo, hi=hi, dz=dz, excl0=excl0,
                fav=fav, sig=sig)


DATASETS = [("tomato", "Tomato (near-OoD)"), ("growli", "GrowliFlower (far-OoD)")]
SCORERS = ["entropy", "MI"]


def main():
    L = []
    L.append("# FRE vs ST-LoRA — full OoD significance cross-tab (M2F, 5 seeds)")
    L.append("")
    L.append("Paired by seed {42,123,456,789,1337}; **d = ST-LoRA − FRE**. "
             "Stats = paired-t + Wilcoxon + 20k bootstrap CI + Cohen's *d_z*. "
             "**No Holm / no BH.** Significant iff (p<0.05 **and** CI excludes 0). "
             "Wilcoxon n=5 floor = 0.0625 (never gates). ↑ higher better, ↓ lower better.")
    L.append("")
    L.append("`Winner` is blank (tie) unless significant. Arrow marks the metric direction. "
             "`d` and CI are on the ST-LoRA−FRE scale.")

    hdr = ("| Level | Dataset | Scorer | Metric | FRE (mean±sd) | ST-LoRA (mean±sd) | "
           "d (S−F) | p | Wilcoxon p | d_z | 95% CI | excl0 | **Winner** |")
    sep = "|" + "---|" * 13

    # print to stdout too
    def emit(s=""):
        print(s); L.append(s)

    print("=" * 120)
    print("FRE vs ST-LoRA — FULL OoD SIGNIFICANCE CROSS-TAB (image+pixel x tomato+growli x entropy+MI x all metrics)")
    print("d = ST-LoRA - FRE; paired-t + Wilcoxon + 20k bootstrap CI + dz; NO Holm/BH; sig iff p<0.05 AND CI excl0")
    print("=" * 120)

    for level, metrics in [("image", IMG_METRICS), ("pixel", PIX_METRICS)]:
        L.append("")
        L.append(f"## {level.capitalize()}-level")
        L.append("")
        L.append(hdr); L.append(sep)
        print(f"\n########## {level.upper()}-LEVEL ##########")
        print(f"{'dataset':22s} {'scorer':7s} {'metric':7s} {'FRE':>8s} {'STL':>8s} "
              f"{'d(S-F)':>9s} {'p':>8s} {'Wp':>8s} {'dz':>7s} {'CI_lo':>8s} {'CI_hi':>8s}  winner")
        for dkey, dlab in DATASETS:
            for scorer in SCORERS:
                for metric, better in metrics:
                    fre, stl = get_vecs(level, dkey, scorer, metric)
                    arrow = "↑" if better == HI else "↓"
                    if fre is None:
                        L.append(f"| {level} | {dlab} | {scorer} | {metric} {arrow} | "
                                 f"— | — | — | — | — | — | — | — | *n/a* |")
                        print(f"{dlab:22s} {scorer:7s} {metric:7s} {'—':>8s} {'—':>8s}"
                              f"{'':>44s}  n/a (not computed)")
                        continue
                    r = stats(fre, stl, better)
                    win = f"**{r['fav']}**" if r["sig"] else ""
                    L.append(f"| {level} | {dlab} | {scorer} | {metric} {arrow} | "
                             f"{r['mF']:.4f}±{r['sF']:.4f} | {r['mS']:.4f}±{r['sS']:.4f} | "
                             f"{r['d']:+.4f} | {r['p']:.4f} | {r['pw']:.4f} | {r['dz']:+.2f} | "
                             f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {str(r['excl0'])} | {win} |")
                    print(f"{dlab:22s} {scorer:7s} {metric:7s} {r['mF']:8.4f} {r['mS']:8.4f} "
                          f"{r['d']:+9.4f} {r['p']:8.4f} {r['pw']:8.4f} {r['dz']:+7.2f} "
                          f"{r['lo']:+8.4f} {r['hi']:+8.4f}  {r['fav'] if r['sig'] else 'tie'}")

    L.append("")
    L.append("**n/a note:** image-level MI was only computed for GrowliFlower AUROC "
             "(`ood_v2_fixed/image_level/MI_AUROC`); tomato image-level MI and image-level "
             "AUPR/FPR95 for MI were never produced. Pixel-level sIoU/PPV/MeanF1 are "
             "segmentation-style scores, defined at pixel level only (no image-level analogue).")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
