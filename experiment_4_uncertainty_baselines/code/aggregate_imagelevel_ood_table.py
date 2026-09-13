# -*- coding: utf-8 -*-
"""
Aggregate the IMAGE-LEVEL OoD results into one table across all methods:
  4 uncertainty methods (Full-FT / ST-LoRA / MC-Dropout / DDU)   -> bup_20/elora jsons
  Uncalibrated + 6 calibrators (TS/Logistic/Dirichlet/LTS/Meta/Selective) -> posthoc jsons

Headline score for every row = MEAN per-pixel predictive entropy (image-level).
Reports, mean +/- std over 5 seeds:
  * GrowliFlower (far-OoD): AUROC, FPR95
  * Tomato       (near-OoD): AUROC, FPR95

Writes results/lora_paper/imagelevel_ood/imagelevel_ood_table.{md,json}.
Missing inputs are skipped with a note (so it can run partially). ssl_cc env.
"""
import os
import json
import numpy as np

ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
LP   = f"{ROOT}/results/lora_paper"
OUT  = f"{LP}/imagelevel_ood"
SEEDS = [42, 123, 456, 789, 1337]

# (row label, json path, headline scorer key) for the 4 UQ methods
UQ = [
    ("Full-FT (FRE)",  f"{LP}/full_ft/ood/fullft_ood_imagelevel.json",     "NegEnt"),
    ("ST-LoRA",        f"{LP}/ood/final_model_ood_imagelevel.json",        "NegEnt"),
    ("MC-Dropout",     f"{LP}/mcdropout/ood/mcdropout_ood_imagelevel.json","NegEnt"),
    ("DDU (density)",  f"{LP}/ddu/ood_summary_imagelevel.json",            "gmm_spatial"),
]
CALIB_ORDER = ["Uncalibrated", "TS", "Logistic", "Dirichlet", "LTS", "Meta", "Selective"]


def ms(vals):
    a = np.asarray([v for v in vals if v is not None], float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0)


def load_uq(path, scorer):
    """Return {src: {AUROC:(m,s), FPR95:(m,s)}} from an aggregate_imagelevel json."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    out = {}
    for src in ("growliflower", "tomato"):
        if src not in d or scorer not in d[src]:
            return None
        out[src] = {mk: (d[src][scorer][mk]["mean"], d[src][scorer][mk]["std"])
                    for mk in ("AUROC", "FPR95")}
    return out


def load_calibrators():
    """Aggregate posthoc per-seed image-level jsons over seeds -> per-method dict."""
    per = {m: {"growliflower": {"AUROC": [], "FPR95": []},
               "tomato":       {"AUROC": [], "FPR95": []}} for m in CALIB_ORDER}
    n_found = 0
    for s in SEEDS:
        p = f"{ROOT}/results/posthoc_calibration_pepper/imagelevel_ood/seed_{s}/ood_imagelevel_metrics.json"
        if not os.path.exists(p):
            continue
        n_found += 1
        d = json.load(open(p))["methods"]
        for m in CALIB_ORDER:
            if m not in d:
                continue
            for src in ("growliflower", "tomato"):
                per[m][src]["AUROC"].append(d[m][src]["AUROC"])
                per[m][src]["FPR95"].append(d[m][src]["FPR95"])
    if n_found == 0:
        return None, 0
    out = {}
    for m in CALIB_ORDER:
        out[m] = {src: {mk: ms(per[m][src][mk]) for mk in ("AUROC", "FPR95")}
                  for src in ("growliflower", "tomato")}
    return out, n_found


def fmt(pair):
    m, s = pair
    if np.isnan(m):
        return "  —  "
    return f"{m:.3f}±{s:.3f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []       # (label, family, dict{src:{metric:(m,s)}})

    for label, path, scorer in UQ:
        r = load_uq(path, scorer)
        if r is None:
            print(f"[skip] UQ {label}: missing/incomplete {path}")
            continue
        rows.append((label, "UQ", r))

    calib, n_cal = load_calibrators()
    if calib is not None:
        for m in CALIB_ORDER:
            rows.append((m, "Calibrator", calib[m]))
        print(f"[calib] aggregated {n_cal}/5 seeds")
    else:
        print("[skip] calibrators: no per-seed jsons yet")

    # ── markdown ─────────────────────────────────────────────────────────────
    lines = []
    lines.append("# Image-level OoD — all methods (headline score = mean per-pixel entropy)")
    lines.append("")
    lines.append(f"ID = sweet-pepper test (93 frames). Far-OoD = GrowliFlower (cauliflower). "
                 f"Near-OoD = tomato val (random {1200} subsample, seed 0). "
                 f"mean±std over {len(SEEDS)} seeds. FPR95 = FPR at 95% TPR (lower better).")
    lines.append("")
    lines.append("| Method | Family | Growli AUROC↑ | Growli FPR95↓ | Tomato AUROC↑ | Tomato FPR95↓ |")
    lines.append("|---|---|---|---|---|---|")
    for label, fam, r in rows:
        lines.append(f"| {label} | {fam} | {fmt(r['growliflower']['AUROC'])} | "
                     f"{fmt(r['growliflower']['FPR95'])} | {fmt(r['tomato']['AUROC'])} | "
                     f"{fmt(r['tomato']['FPR95'])} |")
    md = "\n".join(lines) + "\n"
    open(f"{OUT}/imagelevel_ood_table.md", "w").write(md)

    js = {"seeds": SEEDS, "headline": "mean per-pixel entropy", "rows": []}
    for label, fam, r in rows:
        js["rows"].append({"method": label, "family": fam,
                           "growliflower": {k: list(r["growliflower"][k]) for k in ("AUROC", "FPR95")},
                           "tomato":       {k: list(r["tomato"][k])       for k in ("AUROC", "FPR95")}})
    json.dump(js, open(f"{OUT}/imagelevel_ood_table.json", "w"), indent=2)

    print("\n" + md)
    print(f"[wrote] {OUT}/imagelevel_ood_table.md (+.json)  rows={len(rows)}")


if __name__ == "__main__":
    main()
