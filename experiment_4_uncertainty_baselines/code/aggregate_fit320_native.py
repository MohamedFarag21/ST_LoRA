# -*- coding: utf-8 -*-
"""
Aggregate the 5 seeds of the PEPPER fit@320 -> eval@native study.

Reads  <out_dir>/seed_<S>/stream_metrics_native.json      (ECE/ACE/mIoU/acc)
       <out_dir>/seed_<S>/ood_stream_metrics_native.json  (pixel-OoD, 2 sources)
Writes <out_dir>/pepper_native_summary.md  and  pepper_native_summary.csv

Refuses to publish if a seed is missing or if a run's internal histogram-vs-sklearn
control drifted — a partial/degraded run must fail loudly, not become a plausible table.
"""
import os
import json
import argparse
import numpy as np

METHODS = ["Uncalibrated", "TS", "Logistic", "Dirichlet", "LTS", "Meta", "Selective"]
CAL_METRICS = ["ECE", "ACE", "mIoU", "acc"]
OOD_METRICS = ["AUROC", "AUPR", "FPR95", "sIoU", "PPV", "MeanF1"]
HIGHER_BETTER = {"AUROC": True, "AUPR": True, "FPR95": False, "sIoU": True,
                 "PPV": True, "MeanF1": True, "ECE": False, "ACE": False,
                 "mIoU": True, "acc": True}


def ms(a):
    a = np.asarray(a, dtype=np.float64)
    return f"{np.nanmean(a):.4f} ± {np.nanstd(a):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/lustre/scratch/data/mibrahi2_hpc-my_research/"
                                         "mibrahi2_hpc-my_research-1783386603/"
                                         "mibrahi2_hpc-my_research-1775524204/results/"
                                         "posthoc_calibration_pepper/fit320_native")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    args = ap.parse_args()

    cal, ood, problems = {}, {}, []
    for s in args.seeds:
        pc_ = os.path.join(args.out_dir, f"seed_{s}", "stream_metrics_native.json")
        po_ = os.path.join(args.out_dir, f"seed_{s}", "ood_stream_metrics_native.json")
        if not os.path.exists(pc_):
            problems.append(f"seed {s}: MISSING {pc_}")
        else:
            cal[s] = json.load(open(pc_))["metrics"]
        if not os.path.exists(po_):
            problems.append(f"seed {s}: MISSING {po_}")
            continue
        j = json.load(open(po_))
        ood[s] = j["sources"]
        for src, blk in j["sources"].items():
            v = blk.get("validation")
            if v:
                d = max(abs(v["sklearn"][k] - v["histogram"][k])
                        for k in ("AUROC", "AUPR", "FPR95"))
                print(f"[seed {s}] {src:13s} n={blk['n_frames']:5d} "
                      f"hist-vs-sklearn max|delta|={d:.2e}", flush=True)
                if d > 1e-3:
                    problems.append(f"seed {s}/{src}: histogram vs sklearn {d:.2e}")

    if problems:
        print("\n!! REFUSING TO PUBLISH — unresolved problems:", flush=True)
        for x in problems:
            print(f"   - {x}", flush=True)
        raise SystemExit(1)

    L = ["# PEPPER (in-domain, 8-class) — calibrators fit @320x180, evaluated @ native",
         "",
         f"Seeds: {args.seeds}. Pepper native = 720x1280 (WxH), so 320 is a true 4x "
         f"downscale (320x180), matching the tomato protocol.",
         "",
         "ECE/ACE via float64 streaming (torchmetrics' float32 bin accumulation corrupts "
         "ECE above ~16.7M px/bin; val33 = 30.4M px is already over that line). mIoU is "
         "macro-averaged over classes PRESENT in the eval set (rare subtypes absent -> "
         "skipped, not scored 0).",
         ""]

    # ── calibration ──
    for split in ("val33", "test93"):
        if not any(split in cal[s] for s in cal):
            continue
        L += [f"## Calibration @ native — {split}", "",
              "| Method | " + " | ".join(CAL_METRICS) + " |",
              "|---|" + "---|" * len(CAL_METRICS)]
        for m in METHODS:
            cells = [ms([cal[s][split][m][k] for s in args.seeds]) for k in CAL_METRICS]
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        u = np.nanmean([cal[s][split]["Uncalibrated"]["ECE"] for s in args.seeds])
        best, bv = None, None
        for m in METHODS[1:]:
            v = np.nanmean([cal[s][split][m]["ECE"] for s in args.seeds])
            if bv is None or v < bv:
                best, bv = m, v
        L += ["", f"Best ECE: **{best}** {bv:.4f} vs uncalibrated {u:.4f} "
              f"({100*(bv-u)/u:+.0f}%).", ""]

    # ── OoD ──
    for src in ("tomato", "growliflower"):
        if not any(src in ood[s] for s in ood):
            continue
        n = ood[args.seeds[0]][src]["n_frames"]
        L += [f"## Pixel-OoD @ native — anomaly source: {src} (n={n})", "",
              "| Method | " + " | ".join(OOD_METRICS) + " |",
              "|---|" + "---|" * len(OOD_METRICS)]
        for m in METHODS:
            cells = [ms([ood[s][src]["metrics"][m][k] for s in args.seeds])
                     for k in OOD_METRICS]
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
        # per-seed AUROC/FPR95 (exposes seed lotteries, as in the tomato study)
        for k in ("AUROC", "FPR95"):
            L += [f"Per-seed {k}:", "",
                  "| Method | " + " | ".join(f"seed {s}" for s in args.seeds) + " |",
                  "|---|" + "---|" * len(args.seeds)]
            for m in METHODS:
                v = [f"{ood[s][src]['metrics'][m][k]:.4f}" for s in args.seeds]
                L.append(f"| {m} | " + " | ".join(v) + " |")
            L.append("")

    md = os.path.join(args.out_dir, "pepper_native_summary.md")
    open(md, "w").write("\n".join(L) + "\n")

    csv = os.path.join(args.out_dir, "pepper_native_summary.csv")
    with open(csv, "w") as f:
        f.write("block,method,metric,mean,std," + ",".join(f"seed_{s}" for s in args.seeds) + "\n")
        for split in ("val33", "test93"):
            for m in METHODS:
                for k in CAL_METRICS:
                    a = [cal[s][split][m][k] for s in args.seeds]
                    f.write(f"cal_{split},{m},{k},{np.nanmean(a):.6f},{np.nanstd(a):.6f},"
                            + ",".join(f"{x:.6f}" for x in a) + "\n")
        for src in ("tomato", "growliflower"):
            for m in METHODS:
                for k in OOD_METRICS:
                    a = [ood[s][src]["metrics"][m][k] for s in args.seeds]
                    f.write(f"ood_{src},{m},{k},{np.nanmean(a):.6f},{np.nanstd(a):.6f},"
                            + ",".join(f"{x:.6f}" for x in a) + "\n")

    print("\n".join(L), flush=True)
    print(f"\n[done] wrote {md} and {csv}", flush=True)


if __name__ == "__main__":
    main()
