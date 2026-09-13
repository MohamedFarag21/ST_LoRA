# -*- coding: utf-8 -*-
"""
PEPPER (in-domain, 8-class): fit calibrators @320x180 -> evaluate PIXEL-OoD @ native
WITHOUT recalibrating. Mirrors the tomato §9.5 protocol.

Fit  : cal(30) frames of pepper `valid`, cached at 320x180.
OoD  : two anomaly sources, FULL sets, streamed at each source's native resolution:
         * tomato       (4536 frames @ 1280x720) — anomaly = fruit (sem>0)
         * growliflower (1970 frames @  368x448) — anomaly = cauliflower (raw>0)
       score = per-pixel entropy of the calibrated 8-class probs (higher = more OoD).

Two notes that differ from the binary tomato study:
  * 8-class entropy is bounded by ln(8) ~ 2.0794, NOT ln2 -> OODHist(smax=ln8).
  * GrowliFlower native is only 368x448, so fit@320 is ~87% of native there (a ~13%
    shift, not a 4x one). The fit->native RESOLUTION question is therefore only
    meaningful for the tomato source; growliflower is reported for comparability.

Pass 1: float64 entropy histograms -> AUROC/AUPR/FPR95 + best-F1 threshold.
Pass 2: second native forward -> component sIoU/PPV/MeanF1 at that threshold.
Writes seed_<S>/ood_stream_metrics_native.json. Run via SLURM (ssl env).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")
sys.path.insert(0, f"{ROOT}/code/bup_20_trials/elora")

import pepper_common as pc                                            # noqa: E402
import pepper_stream_common as psc                                    # noqa: E402
from run_seed_ood_stream import OODHist, component_metrics_fast       # noqa: E402
from run_seed_ood_pepper import GrowliRaw                             # noqa: E402  (palette-safe)
from mask2former_lora_train_tomato import TomatoDataset               # noqa: E402
from ood_eval_comprehensive import compute_fpr95                      # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score    # noqa: E402


def run_source(model, ds, proc, dev, fitted, methods, name, validate_n, max_images):
    idx = list(range(len(ds)))
    if max_images:
        idx = idx[:max_images]
    print(f"[{name}] {len(idx)} frames @ native (2 passes); validate_n={validate_n}",
          flush=True)

    hists = {m: OODHist(smax=psc.LN_NUM) for m in methods}
    # Control buffer holds ONLY the uncalibrated scores — that is all the sklearn
    # cross-check consumes. Keeping all 7 methods would retain ~1GB at native for
    # nothing.
    vs, vl = [], []

    def cb1(i, ents, gt):
        g = torch.from_numpy(gt).to(ents[methods[0]].device)
        for m in methods:
            hists[m].update(ents[m], g)
        if len(vl) < validate_n:
            vl.append(gt.ravel().astype(np.uint8))
            vs.append(ents["Uncalibrated"].detach().cpu().numpy().ravel().astype(np.float32))

    t = time.time()
    psc.stream_native(model, ds, idx, proc, dev, fitted, methods, cb1,
                      f"{name}-pass1", binary_anomaly=True)
    print(f"[timing] {name} pass1 {time.time()-t:.1f}s", flush=True)

    res = {m: hists[m].metrics() for m in methods}
    for m in methods:
        r = res[m]
        print(f"[{name} pass1 {m:13s}] AUROC={r['AUROC']:.4f} AUPR={r['AUPR']:.4f} "
              f"FPR95={r['FPR95']:.4f} thr={r['threshold']:.4f}", flush=True)

    # internal control: histogram vs sklearn on the first `validate_n` frames
    validation = None
    if vl:
        lab = np.concatenate(vl)
        sc = np.concatenate(vs)
        h = OODHist(smax=psc.LN_NUM)
        for i in range(0, sc.size, 2_000_000):
            h.update(torch.from_numpy(sc[i:i+2_000_000].astype(np.float64)),
                     torch.from_numpy(lab[i:i+2_000_000].astype(np.int64)))
        hm = h.metrics()
        sk = {"AUROC": float(roc_auc_score(lab, sc)),
              "AUPR": float(average_precision_score(lab, sc)),
              "FPR95": float(compute_fpr95(lab, sc))}
        validation = {"n_frames": len(vl), "n_pixels": int(lab.size),
                      "sklearn": sk,
                      "histogram": {k: hm[k] for k in ("AUROC", "AUPR", "FPR95")}}
        print(f"[{name} validate n={len(vl)}] sklearn  AUROC={sk['AUROC']:.5f} "
              f"AUPR={sk['AUPR']:.5f} FPR95={sk['FPR95']:.5f}", flush=True)
        print(f"[{name} validate n={len(vl)}] histogram AUROC={hm['AUROC']:.5f} "
              f"AUPR={hm['AUPR']:.5f} FPR95={hm['FPR95']:.5f}", flush=True)
    del vs, vl

    # pass 2: component metrics at each method's best-F1 threshold
    comp = {m: {"siou": [], "ppv": [], "f1": []} for m in methods}

    def cb2(i, ents, gt):
        for m in methods:
            thr = res[m]["threshold"]
            pred = (ents[m] >= thr).detach().cpu().numpy().astype(np.uint8)
            s, p, f = component_metrics_fast(gt, pred)
            if not np.isnan(s):
                comp[m]["siou"].append(s); comp[m]["ppv"].append(p); comp[m]["f1"].append(f)

    t = time.time()
    psc.stream_native(model, ds, idx, proc, dev, fitted, methods, cb2,
                      f"{name}-pass2", binary_anomaly=True)
    print(f"[timing] {name} pass2 {time.time()-t:.1f}s", flush=True)

    for m in methods:
        c = comp[m]
        res[m]["sIoU"] = float(np.mean(c["siou"])) if c["siou"] else float("nan")
        res[m]["PPV"] = float(np.mean(c["ppv"])) if c["ppv"] else float("nan")
        res[m]["MeanF1"] = float(np.mean(c["f1"])) if c["f1"] else float("nan")
        res[m]["n_frames_scored"] = len(c["siou"])
        print(f"[{name} pass2 {m:13s}] sIoU={res[m]['sIoU']:.4f} PPV={res[m]['PPV']:.4f} "
              f"MeanF1={res[m]['MeanF1']:.4f}", flush=True)

    return {"n_frames": len(idx), "validation": validation, "metrics": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native")
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--fit_size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_images", type=int, default=None)     # smoke hook
    ap.add_argument("--validate_n", type=int, default=40)
    ap.add_argument("--sources", nargs="+", default=["tomato", "growliflower"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    dev = args.device
    seed_dir = os.path.join(args.out_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)

    ckpt = os.path.join(pc.MODELS_DIR, f"seed_{args.seed}", f"{args.shot}.pt")
    print(f"[load] {ckpt}", flush=True)
    model = pc.load_pepper_model(ckpt, dev)

    sp = pc.load_split()
    cal_ds = pc.COCOSegDataset(image_ids=sp["cal_ids"])
    fitted, _ = psc.fit_calibrators_320(model, cal_ds, dev, args.fit_size,
                                        args.epochs, args.batch_size)
    proc = psc.native_processor()
    methods = psc.METHODS

    all_src = {"tomato": lambda: TomatoDataset(pc.TOMATO_ROOT, split="val"),
               "growliflower": GrowliRaw}
    out_src = {}
    for name in args.sources:
        ds = all_src[name]()
        out_src[name] = run_source(model, ds, proc, dev, fitted, methods, name,
                                   args.validate_n, args.max_images)

    out = {"seed": args.seed, "shot": args.shot, "fit_size": args.fit_size,
           "eval": "native", "num_class": pc.NUM, "smax": psc.LN_NUM,
           "nbins": OODHist().nbins, "elapsed_s": time.time() - t0,
           "sources": out_src}
    outp = os.path.join(seed_dir, "ood_stream_metrics_native.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
