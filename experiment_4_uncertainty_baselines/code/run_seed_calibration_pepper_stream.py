# -*- coding: utf-8 -*-
"""
PEPPER (in-domain, 8-class): fit calibrators @320x180 -> evaluate CALIBRATION (ECE/ACE/
mIoU/acc) @ native 1280x720 WITHOUT recalibrating. Mirrors the tomato §9.4 protocol.

Fit  : cal(30) frames of `valid` (never seen in training), cached at 320x180.
Eval : val(33) [primary] and test(93) [secondary], streamed at native.

ECE MUST use the float64 streaming path: val33 = 30.4M px and test93 = 85.7M px, both
over torchmetrics' float32 ~16.7M-px/bin corruption threshold. See
`reference_torchmetrics_float32_ece` / METHODOLOGY §9.4.

Writes seed_<S>/stream_metrics_native.json. Run via SLURM (ssl env).
"""
import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")
sys.path.insert(0, f"{ROOT}/code/bup_20_trials/elora")

import pepper_common as pc                                            # noqa: E402
import pepper_stream_common as psc                                    # noqa: E402
from run_seed_calibration_stream import StreamBinMetrics              # noqa: E402


def eval_split(model, ds, proc, dev, fitted, methods, desc):
    """Stream one split at native; returns {method: {ECE,ACE,mIoU,acc,...}}."""
    accs = {m: StreamBinMetrics(num_classes=pc.NUM) for m in methods}

    def cb(idx, outs, gt):
        lab = torch.from_numpy(gt)[None].to(dev)                      # (1,H,W)
        for m in methods:
            probs = F.softmax(outs[m], dim=1)                         # (1,NUM,H,W)
            accs[m].update(probs, lab)

    psc.stream_native(model, ds, list(range(len(ds))), proc, dev, fitted, methods,
                      cb, desc, binary_anomaly=False)
    return {m: accs[m].compute() for m in methods}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", default=f"{ROOT}/results/posthoc_calibration_pepper/fit320_native")
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--fit_size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_images", type=int, default=None)   # smoke hook
    ap.add_argument("--no_test", action="store_true")
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
    print(f"[split] cal={len(cal_ds)} val={sp['n_val']} test={len(sp['test_ids'])}", flush=True)

    fitted, _ = psc.fit_calibrators_320(model, cal_ds, dev, args.fit_size,
                                        args.epochs, args.batch_size)
    proc = psc.native_processor()
    methods = psc.METHODS

    results = {}
    splits = [("val33", pc.COCOSegDataset(image_ids=sp["val_ids"]))]
    if not args.no_test:
        splits.append(("test93", pc.COCOSegDataset(image_ids=sp["test_ids"])))
    for name, ds in splits:
        if args.max_images:
            ds.images = ds.images[:args.max_images]
        print(f"[stream] {name}: {len(ds)} frames @ native", flush=True)
        t = time.time()
        results[name] = eval_split(model, ds, proc, dev, fitted, methods, f"ece-{name}")
        print(f"[timing] {name} {time.time()-t:.1f}s", flush=True)
        for m in methods:
            r = results[name][m]
            print(f"[{name} {m:13s}] ECE={r['ECE']:.6f} ACE={r['ACE']:.6f} "
                  f"mIoU={r['mIoU']:.4f} acc={r['acc']:.4f} "
                  f"present={r['n_classes_present']}/{pc.NUM}", flush=True)

    out = {"seed": args.seed, "shot": args.shot, "fit_size": args.fit_size,
           "eval": "native_1280x720", "num_class": pc.NUM,
           "n_val": len(splits[0][1]), "elapsed_s": time.time() - t0,
           "metrics": results}
    outp = os.path.join(seed_dir, "stream_metrics_native.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
