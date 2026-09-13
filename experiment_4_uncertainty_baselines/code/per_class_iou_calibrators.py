# -*- coding: utf-8 -*-
"""
Per-class IoU for the 7 post-hoc calibration methods (Uncalibrated + TS/Logistic/
Dirichlet/LTS/Meta/Selective) on PEPPER test93 @NATIVE (clean), fit@320 -> eval@native.

Companion to code/bup_20_trials/elora/per_class_iou.py (which does the 4 UQ methods:
fullft/lora/mcdropout/ddu). Same output schema and the SAME StreamBinMetrics inter/union
accumulation, so the calibrator per-class JSONs drop straight into the unified 11-method
per-class table. Calibrators sit on the SINGLE model_shot_5 (the study's backbone), fit
once on the clean cal(30) set @320 via psc.fit_calibrators_320 -- identical to the other
fit320_native runners.

Writes <out_dir>/<Method>_seed<seed>.json:
  { "method","seed","M":1, "iou":{class:iou|null}, "inter":{...},"union":{...},
    "n_present", "macro_miou_present", "elapsed_s" }
SLURM only (ssl env). No direct python.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")
sys.path.insert(0, f"{ROOT}/code/bup_20_trials/elora")

import pepper_common as pc                                            # noqa: E402
import pepper_stream_common as psc                                    # noqa: E402
from run_seed_calibration_stream import StreamBinMetrics              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir",
                    default=f"{ROOT}/results/posthoc_calibration_pepper/per_class_iou")
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--fit_size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_images", type=int, default=None, help="smoke cap")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    t0 = time.time()
    dev = a.device
    num = pc.NUM
    names = [pc.ID2LABEL[c] for c in range(num)]
    os.makedirs(a.out_dir, exist_ok=True)

    ckpt = os.path.join(pc.MODELS_DIR, f"seed_{a.seed}", f"{a.shot}.pt")
    print(f"[load] {ckpt}", flush=True)
    model = pc.load_pepper_model(ckpt, dev)

    sp = pc.load_split()
    cal_ds = pc.COCOSegDataset(image_ids=sp["cal_ids"])
    test_ds = pc.COCOSegDataset(image_ids=sp["test_ids"])
    print(f"[split] cal={len(cal_ds)} test={len(test_ds)}", flush=True)

    # fit the 6 calibrators once on clean cal(30) @320 (same as every fit320_native run)
    fitted, _ = psc.fit_calibrators_320(model, cal_ds, dev, a.fit_size,
                                        a.epochs, a.batch_size)
    proc = psc.native_processor()
    tfm = pc._tfm()
    methods = psc.METHODS                                            # Uncalibrated + 6

    idxs = list(range(len(test_ds)))
    if a.max_images:
        idxs = idxs[:a.max_images]

    # one StreamBinMetrics per method; single forward/frame, apply all calibrators
    # (fitting above needs grad; inference below must not -- scope no_grad to the loop)
    accs = {m: StreamBinMetrics(num_classes=num) for m in methods}
    with torch.no_grad():
        for idx in tqdm(idxs, desc=f"test93@native s{a.seed}"):
            pil, sem_np, _ = test_ds[idx]
            W, H = pil.size
            pv = proc([tfm(pil)], return_tensors="pt")["pixel_values"].to(dev)
            seg = pc.build_seg_probs(model(pixel_values=pv), H, W)   # (1,NUM,H,W)
            plog = pc.to_pseudologits(seg)
            img = tfm(pil)[None].to(dev)                             # native, for LTS
            lab = torch.from_numpy(sem_np.astype(np.int64))[None].to(dev)
            for m in methods:
                cl = plog if m == "Uncalibrated" else fitted[m].calibrate(
                    plog, img if m == "LTS" else None)
                probs = F.softmax(cl.float(), dim=1)
                accs[m].update(probs, lab)

    # dump one JSON per method, UQ-compatible schema
    for m in methods:
        inter = accs[m].inter.tolist()
        union = accs[m].union.tolist()
        iou = {names[c]: (inter[c] / union[c] if union[c] > 0 else None)
               for c in range(num)}
        present = [iou[names[c]] for c in range(num) if iou[names[c]] is not None]
        macro = sum(present) / len(present) if present else None
        out = {"method": m, "seed": a.seed, "M": 1,
               "iou": iou,
               "inter": {names[c]: inter[c] for c in range(num)},
               "union": {names[c]: union[c] for c in range(num)},
               "n_present": len(present), "macro_miou_present": macro,
               "elapsed_s": time.time() - t0}
        outp = os.path.join(a.out_dir, f"{m}_seed{a.seed}.json")
        json.dump(out, open(outp, "w"), indent=2)
        print(f"[{m:12s} s{a.seed}] macro-mIoU(present)={macro:.4f}  n_present={len(present)}",
              flush=True)
    print(f"[done] {len(methods)} methods, seed {a.seed} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
