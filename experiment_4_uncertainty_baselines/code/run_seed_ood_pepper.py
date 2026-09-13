# -*- coding: utf-8 -*-
"""
Phase 2b — pixel-level OoD of the CALIBRATED pepper (8-class) outputs, with TWO
anomaly sources (novel-domain detection by an in-domain pepper model):

  * tomato       : anomaly (1) = tomato fruit (sem>0), normal = background
  * growliflower : anomaly (1) = cauliflower plant (mask>0), normal = background

score = per-pixel entropy of the calibrated 8-class probs (higher = more OoD).

For one --seed:
  * fit all 6 calibrators on the pepper cal(30) cache (reuse seed_<S>/cache/cal.pt);
  * run pepper model_shot_5 on N=500 frames of each OoD source (cache 8-class logits
    + anomaly mask at 64px);
  * apply each calibrator (+ uncalibrated), entropy -> compute_pixel_ood_metrics
    (AUROC/AUPR/FPR95/sIoU/PPV/MeanF1), reusing the exact function so numbers are
    comparable to the other methods.

Run via SLURM (ssl env). No direct python.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from PIL import Image

ROOT = "/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204"
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration_pepper")
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")
sys.path.insert(0, f"{ROOT}/code/bup_20_trials/elora")

import pepper_common as pc
from calibrators import Calibrator, NAMES, pixel_entropy
from ood_eval_comprehensive import compute_pixel_ood_metrics, GrowliFlowerDataset
from mask2former_lora_train_tomato import TomatoDataset


class GrowliRaw(torch.utils.data.Dataset):
    """Yields (pil_img, plant_mask_np[HxW 0/1], id) — for predict_cache(binary_anomaly)."""
    def __init__(self, root=pc.GROWLI_ROOT):
        self.pairs = GrowliFlowerDataset(root, transform=None).pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        pil = Image.open(img_path).convert("RGB")
        # GrowliFlower maskPlants are palette-mode (mode "P"): native indices are
        # 0=background, >0=cauliflower plant. Read the RAW palette indices — NOT
        # .convert("L"), which remaps index 0 -> luminance 60 and would mark the
        # whole frame as anomaly.
        raw = np.asarray(Image.open(msk_path))
        if raw.ndim == 3:
            raw = raw[..., 0]
        binm = (raw > 0).astype(np.int64)
        return pil, binm, idx


def ood_from_logits(cal_logits, labels):
    ood_result = {"masks": [labels[i] for i in range(labels.shape[0])],
                  "entropy": pixel_entropy(cal_logits)}
    return compute_pixel_ood_metrics(ood_result, scorer="entropy")


def subsample(n, k, seed):
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(n, size=min(k, n), replace=False).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", default=pc.OUT_DIR)
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--cal_size", type=int, default=64)
    ap.add_argument("--n_ood", type=int, default=500)
    ap.add_argument("--ood_seed", type=int, default=2024)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_images", type=int, default=None)   # smoke
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    seed_dir = os.path.join(args.out_dir, f"seed_{args.seed}")
    ckpt = os.path.join(pc.MODELS_DIR, f"seed_{args.seed}", f"{args.shot}.pt")
    model = pc.load_pepper_model(ckpt, args.device)

    # ── fit calibrators on the pepper cal(30) cache (reuse phase-1 cache) ──
    cal_cache_p = os.path.join(seed_dir, "cache", "cal.pt")
    if os.path.exists(cal_cache_p):
        fit_cache = torch.load(cal_cache_p)
        print(f"[fit] reuse {cal_cache_p} {tuple(fit_cache['logits'].shape)}", flush=True)
    else:
        split = pc.load_split()
        cal_ds = pc.COCOSegDataset(image_ids=split["cal_ids"])
        fit_cache = pc.predict_cache(model, cal_ds, list(range(len(cal_ds))),
                                     args.device, args.cal_size)
    calibrators = {n: Calibrator(n, num_class=pc.NUM).fit(
        fit_cache, args.device, epochs=args.epochs, batch_size=args.batch_size) for n in NAMES}
    print(f"[fit] {len(calibrators)} calibrators ({time.time()-t0:.1f}s)", flush=True)

    # ── OoD sources ──
    k = args.max_images if args.max_images is not None else args.n_ood
    sources = {
        "tomato": TomatoDataset(pc.TOMATO_ROOT, split="val"),
        "growliflower": GrowliRaw(),
    }

    results = {}
    for sname, ds in sources.items():
        idx = subsample(len(ds), k, args.ood_seed)
        cache = pc.predict_cache(model, ds, idx, args.device, args.cal_size,
                                 binary_anomaly=True)
        print(f"[{sname}] cache logits{tuple(cache['logits'].shape)} "
              f"anomaly_frac={(cache['labels']>0).float().mean().item():.3f}", flush=True)
        res = {"Uncalibrated": ood_from_logits(cache["logits"].float(), cache["labels"])}
        print(f"  [Uncalibrated] AUROC={res['Uncalibrated']['AUROC']:.4f}", flush=True)
        for name in NAMES:
            outs = []
            N = cache["logits"].shape[0]
            for i in range(0, N, args.batch_size):
                lo = cache["logits"][i:i+args.batch_size].float().to(args.device)
                im = cache["image"][i:i+args.batch_size].float().to(args.device) if name == "LTS" else None
                outs.append(calibrators[name].calibrate(lo, im).cpu())
            res[name] = ood_from_logits(torch.cat(outs, 0), cache["labels"])
            print(f"  [{name}] AUROC={res[name]['AUROC']:.4f} FPR95={res[name]['FPR95']:.4f}", flush=True)
        results[sname] = res

    out = {"seed": args.seed, "shot": args.shot, "n_ood": k, "ood_seed": args.ood_seed,
           "num_class": pc.NUM, "elapsed_s": time.time() - t0, "metrics": results}
    outp = os.path.join(seed_dir, "ood_metrics.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
