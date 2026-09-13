# -*- coding: utf-8 -*-
"""
Shared native-resolution streaming helpers for the PEPPER fit@320 -> eval@native study.

Pepper native is 720x1280 (W x H) — same geometry as tomato — so `cal_size=320` is a
true 4x downscale to 320x180 (h x w), exactly matching the tomato protocol.

Why streaming: we never materialize a giant (Npix, C) tensor. ECE goes through the
float64 `StreamBinMetrics` (torchmetrics' float32 bin accumulation corrupts ECE above
~16.7M px/bin — pepper val33 is 30.4M px, ALREADY over that line), and OoD goes through
float64 `OODHist` with the 8-class entropy bound ln(8).
"""
import math
import numpy as np
import torch
from tqdm import tqdm
from transformers import Mask2FormerImageProcessor

import pepper_common as pc
from calibrators import Calibrator, NAMES, pixel_entropy

LN_NUM = math.log(pc.NUM)          # 8-class max entropy = ln(8) ~ 2.0794
METHODS = ["Uncalibrated"] + list(NAMES)


def native_processor():
    """do_resize=False -> the model sees full native 1280x720."""
    return Mask2FormerImageProcessor(ignore_index=255, reduce_labels=False,
                                     do_resize=False, do_rescale=False,
                                     do_normalize=False, num_labels=pc.NUM)


def fit_calibrators_320(model, cal_ds, device, fit_size, epochs, batch_size,
                        max_images=None, verbose=True):
    """Fit all 6 calibrators on the cal(30) subset cached at fit_size (320x180)."""
    import time
    idx = list(range(len(cal_ds)))
    cache = pc.predict_cache(model, cal_ds, idx, device, fit_size, max_images=max_images)
    if verbose:
        print(f"[fit cache] {tuple(cache['logits'].shape)}", flush=True)
    fitted = {}
    for name in NAMES:
        t = time.time()
        fitted[name] = Calibrator(name, num_class=pc.NUM).fit(
            cache, device, epochs=epochs, batch_size=batch_size)
        if verbose:
            print(f"[fit {name}] {time.time()-t:.1f}s", flush=True)
    return fitted, cache


@torch.no_grad()
def stream_native(model, ds, indices, proc, dev, fitted, methods, cb, desc,
                  binary_anomaly=False):
    """One native forward per frame; calls cb(idx, out_dict{method: (C,H,W) probs or
    (H,W) entropy}, gt_np).

    `binary_anomaly=True` -> gt is the (sem>0) anomaly mask and cb receives per-method
    ENTROPY maps (OoD path). Otherwise gt is the 8-class label map and cb receives
    per-method PROBS (calibration/ECE path).
    """
    tfm = pc._tfm()
    for idx in tqdm(indices, desc=desc):
        pil, sem_np, _ = ds[idx]
        W, H = pil.size
        pv = proc([tfm(pil)], return_tensors="pt")["pixel_values"].to(dev)
        seg = pc.build_seg_probs(model(pixel_values=pv), H, W)     # (1,NUM,H,W)
        plog = pc.to_pseudologits(seg)                             # (1,NUM,H,W)
        img = tfm(pil)[None].to(dev)
        gt = (sem_np > 0).astype(np.uint8) if binary_anomaly else sem_np.astype(np.int64)
        out = {}
        for m in methods:
            cl = plog if m == "Uncalibrated" else fitted[m].calibrate(
                plog, img if m == "LTS" else None)
            out[m] = pixel_entropy(cl.float())[0] if binary_anomaly else cl.float()
        cb(idx, out, gt)
