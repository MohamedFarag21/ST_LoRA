# -*- coding: utf-8 -*-
"""Lightweight functional unit tests for the parts that don't need GPU/data/model downloads:
  * Experiment 3 corruption functions (shift_corruptions) — apply each to a synthetic image+mask.
  * Experiment 2 config catalog (build_catalog) — config counts per architecture.
Run under the main (ssl) env. Exit code = number of failed checks.
"""
import os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        fails.append(name)


# ---- Experiment 3: shift corruptions ----
sys.path.insert(0, os.path.join(REL, "experiment_3_covariate_shift", "code"))
try:
    import shift_corruptions as sc
    img = Image.fromarray((np.random.rand(368, 448, 3) * 255).astype(np.uint8), "RGB")
    seg = np.zeros((368, 448), dtype=np.uint8); seg[100:200, 100:200] = 1; seg[:10, :] = 255
    params = {"brightness": 0.5, "contrast": 0.5, "gaussian_noise": 0.1, "gaussian_blur": 2,
              "rotation": 30.0, "zoom": 1.25, "translation": 0.1, "hflip": None}
    all_ok = True; detail = ""
    for cname, p in params.items():
        photom = cname in sc.PHOTOMETRIC
        out_img, out_seg = sc.apply(cname, img, None if photom else seg, p)
        if out_img.size != (448, 368):
            all_ok = False; detail = f"{cname}: bad size {out_img.size}"; break
        if not photom and not set(np.unique(out_seg).tolist()).issubset({0, 1, 255}):
            all_ok = False; detail = f"{cname}: mask values leaked"; break
    check("exp3.corruptions apply cleanly (image+mask, labels in {0,1,255})", all_ok, detail)
    n_bup = len(list(sc.iter_conditions(sc.BUP20_SHIFTS)))
    check("exp3.BUP20 shift grid == 19 conditions", n_bup == 19, f"got {n_bup}")
    n_gro = len(list(sc.iter_conditions(sc.GROWLI_SHIFTS, active=sc.GROWLI_ACTIVE)))
    check("exp3.GrowliFlower active grid == 80 (4x20)", n_gro == 80, f"got {n_gro}; active={sc.GROWLI_ACTIVE}")
except Exception as e:
    check("exp3.shift_corruptions import", False, f"{type(e).__name__}: {e}")

# ---- Experiment 2: config catalog counts ----
sys.path.insert(0, os.path.join(REL, "experiment_2_hyperparameters", "code"))
try:
    import build_catalog as bc
    counts = {a: len(bc.build(a)) for a in ("m2f", "segformer", "eomt")}
    check("exp2.catalog counts (m2f=35, segformer=27, eomt=27)",
          counts == {"m2f": 35, "segformer": 27, "eomt": 27}, str(counts))
except Exception as e:
    check("exp2.build_catalog import", False, f"{type(e).__name__}: {e}")

print(f"\n[functional_tests] {len(fails)} failed" + (": " + ", ".join(fails) if fails else " — all passed"))
sys.exit(len(fails))
