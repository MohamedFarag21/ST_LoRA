# -*- coding: utf-8 -*-
"""
Qualitative SUCCESS vs FAILURE panels for the deployed ST-LoRA ensemble (final_model),
on bup20 pepper test93 @native (clean). Backs the per-class story (FINDINGS §11.5-11.7):
  * SUCCESS = common class the low-rank adapter handles as well as full-FT -> best `pepper
    green` per-image IoU (green wins even over FRE on NLL in §11.6).
  * FAILURE = rare "mixed" subtype the adapter under-fits -> worst `pepper mixed_red`
    per-image IoU among frames where mixed_red is actually present (§11.7: members argmax-
    disagree most exactly here).

Frames are chosen by per-image metric (NOT by eye). For each chosen frame we render a
4-panel row: input | GT overlay | ST-LoRA ensemble prediction | per-pixel entropy.
ST-LoRA = mean softmax over shots {1,2,3,4} (as-deployed), fresh base per member.
SLURM only (ssl env). Writes <out_dir>/{success,failure}_seed<seed>.png (+ a combined png).
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")

import calibration_shift_eval as cse                                  # noqa: E402
from run_uq_shift_native_stream import build_get_model               # noqa: E402
from types import SimpleNamespace                                     # noqa: E402
from PIL import Image                                                 # noqa: E402

# contiguous class ids (see cse.LABEL2ID): 0 bg,1 kp,2 red,3 yellow,4 green,5 mixed,
# 6 mixed_red,7 mixed_yellow
GREEN, MIXED_RED = 4, 6
PALETTE = {
    0: None,                       # bg -> transparent
    1: (0.50, 0.50, 0.50),         # pepper_kp
    2: (0.85, 0.10, 0.10),         # red
    3: (0.95, 0.75, 0.10),         # yellow
    4: (0.15, 0.65, 0.20),         # green
    5: (0.60, 0.20, 0.70),         # mixed
    6: (1.00, 0.50, 0.00),         # mixed_red
    7: (0.70, 0.80, 0.20),         # mixed_yellow
}


def per_image_iou(pred, gt, c):
    gc = gt == c
    pc = pred == c
    inter = float((gc & pc).sum())
    union = float((gc | pc).sum())
    return (inter / union) if union > 0 else None, int(gc.sum())


def colorize(lbl):
    """(H,W) int -> (H,W,4) RGBA overlay, bg transparent."""
    H, W = lbl.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    for c, col in PALETTE.items():
        if col is None:
            continue
        m = lbl == c
        rgba[m, 0], rgba[m, 1], rgba[m, 2] = col
        rgba[m, 3] = 0.55
    return rgba


def load_pil(ds, i):
    info = ds.images[i]
    rel = info["path"].lstrip("/datasets/")
    return Image.open(os.path.join(ds.root_dir, rel)).convert("RGB")


def render_row(ax_row, img, gt, pred, ent, title):
    ax_row[0].imshow(img); ax_row[0].set_title(f"{title}\ninput", fontsize=9)
    ax_row[1].imshow(img); ax_row[1].imshow(colorize(gt)); ax_row[1].set_title("ground truth", fontsize=9)
    ax_row[2].imshow(img); ax_row[2].imshow(colorize(pred)); ax_row[2].set_title("ST-LoRA prediction", fontsize=9)
    im = ax_row[3].imshow(ent, cmap="magma", vmin=0.0)
    ax_row[3].set_title("predictive entropy", fontsize=9)
    plt.colorbar(im, ax=ax_row[3], fraction=0.046, pad=0.04)
    for a in ax_row:
        a.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--shot_ids", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--green_min", type=int, default=3000, help="min GT green px for success cand")
    ap.add_argument("--mixedred_min", type=int, default=300, help="min GT mixed_red px for failure cand")
    ap.add_argument("--out_dir", default=f"{ROOT}/results/lora_paper/qualitative_stlora")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    num = cse.NUM_LABELS

    args = SimpleNamespace(
        method="lora", seed=a.seed,
        coco_file=f"{ROOT}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json",
        root_dir=f"{ROOT}/data/bup_20",
        pretrained="facebook/mask2former-swin-base-ade-semantic",
        lora_dir=f"{ROOT}/results/lora_paper/hparam_sweep", config_name="final_model",
        shot_ids=a.shot_ids, batch_size=2, num_workers=4, height=1280, width=720)

    get_model, M = build_get_model(args, dev)
    print(f"[ST-LoRA final_model seed {a.seed}] M={M} shots={a.shot_ids}", flush=True)
    loader = cse.build_loader(args.coco_file, args.root_dir, None, None,
                              args.batch_size, args.num_workers)
    ds = loader.dataset
    probs, gt = cse.run_inference(get_model, M, loader, dev, args.height, args.width)
    torch.cuda.empty_cache()

    pred = probs.argmax(1)                                            # (N,H,W)
    ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(1)  # (N,H,W)
    N = probs.shape[0]

    # per-frame selection metrics
    rows = []
    for i in range(N):
        g = gt[i].numpy(); p = pred[i].numpy()
        g_iou, g_area = per_image_iou(p, g, GREEN)
        m_iou, m_area = per_image_iou(p, g, MIXED_RED)
        rows.append({"i": i, "id": int(ds.images[i]["id"]),
                     "green_iou": g_iou, "green_area": g_area,
                     "mixedred_iou": m_iou, "mixedred_area": m_area,
                     "mean_ent": float(ent[i].mean())})

    # SUCCESS: green present & sizeable -> max green IoU (tie-break: lower mean entropy)
    succ_c = [r for r in rows if r["green_area"] >= a.green_min and r["green_iou"] is not None]
    succ_c.sort(key=lambda r: (-r["green_iou"], r["mean_ent"]))
    success = succ_c[0] if succ_c else max(rows, key=lambda r: (r["green_iou"] or -1))

    # FAILURE: mixed_red present & sizeable -> min mixed_red IoU (tie-break: larger area = more visible)
    fail_c = [r for r in rows if r["mixedred_area"] >= a.mixedred_min and r["mixedred_iou"] is not None]
    fail_c.sort(key=lambda r: (r["mixedred_iou"], -r["mixedred_area"]))
    failure = fail_c[0] if fail_c else min(rows, key=lambda r: (r["mixedred_iou"] if r["mixedred_iou"] is not None else 1.0))

    print(f"[SUCCESS] frame idx={success['i']} id={success['id']} "
          f"green_IoU={success['green_iou']:.3f} green_area={success['green_area']}", flush=True)
    print(f"[FAILURE] frame idx={failure['i']} id={failure['id']} "
          f"mixedred_IoU={failure['mixedred_iou']} mixedred_area={failure['mixedred_area']}", flush=True)

    # shared entropy scale across the two panels
    emax = float(max(ent[success["i"]].max(), ent[failure["i"]].max()))

    def _draw(sel, tag, title):
        img = np.asarray(load_pil(ds, sel["i"]))
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        render_row(axes, img, gt[sel["i"]].numpy(), pred[sel["i"]].numpy(),
                   ent[sel["i"]].numpy(), title)
        for a4 in [axes[3]]:
            a4.images[0].set_clim(0.0, emax)
        fig.tight_layout()
        outp = os.path.join(a.out_dir, f"{tag}_seed{a.seed}.png")
        fig.savefig(outp, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"[saved] {outp}", flush=True)
        return outp

    _draw(success, "success",
          f"SUCCESS — common class (pepper green), IoU={success['green_iou']:.2f}")
    _draw(failure, "failure",
          f"FAILURE — rare subtype (pepper mixed_red), IoU={failure['mixedred_iou']:.2f}")

    # combined 2-row figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    render_row(axes[0], np.asarray(load_pil(ds, success["i"])), gt[success["i"]].numpy(),
               pred[success["i"]].numpy(), ent[success["i"]].numpy(),
               f"SUCCESS · green IoU={success['green_iou']:.2f}")
    render_row(axes[1], np.asarray(load_pil(ds, failure["i"])), gt[failure["i"]].numpy(),
               pred[failure["i"]].numpy(), ent[failure["i"]].numpy(),
               f"FAILURE · mixed_red IoU={failure['mixedred_iou']:.2f}")
    for r in (0, 1):
        axes[r][3].images[0].set_clim(0.0, emax)
    fig.suptitle(f"ST-LoRA (final_model, seed {a.seed}, ensemble shots {a.shot_ids}) — "
                 f"success vs failure on bup20 test93", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    comb = os.path.join(a.out_dir, f"success_failure_seed{a.seed}.png")
    fig.savefig(comb, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {comb}", flush=True)

    json.dump({"seed": a.seed, "shot_ids": a.shot_ids,
               "success": success, "failure": failure,
               "legend": {cse.ID2LABEL[c]: PALETTE[c] for c in range(num)}},
              open(os.path.join(a.out_dir, f"selection_seed{a.seed}.json"), "w"), indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
