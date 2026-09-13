# -*- coding: utf-8 -*-
"""
CORRECTED DDU-for-segmentation OoD detection (marginalized + symmetric).

Fixes the three deviations of the old `gmm_spatial` scorer from the textbook
DDU-for-segmentation recipe:

  1. MARGINALIZE over classes (not max / predicted-class):
        per-pixel   s(z) = log Σ_c π_c · p(z | c)
                         = logsumexp_c [ log π_c + log N_c(z) ]      (nats)
     with class priors  π_c = (#GT pixels of class c in train) / (#all train pixels).
  2. SYMMETRIC scoring: the SAME function and the SAME (full) resolution for ID and
     OoD — no 512-downscale, no predicted-vs-max asymmetry.
  3. VALIDATION-SET model selection: the GMM grid config is chosen by mean marginal
     NLL on the held-out pepper `valid` split, not by in-sample BIC.

Image score = mean over pixels of s(z) (higher = more in-distribution). Per-image
pixels are sampled uniformly from the FULL-resolution feature grid (identical count
and procedure for ID and OoD) — a symmetric Monte-Carlo estimate of the image mean.

Reuses the fitted per-class spatial GMMs already saved by ddu_gmm_fit.py
(results/lora_paper/ddu/seed_<s>/gmm_spatial_<config>.pkl) and the spectral-normed
DDU model checkpoints. Sources = published image-level pipeline (pepper test ID,
tomato near-OoD, GrowliFlower far-OoD).

Writes <out_dir>/ddu_marginal_seed<seed>.json  and, after aggregation, a summary md.
SLURM only (ssl env). 5 seeds {42,123,456,789,1337}.
"""
import os
import sys
import json
import time
import pickle
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)

import ddu_gmm_fit as dfit                      # noqa: E402  model/loader/split/grid helpers
import image_ood_extra as iox                   # noqa: E402  tomato/growli loaders + metrics_all
from torch.utils.data import DataLoader         # noqa: E402
from torchvision import transforms              # noqa: E402
from transformers import Mask2FormerImageProcessor  # noqa: E402

NUM_LABELS = dfit.NUM_LABELS       # 8
NUM_QUERIES = dfit.NUM_QUERIES     # 100
PRETRAINED = "facebook/mask2former-swin-base-ade-semantic"
SEEDS_DEFAULT = [42, 123, 456, 789, 1337]


# ─────────────────────────────────────────────────────────────────────────────
# Pepper loaders (any split) — yields pixel_values (+ seg_maps for prior counting)
# ─────────────────────────────────────────────────────────────────────────────
def build_pepper_loader(coco_file, root_dir, split, batch_size, num_workers):
    pre = Mask2FormerImageProcessor(ignore_index=255, reduce_labels=False,
                                    do_resize=False, do_rescale=False, do_normalize=False,
                                    num_labels=NUM_LABELS)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=dfit.ADE_MEAN, std=dfit.ADE_STD)])
    ds = dfit.COCOSegDataset(coco_file, root_dir, split, transform=tf)

    def collate(batch):
        imgs, segs = zip(*batch)
        out = pre(list(imgs), segmentation_maps=list(segs), return_tensors="pt")
        out["seg_maps"] = torch.stack(list(segs))
        return out

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                      num_workers=num_workers, persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Class priors  π_c = (#train pixels of class c) / (#all train pixels)
# ─────────────────────────────────────────────────────────────────────────────
def class_priors(train_loader):
    counts = np.zeros(NUM_LABELS, dtype=np.float64)
    for batch in train_loader:
        seg = batch["seg_maps"].numpy()
        for c in range(NUM_LABELS):
            counts[c] += float((seg == c).sum())
    total = counts.sum()
    priors = counts / max(total, 1.0)
    return priors, counts


# ─────────────────────────────────────────────────────────────────────────────
# GMM config loading + marginal scoring
# ─────────────────────────────────────────────────────────────────────────────
def load_config_gmms(seed_dir, config_id, normalization):
    with open(os.path.join(seed_dir, f"gmm_spatial_{config_id}.pkl"), "rb") as f:
        gmms = pickle.load(f)
    mean = std = None
    if normalization == "zscore":
        s = np.load(os.path.join(seed_dir, f"gmm_spatial_{config_id}_stats.npz"))
        mean, std = s["mean"], s["std"]
    return gmms, mean, std


def prep_config(gmms, log_priors):
    """Available (class, gmm) pairs + renormalized log-priors over those classes."""
    avail = [(c, g) for c, g in gmms.items() if g is not None]
    lp = np.array([log_priors[c] for c, _ in avail], dtype=np.float64)
    lp = lp - logsumexp(lp)                       # renormalize over available classes
    return avail, lp


def marginal_pixel_scores(X, avail, lp, mean, std):
    """X: (M,100) full-res pixel features -> (M,) per-pixel marginal log-density."""
    if mean is not None:
        X = (X - mean) / (std + 1e-8)
    cols = np.stack([g.score_samples(X) for _, g in avail], axis=1)   # (M, Cavail)
    return logsumexp(cols + lp[None, :], axis=1)                      # (M,)


@torch.no_grad()
def sample_pixels(model, batch, is_id, device, H, W, ppix, rng):
    """Full-res spatial features for one batch -> list of (ppix,100) per-image arrays."""
    pv = batch["pixel_values"].to(device) if is_id else batch.to(device)
    sf = F.interpolate(model(pv).masks_queries_logits.sigmoid(), size=(H, W),
                       mode="bilinear", align_corners=False)
    sf = sf.permute(0, 2, 3, 1).cpu().numpy()                         # (B,H,W,100)
    per_img = []
    for b in range(sf.shape[0]):
        flat = sf[b].reshape(-1, NUM_QUERIES)
        if ppix and flat.shape[0] > ppix:
            flat = flat[rng.choice(flat.shape[0], ppix, replace=False)]
        per_img.append(flat.astype(np.float32))
    return per_img


@torch.no_grad()
def extract_pixels(model, loader, is_id, device, H, W, ppix, seed):
    rng = np.random.RandomState(seed)
    out = []
    for batch in loader:
        out.extend(sample_pixels(model, batch, is_id, device, H, W, ppix, rng))
    return out


@torch.no_grad()
def score_source(model, loader, is_id, avail, lp, mean, std, device, H, W, ppix, seed):
    """Fused extract+score (memory-lean for big OoD sources). Returns per-image scores."""
    rng = np.random.RandomState(seed)
    scores = []
    for batch in loader:
        for flat in sample_pixels(model, batch, is_id, device, H, W, ppix, rng):
            scores.append(float(marginal_pixel_scores(flat, avail, lp, mean, std).mean()))
    return np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Main (single seed)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ddu_root", default=f"{ROOT}/results/lora_paper/ddu")
    ap.add_argument("--out_dir", default=f"{ROOT}/results/lora_paper/ddu/marginal")
    ap.add_argument("--coco_file",
                    default=f"{ROOT}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    ap.add_argument("--root_dir", default=f"{ROOT}/data/bup_20")
    ap.add_argument("--growliflower_dir", default=f"{ROOT}/data/growliflower_l")
    ap.add_argument("--tomato_root", default=iox.TOMATO_ROOT_DEFAULT)
    ap.add_argument("--tomato_split", default="val")
    ap.add_argument("--tomato_max", type=int, default=0)          # 0 => all tomato frames
    ap.add_argument("--pixels_per_image", type=int, default=8192,  # full-res Monte-Carlo mean
                    help="uniform pixels sampled from the FULL-res grid; same for ID & OoD")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--width", type=int, default=720)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W, ppix = a.height, a.width, a.pixels_per_image
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    seed_dir = os.path.join(a.ddu_root, f"seed_{a.seed}")
    ckpt = os.path.join(seed_dir, "model_final.pt")
    print(f"=== CORRECTED DDU (marginal+symmetric)  seed {a.seed}  full-res {H}x{W}  "
          f"ppix={ppix} ===", flush=True)

    # ---- class priors from TRAIN GT (seed-independent, but cheap to recompute) -----
    train_loader = build_pepper_loader(a.coco_file, a.root_dir, "train",
                                       a.batch_size, a.num_workers)
    priors, counts = class_priors(train_loader)
    log_priors = np.log(np.maximum(priors, 1e-12))
    print("  class priors π_c:", {dfit.ID2LABEL[c]: round(float(priors[c]), 5)
                                   for c in range(NUM_LABELS)}, flush=True)

    # ---- model + sources -----------------------------------------------------------
    model = dfit.load_ddu_model(PRETRAINED, ckpt, dev)
    val_loader = build_pepper_loader(a.coco_file, a.root_dir, "valid", a.batch_size, a.num_workers)
    test_loader = build_pepper_loader(a.coco_file, a.root_dir, "test", a.batch_size, a.num_workers)
    tom_loader = iox.build_tomato_ood_loader(a.tomato_root, a.tomato_split, a.batch_size,
                                             a.num_workers, H=H, W=W, max_images=a.tomato_max)
    gro_loader = iox.build_growli_ood_loader(a.growliflower_dir, a.batch_size, a.num_workers,
                                             H=H, W=W)

    # ---- (3) validation-set config selection (min marginal NLL on pepper valid) -----
    val_px = extract_pixels(model, val_loader, True, dev, H, W, ppix, seed=a.seed)
    print(f"  val frames: {len(val_px)}   selecting GMM config by marginal NLL ...", flush=True)
    grid = []
    best = None
    for config_id, cfg in dfit.all_configs():
        gmms, mean, std = load_config_gmms(seed_dir, config_id, cfg["normalization"])
        avail, lp = prep_config(gmms, log_priors)
        s = np.array([marginal_pixel_scores(x, avail, lp, mean, std).mean() for x in val_px])
        nll = float(-np.mean(s))
        grid.append({"config_id": config_id, "val_nll": nll, "n_fitted": len(avail)})
        if best is None or nll < best["val_nll"]:
            best = {"config_id": config_id, "val_nll": nll, "cfg": cfg}
        print(f"    {config_id:<40} val_NLL={nll:.4f}", flush=True)
    print(f"  >>> selected {best['config_id']}  (val_NLL={best['val_nll']:.4f})", flush=True)

    # ---- score ID + OoD with the selected config (symmetric, full-res) --------------
    gmms, mean, std = load_config_gmms(seed_dir, best["config_id"], best["cfg"]["normalization"])
    avail, lp = prep_config(gmms, log_priors)
    s_id = score_source(model, test_loader, True, avail, lp, mean, std, dev, H, W, ppix, a.seed)
    s_tom = score_source(model, tom_loader, False, avail, lp, mean, std, dev, H, W, ppix, a.seed)
    s_gro = score_source(model, gro_loader, False, avail, lp, mean, std, dev, H, W, ppix, a.seed)
    print(f"  ID mean logdens={s_id.mean():.3f}  tom={s_tom.mean():.3f}  gro={s_gro.mean():.3f}",
          flush=True)

    m_tom = iox.metrics_all(s_id, s_tom)          # higher score = more ID
    m_gro = iox.metrics_all(s_id, s_gro)
    print(f"  tomato AUROC={m_tom['AUROC']:.4f} FPR95={m_tom['FPR95']:.4f}  |  "
          f"growli AUROC={m_gro['AUROC']:.4f} FPR95={m_gro['FPR95']:.4f}", flush=True)

    out = {"seed": a.seed, "method": "DDU-marginal (logsumexp over classes, priors, symmetric)",
           "selected_config": best["config_id"], "val_nll": best["val_nll"],
           "eval": f"native_{H}x{W}", "pixels_per_image": ppix,
           "priors": {dfit.ID2LABEL[c]: float(priors[c]) for c in range(NUM_LABELS)},
           "n": {"id_test": len(s_id), "tomato": len(s_tom), "growli": len(s_gro)},
           "scores_mean": {"id": float(s_id.mean()), "tomato": float(s_tom.mean()),
                           "growli": float(s_gro.mean())},
           "tomato": m_tom, "growli": m_gro, "grid": grid,
           "elapsed_s": time.time() - t0}
    outp = os.path.join(a.out_dir, f"ddu_marginal_seed{a.seed}.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
