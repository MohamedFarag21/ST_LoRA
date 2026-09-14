# -*- coding: utf-8 -*-
"""
Spatial uncertainty DECOMPOSITION maps — FRE (Full-FT snapshots) vs ST-LoRA (LoRA-adapter
snapshots) deep/snapshot ensembles, Mask2Former, bup20 sweet-pepper, FULL-AUGMENTATION deployed
models (FRE=full_ft, ST-LoRA=hparam_sweep/final_model — both aug-trained). For M ensemble members
with per-pixel class probabilities p_m(y|x):

  mean prob      p̄ = (1/M) Σ_m p_m
  TOTAL      (predictive) = H[p̄]           = −Σ_c p̄_c log p̄_c        (Shannon entropy of the mean)
  ALEATORIC  (data)       = (1/M) Σ_m H[p_m]                          (weighted-avg member entropy)
  EPISTEMIC  (model)      = MUTUAL INFO     = TOTAL − ALEATORIC        (BALD; ≥0, member disagreement)

Renders, for 2 images × {FRE, ST-LoRA}, a 4-row × 4-col grid [Input | Total | Aleatoric | Epistemic],
with per-(image,component) color scales SHARED across the two methods so FRE vs ST-LoRA is directly
comparable. Reuses calibration_shift_eval (cse) loaders + build_seg_probs, and the diversity script's
member-loading (ST-LoRA re-seeds before each fresh-base PeftModel load — frozen-head bug).

SLURM only, ssl env (read-only inference). Writes results/lora_paper/uncertainty_decomp/*.
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import skimage.draw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")

import calibration_shift_eval as cse                                   # noqa: E402
from peft import PeftModel                                            # noqa: E402

EPS = 1e-12
COMPONENTS = ["Total (Shannon H[p̄])", "Aleatoric (mean H[p_m])", "Epistemic (MI = Total − Aleatoric)"]
CMAP = "magma"


def _seed_all(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_members(method, seed, args, device):
    """Return (get_model, M). get_model(m) loads member m fresh (mirrors run_uq / diversity)."""
    if method == "stlora":
        paths = [os.path.join(args.lora_dir, args.config_name, f"seed_{seed}",
                              f"model_shot_{s}") for s in args.shot_ids]
        def _get(m):
            _seed_all(seed)                                            # frozen-head determinism
            b = cse.load_base_model(args.pretrained, device)
            return PeftModel.from_pretrained(b, paths[m]).eval().to(device)
        return _get, len(paths)
    else:  # fre
        ckpts = [os.path.join(args.fullft_dir, f"seed_{seed}", f"model_shot_{s}.pt")
                 for s in args.shot_ids]
        return (lambda m: cse.load_fullft_snapshot(args.pretrained, ckpts[m], device)), len(ckpts)


def denorm(pv):
    """(3,H,W) normalized tensor -> (H,W,3) uint8 RGB for display."""
    mean = torch.tensor(cse.ADE_MEAN, dtype=pv.dtype).view(3, 1, 1)
    std = torch.tensor(cse.ADE_STD, dtype=pv.dtype).view(3, 1, 1)
    img = (pv.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def auto_pick_foreground(dataset, k=2):
    """Top-k test frames by GT foreground fraction (rasterize polygons; no JPEG load)."""
    fracs = []
    for i, info in enumerate(dataset.images):
        H, W = info["height"], info["width"]
        sem = np.zeros((H, W), dtype=np.uint8)
        for ann in dataset.ann_lookup.get(info["id"], []):
            for poly in ann.get("segmentation", []):
                pts = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(pts[:, 1], pts[:, 0], sem.shape)
                sem[rr, cc] = ann["category_id"]
        fracs.append((cse._REMAP_LUT[sem.astype(np.int64)] > 0).mean())
    order = np.argsort(fracs)[::-1]
    return sorted(order[:k].tolist()), fracs


@torch.no_grad()
def compute_maps(method, seed, args, dev, target_ids):
    get_model, M = build_members(method, seed, args, dev)
    loader = cse.build_loader(args.coco_file, args.root_dir, None, None, 1, args.num_workers)
    acc = {fi: dict(sum_p=None, sum_H=None, img=None, gt=None) for fi in target_ids}
    for mi in range(M):
        model = get_model(mi)
        fi = 0
        for batch in loader:
            pv = batch["pixel_values"]
            seg_maps = batch["seg_maps"]                               # (1,H,W)
            if fi in target_ids:
                Hh, Ww = seg_maps.shape[-2:]
                seg = cse.build_seg_probs(model(pv.to(dev)), Hh, Ww)[0]  # (C,H,W)
                probs = seg.clamp_min(EPS)
                probs = probs / probs.sum(0, keepdim=True)
                ent = -(probs * probs.log()).sum(0)                   # (H,W) member entropy
                a = acc[fi]
                if a["sum_p"] is None:
                    a["sum_p"] = probs.double().cpu(); a["sum_H"] = ent.double().cpu()
                    a["img"] = denorm(pv[0]); a["gt"] = seg_maps[0].cpu().numpy()
                else:
                    a["sum_p"] += probs.double().cpu(); a["sum_H"] += ent.double().cpu()
            fi += 1
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [{method}] member {mi+1}/{M} done", flush=True)

    out = {}
    for fi, a in acc.items():
        p_bar = a["sum_p"] / M
        p_bar = p_bar / p_bar.sum(0, keepdim=True)
        total = -(p_bar * p_bar.clamp_min(EPS).log()).sum(0).numpy()  # H[mean]
        alea = (a["sum_H"] / M).numpy()                               # mean member entropy
        epi = np.clip(total - alea, 0, None)                          # mutual information
        out[fi] = dict(img=a["img"], gt=a["gt"], total=total, alea=alea, epi=epi)
    return out, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shot_ids", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--image_ids", nargs="+", type=int, default=None,
                    help="test-frame indices; default = auto top-2 by GT foreground")
    ap.add_argument("--coco_file",
                    default=f"{ROOT}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    ap.add_argument("--root_dir", default=f"{ROOT}/data/bup_20")
    ap.add_argument("--pretrained", default="facebook/mask2former-swin-base-ade-semantic")
    ap.add_argument("--lora_dir", default=f"{ROOT}/results/lora_paper/hparam_sweep")
    ap.add_argument("--config_name", default="final_model")
    ap.add_argument("--fullft_dir", default=f"{ROOT}/results/lora_paper/full_ft")
    ap.add_argument("--out_dir", default=f"{ROOT}/results/lora_paper/uncertainty_decomp")
    ap.add_argument("--num_workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choose the 2 images
    ds = cse.CorruptedTestDataset(a.coco_file, a.root_dir, None, None)
    if a.image_ids:
        target_ids = sorted(a.image_ids)
    else:
        target_ids, fracs = auto_pick_foreground(ds, k=2)
        print(f"[auto] picked frames {target_ids} "
              f"(fg frac {[round(fracs[i],3) for i in target_ids]})", flush=True)

    res = {}
    for method in ["fre", "stlora"]:
        print(f"=== computing {method} (seed {a.seed}, shots {a.shot_ids}) ===", flush=True)
        res[method], M = compute_maps(method, a.seed, a, dev, target_ids)

    # save raw maps
    np.savez_compressed(
        os.path.join(a.out_dir, f"decomp_maps_seed{a.seed}.npz"),
        **{f"{m}_{fi}_{k}": res[m][fi][k]
           for m in res for fi in target_ids for k in ("total", "alea", "epi")})

    # ── figure: 4 rows (img×method) × 4 cols (input, total, alea, epi) ────────
    methods = [("fre", "FRE (Full-FT)"), ("stlora", "ST-LoRA")]
    nrows = len(target_ids) * len(methods)
    fig, axes = plt.subplots(nrows, 4, figsize=(19, 4.4 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    # per-(image,component) shared clim across the two methods (robust 99th pct)
    clim = {}
    for fi in target_ids:
        for ck in ("total", "alea", "epi"):
            vmax = max(np.percentile(res[m][fi][ck], 99.0) for m, _ in methods)
            clim[(fi, ck)] = max(vmax, 1e-6)

    r = 0
    for fi in target_ids:
        for method, mlabel in methods:
            d = res[method][fi]
            ax = axes[r, 0]
            ax.imshow(d["img"]); ax.set_ylabel(f"{mlabel}\nframe {fi}", fontsize=11, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("Input", fontsize=12, fontweight="bold")
            for ci, (ck, cname) in enumerate(zip(("total", "alea", "epi"), COMPONENTS)):
                ax = axes[r, ci + 1]
                vmax = clim[(fi, ck)]
                im = ax.imshow(d[ck], cmap=CMAP, vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(f"mean {d[ck].mean():.3f}  max {d[ck].max():.3f}", fontsize=9)
                if r == 0:
                    ax.set_title(cname, fontsize=11.5, fontweight="bold")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=7)
            r += 1

    fig.suptitle(
        f"Predictive-uncertainty decomposition — FRE vs ST-LoRA snapshot ensembles "
        f"(M={M}, seed {a.seed}, full-aug Mask2Former, bup20)\n"
        f"Total = Shannon entropy of ensemble mean · Aleatoric = mean member entropy · "
        f"Epistemic = mutual information (Total − Aleatoric).  "
        f"Color scale shared across methods within each (frame, component).",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(a.out_dir, f"uncertainty_decomp_seed{a.seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}", flush=True)
    for fi in target_ids:
        for method, mlabel in methods:
            d = res[method][fi]
            print(f"  frame {fi} {mlabel:14s}: total={d['total'].mean():.4f} "
                  f"alea={d['alea'].mean():.4f} epi={d['epi'].mean():.4f} "
                  f"(epi/total={d['epi'].mean()/max(d['total'].mean(),1e-9):.2%})", flush=True)


if __name__ == "__main__":
    main()
