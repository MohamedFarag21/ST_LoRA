# -*- coding: utf-8 -*-
"""
Qualitative OoD SUCCESS vs FAILURE panels for the deployed ST-LoRA ensemble (final_model),
using BOTH anomaly scorers (predictive entropy = total, and mutual information = epistemic /
member disagreement). Backs FINDINGS §11.2/§11.3/§13:
  * SUCCESS = tomato (FAR-OoD): the pepper model has never seen tomato -> entropy/MI light up
    on the fruit, cleanly flagged (AUROC ~1).
  * FAILURE = GrowliFlower cauliflower (NEAR-OoD): the model confidently segments it as
    bg/pepper -> anomaly score stays DARK on the true anomaly, so it is missed (AUROC <= chance).
    Uses the CORRECTED native-palette mask (raw>0), not the buggy convert("L") loader.

ST-LoRA = mean softmax over shots {1,2,3,4} (as-deployed), FRESH base per member (avoids the
documented adapter-stacking bug). Per source we run the ensemble once, compute entropy & MI
(`run_ensemble_ood`), auto-select one frame per source by fg/bg score separation, and render
a 2-row x 5-col figure: input | GT anomaly | entropy | MI | detection(entropy>tau).
SLURM only (ssl env).
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
from PIL import Image                                                 # noqa: E402
from torch.utils.data import DataLoader                               # noqa: E402
from torchvision import transforms                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, HERE)

import ood_eval_comprehensive_with_ddu as oc                          # noqa: E402
from ood_eval_comprehensive_with_ddu import (                         # noqa: E402
    make_transform, GrowliFlowerDataset, run_ensemble_ood, load_base_model,
    ADE_MEAN, ADE_STD)
from pepper_to_tomato_pixel_ood import TomatoPixelDataset             # noqa: E402
from peft import PeftModel                                            # noqa: E402
from torch.utils.data import Subset                                   # noqa: E402


def subsample(ds, n, seed=2024):
    """Fixed-seed random subset (run_ensemble_ood allocates N*C*H*W in RAM -> keep N small)."""
    if not n or len(ds) <= n:
        return ds
    idx = np.random.default_rng(seed).choice(len(ds), n, replace=False)
    return Subset(ds, sorted(int(i) for i in idx))

try:
    from ood_eval_comprehensive_with_ddu import compute_pixel_ood_metrics
except Exception:                                                     # pragma: no cover
    from ood_eval_comprehensive import compute_pixel_ood_metrics      # noqa: E402


def denorm(img_t):
    """(3,H,W) normalized -> (H,W,3) uint8-ish float in [0,1]."""
    x = img_t.numpy().transpose(1, 2, 0)
    x = x * np.array(ADE_STD) + np.array(ADE_MEAN)
    return np.clip(x, 0, 1)


def mask_collate(H, W):
    def _c(batch):
        images, masks = zip(*batch)
        masks_r = [torch.from_numpy(
            np.array(Image.fromarray(m.numpy().astype(np.uint8)).resize((W, H), Image.NEAREST)))
            for m in masks]
        return torch.stack(images), torch.stack(masks_r)
    return _c


def select_frame(result, want="max", min_area=2000):
    """Pick frame by fg/bg entropy separation; want='max' (success) or 'min' (failure)."""
    N = len(result["masks"])
    best = None
    for i in range(N):
        m = result["masks"][i].numpy()
        e = result["entropy"][i].numpy()
        fg, bg = (m > 0), (m == 0)
        if fg.sum() < min_area or bg.sum() < min_area:
            continue
        sep = float(e[fg].mean() - e[bg].mean())
        if best is None or (want == "max" and sep > best[1]) or (want == "min" and sep < best[1]):
            best = (i, sep, int(fg.sum()))
    if best is None:                                                  # fallback: largest fg
        i = max(range(N), key=lambda k: int((result["masks"][k].numpy() > 0).sum()))
        best = (i, 0.0, int((result["masks"][i].numpy() > 0).sum()))
    return best


def _seed_all(s):
    import random as _r
    torch.manual_seed(s); np.random.seed(s); _r.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_members(seed, shot_ids, pretrained, dev):
    paths = [os.path.join(ROOT, "results/lora_paper/hparam_sweep/final_model",
                          f"seed_{seed}", f"model_shot_{s}") for s in shot_ids]
    for p in paths:
        if not os.path.isdir(p):
            raise FileNotFoundError(p)

    def _get(m):                                                      # FRESH base per member
        _seed_all(seed)                                               # reproduce frozen random head
        b = load_base_model(pretrained, dev)
        return PeftModel.from_pretrained(b, paths[m]).eval().to(dev)
    return _get, len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--shot_ids", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--pretrained", default="facebook/mask2former-swin-base-ade-semantic")
    ap.add_argument("--tomato_root", default=f"{ROOT}/data/tomato_esra")
    ap.add_argument("--tomato_split", default="val")
    ap.add_argument("--growli_root", default=f"{ROOT}/data/growliflower_l")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=300, help="per-source subsample cap")
    ap.add_argument("--out_dir", default=f"{ROOT}/results/lora_paper/qualitative_stlora_ood")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = a.height, a.width
    os.makedirs(a.out_dir, exist_ok=True)
    get_model, M = build_members(a.seed, a.shot_ids, a.pretrained, dev)
    print(f"[ST-LoRA final_model seed {a.seed}] M={M} shots={a.shot_ids}", flush=True)

    # ---- tomato (success) ----
    tom_ds = TomatoPixelDataset(a.tomato_root, a.tomato_split, transform=make_transform(H, W))
    tom_ds = subsample(tom_ds, a.max_images)
    tom_loader = DataLoader(tom_ds, batch_size=a.batch_size, shuffle=False,
                            collate_fn=mask_collate(H, W), num_workers=a.num_workers)
    print(f"[tomato:{a.tomato_split}] {len(tom_ds)} imgs", flush=True)
    tom_res = run_ensemble_ood(get_model, tom_loader, dev, H, W, M)

    # ---- GrowliFlower (failure) ----
    gf_ds = GrowliFlowerDataset(a.growli_root, transform=make_transform(H, W), native_palette=True)
    gf_ds = subsample(gf_ds, a.max_images)
    gf_loader = DataLoader(gf_ds, batch_size=a.batch_size, shuffle=False,
                           collate_fn=mask_collate(H, W), num_workers=a.num_workers)
    print(f"[growliflower] {len(gf_ds)} imgs", flush=True)
    gf_res = run_ensemble_ood(get_model, gf_loader, dev, H, W, M)

    # dataset-level AUROC (both scorers) for the caption
    auroc = {}
    for tag, res in (("tomato", tom_res), ("growliflower", gf_res)):
        auroc[tag] = {sc: compute_pixel_ood_metrics(res, scorer=sc)["AUROC"]
                      for sc in ("entropy", "MI")}
        print(f"[{tag}] AUROC entropy={auroc[tag]['entropy']:.3f} MI={auroc[tag]['MI']:.3f}",
              flush=True)

    si, ssep, sarea = select_frame(tom_res, "max")
    fi, fsep, farea = select_frame(gf_res, "min")
    print(f"[SUCCESS tomato] idx={si} sep={ssep:.3f} fg_area={sarea}", flush=True)
    print(f"[FAILURE growli] idx={fi} sep={fsep:.3f} fg_area={farea}", flush=True)

    # ---- re-infer the two chosen frames at NATIVE aspect ratio for display ----
    # (the fixed-size pass above is only for selection + dataset AUROC; rendering at the
    # frame's own proportions avoids the squished look.)
    members = [get_model(m) for m in range(M)]   # load once, reuse for both frames

    def native_ensemble(pil, L=1024):
        w0, h0 = pil.size
        s = L / max(w0, h0)
        Wd = max(32, int(round(w0 * s / 32)) * 32)
        Hd = max(32, int(round(h0 * s / 32)) * 32)
        tfm = transforms.Compose([
            transforms.Resize((Hd, Wd)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)])
        pv = tfm(pil)[None].to(dev)
        sump = sumH = None
        with torch.no_grad():
            for md in members:
                seg = oc.build_seg_probs(md(pv), Hd, Wd)           # (1,C,Hd,Wd)
                H_t = -(seg * (seg + 1e-12).log2()).sum(1)
                sump = seg if sump is None else sump + seg
                sumH = H_t if sumH is None else sumH + H_t
        mean = sump / M
        Hmean = -(mean * (mean + 1e-12).log2()).sum(1)
        ent = Hmean[0].cpu().numpy()
        mi = (Hmean - sumH / M)[0].cpu().numpy()
        img = np.asarray(pil.resize((Wd, Hd))) / 255.0
        return img, ent, mi, (Wd, Hd)

    def tomato_native(ds, i):
        base = ds.dataset if isinstance(ds, Subset) else ds
        oi = ds.indices[i] if isinstance(ds, Subset) else i
        pil, sem, _ = base.ds[oi]
        return pil, (np.asarray(sem) > 0).astype(np.uint8)

    def growli_native(ds, i):
        base = ds.dataset if isinstance(ds, Subset) else ds
        oi = ds.indices[i] if isinstance(ds, Subset) else i
        ip, mp = base.pairs[oi]
        pil = Image.open(ip).convert("RGB")
        raw = np.asarray(Image.open(mp))
        raw = raw[..., 0] if raw.ndim == 3 else raw
        return pil, (raw > 0).astype(np.uint8)

    s_pil, s_mask_n = tomato_native(tom_ds, si)
    f_pil, f_mask_n = growli_native(gf_ds, fi)
    s_img, s_e, s_mi, s_sz = native_ensemble(s_pil)
    f_img, f_e, f_mi, f_sz = native_ensemble(f_pil)

    def rs_mask(mn, sz):
        return np.array(Image.fromarray(mn.astype(np.uint8)).resize(sz, Image.NEAREST))
    s_mask = rs_mask(s_mask_n, s_sz)
    f_mask = rs_mask(f_mask_n, f_sz)

    e_vmax = float(np.percentile(np.concatenate([s_e.ravel(), f_e.ravel()]), 99.5))
    mi_vmax = float(max(np.percentile(np.concatenate([s_mi.ravel(), f_mi.ravel()]), 99.5), 1e-3))
    tau = float(0.5 * (s_e[s_mask > 0].mean() + s_e[s_mask == 0].mean()))

    rows = [("SUCCESS — tomato (far-OoD)", s_img, s_mask, s_e, s_mi, auroc["tomato"]),
            ("FAILURE — GrowliFlower (near-OoD)", f_img, f_mask, f_e, f_mi, auroc["growliflower"])]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for r, (title, img, m, e, mi, au) in enumerate(rows):
        ax = axes[r]
        ax[0].imshow(img)
        ax[0].set_title(f"{title}\ninput", fontsize=10)
        ov = np.zeros((*m.shape, 4)); ov[m > 0] = (1, 0, 0, 0.45)
        ax[1].imshow(img); ax[1].imshow(ov)
        ax[1].set_title("GT anomaly (red)", fontsize=10)
        im2 = ax[2].imshow(e, cmap="magma", vmin=0, vmax=e_vmax)
        ax[2].set_title(f"entropy  (AUROC={au['entropy']:.2f})", fontsize=10)
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        im3 = ax[3].imshow(mi, cmap="viridis", vmin=0, vmax=mi_vmax)
        ax[3].set_title(f"MI / epistemic  (AUROC={au['MI']:.2f})", fontsize=10)
        plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
        det = (e > tau)
        dov = np.zeros((*det.shape, 4)); dov[det] = (0, 1, 1, 0.5)
        ax[4].imshow(img); ax[4].imshow(dov)
        ax[4].set_title(f"detection  (entropy > {tau:.2f})", fontsize=10)
        for a4 in ax:
            a4.axis("off")
    fig.suptitle(f"ST-LoRA (final_model, seed {a.seed}, shots {a.shot_ids}) — OoD detection: "
                 f"tomato (success) vs GrowliFlower (failure)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    comb = os.path.join(a.out_dir, f"ood_success_failure_seed{a.seed}.png")
    fig.savefig(comb, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {comb}", flush=True)

    json.dump({"seed": a.seed, "shot_ids": a.shot_ids, "auroc": auroc,
               "tau": tau, "e_vmax": e_vmax, "mi_vmax": mi_vmax,
               "success": {"idx": si, "sep": ssep, "fg_area": sarea},
               "failure": {"idx": fi, "sep": fsep, "fg_area": farea}},
              open(os.path.join(a.out_dir, f"ood_selection_seed{a.seed}.json"), "w"), indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
