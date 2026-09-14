# -*- coding: utf-8 -*-
"""
Per-class ENSEMBLE-MEMBER DIVERSITY for the two snapshot ensembles used in the pepper
UQ study — FRE (Full-FT snapshots) and ST-LoRA (LoRA-adapter snapshots) — on bup20
test93 @native (1280x720, clean).

This is the diversity companion to the per-class IoU / Brier / NLL tables. The paper
attributes FRE's edge over ST-LoRA to greater member diversity; this script measures that
directly, and does so PER GROUND-TRUTH CLASS c (over the pixels whose true label is c),
exactly like per-class IoU — so we can see *which pepper classes* the two ensembles
disagree on, not just a global average.

Member convention: to compare diversity fairly the two ensembles use the SAME member
count and the SAME snapshot epochs. Default shots=[1,2,3,4] (M=4), each snapshot saved
every 20 epochs (shot_s = epoch 20*s), available for BOTH methods. (ST-LoRA-as-deployed
also uses these 4 adapters; FRE-as-deployed adds a 5th snapshot — dropped here only to
match M so the diversity magnitudes are directly comparable. Use --shot_ids 1 2 3 4 5 to
reproduce the as-deployed M=5 ensembles.)

Diversity metrics (per pixel, then reduced into per-GT-class buckets):
  * disagreement        mean pairwise argmax DISAGREEMENT rate               [prediction-level]
  * sym_kl              mean pairwise SYMMETRIC KL between members           [distribution-level]
  * mutual_info         H[mean] - mean_m H[member] = generalized JSD = EPISTEMIC uncertainty
  * expected_entropy    mean_m H[member]                                     [aleatoric]
  * predictive_entropy  H[mean]  ( = expected_entropy + mutual_info )        [total]
  * frac_disagree       fraction of pixels with ANY pairwise argmax disagreement

Streaming per-member running sums (never hold all M members at once):
  S_p = sum_m p_m,  S_L = sum_m log p_m,  S_H = sum_m H[p_m],  votes = sum_m onehot(argmax_m).
  mean pairwise symmetric KL = (-M*S_H - sum_c S_p*S_L) / (M(M-1))     [exact]

Runs in the ssl env (Mask2Former + peft), SLURM only. Writes
  <out_dir>/<method>_seed<seed>_diversity.json
with per_class / macro / all / fg blocks.
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

import calibration_shift_eval as cse                                  # noqa: E402
from peft import PeftModel                                            # noqa: E402

EPS = 1e-12


def _seed_all(s):
    import random as _r
    torch.manual_seed(s); np.random.seed(s); _r.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_members(method, seed, args, device):
    """Return (get_model, M). get_model(m) loads member m fresh (mirrors run_uq)."""
    if method == "lora":
        paths = [os.path.join(args.lora_dir, args.config_name, f"seed_{seed}",
                              f"model_shot_{s}") for s in args.shot_ids]
        for p in paths:
            if not os.path.isdir(p):
                raise FileNotFoundError(p)

        # FRESH base per member: reusing an adapted base silently re-wraps it and corrupts
        # the effective weights of members 1..M-1 (documented in run_uq_shift_native_stream).
        # SEED before each load: the adapter (modules_to_save=null) does NOT store the frozen,
        # randomly-initialized class_predictor head (ignore_mismatched_sizes). Training fixes it
        # via seed_everything(seed) before from_pretrained; without re-seeding here EACH member
        # draws a DIFFERENT random head -> nondeterministic AND spurious member "diversity".
        def _get(m):
            _seed_all(seed)
            b = cse.load_base_model(args.pretrained, device)
            return PeftModel.from_pretrained(b, paths[m]).eval().to(device)
        return _get, len(paths)
    if method == "fullft":
        ckpts = [os.path.join(args.fullft_dir, f"seed_{seed}", f"model_shot_{s}.pt")
                 for s in args.shot_ids]
        for p in ckpts:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        return (lambda m: cse.load_fullft_snapshot(args.pretrained, ckpts[m], device)), len(ckpts)
    raise ValueError(method)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["lora", "fullft"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shot_ids", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--coco_file",
                    default=f"{ROOT}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    ap.add_argument("--root_dir", default=f"{ROOT}/data/bup_20")
    ap.add_argument("--pretrained", default="facebook/mask2former-swin-base-ade-semantic")
    ap.add_argument("--lora_dir", default=f"{ROOT}/results/lora_paper/hparam_sweep")
    ap.add_argument("--config_name", default="final_model")
    ap.add_argument("--fullft_dir", default=f"{ROOT}/results/lora_paper/full_ft")
    ap.add_argument("--out_dir",
                    default=f"{ROOT}/results/lora_paper/calibration_shift/per_class_diversity")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--max_images", type=int, default=0, help="smoke cap (0=all)")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = cse.NUM_LABELS
    names = [cse.ID2LABEL[c] for c in range(C)]
    H, W = a.height, a.width
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    get_model, M = build_members(a.method, a.seed, a, dev)
    assert M >= 2, "need >=2 members for diversity"
    print(f"=== per-class DIVERSITY {a.method} seed {a.seed} M={M} shots={a.shot_ids} "
          f"test93 @native ===", flush=True)

    loader = cse.build_loader(a.coco_file, a.root_dir, None, None,
                              a.batch_size, a.num_workers)

    # per-frame running sums over members (CPU float32; votes uint8)
    S_p, S_L, S_H, votes, gts = [], [], [], [], []
    for mi in range(M):
        model = get_model(mi)
        fi = 0
        stop = False
        for batch in tqdm(loader, desc=f"member {mi+1}/{M}"):
            pv = batch["pixel_values"].to(dev)
            seg = cse.build_seg_probs(model(pv), H, W)               # (B,C,H,W) probs>0
            for b in range(seg.shape[0]):
                if a.max_images and fi >= a.max_images:
                    stop = True
                    break
                probs = seg[b].clamp_min(EPS)                        # (C,H,W)
                probs = probs / probs.sum(0, keepdim=True)
                logp = probs.log()
                ent = -(probs * logp).sum(0)                        # (H,W)
                am = probs.argmax(0)
                oneh = F.one_hot(am, C).permute(2, 0, 1).to(torch.uint8)
                p_cpu, l_cpu = probs.float().cpu(), logp.float().cpu()
                e_cpu, o_cpu = ent.float().cpu(), oneh.cpu()
                if mi == 0:
                    S_p.append(p_cpu); S_L.append(l_cpu); S_H.append(e_cpu)
                    votes.append(o_cpu)
                    gts.append(batch["seg_maps"][b].clone())
                else:
                    S_p[fi] += p_cpu; S_L[fi] += l_cpu; S_H[fi] += e_cpu
                    votes[fi] += o_cpu
                fi += 1
            if stop:
                break
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    keys = ["disagreement", "sym_kl", "mutual_info", "expected_entropy", "predictive_entropy"]
    allk = keys + ["frac_disagree"]
    # per-class + all + fg accumulators: sum & n
    per_sum = {k: np.zeros(C) for k in allk}
    per_n = np.zeros(C)
    scope = {sc: {k + "_sum": 0.0 for k in allk} | {"n": 0.0} for sc in ("all", "fg")}

    Mf = float(M)
    tot_pairs = Mf * (Mf - 1.0) / 2.0
    for sp, sl, sh, vt, gt in zip(S_p, S_L, S_H, votes, gts):
        sp = sp.double(); sl = sl.double(); sh = sh.double(); vt = vt.double()
        mean = sp / Mf
        H_mean = -(mean * (mean + EPS).log()).sum(0)                # predictive (total)
        mean_H = sh / Mf                                            # expected (aleatoric)
        MI = (H_mean - mean_H).clamp_min(0.0)                       # epistemic / JSD
        cross = (sp * sl).sum(0)
        symkl = ((-Mf * sh - cross) / (Mf * (Mf - 1.0))).clamp_min(0.0)
        agree = 0.5 * ((vt * vt).sum(0) - Mf)
        disag = (1.0 - agree / tot_pairs).clamp(0.0, 1.0)
        maps = {"disagreement": disag, "sym_kl": symkl, "mutual_info": MI,
                "expected_entropy": mean_H, "predictive_entropy": H_mean,
                "frac_disagree": (disag > 0).double()}
        g = gt.long()
        valid = g != 255
        fg = valid & (g > 0)
        # per-GT-class reduction
        for c in range(C):
            mc = valid & (g == c)
            nc = int(mc.sum().item())
            if nc == 0:
                continue
            per_n[c] += nc
            for k in allk:
                per_sum[k][c] += float(maps[k][mc].sum().item())
        # scope reduction
        for sc, mask in (("all", valid), ("fg", fg)):
            nn = int(mask.sum().item())
            if nn == 0:
                continue
            scope[sc]["n"] += nn
            for k in allk:
                scope[sc][k + "_sum"] += float(maps[k][mask].sum().item())

    def _means_scope(d):
        n = d["n"]
        return {k: (d[k + "_sum"] / n if n else None) for k in allk}

    per_class = {}
    present = []
    for c in range(C):
        n = per_n[c]
        if n > 0:
            per_class[names[c]] = {k: per_sum[k][c] / n for k in allk}
            per_class[names[c]]["npix"] = int(n)
            present.append(c)
        else:
            per_class[names[c]] = {k: None for k in allk}
            per_class[names[c]]["npix"] = 0
    macro = {k: float(np.mean([per_sum[k][c] / per_n[c] for c in present])) for k in allk}

    result = {"method": a.method, "seed": a.seed, "M": M, "shot_ids": a.shot_ids,
              "eval": "native_1280x720_test93", "num_class": C,
              "per_class": per_class, "macro": macro,
              "all_pixels": _means_scope(scope["all"]),
              "foreground": _means_scope(scope["fg"]),
              "n_frames": len(S_p), "elapsed_s": time.time() - t0}
    tag = "_smoke" if a.max_images else ""
    outp = os.path.join(a.out_dir, f"{a.method}_seed{a.seed}_diversity{tag}.json")
    json.dump(result, open(outp, "w"), indent=2)
    fgm = result["foreground"]
    print(f"[{a.method} s{a.seed}] macro MI={macro['mutual_info']:.4f} "
          f"disag={macro['disagreement']:.4f} symKL={macro['sym_kl']:.4f} | "
          f"(fg) MI={fgm['mutual_info']:.4f} disag={fgm['disagreement']:.4f}", flush=True)
    for c in range(C):
        pc = per_class[names[c]]
        if pc["npix"]:
            print(f"    {names[c]:22s} MI={pc['mutual_info']:.4f} disag={pc['disagreement']:.4f} "
                  f"symKL={pc['sym_kl']:.4f} predH={pc['predictive_entropy']:.4f} "
                  f"npix={pc['npix']}", flush=True)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
