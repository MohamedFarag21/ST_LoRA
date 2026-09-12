# -*- coding: utf-8 -*-
"""
Evaluate a trained EoMT snapshot on bup20 sweet-pepper test93 (native 1280x720):
  * overall mIoU (dataset-pooled, macro over PRESENT classes)
  * per-class IoU (all 8 classes)
  * calibration: ECE / ACE (float64, matches StreamBinMetrics) + Brier / NLL (float64,
    unbinned proper scoring rules) on the UNCALIBRATED softmax-normalised semantic probs.

Per-pixel 8-class probabilities come from post_process_semantic_segmentation(
return_segmentation_scores=True) -> (8,H,W) native score maps, then L1-normalised
(== softmax(log(score)) ), reproducing the pepper Mask2Former study's build_seg_probs
convention so the numbers are directly comparable to that study.

Runs in the ISOLATED `eomt` env (SLURM only). Writes seed_<S>/eval_metrics.json.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eomt_full_train_seeded import PepperEomtDataModule, IGNORE_INDEX   # noqa: E402
from mask2former_full_train_seeded import ID2LABEL, NUM_LABELS          # noqa: E402
from transformers import EomtForUniversalSegmentation                   # noqa: E402


class StreamMetrics:
    """Self-contained float64 streaming metrics (mirrors StreamBinMetrics), plus
    per-class IoU exposure. probs (1,C,H,W) softmax; labels (1,H,W) long, ignore=255."""

    def __init__(self, num_classes=NUM_LABELS, n_ece=10, n_fine=4096, eps=1e-12):
        self.C, self.n_ece, self.n_fine, self.eps = num_classes, n_ece, n_fine, eps
        z = lambda n: torch.zeros(n, dtype=torch.float64)
        self.e_cnt, self.e_conf, self.e_corr = z(n_ece), z(n_ece), z(n_ece)
        self.f_cnt, self.f_conf, self.f_corr = z(n_fine), z(n_fine), z(n_fine)
        self.inter, self.union = z(self.C), z(self.C)
        self.correct, self.total = 0.0, 0.0
        self.sq_sum, self.pgt_sum, self.nll_sum, self.npix = 0.0, 0.0, 0.0, 0

    @torch.no_grad()
    def update(self, probs, labels):
        valid = labels != 255
        conf, pred = probs.max(1)
        conf, pred, lab = conf[valid], pred[valid], labels[valid]
        corr = (pred == lab).double()
        self.correct += corr.sum().item(); self.total += corr.numel()
        for cnt, cf, cr, nb in ((self.e_cnt, self.e_conf, self.e_corr, self.n_ece),
                                (self.f_cnt, self.f_conf, self.f_corr, self.n_fine)):
            bi = torch.clamp((conf * nb).long(), 0, nb - 1)
            cnt += torch.bincount(bi, minlength=nb).double().cpu()
            cf  += torch.bincount(bi, weights=conf.double(), minlength=nb).cpu()
            cr  += torch.bincount(bi, weights=corr, minlength=nb).cpu()
        C = self.C
        eq = pred == lab
        self.inter += torch.bincount(pred[eq], minlength=C).double().cpu()
        self.union += (torch.bincount(pred, minlength=C)
                       + torch.bincount(lab, minlength=C)
                       - torch.bincount(pred[eq], minlength=C)).double().cpu()
        pv = probs.permute(0, 2, 3, 1)[valid].double()
        pgt = pv.gather(1, lab.long().unsqueeze(1)).squeeze(1).clamp_min(self.eps)
        self.sq_sum += (pv * pv).sum().item()
        self.pgt_sum += pgt.sum().item()
        self.nll_sum += (-pgt.log()).sum().item()
        self.npix += pv.shape[0]

    def _binned_ce(self, cnt, conf, corr):
        N = cnt.sum()
        if N == 0:
            return float("nan")
        nz = cnt > 0
        acc = torch.zeros_like(cnt); cf = torch.zeros_like(cnt)
        acc[nz] = corr[nz] / cnt[nz]; cf[nz] = conf[nz] / cnt[nz]
        return ((cnt / N) * (acc - cf).abs()).sum().item()

    def _ace(self, nbins=10):
        N = self.f_cnt.sum()
        if N == 0:
            return float("nan")
        cum = torch.cumsum(self.f_cnt, 0)
        edges = [(i + 1) * N / nbins for i in range(nbins)]
        gc, gf, gr = (torch.zeros(nbins, dtype=torch.float64) for _ in range(3))
        g = 0
        for k in range(self.n_fine):
            gc[g] += self.f_cnt[k]; gf[g] += self.f_conf[k]; gr[g] += self.f_corr[k]
            if cum[k] >= edges[g] and g < nbins - 1:
                g += 1
        return self._binned_ce(gc, gf, gr)

    def compute(self):
        present = self.union > 0
        per_class = {}
        for c in range(self.C):
            per_class[ID2LABEL[c]] = (float(self.inter[c] / self.union[c])
                                      if self.union[c] > 0 else None)
        miou = (float((self.inter[present] / self.union[present]).mean())
                if bool(present.any()) else float("nan"))
        npx = max(self.npix, 1)
        return {"mIoU": miou,
                "per_class_IoU": per_class,
                "n_classes_present": int(present.sum()),
                "ECE": self._binned_ce(self.e_cnt, self.e_conf, self.e_corr),
                "ACE": self._ace(10),
                "Brier": (self.sq_sum - 2.0 * self.pgt_sum + self.npix) / npx,
                "NLL": self.nll_sum / npx,
                "acc": self.correct / max(self.total, 1)}


@torch.no_grad()
def main():
    R = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
         "mibrahi2_hpc-my_research-1775524204")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--snapshot", default=None,
                    help="path to model_shot_*.pt; default = <base>/seed_<S>/<shot>.pt")
    ap.add_argument("--base_save_dir", default=f"{R}/results/lora_paper/eomt_full")
    ap.add_argument("--shot", default="model_shot_5")
    ap.add_argument("--ckpt", default="tue-mps/ade20k_semantic_eomt_large_512")
    ap.add_argument("--coco_file", default=f"{R}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json")
    ap.add_argument("--root_dir", default=f"{R}/data/bup_20")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()

    snap = args.snapshot or os.path.join(args.base_save_dir, f"seed_{args.seed}",
                                         f"{args.shot}.pt")
    out_dir = args.out_dir or os.path.join(args.base_save_dir, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    dm = PepperEomtDataModule(coco_file=args.coco_file, root_dir=args.root_dir,
                              ckpt=args.ckpt, batch_size=1, num_workers=4,
                              use_augmentation=False)
    dm.setup()
    loader = dm.val_dataloader() if args.split == "valid" else dm.test_dataloader()

    model = EomtForUniversalSegmentation.from_pretrained(
        args.ckpt, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    sd = torch.load(snap, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {snap}\n  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(dev).eval()

    met = StreamMetrics(num_classes=NUM_LABELS)
    for batch in tqdm(loader, desc=f"eval {args.split}"):
        out = model(pixel_values=batch["pixel_values"].to(dev),
                    patch_offsets=batch.get("patch_offsets"))
        results = dm.processor.post_process_semantic_segmentation(
            out, target_sizes=batch["target_sizes"], return_segmentation_scores=True)
        for res, gt in zip(results, batch["original_segmentation_maps"]):
            scores = res.segmentation_scores.to(dev).clamp_min(0)      # (C,H,W) >= 0
            probs = (scores + met.eps) / (scores.sum(0, keepdim=True) + met.C * met.eps)
            met.update(probs.unsqueeze(0), gt.unsqueeze(0).to(dev))

    m = met.compute()
    m.update({"seed": args.seed, "shot": args.shot, "split": args.split,
              "snapshot": snap, "elapsed_s": time.time() - t0})
    outp = os.path.join(out_dir, f"eval_metrics_{args.split}.json")
    json.dump(m, open(outp, "w"), indent=2)
    print(f"\n[seed {args.seed}] mIoU={m['mIoU']:.4f} ECE={m['ECE']:.4f} "
          f"Brier={m['Brier']:.4f} NLL={m['NLL']:.4f}", flush=True)
    print("  per-class IoU:", {k: (round(v, 3) if v is not None else None)
                               for k, v in m["per_class_IoU"].items()}, flush=True)
    print(f"[done] wrote {outp} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
