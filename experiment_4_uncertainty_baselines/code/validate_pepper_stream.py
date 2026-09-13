# -*- coding: utf-8 -*-
"""
Correctness proofs for generalizing the two streaming estimators from the binary
tomato study to 8-class pepper. SYNTHETIC data, CPU-only, no dataset/model reads.

  PART A  REGRESSION: StreamBinMetrics(num_classes=2) mIoU — the new vectorized
          bincount confusion vs the ORIGINAL `for c in (0,1)` loop. Must be EXACT,
          or today's published tomato mIoU numbers silently change.

  PART B  StreamBinMetrics(num_classes=8) mIoU vs an independent numpy reference
          that uses nanmean-over-present-classes (evaluate/mean_iou semantics).
          Includes the case that motivated the change: a class absent from the
          eval set (rare pepper subtype), which must be SKIPPED, not scored 0.

  PART C  OODHist(smax=ln 8) vs sklearn on scores spanning [0, ln8] — proves the
          bound generalization bins correctly and doesn't clip.

  PART D  REGRESSION: OODHist() default is still exactly smax=ln2 binary behaviour.

Run via SLURM (ssl env, CPU partition). No direct python on login nodes.
"""
import os
import sys
import math
import numpy as np
import torch

ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")
sys.path.insert(0, f"{ROOT}/code/bup_20_trials/elora")

from sklearn.metrics import roc_auc_score, average_precision_score   # noqa: E402
from ood_eval_comprehensive import compute_fpr95                     # noqa: E402
from run_seed_ood_stream import OODHist, LN2                         # noqa: E402
from run_seed_calibration_stream import StreamBinMetrics             # noqa: E402

LN8 = math.log(8.0)


def old_miou_loop(preds, labs, num_classes=2):
    """The ORIGINAL implementation, verbatim: `for c in (0,1)` + clamp(min=1).mean()."""
    inter = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    for pred, lab in zip(preds, labs):
        valid = lab != 255
        pred, lab = pred[valid], lab[valid]
        for c in range(num_classes):
            pc, lc = (pred == c), (lab == c)
            inter[c] += (pc & lc).sum().item()
            union[c] += (pc | lc).sum().item()
    return float((inter / union.clamp(min=1)).mean())


def ref_miou_numpy(preds, labs, num_classes):
    """Independent numpy reference: dataset-pooled IoU, nanmean over PRESENT classes."""
    inter = np.zeros(num_classes, np.float64)
    union = np.zeros(num_classes, np.float64)
    for pred, lab in zip(preds, labs):
        p = pred.numpy().ravel(); l = lab.numpy().ravel()
        m = l != 255
        p, l = p[m], l[m]
        for c in range(num_classes):
            inter[c] += np.sum((p == c) & (l == c))
            union[c] += np.sum((p == c) | (l == c))
    present = union > 0
    return float(np.mean(inter[present] / union[present])) if present.any() else float("nan")


def fake_probs(pred, num_classes, rng):
    """Build (1,C,H,W) probs whose argmax == pred (confidence irrelevant to mIoU)."""
    H, W = pred.shape
    p = torch.from_numpy(rng.uniform(0.0, 0.2, (1, num_classes, H, W)))
    p.scatter_(1, pred[None, None].long(), 0.9)
    return (p / p.sum(1, keepdim=True)).float()


def part_a(n_frames=40, seed=0):
    print("=== PART A: REGRESSION StreamBinMetrics(num_classes=2) mIoU ===", flush=True)
    rng = np.random.default_rng(seed)
    preds, labs = [], []
    for _ in range(n_frames):
        H, W = 48, 64
        preds.append(torch.from_numpy(rng.integers(0, 2, (H, W))).long())
        lab = rng.integers(0, 2, (H, W))
        lab[rng.random((H, W)) < 0.05] = 255                 # ignore pixels
        labs.append(torch.from_numpy(lab).long())
    m = StreamBinMetrics(num_classes=2)
    for pred, lab in zip(preds, labs):
        m.update(fake_probs(pred, 2, rng), lab[None])
    new = m.compute()["mIoU"]
    old = old_miou_loop(preds, labs, 2)
    d = abs(new - old)
    print(f"  new(bincount)={new:.12f}  old(loop)={old:.12f}  |delta|={d:.3e}", flush=True)
    ok = d < 1e-12
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}  (binary tomato mIoU must be unchanged)",
          flush=True)
    return ok


def part_b(n_frames=40, seed=1):
    print("\n=== PART B: StreamBinMetrics(num_classes=8) mIoU vs numpy ref ===", flush=True)
    rng = np.random.default_rng(seed)
    allok = True
    for case, hi in (("all 8 classes present", 8), ("classes 6,7 ABSENT (rare subtype)", 6)):
        preds, labs = [], []
        for _ in range(n_frames):
            H, W = 48, 64
            preds.append(torch.from_numpy(rng.integers(0, hi, (H, W))).long())
            lab = rng.integers(0, hi, (H, W))
            lab[rng.random((H, W)) < 0.05] = 255
            labs.append(torch.from_numpy(lab).long())
        m = StreamBinMetrics(num_classes=8)
        for pred, lab in zip(preds, labs):
            m.update(fake_probs(pred, 8, rng), lab[None])
        out = m.compute()
        ref = ref_miou_numpy(preds, labs, 8)
        d = abs(out["mIoU"] - ref)
        ok = d < 1e-12
        allok &= ok
        print(f"  [{case:34s}] stream={out['mIoU']:.12f} ref={ref:.12f} "
              f"|delta|={d:.3e} present={out['n_classes_present']}/8  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    print(f"  RESULT: {'PASS' if allok else 'FAIL'}", flush=True)
    return allok


def _hist_vs_sklearn(smax, label, seed):
    rng = np.random.default_rng(seed)
    ok_all = True
    for name, mu_n, mu_a, sd in (("separated", 0.15, 0.75, 0.05),
                                 ("overlapping", 0.45, 0.55, 0.12),
                                 ("near-random", 0.50, 0.51, 0.15)):
        n_norm, n_anom = 3_000_000, 150_000
        s_norm = np.clip(rng.normal(mu_n * smax, sd * smax, n_norm), 0, smax)
        s_anom = np.clip(rng.normal(mu_a * smax, sd * smax, n_anom), 0, smax)
        scores = np.concatenate([s_norm, s_anom]).astype(np.float64)
        labels = np.concatenate([np.zeros(n_norm, np.uint8), np.ones(n_anom, np.uint8)])
        sk = {"AUROC": float(roc_auc_score(labels, scores)),
              "AUPR": float(average_precision_score(labels, scores)),
              "FPR95": float(compute_fpr95(labels, scores))}
        h = OODHist(smax=smax)
        for i in range(0, scores.size, 500_000):
            h.update(torch.from_numpy(scores[i:i + 500_000]),
                     torch.from_numpy(labels[i:i + 500_000].astype(np.int64)))
        hm = h.metrics()
        d = {k: abs(sk[k] - hm[k]) for k in sk}
        ok = all(v < 1e-3 for v in d.values())
        ok_all &= ok
        print(f"  [{label} {name:12s}] sk AUROC={sk['AUROC']:.6f} AUPR={sk['AUPR']:.6f} "
              f"FPR95={sk['FPR95']:.6f}", flush=True)
        print(f"  {'':>14s}{'':12s}  hi AUROC={hm['AUROC']:.6f} AUPR={hm['AUPR']:.6f} "
              f"FPR95={hm['FPR95']:.6f}  max|d|={max(d.values()):.2e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok_all


def part_c(seed=2):
    print(f"\n=== PART C: OODHist(smax=ln8={LN8:.4f}) vs sklearn (8-class range) ===",
          flush=True)
    ok = _hist_vs_sklearn(LN8, "ln8", seed)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def part_d(seed=3):
    print(f"\n=== PART D: REGRESSION OODHist() default smax==ln2 ({LN2:.4f}) ===",
          flush=True)
    assert OODHist().smax == LN2, f"default smax changed! {OODHist().smax} != {LN2}"
    ok = _hist_vs_sklearn(LN2, "ln2", seed)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}  (binary tomato OoD must be unchanged)",
          flush=True)
    return ok


if __name__ == "__main__":
    r = [part_a(), part_b(), part_c(), part_d()]
    print(f"\n=== OVERALL: {'PASS' if all(r) else 'FAIL'} ===", flush=True)
    sys.exit(0 if all(r) else 1)
