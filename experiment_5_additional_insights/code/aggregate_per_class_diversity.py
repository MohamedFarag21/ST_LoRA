# -*- coding: utf-8 -*-
"""
Aggregate per-class ensemble-member diversity (FRE vs ST-LoRA) across seeds -> mean±std
markdown tables, ready to paste into FINDINGS. Reads
  <dir>/<method>_seed<seed>_diversity.json  (method in {fullft, lora}).

Prints, for the two headline diversity metrics (mutual_info = epistemic, disagreement),
a per-class mean±std table with both methods side by side, plus macro / foreground /
all-pixels summary rows. No plotting deps; pure stdlib. Run via SLURM/.sh (no direct py).
"""
import os
import sys
import json
import glob
from collections import defaultdict

ROOT = ("/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/"
        "mibrahi2_hpc-my_research-1775524204")
DIR = sys.argv[1] if len(sys.argv) > 1 else \
    f"{ROOT}/results/lora_paper/calibration_shift/per_class_diversity"
METHODS = [("fullft", "FRE"), ("lora", "ST-LoRA")]
# present pepper classes in test93 (drop bg-only? keep bg; drop null classes)
CLASS_ORDER = ["bg", "pepper red", "pepper yellow", "pepper green",
               "pepper mixed_red", "pepper mixed_yellow"]
METRICS = [("mutual_info", "MI (epistemic)"), ("disagreement", "argmax disagree"),
           ("sym_kl", "sym-KL"), ("predictive_entropy", "pred entropy"),
           ("expected_entropy", "exp entropy"), ("frac_disagree", "frac disagree")]


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, v ** 0.5


def load(method):
    """seed -> record."""
    out = {}
    for f in sorted(glob.glob(os.path.join(DIR, f"{method}_seed*_diversity.json"))):
        d = json.load(open(f))
        out[d["seed"]] = d
    return out


def main():
    data = {m: load(m) for m, _ in METHODS}
    seeds = sorted(set().union(*[set(v.keys()) for v in data.values()]))
    print(f"# Per-class ensemble diversity — FRE vs ST-LoRA (pepper test93 @native)\n")
    m0 = data["fullft"].get(seeds[0]) if seeds else None
    M = m0["M"] if m0 else "?"
    shots = m0["shot_ids"] if m0 else "?"
    print(f"seeds={seeds}  M={M} (shots={shots})  n_frames={m0['n_frames'] if m0 else '?'}\n")

    # collect per-class per-metric across seeds
    for metric, mlabel in METRICS:
        print(f"\n## {mlabel}  (mean±std over {len(seeds)} seeds)\n")
        print(f"| class | FRE | ST-LoRA | Δ(LoRA−FRE) |")
        print(f"|---|---|---|---|")
        for cls in CLASS_ORDER:
            row = [cls]
            means = {}
            for meth, _ in METHODS:
                xs = []
                for s in seeds:
                    rec = data[meth].get(s)
                    if rec and rec["per_class"].get(cls) is not None:
                        xs.append(rec["per_class"][cls].get(metric))
                mu, sd = _stats(xs)
                means[meth] = mu
                row.append("--" if mu is None else f"{mu:.4f}±{sd:.4f}")
            if means["fullft"] is not None and means["lora"] is not None:
                row.append(f"{means['lora'] - means['fullft']:+.4f}")
            else:
                row.append("--")
            print("| " + " | ".join(row) + " |")
        # summary scopes
        for scope, slab in (("macro", "macro"), ("foreground", "foreground"),
                            ("all_pixels", "all-pixels")):
            row = [f"**{slab}**"]
            mv = {}
            for meth, _ in METHODS:
                xs = [data[meth][s][scope].get(metric) for s in seeds
                      if s in data[meth] and scope in data[meth][s]]
                mu, sd = _stats(xs)
                mv[meth] = mu
                row.append("--" if mu is None else f"**{mu:.4f}±{sd:.4f}**")
            if mv["fullft"] is not None and mv["lora"] is not None:
                row.append(f"**{mv['lora'] - mv['fullft']:+.4f}**")
            else:
                row.append("--")
            print("| " + " | ".join(row) + " |")
    print("\n(Δ>0 ⇒ ST-LoRA more diverse on that metric/class.)")


if __name__ == "__main__":
    main()
