# -*- coding: utf-8 -*-
"""
Collision-safe LoRA target resolver for the ST-LoRA hyperparameter study (Experiment 2).

Bare PEFT suffixes are ambiguous in these models (M2F `dense` matches 72 modules; `fc1/fc2`
match both pixel-decoder and transformer-decoder; EoMT `fc1/fc2` match both encoder MLP and the
mask head). This module loads a model, classifies every nn.Linear / (SegFormer) Conv2d leaf into
one semantic ROLE, and returns the EXPLICIT full module names for the requested roles — which PEFT
matches unambiguously (a full name ends-with-matches exactly one module).

Roles (per architecture):
  m2f       : enc_attn, enc_ffn, dec_attn, dec_ffn, head
  segformer : enc_attn, enc_ffn, dec_ffn, head        (decode_head is all-MLP -> no dec_attn)
  eomt      : enc_attn, enc_ffn, mask_ffn, head        (no decoder; mask_ffn = mask_head)

Convention (matches the study): "attention" = q/k/v projections only (not the output projection),
per the Experiment-2 baseline "attention query, key, value".

Usage:
  # print space-joined explicit target list for a role set (feed to --target_modules):
  python resolve_targets.py --arch m2f --roles enc_attn enc_ffn dec_attn dec_ffn
  # dry-run asserter: build the peft model, print module counts + trainable %, assert non-empty:
  python resolve_targets.py --arch eomt --roles enc_attn enc_ffn --assert
  # list the head module (feed to --modules_to_save):
  python resolve_targets.py --arch segformer --roles head --names_only
"""
import re
import sys
import argparse
import torch.nn as nn

MODEL_IDS = {
    "m2f":       ("Mask2FormerForUniversalSegmentation", "facebook/mask2former-swin-base-ade-semantic"),
    "segformer": ("SegformerForSemanticSegmentation",    "nvidia/segformer-b2-finetuned-ade-512-512"),
    "eomt":      ("EomtForUniversalSegmentation",        "tue-mps/ade20k_semantic_eomt_large_512"),
}

# role -> compiled regex on the FULL module name (anchored at end)
ROLE_PATTERNS = {
    "m2f": {
        "enc_attn": r"pixel_level_module\.encoder\..*\.attention\.self\.(query|key|value)$",
        # Swin block FFN = intermediate.dense (in) + block output.dense (out); EXCLUDE attention.output.dense
        "enc_ffn":  r"pixel_level_module\.encoder\.(?:(?!attention\.output).)*\.(intermediate\.dense|output\.dense)$",
        "dec_attn": r"transformer_module\.decoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)$",
        "dec_ffn":  r"transformer_module\.decoder\.layers\.\d+\.(fc1|fc2)$",
        "head":     r"(^|\.)class_predictor$",
    },
    "segformer": {
        "enc_attn": r"segformer\.encoder\..*\.attention\.self\.(query|key|value)$",
        "enc_ffn":  r"segformer\.encoder\..*\.mlp\.(dense1|dense2)$",
        "dec_ffn":  r"decode_head\.(linear_c\.\d+\.proj|linear_fuse)$",
        "head":     r"decode_head\.classifier$",
    },
    "eomt": {
        "enc_attn": r"layers\.\d+\.attention\.(q_proj|k_proj|v_proj)$",
        "enc_ffn":  r"layers\.\d+\.mlp\.(fc1|fc2)$",
        "mask_ffn": r"mask_head\.(fc1|fc2|fc3)$",
        "head":     r"(^|\.)class_predictor$",
    },
}
# SegFormer decode_head uses Conv2d for linear_fuse/classifier; include Conv2d leaves for those roles.
LORA_LAYER_TYPES = (nn.Linear, nn.Conv2d)


def load_model(arch):
    import transformers
    cls_name, mid = MODEL_IDS[arch]
    cls = getattr(transformers, cls_name)
    kw = dict(ignore_mismatched_sizes=True)
    if arch == "segformer":
        kw.update(num_labels=2)
    return cls.from_pretrained(mid, **kw)


def resolve(arch, roles):
    """Return dict role -> sorted list of explicit full module names present in the model."""
    model = load_model(arch)
    pats = {r: re.compile(ROLE_PATTERNS[arch][r]) for r in roles}
    out = {r: [] for r in roles}
    for name, mod in model.named_modules():
        if not isinstance(mod, LORA_LAYER_TYPES):
            continue
        for r, pat in pats.items():
            if pat.search(name):
                out[r].append(name)
    for r in out:
        out[r] = sorted(out[r])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(MODEL_IDS))
    ap.add_argument("--roles", nargs="+", required=True)
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    help="build the peft model and print module counts + trainable %%")
    ap.add_argument("--names_only", action="store_true",
                    help="print names one-per-line instead of space-joined")
    args = ap.parse_args()

    for r in args.roles:
        if r not in ROLE_PATTERNS[args.arch]:
            sys.exit(f"role '{r}' invalid for arch {args.arch}; "
                     f"valid: {sorted(ROLE_PATTERNS[args.arch])}")

    res = resolve(args.arch, args.roles)
    names = sorted(set(n for r in args.roles for n in res[r]))

    if args.do_assert:
        from peft import LoraConfig, get_peft_model
        print(f"[resolve] arch={args.arch} roles={args.roles}", file=sys.stderr)
        for r in args.roles:
            print(f"  role {r:<9}: {len(res[r])} modules", file=sys.stderr)
        assert names, "no modules matched — target set is EMPTY"
        model = load_model(args.arch)
        cfg = LoraConfig(r=8, lora_alpha=8, target_modules=names, lora_dropout=0.1, bias="none")
        pm = get_peft_model(model, cfg)
        tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in pm.parameters())
        print(f"  TOTAL matched modules: {len(names)}", file=sys.stderr)
        print(f"  trainable {tr:,} / {tot:,} ({100*tr/tot:.3f}%)", file=sys.stderr)
        print("  ASSERT OK (non-empty, adapters attach)", file=sys.stderr)

    if args.names_only:
        print("\n".join(names))
    else:
        print(" ".join(names))


if __name__ == "__main__":
    main()
