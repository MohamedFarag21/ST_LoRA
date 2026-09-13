# -*- coding: utf-8 -*-
"""
Turn one catalog entry (configs/<arch>.json[index]) into collision-safe trainer flags.

Emits shell KEY=VALUE lines (eval this in the job). Handles the per-architecture head/decoder
conventions so the ablation isolates attn/FFN placement:

  m2f       : LoRA on the requested roles + class_predictor (head as LoRA target, deployed setup);
              decoder full-FT (structural) via --fullft_modules transformer_module.decoder.
  segformer : LoRA on the requested roles; classifier always trained via --modules_to_save;
              decoder full-FT via --fullft_modules decode_head (covers classifier too).
  eomt      : LoRA on roles + class_predictor; upscale_block always fully trained
              (--modules_to_save, its ConvTranspose2d is peft-unsupported); mask-head full-FT
              (structural) via --fullft_modules mask_head.

Target modules are resolved to EXPLICIT full names by resolve_targets.py (loads the model once).
Run inside the arch's env (ssl for m2f/segformer, eomt for eomt).
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import resolve_targets as rt   # noqa: E402


def sh(key, val):
    if isinstance(val, (list, tuple)):
        val = " ".join(val)
    print(f"{key}='{val}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["m2f", "segformer", "eomt"])
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--catalog_dir", default=os.path.join(os.path.dirname(HERE), "configs"))
    args = ap.parse_args()

    cfgs = json.load(open(os.path.join(args.catalog_dir, f"{args.arch}.json")))
    if not (0 <= args.index < len(cfgs)):
        sys.exit(f"index {args.index} out of range 0..{len(cfgs)-1} for {args.arch}")
    e = cfgs[args.index]
    roles = e["roles"]
    mode = e["decoder_mode"]

    res = rt.resolve(args.arch, roles + ["head"])
    target = [n for r in roles for n in res[r]]
    head = res["head"]

    mts, fullft = [], []
    if args.arch == "m2f":
        target += head                                   # class_predictor as LoRA target
        if mode == "fullft":
            fullft = ["transformer_module.decoder"]
    elif args.arch == "eomt":
        target += head                                   # class_predictor as LoRA target
        mts = ["upscale_block"]                           # always fully trained
        if mode == "fullft":
            fullft = ["mask_head"]
    elif args.arch == "segformer":
        if mode == "fullft":
            fullft = ["decode_head"]                      # trains linear_c/linear_fuse/classifier
        else:
            mts = head                                    # decode_head.classifier

    # sanity: no target module should also be in a fully-trained container
    for f in fullft:
        clash = [t for t in target if f in t]
        assert not clash, f"target/fullft clash for '{f}': {clash[:3]}"
    assert target, "empty target set"

    sh("NAME", e["name"])
    sh("FAMILY", e["family"])
    sh("LORA_R", e["lora_r"])
    sh("LORA_ALPHA", e["lora_alpha"])
    sh("LORA_DROPOUT", e["lora_dropout"])
    sh("DECODER_MODE", mode)
    sh("TARGET_MODULES", target)
    sh("MODULES_TO_SAVE", mts)
    sh("FULLFT_MODULES", fullft)
    print(f"# {args.arch}[{args.index}] {e['name']}: "
          f"{len(target)} targets, mts={mts or '-'}, fullft={fullft or '-'}", file=sys.stderr)


if __name__ == "__main__":
    main()
