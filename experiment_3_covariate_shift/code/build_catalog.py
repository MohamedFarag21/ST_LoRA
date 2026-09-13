# -*- coding: utf-8 -*-
"""
Generate the Experiment-2 config catalog (one JSON per architecture).

Each entry is one ST-LoRA run: rank / alpha / dropout / ablated component roles / decoder mode.
Target modules are NOT stored here — they are resolved collision-safely at run time by
resolve_targets.py from the `roles` list. The task head is handled CONSTANTLY across all configs
so the ablation isolates attn/FFN placement:
  m2f       : class_predictor is always a LoRA target
  segformer : classifier is always modules_to_save
  eomt      : class_predictor always a LoRA target; upscale_block always modules_to_save

Baseline: r=alpha=8, dropout=0.1, all components (attn q/k/v + FFN of encoder and decoder).
Sweeps (one axis varies, rest at baseline): rank {2..256}, alpha {1..64}, dropout {0.05..0.30}.
Module subsets: all non-empty subsets of the architecture's components.
Structural: encoder-only (decoder frozen), encoder-LoRA + decoder full-FT, encoder+decoder LoRA.

Pure stdlib (no torch); emits configs/<arch>.json. Run once to (re)generate the catalog.
"""
import os
import json
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "configs")

COMPONENTS = {
    "m2f":       ["enc_attn", "enc_ffn", "dec_attn", "dec_ffn"],
    "segformer": ["enc_attn", "enc_ffn", "dec_ffn"],
    "eomt":      ["enc_attn", "enc_ffn", "mask_ffn"],
}
ENC_ROLES = {"m2f": ["enc_attn", "enc_ffn"], "segformer": ["enc_attn", "enc_ffn"],
             "eomt": ["enc_attn", "enc_ffn"]}
# decoder container module (for decoder full-FT mode) and per-arch head handling
DECODER_CONTAINER = {"m2f": "transformer_module.decoder", "segformer": "decode_head",
                     "eomt": "mask_head"}
HEAD_TARGET = {"m2f": ["class_predictor"], "segformer": [], "eomt": ["class_predictor"]}
HEAD_SAVE   = {"m2f": [], "segformer": ["classifier"], "eomt": ["upscale_block"]}

RANKS = [2, 4, 8, 16, 32, 64, 128, 256]
ALPHAS = [1, 2, 4, 8, 16, 32, 64]
DROPOUTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BASE_R, BASE_A, BASE_DO = 8, 8, 0.10


def entry(name, family, roles, r=BASE_R, a=BASE_A, do=BASE_DO, decoder_mode="lora"):
    return {"name": name, "family": family, "roles": list(roles),
            "lora_r": r, "lora_alpha": a, "lora_dropout": round(do, 2),
            "decoder_mode": decoder_mode}


def build(arch):
    comps = COMPONENTS[arch]
    enc = ENC_ROLES[arch]
    cfgs = []
    sig_seen = {}   # dedup identical runs; record all families an entry satisfies

    def add(e):
        sig = (tuple(sorted(e["roles"])), e["lora_r"], e["lora_alpha"],
               e["lora_dropout"], e["decoder_mode"])
        if sig in sig_seen:
            fam = cfgs[sig_seen[sig]]["family"]
            if e["family"] not in fam.split("+"):
                cfgs[sig_seen[sig]]["family"] = fam + "+" + e["family"]
            return
        sig_seen[sig] = len(cfgs)
        cfgs.append(e)

    add(entry("baseline", "baseline", comps))
    for r in RANKS:
        add(entry(f"rank_r{r}", "rank", comps, r=r, a=BASE_A))
    for a in ALPHAS:
        add(entry(f"alpha_a{a}", "alpha", comps, r=BASE_R, a=a))
    for do in DROPOUTS:
        add(entry(f"drop_{int(round(do*100)):02d}", "dropout", comps, do=do))
    # all non-empty subsets of components
    for k in range(1, len(comps) + 1):
        for sub in itertools.combinations(comps, k):
            tag = "+".join(s.replace("_", "") for s in sub)
            add(entry(f"mod_{tag}", "modules", list(sub)))
    # structural
    add(entry("struct_encoder_only", "structural", enc, decoder_mode="frozen"))
    add(entry("struct_enc_lora_dec_fullft", "structural", enc, decoder_mode="fullft"))
    add(entry("struct_enc_dec_lora", "structural", comps, decoder_mode="lora"))
    return cfgs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {"components": COMPONENTS, "decoder_container": DECODER_CONTAINER,
            "head_target": HEAD_TARGET, "head_save": HEAD_SAVE,
            "ranks": RANKS, "alphas": ALPHAS, "dropouts": DROPOUTS,
            "baseline": {"r": BASE_R, "alpha": BASE_A, "dropout": BASE_DO},
            "dataset": "BUP20"}
    json.dump(meta, open(os.path.join(OUT_DIR, "_meta.json"), "w"), indent=2)
    for arch in COMPONENTS:
        cfgs = build(arch)
        json.dump(cfgs, open(os.path.join(OUT_DIR, f"{arch}.json"), "w"), indent=2)
        by_fam = {}
        for c in cfgs:
            for f in c["family"].split("+"):
                by_fam[f] = by_fam.get(f, 0) + 1
        print(f"[{arch}] {len(cfgs)} unique configs  " +
              "  ".join(f"{k}:{v}" for k, v in sorted(by_fam.items())))


if __name__ == "__main__":
    main()
