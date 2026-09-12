# Experiment 2 — ST-LoRA hyperparameter ablation

How ST-LoRA's **rank**, **scaling factor α**, **dropout**, and **target modules** affect a single
fine-tuned model, across **Mask2Former**, **SegFormer-B2**, and **EoMT**. Dataset: **BUP20**
(sweet pepper). **All runs use `--no_augmentation`.** Single model (last snapshot, no ensembling).
We report **mIoU + all calibration metrics** (ECE / MECE / ACE / MACE / Brier / NLL).

**Baseline** (fixed point of every 1-D sweep): `r = α = 8`, `dropout = 0.1`, LoRA on
**attention q/k/v + the encoder & decoder FFN** (for EoMT, which has no decoder, the "decoder FFN"
is the mask head that produces the masks).

## Ablations (per architecture)

| Family | Grid | Fixed |
|---|---|---|
| **rank** | r ∈ {2,4,8,16,32,64,128,256} | α=8, dropout=0.1, all components |
| **alpha** | α ∈ {1,2,4,8,16,32,64} | r=8, dropout=0.1, all components |
| **dropout** | {0.05,0.10,0.15,0.20,0.25,0.30} | r=α=8, all components |
| **modules** | all non-empty subsets of the components | r=α=8, dropout=0.1 |
| **structural** | encoder-only (decoder frozen); encoder-LoRA + decoder full-FT; encoder+decoder LoRA | r=α=8, dropout=0.1 |
| **efficiency** | FFN-only vs FFN+attn (trainable params; CO₂/energy via Exp-1 tooling) | r=8 |

**Components** (the ablation isolates attn/FFN placement; the task head is handled *constantly*):
- M2F: `enc_attn, enc_ffn, dec_attn, dec_ffn` → 15 module subsets, **35 configs**
- SegFormer-B2: `enc_attn, enc_ffn, dec_ffn` (decode-head is all-MLP → no dec-attn) → 7 subsets, **27 configs**
- EoMT: `enc_attn, enc_ffn, mask_ffn` (no decoder; mask_ffn = mask head) → 7 subsets, **27 configs**

## How targets are resolved (collision-safe)

Bare PEFT suffixes are ambiguous in these models (`dense` matches 72 M2F modules; `fc1/fc2` match
both encoder MLP and mask/pixel decoders). `code/resolve_targets.py` loads the model and returns the
**explicit full module names** for each semantic role, validated by a dry-run asserter
(`--assert`). `code/prepare_run.py` turns a catalog entry into the exact trainer flags
(`--target_modules`, `--modules_to_save`, `--fullft_modules`) with per-architecture head handling:
- M2F: `class_predictor` is a LoRA target; decoder full-FT via `--fullft_modules transformer_module.decoder`.
- SegFormer: `classifier` always trained (`--modules_to_save`); decoder full-FT via `--fullft_modules decode_head`.
- EoMT: `class_predictor` a LoRA target; `upscale_block` always fully trained; mask-head full-FT via `--fullft_modules mask_head`.

## Contents

```
experiment_2_hyperparameters/
├── env.sh                 # set ST_LORA_ROOT + BUP20_DIR
├── configs/               # generated catalog: m2f.json, segformer.json, eomt.json, _meta.json
├── code/
│   ├── build_catalog.py       # (re)generates configs/*.json
│   ├── resolve_targets.py     # collision-safe role -> explicit module names (+ --assert)
│   ├── prepare_run.py         # catalog entry -> trainer flags
│   ├── *_lora_train_seeded*.py  # the 3 trainers (patched: --fullft_modules)
│   └── *_eval*.py             # single-model evaluation (mIoU + calibration)
└── jobs/
    ├── run_ablation.sh        # array runner: one task = one config (per arch, per seed)
    └── efficiency_ffn_vs_attn.sh
```

## Setup

```bash
conda env create -f ../environment.yml       # st_lora     (M2F + SegFormer)
conda env create -f ../environment_eomt.yml   # st_lora_eomt (EoMT)
$EDITOR env.sh                                # set BUP20_DIR
source env.sh
# regenerate the catalog if you change the grids (pure stdlib):
python code/build_catalog.py
```

## Run — SLURM

One array task = one catalog config, for a fixed seed. Array upper bound = (#configs − 1):
M2F 34, SegFormer/EoMT 26. Loop seeds `{42,123,456,789,1337}`:

```bash
source env.sh
for S in 42 123 456 789 1337; do
  ARCH=m2f       SEED=$S sbatch --array=0-34 jobs/run_ablation.sh
  ARCH=segformer SEED=$S sbatch --array=0-26 jobs/run_ablation.sh
  ARCH=eomt      SEED=$S sbatch --array=0-26 jobs/run_ablation.sh
done
# efficiency (trainable params, FFN-only vs FFN+attn):
sbatch jobs/efficiency_ffn_vs_attn.sh
```
Each task trains (110 ep, cosine T0=20, snapshots every 20, **no augmentation**) then evaluates the
single last snapshot (shot 5). Results land in `results/exp2/<arch>/<config>/seed_<S>/`.

## Run — no SLURM (single GPU)

```bash
source env.sh; conda activate "$ENV_MAIN"           # or "$ENV_EOMT" for eomt
ARCH=m2f; SEED=42; IDX=0                              # pick a config index (see configs/m2f.json)
eval "$(PYTHONPATH=code python code/prepare_run.py --arch $ARCH --index $IDX)"
PYTHONPATH=code python code/mask2former_lora_train_seeded_aug.py \
  --seed $SEED --no_augmentation --epochs 110 --cosine_T0 20 --snapshot_every 20 \
  --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT \
  --target_modules $TARGET_MODULES ${FULLFT_MODULES:+--fullft_modules $FULLFT_MODULES} \
  --precision 32 --no_dora --gpus 1 --num_workers 8 --batch_size 4 \
  --coco_file "$BUP20_COCO" --root_dir "$BUP20_DIR" \
  --base_save_dir results/exp2/$ARCH/$NAME --log_dir results/exp2/$ARCH/$NAME/logs
```
(SegFormer/EoMT differ only in script name, env, LR/precision, and the `--modules_to_save` flag —
see `jobs/run_ablation.sh` for the exact per-arch commands.)

## Efficiency: FFN-only vs FFN+attn

`jobs/efficiency_ffn_vs_attn.sh` prints trainable-parameter counts (the deterministic proxy) for
the FFN-only vs FFN+attn target sets on all three architectures. For **wall-clock / GPU-energy /
CO₂**, run Experiment 1's `co2_compare_train.py` family with the same two target sets (resolve them
with `code/resolve_targets.py`).

## Notes

- **No augmentation** on every Experiment-2 run, by design.
- **Seeds** `{42,123,456,789,1337}`; report mean ± std.
- **Structural semantics:** *encoder-only* = LoRA on encoder attn+FFN, decoder frozen (head still
  trained); *decoder full-FT* = encoder LoRA + decoder fully fine-tuned; *encoder+decoder* = the baseline.
- Targets are always the **explicit full module names** from `resolve_targets.py` — never bare
  suffixes — so the FFN/attn/encoder/decoder partition is collision-safe.
- Pre-flight tip: run one config as a smoke (e.g. `--array=0-0`, one seed) before launching the grid.
