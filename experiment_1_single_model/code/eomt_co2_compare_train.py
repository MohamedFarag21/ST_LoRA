# -*- coding: utf-8 -*-
"""
EoMT analog of co2_compare_train.py — measure CO2/energy of ONE EoMT training run
(FRE full-FT vs ST-LoRA r8) with CodeCarbon, WITHOUT modifying the EoMT trainers.

Mirrors the Mask2Former CO2 protocol exactly (OfflineEmissionsTracker country=DEU +
background pynvml per-epoch power trace), reusing the generic instrumentation helpers
from co2_compare_train.py. Only build_argv + the wrapped trainer module differ.

Both methods run NO-augmentation, batch_size 2, 1 GPU, same cadence — so energies are
directly comparable. FRE uses the deployed EoMT recipe (lr 1e-4, backbone_lr_mult 0.1);
ST-LoRA r8 uses the CORRECTED flat LR (2e-4) that fixed the starved-adapter bug.

Runs in the ISOLATED eomt_cc env (eomt clone + codecarbon). SLURM only.
Emits <out_dir>/{emissions,summary,perepoch,powertrace}_<method>_seed<seed>.{csv,json}.
"""
import os
import sys
import json
import time
import csv
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the generic, trainer-agnostic instrumentation from the Mask2Former wrapper.
from co2_compare_train import PowerSampler, _make_epoch_callback, _resolve_gpu_index  # noqa: E402


def build_argv(method, a):
    """EoMT trainer argv. NO-aug both; batch_size 2; snapshot cadence matched. FRE uses the
    deployed EoMT recipe; ST-LoRA r8 uses the corrected flat LR (2e-4)."""
    save = a.save_root
    common = ["prog",
              "--seed", str(a.seed),
              "--base_save_dir", save,
              "--log_dir", os.path.join(save, "tb_logs"),
              "--epochs", str(a.epochs),
              "--cosine_T0", str(a.cosine_T0),
              "--snapshot_every", str(a.snapshot_every),
              "--batch_size", str(a.batch_size),
              "--num_workers", str(a.num_workers),
              "--gpus", "1",
              "--no_augmentation"]
    if method == "fre":
        # EoMT full-FT deployed recipe (eomt_noaug_train_5seeds.sh)
        return common + ["--lr", "1e-4", "--lr_min", "1e-6",
                         "--backbone_lr_mult", "0.1", "--weight_decay", "0.05"]
    # ST-LoRA r8 — corrected flat-LR recipe (eomt_lora_flatlr_train_5seeds.sh).
    # stlora_ffn = same recipe, FFN-only targets (drop attention q/k/v/out_proj).
    argv = common + ["--lora_r", str(a.lora_r), "--lora_alpha", str(a.lora_r),
                     "--lora_dropout", "0.1", "--lr", "2e-4", "--lr_min", "2e-6"]
    tm = a.target_modules
    if method == "stlora_ffn" and tm is None:
        tm = ["fc1", "fc2", "fc3", "class_predictor"]   # FFN + heads, NO attention
    if tm:
        argv += ["--target_modules", *tm]
    return argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fre", "stlora", "stlora_ffn"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=8)           # ST-LoRA rank (r8 per user)
    ap.add_argument("--target_modules", nargs="+", default=None,
                    help="override LoRA target modules (stlora_ffn defaults to fc1 fc2 fc3 class_predictor)")
    ap.add_argument("--cosine_T0", type=int, default=20)
    ap.add_argument("--snapshot_every", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)       # EoMT deployed bs
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_root", required=True, help="base_save_dir for checkpoints")
    ap.add_argument("--out_dir", required=True, help="where emissions csv + summary go")
    ap.add_argument("--measure_secs", type=int, default=15)
    ap.add_argument("--tracking_mode", default="machine", choices=["machine", "process"])
    ap.add_argument("--log_level", default="info")
    ap.add_argument("--per_epoch_trace", dest="per_epoch_trace", action="store_true", default=True)
    ap.add_argument("--no_per_epoch_trace", dest="per_epoch_trace", action="store_false")
    ap.add_argument("--sample_secs", type=float, default=3.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.save_root, exist_ok=True)

    from codecarbon import OfflineEmissionsTracker  # noqa: E402
    import codecarbon                                # noqa: E402
    print(f"[codecarbon] version={getattr(codecarbon,'__version__','?')} "
          f"mode={a.tracking_mode} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)

    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU",
        output_dir=a.out_dir,
        output_file=f"emissions_{a.method}_seed{a.seed}.csv",
        measure_power_secs=a.measure_secs,
        tracking_mode=a.tracking_mode,
        project_name=f"eomt_{a.method}",
        log_level=a.log_level,
        save_to_file=True,
    )

    if a.method == "fre":
        import eomt_full_train_seeded as mod
    else:
        import eomt_lora_train_seeded as mod

    sys.argv = build_argv(a.method, a)
    print(f"[wrap] calling {mod.__name__}.main() argv={sys.argv[1:]}", flush=True)

    # ---- per-epoch power trace: inject a Lightning callback + background sampler ----
    rows, sampler = [], None
    if a.per_epoch_trace:
        import pytorch_lightning as pl
        epoch_cb = _make_epoch_callback(pl, rows)
        _orig_trainer_init = pl.Trainer.__init__

        def _patched_trainer_init(self, *pa, **kw):
            cbs = kw.get("callbacks")
            if cbs is None:
                cbs = []
            elif isinstance(cbs, (list, tuple)):
                cbs = list(cbs)
            else:
                cbs = [cbs]
            cbs.append(epoch_cb)
            kw["callbacks"] = cbs
            return _orig_trainer_init(self, *pa, **kw)

        pl.Trainer.__init__ = _patched_trainer_init
        gi = _resolve_gpu_index()
        sampler = PowerSampler(interval_s=a.sample_secs, gpu_index=gi)
        print(f"[per-epoch] tracing ON (sample_secs={a.sample_secs}, gpu_index={gi}); "
              f"train window = whole epoch incl. augmentation; validation excluded", flush=True)

    tracker.start()
    if sampler is not None:
        sampler.start()
    err = None
    t_wall0 = time.perf_counter()
    try:
        mod.main()
    except SystemExit as e:
        err = f"SystemExit({e.code})"
    except Exception as e:             # noqa: BLE001
        err = repr(e)
        raise
    finally:
        wall_s = time.perf_counter() - t_wall0
        if sampler is not None:
            sampler.stop()
        kg = tracker.stop()
        d = tracker.final_emissions_data
        summary = {
            "method": a.method, "seed": a.seed, "epochs": a.epochs,
            "lora_r": (a.lora_r if a.method in ("stlora", "stlora_ffn") else None),
            "error": err,
            "wall_clock_s": wall_s,
            "wall_clock_hms": time.strftime("%H:%M:%S", time.gmtime(wall_s)),
            "sec_per_epoch": wall_s / a.epochs if a.epochs else None,
            "duration_s": getattr(d, "duration", None),
            "emissions_kgCO2e": kg,
            "energy_consumed_kWh": getattr(d, "energy_consumed", None),
            "cpu_energy_kWh": getattr(d, "cpu_energy", None),
            "gpu_energy_kWh": getattr(d, "gpu_energy", None),
            "ram_energy_kWh": getattr(d, "ram_energy", None),
            "cpu_power_W": getattr(d, "cpu_power", None),
            "gpu_power_W": getattr(d, "gpu_power", None),
            "gpu_count": getattr(d, "gpu_count", None),
            "gpu_model": getattr(d, "gpu_model", None),
            "country_iso_code": getattr(d, "country_iso_code", None),
        }
        if a.per_epoch_trace:
            for r in rows:
                ts = r.get("train_start")
                te = r.get("val_start") or r.get("train_epoch_end")
                e_kwh, avg_w, n = sampler.window(ts, te) if sampler else (None, None, 0)
                r["train_dur_s"] = (te - ts) if (ts and te) else None
                r["train_gpu_energy_kWh"], r["train_avg_gpu_W"], r["train_n_samples"] = e_kwh, avg_w, n
                if r.get("val_start") and r.get("val_end"):
                    ve, va, _ = (sampler.window(r["val_start"], r["val_end"])
                                 if sampler else (None, None, 0))
                    r["val_dur_s"] = r["val_end"] - r["val_start"]
                    r["val_gpu_energy_kWh"], r["val_avg_gpu_W"] = ve, va
                else:
                    r["val_dur_s"] = r["val_gpu_energy_kWh"] = r["val_avg_gpu_W"] = None
                r["is_warmup"] = int(r.get("epoch") == 0)
            fields = ["epoch", "is_warmup", "train_dur_s", "train_avg_gpu_W",
                      "train_gpu_energy_kWh", "train_n_samples",
                      "val_dur_s", "val_avg_gpu_W", "val_gpu_energy_kWh"]
            pe_path = os.path.join(a.out_dir, f"perepoch_{a.method}_seed{a.seed}.csv")
            with open(pe_path, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                wr.writeheader()
                for r in rows:
                    wr.writerow(r)
            tr_path = None
            if sampler is not None:
                tr_path = os.path.join(a.out_dir, f"powertrace_{a.method}_seed{a.seed}.csv")
                with open(tr_path, "w", newline="") as f:
                    wr = csv.writer(f)
                    wr.writerow(["t_epoch_s", "gpu_W"])
                    for (t, wp) in sampler.samples():
                        wr.writerow([f"{t:.3f}", "" if wp is None else f"{wp:.2f}"])
            steady = [r["train_avg_gpu_W"] for r in rows
                      if r.get("train_avg_gpu_W") is not None and r.get("epoch") != 0]
            sum_tr = sum(r["train_gpu_energy_kWh"] for r in rows
                         if r.get("train_gpu_energy_kWh") is not None)
            summary["per_epoch"] = {
                "n_epochs_traced": len(rows),
                "sample_secs": a.sample_secs,
                "window_note": "train=[epoch_start,val_start) incl. augmentation; validation excluded",
                "steady_state_avg_gpu_W": (sum(steady) / len(steady)) if steady else None,
                "epoch0_warmup_avg_gpu_W": next(
                    (r.get("train_avg_gpu_W") for r in rows if r.get("epoch") == 0), None),
                "sum_train_gpu_energy_kWh": sum_tr,
                "codecarbon_gpu_energy_kWh": summary.get("gpu_energy_kWh"),
                "perepoch_csv": pe_path,
                "powertrace_csv": tr_path,
            }
            print(f"[per-epoch] wrote {pe_path} ({len(rows)} epochs); steady-state avg GPU "
                  f"{summary['per_epoch']['steady_state_avg_gpu_W']} W; sum train GPU energy "
                  f"{sum_tr:.4f} kWh vs codecarbon {summary.get('gpu_energy_kWh')} kWh", flush=True)
        sp = os.path.join(a.out_dir, f"summary_{a.method}_seed{a.seed}.json")
        json.dump(summary, open(sp, "w"), indent=2)
        print(f"[WALL-CLOCK] {a.method}: {summary['wall_clock_hms']} "
              f"({wall_s:.1f}s, {summary['sec_per_epoch']:.2f} s/epoch over {a.epochs} epochs)",
              flush=True)
        print("[codecarbon SUMMARY] " + json.dumps(summary), flush=True)
        print(f"[saved] {sp}", flush=True)


if __name__ == "__main__":
    main()
