# -*- coding: utf-8 -*-
"""
SEGFORMER CO2/energy comparison (FRE full-FT vs ST-LoRA) — mirrors co2_compare_train.py (M2F)
and eomt_co2_compare_train.py (EoMT). Measure the CO2/energy of ONE SegFormer training run with
CodeCarbon WITHOUT modifying the trainers: import the target trainer module and call its main()
inside an OfflineEmissionsTracker (country=DEU, no network). Both SegFormer trainers use the SAME
aug+CutMix pipeline, batch size, cadence, and 1 GPU — the only difference is full-FT vs LoRA
(deployed final_model config r8/a8, targets q/k/v/dense, modules_to_save decode_head) — so the
resulting energies are directly comparable, and comparable to the M2F/EoMT CO2 studies.

Runs in the ISOLATED ssl_cc env (ssl clone + codecarbon). SLURM only.
Emits <out_dir>/emissions_<method>_seed<seed>.csv (codecarbon) + summary/per-epoch/powertrace.
"""
import os
import sys
import json
import time
import csv
import threading
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _resolve_gpu_index():
    """Physical NVML index of the (single) allocated GPU. Under SLURM cgroup isolation
    NVML often sees only the one allocated device (index 0); otherwise honor the first
    CUDA_VISIBLE_DEVICES entry."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        import pynvml
        pynvml.nvmlInit(); n = pynvml.nvmlDeviceGetCount(); pynvml.nvmlShutdown()
    except Exception:                 # noqa: BLE001
        return 0
    if n <= 1:
        return 0
    first = cvd.split(",")[0].strip() if cvd else ""
    return int(first) if first.isdigit() else 0


class PowerSampler:
    """Background thread that polls instantaneous GPU power (pynvml) at a fixed interval and
    integrates it (trapezoidal) over arbitrary [t0,t1] windows. Runs ALONGSIDE CodeCarbon,
    which still owns the authoritative whole-run total; this only supplies the per-epoch
    decomposition. Degrades to a no-op (energy None; timestamps still usable) without pynvml."""

    def __init__(self, interval_s=3.0, gpu_index=0):
        self.interval = float(interval_s)
        self._stop = threading.Event()
        self._samples = []            # list[(t_epoch_s, gpu_W|None)]
        self._lock = threading.Lock()
        self._thr = None
        self._h = None
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as e:        # noqa: BLE001
            print(f"[power-sampler] pynvml unavailable ({e!r}); per-epoch GPU energy disabled",
                  flush=True)
            self.pynvml = None

    def _read_w(self):
        if self._h is None:
            return None
        try:
            return self.pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0   # mW -> W
        except Exception:             # noqa: BLE001
            return None

    def _loop(self):
        while not self._stop.is_set():
            t, w = time.time(), self._read_w()
            with self._lock:
                self._samples.append((t, w))
            self._stop.wait(self.interval)

    def start(self):
        if self.pynvml is None:
            return
        self._thr = threading.Thread(target=self._loop, name="power-sampler", daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=5)
        if self.pynvml is not None:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:         # noqa: BLE001
                pass

    def window(self, t0, t1):
        """(gpu_energy_kWh, avg_gpu_W, n_samples) integrated over [t0,t1]."""
        if t0 is None or t1 is None or t1 <= t0:
            return (None, None, 0)
        with self._lock:
            s = [(t, w) for (t, w) in self._samples if w is not None and t0 <= t <= t1]
        if len(s) < 2:
            return (None, None, len(s))
        e_j = 0.0
        for (ta, wa), (tb, wb) in zip(s, s[1:]):
            e_j += 0.5 * (wa + wb) * (tb - ta)
        dur = s[-1][0] - s[0][0]
        return (e_j / 3.6e6, (e_j / dur if dur > 0 else None), len(s))    # J -> kWh

    def samples(self):
        with self._lock:
            return list(self._samples)


def _make_epoch_callback(pl, rows):
    """Lightning callback (injected via monkeypatch, so the trainer scripts stay untouched)
    that records per-epoch wall-clock boundaries. The TRAIN window is
    [on_train_epoch_start, on_validation_epoch_start): it spans the WHOLE training epoch —
    every batch, augmentation/CutMix included, since those run per-batch inside the loop —
    and ends exactly where validation begins, so validation energy is excluded regardless of
    PL's hook-nesting order. If an epoch has no validation, the window ends at train_epoch_end."""

    class _EnergyCB(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if getattr(trainer, "sanity_checking", False):
                return
            self._cur = {"epoch": int(trainer.current_epoch), "train_start": time.time()}

        def on_validation_epoch_start(self, trainer, pl_module):
            if getattr(trainer, "sanity_checking", False):
                return
            if getattr(self, "_cur", None):
                self._cur["val_start"] = time.time()

        def on_validation_epoch_end(self, trainer, pl_module):
            if getattr(trainer, "sanity_checking", False):
                return
            if getattr(self, "_cur", None):
                self._cur["val_end"] = time.time()

        def on_train_epoch_end(self, trainer, pl_module):
            cur = getattr(self, "_cur", None)
            if not cur:
                return
            cur["train_epoch_end"] = time.time()
            rows.append(dict(cur))
            self._cur = None

    return _EnergyCB()


def build_argv(method, a):
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
              "--gpus", "1"]
    if method == "fre":
        return common
    lora = common + ["--lora_r", "8", "--lora_alpha", "8", "--lora_dropout", "0.1",
                     "--no_dora"]
    # stlora_ffn — FFN-only mirror: target MixFFN dense1/dense2, drop attention q/k/v/dense.
    # decode_head still fully trained (trainer default modules_to_save). Same aug+CutMix.
    if method == "stlora_ffn":
        return lora + ["--target_modules", "dense1", "dense2"]
    # stlora_ffnattn — NESTED arm (mirror of EoMT/M2F): FFN (dense1/dense2) + attention
    # (query/key/value/dense). Strict superset of stlora_ffn. decode_head still fully trained.
    if method == "stlora_ffnattn":
        return lora + ["--target_modules", "dense1", "dense2", "query", "key", "value", "dense"]
    # ST-LoRA — deployed SegFormer final_model config (r8/a8, targets q/k/v/dense = trainer
    # defaults, modules_to_save decode_head). Same aug+CutMix as full-FT.
    return lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fre", "stlora", "stlora_ffn", "stlora_ffnattn"],
                    required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cosine_T0", type=int, default=20)
    ap.add_argument("--snapshot_every", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_root", required=True, help="base_save_dir for checkpoints")
    ap.add_argument("--out_dir", required=True, help="where emissions csv + summary go")
    ap.add_argument("--measure_secs", type=int, default=15)
    ap.add_argument("--tracking_mode", default="machine", choices=["machine", "process"])
    ap.add_argument("--log_level", default="info")
    ap.add_argument("--per_epoch_trace", dest="per_epoch_trace", action="store_true", default=True,
                    help="record a per-epoch GPU power/energy trace (default ON)")
    ap.add_argument("--no_per_epoch_trace", dest="per_epoch_trace", action="store_false")
    ap.add_argument("--sample_secs", type=float, default=3.0,
                    help="power sampling interval (s) for the per-epoch trace")
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
        project_name=f"segformer_{a.method}",
        log_level=a.log_level,
        save_to_file=True,
    )

    if a.method == "fre":
        import segformer_fullft_train_seeded as mod
    else:
        import segformer_lora_train_seeded as mod

    sys.argv = build_argv(a.method, a)
    print(f"[wrap] calling {mod.__name__}.main() argv={sys.argv[1:]}", flush=True)

    # ---- per-epoch power trace: inject a Lightning callback + start a background sampler ----
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
    t_wall0 = time.perf_counter()      # WALL-CLOCK training time (reviewer-requested)
    try:
        mod.main()
    except SystemExit as e:            # trainer should not sys.exit, but be safe
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
        # ---- per-epoch decomposition (train window incl. augmentation; validation excluded) ----
        if a.per_epoch_trace:
            for r in rows:
                ts = r.get("train_start")
                te = r.get("val_start") or r.get("train_epoch_end")   # val_start = train-window end
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
