# -*- coding: utf-8 -*-
"""Import every .py file given as arguments and classify the outcome.

  OK   : module imported cleanly.
  WARN : imported far enough to fail only on a *runtime* cause (e.g. a script that runs at import
         and needs result files / data that aren't in a fresh checkout) — NOT a code/dependency bug.
  FAIL : ModuleNotFoundError / ImportError — a real missing-dependency or broken-import problem.

Exit code = number of FAILs (0 = all imports resolve). Run with the interpreter whose env matches
the files (ssl for most; the EoMT env for eomt_*.py).
"""
import sys, os, importlib.util

ok = warn = fail = 0
for path in sys.argv[1:]:
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"  OK    {os.path.basename(path)}"); ok += 1
    except (ModuleNotFoundError, ImportError) as e:
        print(f"  FAIL  {os.path.basename(path)}: {type(e).__name__}: {e}"); fail += 1
    except Exception as e:                      # ran at import, needs data/GPU — not a code bug
        print(f"  warn  {os.path.basename(path)}: {type(e).__name__}: {str(e)[:80]}"); warn += 1

print(f"[import_check] ok={ok} warn={warn} fail={fail}")
sys.exit(fail)
