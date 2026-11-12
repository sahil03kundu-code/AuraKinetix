"""V4 Step 6: first closed-loop test + supervisor tuning (validation seeds 300-315 only).

Same objective, seeds and grid as V3's shield tuning (tune_v3.py), so the V4 family
gets matched tuning effort. Also scores V4 without a supervisor and the V3
triggered observer on the same runs for reference.
"""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import tune_v3 as TV
import benchmark_v4 as B4
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.safety import SafetyShield, ShieldedController

GRID = [dict(gamma=g, kappa=k, stale_deg=s) for g, k, s in itertools.product((0.03, 0.05), (0.0, 1.0, 2.0), (4.0, 8.0))]


def job(spec):
    torch.set_num_threads(1)
    kind, i = spec
    p = PlantParams(); lqr = LQRController(p)
    if kind.startswith("grid"):
        fam_cfg = "geo" if kind == "grid-geo" else "v4-s0"
        mk = lambda: B4.build(fam_cfg)[0]
        rows, calls = TV.val_run(mk, lambda pipe: ShieldedController(lqr, SafetyShield(p, **GRID[i]), pipe))
    else:
        mk = lambda: B4.build(kind)[0]
        rows, calls = TV.val_run(mk, lambda pipe: lqr)
    surv = float(np.mean([r[0] for r in rows])); safe = float(np.mean([r[0] and r[1] <= 12 for r in rows]))
    return kind, i, TV.score(rows), surv, safe


if __name__ == "__main__":
    os.makedirs("results/v4", exist_ok=True)
    specs = [("ev-trig-s0", None), ("v4-s0", None), ("geo", None)] + [(g, i) for g in ("grid", "grid-geo") for i in range(len(GRID))]
    with Pool(2) as pool:
        res = pool.map(job, specs, chunksize=1)
    out = dict(reference={}, grid=[])
    for kind, i, sc, sv, sf in res:
        if kind.startswith("grid"):
            fam = "geo" if kind == "grid-geo" else "v4"
            out["grid"].append(dict(family=fam, params=GRID[i], score=sc, survival=sv, safe=sf))
            print(f"{fam}+sup {GRID[i]}  score {sc:.3f} surv {sv:.2f} safe {sf:.2f}", flush=True)
        else:
            out["reference"][kind] = dict(score=sc, survival=sv, safe=sf)
            print(f"{kind:12s} (no supervisor)  score {sc:.3f} surv {sv:.2f} safe {sf:.2f}", flush=True)
    out["supervisor_by_family"] = {}
    for fam in ("v4", "geo"):
        best = max((r for r in out["grid"] if r["family"] == fam), key=lambda r: r["score"])
        out["supervisor_by_family"][fam] = best["params"]
        print("chosen supervisor", fam, best, flush=True)
    out["note"] = "validation seeds 300-315, hits 1-4 rad/s; same objective and grid style as V3 shield tuning"
    json.dump(out, open("results/v4/tuning_v4.json", "w"), indent=1)
