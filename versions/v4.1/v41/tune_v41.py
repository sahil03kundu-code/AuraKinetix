"""Supervisor tuning for the V4.1 families (validation seeds 300-303 x hits 1-4, exactly the grid,
objective and runs used for V3 / V4 / Baseline D). Writes supervisor_by_family into
results/v41/tuning_v41.json (pipeline parameters there are kept)."""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import tune_v3 as TV
import benchmark_v41 as BV
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.safety import SafetyShield, ShieldedController

GRID = [dict(gamma=g, kappa=k, stale_deg=s) for g, k, s in itertools.product((0.03, 0.05), (0.0, 1.0, 2.0), (4.0, 8.0))]
FAMS = {"geoev": "geoev", "v41": "v41-s0"}

def job(spec):
    torch.set_num_threads(1)
    fam, i = spec
    p = PlantParams(); lqr = LQRController(p)
    mk = lambda: BV.build(FAMS[fam])[0]
    if i is None:
        rows, _ = TV.val_run(mk, lambda pipe: lqr)
    else:
        rows, _ = TV.val_run(mk, lambda pipe: ShieldedController(lqr, SafetyShield(p, **GRID[i]), pipe))
    return fam, i, TV.score(rows), float(np.mean([r[0] for r in rows])), float(np.mean([r[0] and r[1] <= 12 for r in rows]))

if __name__ == "__main__":
    specs = [(f, None) for f in FAMS] + [(f, i) for f in FAMS for i in range(len(GRID))]
    with Pool(2) as pool:
        res = pool.map(job, specs, chunksize=1)
    path = "results/v41/tuning_v41.json"
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["supervisor_grid"] = []; out["reference_no_supervisor"] = {}
    for fam, i, sc, sv, sf in res:
        if i is None:
            out["reference_no_supervisor"][fam] = dict(score=sc, survival=sv, safe=sf)
        else:
            out["supervisor_grid"].append(dict(family=fam, params=GRID[i], score=sc, survival=sv, safe=sf))
    out["supervisor_by_family"] = {}
    for fam in FAMS:
        best = max((r for r in out["supervisor_grid"] if r["family"] == fam), key=lambda r: r["score"])
        out["supervisor_by_family"][fam] = best["params"]
        print(fam, "chosen", best, "| no supervisor:", out["reference_no_supervisor"][fam], flush=True)
    out["note"] = "validation seeds 300-303, hits 1-4 rad/s, nominal; same objective/grid as V3/V4 shield tuning"
    json.dump(out, open(path, "w"), indent=1)
