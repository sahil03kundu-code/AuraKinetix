"""Validation runs (seeds 300+ only) for V4.1 design decisions.
usage: TAG=name CONFIGS=a,b OVERRIDES='{"min_gap":20}' python3 v41/validate.py
Writes results/v41/val/<TAG>.json with per-condition summaries."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from multiprocessing import Pool
import numpy as np
import benchmark_v41 as BV

N = {"nominal": 4, "occlusion": 12, "combo": 12}
CONDS = os.environ.get("CONDS", ",".join(BV.CONDS)).split(",")
OVR = json.loads(os.environ.get("OVERRIDES", "{}"))

def job(a):
    cond, cfg = a
    if OVR:
        orig = BV.build
        BV.build = lambda c, overrides=None: orig(c, OVR if (c.startswith("v41") or c.startswith("geoev")) else None)
    return BV.run_cfg((cond, cfg, 300, N.get(cond, 8)))

if __name__ == "__main__":
    tag = os.environ["TAG"]; cfgs = os.environ["CONFIGS"].split(",")
    jobs = [(c, g) for c in CONDS for g in cfgs]
    out = {}
    with Pool(2) as pool:
        for cond, cfg, rows in pool.imap_unordered(job, jobs):
            m = lambda k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None
            out.setdefault(cond, {})[cfg] = {k: m(k) for k in ("survived", "safe", "est_th_deg", "est_dth_degs", "track_violation",
                                                                 "calls_per_s", "interventions", "false_interventions")}
            out[cond][cfg]["n"] = len(rows)
            s = out[cond][cfg]
            print(f"{cond:15s} {cfg:16s} n{len(rows):3d} surv {s['survived']:.2f} safe {s['safe']:.2f} est {s['est_th_deg']:5.2f} "
                  f"dth {s['est_dth_degs']:5.0f} calls/s {s['calls_per_s']:4.0f} FI {s['false_interventions'] or 0:.3f}", flush=True)
    os.makedirs("results/v41/val", exist_ok=True)
    json.dump(dict(overrides=OVR, results=out), open(f"results/v41/val/{tag}.json", "w"), indent=1)
