"""Post-hoc DIAGNOSTIC (no design change): ood_plant held-out trials at 3 rad/s without the supervisor,
to attribute V4.1's track-end falls. Writes results/v41/diag_ood_nosup.json"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from multiprocessing import Pool
import numpy as np
import benchmark_v41 as BV
def job(cfg):
    _, _, rows = BV.run_cfg(("ood_plant", cfg, 4000, None))
    rows = [r for r in rows if r["kick"] == 3.0]
    return cfg, dict(n=len(rows), survival=float(np.mean([r["survived"] for r in rows])), track=float(np.mean([r["track_violation"] for r in rows])),
                     safe=float(np.mean([r["safe"] for r in rows])), th=float(np.mean([r["est_th_deg"] for r in rows])))
if __name__ == "__main__":
    with Pool(2) as pool:
        out = dict(pool.map(job, ["geoev", "v41-s0", "geo"]))
    for k, v in out.items(): print(k, v)
    json.dump(out, open("results/v41/diag_ood_nosup.json", "w"), indent=1)
