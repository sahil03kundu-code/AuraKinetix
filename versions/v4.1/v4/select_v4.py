"""V4 Step 6b: design selection on VALIDATION seeds (300-315 x 4 hit levels = 64 runs).

Compares the candidate V4 designs and Baseline D before any test run, so the
'proposed V4' is fixed by validation data only. Paired by run; reports survival,
safe recoveries, mean estimation error and the score used in all tuning.
"""
import os, sys, json
from multiprocessing import Pool
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import tune_v3 as TV
import benchmark_v4 as B4
from aurasync.physics import PlantParams
from aurasync.simulate import run_episode

TV.VAL_SEEDS = list(range(300, 316))
CANDIDATES = ["ev-trig-s0", "geo", "geo+fb", "v4-s0", "v4+fb-s0", "v4-kf-s0", "v4-kf+fb-s0"]
if os.environ.get("ROUND") == "2":     # after the detector fix + position-only neural fusion
    CANDIDATES = ["v4-kp-s0", "v4-kp+fb-s0", "v4-kp+sup-s0"]


def job(cfg):
    torch.set_num_threads(1)
    p = PlantParams(); rows = []
    for kick in TV.VAL_KICKS:
        for sd in TV.VAL_SEEDS:
            pipe, ctrl_f = B4.build(cfg)
            rng = np.random.default_rng(sd); s0 = [rng.uniform(-.1, .1), 0, rng.uniform(-.03, .03), 0]
            lg = run_episode(p, ctrl_f(pipe), pipe, 1000, s0, kicks=[(200, kick * (1 if sd % 2 else -1))], cam_seed=sd)
            pk = float(np.degrees(np.abs(lg["s"][200:, 2]).max())) if len(lg["s"]) > 200 else 90.0
            est = float(np.degrees(np.abs(lg["s_hat"] - lg["s"])[50:, 2].mean()))
            rows.append((lg["fail_tick"] is None, pk, est))
    return cfg, rows


if __name__ == "__main__":
    with Pool(2) as pool:
        res = dict(pool.map(job, CANDIDATES, chunksize=1))
    out = {}
    for cfg, rows in res.items():
        sv = np.array([r[0] for r in rows], float); sf = np.array([r[0] and r[1] <= 12 for r in rows], float)
        out[cfg] = dict(score=TV.score([(r[0], r[1]) for r in rows]), survival=float(sv.mean()), safe=float(sf.mean()),
                        est_deg=float(np.mean([r[2] for r in rows])), n=len(rows),
                        survived=sv.tolist(), safe_list=sf.tolist())
        print(f"{cfg:14s} score {out[cfg]['score']:.3f}  survival {sv.mean()*100:5.1f}%  safe {sf.mean()*100:5.1f}%  "
              f"est error {out[cfg]['est_deg']:.2f} deg  (n={len(rows)})", flush=True)
    json.dump(out, open("results/v4/select_v4%s.json" % ("_round2" if os.environ.get("ROUND") == "2" else ""), "w"), indent=1)
