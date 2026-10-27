"""V4 Step 2a: reproduce V3 benchmark results from scratch.

Re-runs a deterministic sample of V3 test trials for several configurations and
compares survival, peak tilt and estimation error with the stored raw results
(results/v3/raw). Writes results/v4/repro_v3.json.
"""
import os, sys, json, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import benchmark_v3 as B
from aurasync.simulate import run_episode

torch.set_num_threads(1)
SAMPLE = [("nominal", "ev-dense-s1"), ("nominal", "ev-trig-s2"), ("nominal", "ev-trig+shield-s0"),
          ("nominal", "imu+shield"), ("hdr", "eventonly-dense-s0"), ("snap", "frame120-s1")]


def rerun(cond_name, cfg, rows):
    cond, plant, snap, _, _ = B.cond_spec(cond_name)
    pipe, ctrl_f = B.build(cfg)
    out = []
    for r in rows:
        i = r["trial"]; seed = (800 if cond_name == "snap" else 500) + i
        rng = np.random.default_rng(seed); s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
        log = run_episode(plant, ctrl_f(pipe), pipe, B.N_TICKS, s0, kicks=[(B.KICK_TICK, r["kick"] * (1 if i % 2 == 0 else -1))],
                          snap=snap, cam_seed=seed, cond=cond)
        th = log["s"][:, 2]
        peak = float(np.degrees(np.abs(th[B.KICK_TICK:]).max())) if len(th) > B.KICK_TICK else 90.0
        est = float(np.degrees(np.abs(log["s_hat"] - log["s"])[50:, 2].mean()))
        out.append(dict(kick=r["kick"], trial=i, survived_v3=r["survived"], survived_now=log["fail_tick"] is None,
                        peak_v3=r["peak_deg"], peak_now=min(peak, 90.0), est_v3=r["est_th_deg"], est_now=est))
    return out


if __name__ == "__main__":
    os.makedirs("results/v4", exist_ok=True)
    res = {}
    for cond_name, cfg in SAMPLE:
        rows = json.load(open(f"results/v3/raw/{cond_name}/{cfg}.json"))["rows"]
        pick = [rows[j] for j in np.linspace(0, len(rows) - 1, 6).astype(int)]     # spread over hit strengths
        t0 = time.time()
        out = rerun(cond_name, cfg, pick)
        exact = all(o["survived_v3"] == o["survived_now"] and abs(o["peak_v3"] - o["peak_now"]) < 1e-9
                    and abs(o["est_v3"] - o["est_now"]) < 1e-9 for o in out)
        res[f"{cond_name}|{cfg}"] = dict(trials=out, exact=exact)
        print(f"{cond_name:8s} {cfg:22s} {len(out)} trials  exact match: {exact}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("results/v4/repro_v3.json", "w"), indent=1)
