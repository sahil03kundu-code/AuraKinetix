"""V4 Step 2b: development set of closed-loop runs (VALIDATION seeds only).

Runs the V3 triggered event observer (Baseline B, seed-0 model) with and without the
shield on validation seeds 300-315 and five hit strengths, logging every tick:
true state, estimate, covariance, control, and the full sensor surface (float16).
V4's detector and corrector are developed by replaying these runs offline;
test seeds (500+/800+/7000+) are never used for design decisions.
Output: data/v4dev/<cfg>_<seed>_<kick>.npz and results/v4/dev_catalogue.json
"""
import os, sys, json, time
from multiprocessing import Pool
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import benchmark_v3 as B
from aurasync.physics import PlantParams
from aurasync.simulate import run_episode

DEV_SEEDS = list(range(300, 316))
DEV_KICKS = [1.5, 2.5, 3.0, 3.5, 4.0]
CFGS = ["ev-trig-s0", "ev-trig+shield-s0"]
OUT = "data/v4dev"


class Logged:
    """Wraps a pipeline to record its covariance at every tick (no behaviour change)."""

    def __init__(self, pipe):
        self.p = pipe; self.sig = []

    def __getattr__(self, k):
        return getattr(self.p, k)

    def reset(self, s0):
        self.sig = []; self.p.reset(s0)

    def update(self, *a):
        self.p.update(*a); self.sig.append(np.sqrt(np.clip(np.diag(self.p.covariance()), 0, None)))

    def estimate(self):
        return self.p.estimate()


def job(args):
    cfg, sd, kick = args
    torch.set_num_threads(1)
    path = f"{OUT}/{cfg}_{sd}_{kick}.npz"
    if os.path.exists(path):
        return path
    pipe, ctrl_f = B.build(cfg)
    lp = Logged(pipe)
    rng = np.random.default_rng(sd); s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
    k = kick * (1 if sd % 2 == 0 else -1)
    log = run_episode(PlantParams(), ctrl_f(pipe), lp, B.N_TICKS, s0, kicks=[(B.KICK_TICK, k)], cam_seed=sd, record=True)
    np.savez(path, s=log["s"], s_hat=log["s_hat"], sig=np.array(lp.sig), u=log["u"], events=log["events"],
             surf=log["surf"], s0=np.array(s0), fail=-1 if log["fail_tick"] is None else log["fail_tick"], kick=k)
    return path


def catalogue():
    rows = []
    for f in sorted(os.listdir(OUT)):
        d = np.load(f"{OUT}/{f}")
        th, eh, sg = np.degrees(d["s"][:, 2]), np.degrees(d["s_hat"][:, 2]), np.degrees(d["sig"][:, 2])
        e = np.abs(eh - th)
        conf_wrong = (e > 5) & (e > 3 * sg)
        sign_err = (np.sign(eh) != np.sign(th)) & (np.abs(th) > 3) & (np.abs(eh) > 1)
        first_cw = int(np.argmax(conf_wrong)) if conf_wrong.any() else None
        rows.append(dict(file=f, fail=int(d["fail"]), max_err=float(e.max()), conf_wrong_ticks=int(conf_wrong.sum()),
                         sign_err_ticks=int(sign_err.sum()), first_conf_wrong=first_cw,
                         max_err_over_sigma=float((e / np.maximum(sg, 1e-6)).max())))
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs("results/v4", exist_ok=True)
    jobs = [(c, s, k) for c in CFGS for s in DEV_SEEDS for k in DEV_KICKS]
    t0 = time.time()
    with Pool(2) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, jobs)):
            if (i + 1) % 40 == 0:
                print(f"{i+1}/{len(jobs)} runs {time.time()-t0:.0f}s", flush=True)
    rows = catalogue()
    json.dump(rows, open("results/v4/dev_catalogue.json", "w"), indent=1)
    for cfg in CFGS:
        r = [x for x in rows if x["file"].startswith(cfg + "_")]
        fails = [x for x in r if x["fail"] >= 0]
        cw = [x for x in r if x["conf_wrong_ticks"] > 0]
        print(f"{cfg}: {len(r)} runs, {len(fails)} falls, {len(cw)} runs with confidently-wrong ticks, "
              f"{sum(1 for x in fails if x['conf_wrong_ticks'] > 0)}/{len(fails)} falls preceded by confident errors, "
              f"{sum(1 for x in r if x['sign_err_ticks'] > 0)} runs with sign errors")
