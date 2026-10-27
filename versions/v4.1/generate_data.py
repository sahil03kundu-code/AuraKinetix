"""Generate simulated event-camera training data.

Closed-loop episodes on randomised plants, with exploration noise on the
actuator and random impulse 'hits' to the pole, so the observer sees calm
balancing, violent recoveries and everything between.
Output: data/{train,val}.npz with time surfaces (float16), forces, true states.
"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.simulate import run_episode, OraclePipeline, EventPipeline

POLICY = None  # path to an observer checkpoint for on-policy (DAgger) data


def episode_setup(seed, n_ticks=1500):
    """Randomised plant, exploration noise, initial state and impulse schedule for one episode."""
    rng = np.random.default_rng(seed)
    nominal = PlantParams()
    p = nominal.perturbed(rng)
    ctrl = LQRController(nominal)
    ou = [0.0]
    sigma = rng.uniform(1.0, 6.0)

    def explore():
        ou[0] += -ou[0] * 0.02 + sigma * 0.2 * rng.normal()
        return ou[0]

    s0 = [rng.uniform(-0.3, 0.3), 0, rng.uniform(-0.12, 0.12), rng.uniform(-0.5, 0.5)]
    kicks, k = [], int(rng.integers(100, 400))
    while k < n_ticks:
        kicks.append((k, rng.uniform(-3.0, 3.0)))
        k += int(rng.integers(250, 600))
    return p, ctrl, explore, s0, kicks, sigma


def make_episode(seed, n_ticks=1500):
    p, ctrl, explore, s0, kicks, sigma = episode_setup(seed, n_ticks)
    pipe = OraclePipeline()
    if POLICY:  # DAgger: the learned observer drives the loop, labels stay ground truth
        import torch
        from aurasync.estimator import EventObserver
        torch.set_num_threads(1)
        m = EventObserver(); m.load_state_dict(torch.load(POLICY)); pipe = EventPipeline(m)
        explore_base = explore
        explore = lambda: 0.3 * explore_base()
    log = run_episode(p, ctrl, pipe, n_ticks, s0, kicks=kicks,
                      exploration=explore, record=True, cam_seed=seed, stop_on_fail=True)
    return log["surf"], log["u"].astype(np.float32), log["s"].astype(np.float32), np.array(s0, np.float32)


def build(seeds, path):
    global POLICY
    t0 = time.time()
    with Pool(2) as pool:
        eps = pool.map(make_episode, seeds)
    eps = [e for e in eps if len(e[1]) >= 300]
    lens = np.array([len(e[1]) for e in eps])
    np.savez(path, surf=np.concatenate([e[0] for e in eps]), u=np.concatenate([e[1] for e in eps]),
             s=np.concatenate([e[2] for e in eps]), s0=np.stack([e[3] for e in eps]), lens=lens)
    print(f"{path}: {len(eps)} episodes, {lens.sum()} ticks, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--val", type=int, default=6)
    ap.add_argument("--policy", default=None, help="observer checkpoint -> on-policy DAgger data")
    ap.add_argument("--out", default="data/train.npz")
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    if a.policy:
        POLICY = a.policy
        build(range(20000, 20000 + a.train), a.out)
    else:
        build(range(1000, 1000 + a.train), "data/train.npz")
        build(range(9000, 9000 + a.val), "data/val.npz")
