"""Frame-camera training data on the SAME trajectories the event observer was trained on.

Re-runs the identical seeded episodes (seeds 1000.. train, 9000.. val) and stores
one sharp 1 ms render per tick (uint8) plus the true state. Blur, frame rate and
read noise are applied at training time, so one dataset covers 30/60/120 fps
and every exposure setting.
Output: data/frames_{train,val}.npz
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from aurasync.simulate import run_episode, OraclePipeline
from aurasync.event_camera import SceneRenderer
from generate_data import episode_setup

PAD = 96  # ticks of pre-roll (initial pose) so the oldest frame/exposure is always defined


def frame_episode(seed, n_ticks=1500):
    p, ctrl, explore, s0, kicks, _ = episode_setup(seed, n_ticks)
    log = run_episode(p, ctrl, OraclePipeline(), n_ticks, s0, kicks=kicks,
                      exploration=explore, cam_seed=seed)
    r = SceneRenderer()
    first = r.render(np.array(s0, float), p.l)
    imgs = [first] * PAD + [r.render(s, p.l) for s in log["s"]]
    return (np.clip(np.stack(imgs) * 255 + 0.5, 0, 255).astype(np.uint8), log["s"].astype(np.float32))


def build(seeds, path):
    t0 = time.time()
    with Pool(2) as pool:
        eps = pool.map(frame_episode, seeds)
    eps = [e for e in eps if len(e[1]) >= 300]   # same filter as the event dataset
    np.savez(path, img=np.concatenate([e[0] for e in eps]), s=np.concatenate([e[1] for e in eps]),
             lens=np.array([len(e[1]) for e in eps]))
    print(f"{path}: {len(eps)} episodes, {sum(len(e[1]) for e in eps)} ticks, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    build(range(1000, 1040), "data/frames_train.npz")
    build(range(9000, 9006), "data/frames_val.npz")
