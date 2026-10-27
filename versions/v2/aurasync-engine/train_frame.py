"""Train the frame-camera state estimator (the fair baseline).

Each sample: the last 3 frames at a random frame rate (30/60/120 fps) and random
exposure (1/2/4/8 ms), blurred from 1 ms renders and given read noise; target =
true state at capture time. Same tilt-weighted loss as the event observer.
"""
import argparse, os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from aurasync.frame_camera import FrameEstimator, N_FRAMES, READ_NOISE, cond_vector
from aurasync.estimator import SCALE
from generate_frame_data import PAD

torch.set_num_threads(2)
FPS = (30, 60, 120)
EXPOSURES = (0.001, 0.002, 0.004, 0.008)
W_STATE = torch.tensor([1.0, 1.0, 4.0, 2.0])


class FrameData:
    def __init__(self, path):
        d = np.load(path)
        self.img, self.s, self.lens = d["img"], d["s"], d["lens"]
        self.img_start = np.concatenate([[0], np.cumsum(self.lens + PAD)[:-1]])
        self.s_start = np.concatenate([[0], np.cumsum(self.lens)[:-1]])

    def item(self, e, i, fps, exposure, rng):
        """Frame stack for a capture at the end of tick i of episode e."""
        E = int(round(exposure * 1e3))
        base = self.img_start[e] + PAD
        frames = []
        for j in range(N_FRAMES):
            c = i - int(round(j * 1000.0 / fps))
            lo = max(c - E + 1, -PAD)
            img = self.img[base + lo: base + c + 1].astype(np.float32).mean(0) / 255.0
            frames.append(img + rng.normal(0, READ_NOISE, img.shape).astype(np.float32))
        return np.stack(frames), self.s[self.s_start[e] + i]

    def batch(self, rng, B):
        X, Y, C = [], [], []
        for _ in range(B):
            e = rng.integers(len(self.lens)); i = rng.integers(self.lens[e])
            fps = FPS[rng.integers(3)]; exp = EXPOSURES[rng.integers(4)]
            x, y = self.item(e, i, fps, exp, rng)
            X.append(x); Y.append(y); C.append(cond_vector(fps, exp)[0])
        return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y)), torch.stack(C)


def loss_fn(pred, y):
    w = W_STATE.to(pred)
    return ((((pred - y) / SCALE) ** 2) * w).mean() / w.mean()


@torch.no_grad()
def evaluate(model, val, exposure=0.004, stride=5):
    model.eval()
    rng = np.random.default_rng(123)
    out = {}
    for fps in FPS:
        errs = []
        for e in range(len(val.lens)):
            idx = np.arange(80, val.lens[e], stride)
            X, Y = zip(*[val.item(e, i, fps, exposure, rng) for i in idx])
            pred = model(torch.from_numpy(np.stack(X)), cond_vector(fps, exposure).expand(len(idx), -1))
            errs.append((pred - torch.from_numpy(np.stack(Y))).abs().numpy())
        err = np.concatenate(errs)
        out[fps] = dict(mae_x_mm=float(err[:, 0].mean() * 1e3), mae_xd=float(err[:, 1].mean()),
                        mae_th_deg=float(np.degrees(err[:, 2].mean())),
                        mae_thd_degs=float(np.degrees(err[:, 3].mean())))
    model.train()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    rng = np.random.default_rng(0); torch.manual_seed(0)
    train, val = FrameData("data/frames_train.npz"), FrameData("data/frames_val.npz")
    model = FrameEstimator()
    print("params:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    hist, best, t0 = [], 1e9, time.time()
    for step in range(1, a.steps + 1):
        X, Y, C = train.batch(rng, a.batch)
        loss = loss_fn(model(X, C), Y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if step % 1000 == 0 or step == a.steps:
            m = evaluate(model, val)
            print("  val:", json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in m.items()}), flush=True)
            score = np.mean([v["mae_th_deg"] + v["mae_x_mm"] / 10 for v in m.values()])
            hist.append(dict(step=step, val=m, score=float(score)))
            if score < best:
                best = score
                torch.save(model.state_dict(), "checkpoints/frame_estimator.pt")
    json.dump(hist, open("checkpoints/frame_train_history.json", "w"), indent=1)
    print("done", time.time() - t0, "best score", best)
