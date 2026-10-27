"""Train the physics-informed event observer with truncated BPTT."""
import argparse, os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from aurasync.estimator import EventObserver, physics_informed_loss, SCALE

torch.set_num_threads(2)


class Windows:
    def __init__(self, path):
        base = path[:-4]
        if os.path.exists(base + "_surf.npy"):   # memory-mapped: parallel jobs share one copy in page cache
            ld = lambda k: np.load(f"{base}_{k}.npy", mmap_mode="r" if k == "surf" else None)
            self.surf, self.u, self.s, self.s0, lens = ld("surf"), ld("u"), ld("s"), ld("s0"), ld("lens")
        else:
            d = np.load(path)
            self.surf, self.u, self.s, self.s0, lens = d["surf"], d["u"], d["s"], d["s0"], d["lens"]
        self.starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        self.lens = lens

    def sample(self, rng, B, T, init_noise):
        surf, u, s, s0 = [], [], [], []
        for _ in range(B):
            e = rng.integers(len(self.lens))
            i = self.starts[e] + rng.integers(0, self.lens[e] - T + 1)
            init = self.s0[e] if i == self.starts[e] else self.s[i - 1]
            surf.append(self.surf[i:i + T]); u.append(self.u[i:i + T]); s.append(self.s[i:i + T]); s0.append(init)
        s0_true = torch.from_numpy(np.stack(s0))
        s0_hat = s0_true + init_noise * SCALE * torch.randn_like(s0_true)
        return (torch.from_numpy(np.stack(surf)).float(), torch.from_numpy(np.stack(u)),
                torch.from_numpy(np.stack(s)), s0_hat, s0_true)


@torch.no_grad()
def evaluate(model, val, T=1500):
    """Run the observer over each whole validation episode from a perturbed start."""
    model.eval()
    errs = []
    g = torch.Generator().manual_seed(0)
    for e in range(len(val.lens)):
        a, n = val.starts[e], val.lens[e]
        surf = torch.from_numpy(val.surf[a:a + n]).float()[None]
        u = torch.from_numpy(val.u[a:a + n])[None]
        s = torch.from_numpy(val.s[a:a + n])[None]
        s0 = torch.from_numpy(val.s0[e])[None] + 0.3 * SCALE * torch.randn(1, 4, generator=g)
        s_hat, _ = model(surf, u, s0)
        errs.append((s_hat - s).abs()[0, 50:].numpy())  # skip 50 ms convergence
    model.train()
    err = np.concatenate(errs)
    return dict(mae_x_mm=float(err[:, 0].mean() * 1e3), mae_xd=float(err[:, 1].mean()),
                mae_th_deg=float(np.degrees(err[:, 2].mean())), mae_thd_degs=float(np.degrees(err[:, 3].mean())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--extra", default=None, help="extra (e.g. DAgger) dataset mixed 50/50")
    ap.add_argument("--fixed_T", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(0); torch.manual_seed(0)
    train, val = Windows("data/train.npz"), Windows("data/val.npz")
    model = EventObserver()
    if a.init:
        model.load_state_dict(torch.load(a.init))
    extra = Windows(a.extra) if a.extra else None
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    os.makedirs("checkpoints", exist_ok=True)
    hist, t0, best = [], time.time(), 1e9
    if a.init:
        m = evaluate(model, val); best = m["mae_th_deg"] + m["mae_x_mm"] / 10
        print("init val:", m, flush=True)
    for step in range(1, a.steps + 1):
        T = a.fixed_T or (48 if step < a.steps * 0.4 else (128 if step < a.steps * 0.8 else 256))
        B = a.batch if T < 200 else max(4, a.batch // 2)
        src = extra if (extra is not None and step % 2 == 0) else train
        surf, u, s, s0_hat, s0_true = src.sample(rng, B, T, init_noise=0.3)
        s_hat, corr = model(surf, u, s0_hat)
        loss, parts = physics_informed_loss(s_hat, s, corr, s0_hat, model.dt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            print(f"step {step} T={T} loss {loss.item():.4f} {parts} {time.time()-t0:.0f}s", flush=True)
        if step % 500 == 0 or step == a.steps:
            m = evaluate(model, val); m["step"] = step; hist.append(m)
            print("  val:", m, flush=True)
            score = m["mae_th_deg"] + m["mae_x_mm"] / 10
            if score < best:
                best = score
                torch.save(model.state_dict(), "checkpoints/observer.pt")
    json.dump(hist, open("checkpoints/train_history.json", "w"), indent=1)
    print("done", time.time() - t0)
