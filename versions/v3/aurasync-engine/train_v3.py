"""Train the v3 confidence-aware, gap-aware event observer (Upgrades A + C, D seeds).

Same data, step budget, learning-rate schedule and checkpoint-selection rule as
the v2 observer, plus:
  * Gaussian NLL on the variance head (detached residuals -> mean training unchanged)
  * random correction skipping (half the sequences dense, half with skip prob
    U(0, 0.85)) so the model works when corrections are event-triggered
  * --seed for independent runs, --no_aps for the event-only ablation
"""
import argparse, os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from aurasync.estimator import ProbEventObserver, physics_informed_loss, gaussian_nll, SCALE
from train import Windows


@torch.no_grad()
def evaluate(model, val, skip=0.0):
    """Whole validation episodes from a perturbed start. Returns MAE plus
    calibration of the tilt variance (NLL, 1-sigma / 2-sigma coverage)."""
    model.eval()
    g = torch.Generator().manual_seed(0)
    errs, zs, lvs = [], [], []
    for e in range(len(val.lens)):
        a, n = val.starts[e], val.lens[e]
        surf = torch.from_numpy(val.surf[a:a + n]).float()[None]
        u = torch.from_numpy(val.u[a:a + n])[None]
        s = torch.from_numpy(val.s[a:a + n])[None]
        s0 = torch.from_numpy(val.s0[e])[None] + 0.3 * SCALE * torch.randn(1, 4, generator=g)
        sp = None if skip == 0 else torch.tensor([skip])
        s_hat, _, lv, mask = model(surf, u, s0, skip_p=sp, gen=g)
        err = (s_hat - s)[0, 50:]
        errs.append(err.abs().numpy())
        m = mask[0, 50:]
        zs.append((err[m] / SCALE / torch.exp(0.5 * lv[0, 50:][m])).numpy())
        lvs.append(lv[0, 50:][m].numpy())
    model.train()
    err = np.concatenate(errs); z = np.concatenate(zs)
    return dict(mae_x_mm=float(err[:, 0].mean() * 1e3), mae_xd=float(err[:, 1].mean()),
                mae_th_deg=float(np.degrees(err[:, 2].mean())), mae_thd_degs=float(np.degrees(err[:, 3].mean())),
                th_cov1=float((np.abs(z[:, 2]) < 1).mean()), th_cov2=float((np.abs(z[:, 2]) < 2).mean()),
                th_z_rms=float(np.sqrt((z[:, 2] ** 2).mean())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_aps", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--w_nll", type=float, default=0.1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    train, val = Windows("data/train.npz"), Windows("data/val.npz")
    model = ProbEventObserver(use_aps=not a.no_aps)
    print("params:", sum(p.numel() for p in model.parameters()), "seed", a.seed, "aps", not a.no_aps, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    os.makedirs("checkpoints/v3", exist_ok=True)
    ckpt = f"checkpoints/v3/{a.name}.pt"
    hist, t0, best, start = [], time.time(), 1e9, 1
    state_path = f"checkpoints/v3/{a.name}_state.pt"
    if os.path.exists(state_path):      # resume after the 2 h background-job limit
        st = torch.load(state_path, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]; gen.set_state(st["gen"]); torch.set_rng_state(st["torch"])
        hist, best, start = st["hist"], st["best"], st["step"] + 1
        t0 -= st["elapsed"]
        print("resumed at step", start, flush=True)
    for step in range(start, a.steps + 1):
        T = 48 if step < a.steps * 0.4 else (128 if step < a.steps * 0.8 else 256)
        B = a.batch if T < 200 else max(4, a.batch // 2)
        surf, u, s, s0_hat, _ = train.sample(rng, B, T, init_noise=0.3)
        dense = torch.rand(B, generator=gen) < 0.5
        skip_p = torch.where(dense, torch.zeros(B), torch.rand(B, generator=gen) * 0.85)
        s_hat, corr, lv, mask = model(surf, u, s0_hat, skip_p=skip_p, gen=gen)
        loss, parts = physics_informed_loss(s_hat, s, corr, s0_hat, model.dt)
        nll = gaussian_nll(s_hat, s, lv, mask)
        loss = loss + a.w_nll * nll
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            print(f"step {step} T={T} loss {loss.item():.4f} nll {nll.item():.3f} {parts} {time.time()-t0:.0f}s", flush=True)
        if step % 500 == 0 or step == a.steps:
            m = evaluate(model, val); m["step"] = step
            m["sparse"] = evaluate(model, val, skip=0.7)
            hist.append(m)
            print("  val:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "sparse"}),
                  "| sparse th", round(m["sparse"]["mae_th_deg"], 2), flush=True)
            score = m["mae_th_deg"] + m["mae_x_mm"] / 10      # same selection rule as v2
            if score < best:
                best = score
                torch.save(model.state_dict(), ckpt)
            json.dump(hist, open(f"checkpoints/v3/{a.name}_history.json", "w"), indent=1)
        if step % 50 == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            rng=rng.bit_generator.state, gen=gen.get_state(), torch=torch.get_rng_state(),
                            hist=hist, best=best, step=step, elapsed=time.time() - t0), state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
    print("done", time.time() - t0, "best", best, flush=True)
