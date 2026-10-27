"""AuraSync benchmark: event pipeline vs frame-camera pipelines.

1. Latency  - wall-clock cost of one perception tick on this CPU (single thread).
2. Recovery - sweep impulse 'hits' to the pole and measure, per pipeline:
              survival, safety-envelope violations (peak tilt > 12 deg),
              peak tilt, and settle time.
3. Snap     - mid-run mechanical failure: pole mass x1.8 plus a hit.
All pipelines drive the same plant with the same 1 kHz LQR controller.
"""
import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import replace
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.estimator import EventObserver
from aurasync.event_camera import SceneRenderer, EventCamera, TimeSurfaceEncoder
from aurasync.simulate import run_episode, OraclePipeline, FramePipeline, EventPipeline

torch.set_num_threads(1)
OUT = "results"
SAFE_DEG = 12.0
KICK_TICK = 200
N_TICKS = 1200


def load_model():
    m = EventObserver()
    m.load_state_dict(torch.load("checkpoints/observer.pt"))
    return m.eval()


def pipelines(model):
    return [OraclePipeline(), EventPipeline(model, delay_ticks=1),
            FramePipeline(120, 0.008), FramePipeline(60, 0.015), FramePipeline(30, 0.025)]


@torch.inference_mode()
def measure_latency(model, n=3000):
    r = SceneRenderer(); enc = TimeSurfaceEncoder(r.H, r.W)
    enc.t_pos[:] = np.random.default_rng(0).uniform(-0.05, 0, enc.t_pos.shape)
    enc.t_neg[:] = enc.t_pos
    s = torch.zeros(1, 4); h = model.init_hidden(1); u = torch.zeros(1)
    t_enc, t_net = [], []
    for i in range(n + 200):
        a = time.perf_counter()
        surf = torch.from_numpy(np.concatenate([enc.tensor(0.0), r.render(np.zeros(4), 0.25)[None].astype(np.float32)]))[None]
        b = time.perf_counter()
        z = model.enc(surf); s, h, _ = model.cell(z, h, s, u)
        c = time.perf_counter()
        if i >= 200:
            t_enc.append(b - a); t_net.append(c - b)
    tot = np.array(t_enc) + np.array(t_net)
    q = lambda x, p: float(np.percentile(x, p) * 1e3)
    return dict(encode_median_ms=q(t_enc, 50), observer_median_ms=q(t_net, 50),
                total_median_ms=q(tot, 50), total_p99_ms=q(tot, 99),
                fits_1khz_budget_fraction=float((tot < 1e-3).mean()),
                params=sum(p.numel() for p in model.parameters()))


def settle_time(theta, start, tol=np.radians(1.0)):
    out = np.abs(theta[start:]) > tol
    if not out.any():
        return 0.0
    last = np.nonzero(out)[0][-1]
    return None if last >= len(out) - 50 else (last + 1) * 1e-3


def trial(p, ctrl, pipe, kick, seed, snap=None):
    rng = np.random.default_rng(seed)
    s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
    log = run_episode(p, ctrl, pipe, N_TICKS, s0, kicks=[(KICK_TICK, kick)], snap=snap,
                      cam_seed=seed)
    th = log["s"][:, 2]
    peak = float(np.degrees(np.abs(th[KICK_TICK:]).max())) if len(th) > KICK_TICK else 90.0
    survived = log["fail_tick"] is None
    return dict(survived=survived, safe=survived and peak <= SAFE_DEG, peak_deg=min(peak, 90.0),
                settle_s=settle_time(th, KICK_TICK) if survived else None), log


def recovery_sweep(model, kicks, n_trials):
    p = PlantParams(); ctrl = LQRController(p)
    res = {}
    for pipe in pipelines(model):
        rows = []
        for kick in kicks:
            trials = [trial(p, ctrl, pipe, kick * (1 if i % 2 == 0 else -1), seed=500 + i)[0]
                      for i in range(n_trials)]
            settles = [t["settle_s"] for t in trials if t["settle_s"] is not None]
            rows.append(dict(kick=kick,
                             survival=np.mean([t["survived"] for t in trials]),
                             safe=np.mean([t["safe"] for t in trials]),
                             peak_deg=np.mean([t["peak_deg"] for t in trials]),
                             settle_s=float(np.mean(settles)) if settles else None))
            print(f"{pipe.name:16s} kick {kick:.2f}: surv {rows[-1]['survival']:.2f} safe {rows[-1]['safe']:.2f} "
                  f"peak {rows[-1]['peak_deg']:.1f}deg", flush=True)
        res[pipe.name] = rows
    return res


def snap_test(model, n_trials):
    p = PlantParams(); ctrl = LQRController(p)
    broken = replace(p, m=p.m * 1.8)
    res = {}
    for pipe in pipelines(model):
        tr = [trial(p, ctrl, pipe, 2.0 * (1 if i % 2 == 0 else -1), seed=800 + i,
                    snap=(KICK_TICK, broken))[0] for i in range(n_trials)]
        res[pipe.name] = dict(survival=float(np.mean([t["survived"] for t in tr])),
                              safe=float(np.mean([t["safe"] for t in tr])),
                              peak_deg=float(np.mean([t["peak_deg"] for t in tr])))
        print("snap", pipe.name, res[pipe.name], flush=True)
    return res


COLORS = {"oracle": "#888888", "aurasync-event": "#d6336c", "frame-120fps": "#1c7ed6",
          "frame-60fps": "#4dabf7", "frame-30fps": "#a5d8ff"}


def plot_sweep(res, kicks):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in res.items():
        ls = "--" if name == "oracle" else "-"
        ax[0].plot(kicks, [r["safe"] * 100 for r in rows], ls, marker="o", color=COLORS[name], label=name, lw=2)
        ax[1].plot(kicks, [r["peak_deg"] for r in rows], ls, marker="o", color=COLORS[name], label=name, lw=2)
    ax[0].set(xlabel="impulse hit to pole (rad/s)", ylabel=f"% recoveries within {SAFE_DEG:.0f}° envelope",
              title="Safe-recovery rate")
    ax[1].axhline(SAFE_DEG, color="k", lw=0.8, ls=":")
    ax[1].set(xlabel="impulse hit to pole (rad/s)", ylabel="mean peak tilt (deg)", title="Peak excursion after hit",
              ylim=(0, 50))
    ax[0].legend(); [a.grid(alpha=0.3) for a in ax]
    fig.tight_layout(); fig.savefig(f"{OUT}/recovery_sweep.png", dpi=140); plt.close(fig)


def plot_trajectories(model, kick):
    p = PlantParams(); ctrl = LQRController(p)
    fig, ax = plt.subplots(figsize=(9, 4))
    for pipe in pipelines(model):
        _, log = trial(p, ctrl, pipe, kick, seed=42)
        t = np.arange(len(log["s"])) * 1e-3
        ax.plot(t, np.degrees(log["s"][:, 2]), color=COLORS[pipe.name], lw=2 if "event" in pipe.name else 1.3,
                ls="--" if pipe.name == "oracle" else "-", label=pipe.name)
    ax.axvline(KICK_TICK * 1e-3, color="k", lw=0.8); ax.text(KICK_TICK * 1e-3 + 0.005, 1, "hit", fontsize=9)
    ax.axhspan(-SAFE_DEG, SAFE_DEG, color="green", alpha=0.06)
    ax.set(xlabel="time (s)", ylabel="pole tilt (deg)", title=f"Recovery from a {kick} rad/s hit", ylim=(-50, 50))
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/trajectories.png", dpi=140); plt.close(fig)


def plot_events_and_tracking(model):
    """Event raster + time surface + observer tracking on one closed-loop run."""
    p = PlantParams(); ctrl = LQRController(p)
    r = SceneRenderer()
    pipe = EventPipeline(model)
    log = run_episode(p, ctrl, pipe, 700, [0, 0, 0.02, 0], kicks=[(200, 2.0)], record=True, cam_seed=7)
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    ax[0, 0].imshow(r.render(log["s"][230], p.l), cmap="gray"); ax[0, 0].set_title("What a frame camera sees (t=230 ms)")
    surf = log["surf"][230].astype(np.float32)
    rgb = np.stack([surf[0], 0.25 * surf[2], surf[1]], -1)
    ax[0, 1].imshow(np.clip(rgb, 0, 1)); ax[0, 1].set_title("Event time surface (red=ON, blue=OFF, fading = older)")
    for a in ax[0]: a.axis("off")
    t = np.arange(len(log["s"])) * 1e-3
    ax[1, 0].plot(t, np.degrees(log["s"][:, 2]), "k", lw=1.5, label="true tilt")
    ax[1, 0].plot(t, np.degrees(log["s_hat"][:, 2]), color=COLORS["aurasync-event"], lw=1.2, ls="--", label="AuraSync estimate")
    ax[1, 0].set(xlabel="time (s)", ylabel="deg", title="Observer tracking (closed loop)"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
    ax[1, 1].plot(t, log["events"] * 1e3 / 1e3, color="#5c7cfa")
    ax[1, 1].set(xlabel="time (s)", ylabel="k events / s", title="Event rate: silence when still, burst on impact"); ax[1, 1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/events_and_tracking.png", dpi=140); plt.close(fig)
    err = np.abs(log["s_hat"] - log["s"])[50:]
    return dict(closed_loop_mae_th_deg=float(np.degrees(err[:, 2].mean())),
                closed_loop_mae_x_mm=float(err[:, 0].mean() * 1e3),
                mean_event_rate_keps=float(log["events"].mean()),
                peak_event_rate_keps=float(log["events"].max()))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n_trials = int(os.environ.get("TRIALS", 8))
    kicks = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    model = load_model()
    out = {"latency": measure_latency(model)}
    print("latency", out["latency"], flush=True)
    out["events"] = plot_events_and_tracking(model)
    print("events", out["events"], flush=True)
    out["sweep"] = recovery_sweep(model, kicks, n_trials)
    plot_sweep(out["sweep"], kicks)
    plot_trajectories(model, 2.5)
    out["snap"] = snap_test(model, n_trials)
    out["config"] = dict(trials_per_point=n_trials, safe_envelope_deg=SAFE_DEG,
                         frame_assumptions="true pose+velocity at capture; latency = readout+inference",
                         event_assumptions="1 ms (one tick) compute delay; learned estimate, no ground truth")
    json.dump(out, open(f"{OUT}/benchmark.json", "w"), indent=1, default=float)
    print("saved", OUT)
