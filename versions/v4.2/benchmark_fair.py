"""Phase 2 benchmark: learned event stack vs LEARNED frame stack.

Phase 1 handed the frame baselines the true state. Here every frame pipeline
must estimate the state from blurred, noisy frames with FrameEstimator, then
roll it forward through physics to cover its latency.

Pipelines (same plant, controller, seeds and hits as phase 1):
  oracle                       perfect instantaneous state (upper bound)
  aurasync-event / -2ms        event observer, 1 tick and 2 ticks compute delay
                               (2 ms covers its measured 1.8 ms median CPU cost)
  learned-frame-{fps}fps       FrameEstimator + physics latency compensation, 4 ms exposure
  learned-frame-{fps}fps-zoh   same estimator, naive hold (ablation)
  learned-frame-{fps}fps-exp1ms  1 ms exposure, i.e. near-sharp frames (ablation)
  frame-{fps}fps               phase-1 ground-truth frame baseline (reference)
Latency budgets (capture -> estimate available): 120 fps 8 ms, 60 fps 15 ms, 30 fps 25 ms.
"""
import os, sys, json, time
from multiprocessing import Pool
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import replace
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.frame_camera import FrameEstimator, LearnedFramePipeline, N_FRAMES, cond_vector
from aurasync.simulate import OraclePipeline, FramePipeline, EventPipeline, run_episode
from benchmark import load_model, trial, KICK_TICK, N_TICKS, SAFE_DEG

OUT = "results"
LAT = {120: 0.008, 60: 0.015, 30: 0.025}
LAT_HONEST = {120: 0.012}
KICKS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def load_frame_model():
    m = FrameEstimator()
    m.load_state_dict(torch.load("checkpoints/frame_estimator.pt"))
    return m.eval()


def make_pipeline(spec):
    kind, fps = spec
    if kind == "oracle":
        return OraclePipeline()
    if kind.startswith("event"):
        pipe = EventPipeline(load_model(), delay_ticks=1 if kind == "event" else 2)
        if kind == "event2":
            pipe.name = "aurasync-event-2ms"
        return pipe
    if kind == "gt":
        return FramePipeline(fps, LAT[fps])
    fm = load_frame_model()
    if kind == "learned":
        return LearnedFramePipeline(fm, fps, LAT[fps])
    if kind == "zoh":
        return LearnedFramePipeline(fm, fps, LAT[fps], mode="zoh")
    if kind == "sharp":
        return LearnedFramePipeline(fm, fps, LAT[fps], exposure=0.001)
    if kind == "honest":  # budget that also covers the estimator's measured CPU cost (~6 ms median)
        pipe = LearnedFramePipeline(fm, fps, LAT_HONEST[fps])
        pipe.name += f"-{LAT_HONEST[fps]*1e3:.0f}ms"
        return pipe
    raise ValueError(kind)


SPECS = [("oracle", 0), ("event", 0), ("event2", 0)] + \
        [(k, f) for k in ("learned", "zoh", "sharp", "gt") for f in (120, 60, 30)] + [("honest", 120)]


def est_error(log):
    """Closed-loop estimation error (tilt, deg) from 50 ms after start."""
    e = np.abs(log["s_hat"] - log["s"])[50:]
    return float(np.degrees(e[:, 2].mean())), float(e[:, 0].mean() * 1e3)


def run_spec(args):
    spec, n_trials = args
    torch.set_num_threads(1)
    pipe = make_pipeline(spec)
    p = PlantParams(); ctrl = LQRController(p)
    sweep = []
    for kick in KICKS:
        for i in range(n_trials):
            k = kick * (1 if i % 2 == 0 else -1)
            r, log = trial(p, ctrl, pipe, k, seed=500 + i)
            r["est_th_deg"], r["est_x_mm"] = est_error(log)
            r.update(kick=kick, seed=500 + i)
            sweep.append(r)
    broken = replace(p, m=p.m * 1.8)
    snap = []
    for i in range(n_trials):
        r, log = trial(p, ctrl, pipe, 2.0 * (1 if i % 2 == 0 else -1), seed=800 + i, snap=(KICK_TICK, broken))
        snap.append(r)
    print(f"{pipe.name:30s} done", flush=True)
    return pipe.name, dict(sweep=sweep, snap=snap)


def summarise(rows):
    out = []
    for kick in KICKS:
        t = [r for r in rows if r["kick"] == kick]
        out.append(dict(kick=kick, survival=float(np.mean([r["survived"] for r in t])),
                        safe=float(np.mean([r["safe"] for r in t])),
                        peak_deg=float(np.mean([r["peak_deg"] for r in t]))))
    return out


def paired_ci(a, b, n_boot=5000, seed=0):
    """Mean of (a - b) over paired trials with a 95% bootstrap CI."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


@torch.inference_mode()
def frame_latency(n=2000):
    torch.set_num_threads(1)
    m = load_frame_model()
    x = torch.rand(1, N_FRAMES, 32, 96); c = cond_vector(120, 0.004)
    ts = []
    for i in range(n + 100):
        a = time.perf_counter(); m(x, c); b = time.perf_counter()
        if i >= 100: ts.append(b - a)
    return dict(median_ms=float(np.median(ts) * 1e3), p99_ms=float(np.percentile(ts, 99) * 1e3),
                params=sum(p.numel() for p in m.parameters()))


# ---- plots ------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_EVENT, C_LEARN, C_GT, C_ORACLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def style(ax):
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#bdbcb7")
    ax.tick_params(colors=INK2, labelsize=8)


def plot_fair(summ):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    for col, fps in enumerate((30, 60, 120)):
        series = [("oracle", "Oracle (perfect state)", C_ORACLE, "--", "o"),
                  ("aurasync-event", "AuraSync event (1 ms)", C_EVENT, "-", "o"),
                  ("aurasync-event-2ms", "AuraSync event (2 ms)", C_EVENT, ":", "s"),
                  (f"learned-frame-{fps}fps", f"Learned frame {fps} fps", C_LEARN, "-", "o"),
                  (f"frame-{fps}fps", f"Frame {fps} fps, true state (phase 1)", C_GT, "-", "^")]
        if fps == 120 and "learned-frame-120fps-12ms" in summ:
            series.append(("learned-frame-120fps-12ms", "Learned frame 120 fps, 12 ms (incl. compute)", C_LEARN, ":", "s"))
        for row, (key, ylab) in enumerate((("safe", f"safe recoveries (%, peak ≤ {SAFE_DEG:.0f}°)"),
                                           ("peak_deg", "mean peak tilt (deg)"))):
            ax = axes[row, col]; style(ax)
            for name, label, colr, ls, mk in series:
                y = [r[key] * (100 if key == "safe" else 1) for r in summ[name]]
                ax.plot(KICKS, y, ls=ls, color=colr, lw=2, marker=mk, ms=6, label=label,
                        markeredgecolor="white", markeredgewidth=1.2)
            if key == "peak_deg":
                ax.axhline(SAFE_DEG, color=INK2, lw=0.8, ls=(0, (2, 3)))
                ax.set_ylim(0, 50); ax.set_xlabel("impulse hit to pole (rad/s)", color=INK2, fontsize=9)
            else:
                ax.set_ylim(-4, 104); ax.set_title(f"{fps} fps frame camera ({LAT[fps]*1e3:.0f} ms latency)",
                                                   color=INK, fontsize=10, loc="left")
            if col == 0: ax.set_ylabel(ylab, color=INK2, fontsize=9)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    labels = [l.replace("120 fps", "(panel fps)").replace("(panel fps), 12 ms", "120 fps, 12 ms") for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Event stack vs learned frame stack: same plant, controller, seeds and hits (8 trials per point)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    fig.savefig(f"{OUT}/fair_recovery_sweep.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n_trials = int(os.environ.get("TRIALS", 8))
    t0 = time.time()
    only = os.environ.get("ONLY")  # e.g. ONLY=honest:120 -> run that spec, reuse the rest from disk
    specs = [s for s in SPECS if f"{s[0]}:{s[1]}" in only.split(",")] if only else SPECS
    raw = json.load(open(f"{OUT}/benchmark_fair.json"))["raw"] if only else {}
    with Pool(2) as pool:
        raw.update(dict(pool.map(run_spec, [(s, n_trials) for s in specs], chunksize=1)))
    print(f"sweep {time.time()-t0:.0f}s")
    summ = {k: summarise(v["sweep"]) for k, v in raw.items()}
    overall = {k: dict(survival=float(np.mean([r["survived"] for r in v["sweep"]])),
                       safe=float(np.mean([r["safe"] for r in v["sweep"]])),
                       peak_deg=float(np.mean([r["peak_deg"] for r in v["sweep"]])),
                       est_th_deg=float(np.mean([r["est_th_deg"] for r in v["sweep"]])),
                       est_x_mm=float(np.mean([r["est_x_mm"] for r in v["sweep"]])),
                       snap_survival=float(np.mean([r["survived"] for r in v["snap"]])),
                       snap_safe=float(np.mean([r["safe"] for r in v["snap"]])),
                       snap_peak_deg=float(np.mean([r["peak_deg"] for r in v["snap"]])))
               for k, v in raw.items()}
    # paired comparison: event (1 ms) minus each frame stack, per (kick, seed)
    ev = [r["peak_deg"] for r in raw["aurasync-event"]["sweep"]]
    paired = {k: paired_ci(ev, [r["peak_deg"] for r in v["sweep"]]) for k, v in raw.items() if k != "aurasync-event"}
    ev_s = [float(r["survived"]) for r in raw["aurasync-event"]["sweep"]]
    paired_surv = {k: paired_ci(ev_s, [float(r["survived"]) for r in v["sweep"]]) for k, v in raw.items() if k != "aurasync-event"}
    out = dict(frame_estimator_latency=frame_latency(), overall=overall, sweep=summ,
               paired_peak_diff_event_minus_other=paired,
               paired_survival_diff_event_minus_other=paired_surv,
               config=dict(trials_per_point=n_trials, kicks=KICKS, safe_envelope_deg=SAFE_DEG,
                           frame_latency_budget_s=LAT, default_exposure_s=0.004, read_noise=0.02),
               raw=raw)
    json.dump(out, open(f"{OUT}/benchmark_fair.json", "w"), indent=1, default=float)
    plot_fair(summ)
    print(json.dumps(out["frame_estimator_latency"]))
    for k, v in overall.items():
        print(f"{k:32s} " + " ".join(f"{kk}={vv:.2f}" for kk, vv in v.items()))
    for k, v in paired.items():
        w = paired_surv[k]
        print(f"event - {k:28s} peak diff {v[0]:+.1f} deg [{v[1]:+.1f}, {v[2]:+.1f}]   "
              f"survival diff {w[0]*100:+.0f} pts [{w[1]*100:+.0f}, {w[2]*100:+.0f}]")
