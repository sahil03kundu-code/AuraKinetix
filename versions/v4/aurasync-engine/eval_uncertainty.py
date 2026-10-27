"""Upgrade A evaluation: is the event observer's confidence trustworthy?

Open-loop study: held-out episodes (seeds 7000+, never used for training,
validation or tuning) are driven by a perfect-state controller with
exploration noise and impulse hits, recorded once per condition, then replayed
into every estimator. Because control never depends on the estimate here,
every method sees exactly the same sensor data.

Methods
  const        constant sigma = validation RMSE of the dense single model (no uncertainty model)
  varhead      dense single model (seed 0), learned variance head, temperature-calibrated on validation
  ensemble     dense 3-seed deep ensemble: mean of members; var = mean(var_i) + spread (calibrated)
  spread       same ensemble, epistemic spread only
  triggered    seed 0 under the event-triggered scheduler: variance head at corrections,
               physics-propagated covariance in between (filter-style)
  eventonly    dense event-only ablation (no APS frame), variance head (calibrated)
  monitor      varhead sigma inflated by the event-consistency monitor (model-free check
               against raw events): sigma^2 + (beta * max(0, m - m0))^2, beta = 20 deg,
               m0 = 95th percentile of m on validation episodes
  imu-ekf      IMU + encoder EKF covariance (reference, different sensor)

Metrics on tilt (theta): MAE, Gaussian NLL, coverage of the 1-sigma / 2-sigma
intervals (ideal 68.3% / 95.4%), expected calibration error over 10 levels,
false-confidence rate (error > 3 sigma AND > 2 deg), and AUROC of sigma for
flagging unreliable ticks (error > 3 deg). Subsets: quiet ticks (< 3 pixels
fired in the last ms), bursts (>= 30), occluded ticks.
"""
import os, sys, json, time
from dataclasses import replace
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.estimator import ProbEventObserver, SCALE
from aurasync.conditions import PRESETS, NOMINAL
from aurasync.scheduler import TriggeredEventPipeline, FRESH
from aurasync.imu import IMUPipeline
from aurasync.monitor import monitor_trajectory
from aurasync.simulate import run_episode, OraclePipeline
from generate_data import episode_setup
from train import Windows

torch.set_num_threads(1)
OUT = "results/v3"
CAL = "checkpoints/v3/calibration.json"
SCALE_NP = SCALE.numpy().astype(np.float64)
OOD_PLANT = PlantParams(M=1.45, m=0.155, l=0.29)      # outside the training randomisation (+-30%, +-10%)
LEVELS = np.linspace(0.1, 0.9, 9)
OCC_STARTS = (0.25, 0.65, 1.05)


def load_obs(name):
    m = ProbEventObserver(use_aps=not name.startswith("eventonly"))
    m.load_state_dict(torch.load(f"checkpoints/v3/{name}.pt"))
    return m.eval()


# ---------------------------------------------------------------------------------------------
def replay(pipe, log, s0, p_true=None):
    """Feed recorded data into a pipeline; return per-tick estimate (at the tick's own time) and sigma."""
    pipe.reset(np.asarray(s0, float))
    est, sig = [], []
    for k in range(len(log["s"])):
        if p_true is not None:
            pipe.p_true = p_true[k]
        pipe.update(k, (k + 1) * 1e-3, log["s"][k], log["surf"][k].astype(np.float32), log["u"][k])
        if isinstance(pipe, IMUPipeline):
            e, S = pipe.x[:4], pipe.P[:4, :4]
        else:
            e, S = pipe.out[-1]
        est.append(np.array(e)); sig.append(np.sqrt(np.clip(np.diag(S), 1e-16, None)))
    return np.array(est), np.array(sig)


MON_BETA = np.radians(20.0)


def calibrate_monitor(models, cal):
    """m0 = 95th percentile of the monitor score on validation episodes (healthy tracking)."""
    val = Windows("data/val.npz")
    ms = []
    for e in range(len(val.lens)):
        a, n = val.starts[e], val.lens[e]
        log = dict(s=np.asarray(val.s[a:a + n], float), surf=val.surf[a:a + n], u=np.asarray(val.u[a:a + n], float))
        est, _ = replay(TriggeredEventPipeline(models["event_s0"], mode="dense", temp=np.array(cal["temp"]["varhead"])),
                        log, np.asarray(val.s0[e], float))
        ms.append(monitor_trajectory(est, np.asarray(log["surf"], np.float32))[50:])
    cal["monitor"] = dict(m0=float(np.percentile(np.concatenate(ms), 95)), beta=float(MON_BETA))
    json.dump(cal, open(CAL, "w"), indent=1)
    print("monitor m0", round(cal["monitor"]["m0"], 3), flush=True)
    return cal


def make_methods(models, cal):
    s0, ens, eo = models["event_s0"], [models[f"event_s{i}"] for i in range(3)], models["eventonly_s0"]
    T = {k: np.array(v) for k, v in cal["temp"].items()}
    return {
        "varhead": lambda: TriggeredEventPipeline(s0, mode="dense", temp=T["varhead"]),
        "ensemble": lambda: TriggeredEventPipeline(ens, mode="dense", temp=T["ensemble"]),
        "spread": lambda: TriggeredEventPipeline(ens, mode="dense", temp=np.full(4, 1e-6)),
        "triggered": lambda: TriggeredEventPipeline(s0, mode="triggered", temp=T["varhead"],
                                                    Q=np.diag(cal["Q_diag"]), latency_samples=cal["latency"]),
        "eventonly": lambda: TriggeredEventPipeline(eo, mode="dense", temp=T["eventonly"]),
    }


def calibrate(models):
    """Fit per-state temperatures (and the constant-sigma baseline) on the VALIDATION episodes,
    then the process-noise scale of the triggered filter. Never touches test data."""
    val = Windows("data/val.npz")
    res = dict(temp={}, const_sigma=None)
    eps = []
    for e in range(len(val.lens)):
        a, n = val.starts[e], val.lens[e]
        eps.append(dict(log=dict(s=np.asarray(val.s[a:a + n], float), surf=val.surf[a:a + n],
                                 u=np.asarray(val.u[a:a + n], float)), s0=np.asarray(val.s0[e], float)))
    ens = [models[f"event_s{i}"] for i in range(3)]
    for key, mk in (("varhead", lambda: TriggeredEventPipeline(models["event_s0"], mode="dense")),
                    ("ensemble", lambda: TriggeredEventPipeline(ens, mode="dense")),
                    ("eventonly", lambda: TriggeredEventPipeline(models["eventonly_s0"], mode="dense"))):
        Z, E = [], []
        for ep in eps:
            est, sig = replay(mk(), ep["log"], ep["s0"])
            err = (est - ep["log"]["s"])[50:]
            Z.append(err / sig[50:]); E.append(err)
        Z, E = np.concatenate(Z), np.concatenate(E)
        res["temp"][key] = np.sqrt((Z ** 2).mean(0)).tolist()
        if key == "varhead":
            res["const_sigma"] = np.sqrt((E ** 2).mean(0)).tolist()
        print("calibrated", key, np.round(res["temp"][key], 2), flush=True)
    res["latency"] = json.load(open(f"{OUT}/latency.json"))["correction_samples_s"]
    best = None
    for q in (1.0, 1e2, 1e4, 1e6):
        Qd = (np.array([1e-9, 1e-5, 1e-8, 1e-4]) * q).tolist()
        nll = []
        for ep in eps[:3]:
            pipe = TriggeredEventPipeline(models["event_s0"], mode="triggered", temp=np.array(res["temp"]["varhead"]),
                                          Q=np.diag(Qd), latency_samples=res["latency"])
            est, sig = replay(pipe, ep["log"], ep["s0"])
            e, s = (est - ep["log"]["s"])[50:, 2], sig[50:, 2]
            nll.append(np.mean(0.5 * (e / s) ** 2 + np.log(s)))
        print("Q scale", q, "theta NLL", round(float(np.mean(nll)), 3), flush=True)
        if best is None or np.mean(nll) < best[0]:
            best = (np.mean(nll), Qd)
    res["Q_diag"] = best[1]
    json.dump(res, open(CAL, "w"), indent=1)
    return res


# ---------------------------------------------------------------------------------------------
CONDITIONS = ["nominal", "low_light", "hdr", "event_noise", "thresh_mismatch", "dropped_events",
              "occlusion", "ood_plant", "snap", "fast"]


def record_episode(cond_name, seed, n_ticks=1500):
    p, ctrl, explore, s0, kicks, _ = episode_setup(seed, n_ticks)
    cond = PRESETS.get(cond_name, NOMINAL)
    snap = None
    if cond_name == "ood_plant":
        p = OOD_PLANT
    if cond_name == "snap":
        snap = (700, replace(p, m=p.m * 1.8))
    if cond_name == "fast":
        rng = np.random.default_rng(seed + 1)
        kicks = [(k, np.sign(v) * rng.uniform(3.0, 5.0)) for k, v in kicks]
    if cond_name == "occlusion":       # several 100 ms occlusion windows across the episode
        cond = replace(cond, occlusion=tuple((t, t + 0.1) for t in OCC_STARTS))
    log = run_episode(p, LQRController(PlantParams()), OraclePipeline(), n_ticks, s0, kicks=kicks,
                      exploration=lambda: 0.5 * explore(), record=True, cam_seed=seed, cond=cond,
                      snap=snap, stop_on_fail=True)
    plants = [p if (snap is None or k < snap[0]) else snap[1] for k in range(len(log["s"]))]
    occ = np.zeros(len(log["s"]), bool)
    if cond_name == "occlusion":
        tt = (np.arange(len(log["s"])) + 1) * 1e-3
        for t0 in OCC_STARTS:
            occ |= (tt >= t0) & (tt < t0 + 0.1)
    return log, s0, plants, occ


def metrics(err, sig, mask=None):
    e, s = np.abs(err), sig
    if mask is not None:
        e, s = e[mask], s[mask]
    if len(e) < 20:
        return None
    z = e / s
    cov = [(z < norm.ppf(0.5 + lv / 2)).mean() for lv in LEVELS]
    bad = e > np.radians(3)
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(bad, s)) if 0 < bad.mean() < 1 else None
    except Exception:
        auroc = _auroc(bad, s)
    return dict(n=int(len(e)), mae_deg=float(np.degrees(e.mean())), nll=float(np.mean(0.5 * z ** 2 + np.log(s) + 0.919)),
                cov1=float((z < 1).mean()), cov2=float((z < 2).mean()),
                ece=float(np.mean(np.abs(np.array(cov) - LEVELS))),
                false_conf=float(((z > 3) & (e > np.radians(2))).mean()),
                frac_bad=float(bad.mean()), auroc=auroc, coverage_curve=[float(c) for c in cov])


MCOL = {"const": "#8a8984", "varhead": "#2a78d6", "monitor": "#eda100", "ensemble": "#eb6834", "spread": "#e87ba4",
        "triggered": "#1baf7a", "eventonly": "#6250d6", "imu-ekf": "#008300"}


def plot_uncertainty(summary, methods):
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    ev = [m for m in methods if m != "imu-ekf"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    for ax in axes:
        ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(colors=ink2, labelsize=8)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=ink2, lw=1, ls=(0, (2, 3)))
    for m in ev:
        stress = [summary[f"{c}|{m}"]["all"] for c in CONDITIONS if c != "nominal"]
        w = np.array([x["n"] for x in stress]); cc = np.array([x["coverage_curve"] for x in stress])
        ax.plot(LEVELS, (cc * w[:, None]).sum(0) / w.sum(), "-o", color=MCOL[m], lw=2, ms=4, label=m)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_title("Reliability under stress (pooled, tilt)", fontsize=10, color=ink, loc="left")
    ax.set_xlabel("nominal coverage of the predicted interval", fontsize=8, color=ink2)
    ax.set_ylabel("observed coverage (diagonal = calibrated)", fontsize=8, color=ink2)
    ax.legend(fontsize=7.5, frameon=False)
    x = np.arange(len(CONDITIONS)); wdt = 0.8 / len(ev)
    for ax, key, ttl in ((axes[1], "false_conf", "False confidence (error > 3σ and > 2°)"),
                         (axes[2], "auroc", "AUROC: σ flags ticks with error > 3°")):
        for j, m in enumerate(ev):
            vals = [summary[f"{c}|{m}"]["all"][key] for c in CONDITIONS]
            vals = [np.nan if v is None else v * (100 if key == "false_conf" else 1) for v in vals]
            ax.bar(x + (j - len(ev) / 2 + 0.5) * wdt, vals, wdt * 0.92, color=MCOL[m], label=m)
        ax.set_xticks(x); ax.set_xticklabels([c.replace("_", " ") for c in CONDITIONS], fontsize=7, color=ink,
                                             rotation=35, ha="right")
        ax.set_title(ttl, fontsize=10, color=ink, loc="left")
    axes[1].set_ylabel("% of ticks", fontsize=8, color=ink2)
    axes[2].axhline(0.5, color=ink2, lw=0.8, ls=(0, (2, 3))); axes[2].set_ylim(0.3, 1.0)
    axes[2].set_ylabel("AUROC (0.5 = chance)", fontsize=8, color=ink2)
    fig.suptitle("Upgrade A: is the event observer's confidence trustworthy? (held-out episodes, seeds 7000+)",
                 x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout(); fig.savefig(f"{OUT}/v3_uncertainty.png", dpi=140); plt.close(fig)


def _auroc(y, s):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n_eps = int(os.environ.get("EPS", 4))
    names = ["event_s0", "event_s1", "event_s2", "eventonly_s0"]
    models = {n: load_obs(n) for n in names}
    cal = json.load(open(CAL)) if os.path.exists(CAL) and not os.environ.get("RECAL") else calibrate(models)
    if "monitor" not in cal:
        cal = calibrate_monitor(models, cal)
    methods = make_methods(models, cal)
    const_sig = np.array(cal["const_sigma"])
    raw_path = f"{OUT}/uncertainty_raw.npz"
    store = dict(np.load(raw_path, allow_pickle=True)) if os.path.exists(raw_path) else {}
    for cond in CONDITIONS:
        for i in range(n_eps):
            key = f"{cond}|{i}"
            if f"{key}|truth" in store:
                continue
            t0 = time.time()
            log, s0, plants, occ = record_episode(cond, 7000 + i)
            store[f"{key}|truth"] = log["s"]
            store[f"{key}|events"] = (log["surf"][:, 0] >= FRESH).sum((1, 2)) + (log["surf"][:, 1] >= FRESH).sum((1, 2))
            store[f"{key}|occ"] = occ
            for mname, mk in methods.items():
                est, sig = replay(mk(), log, s0)
                store[f"{key}|{mname}|est"], store[f"{key}|{mname}|sig"] = est, sig
            est, sig = store[f"{key}|varhead|est"], store[f"{key}|varhead|sig"].copy()
            mm = monitor_trajectory(est, log["surf"].astype(np.float32))
            sig[:, 2] = np.sqrt(sig[:, 2] ** 2 + (cal["monitor"]["beta"] * np.maximum(0, mm - cal["monitor"]["m0"])) ** 2)
            store[f"{key}|monitor|est"], store[f"{key}|monitor|sig"], store[f"{key}|monitor|m"] = est, sig, mm
            est, sig = replay(IMUPipeline(), log, s0, p_true=plants)
            store[f"{key}|imu-ekf|est"], store[f"{key}|imu-ekf|sig"] = est, sig
            np.savez(raw_path, **store)
            print(f"{cond} ep{i} {len(log['s'])} ticks {time.time()-t0:.0f}s", flush=True)
    # ---- metrics -------------------------------------------------------------------------
    all_methods = ["const", "varhead", "monitor", "ensemble", "spread", "triggered", "eventonly", "imu-ekf"]
    summary = {}
    for cond in CONDITIONS:
        for mname in all_methods:
            E, S, Q, B, O = [], [], [], [], []
            for i in range(n_eps):
                key = f"{cond}|{i}"
                truth = store[f"{key}|truth"]
                src = "varhead" if mname == "const" else mname
                est = store[f"{key}|{src}|est"]
                sig = np.broadcast_to(const_sig, est.shape) if mname == "const" else store[f"{key}|{src}|sig"]
                ev = store[f"{key}|events"]
                sl = slice(50, None)
                E.append((est - truth)[sl, 2]); S.append(sig[sl, 2])
                Q.append(ev[sl] < 3); B.append(ev[sl] >= 30); O.append(store[f"{key}|occ"][sl])
            E, S, Q, B, O = map(np.concatenate, (E, S, Q, B, O))
            summary[f"{cond}|{mname}"] = dict(all=metrics(E, S), quiet=metrics(E, S, Q), burst=metrics(E, S, B),
                                               occluded=metrics(E, S, O) if cond == "occlusion" else None)
    json.dump(dict(summary=summary, calibration={k: v for k, v in cal.items() if k != "latency"},
                   conditions=CONDITIONS, methods=all_methods, episodes=n_eps),
              open(f"{OUT}/uncertainty.json", "w"), indent=1)
    plot_uncertainty(summary, all_methods)
    for cond in CONDITIONS:
        print(f"--- {cond}")
        for mname in all_methods:
            m = summary[f"{cond}|{mname}"]["all"]
            print(f"  {mname:10s} mae {m['mae_deg']:5.2f} nll {m['nll']:7.2f} cov1 {m['cov1']:.2f} cov2 {m['cov2']:.2f} "
                  f"ece {m['ece']:.3f} falseconf {m['false_conf']:.3f} auroc {m['auroc'] if m['auroc'] is None else round(m['auroc'], 3)}")
