"""v3 benchmark: confidence-aware event observer + safety shield + event-triggered scheduling,
multi-seed, against learned frame cameras, an IMU+encoder EKF and an oracle, under stress conditions.

Protocol (Upgrade D)
  * training seeds      event 0/1/2, frame 0/1/2, event-only 0 (identical data and selection rule)
  * validation          training-time checkpoint selection: data/val episodes (seeds 9000-9005)
                        temperature + filter calibration: same validation episodes
                        shield / trigger tuning: closed-loop seeds 300-315 (tune_v3.py)
  * test                closed-loop seeds 500+ (hits) and 800+ (snap), open-loop seeds 7000+
                        (eval_uncertainty.py). Never used for any choice.
  * parameter regime    'ood_plant' uses masses/length outside the training randomisation
Conditions: nominal (7 hit levels x 16 trials) and 9 stress conditions (2 hit levels x 8 trials).
Each (condition, configuration) result is written to results/v3/raw/ as soon as it
finishes, so the run can be resumed after the 2 h job limit.
"""
import os, sys, json, time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from aurasync.physics import PlantParams, linearise, rk4_step
from aurasync.controller import LQRController
from aurasync.estimator import ProbEventObserver
from aurasync.frame_camera import FrameEstimator, LearnedFramePipeline
from aurasync.conditions import PRESETS, NOMINAL
from aurasync.scheduler import TriggeredEventPipeline
from aurasync.safety import SafetyShield, ShieldedController
from aurasync.imu import IMUPipeline
from aurasync.simulate import run_episode, OraclePipeline

OUT = "results/v3"
RAW = f"{OUT}/raw"
SAFE_DEG, KICK_TICK, N_TICKS = 12.0, 200, 1200
LAT = {120: 0.008, 60: 0.015, 30: 0.025}
OOD_PLANT = PlantParams(M=1.45, m=0.155, l=0.29)
NOMINAL_KICKS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
STRESS = ["low_light", "hdr", "event_noise", "thresh_mismatch", "dropped_events", "occlusion",
          "ood_plant", "snap", "fast"]


def cond_spec(name):
    """-> (sensor conditions, plant, snap, kicks, trials per kick)"""
    if name == "nominal":
        return NOMINAL, PlantParams(), None, NOMINAL_KICKS, int(os.environ.get("TRIALS", 16))
    n = int(os.environ.get("STRESS_TRIALS", 8))
    if name == "ood_plant":
        return NOMINAL, OOD_PLANT, None, [1.5, 3.0], n
    if name == "snap":
        p = PlantParams()
        return NOMINAL, p, (KICK_TICK, replace(p, m=p.m * 1.8)), [1.5, 3.0], n
    if name == "fast":
        return NOMINAL, PlantParams(), None, [4.0, 5.0], n
    return PRESETS[name], PlantParams(), None, [1.5, 3.0], n


# ---- model loading --------------------------------------------------------------------------
_cache = {}


def event_model(name):
    if name not in _cache:
        m = ProbEventObserver(use_aps=not name.startswith("eventonly"))
        m.load_state_dict(torch.load(f"checkpoints/v3/{name}.pt")); _cache[name] = m.eval()
    return _cache[name]


def frame_model(seed):
    key = f"frame_s{seed}"
    if key not in _cache:
        m = FrameEstimator(); m.load_state_dict(torch.load(f"checkpoints/v3/{key}.pt")); _cache[key] = m.eval()
    return _cache[key]


def tuning():
    return json.load(open(f"{OUT}/tuning.json"))


def calibration():
    return json.load(open("checkpoints/v3/calibration.json"))


class FrameWithCov(LearnedFramePipeline):
    """Learned frame pipeline that also reports a covariance: validation error at capture,
    propagated through the linearised physics since capture (filter-style), so the
    shield treats frame and event estimates the same way."""

    def __init__(self, *a, sigma=None, **kw):
        super().__init__(*a, **kw)
        self.sig0 = np.diag(np.square(sigma))
        self.A, _ = linearise(self.p, 1e-3)
        self.Qp = np.diag([1e-9, 1e-5, 1e-8, 1e-4])

    def reset(self, s0):
        super().reset(s0); self.Sigma = self.sig0.copy(); self.t_cap = 0.0

    def update(self, k, t, s_true, surface, u):
        n_before = len(self.pending)
        arrivals = [x for x in self.pending if x[0] <= t + 1e-9]
        super().update(k, t, s_true, surface, u)
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.Qp
        if arrivals:
            Sg = self.sig0.copy()
            for _ in range(int(round((t - arrivals[-1][1]) / 1e-3))):
                Sg = self.A @ Sg @ self.A.T + self.Qp
            self.Sigma = Sg

    def covariance(self):
        return self.Sigma


def frame_sigma(seed, fps):
    """Per-state sigma from the frame model's validation MAE (Gaussian: sigma = 1.2533 * MAE)."""
    h = json.load(open(f"checkpoints/v3/frame_s{seed}_history.json"))
    best = min(h, key=lambda r: r["score"])["val"][str(fps)]
    mae = np.array([best["mae_x_mm"] / 1e3, best["mae_xd"], np.radians(best["mae_th_deg"]), np.radians(best["mae_thd_degs"])])
    return 1.2533 * mae


# ---- configurations: name -> (pipeline factory, controller factory) ---------------------------
def build(cfg):
    p = PlantParams(); lqr = LQRController(p)
    tu = tuning(); cal = calibration()
    fam = ("oracle" if cfg.startswith("oracle") else "imu" if cfg.startswith("imu")
           else "frame" if cfg.startswith("frame") else "event")
    prm = tu["shield_by_family"][fam]                     # each family's own validation-tuned shield
    plain = lambda pipe: lqr
    shielded = lambda pipe: ShieldedController(lqr, SafetyShield(p, **prm), pipe)
    lat = cal["latency"]; T = np.array(cal["temp"]["varhead"]); Tens = np.array(cal["temp"]["ensemble"])
    Teo = np.array(cal["temp"]["eventonly"]); Qd = np.diag(cal["Q_diag"]); trig = tu["trigger"]
    base, _, sfx = cfg.rpartition("-s")
    seed = int(sfx) if sfx.isdigit() and base else None
    name = base if seed is not None else cfg
    ctrl = shielded if "+shield" in name else plain
    kind = name.replace("+shield", "")
    if kind == "oracle":
        return OraclePipeline(), ctrl
    if kind == "imu":
        return IMUPipeline(), ctrl
    if kind == "ens-dense":
        return TriggeredEventPipeline([event_model(f"event_s{i}") for i in range(3)], mode="dense", temp=Tens), ctrl
    if kind.startswith("eventonly"):
        m = event_model(f"eventonly_s{seed}"); mode = kind.split("-")[1]
        kw = dict(latency_samples=lat, Q=Qd, **trig) if mode == "trig" else {}
        return TriggeredEventPipeline(m, mode={"trig": "triggered"}.get(mode, mode), temp=Teo, **kw), ctrl
    if kind.startswith("ev-"):
        m = event_model(f"event_s{seed}"); mode = kind[3:]
        if mode == "dense":
            return TriggeredEventPipeline(m, mode="dense", temp=T), ctrl
        if mode == "trig":
            return TriggeredEventPipeline(m, mode="triggered", temp=T, latency_samples=lat, Q=Qd, **trig), ctrl
        if mode == "periodic":
            return TriggeredEventPipeline(m, mode="periodic", period=tu["matched_period"], temp=T,
                                          latency_samples=lat, Q=Qd, deadline=trig["deadline"]), ctrl
        if mode == "greedy":
            return TriggeredEventPipeline(m, mode="greedy", temp=T, latency_samples=lat, Q=Qd,
                                          deadline=trig["deadline"]), ctrl
    if kind.startswith("frame"):
        fps = int(kind[5:])
        return FrameWithCov(frame_model(seed), fps, LAT[fps], sigma=frame_sigma(seed, fps)), ctrl
    raise ValueError(cfg)


# Compute budget (2 CPU cores): all 3 training seeds for the main families, seed 0 for ablations.
SEEDED_3 = ["ev-dense", "ev-dense+shield", "ev-trig", "ev-trig+shield", "frame120", "frame30"]
SEEDED_1 = ["ev-periodic+shield", "frame60", "frame120+shield", "frame30+shield"]
NOMINAL_CONFIGS = (["oracle", "oracle+shield", "imu", "imu+shield", "ens-dense+shield",
                    "eventonly-dense-s0", "eventonly-trig+shield-s0"]
                   + [f"{f}-s{i}" for f in SEEDED_3 for i in range(3)] + [f"{f}-s0" for f in SEEDED_1])
STRESS_CONFIGS = (["imu", "eventonly-dense-s0", "ev-trig-s0", "frame30-s0"]
                  + [f"{f}-s{i}" for f in ("ev-dense", "frame120") for i in range(3)])
PLANT_CONDITIONS = ("ood_plant", "snap", "fast")      # the oracle is only affected by plant changes


def run_one(args):
    cond_name, cfg = args
    path = f"{RAW}/{cond_name}/{cfg}.json"
    if os.path.exists(path):
        return path
    torch.set_num_threads(1)
    cond, plant, snap, kicks, n = cond_spec(cond_name)
    pipe, ctrl_f = build(cfg)
    rows, t0 = [], time.time()
    for kick in kicks:
        for i in range(n):
            seed = (800 if cond_name == "snap" else 500) + i
            rng = np.random.default_rng(seed)
            s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
            ctrl = ctrl_f(pipe)
            log = run_episode(plant, ctrl, pipe, N_TICKS, s0, kicks=[(KICK_TICK, kick * (1 if i % 2 == 0 else -1))],
                              snap=snap, cam_seed=seed, cond=cond)
            th = log["s"][:, 2]
            peak = float(np.degrees(np.abs(th[KICK_TICK:]).max())) if len(th) > KICK_TICK else 90.0
            surv = log["fail_tick"] is None
            err = np.abs(log["s_hat"] - log["s"])[50:]
            r = dict(kick=kick, trial=i, survived=surv, safe=bool(surv and peak <= SAFE_DEG), peak_deg=min(peak, 90.0),
                     est_th_deg=float(np.degrees(err[:, 2].mean())), est_x_mm=float(err[:, 0].mean() * 1e3),
                     effort=float(np.mean(np.square(log["u"]))) if len(log["u"]) else 0.0)
            if isinstance(ctrl, ShieldedController):
                st = ctrl.stats; tk = max(st["ticks"], 1)
                r.update(shield_modified=(st["modified"] + st["soft_dropped"]) / tk, shield_soft_dropped=st["soft_dropped"] / tk, shield_infeasible=st["infeasible"] / tk,
                         shield_stale=st["stale"] / tk, ctrl_p50_ms=float(np.median(ctrl.times) * 1e3),
                         ctrl_p99_ms=float(np.percentile(ctrl.times, 99) * 1e3), ctrl_max_ms=float(np.max(ctrl.times) * 1e3))
            if isinstance(pipe, TriggeredEventPipeline):
                lg = pipe.log; ticks = max(len(log["s"]), 1)
                r.update(calls_per_s=lg["calls"] / ticks * 1000, misses=lg["misses"],
                         miss_rate=lg["misses"] / max(lg["calls"], 1), triggers=dict(lg["triggers"]))
            rows.append(r)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(cond=cond_name, cfg=cfg, rows=rows, seconds=time.time() - t0), open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    print(f"{cond_name:16s} {cfg:28s} surv {np.mean([r['survived'] for r in rows]):.2f} "
          f"safe {np.mean([r['safe'] for r in rows]):.2f} {time.time()-t0:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    which = os.environ.get("WHICH", "all")
    jobs = []
    if which in ("all", "nominal"):
        jobs += [("nominal", c) for c in NOMINAL_CONFIGS]
    if which in ("all", "stress"):
        jobs += [(cn, c) for cn in STRESS for c in STRESS_CONFIGS]
        jobs += [(cn, "oracle") for cn in PLANT_CONDITIONS]
    jobs = [j for j in jobs if not os.path.exists(f"{RAW}/{j[0]}/{j[1]}.json")]
    print(len(jobs), "jobs to run", flush=True)
    with Pool(int(os.environ.get("PROCS", 2))) as pool:
        for _ in pool.imap_unordered(run_one, jobs):
            pass
    print("all done", flush=True)
