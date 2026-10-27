"""V4 benchmark: Detect -> Correct -> Recover, against the V3 baselines.

Baselines (spec section 9)
  A  ev-dense-s*            V3 event observer + LQR (dense, v2-style)
  B  ev-trig-s*             V3 event-triggered observer (its consistency monitor is diagnostic only)
  C  ev-trig+shield-s*      V3 with uncertainty + safety shield
  D  geo                    independent event-geometry estimator (physics + gated fusion, no network)
  E  frame120-s*            learned frame camera, 120 fps
  F  imu, oracle            IMU + encoder EKF / perfect state (references, not camera methods)
Proposed
  v4+sup-s*                 V4 pipeline (triggered, position-only neural fusion) + recovery-aware supervisor
Ablations
  v4-s*                     without the safety supervisor
  v4-nocorr+sup-s0          without state correction (detection + inflation only)
  v4-noinfl+sup-s0          without uncertainty feedback (no inflation; supervisor sees un-inflated covariance)
  v4-dense+sup-s0           without event-triggered scheduling
  eventonly-v4+sup-s0       event-only model (no APS channel)
  v4-ow+sup-s0              first V4 design: neural correction overwrites the state (incl. velocities)
  geo+sup                   Baseline D with the same supervisor
  v4-b250+sup-s0, v4-b100+sup-s0   Module E: neural-correction budget of 250 / 100 per second
Configurations identical to V3 (A, B, C, E, F) reuse V3's stored results on the same test
trials, except where V4 fixed the simulation (frame camera under 'ood_plant').
Supervisor parameters for the V4 family are tuned on validation seeds like every other family.

New scenarios (spec section 10, hits 1.5 and 3.0 rad/s, 8 trials each, test seeds 500+):
  sign_flip      at 60 ms after the hit the estimator's angle and rate are flipped (pole-angle sign error)
  conf_wrong     at 60 ms after the hit the estimate's angle is offset by 15 deg, covariance untouched
  stale_stream   the camera output freezes for 40 ms starting 10 ms after the hit
  miscalib       camera mounted 1.2 px off-centre and 4% scale error (estimators assume the nominal calibration)
  init_error     estimators start 5 deg / 5 cm away from the true initial state
  combo          low light + 5x event noise + unseen plant parameters together (never trained or tuned on)
"""
import os, sys, json, time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import benchmark_v3 as B3
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.conditions import PRESETS, NOMINAL, SensorConditions
from aurasync.safety import SafetyShield, ShieldedController
from aurasync.simulate import run_episode
from aurasync.v4.pipeline import V4Pipeline
from aurasync.v4.geo_pipeline import GeoPipeline

OUT = "results/v4"
RAW4 = f"{OUT}/raw"
KICK_TICK, N_TICKS, SAFE_DEG = B3.KICK_TICK, B3.N_TICKS, B3.SAFE_DEG
NEW_SCENARIOS = ["sign_flip", "conf_wrong", "stale_stream", "miscalib", "init_error", "combo"]


def v4_tuning():
    p = f"{OUT}/tuning_v4.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def build(cfg):
    """V4 / geometry configurations; everything else is delegated to the V3 builder."""
    base, _, sfx = cfg.rpartition("-s")
    seed = int(sfx) if (base and sfx.isdigit()) else None
    name = base if seed is not None else cfg
    if not (name.startswith("v4") or name.startswith("eventonly-v4") or name.startswith("geo")):
        return B3.build(cfg)
    p = PlantParams(); lqr = LQRController(p)
    cal, tu = B3.calibration(), B3.tuning()
    vt = v4_tuning()
    fam = "geo" if name.startswith("geo") else "v4"
    sup_prm = vt.get("supervisor_by_family", {}).get(fam, tu["shield_by_family"]["event"])
    if "+sup" in name:            # full supervisor: delay compensation + stale fallback + CBF shield
        sup = lambda pipe: ShieldedController(lqr, SafetyShield(p, **sup_prm), pipe)
    elif "+fb" in name:           # recovery-aware fallback only: delay compensation + stale fallback, no CBF filter
        sup = lambda pipe: ShieldedController(lqr, SafetyShield(p, **sup_prm), pipe, enabled=False)
    else:
        sup = lambda pipe: lqr
    if name.startswith("geo"):
        return GeoPipeline(Q=np.diag(cal["Q_diag"])), sup
    eo = name.startswith("eventonly")
    model = B3.event_model(f"eventonly_s{seed}" if eo else f"event_s{seed}")
    temp = np.array(cal["temp"]["eventonly" if eo else "varhead"])
    mode = "dense" if "-dense" in name else "triggered"
    kw = dict(temp=temp, Q=np.diag(cal["Q_diag"]))
    if mode == "triggered":
        kw.update(latency_samples=cal["latency"], **tu["trigger"])
        if "-b" in name:                                   # Module E budget: '-b100' = at most 100 corrections/s
            budget = int(name.split("-b")[1].split("+")[0].split("-")[0])
            kw["min_gap"] = int(np.ceil(1000 / budget))
    v = dict(correct="-nocorr" not in name, inflate="-noinfl" not in name,
             # V4 default (chosen on validation): the network contributes position only ('kalman_pos').
             # '-ow' = first V4 design (V3-style overwrite), '-kf' = full-state Kalman weighting (ablations)
             neural_fusion="overwrite" if "-ow" in name else ("kalman" if "-kf" in name else "kalman_pos"))
    return V4Pipeline(model, mode=mode, name=cfg, **kw, **v), sup


# ---- scenario helpers -----------------------------------------------------------------------
class Scenario:
    """Wraps a pipeline to inject estimator faults / sensor stalls without touching its code."""

    def __init__(self, pipe, kind, rng):
        self.p, self.kind, self.rng = pipe, kind, rng
        for a in ("needs_true_plant",):
            if hasattr(pipe, a):
                setattr(self, a, getattr(pipe, a))

    def __getattr__(self, k):
        return getattr(self.p, k)

    def __setattr__(self, k, v):
        if k in ("p", "kind", "rng", "frozen", "needs_true_plant"):
            object.__setattr__(self, k, v)
        elif k == "p_true":
            setattr(self.p, k, v)
        else:
            object.__setattr__(self, k, v)

    def reset(self, s0):
        s0 = np.asarray(s0, float).copy()
        if self.kind == "init_error":
            s0 = s0 + np.array([0.05, 0, np.radians(5), 0]) * self.rng.choice([-1, 1])
        self.frozen = None
        self.p.reset(s0)

    def _state_ref(self):
        for attr in ("s", "x"):
            if hasattr(self.p, attr) and isinstance(getattr(self.p, attr), np.ndarray):
                return attr
        return None

    def update(self, k, t, s_true, surface, u):
        if self.kind == "stale_stream" and KICK_TICK + 10 <= k < KICK_TICK + 50:
            if self.frozen is None:
                self.frozen = (s_true.copy(), surface.copy())
            s_true, surface = self.frozen
        self.p.update(k, t, s_true, surface, u)
        if self.kind in ("sign_flip", "conf_wrong") and k == KICK_TICK + 60:
            a = self._state_ref()
            if hasattr(self.p, "job"):
                self.p.job = None          # cancel any in-flight neural correction (same treatment for V3 and V4)
            if a is not None:
                st = getattr(self.p, a)
                if self.kind == "sign_flip":
                    st[2], st[3] = -st[2], -st[3]
                else:
                    st[2] += np.radians(15) * (1 if st[2] >= 0 else -1)
                if hasattr(self.p, "out"):
                    self.p.out[-1] = (st[:4].copy(), self.p.out[-1][1])
                if hasattr(self.p, "buf") and len(getattr(self.p, "buf")) and isinstance(self.p.buf[-1], np.ndarray):
                    self.p.buf[-1] = st[:4].copy()

    def estimate(self):
        return self.p.estimate()


def scenario_spec(name):
    """-> (sensor conditions, plant, snap, kicks, trials, fault kind)"""
    n = int(os.environ.get("STRESS_TRIALS", 8))
    if name in ("sign_flip", "conf_wrong", "stale_stream", "init_error"):
        return NOMINAL, PlantParams(), None, [1.5, 3.0], n, name
    if name == "miscalib":
        return SensorConditions("miscalib", x_offset_m=0.03, scale=1.04), PlantParams(), None, [1.5, 3.0], n, None
    if name == "combo":
        return (SensorConditions("combo", brightness=0.05, ev_noise_mult=5.0), B3.OOD_PLANT, None, [1.5, 3.0], n, None)
    cond, plant, snap, kicks, trials = B3.cond_spec(name)
    return cond, plant, snap, kicks, (trials if name == "nominal" else n), None


def run_one(args):
    cond_name, cfg = args
    path = f"{RAW4}/{cond_name}/{cfg}.json"
    if os.path.exists(path):
        return path
    torch.set_num_threads(1)
    cond, plant, snap, kicks, n, fault = scenario_spec(cond_name)
    pipe, ctrl_f = build(cfg)
    rows, t0 = [], time.time()
    for kick in kicks:
        for i in range(n):
            seed = (800 if cond_name == "snap" else 500) + i
            rng = np.random.default_rng(seed); s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
            run_pipe = Scenario(pipe, fault, np.random.default_rng(seed + 7)) if fault else pipe
            ctrl = ctrl_f(run_pipe)
            log = run_episode(plant, ctrl, run_pipe, N_TICKS, s0, kicks=[(KICK_TICK, kick * (1 if i % 2 == 0 else -1))],
                              snap=snap, cam_seed=seed, cond=cond)
            th = log["s"][:, 2]
            peak = float(np.degrees(np.abs(th[KICK_TICK:]).max())) if len(th) > KICK_TICK else 90.0
            surv = log["fail_tick"] is None
            err = np.abs(log["s_hat"] - log["s"])[50:]
            e_deg = np.degrees(err[:, 2])
            wrong_sign = (np.sign(log["s_hat"][50:, 2]) != np.sign(log["s"][50:, 2])) & (np.abs(np.degrees(log["s"][50:, 2])) > 3)
            u = np.asarray(log["u"])
            r = dict(kick=kick, trial=i, survived=surv, safe=bool(surv and peak <= SAFE_DEG), peak_deg=min(peak, 90.0),
                     est_th_deg=float(e_deg.mean()), est_th_median=float(np.median(e_deg)), est_th_max=float(e_deg.max()),
                     frac_err_gt10=float((e_deg > 10).mean()), wrong_sign=float(wrong_sign.mean()),
                     est_x_mm=float(err[:, 0].mean() * 1e3), effort=float(np.mean(u ** 2)) if len(u) else 0.0,
                     sat_frac=float((np.abs(u) >= 19.99).mean()) if len(u) else 0.0,
                     max_du=float(np.abs(np.diff(u)).max()) if len(u) > 1 else 0.0,
                     track_violation=bool(np.abs(log["s"][:, 0]).max() >= plant.x_limit),
                     fail_tick=log["fail_tick"])
            inner = run_pipe.p if isinstance(run_pipe, Scenario) else run_pipe
            if isinstance(inner, V4Pipeline):
                dl = np.array(inner.det_log, dtype=float) if inner.det_log else np.zeros((1, 5))
                alarm = dl[:, 1] > 0
                r.update(n_fuse=int(dl[:, 2].sum()), n_reinit=int(dl[:, 3].sum()),
                         n_inflate=sum(1 for e in inner.ev_log if e["action"] == "inflate"),
                         first_alarm=int(np.argmax(alarm)) if alarm.any() else None, valid_meas=float(dl[:, 4].mean()),
                         reinit_ticks=[e["k"] for e in inner.ev_log if e["action"] == "reinit"],
                         reasons=sorted({x for e in inner.ev_log for x in e.get("reasons", [])}))
            if hasattr(inner, "log") and isinstance(inner.log, dict) and "calls" in inner.log:
                lg = inner.log; ticks = max(len(log["s"]), 1)
                r.update(calls_per_s=lg["calls"] / ticks * 1000, misses=lg["misses"], miss_rate=lg["misses"] / max(lg["calls"], 1))
            if isinstance(ctrl, ShieldedController):
                st = ctrl.stats; tk = max(st["ticks"], 1)
                r.update(shield_modified=(st["modified"] + st["soft_dropped"]) / tk, shield_infeasible=st["infeasible"] / tk,
                         shield_stale=st["stale"] / tk)
            rows.append(r)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(cond=cond_name, cfg=cfg, rows=rows, seconds=time.time() - t0), open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    print(f"{cond_name:16s} {cfg:26s} surv {np.mean([r['survived'] for r in rows]):.2f} safe {np.mean([r['safe'] for r in rows]):.2f} "
          f"est {np.mean([r['est_th_deg'] for r in rows]):.2f} {time.time()-t0:.0f}s", flush=True)
    return path


V4_SEEDED_3 = ["v4+sup", "v4"]
NOMINAL_CONFIGS = ([f"{f}-s{i}" for f in V4_SEEDED_3 for i in range(3)]
                   + ["v4-nocorr+sup-s0", "v4-noinfl+sup-s0", "v4-dense+sup-s0", "eventonly-v4+sup-s0", "v4-ow+sup-s0",
                      "v4-b250+sup-s0", "v4-b100+sup-s0", "geo", "geo+sup"])
STRESS_CONFIGS = ["v4+sup-s0", "v4-s0", "eventonly-v4+sup-s0", "geo", "geo+sup"]
REDO_V3 = [("ood_plant", f"frame120-s{i}") for i in range(3)] + [("ood_plant", "frame30-s0")]   # simulation fix
NEW_CONFIGS = ["ev-dense-s0", "ev-trig-s0", "ev-trig+shield-s0", "geo", "geo+sup", "frame120-s0", "imu",
               "v4+sup-s0", "v4-s0", "v4-nocorr+sup-s0"]


if __name__ == "__main__":
    os.makedirs(RAW4, exist_ok=True)
    which = os.environ.get("WHICH", "all")
    jobs = []
    if which in ("all", "nominal"):
        jobs += [("nominal", c) for c in NOMINAL_CONFIGS]
    if which in ("all", "stress"):
        jobs += [(cn, c) for cn in B3.STRESS for c in STRESS_CONFIGS] + REDO_V3
    if which in ("all", "new"):
        jobs += [(cn, c) for cn in NEW_SCENARIOS for c in NEW_CONFIGS]
    jobs = [j for j in jobs if not os.path.exists(f"{RAW4}/{j[0]}/{j[1]}.json")]
    print(len(jobs), "jobs", flush=True)
    with Pool(int(os.environ.get("PROCS", 2))) as pool:
        for _ in pool.imap_unordered(run_one, jobs):
            pass
    print("all done", flush=True)
