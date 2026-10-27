"""V4.1 benchmark: does selective neural assistance beat the geometry-first system?

Configurations
  geo+sup            Baseline D (V4 test protocol, unchanged)
  geoev+sup          D + the evidence fix (Step 3), no network      <- isolates the fix
  v4+sup-s0          existing V4 (neural every ~2 ms)
  v4-b100+sup-s0     existing V4 capped at 100 neural calls/s
  v41+sup-s0         proposed V4.1: geometry-first + evidence fix + selective neural assistance
Conditions: nominal, occlusion, dropped_events, low_light, combo, ood_plant.

Seeds
  validation (design + supervisor tuning):  300-331     (never used for the final numbers)
  FINAL held-out test:                      4000+        (fresh: never used by V1-V4 or V4.1 design)
Trials (final): nominal 7 hits x 12; occlusion and combo 2 hits x 32; others 2 hits x 24.

Extra metrics vs V4: velocity-estimate error, false interventions (the supervisor intervened -
modified the command, dropped the soft target or used the stale fallback - on a tick where the
same supervisor fed the TRUE state with zero uncertainty would have passed the command
unchanged), per-tick estimator CPU time (indicative only: 2 workers share 2 cores), neural
calls/s, evidence-state shares.
"""
import os, sys, json, time
from multiprocessing import Pool
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import benchmark_v3 as B3
import benchmark_v4 as B4
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.safety import SafetyShield, ShieldedController
from aurasync.simulate import run_episode
from aurasync.v41.pipeline import EvidencePipeline, STATE_NAMES

OUT = "results/v41"
RAW = f"{OUT}/raw"
KICK_TICK, N_TICKS, SAFE_DEG = B3.KICK_TICK, B3.N_TICKS, B3.SAFE_DEG
CONDS = ["nominal", "occlusion", "dropped_events", "low_light", "combo", "ood_plant"]
CONFIGS = ["geo+sup", "geoev+sup", "v4+sup-s0", "v4-b100+sup-s0", "v41+sup-s0"]
SEED0 = int(os.environ.get("SEED0", 4000))


def v41_tuning():
    p = f"{OUT}/tuning_v41.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def build(cfg, overrides=None):
    base, _, sfx = cfg.rpartition("-s")
    seed = int(sfx) if (base and sfx.isdigit()) else None
    name = base if seed is not None else cfg
    if not (name.startswith("geoev") or name.startswith("v41")):
        return B4.build(cfg)
    p = PlantParams(); lqr = LQRController(p)
    cal, tu = B3.calibration(), B3.tuning()
    vt = v41_tuning()
    fam = "v41" if name.startswith("v41") else "geoev"
    prm = dict(vt.get("pipeline_by_family", {}).get(fam, {}))
    prm.update(overrides or {})
    sup_prm = vt.get("supervisor_by_family", {}).get(fam, B4.v4_tuning()["supervisor_by_family"]["geo"])
    if "+sup" in name:
        sup = lambda pipe: ShieldedController(lqr, SafetyShield(p, **sup_prm), pipe)
    else:
        sup = lambda pipe: lqr
    if fam == "v41":
        model = B3.event_model(f"event_s{seed}")
        pipe = EvidencePipeline(np.diag(cal["Q_diag"]), neural=model, temp=np.array(cal["temp"]["varhead"]),
                                latency_samples=cal["latency"], name=cfg, **prm)
    else:
        pipe = EvidencePipeline(np.diag(cal["Q_diag"]), name=cfg, **prm)
    return pipe, sup


class Timed:
    """Times the estimator update (pass-through otherwise)."""

    def __init__(self, p):
        object.__setattr__(self, "p", p); object.__setattr__(self, "dt", [])

    def __getattr__(self, k):
        return getattr(self.p, k)

    def __setattr__(self, k, v):
        setattr(self.p, k, v)

    def update(self, *a):
        t = time.perf_counter(); self.p.update(*a); self.dt.append(time.perf_counter() - t)

    def reset(self, s0):
        self.p.reset(s0)

    def estimate(self):
        return self.p.estimate()


class RecShield(ShieldedController):
    def reset(self):
        super().reset(); self.modes = []

    def act(self, est):
        st0 = self.stats["stale"]
        f = self.shield.filter
        mode = ["pass"]
        def rec(s, S, u):
            r = f(s, S, u); mode[0] = r[1]; return r
        self.shield.filter = rec
        u = super().act(est)
        self.shield.filter = f
        self.modes.append((mode[0] != "pass") or (self.stats["stale"] > st0))
        return u


def false_interventions(plant_nominal, sup_prm, s_log, s0, modes):
    """Reference: same supervisor on the TRUE state (zero covariance, no delay)."""
    sh = SafetyShield(plant_nominal, **sup_prm); lqr = LQRController(plant_nominal)
    Z = np.zeros((4, 4)); fi = 0
    prev = np.asarray(s0, float)
    for k, inter in enumerate(modes):
        _, ref = sh.filter(prev, Z, lqr.act(prev))
        if inter and ref == "pass":
            fi += 1
        if k < len(s_log):
            prev = s_log[k]
    return fi


def spec(cond_name, final=True):
    cond, plant, snap, kicks, n, fault = B4.scenario_spec(cond_name)
    if final:
        n = {"nominal": 12, "occlusion": 32, "combo": 32}.get(cond_name, 24)
    return cond, plant, snap, kicks, n, fault


def run_cfg(args):
    cond_name, cfg, seeds_from, n_override = args
    torch.set_num_threads(1)
    cond, plant, snap, kicks, n, fault = spec(cond_name)
    if n_override:
        n = n_override
    pipe, ctrl_f = build(cfg)
    sup_on = "+sup" in cfg
    fam = "v41" if cfg.startswith("v41") else ("geoev" if cfg.startswith("geoev") else ("geo" if cfg.startswith("geo") else "v4"))
    if fam in ("v41", "geoev"):
        sup_prm = v41_tuning().get("supervisor_by_family", {}).get(fam, B4.v4_tuning()["supervisor_by_family"]["geo"])
    else:
        sup_prm = B4.v4_tuning()["supervisor_by_family"]["geo" if fam == "geo" else "v4"]
    p_nom = PlantParams(); lqr = LQRController(p_nom)
    rows = []
    for kick in kicks:
        for i in range(n):
            seed = seeds_from + i
            rng = np.random.default_rng(seed); s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
            tp = Timed(pipe)
            ctrl = RecShield(lqr, SafetyShield(p_nom, **sup_prm), tp) if sup_on else lqr
            lg = run_episode(plant, ctrl, tp, N_TICKS, s0, kicks=[(KICK_TICK, kick * (1 if i % 2 == 0 else -1))],
                             snap=snap, cam_seed=seed, cond=cond)
            th = lg["s"][:, 2]
            peak = float(np.degrees(np.abs(th[KICK_TICK:]).max())) if len(th) > KICK_TICK else 90.0
            surv = lg["fail_tick"] is None
            err = np.abs(lg["s_hat"] - lg["s"])[50:]
            e_th, e_dth = np.degrees(err[:, 2]), np.degrees(err[:, 3])
            u = np.asarray(lg["u"])
            r = dict(kick=kick, trial=i, seed=seed, survived=surv, safe=bool(surv and peak <= SAFE_DEG), peak_deg=min(peak, 90.0),
                     est_th_deg=float(e_th.mean()), est_th_p95=float(np.percentile(e_th, 95)),
                     est_dth_degs=float(e_dth.mean()), est_x_mm=float(err[:, 0].mean() * 1e3),
                     frac_err_gt10=float((e_th > 10).mean()),
                     track_violation=bool(np.abs(lg["s"][:, 0]).max() >= plant.x_limit),
                     sat_frac=float((np.abs(u) >= 19.99).mean()) if len(u) else 0.0,
                     fail_tick=lg["fail_tick"], ticks=len(lg["s"]),
                     est_cpu_ms_mean=float(np.mean(tp.dt) * 1e3), est_cpu_ms_p99=float(np.percentile(tp.dt, 99) * 1e3))
            lgp = getattr(pipe, "log", {})
            r["calls_per_s"] = float(lgp.get("calls", 0) / (len(lg["s"]) * 1e-3))
            r["misses"] = int(lgp.get("misses", 0))
            if "states" in lgp:
                r["state_share"] = {STATE_NAMES[j]: c / max(sum(lgp["states"]), 1) for j, c in enumerate(lgp["states"])}
                r["triggers"] = dict(lgp["triggers"]); r["neural_accepted"] = lgp["accepted"]; r["neural_rejected"] = lgp["rejected"]
            if sup_on:
                r["interventions"] = float(np.mean(ctrl.modes))
                r["false_interventions"] = false_interventions(p_nom, sup_prm, lg["s"], s0, ctrl.modes) / max(len(ctrl.modes), 1)
            rows.append(r)
    return cond_name, cfg, rows


def run_and_save(args):
    cond_name, cfg = args[:2]
    path = f"{RAW}/{cond_name}/{cfg}.json"
    if os.path.exists(path):
        return path
    t0 = time.time()
    _, _, rows = run_cfg(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(cfg=cfg, cond=cond_name, seed0=args[2], rows=rows), open(path, "w"))
    m = lambda k: np.mean([r[k] for r in rows if r.get(k) is not None])
    print(f"{cond_name:15s} {cfg:16s} surv {m('survived'):.2f} safe {m('safe'):.2f} est {m('est_th_deg'):.2f} "
          f"dth {m('est_dth_degs'):.0f} calls/s {m('calls_per_s'):.0f} {time.time()-t0:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    which = os.environ.get("CONFIGS")
    cfgs = which.split(",") if which else CONFIGS
    jobs = [(c, g, SEED0, None) for c in CONDS for g in cfgs]
    jobs = [j for j in jobs if not os.path.exists(f"{RAW}/{j[0]}/{j[1]}.json")]
    print(len(jobs), "jobs", flush=True)
    with Pool(int(os.environ.get("PROCS", 2))) as pool:
        for _ in pool.imap_unordered(run_and_save, jobs):
            pass
    print("all done", flush=True)
