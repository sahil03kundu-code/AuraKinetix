"""Validation-only tuning for v3 (never touches test seeds 500+/800+/7000+).

Stage 0  latency.json      correction / frame-model / always-on tick cost on this CPU (1 thread)
Stage 1  calibration.json  temperatures, constant sigma, filter process noise (eval_uncertainty.calibrate)
Stage 2  shield            same grid for every family (event, frame, IMU, oracle) -> each family's best
Stage 3  trigger           event-triggered scheduler grid with the event shield; matched periodic budget
Objective (validation seeds 300-315, hits 1-4 rad/s): survival + 0.5 * safe - 0.002 * mean peak tilt.
"""
import os, sys, json, itertools, time
from multiprocessing import Pool
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.safety import SafetyShield, ShieldedController
from aurasync.scheduler import TriggeredEventPipeline, measure_correction_latency, physics_tick_cost
from aurasync.imu import IMUPipeline
from aurasync.simulate import run_episode, OraclePipeline
import benchmark_v3 as B

OUT = "results/v3"
VAL_SEEDS = list(range(300, 304))
VAL_KICKS = [1.0, 2.0, 3.0, 4.0]
SHIELD_GRID = [dict(gamma=g, kappa=k, stale_deg=s) for g, k, s in itertools.product((0.02, 0.03, 0.05), (0.0, 1.0, 2.0), (4.0, 8.0))]
TRIG_GRID = [dict(burst_thr=b, sigma_trig_deg=sg, max_gap=mg, margin_trig_deg=7.0, deadline=6)
             for b, sg, mg in itertools.product((8, 16), (1.0, 2.0), (10, 25))]


def score(rows):
    return float(np.mean([r[0] for r in rows]) + 0.5 * np.mean([r[0] and r[1] <= 12 for r in rows])
                 - 0.002 * np.mean([min(r[1], 46) for r in rows]))


def val_run(make_pipe, make_ctrl):
    p = PlantParams(); rows, calls = [], []
    for kick in VAL_KICKS:
        for sd in VAL_SEEDS:
            rng = np.random.default_rng(sd); s0 = [rng.uniform(-.1, .1), 0, rng.uniform(-.03, .03), 0]
            pipe = make_pipe(); ctrl = make_ctrl(pipe)
            lg = run_episode(p, ctrl, pipe, 1000, s0, kicks=[(200, kick * (1 if sd % 2 else -1))], cam_seed=sd)
            pk = float(np.degrees(np.abs(lg["s"][200:, 2]).max())) if len(lg["s"]) > 200 else 90.0
            rows.append((lg["fail_tick"] is None, pk))
            if hasattr(pipe, "log"):
                calls.append(pipe.log["calls"] / len(lg["s"]) * 1000)
    return rows, (float(np.mean(calls)) if calls else None)


def shield_job(args):
    fam, i = args
    torch.set_num_threads(1)
    prm = SHIELD_GRID[i]; p = PlantParams(); lqr = LQRController(p)
    cal = B.calibration()
    if fam == "event":
        mk = lambda: TriggeredEventPipeline(B.event_model("event_s0"), mode="dense", temp=np.array(cal["temp"]["varhead"]))
    elif fam == "frame":
        mk = lambda: B.FrameWithCov(B.frame_model(0), 120, B.LAT[120], sigma=B.frame_sigma(0, 120))
    elif fam == "imu":
        mk = IMUPipeline
    else:
        mk = OraclePipeline
    rows, _ = val_run(mk, lambda pipe: ShieldedController(lqr, SafetyShield(p, **prm), pipe))
    return fam, i, score(rows), rows


def trig_job(args):
    i, shield_prm = args
    torch.set_num_threads(1)
    p = PlantParams(); lqr = LQRController(p); cal = B.calibration()
    mk = lambda: TriggeredEventPipeline(B.event_model("event_s0"), mode="triggered", temp=np.array(cal["temp"]["varhead"]),
                                        latency_samples=cal["latency"], Q=np.diag(cal["Q_diag"]), **TRIG_GRID[i])
    rows, calls = val_run(mk, lambda pipe: ShieldedController(lqr, SafetyShield(p, **shield_prm), pipe))
    return i, score(rows), rows, calls


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    stage = os.environ.get("STAGE", "all")
    if stage in ("all", "latency"):
        torch.set_num_threads(1)
        m = B.event_model("event_s0")
        lat = measure_correction_latency(m, n=3000)
        fm = B.frame_model(0); x = torch.rand(1, 3, 32, 96)
        from aurasync.frame_camera import cond_vector
        ft = []
        with torch.inference_mode():
            for i in range(1200):
                a = time.perf_counter(); fm(x, cond_vector(120, 0.004)); b = time.perf_counter()
                if i >= 200: ft.append(b - a)
        tick = physics_tick_cost(5000)
        q = lambda a, p: float(np.percentile(a, p) * 1e3)
        res = dict(correction_samples_s=[float(v) for v in lat[::3]],
                   correction_ms=dict(p50=q(lat, 50), p95=q(lat, 95), p99=q(lat, 99), max=float(lat.max() * 1e3)),
                   frame_model_ms=dict(p50=q(ft, 50), p95=q(ft, 95), p99=q(ft, 99), max=float(np.max(ft) * 1e3)),
                   always_on_tick_ms=dict(p50=q(tick, 50), p95=q(tick, 95), p99=q(tick, 99), max=float(tick.max() * 1e3)))
        json.dump(res, open(f"{OUT}/latency.json", "w"), indent=1)
        print("latency", {k: v for k, v in res.items() if k != "correction_samples_s"}, flush=True)
    if stage in ("all", "calibration"):
        from eval_uncertainty import calibrate, load_obs
        calibrate({n: load_obs(n) for n in ["event_s0", "event_s1", "event_s2", "eventonly_s0"]})
    if stage in ("all", "shield", "trigger"):
        tu = json.load(open(f"{OUT}/tuning.json")) if os.path.exists(f"{OUT}/tuning.json") else {}
        if stage in ("all", "shield"):
            jobs = [(f, i) for f in ("event", "frame", "imu", "oracle") for i in range(len(SHIELD_GRID))]
            with Pool(2) as pool:
                res = pool.map(shield_job, jobs, chunksize=1)
            tu["shield_grid"] = {f"{f}|{i}": dict(params=SHIELD_GRID[i], score=s) for f, i, s, _ in res}
            tu["shield_by_family"] = {}
            for fam in ("event", "frame", "imu", "oracle"):
                best = max((r for r in res if r[0] == fam), key=lambda r: r[2])
                tu["shield_by_family"][fam] = SHIELD_GRID[best[1]]
                print("shield", fam, SHIELD_GRID[best[1]], round(best[2], 3), flush=True)
            json.dump(tu, open(f"{OUT}/tuning.json", "w"), indent=1)
        if stage in ("all", "trigger"):
            sp = tu["shield_by_family"]["event"]
            with Pool(2) as pool:
                res = pool.map(trig_job, [(i, sp) for i in range(len(TRIG_GRID))], chunksize=1)
            for i, s, _, c in res:
                print("trigger", TRIG_GRID[i], round(s, 3), "calls/s", round(c, 1), flush=True)
            top = max(r[1] for r in res)
            best = min((r for r in res if r[1] >= top - 0.02), key=lambda r: r[3])   # cheapest within 0.02 of best
            tu["trigger"] = TRIG_GRID[best[0]]
            tu["trigger_calls_per_s_val"] = best[3]
            tu["matched_period"] = int(max(1, round(1000.0 / best[3])))
            tu["trigger_grid"] = [dict(params=TRIG_GRID[i], score=s, calls_per_s=c) for i, s, _, c in res]
            print("chosen trigger", tu["trigger"], "calls/s", round(best[3], 1), "matched period", tu["matched_period"], flush=True)
            json.dump(tu, open(f"{OUT}/tuning.json", "w"), indent=1)
