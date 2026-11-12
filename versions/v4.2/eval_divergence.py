"""Closed-loop failure warning: which signal warns before the event observer loses the pole?

The event stack's dominant failure is observer divergence (the estimate drifts
to a wrong, even mirror-image, angle) after large hits. Three warning signals
are recorded on the same runs:
  sigma     calibrated variance-head sigma_theta of the controlling model (seed 0)
  spread    disagreement (std of theta) between the 3 seed models run in shadow on
            the same sensor data, each with its own recurrent state
  monitor   EventConsistencyMonitor score m (model-free check against raw events)
Thresholds: chosen on validation seeds 300-315 so that at most 5% of the runs that
survive raise any alarm. Test: seeds 500-515 x 7 hit levels (the benchmark trials).
Metrics per detector: detection rate (failed runs with an alarm >= 20 ms before the
fall), median warning lead time, false-alarm rate (surviving runs with any alarm).
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.safety import SafetyShield, ShieldedController
from aurasync.scheduler import TriggeredEventPipeline
from aurasync.monitor import EventConsistencyMonitor
from aurasync.simulate import run_episode
import benchmark_v3 as B

torch.set_num_threads(1)
OUT = "results/v3"
KICKS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


class Instrumented(TriggeredEventPipeline):
    """Dense seed-0 observer in control + shadow seed models + consistency monitor."""

    def __init__(self, models, temp):
        super().__init__(models[0], mode="dense", temp=temp)
        self.shadows = [TriggeredEventPipeline(m, mode="dense", temp=temp) for m in models[1:]]

    def reset(self, s0):
        super().reset(s0)
        for sh in self.shadows:
            sh.reset(s0)
        self.mon = EventConsistencyMonitor()
        self.rec = dict(sigma=[], spread=[], monitor=[])

    def update(self, k, t, s_true, surface, u):
        super().update(k, t, s_true, surface, u)
        ths = [self.out[-1][0][2]]
        for sh in self.shadows:
            sh.update(k, t, s_true, surface, u); ths.append(sh.out[-1][0][2])
        self.rec["sigma"].append(np.sqrt(self.out[-1][1][2, 2]))
        self.rec["spread"].append(float(np.std(ths)))
        self.rec["monitor"].append(self.mon.update(self.out[-1][0], surface))


def run(seeds, shield, models, temp, sp):
    p = PlantParams(); lqr = LQRController(p)
    out = []
    for kick in KICKS:
        for i, sd in enumerate(seeds):
            rng = np.random.default_rng(sd); s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
            pipe = Instrumented(models, temp)
            ctrl = ShieldedController(lqr, SafetyShield(p, **sp), pipe, enabled=shield)
            lg = run_episode(p, ctrl, pipe, B.N_TICKS, s0, kicks=[(B.KICK_TICK, kick * (1 if i % 2 == 0 else -1))], cam_seed=sd)
            err = np.degrees(np.abs(lg["s_hat"][:, 2] - lg["s"][:, 2]))
            out.append(dict(kick=kick, seed=sd, fail=lg["fail_tick"], err=err.tolist(),
                            **{k: np.asarray(v).tolist() for k, v in pipe.rec.items()}))
    return out


def first_alarm(sig, thr, start=50):
    idx = np.nonzero(np.asarray(sig[start:]) > thr)[0]
    return None if len(idx) == 0 else int(idx[0] + start)


def pick_threshold(runs, key, max_fa=0.05):
    ok = [r for r in runs if r["fail"] is None]
    peaks = np.sort([max(r[key][50:]) for r in ok])
    return float(peaks[int(np.ceil((1 - max_fa) * len(peaks))) - 1]) + 1e-9 if len(peaks) else np.inf


def score(runs, key, thr):
    failed = [r for r in runs if r["fail"] is not None]
    ok = [r for r in runs if r["fail"] is None]
    leads = []
    for r in failed:
        a = first_alarm(r[key], thr)
        if a is not None and r["fail"] - a >= 20:
            leads.append(r["fail"] - a)
    fa = np.mean([first_alarm(r[key], thr) is not None for r in ok]) if ok else float("nan")
    return dict(threshold=thr, n_failed=len(failed), n_ok=len(ok),
                detection_rate=len(leads) / max(len(failed), 1),
                median_lead_ms=float(np.median(leads)) if leads else None,
                min_lead_ms=float(np.min(leads)) if leads else None, false_alarm_rate=float(fa))


if __name__ == "__main__":
    cal, tu = B.calibration(), B.tuning()
    temp = np.array(cal["temp"]["varhead"]); sp = tu["shield_by_family"]["event"]
    models = [B.event_model(f"event_s{i}") for i in range(3)]
    res = {}
    for shield in (True, False):
        tag = "shield" if shield else "no_shield"
        val = run(list(range(300, 316)), shield, models, temp, sp)
        test = run(list(range(500, 516)), shield, models, temp, sp)
        res[tag] = {}
        for key in ("sigma", "spread", "monitor"):
            thr = pick_threshold(val, key)
            res[tag][key] = dict(val=score(val, key, thr), test=score(test, key, thr))
            print(tag, key, json.dumps(res[tag][key]["test"]), flush=True)
    json.dump(res, open(f"{OUT}/divergence_warning.json", "w"), indent=1)
