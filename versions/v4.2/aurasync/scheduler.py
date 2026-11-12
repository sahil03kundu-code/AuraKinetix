"""Event-triggered neural correction with a deadline guarantee (Upgrade C).

Analogy: an air-traffic controller who dead-reckons every plane each second
(cheap) and only asks for a radar sweep (expensive) when something happens: a
burst of activity, growing doubt, a plane near another one, or simply because it
has been too long. The radar takes time, so the controller never waits for it.

Per 1 ms control tick (never waits on the network):
  1. physics prediction s <- RK4(s, u); covariance  Sigma <- A Sigma A^T + Q
  2. if a correction result has arrived: roll the corrected state at its capture
     time forward through the controls applied since (replay), take its covariance
  3. if the worker is idle and a trigger fires, capture the current time surface
     and start a correction job; it completes after a compute latency sampled
     from the latency measured on this CPU
Triggers (mode 'triggered'):
  burst      >= burst_thr pixels fired an event in the last tick
  doubt      propagated sigma_theta > sigma_trig
  margin     |psi| = |theta + 0.1 theta_dot| > margin_trig (close to the safety envelope)
  max gap    >= max_gap ticks since the last correction (quiet scenes still get checked)
Deadline: a job not finished within `deadline` ticks is abandoned (deadline
miss, logged). Fallback is deterministic: keep the physics prediction and let
the covariance grow, which tightens the safety shield and can trip its stale mode.
Other modes, for comparison at matched compute:
  dense      a correction every tick, synchronous, 1 tick delay (the v2 behaviour;
             infeasible in real time on this CPU, kept as reference)
  greedy     async; start a new job whenever the worker is free
  periodic   async; start a job every `period` ticks
"""
from collections import deque
import numpy as np
import torch

from .physics import PlantParams, rk4_step, linearise
from .estimator import SCALE

DT = 1e-3
SCALE_NP = SCALE.numpy().astype(np.float64)
FRESH = float(np.exp(-1e-3 / 0.008))     # fast time-surface value of an event < 1 ms old


class TriggeredEventPipeline:
    def __init__(self, models, mode="triggered", latency_samples=(0.0018,), deadline=6,
                 period=5, max_gap=20, burst_thr=12, sigma_trig_deg=1.5, margin_trig_deg=7.0,
                 temp=None, Q=None, seed=0, name=None, plant=None):
        self.models = [m.eval() for m in (models if isinstance(models, (list, tuple)) else [models])]
        self.mode, self.deadline, self.period, self.max_gap = mode, deadline, period, max_gap
        self.burst_thr = burst_thr
        self.sigma_trig, self.margin_trig = np.radians(sigma_trig_deg), np.radians(margin_trig_deg)
        self.lat = np.asarray(latency_samples, float) * len(self.models)   # members run sequentially
        self.temp = np.ones(4) if temp is None else np.asarray(temp, float)
        self.p = plant or PlantParams()
        self.A, _ = linearise(self.p, DT)
        self.Q = np.diag([1e-9, 1e-5, 1e-8, 1e-4]) if Q is None else Q
        self.seed = seed
        tag = "" if len(self.models) == 1 else f"-ens{len(self.models)}"
        self.name = name or f"event-{mode}{tag}"
        self.delay_ticks = 1 if mode == "dense" else 0

    # ---- interface used by run_episode / ShieldedController ---------------------
    def reset(self, s0):
        self.rng = np.random.default_rng(self.seed)
        self.s = np.asarray(s0, float).copy()
        self.Sigma = np.diag((0.05 * SCALE_NP) ** 2)
        self.h = [m.init_hidden(1) for m in self.models]
        self.u_hist = deque(maxlen=256)
        self.k = -1
        self.last_capture = 0
        self.job = None                        # (capture_k, done_k, s_corr, Sigma_corr, new_h)
        self.out = deque([(self.s.copy(), self.Sigma.copy())] * 2, maxlen=2)
        self.log = dict(calls=0, misses=0, triggers=dict(burst=0, doubt=0, margin=0, gap=0, idle=0, periodic=0),
                        busy_ticks=0, sigma_th=[], corrected=[])

    def estimate(self):
        return self.out[0][0] if self.mode == "dense" else self.out[-1][0]

    def covariance(self):
        return self.out[0][1] if self.mode == "dense" else self.out[-1][1]

    def age_ticks(self):
        return self.delay_ticks

    # ---- internals ----------------------------------------------------------------
    @torch.inference_mode()
    def _correct(self, surface, s_pred, gap):
        ts = torch.from_numpy(np.ascontiguousarray(surface, dtype=np.float32))[None]
        sp = torch.tensor(s_pred, dtype=torch.float32)[None]
        g = torch.tensor([float(gap)])
        outs, vars_, hs = [], [], []
        for m, h in zip(self.models, self.h):
            z, act = m.encode(ts)
            s_new, h_new, _, lv = m.correct(z, act, h, sp, g)
            outs.append(s_new[0].double().numpy()); hs.append(h_new)
            vars_.append((np.exp(lv[0].double().numpy()) * SCALE_NP ** 2) * self.temp ** 2)
        outs = np.stack(outs)
        mean = outs.mean(0)
        var = np.mean(vars_, 0) + (outs.var(0) if len(outs) > 1 else 0.0)   # aleatoric + epistemic
        return mean, np.diag(var), hs

    def _propagate(self, s, Sigma, u):
        return rk4_step(s, u, DT, self.p), self.A @ Sigma @ self.A.T + self.Q

    def _trigger(self, surface):
        if self.mode == "greedy":
            return "idle"
        gap = self.k - self.last_capture
        if self.mode == "periodic":
            return "periodic" if gap >= self.period else None
        if gap >= self.max_gap:
            return "gap"
        if (surface[0] >= FRESH).sum() + (surface[1] >= FRESH).sum() >= self.burst_thr:
            return "burst"
        if np.sqrt(self.Sigma[2, 2]) > self.sigma_trig:
            return "doubt"
        if abs(self.s[2] + 0.1 * self.s[3]) > self.margin_trig:
            return "margin"
        return None

    def update(self, k, t, s_true, surface, u):
        self.k = k
        self.u_hist.append(u)
        if self.mode == "dense":
            s_pred, _ = self._propagate(self.s, self.Sigma, u)
            self.s, self.Sigma, self.h = self._correct(surface, s_pred, 1)
            self.log["calls"] += 1
            self.out.append((self.s.copy(), self.Sigma.copy()))
            self.log["sigma_th"].append(np.sqrt(self.Sigma[2, 2])); self.log["corrected"].append(True)
            return
        # 1. cheap physics prediction, every tick
        self.s, self.Sigma = self._propagate(self.s, self.Sigma, u)
        corrected = False
        # 2. collect a finished job (or abandon one past its deadline)
        if self.job is not None:
            cap_k, done_k, s_c, S_c, h_new = self.job
            if done_k - cap_k > self.deadline and k >= cap_k + self.deadline:
                self.log["misses"] += 1                          # deadline miss -> discard
                self.job = None
            elif k >= done_k:
                for uu in list(self.u_hist)[-(k - cap_k):] if k > cap_k else []:
                    s_c, S_c = self._propagate(s_c, S_c, uu)    # replay controls since capture
                self.s, self.Sigma, self.h = s_c, S_c, h_new
                self.job = None
                corrected = True
            else:
                self.log["busy_ticks"] += 1
        # 3. start a new job if idle and triggered
        if self.job is None:
            why = self._trigger(surface)
            if why is not None:
                gap = max(1, k - self.last_capture)
                s_c, S_c, h_new = self._correct(surface, self.s, gap)
                lat_ticks = int(np.ceil(self.rng.choice(self.lat) / DT))
                self.job = (k, k + lat_ticks, s_c, S_c, h_new)
                self.last_capture = k
                self.log["calls"] += 1
                self.log["triggers"][why] += 1
        self.out.append((self.s.copy(), self.Sigma.copy()))
        self.log["sigma_th"].append(np.sqrt(self.Sigma[2, 2])); self.log["corrected"].append(corrected)


@torch.inference_mode()
def measure_correction_latency(model, n=3000, warm=200):
    """Wall-clock cost of one neural correction (encode + correct) on one CPU thread."""
    import time
    torch.set_num_threads(1)
    ts = torch.rand(1, 4, 32, 96); h = model.init_hidden(1); s = torch.zeros(1, 4); g = torch.ones(1)
    out = []
    for i in range(n + warm):
        a = time.perf_counter()
        z, act = model.encode(ts)
        model.correct(z, act, h, s, g)
        b = time.perf_counter()
        if i >= warm:
            out.append(b - a)
    return np.array(out)


def physics_tick_cost(n=5000):
    """Cost of the always-on part of the control tick: prediction + covariance + LQR + shield."""
    import time
    from .controller import LQRController
    from .safety import SafetyShield
    p = PlantParams(); A, _ = linearise(p, DT); Q = np.diag([1e-9, 1e-5, 1e-8, 1e-4])
    lqr = LQRController(p); sh = SafetyShield(p)
    s = np.array([0.01, 0.0, 0.02, 0.1]); S = np.eye(4) * 1e-4
    out = []
    for i in range(n):
        a = time.perf_counter()
        s2 = rk4_step(s, 0.5, DT, p); S2 = A @ S @ A.T + Q
        sh.filter(s2, S2, lqr.act(s2))
        out.append(time.perf_counter() - a)
    return np.array(out)
