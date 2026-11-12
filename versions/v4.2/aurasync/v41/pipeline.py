"""AuraSync V4.1: geometry-first, evidence-aware estimation with selective neural assistance.

Analogy: a surveyor with a theodolite (event geometry) does the work. Before trusting a
reading he asks what kind of silence or noise he is facing:
  QUIET          nothing moves, the instrument is alive, and nothing *should* move
                 -> the last fix still holds; uncertainty stays bounded
  UNOBSERVED     something is moving (activity in the pole region) or should be moving
                 (the estimate predicts motion), but no usable fit -> uncertainty grows
                 with the full process noise: absence of evidence is NOT proof of stillness
  CONTRADICTORY  a good-looking fit disagrees with the estimate -> wait for corroboration
                 (re-initialise after several consistent fits), meanwhile widen the
                 uncertainty so a real change cannot be locked out
  CORRUPT        burst of events far above the background, or fits that disagree with
                 each other -> ignore the sensor for now, widen uncertainty
Only when the theodolite is not enough (UNOBSERVED long enough, unresolved contradiction,
corruption, or high uncertainty) does he radio the expensive helper (the V3 neural
observer), within a budget, with a periodic check-in at most `max_interval` ticks apart.
The helper's answer is accepted only through a gate, and only for positions (x, theta):
velocities stay with the physics + geometry filter.

neural=None gives the network-free variant ("D-ev": Baseline D + the evidence fix).
No ground truth is used: inputs are the time surface, the applied controls and the
estimator's own state.
"""
from collections import deque
import numpy as np
import torch

from ..physics import PlantParams, rk4_step, linearise
from ..estimator import SCALE
from ..scheduler import FRESH
from ..v4.geometry import EventGeometry

DT = 1e-3
SCALE_NP = SCALE.numpy().astype(np.float64)
QUIET, UNOBSERVED, FUSED, CONTRADICTORY, CORRUPT, REINIT = range(6)
STATE_NAMES = ["quiet", "unobserved", "fused", "contradictory", "corrupt", "reinit"]


class EvidencePipeline:
    def __init__(self, Q, neural=None, temp=None, latency_samples=(0.002,), deadline=6,
                 # measurement model (calibrated on validation seeds, results/v41/geometry_calibration.json)
                 meas_inflate=1.4, low_support_inflate=1.6, sigma_x_floor=0.004, gate=9.0,
                 # evidence classification
                 roi_quiet=4, v_quiet=0.15, alive_window=10, sigma_quiet_deg=3.0, sigma_dquiet=0.5,
                 burst_factor=4.0, burst_min=40, consistency_deg=15.0,
                 # contradictions
                 confirm_n=4, confirm_window=12, confirm_spread_deg=3.0, cooldown=40, contra_growth=1.3,
                 corrupt_q_deg=1.0,
                 # selective neural assistance (Module E)
                 min_gap=10, max_interval=50, unobs_ticks=3, sigma_req_deg=3.0, neural_gate=9.0,
                 Q_quiet=None, plant=None, seed=0, name=None):
        self.p = plant or PlantParams()
        self.A, _ = linearise(self.p, DT)
        self.Q = Q
        self.Q_quiet = np.diag([1e-9, 1e-5, 1e-8, 1e-4]) if Q_quiet is None else Q_quiet
        self.geo = EventGeometry(seed=seed + 1)
        self.models = None if neural is None else [m.eval() for m in (neural if isinstance(neural, (list, tuple)) else [neural])]
        self.temp = np.ones(4) if temp is None else np.asarray(temp, float)
        self.lat = np.asarray(latency_samples, float)
        self.deadline = deadline
        self.mi, self.mi_low, self.sx, self.gate = meas_inflate, low_support_inflate, sigma_x_floor, gate
        self.roi_quiet, self.v_quiet, self.alive_window = roi_quiet, v_quiet, alive_window
        self.sig_quiet, self.sig_dquiet = np.radians(sigma_quiet_deg), sigma_dquiet
        self.burst_factor, self.burst_min, self.consistency = burst_factor, burst_min, np.radians(consistency_deg)
        self.confirm_n, self.confirm_window, self.confirm_spread = confirm_n, confirm_window, np.radians(confirm_spread_deg)
        self.cooldown, self.contra_growth, self.q_corrupt = cooldown, contra_growth, np.radians(corrupt_q_deg) ** 2
        self.min_gap, self.max_interval, self.unobs_ticks = min_gap, max_interval, unobs_ticks
        self.sigma_req, self.neural_gate = np.radians(sigma_req_deg), neural_gate
        self.seed = seed
        self.name = name or ("v41" if neural is not None else "geo-ev")
        self.delay_ticks = 0

    # ---- interface --------------------------------------------------------------------------
    def reset(self, s0):
        self.rng = np.random.default_rng(self.seed)
        self.geo.rng = np.random.default_rng(self.seed + 1)
        self.s = np.asarray(s0, float).copy()
        self.Sigma = np.diag((0.05 * SCALE_NP) ** 2)
        self.k = -1
        self.u_hist = deque(maxlen=256)
        self.buf = deque(maxlen=32)               # recent valid fits (k, theta, sigma, x)
        self.cbuf = deque(maxlen=32)              # recent fits that CONTRADICTED the estimate
        self.last_alive = -10 ** 9                # last tick with any background/scene activity
        self.ev_base = None
        self.unobs_run = 0
        self.last_reinit = -10 ** 9
        self.last_call = -10 ** 9
        self.job = None                           # (capture_k, done_k, s_corr, Sigma_corr, h_new)
        self.h = None if self.models is None else [m.init_hidden(1) for m in self.models]
        self.state = QUIET
        self.log = dict(calls=0, misses=0, accepted=0, rejected=0, states=[0] * 6, triggers=dict(
            unobserved=0, contradictory=0, corrupt=0, doubt=0, check=0), sigma_th=[], state_seq=[])

    def estimate(self):
        return self.s

    def covariance(self):
        return self.Sigma

    def age_ticks(self):
        return 0

    # ---- neural helper (identical computation to the V3 observer) ----------------------------
    @torch.inference_mode()
    def _neural(self, surface, s_pred, gap):
        ts = torch.from_numpy(np.ascontiguousarray(surface, dtype=np.float32))[None]
        sp = torch.tensor(s_pred, dtype=torch.float32)[None]
        g = torch.tensor([float(gap)])
        outs, vars_, hs = [], [], []
        for m, h in zip(self.models, self.h):
            z, act = m.encode(ts)
            s_new, h_new, _, lv = m.correct(z, act, h, sp, g)
            outs.append(s_new[0].double().numpy()); hs.append(h_new)
            vars_.append((np.exp(lv[0].double().numpy()) * SCALE_NP ** 2) * self.temp ** 2)
        return np.mean(outs, 0), np.diag(np.mean(vars_, 0)), hs

    def _propagate(self, s, Sigma, u, Q):
        return rk4_step(s, u, DT, self.p), self.A @ Sigma @ self.A.T + Q

    # ---- Kalman helpers ---------------------------------------------------------------------
    def _kf(self, idx, z, R):
        H = np.eye(4)[idx]
        S = H @ self.Sigma @ H.T + R
        K = np.linalg.solve(S, H @ self.Sigma).T
        self.s = self.s + K @ (z - H @ self.s)
        I_KH = np.eye(4) - K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ R @ K.T

    def _cap(self, i, sig):
        """Bound the std of state i at sig by scaling its row/column (keeps Sigma PSD)."""
        v = self.Sigma[i, i]
        if v > sig ** 2:
            f = sig / np.sqrt(v)
            self.Sigma[i, :] *= f; self.Sigma[:, i] *= f

    def _widen(self, i, var):
        self.Sigma[i, i] += var

    # ---- one tick ---------------------------------------------------------------------------
    def update(self, k, t, s_true, surface, u):
        self.k = k
        self.u_hist.append(u)
        # evidence counts (no model, no truth)
        ev = self.geo.events(surface)
        roi = (self.geo.wy >= self.geo.y_lo) & (self.geo.wy <= self.geo.y_hi) & (np.abs(self.geo.wx - self.s[0]) <= self.geo.band)
        n_glob, n_roi = int(ev.sum()), int((ev & roi).sum())
        if n_glob > 0:
            self.last_alive = k
        burst = self.ev_base is not None and n_glob > max(self.burst_factor * self.ev_base, self.burst_min)
        self.ev_base = n_glob if self.ev_base is None else (0.995 * self.ev_base + 0.005 * min(n_glob, 3 * self.ev_base + 5))
        alive = k - self.last_alive <= self.alive_window
        v_hat = abs(self.s[1] + 2 * self.geo.l * self.s[3] * np.cos(self.s[2]))

        m = self.geo.measure(surface, self.s[0])
        valid = m is not None and m["valid"]
        burst_corrupt = burst and not valid
        # classify the evidence BEFORE propagation decides how much process noise to add
        if burst_corrupt:
            state = CORRUPT
        elif valid:
            recent = [b[1] for b in self.buf if k - b[0] <= 5]
            if recent and abs(m["theta"] - np.median(recent)) > self.consistency and len(recent) >= 3:
                state = CORRUPT                                  # fits disagree with each other
            else:
                state = FUSED                                    # provisional; gate decides below
        elif n_roi < self.roi_quiet and v_hat < self.v_quiet and alive:
            state = QUIET
        else:
            state = UNOBSERVED

        # 1. physics prediction: full process noise unless the scene is verified quiet
        self.s, self.Sigma = self._propagate(self.s, self.Sigma, u, self.Q_quiet if state == QUIET else self.Q)
        if state == QUIET:
            self._cap(2, self.sig_quiet); self._cap(3, self.sig_dquiet)

        # 2. finished neural job -> gated, position-only fusion
        if self.job is not None:
            self._finish_job(k)

        # 3. geometry
        if state in (FUSED,):
            sig = m["sigma_theta"] * (self.mi_low if (m["span"] < 0.6 or m["n_inliers"] < 12) else self.mi)
            sx = max(self.sx, 0.5 / self.geo.S / np.sqrt(m["n_inliers"])) * self.mi
            nu = m["theta"] - self.s[2]
            d2 = nu ** 2 / (self.Sigma[2, 2] + sig ** 2)
            self.buf.append((k, m["theta"], sig, m["x_pivot"]))
            if d2 <= self.gate:
                self._kf([2, 0], np.array([m["theta"], m["x_pivot"]]), np.diag([sig ** 2, sx ** 2]))
                self.cbuf.clear()
            else:
                state = CONTRADICTORY
                self.cbuf.append((k, m["theta"], sig, m["x_pivot"]))
                if k - self.last_reinit >= self.cooldown and self._confirmed(k):
                    self._reinit(k); state = REINIT
                else:                                            # covariance matching, bounded growth
                    target = nu ** 2 / self.gate - sig ** 2
                    self.Sigma[2, 2] = max(self.Sigma[2, 2], min(target, self.Sigma[2, 2] * self.contra_growth ** 2))
        elif state == CORRUPT:
            self._widen(2, self.q_corrupt)
        self.unobs_run = self.unobs_run + 1 if state == UNOBSERVED else 0

        # 4. selective neural assistance
        if self.models is not None and self.job is None and k - self.last_call >= self.min_gap:
            why = None
            if state == UNOBSERVED and self.unobs_run >= self.unobs_ticks:
                why = "unobserved"
            elif state == CONTRADICTORY:
                why = "contradictory"
            elif state == CORRUPT:
                why = "corrupt"
            elif np.sqrt(self.Sigma[2, 2]) > self.sigma_req:
                why = "doubt"
            elif k - self.last_call >= self.max_interval:
                why = "check"
            if why is not None:
                gap = max(1, k - self.last_call) if self.last_call > -10 ** 8 else 1
                s_c, S_c, h_new = self._neural(surface, self.s, min(gap, 1000))
                lat = int(np.ceil(self.rng.choice(self.lat) / DT))
                self.job = (k, k + lat, s_c, S_c, h_new, self.s.copy())
                self.last_call = k
                self.log["calls"] += 1; self.log["triggers"][why] += 1

        self.state = state
        self.log["states"][state] += 1
        self.log["state_seq"].append(state)
        self.log["sigma_th"].append(np.sqrt(self.Sigma[2, 2]))

    def _finish_job(self, k):
        cap_k, done_k, s_c, S_c, h_new, s_cap = self.job
        if done_k - cap_k > self.deadline and k >= cap_k + self.deadline:
            self.log["misses"] += 1; self.job = None
            return
        if k < done_k:
            return
        Sc = S_c.copy()
        for uu in list(self.u_hist)[-(k - cap_k):] if k > cap_k else []:
            s_c, Sc = self._propagate(s_c, Sc, uu, self.Q)
        self.h = h_new
        R = np.diag(np.clip(np.diag(Sc)[[0, 2]], 1e-12, None))
        nu = s_c[2] - self.s[2]
        d2 = nu ** 2 / (self.Sigma[2, 2] + R[1, 1])
        if d2 <= self.neural_gate:
            self._kf([0, 2], s_c[[0, 2]], R)                    # positions only; velocities via cross-covariance
            self.log["accepted"] += 1
        else:
            self.log["rejected"] += 1
        self.job = None

    def _confirmed(self, k):
        recent = [b for b in self.cbuf if k - b[0] <= self.confirm_window]
        if len(recent) < self.confirm_n:
            return False
        th = np.array([b[1] for b in recent])
        return float(np.abs(th - np.median(th)).max()) <= self.confirm_spread

    def _reinit(self, k):
        recent = [b for b in self.cbuf if k - b[0] <= self.confirm_window]
        ks = np.array([b[0] for b in recent], float); th = np.array([b[1] for b in recent])
        self.s[2] = float(np.median(th))
        if ks.max() - ks.min() >= 6:
            self.s[3] = float(np.polyfit((ks - ks.max()) * DT, th, 1)[0])
        self.s[0] = float(np.median([b[3] for b in recent]))
        sig_m = float(np.median([b[2] for b in recent]))
        self.Sigma = np.diag([max(self.Sigma[0, 0], 1e-4), max(self.Sigma[1, 1], 0.04), sig_m ** 2, 1.0])
        if self.models is not None:
            self.h = [m.init_hidden(1) for m in self.models]
            self.job = None
        self.last_reinit = k
        self.cbuf.clear()
