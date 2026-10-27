"""AuraSync V4: Detect -> Correct -> Recover.

Analogy: a ship's navigator (the neural observer) dead-reckons and reads the
instruments; a lookout (EventGeometry) shouts bearings to the mast whenever he can
see it. Small disagreements are averaged in. A large, *repeated*, *self-consistent*
disagreement means the navigator has lost the plot, so his position is reset to the
lookout's bearing. If the lookout is unsure (fog, spray), nobody resets anything;
the captain is simply told the position is uncertain.

Module A  ConsistencyDetector   per-tick anomaly score (CUSUM of the geometric
                                innovation), alarm + timestamp, reliability, reasons
Module B  V4Pipeline correction gated Kalman fusion (theta, x) / confirmed
                                re-initialisation / uncertainty inflation
Module C  covariance after every step is the posterior of the fusion, and is
                                inflated when the evidence contradicts it without
                                allowing a correction
No ground truth is used: inputs are the sensor surface, the controls and the
observer's own estimate.
"""
from collections import deque
import numpy as np
import torch

from ..scheduler import TriggeredEventPipeline, FRESH
from .geometry import EventGeometry

DT = 1e-3


class ConsistencyDetector:
    """Per-tick consistency check between the estimate and independent event geometry."""

    def __init__(self, cusum_k=4.0, cusum_h=40.0, sign_deg=3.0, stale_ticks=30, motion_speed=0.3,
                 burst_factor=4.0, l=0.25):
        self.k, self.h = cusum_k, cusum_h
        self.sign = np.radians(sign_deg)
        self.stale_ticks, self.motion_speed, self.burst_factor, self.l = stale_ticks, motion_speed, burst_factor, l
        self.reset()

    def reset(self):
        self.g = 0.0
        self.alarm = False
        self.t_detect = None
        self.last_valid = 0
        self.rel = 0.0
        self.ev_baseline = None
        self.reasons = []

    def update(self, k, s_hat, Sigma, m, n_fresh):
        reasons = []
        d2 = None
        # running estimate of the normal fresh-event count (for burst detection)
        self.ev_baseline = n_fresh if self.ev_baseline is None else 0.99 * self.ev_baseline + 0.01 * n_fresh
        burst = n_fresh > self.burst_factor * max(self.ev_baseline, 5.0)
        if m is not None and m["valid"]:
            nu = m["theta"] - s_hat[2]
            d2 = nu ** 2 / (Sigma[2, 2] + m["sigma_theta"] ** 2)
            # CUSUM on the normalised innovation; each tick adds at most 16 - k (= 12), so a single
            # outlier can never raise the alarm alone (>= 4 consecutive strong disagreements needed)
            self.g = max(0.0, self.g + min(d2, 16.0) - self.k)
            self.last_valid = k
            self.rel = 0.9 * self.rel + 0.1 * m["quality"]
            if d2 > 9:
                reasons.append("angle_mismatch")
            if np.sign(m["theta"]) != np.sign(s_hat[2]) and min(abs(m["theta"]), abs(s_hat[2])) > self.sign:
                reasons.append("sign_mismatch")
        else:
            self.rel *= 0.98
            v_top = s_hat[1] + 2 * self.l * s_hat[3] * np.cos(s_hat[2])
            if abs(v_top) > self.motion_speed and n_fresh < 5:
                reasons.append("missing_events")                              # estimate says motion, sensor sees none
            if k - self.last_valid > self.stale_ticks:
                reasons.append("stale_evidence")
        if burst:
            reasons.append("noise_burst")
        self.alarm = self.g > self.h
        if self.alarm and self.t_detect is None:
            self.t_detect = k
        if not self.alarm and self.g == 0.0:
            self.t_detect = None
        self.reasons = reasons
        return dict(score=self.g / self.h, alarm=self.alarm, t_detect=self.t_detect, reliability=self.rel,
                    reasons=reasons, d2=d2)


class V4Pipeline(TriggeredEventPipeline):
    """V3 observer + event geometry + detector + gated correction.

    correct        apply gated fusion and confirmed re-initialisation (False -> detection only)
    inflate        inflate the covariance when the detector alarms without a usable correction
    hidden_reset   'zero' | 'keep': what happens to the observer's SSM memory on re-initialisation
    fuse_x         also fuse the event-measured pivot position
    """

    def __init__(self, models, mode="triggered", correct=True, inflate=True, hidden_reset="zero", fuse_x=True,
                 gate=9.0, confirm_n=4, confirm_window=12, confirm_spread_deg=3.0, cooldown=40,
                 inflate_sigma_deg=10.0, sigma_x_floor=0.004, neural_fusion="overwrite", min_gap=0,
                 residual_trigger=None, name=None, **kw):
        super().__init__(models, mode=mode, **kw)
        self.min_gap = int(min_gap)                # Module E: hard compute budget (>= min_gap ticks between corrections)
        # Module E: also trigger on the event-consistency residual (default: only when a budget is set,
        # so the unbudgeted V4 is exactly the design selected on validation)
        self.residual_trigger = (min_gap > 0) if residual_trigger is None else residual_trigger
        self._last_d2 = None
        self.neural_fusion = neural_fusion        # 'overwrite' (V3 semantics) | 'kalman' (full-state measurement)
                                                  # | 'kalman_pos' (measurement of x and theta only; velocities come
                                                  #   from physics + the measurement stream via the cross-covariance)
        self.geo = EventGeometry()
        self.det = ConsistencyDetector()
        self.correct_on, self.inflate_on, self.hidden_reset, self.fuse_x = correct, inflate, hidden_reset, fuse_x
        self.gate, self.confirm_n, self.confirm_window = gate, confirm_n, confirm_window
        self.confirm_spread = np.radians(confirm_spread_deg)
        self.cooldown, self.inflate_sigma, self.sigma_x_floor = cooldown, np.radians(inflate_sigma_deg), sigma_x_floor
        tag = ("" if correct else "-nocorr") + ("" if inflate else "-noinfl") + ("" if hidden_reset == "zero" else "-keeph")
        self.name = name or f"v4-{mode}{tag}"

    def reset(self, s0):
        super().reset(s0)
        self.geo.rng = np.random.default_rng(self.seed + 1)
        self.det.reset()
        self.buf = deque(maxlen=32)            # (k, theta_meas, sigma, x_pivot) of recent valid measurements
        self.last_reinit = -10 ** 9
        self.cap_state = None                  # state at the capture of the in-flight neural job
        self.ev_log = []                       # correction / reinit / inflation events (diagnostics)
        self.det_log = []

    # ---- neural job bookkeeping: apply a finished correction as a DELTA ---------------------
    def _trigger(self, surface):
        """V3 triggers (burst, doubt, safety margin, max gap) + the consistency residual,
        all subject to a hard budget of one correction per `min_gap` ticks."""
        if self.min_gap and self.k - self.last_capture < self.min_gap:
            return None
        why = super()._trigger(surface)
        if why is None and self.residual_trigger and self._last_d2 is not None and self._last_d2 > self.gate:
            self.log["triggers"]["residual"] = self.log["triggers"].get("residual", 0)
            why = "residual"
        return why

    def _correct(self, surface, s_pred, gap):
        out = super()._correct(surface, s_pred, gap)
        self.cap_state = np.array(s_pred, float)
        return out

    def _finish_job(self, k):
        """Like V3, but the neural correction is applied as a delta on top of the current
        (possibly event-corrected) state; identical to V3 when no event correction happened."""
        cap_k, done_k, s_c, S_c, h_new = self.job
        if done_k - cap_k > self.deadline and k >= cap_k + self.deadline:
            self.log["misses"] += 1; self.job = None
            return False
        if k < done_k:
            self.log["busy_ticks"] += 1
            return False
        s_base = self.cap_state.copy()
        for uu in list(self.u_hist)[-(k - cap_k):] if k > cap_k else []:
            s_c, S_c = self._propagate(s_c, S_c, uu)
            s_base, _ = self._propagate(s_base, S_c, uu)
        if self.neural_fusion.startswith("kalman"):
            self._fuse_neural(s_c, S_c)
            self.h = h_new
        else:
            self.s = self.s + (s_c - s_base)
            self.Sigma, self.h = S_c, h_new
        self.job = None
        return True

    def _fuse_neural(self, s_c, S_c):
        """Treat the neural observer's output as a measurement with its calibrated (diagonal)
        covariance, weighted against the current fused estimate. 'kalman_pos' uses only the
        position components: the network's velocity outputs were found (validation) to be
        confidently wrong after large hits."""
        idx = [0, 2] if self.neural_fusion == "kalman_pos" else [0, 1, 2, 3]
        H = np.eye(4)[idx]
        R = np.diag(np.clip(np.diag(S_c)[idx], 1e-12, None))
        S = H @ self.Sigma @ H.T + R
        K = np.linalg.solve(S, H @ self.Sigma).T
        self.s = self.s + K @ (s_c[idx] - self.s[idx])
        I_KH = np.eye(4) - K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ R @ K.T

    def update(self, k, t, s_true, surface, u):
        self.k = k
        self.u_hist.append(u)
        corrected = False
        if self.mode == "dense":
            s_pred, S_pred = self._propagate(self.s, self.Sigma, u)
            s_c, S_c, h_new = self._correct(surface, s_pred, 1)
            if self.neural_fusion.startswith("kalman"):
                self.s, self.Sigma = s_pred, S_pred
                self._fuse_neural(s_c, S_c); self.h = h_new
            else:
                self.s, self.Sigma, self.h = s_c, S_c, h_new
            self.log["calls"] += 1
            corrected = True
        else:
            self.s, self.Sigma = self._propagate(self.s, self.Sigma, u)
            if self.job is not None:
                corrected = self._finish_job(k)
            if self.job is None:
                why = self._trigger(surface)
                if why is not None:
                    gap = max(1, k - self.last_capture)
                    s_c, S_c, h_new = self._correct(surface, self.s, gap)
                    lat_ticks = int(np.ceil(self.rng.choice(self.lat) / DT))
                    self.job = (k, k + lat_ticks, s_c, S_c, h_new)
                    self.last_capture = k
                    self.log["calls"] += 1; self.log["triggers"][why] += 1
        # ---- Module A: independent event geometry + consistency detection ------------------
        m = self.geo.measure(surface, self.s[0])
        n_fresh = int(((surface[0] >= FRESH) | (surface[1] >= FRESH)).sum())
        d = self.det.update(k, self.s, self.Sigma, m, n_fresh)
        self._last_d2 = d["d2"]
        if m is not None and m["valid"] and "noise_burst" not in d["reasons"]:
            self.buf.append((k, m["theta"], m["sigma_theta"], m["x_pivot"]))
        action = None
        # ---- Module B: correction ------------------------------------------------------------
        if self.correct_on and m is not None and m["valid"] and "noise_burst" not in d["reasons"]:
            if d["d2"] is not None and d["d2"] <= self.gate:
                self._fuse(m); action = "fuse"
            elif d["alarm"] and k - self.last_reinit >= self.cooldown and self._confirmed(k):
                self._reinit(k); action = "reinit"
        # ---- Module C: inflate when contradicted but not correctable --------------------------
        if action is None and self.inflate_on and d["alarm"]:
            if self.Sigma[2, 2] < self.inflate_sigma ** 2:
                self.Sigma[2, 2] = self.inflate_sigma ** 2
                self.Sigma[3, 3] = max(self.Sigma[3, 3], 4.0)
                action = "inflate"
        if action in ("reinit", "inflate"):
            self.ev_log.append(dict(k=k, action=action, reasons=list(d["reasons"]), score=d["score"]))
        self.det_log.append((d["score"], d["alarm"], action == "fuse", action == "reinit", m is not None and m["valid"]))
        self.out.append((self.s.copy(), self.Sigma.copy()))
        self.log["sigma_th"].append(np.sqrt(self.Sigma[2, 2])); self.log["corrected"].append(corrected)

    def _fuse(self, m):
        """Kalman update with the event measurement of theta (and the pivot x)."""
        H = [np.array([0, 0, 1.0, 0])]; z = [m["theta"]]; R = [m["sigma_theta"] ** 2]
        if self.fuse_x:
            H.append(np.array([1.0, 0, 0, 0])); z.append(m["x_pivot"])
            R.append(max(self.sigma_x_floor, 0.5 / self.geo.S / np.sqrt(m["n_inliers"])) ** 2)
        H, z, R = np.stack(H), np.array(z), np.diag(R)
        S = H @ self.Sigma @ H.T + R
        K = np.linalg.solve(S, H @ self.Sigma).T
        self.s = self.s + K @ (z - H @ self.s)
        I_KH = np.eye(4) - K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ R @ K.T

    def _confirmed(self, k):
        """Re-initialise only on several recent, mutually consistent measurements."""
        recent = [b for b in self.buf if k - b[0] <= self.confirm_window]
        if len(recent) < self.confirm_n:
            return False
        th = np.array([b[1] for b in recent])
        return float(np.abs(th - np.median(th)).max()) <= self.confirm_spread

    def _reinit(self, k):
        recent = [b for b in self.buf if k - b[0] <= self.confirm_window]
        ks = np.array([b[0] for b in recent], float); th = np.array([b[1] for b in recent])
        before = self.s.copy()
        theta = float(np.median(th))
        if ks.max() - ks.min() >= 6:                                   # rate from the measurement history
            slope = float(np.polyfit((ks - ks.max()) * DT, th, 1)[0])
        else:
            slope = self.s[3]
        self.s[2], self.s[3] = theta, slope
        if self.fuse_x:
            self.s[0] = float(np.median([b[3] for b in recent]))
        sig_m = float(np.median([b[2] for b in recent]))
        self.Sigma = np.diag([max(self.Sigma[0, 0], 1e-4), max(self.Sigma[1, 1], 0.04), sig_m ** 2, 1.0])
        if self.hidden_reset == "zero":
            self.h = [m.init_hidden(1) for m in self.models]
        self.job = None                                                # in-flight correction is based on the old state
        self.last_reinit = k
        self.det.g = 0.0
        self.ev_log.append(dict(k=k, action="reinit_detail", before=before.tolist(), after=self.s.tolist()))
