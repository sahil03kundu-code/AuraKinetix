"""Uncertainty- and delay-aware safety shield (Upgrade B).

Analogy: a driving instructor with a second brake pedal. The student (LQR)
steers; the instructor only intervenes when the car is heading for the kerb
faster than it could stop, and leans in earlier when visibility is poor
(high estimate uncertainty) or when the student's view is out of date (delay).

Formulation (discrete-time control barrier function, one input, closed form):
  Look-ahead coordinates (relative degree 1 in the force u):
      psi = theta + c_th * theta_dot        chi = x + x_dot |x_dot| / (2 a_brake)
  Safe set S = { |psi| <= psi_max  and  |chi| <= chi_max }. If |psi| <= psi_max
  is kept forever, theta converges into the same band (first-order filter), so
  this bounds the pole angle itself.
  At each 1 ms tick, with the estimate first rolled forward over its age
  (delay compensation, replaying the controls already applied), the safe set is
  tightened by the estimate's uncertainty and the discrete CBF condition is
  imposed on the tightened barriers:
      h~ = (psi_max - kappa * sigma_psi) -/+ psi,   h~(s_{k+1}) >= (1 - gamma) h~(s_k)
  (same for chi). sigma_psi is the barrier's standard deviation from the
  estimator covariance propagated one step through the linearised physics, so
  a vaguer or older estimate keeps the pole further from the edge (floor: 25%
  of the nominal set). Dynamics are affine in u, so each constraint is an interval on u;
  the QP  min (u - u_lqr)^2  becomes a clip onto the intersection with the
  actuator limits.
Priorities and fallback (hard limits before soft targets):
  HARD  track end   |chi| <= chi_max, chi = x + x_dot|x_dot| / (2 a_brake): position plus the
                    distance needed to stop at a_brake = 12 m/s^2 (actuator limit ~18 m/s^2)
                    must stay inside 0.95 m (track 1.0 m) -> leaving it is a failure
  HARD  pole        |psi| <= psi_hard (35 deg, falling is 46 deg)
  SOFT  envelope    |psi| <= psi_max  (10 deg, the safety target), pursued only while
                    |chi| <= chi_soft (0.6 m): the last part of the track is a reserve the
                    soft target may never spend, so chasing it cannot strand the cart;
                    and only from states with |psi| <= 2 psi_max (reachability gate, latched
                    until settled: |psi| <= psi_max, |x_dot| < 0.3 m/s, |theta_dot| < 0.5 rad/s).
                    After a hit too big to contain, chasing the target mid-recovery fights
                    the nominal controller and walks the cart off the track
  1. hard + soft + actuator feasible -> project u_lqr
  2. else drop the soft envelope     -> project onto hard + actuator (logged 'soft_dropped')
  3. else infeasible                 -> do no harm: pass the nominal command through (logged).
                                        An earlier variant that slammed the actuator to the
                                        most-violated barrier lost runs plain LQR survives.
  (v3 first shipped with the soft envelope ranked above the track limit; on test trials
  that drove the cart off the track in 21% of oracle runs. Fixed before reporting.)
  Stale estimate (sigma_theta > stale_deg) -> the nominal command comes from a
  conservative low-gain LQR before shielding (logged).
"""
import time
from collections import deque
import numpy as np

from .physics import PlantParams, rk4_step, linearise
from .controller import LQRController

DT = 1e-3


class SafetyShield:
    def __init__(self, plant=None, psi_max_deg=10.0, c_th=0.10, chi_max=0.95, a_brake=12.0,
                 gamma=0.03, kappa=2.0, stale_deg=4.0, Q_proc=None, psi_hard_deg=35.0, chi_soft=0.6,
                 gamma_hard=0.03, soft_gate=2.0):
        self.p = plant or PlantParams()
        self.psi_max = np.radians(psi_max_deg)
        self.psi_hard = np.radians(psi_hard_deg)
        self.c_th, self.chi_max, self.a_brake, self.chi_soft = c_th, chi_max, a_brake, chi_soft
        self.gamma, self.kappa, self.gamma_hard, self.soft_gate = gamma, kappa, gamma_hard, soft_gate
        self.soft_off = False
        self.stale = np.radians(stale_deg)
        self.A, _ = linearise(self.p, DT)
        self.Q = np.diag([1e-9, 1e-5, 1e-8, 1e-4]) if Q_proc is None else Q_proc
        self.psi = lambda z: z[2] + c_th * z[3]
        self.chi = lambda z: z[0] + z[1] * abs(z[1]) / (2 * a_brake)
        self.fallback_ctrl = LQRController(self.p, Q=(10.0, 1.0, 100.0, 2.0), R=0.1)
        self.u_max = self.p.u_max

    @staticmethod
    def _interval(a0, a1, lo, hi):
        """{u : lo <= a0 + a1 u <= hi} as (u_lo, u_hi)."""
        if abs(a1) < 1e-12:
            return (-np.inf, np.inf) if lo <= a0 <= hi else (np.inf, -np.inf)
        u1, u2 = (lo - a0) / a1, (hi - a0) / a1
        return (min(u1, u2), max(u1, u2))

    def _bounds(self, s, Sigma, fn, h_max, gamma=None):
        """Interval on u for the pair of barriers h_max -/+ fn(s_next) (fn affine in u over one tick)."""
        n0 = rk4_step(s, 0.0, DT, self.p)
        n1 = rk4_step(s, 1.0, DT, self.p)
        a0, a1, cur = fn(n0), fn(n1) - fn(n0), fn(s)
        g = np.array([(fn(n0 + e) - fn(n0 - e)) / 2e-6 for e in np.eye(4) * 1e-6])   # barrier gradient
        S1 = self.A @ Sigma @ self.A.T + self.Q
        m = self.kappa * np.sqrt(max(g @ S1 @ g, 0.0))
        hm = max(h_max - m, 0.25 * h_max)                 # robust tightened safe set |g.s| <= h_max - m
        gm = self.gamma if gamma is None else gamma
        hi = hm - (1 - gm) * (hm - cur)
        lo = -hm + (1 - gm) * (hm + cur)
        return self._interval(a0, a1, lo, hi), (a0, a1, lo, hi)



    def filter(self, s, Sigma, u_nom):
        """Returns (u_safe, mode). mode: 'pass', 'modified', 'soft_dropped', 'infeasible'."""

        if not np.isfinite(u_nom):
            raise ValueError("Nominal actuator force must be finite")

        s = np.asarray(s, dtype=float)
        Sigma = np.asarray(Sigma, dtype=float)

        if s.shape != (4,):
            raise ValueError("state must contain exactly 4 values")

        if Sigma.shape != (4, 4):
            raise ValueError("covariance must be a 4x4 matrix")

        if not np.all(np.isfinite(s)):
            raise ValueError("state must contain only finite values")

        if not np.all(np.isfinite(Sigma)):
            raise ValueError("covariance must contain only finite values")


        (sl, sh), _ = self._bounds(
            s, Sigma, self.psi, self.psi_max
        )
        (pl, ph), pk = self._bounds(
            s, Sigma, self.psi, self.psi_hard, self.gamma_hard
        )
        (cl, ch), ck = self._bounds(
            s, Sigma, self.chi, self.chi_max, self.gamma_hard
        )
        (rl, rh), _ = self._bounds(
            s, Sigma, self.chi, self.chi_soft
        )

        ul, uh = -self.u_max, self.u_max
        L = max(sl, rl, pl, cl, ul)
        H = min(sh, rh, ph, ch, uh)

        a = abs(self.psi(s))

        if a > self.soft_gate * self.psi_max:
            self.soft_off = True
        elif (
            a <= self.psi_max
            and abs(s[1]) < 0.3
            and abs(s[3]) < 0.5
        ):
            self.soft_off = False

        if self.soft_off:
            L, H = 1.0, -1.0

        if L <= H:
            u = float(np.clip(u_nom, L, H))
            return u, (
                "pass" if abs(u - u_nom) < 1e-9 else "modified"
            )

        L = max(pl, cl, ul)
        H = min(ph, ch, uh)

        if L <= H:
            return float(np.clip(u_nom, L, H)), "soft_dropped"

        return float(np.clip(u_nom, ul, uh)), "infeasible"


class ShieldedController:
    """LQR -> (stale check) -> delay compensation -> SafetyShield.

    Reads the pipeline's covariance (`covariance()`) and estimate age
    (`age_ticks()`) when it offers them; otherwise uses `sigma_const` and the
    pipeline's fixed delay."""

    def __init__(self, lqr, shield, pipeline, sigma_const=None, enabled=True):
        self.lqr, self.shield, self.pipe = lqr, shield, pipeline
        self.sigma_const = np.diag(np.square(sigma_const)) if sigma_const is not None else np.zeros((4, 4))
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.u_hist = deque(maxlen=64)
        self.stats = dict(pass_=0, modified=0, soft_dropped=0, infeasible=0, stale=0, ticks=0)
        self.times = []

    def _age(self):
        if hasattr(self.pipe, "age_ticks"):
            return int(self.pipe.age_ticks())
        return int(getattr(self.pipe, "delay_ticks", 0))

    def _cov(self):
        if hasattr(self.pipe, "covariance"):
            return np.asarray(self.pipe.covariance())
        return self.sigma_const

    def act(self, est):
        t0 = time.perf_counter()
        s = np.asarray(est, float)
        Sigma = self._cov()
        age = self._age()
        us = list(self.u_hist)[-age:] if age > 0 else []
        for u in us:                                     # delay compensation
            s = rk4_step(s, u, DT, self.shield.p)
            Sigma = self.shield.A @ Sigma @ self.shield.A.T + self.shield.Q
        self.stats["ticks"] += 1
        stale = np.sqrt(max(Sigma[2, 2], 0)) > self.shield.stale
        if stale:
            self.stats["stale"] += 1
            u_nom = self.shield.fallback_ctrl.act(s)
        else:
            u_nom = self.lqr.act(s)
        if self.enabled:
            u, mode = self.shield.filter(s, Sigma, u_nom)
            self.stats["pass_" if mode == "pass" else mode] += 1
        else:
            u = u_nom
        self.u_hist.append(u)
        self.times.append(time.perf_counter() - t0)
        return u
