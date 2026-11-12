"""Baseline D: independent event-geometry estimator (no neural network).

Physics prediction every tick (same nominal RK4 model and process noise as the
V3 filter) + gated Kalman fusion of the event-geometry measurement (theta and
pivot x). Velocities are inferred by the filter through the dynamics. When no
valid measurement exists (quiet scene) it coasts on physics with growing
covariance. Same pipeline interface as every other estimator.
"""
import numpy as np

from ..physics import PlantParams, rk4_step, linearise
from ..estimator import SCALE
from .geometry import EventGeometry

DT = 1e-3
SCALE_NP = SCALE.numpy().astype(np.float64)


class GeoPipeline:
    def __init__(self, Q=None, gate=9.0, sigma_x_floor=0.004, plant=None, seed=0, name="event-geometry"):
        self.p = plant or PlantParams()
        self.A, _ = linearise(self.p, DT)
        self.Q = np.diag([1e-9, 1e-5, 1e-8, 1e-4]) * 1e4 if Q is None else Q
        self.gate, self.sx = gate, sigma_x_floor
        self.geo = EventGeometry(seed=seed + 1)
        self.seed, self.name = seed, name
        self.delay_ticks = 0

    def reset(self, s0):
        self.geo.rng = np.random.default_rng(self.seed + 1)
        self.s = np.asarray(s0, float).copy()
        self.Sigma = np.diag((0.05 * SCALE_NP) ** 2)
        self.n_fused = 0

    def update(self, k, t, s_true, surface, u):
        self.s = rk4_step(self.s, u, DT, self.p)
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.Q
        m = self.geo.measure(surface, self.s[0])
        if m is None or not m["valid"]:
            return
        H = np.array([[0, 0, 1.0, 0], [1.0, 0, 0, 0]])
        z = np.array([m["theta"], m["x_pivot"]])
        R = np.diag([m["sigma_theta"] ** 2, max(self.sx, 0.5 / self.geo.S / np.sqrt(m["n_inliers"])) ** 2])
        S = H @ self.Sigma @ H.T + R
        nu = z - H @ self.s
        if nu[0] ** 2 / S[0, 0] > self.gate and self.n_fused > 50:        # outlier rejection once tracking
            return
        K = np.linalg.solve(S, H @ self.Sigma).T
        self.s = self.s + K @ nu
        I_KH = np.eye(4) - K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ R @ K.T
        self.n_fused += 1

    def estimate(self):
        return self.s

    def covariance(self):
        return self.Sigma

    def age_ticks(self):
        return 0
