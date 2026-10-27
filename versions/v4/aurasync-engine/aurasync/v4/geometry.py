"""Independent event-derived geometric measurement of the pole (V4, Modules A + B).

Analogy: instead of asking the navigator where the ship is, look out of the window
and line up the mast against the horizon. No memory, no learned model: just the
geometry of where events are firing right now.

The pole is a straight segment from the pivot (x_p, PIVOT_Y) at angle theta.
A moving pole fires events along its edges, so the fresh events (younger than
`window` seconds; their age is exactly recoverable from the time surface) lie
near the line
        x = x_p + (y - PIVOT_Y) * tan(theta)
Fitting x as a function of y is well conditioned for the angles that matter
(|theta| < 60 deg). The fit is robust (RANSAC + least-squares refinement on
inliers) because background noise events are scattered uniformly.

Output per call (or None when the evidence is insufficient):
    theta, x_pivot      measured pole angle and pivot position
    sigma_theta         measurement standard deviation (from inlier scatter,
                        inlier count and vertical extent, floored)
    n_inliers, span     support of the fit (span = fraction of pole length covered)
    snr                 inlier density / background event density
    quality in [0, 1]   combined reliability used for gating
Uses nothing but the sensor surface and the nominal pole length (no ground truth).
"""
import numpy as np

from ..event_camera import SceneRenderer

PIVOT_Y = 0.04
TAU_FAST = 0.008


class EventGeometry:
    def __init__(self, l=0.25, window=0.003, band_halfwidth=0.6, inlier_px=1.5, n_iter=60,
                 min_inliers=8, min_span=0.35, min_snr=4.0, sigma_floor_deg=0.4, seed=0):
        r = SceneRenderer()
        self.S = r.S
        self.wx, self.wy = r.wx, r.wy
        self.l, self.band = l, band_halfwidth
        self.fresh = float(np.exp(-window / TAU_FAST))
        self.inl = inlier_px / r.S
        self.n_iter, self.min_inliers, self.min_span, self.min_snr = n_iter, min_inliers, min_span, min_snr
        self.sigma_floor = np.radians(sigma_floor_deg)
        self.rng = np.random.default_rng(seed)
        self.y_lo, self.y_hi = PIVOT_Y + 0.06, PIVOT_Y + 2 * l + 0.04     # above the cart, up to the tip

    def events(self, surface):
        ev = (surface[0] >= self.fresh) | (surface[1] >= self.fresh)
        return ev

    def measure(self, surface, x_hint):
        """Fit the pole line to fresh events in a vertical band around x_hint (the current
        pivot estimate; only used to choose where to look)."""
        ev = self.events(surface)
        roi = (self.wy >= self.y_lo) & (self.wy <= self.y_hi) & (np.abs(self.wx - x_hint) <= self.band)
        sel = ev & roi
        n = int(sel.sum())
        if n < self.min_inliers:
            return None
        X, Y = self.wx[sel], self.wy[sel] - PIVOT_Y
        # background density from fresh events outside the region (per pixel)
        out = ev & ~roi & (self.wy > self.y_lo)
        bg_density = out.sum() / max((~roi & (self.wy > self.y_lo)).sum(), 1)
        # vectorised RANSAC: all hypotheses scored in one array operation
        i = self.rng.integers(0, n, self.n_iter); j = self.rng.integers(0, n, self.n_iter)
        dy = Y[i] - Y[j]
        ok = np.abs(dy) >= 0.08
        if not ok.any():
            return None
        i, j, dy = i[ok], j[ok], dy[ok]
        slope = (X[i] - X[j]) / dy                                          # tan(theta) per hypothesis
        keep = np.abs(slope) <= np.tan(np.radians(70))
        if not keep.any():
            return None
        slope, x0 = slope[keep], (X[i] - slope * Y[i])[keep]
        inl_all = np.abs(X[None, :] - (x0[:, None] + slope[:, None] * Y[None, :])) <= self.inl
        b = int(inl_all.sum(1).argmax())
        best_in = inl_all[b]
        if best_in.sum() < self.min_inliers:
            return None
        Xi, Yi = X[best_in], Y[best_in]
        A = np.stack([np.ones_like(Yi), Yi], 1)
        (x0, slope), *_ = np.linalg.lstsq(A, Xi, rcond=None)                # refine on inliers
        res = Xi - (x0 + slope * Yi)
        theta = float(np.arctan(slope))
        span = float((Yi.max() - Yi.min()) / (2 * self.l))
        # inlier band area (pixels) for the noise comparison
        band_px = 2 * self.inl * self.S * (Yi.max() - Yi.min()) * self.S + 1e-9
        snr = float(best_in.sum() / max(bg_density * band_px, 0.5))
        # angle uncertainty: horizontal scatter over the vertical lever arm (Gauss-Markov), floored
        sx = max(float(res.std()), 0.5 / self.S)
        sy = max(float(Yi.std()), 1e-3)
        sigma = float(np.hypot(sx / (sy * np.sqrt(best_in.sum())), self.sigma_floor)) / np.cos(theta) ** 2
        quality = float(np.clip(min(span / 0.8, snr / 10.0, best_in.sum() / 30.0), 0, 1))
        valid = span >= self.min_span and snr >= self.min_snr
        return dict(theta=theta, x_pivot=float(x0), sigma_theta=sigma, n_inliers=int(best_in.sum()),
                    n_events=n, span=span, snr=snr, quality=quality, valid=bool(valid))
