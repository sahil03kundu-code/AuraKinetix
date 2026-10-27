"""Event-consistency monitor: an independent check on the observer's estimate.

Analogy: a lifeguard told "the swimmer is by the red buoy" glances at where
the splashes actually are. If the splashes keep coming from somewhere else,
the report is wrong, however confident the person who gave it sounds.

Why it is needed: the learned variance head stays confident while the observer
diverges onto a wrong (even mirror-image) pole angle, because the divergence
happens on trajectories it never saw in training. This monitor makes no use of
the network at all; it checks the estimate against raw events:
  * a moving edge fires events where the pole actually is
  * inside a fan around the estimated pivot (where the pole could be), compare
    the density of fresh events (< 2 ms old) close to the PREDICTED pole
    (<= 2 px) with the density elsewhere in the fan (> 3 px away)
  * densities, not counts, so uniform background noise cancels out
      r = d_far / (d_near + d_far)    ~0 consistent, ~0.5 uninformative, -> 1 contradicted
  * ticks with fewer than `min_events` fresh events in the fan carry no
    information (quiet scene) and leave the running score unchanged
  * m = exponential moving average of r over informative ticks
The monitor turns into an uncertainty by inflating the observer's variance:
  sigma_theta^2 <- sigma_theta^2 + (beta * max(0, m - m0))^2
with m0 and beta fitted on validation episodes only.
"""
import numpy as np

from .event_camera import SceneRenderer

FRESH2 = float(np.exp(-2e-3 / 0.008))     # fast time-surface value of an event < 2 ms old


class EventConsistencyMonitor:
    def __init__(self, l=0.25, fan_deg=60.0, near_px=2.0, far_px=3.0, min_events=6, ema=0.1,
                 m0=0.25, beta=0.5):
        self.r = SceneRenderer()
        self.l, self.fan = l, np.radians(fan_deg)
        self.near_px, self.far_px, self.min_events, self.ema = near_px, far_px, min_events, ema
        self.m0, self.beta = m0, beta
        self.reset()

    def reset(self):
        self.m = self.m0
        self.info_ticks = 0

    def raw_score(self, s, surface):
        """r for one tick, or None if the tick carries no information."""
        S, wx, wy = self.r.S, self.r.wx, self.r.wy
        x, th = float(s[0]), float(s[2])
        px, py = x, 0.04
        dx, dy = wx - px, wy - py
        dist_piv = np.hypot(dx, dy)
        ang = np.arctan2(dx, dy)                                   # 0 = straight up, + = right
        fan = (dist_piv <= 2 * self.l + 3 / S) & (dist_piv >= 0.06) & (np.abs(ang) <= self.fan)
        tx, ty = px + 2 * self.l * np.sin(th), py + 2 * self.l * np.cos(th)
        vx, vy = tx - px, ty - py
        t = np.clip((dx * vx + dy * vy) / (vx * vx + vy * vy), 0, 1)
        d_pole_px = np.hypot(wx - (px + t * vx), wy - (py + t * vy)) * S
        near = fan & (d_pole_px <= self.near_px)
        far = fan & (d_pole_px > self.far_px)
        ev = (surface[0] >= FRESH2) | (surface[1] >= FRESH2)
        n_near, n_far = int((ev & near).sum()), int((ev & far).sum())
        if n_near + n_far < self.min_events or near.sum() == 0 or far.sum() == 0:
            return None
        d_near, d_far = n_near / near.sum(), n_far / far.sum()
        return d_far / (d_near + d_far + 1e-9)

    def update(self, s, surface):
        r = self.raw_score(s, surface)
        if r is not None:
            self.m += self.ema * (r - self.m)
            self.info_ticks += 1
        return self.m

    def inflate(self, sigma_theta):
        return np.sqrt(sigma_theta ** 2 + (self.beta * max(0.0, self.m - self.m0)) ** 2)


def monitor_trajectory(est, surf, l=0.25, **kw):
    """Run the monitor over a recorded trajectory. est: (T, 4); surf: (T, 4, H, W). Returns m_t."""
    mon = EventConsistencyMonitor(l=l, **kw)
    return np.array([mon.update(est[k], surf[k]) for k in range(len(est))])
