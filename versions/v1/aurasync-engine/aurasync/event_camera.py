"""Event-camera simulator (ESIM-style) and the scene renderer.

A real event camera pixel does not take pictures. Each pixel watches its own
log-brightness and fires an event (x, y, t, polarity) every time that value has
moved by a contrast threshold C since the pixel's last event. Static scenes
produce almost nothing; moving edges produce microsecond-timestamped spikes.

We reproduce that from a finely-sampled rendered scene:
  * per-pixel log-intensity reference, per-pixel threshold mismatch
  * multiple crossings per step, with linearly interpolated timestamps
  * background-activity noise events
"""
import numpy as np


class SceneRenderer:
    """Renders the cart-pole as a grayscale image (antialiased, H x W)."""

    def __init__(self, H=32, W=96, px_per_m=40.0, y_min=-0.12, bg=0.15, fg=0.9):
        self.H, self.W, self.S = H, W, px_per_m
        jj, ii = np.meshgrid(np.arange(W), np.arange(H))
        self.wx = (jj + 0.5 - W / 2) / px_per_m
        self.wy = y_min + (H - ii - 0.5) / px_per_m
        self.bg, self.fg = bg, fg

    def render(self, s, l):
        x, th = s[0], s[2]
        S, wx, wy = self.S, self.wx, self.wy
        # cart: 0.2 m x 0.08 m box (signed-distance antialiasing)
        dx = np.abs(wx - x) - 0.10
        dy = np.abs(wy - 0.0) - 0.04
        d_box = np.maximum(dx, dy)
        cov_cart = np.clip(0.5 - d_box * S, 0, 1)
        # pole: segment from pivot to tip, ~1 px thick
        px, py = x, 0.04
        tx, ty = x + 2 * l * np.sin(th), py + 2 * l * np.cos(th)
        vx, vy = tx - px, ty - py
        t = np.clip(((wx - px) * vx + (wy - py) * vy) / (vx * vx + vy * vy), 0, 1)
        d_seg = np.hypot(wx - (px + t * vx), wy - (py + t * vy))
        cov_pole = np.clip(0.5 - (d_seg - 0.013) * S, 0, 1)
        cov = np.maximum(cov_cart, cov_pole)
        return self.bg + (self.fg - self.bg) * cov


class EventCamera:
    """Converts a stream of rendered frames into asynchronous events.

    Rather than materialising every event object (memory-hungry at MHz rates)
    the simulator reports, per pixel and polarity, the timestamp of the latest
    crossing plus counts. `record=True` additionally keeps a raw (x, y, t, p)
    list for visualisation.
    """

    def __init__(self, H, W, C=0.18, C_sigma=0.03, noise_rate_hz=0.3, eps=0.02,
                 seed=0, record=False):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.Cp = np.clip(rng.normal(C, C_sigma, (H, W)), 0.05, None)
        self.Cn = np.clip(rng.normal(C, C_sigma, (H, W)), 0.05, None)
        self.noise_rate = noise_rate_hz
        self.eps = eps
        self.H, self.W = H, W
        self.L_ref = None
        self.L_prev = None
        self.record = record
        self.events = []
        self.n_events = 0

    def reset(self, img):
        self.L_ref = np.log(img + self.eps)
        self.L_prev = self.L_ref.copy()

    def step(self, img, t0, dt):
        """Process scene change between t0 and t0+dt.

        Returns (last_t_pos, last_t_neg) arrays with the crossing timestamps in
        this interval (NaN where the pixel was silent) and the event count.
        """
        L = np.log(img + self.eps)
        dL = L - self.L_prev
        diff = L - self.L_ref
        out_p = np.full((self.H, self.W), np.nan)
        out_n = np.full((self.H, self.W), np.nan)
        count = 0
        safe = np.where(np.abs(dL) > 1e-9, dL, 1e-9)

        n_pos = np.floor(np.maximum(diff, 0) / self.Cp)
        mp = n_pos >= 1
        if mp.any():
            level = self.L_ref[mp] + n_pos[mp] * self.Cp[mp]
            frac = np.clip((level - self.L_prev[mp]) / safe[mp], 0, 1)
            out_p[mp] = t0 + frac * dt
            self.L_ref[mp] = level
            count += int(n_pos[mp].sum())
        n_neg = np.floor(np.maximum(-diff, 0) / self.Cn)
        mn = n_neg >= 1
        if mn.any():
            level = self.L_ref[mn] - n_neg[mn] * self.Cn[mn]
            frac = np.clip((level - self.L_prev[mn]) / safe[mn], 0, 1)
            out_n[mn] = t0 + frac * dt
            self.L_ref[mn] = level
            count += int(n_neg[mn].sum())

        # background-activity noise (leak / shot noise)
        p_noise = self.noise_rate * dt
        noise = self.rng.random((self.H, self.W)) < p_noise
        if noise.any():
            pol = self.rng.random(noise.sum()) < 0.5
            ts = t0 + self.rng.random(noise.sum()) * dt
            tgt_p = out_p[noise]; tgt_n = out_n[noise]
            tgt_p[pol] = np.fmax(tgt_p[pol], ts[pol])
            tgt_n[~pol] = np.fmax(tgt_n[~pol], ts[~pol])
            out_p[noise] = tgt_p; out_n[noise] = tgt_n
            count += int(noise.sum())

        if self.record:
            for pol, arr in ((1, out_p), (-1, out_n)):
                ys, xs = np.nonzero(~np.isnan(arr))
                for y, x in zip(ys, xs):
                    self.events.append((x, y, arr[y, x], pol))
        self.L_prev = L
        self.n_events += count
        return out_p, out_n, count


class TimeSurfaceEncoder:
    """Event-to-tensor compression: fixed-size, exponentially-decaying time surface.

    Every pixel/polarity keeps only the timestamp of its latest event. When the
    controller asks for a state at time t, the surface is
        T[c, y, x] = exp(-(t - t_last[c, y, x]) / tau_c)
    Memory is constant (H x W x channels) no matter how many events arrive, so
    there is no buffer bloat and no dropped events at high event rates; the
    newest events dominate and old ones fade continuously.

    Channels: [pos fast, neg fast, any slow]
    """

    def __init__(self, H, W, tau_fast=0.008, tau_slow=0.040):
        self.tau_fast, self.tau_slow = tau_fast, tau_slow
        self.t_pos = np.full((H, W), -1e9)
        self.t_neg = np.full((H, W), -1e9)

    def update(self, last_p, last_n):
        self.t_pos = np.fmax(self.t_pos, last_p)
        self.t_neg = np.fmax(self.t_neg, last_n)

    def tensor(self, t):
        tp = np.exp(-(t - self.t_pos) / self.tau_fast)
        tn = np.exp(-(t - self.t_neg) / self.tau_fast)
        ta = np.exp(-(t - np.maximum(self.t_pos, self.t_neg)) / self.tau_slow)
        return np.stack([tp, tn, ta]).astype(np.float32)

    @property
    def nbytes(self):
        return self.t_pos.nbytes + self.t_neg.nbytes
