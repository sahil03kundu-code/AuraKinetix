"""Closed-loop hardware-in-the-loop style simulation.

Plant physics runs at 5 kHz (200 us substeps); the event camera samples every
substep; the controller ticks at 1 kHz. Perception pipelines plug in through a
small interface so event-based and frame-based stacks are compared on the
exact same plant, disturbances and controller.
"""
from collections import deque
import numpy as np
import torch

from .physics import rk4_step, failed
from .event_camera import SceneRenderer, EventCamera, TimeSurfaceEncoder

DT = 1e-3
SUB = 5


class OraclePipeline:
    """Perfect, instantaneous state (upper bound - not physically achievable)."""
    name = "oracle"

    def reset(self, s0): self.s = s0.copy()

    def update(self, k, t, s_true, surface, u): self.s = s_true.copy()

    def estimate(self): return self.s


class FramePipeline:
    """Conventional frame camera + vision model.

    A frame is captured every 1/fps seconds; its result is available `latency`
    seconds later (readout + inference). Deliberately generous: the frame model
    is given the TRUE pose and velocity at capture time (no estimation error,
    no motion blur). Only the timing is realistic.
    """

    def __init__(self, fps, latency):
        self.fps, self.latency = fps, latency
        self.name = f"frame-{fps}fps"

    def reset(self, s0):
        self.pending = deque()
        self.s = s0.copy()
        self.next_capture = 0.0

    def update(self, k, t, s_true, surface, u):
        if t + 1e-9 >= self.next_capture:
            self.pending.append((t + self.latency, s_true.copy()))
            self.next_capture += 1.0 / self.fps
        while self.pending and self.pending[0][0] <= t + 1e-9:
            self.s = self.pending.popleft()[1]

    def estimate(self): return self.s


class EventPipeline:
    """AuraSync: event time-surface -> physics-informed observer, 1 tick latency."""
    name = "aurasync-event"

    def __init__(self, model, delay_ticks=1):
        self.model = model.eval()
        self.delay_ticks = delay_ticks

    def reset(self, s0):
        self.s_hat = torch.tensor(s0, dtype=torch.float32)[None]
        self.h = self.model.init_hidden(1)
        self.buf = deque([s0.copy()] * (self.delay_ticks + 1), maxlen=self.delay_ticks + 1)

    @torch.inference_mode()
    def update(self, k, t, s_true, surface, u):
        z = self.model.enc(torch.from_numpy(surface)[None])
        self.s_hat, self.h, _ = self.model.cell(z, self.h, self.s_hat,
                                                torch.tensor([u], dtype=torch.float32))
        self.buf.append(self.s_hat[0].numpy().astype(np.float64))

    def estimate(self): return self.buf[0]


def run_episode(p, controller, pipeline, n_ticks, s0, kicks=(), snap=None,
                exploration=None, rng=None, record=False, cam_seed=0,
                stop_on_fail=True, renderer=None, aps_fps=30, aps_latency=0.005):
    """kicks: list of (tick, delta_theta_dot). snap: (tick, new_params) mid-episode
    plant change (e.g. mass shift from a mechanical failure).

    The sensor is modelled as a hybrid DAVIS-style pixel array: asynchronous
    events PLUS a slow absolute intensity (APS) frame at `aps_fps`, delivered
    `aps_latency` after capture. Events give speed; the slow frame anchors
    drift when the scene is still and no events are produced."""
    renderer = renderer or SceneRenderer()
    cam = EventCamera(renderer.H, renderer.W, seed=cam_seed)
    enc = TimeSurfaceEncoder(renderer.H, renderer.W)
    s = np.array(s0, dtype=np.float64)
    cam.reset(renderer.render(s, p.l))
    kicks = dict(kicks)
    pipeline.reset(s)
    log = dict(s=[], s_hat=[], u=[], surf=[], events=[], fail_tick=None)
    aps = renderer.render(s, p.l).astype(np.float32)
    aps_pending, next_aps = deque(), 0.0
    t = 0.0
    u = 0.0
    for k in range(n_ticks):
        if snap is not None and k == snap[0]:
            p = snap[1]
        if k in kicks:
            s[3] += kicks[k]
        est = pipeline.estimate()
        u = controller.act(est)
        if exploration is not None:
            u = float(np.clip(u + exploration(), -p.u_max, p.u_max))
        n_ev = 0
        for j in range(SUB):
            s = rk4_step(s, u, DT / SUB, p)
            img = renderer.render(s, p.l)
            lp, ln, c = cam.step(img, t + j * DT / SUB, DT / SUB)
            enc.update(lp, ln)
            n_ev += c
            if aps_fps and t + j * DT / SUB >= next_aps - 1e-9:
                aps_pending.append((next_aps + aps_latency, img.astype(np.float32)))
                next_aps += 1.0 / aps_fps
        t += DT
        while aps_pending and aps_pending[0][0] <= t + 1e-9:
            aps = aps_pending.popleft()[1]
        surface = np.concatenate([enc.tensor(t), aps[None]])
        pipeline.update(k, t, s, surface, u)
        log["s"].append(s.copy()); log["u"].append(u); log["events"].append(n_ev)
        log["s_hat"].append(np.array(pipeline.estimate()))
        if record:
            log["surf"].append(surface.astype(np.float16))
        if failed(s, p):
            log["fail_tick"] = k
            if stop_on_fail:
                break
    for key in ("s", "s_hat", "u", "events"):
        log[key] = np.array(log[key])
    if record:
        log["surf"] = np.stack(log["surf"])
    return log
