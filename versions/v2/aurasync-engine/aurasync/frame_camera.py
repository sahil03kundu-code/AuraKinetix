"""Learned frame-camera baseline (fair comparison for the event stack).

The phase-1 frame baselines were handed the TRUE state at capture time. Here
the frame stack has to *see* the state the same way the event stack does:

  sensor     global-shutter grayscale camera on the same 96x32 scene,
             exposure-integrated motion blur and read noise
  estimator  FrameEstimator: ResNet-lite trunk over the last 3 frames plus a
             shared per-frame spatial-softmax keypoint encoder (the same
             sub-pixel tool the event observer uses), conditioned on frame
             period and exposure. Regresses the full state at capture time.
  pipeline   LearnedFramePipeline: at each frame arrival (capture + latency)
             the estimate is rolled forward to "now" through the same nominal
             physics model the event observer uses, replaying the controls
             applied since capture (latency compensation, a Smith predictor).
             Between frames the estimate keeps propagating at 1 kHz.
             mode="zoh" disables both (naive hold, for the ablation).

Analogy: the frame stack is a goalkeeper watching a strobe light. Between
flashes he still knows physics, so he extrapolates the ball's path rather than
freezing on the last flash. The question is whether that beats someone who
sees a continuous trickle of motion (the event camera).
"""
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import PlantParams, rk4_step
from .event_camera import SceneRenderer
from .estimator import SpatialSoftmaxKeypoints, SCALE

N_FRAMES = 3
READ_NOISE = 0.02  # std of per-pixel read noise (scene intensities are 0.15..0.9)


def blur_frame(history, exposure_ticks):
    """Average the last `exposure_ticks` sharp 1 ms renders = shutter open over that window."""
    return np.mean(np.stack(list(history)[-exposure_ticks:]), 0)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None if (stride == 1 and cin == cout) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class FrameEstimator(nn.Module):
    """frames (N, 3, H, W) newest-first, cond (N, 2) = [frame period/33ms, exposure/8ms] -> state (N, 4)."""

    def __init__(self, H=32, W=96, K=12):
        super().__init__()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy])[None])
        self.trunk = nn.Sequential(
            nn.Conv2d(N_FRAMES + 2, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            BasicBlock(16, 16), BasicBlock(16, 32, 2), BasicBlock(32, 32),
            BasicBlock(32, 48, 2), BasicBlock(48, 48, 2))
        self.kp = SpatialSoftmaxKeypoints(1, H, W, K=K)
        self.fc = nn.Linear(48 * (H // 8) * (W // 8), 128)
        self.head = nn.Sequential(nn.Linear(128 + N_FRAMES * self.kp.out_dim + 2, 128), nn.GELU(),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4))
        self.register_buffer("scale", SCALE.clone())

    def forward(self, frames, cond):
        n = frames.shape[0]
        x = (frames - 0.5) / 0.3
        g = F.gelu(self.fc(self.trunk(torch.cat([x, self.coords.expand(n, -1, -1, -1)], 1)).flatten(1)))
        kps = self.kp(x.reshape(n * N_FRAMES, 1, *x.shape[2:])).view(n, -1)
        return self.head(torch.cat([g, kps, cond], 1)) * self.scale


def cond_vector(fps, exposure):
    return torch.tensor([[(1.0 / fps) / 0.0333, exposure / 0.008]], dtype=torch.float32)


class LearnedFramePipeline:
    """Frame camera + FrameEstimator, with physics latency compensation (mode='physics') or naive hold ('zoh')."""

    def __init__(self, model, fps, latency, exposure=0.004, mode="physics", l=0.25,
                 plant=None, seed=0, read_noise=READ_NOISE):
        self.model = model.eval()
        self.fps, self.latency, self.exposure, self.mode = fps, latency, exposure, mode
        self.exp_ticks = max(1, int(round(exposure / 1e-3)))
        self.l = l
        self.p = plant or PlantParams()  # nominal physics model, same as the event observer's
        self.seed, self.read_noise = seed, read_noise
        self.renderer = SceneRenderer()
        tag = "" if mode == "physics" else "-zoh"
        self.name = f"learned-frame-{fps}fps{tag}" + ("" if abs(exposure - 0.004) < 1e-9 else f"-exp{exposure*1e3:g}ms")

    def reset(self, s0):
        self.rng = np.random.default_rng(self.seed)
        sharp = self.renderer.render(s0, self.l)
        self.hist = deque([sharp] * 16, maxlen=16)      # sharp 1 ms renders (for blur)
        self.frames = deque(maxlen=N_FRAMES)            # captured (blurred, noisy) frames
        self.pending = deque()                          # (arrival_t, capture_t, frame stack)
        self.u_hist = deque(maxlen=200)                 # controls applied per tick
        self.s = np.array(s0, dtype=np.float64)
        self.next_capture = 0.0
        self.cond = cond_vector(self.fps, self.exposure)

    def _capture(self):
        img = blur_frame(self.hist, self.exp_ticks)
        img = img + self.rng.normal(0, self.read_noise, img.shape)
        self.frames.appendleft(img.astype(np.float32))
        while len(self.frames) < N_FRAMES:      # very first capture: repeat it
            self.frames.append(self.frames[-1])
        return np.stack(self.frames)

    @torch.inference_mode()
    def update(self, k, t, s_true, surface, u):
        self.u_hist.append(u)
        if self.mode == "physics":            # propagate current estimate across this tick
            self.s = rk4_step(self.s, u, 1e-3, self.p)
        self.hist.append(self.renderer.render(s_true, self.l))
        if t + 1e-9 >= self.next_capture:
            self.pending.append((t + self.latency, t, self._capture()))
            self.next_capture += 1.0 / self.fps
        while self.pending and self.pending[0][0] <= t + 1e-9:
            _, t_cap, stack = self.pending.popleft()
            s_cap = self.model(torch.from_numpy(stack)[None], self.cond)[0].double().numpy()
            if self.mode == "physics":         # roll forward from capture time to now
                n = int(round((t - t_cap) / 1e-3))
                us = list(self.u_hist)[-n:] if n > 0 else []
                for uu in us:
                    s_cap = rk4_step(s_cap, uu, 1e-3, self.p)
            self.s = s_cap

    def estimate(self):
        return self.s
