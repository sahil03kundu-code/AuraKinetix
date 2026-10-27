"""Physics-informed event observer (PINN + selective state-space model).

Analogy: a blindfolded juggler who knows physics. Between glimpses, the juggler
predicts where the ball is from Newton's laws (the physics prior). Every event
spike is a tiny glimpse; the network only learns how much to *correct* the
physical prediction. When the scene is still and no events arrive, the
estimate simply keeps following the physics instead of going blind.

    s_{k+1} = RK4(s_k, u_k)                      # differentiable physics
            + G * MLP([z_{k+1}, h_{k+1}, s_k])   # learned event correction
    z_k     = CNN(time_surface_k)                # event features
    h_{k+1} = a(z) * h_k + (1 - a(z)) * B z      # selective SSM memory
    a(z)    = exp(-softplus(W_d z) * lambda)     # input-dependent step (Mamba-style)

Training loss = state error + physics residuals (kinematic consistency and a
penalty on unexplained corrections) - i.e. the network is told that the
physics is mostly right and must justify any deviation with evidence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE = torch.tensor([0.3, 1.0, 0.2, 2.0])  # typical magnitude of x, xd, th, thd


class TorchCartPole(nn.Module):
    def __init__(self, M=1.0, m=0.1, l=0.25, g=9.81, b_cart=0.1, b_pole=0.002):
        super().__init__()
        self.M, self.m, self.l, self.g = M, m, l, g
        self.b_cart, self.b_pole = b_cart, b_pole

    def deriv(self, s, u):
        x, xd, th, thd = s.unbind(-1)
        sin, cos = torch.sin(th), torch.cos(th)
        total = self.M + self.m
        temp = (u + self.m * self.l * thd ** 2 * sin - self.b_cart * xd) / total
        thdd = (self.g * sin - cos * temp - self.b_pole * thd / (self.m * self.l)) / (
            self.l * (4.0 / 3.0 - self.m * cos ** 2 / total))
        xdd = temp - self.m * self.l * thdd * cos / total
        return torch.stack([xd, xdd, thd, thdd], -1)

    def step(self, s, u, dt):
        k1 = self.deriv(s, u)
        k2 = self.deriv(s + 0.5 * dt * k1, u)
        k3 = self.deriv(s + 0.5 * dt * k2, u)
        k4 = self.deriv(s + dt * k3, u)
        return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class SpatialSoftmaxKeypoints(nn.Module):
    """Full-resolution keypoint layer (sub-pixel precise).

    Each of K learned heatmaps is turned into a probability map; its mean gives
    a sub-pixel (x, y) location and its covariance gives the orientation and
    spread of the blob - e.g. the angle of the pole. This preserves the fine
    geometry that strided pooling throws away (1 px at the pole tip ~ 3 deg).
    """

    def __init__(self, cin, H, W, K=12):
        super().__init__()
        self.c1 = nn.Conv2d(cin, 16, 3, 1, 1)
        self.c2 = nn.Conv2d(16, K, 3, 1, 1)
        self.log_temp = nn.Parameter(torch.zeros(K))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W) * (W / H), indexing="ij")
        self.register_buffer("gx", xx.flatten())
        self.register_buffer("gy", yy.flatten())
        self.out_dim = 5 * K

    def forward(self, x):
        logits = self.c2(F.gelu(self.c1(x))).flatten(2) * torch.exp(self.log_temp)[None, :, None]
        p = torch.softmax(logits, -1)
        mx, my = (p * self.gx).sum(-1), (p * self.gy).sum(-1)
        vxx = (p * self.gx ** 2).sum(-1) - mx ** 2
        vyy = (p * self.gy ** 2).sum(-1) - my ** 2
        vxy = (p * self.gx * self.gy).sum(-1) - mx * my
        return torch.cat([mx, my, 10 * vxx, 10 * vyy, 10 * vxy], -1)


class EventEncoder(nn.Module):
    def __init__(self, H=32, W=96, d=128, cin=4):
        super().__init__()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy])[None])
        self.c1 = nn.Conv2d(cin + 2, 16, 5, 2, 2)
        self.c2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.c3 = nn.Conv2d(32, 32, 3, 2, 1)
        self.kp = SpatialSoftmaxKeypoints(cin, H, W)
        self.fc = nn.Linear(32 * (H // 8) * (W // 8) + self.kp.out_dim, d)

    def forward(self, ts):  # ts: (N, 4, H, W): ON, OFF, slow-any, APS frame
        x = torch.cat([ts, self.coords.expand(ts.shape[0], -1, -1, -1)], 1)
        x = F.gelu(self.c1(x)); x = F.gelu(self.c2(x)); x = F.gelu(self.c3(x))
        return F.gelu(self.fc(torch.cat([x.flatten(1), self.kp(ts)], 1)))


class EventObserver(nn.Module):
    def __init__(self, dt=1e-3, d=128, n_state=64):
        super().__init__()
        self.dt = dt
        self.physics = TorchCartPole()
        self.enc = EventEncoder(d=d)
        self.B = nn.Linear(d, n_state)
        self.W_delta = nn.Linear(d, n_state)
        self.log_lambda = nn.Parameter(torch.zeros(n_state))
        self.head = nn.Sequential(nn.Linear(d + n_state + 4, 128), nn.GELU(),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("scale", SCALE.clone())
        self.n_state = n_state

    def init_hidden(self, batch):
        return torch.zeros(batch, self.n_state)

    def cell(self, z, h, s, u):
        """One 1 ms observer tick. z: event features at t_{k+1}; s, u at t_k."""
        a = torch.exp(-F.softplus(self.W_delta(z)) * torch.exp(self.log_lambda))
        h = a * h + (1 - a) * self.B(z)
        s_pred = self.physics.step(s, u, self.dt)
        corr = self.head(torch.cat([z, h, s / self.scale], -1))
        # correction expressed in units of "typical magnitude per 50 ms"
        s_new = s_pred + corr * self.scale * (self.dt / 0.05)
        return s_new, h, corr

    def forward(self, ts_seq, u_seq, s0):
        """ts_seq: (B, T, 4, H, W) surfaces at t_1..t_T; u_seq: (B, T); s0: (B, 4)."""
        Bsz, T = ts_seq.shape[:2]
        z_all = self.enc(ts_seq.flatten(0, 1)).view(Bsz, T, -1)
        h = self.init_hidden(Bsz)
        s = s0
        out, corrs = [], []
        for k in range(T):
            s, h, c = self.cell(z_all[:, k], h, s, u_seq[:, k])
            out.append(s); corrs.append(c)
        return torch.stack(out, 1), torch.stack(corrs, 1)


def physics_informed_loss(s_hat, s_true, corr, s0, dt, w_kin=0.05, w_corr=1e-3):
    scale = SCALE.to(s_hat)
    # tilt and tilt-rate errors matter most to the controller: weight them up
    w = torch.tensor([1.0, 1.0, 4.0, 2.0]).to(s_hat)
    data = ((((s_hat - s_true) / scale) ** 2) * w).mean() / w.mean()
    # kinematic residual: position change must match estimated velocity
    s_prev = torch.cat([s0[:, None], s_hat[:, :-1]], 1)
    dpos = (s_hat[..., [0, 2]] - s_prev[..., [0, 2]]) / dt
    vel = 0.5 * (s_hat[..., [1, 3]] + s_prev[..., [1, 3]])
    kin = (((dpos - vel) / scale[[1, 3]]) ** 2).mean()
    # parsimony: corrections must be justified by evidence
    reg = (corr ** 2).mean()
    return data + w_kin * kin + w_corr * reg, dict(data=data.item(), kin=kin.item(), reg=reg.item())
