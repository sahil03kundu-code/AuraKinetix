"""Step 1 probe (VALIDATION seeds 300+): what does event geometry see, tick by tick, in
nominal / occlusion / dropped events / low light? Logs per-tick validity, ROI and global fresh-event
counts, true tip speed, and the Baseline D estimate error."""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import benchmark_v4 as B4
from aurasync.simulate import run_episode
from aurasync.scheduler import FRESH

class Probe:
    def __init__(self, pipe): self.p = pipe; self.rows = []
    def __getattr__(self, k): return getattr(self.p, k)
    def reset(self, s0): self.p.reset(s0)
    def update(self, k, t, s_true, surface, u):
        g = self.p.geo
        ev = g.events(surface)
        roi = (g.wy >= g.y_lo) & (g.wy <= g.y_hi) & (np.abs(g.wx - self.p.s[0]) <= g.band)
        self.p.update(k, t, s_true, surface, u)
        m = g.measure(surface, self.p.s[0])
        l = 0.25; v_tip = abs(s_true[1] + 2 * l * s_true[3] * np.cos(s_true[2]))
        vhat = abs(self.p.s[1] + 2 * l * self.p.s[3] * np.cos(self.p.s[2]))
        self.rows.append((k, int((ev & roi).sum()), int(ev.sum()), m is not None and m["valid"], v_tip, vhat,
                          np.degrees(self.p.s[2] - s_true[2]), np.degrees(np.sqrt(self.p.Sigma[2, 2]))))
    def estimate(self): return self.p.estimate()

out = {}
for cond_name in ["nominal", "occlusion", "dropped_events", "low_light"]:
    cond, plant, snap, kicks, n, fault = B4.scenario_spec(cond_name)
    R = []
    for i, kick in enumerate([1.5, 3.0, 1.5, 3.0]):
        seed = 300 + i
        pipe, ctrl_f = B4.build("geo+sup"); pr = Probe(pipe)
        rng = np.random.default_rng(seed); s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
        lg = run_episode(plant, ctrl_f(pr), pr, 1200, s0, kicks=[(200, kick * (1 if i % 2 == 0 else -1))], cond=cond, cam_seed=seed)
        a = np.array(pr.rows, float); R.append(a)
        post = a[(a[:, 0] >= 200) & (a[:, 0] < 320)]
        print(f"{cond_name:15s} kick {kick} surv {lg['fail_tick'] is None}  ticks200-320: valid {post[:,3].mean():.2f} roiEv {post[:,1].mean():5.1f} globEv {post[:,2].mean():5.1f} "
              f"vtip {post[:,4].mean():.2f} vhat {post[:,5].mean():.2f} |err| {np.abs(post[:,6]).mean():5.1f} sig {post[:,7].mean():.2f} | all: valid {a[:,3].mean():.2f}", flush=True)
    quiet = np.concatenate([r[(r[:, 0] > 50) & (r[:, 0] < 200)] for r in R])
    print(f"   pre-hit quiet: valid {quiet[:,3].mean():.2f} roiEv {quiet[:,1].mean():.1f} globEv {quiet[:,2].mean():.1f} vtip {quiet[:,4].mean():.3f}")
    np.save(f"/tmp/claude-0/-home-claude/092a0d58-f4fa-5ded-9401-5638676f1bf6/scratchpad/probe_{cond_name}.npy", np.concatenate(R))
