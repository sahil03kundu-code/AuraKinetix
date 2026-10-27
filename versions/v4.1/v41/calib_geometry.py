"""Step 3a (VALIDATION seeds 300-311 only): is EventGeometry's sigma_theta honest, as a function of
fit support? Collects every valid measurement with its true angle from Baseline D closed-loop runs
in nominal / occlusion / dropped events / low light, and reports |z| = |theta_m - theta_true| / sigma
by support bin. Output: results/v41/geometry_calibration.json"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from multiprocessing import Pool
import benchmark_v4 as B4
from aurasync.simulate import run_episode

class Rec:
    def __init__(self, p): self.p = p; self.rows = []
    def __getattr__(self, k): return getattr(self.p, k)
    def reset(self, s0): self.p.reset(s0)
    def estimate(self): return self.p.estimate()
    def update(self, k, t, st, surf, u):
        self.p.update(k, t, st, surf, u)
        g = self.p.geo; state = g.rng.bit_generator.state
        m = g.measure(surf, self.p.s[0]); g.rng.bit_generator.state = state
        if m is not None and m["valid"]:
            self.rows.append((m["theta"] - st[2], m["sigma_theta"], m["span"], m["n_inliers"], m["snr"], m["quality"], k))

def job(arg):
    cond_name, seed, kick = arg
    cond, plant, snap, kicks, n, fault = B4.scenario_spec(cond_name)
    pipe, cf = B4.build("geo+sup"); r = Rec(pipe)
    rng = np.random.default_rng(seed); s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
    run_episode(plant, cf(r), r, 1200, s0, kicks=[(200, kick * (1 if seed % 2 else -1))], cond=cond, cam_seed=seed)
    return cond_name, r.rows

if __name__ == "__main__":
    os.makedirs("results/v41", exist_ok=True)
    jobs = [(c, s, k) for c in ["nominal", "occlusion", "dropped_events", "low_light"] for s in range(300, 312) for k in (1.5, 3.0)]
    with Pool(2) as pool:
        res = pool.map(job, jobs)
    D = np.array([r for _, rows in res for r in rows])
    np.save("results/v41/geometry_calib_rows.npy", D)
    err, sig, span, nin = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
    z = np.abs(err) / sig
    out = {}
    print(f"all valid measurements: n={len(D)}  median|err| {np.degrees(np.median(np.abs(err))):.2f} deg  |z|<2: {(z<2).mean():.2f}  p95|z| {np.percentile(z,95):.1f}")
    for lo, hi in [(0.35, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 2)]:
        for nlo, nhi in [(8, 12), (12, 20), (20, 999)]:
            s = (span >= lo) & (span < hi) & (nin >= nlo) & (nin < nhi)
            if s.sum() < 30: continue
            key = f"span{lo}-{hi}_n{nlo}-{nhi}"
            # inflation needed so that 95% of |err| fall within 2 sigma
            f = float(np.percentile(z[s], 95) / 2)
            out[key] = dict(n=int(s.sum()), median_err_deg=float(np.degrees(np.median(np.abs(err[s])))),
                            p95_err_deg=float(np.degrees(np.percentile(np.abs(err[s]), 95))),
                            cov2=float((z[s] < 2).mean()), inflation_for_95=f)
            print(f"  {key:22s} n {s.sum():6d}  med|err| {out[key]['median_err_deg']:5.2f}  p95 {out[key]['p95_err_deg']:6.2f}  |z|<2 {out[key]['cov2']:.2f}  needed x{f:.1f}")
    json.dump(out, open("results/v41/geometry_calibration.json", "w"), indent=1)
