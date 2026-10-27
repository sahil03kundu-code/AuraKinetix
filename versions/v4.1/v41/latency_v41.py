"""Measured CPU cost per control tick (estimator + supervisor), idle CPU, 1 thread, replaying
recorded validation sensor data (no ground truth). Neural work is computed inline here, so the
per-tick figures include it on ticks that start a neural job (in deployment it runs on a second core).
Writes results/v41/latency_v41.json."""
import os, sys, json, glob, time
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
torch.set_num_threads(1)
import benchmark_v41 as BV

files = sorted(glob.glob("data/v4dev/ev-trig+shield-s0_*.npz"))[:4] + sorted(glob.glob("data/v4dev/ev-trig-s0_*.npz"))[:4]
out = {}
for cfg in ["geo+sup", "geoev+sup", "v41+sup-s0", "v4-b100+sup-s0", "v4+sup-s0"]:
    est, sup, calls, n_all = [], [], 0, 0
    for f in files:
        d = dict(np.load(f)); surf = d["surf"].astype(np.float32); n = min(len(surf), len(d["u"]))
        pipe, sup_f = BV.build(cfg); ctrl = sup_f(pipe)
        pipe.reset(d["s0"].astype(float))
        for k in range(n):
            t0 = time.perf_counter(); pipe.update(k, (k + 1) * 1e-3, None, surf[k], float(d["u"][k])); t1 = time.perf_counter()
            ctrl.act(np.asarray(pipe.estimate())); t2 = time.perf_counter()
            est.append(t1 - t0); sup.append(t2 - t1)
        calls += getattr(pipe, "log", {}).get("calls", 0); n_all += n
    est, sup = np.array(est) * 1e3, np.array(sup) * 1e3
    tot = est + sup
    out[cfg] = dict(ticks=int(n_all), calls_per_s_replay=calls / (n_all * 1e-3),
                    estimator_ms=dict(p50=float(np.percentile(est, 50)), p99=float(np.percentile(est, 99)), max=float(est.max())),
                    supervisor_ms=dict(p50=float(np.percentile(sup, 50)), p99=float(np.percentile(sup, 99))),
                    tick_ms=dict(p50=float(np.percentile(tot, 50)), p99=float(np.percentile(tot, 99)), max=float(tot.max())),
                    cpu_cores=float(tot.sum() / n_all))
    print(cfg, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[cfg].items() if k != "estimator_ms"}), flush=True)
json.dump(out, open("results/v41/latency_v41.json", "w"), indent=1)
