"""V4 Module E: compute / latency report (run on an otherwise idle CPU, 1 thread).

Replays recorded validation sensor data through the V4 pipeline and times every stage
on this machine. Reports measured software costs only. It does NOT measure
sensor-to-actuator latency (no real sensor, bus or actuator exists here); the modelled
end-to-end figure is labelled as a model.
"""
import os, sys, json, glob, time, resource
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
torch.set_num_threads(1)
import benchmark_v4 as B4
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.safety import SafetyShield, ShieldedController


def pct(a):
    a = np.asarray(a) * 1e3
    return dict(p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                max=float(a.max()), mean=float(a.mean()), n=int(a.size))


def timed(obj, name, store):
    f = getattr(obj, name)
    def w(*a, **k):
        t = time.perf_counter(); r = f(*a, **k); store.append(time.perf_counter() - t); return r
    setattr(obj, name, w)


if __name__ == "__main__":
    N = int(os.environ.get("NFILES", 6))
    files = sorted(glob.glob("data/v4dev/ev-trig+shield-s0_*.npz"))[:N] + sorted(glob.glob("data/v4dev/ev-trig-s0_*.npz"))[:N]
    T = dict(tick=[], geometry=[], detector=[], fuse=[], neural_correct=[], supervisor=[])
    calls = ticks = misses = 0
    p = PlantParams(); lqr = LQRController(p)
    for f in files:
        d = dict(np.load(f)); surf = d["surf"].astype(np.float32); n = min(len(surf), len(d["u"]))
        pipe, sup_f = B4.build("v4+sup-s0")
        ctrl = sup_f(pipe)
        timed(pipe.geo, "measure", T["geometry"]); timed(pipe.det, "update", T["detector"])
        timed(pipe, "_fuse", T["fuse"]); timed(pipe, "_correct", T["neural_correct"])
        pipe.reset(d["s0"].astype(float))
        for k in range(n):
            t0 = time.perf_counter()
            pipe.update(k, (k + 1) * 1e-3, None, surf[k], float(d["u"][k]))
            T["tick"].append(time.perf_counter() - t0)
            s_hat, S = pipe.out[-1]
            t1 = time.perf_counter()
            ctrl.act(s_hat)                    # command not applied: replay uses the recorded controls
            T["supervisor"].append(time.perf_counter() - t1)
        calls += pipe.log["calls"]; misses += pipe.log["misses"]; ticks += n
    tick = np.array(T["tick"]); nc = np.array(T["neural_correct"])
    # The neural correction is computed inline here (timed separately) but runs asynchronously in the
    # design, so the always-on per-tick cost is tick time minus the neural call on ticks that made one.
    always_on = tick.copy()
    out = dict(
        machine=dict(cpu=os.popen("lscpu | grep 'Model name' | head -1").read().split(":")[-1].strip(), threads=1),
        ticks=ticks, files=len(files),
        stage_ms={k: pct(v) for k, v in T.items() if len(v)},
        neural_calls_per_s=calls / (ticks * 1e-3), deadline_miss_rate=misses / max(calls, 1),
        always_on_cost_per_s_ms=float((tick.sum() - nc.sum()) / (ticks * 1e-3) * 1e3),
        neural_cost_per_s_ms=float(nc.sum() / (ticks * 1e-3) * 1e3),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    always = out["always_on_cost_per_s_ms"]; neural = out["neural_cost_per_s_ms"]
    out["cpu_utilisation_one_core"] = dict(always_on=always / 1000, neural=neural / 1000, total=(always + neural) / 1000)
    # Modelled sensor-to-actuator latency (a MODEL, not a measurement): event batch at 1 ms tick ->
    # geometry + detector + fusion (+ supervisor) -> command at the next tick.
    st = out["stage_ms"]
    out["modelled_latency_ms"] = dict(
        note="model: worst-case tick quantisation (1 ms) + measured p99 compute of the always-on path; excludes "
             "sensor readout, USB/MIPI transport, OS scheduling and actuator delay, which were not measured",
        event_path_p99=1.0 + st["geometry"]["p99"] + st["detector"]["p99"] + st.get("fuse", {"p99": 0})["p99"] + st["supervisor"]["p99"],
        neural_path_p99=1.0 + st["neural_correct"]["p99"])
    json.dump(out, open("results/v4/latency_v4.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
