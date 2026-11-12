"""CPU-only Python-loop timing audit, NOT sensor-to-action hardware latency.

Measures separately the controller.act and pipeline.update calls inside the
actual closed-loop simulator. GitHub-hosted runner timing is variable and
cannot substantiate physical sub-millisecond recovery claims.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

try:
    import resource  # Unix process peak resident set; not available on native Windows
except ImportError:  # pragma: no cover - Windows fallback
    resource = None

import numpy as np

from aurasync.controller import LQRController
from aurasync.physics import PlantParams
from aurasync.simulate import run_episode
from aurasync.v4.geo_pipeline import GeoPipeline
from aurasync.v41.pipeline import EvidencePipeline


class TimedController:
    def __init__(self, inner):
        self.inner = inner
        self.timings_ns = []

    def act(self, estimate):
        start = time.perf_counter_ns()
        result = self.inner.act(estimate)
        self.timings_ns.append(time.perf_counter_ns() - start)
        return result


class TimedPipeline:
    def __init__(self, inner):
        self.inner = inner
        self.timings_ns = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset(self, initial):
        return self.inner.reset(initial)

    def estimate(self):
        return self.inner.estimate()

    def update(self, k, t, s_true, surface, u):
        start = time.perf_counter_ns()
        result = self.inner.update(k, t, s_true, surface, u)
        self.timings_ns.append(time.perf_counter_ns() - start)
        return result


def describe_ns(samples):
    values = np.asarray(samples, dtype=np.float64) / 1e6
    if values.size == 0:
        raise ValueError("cannot summarize empty timing samples")
    return {
        "samples": int(values.size),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }



def process_peak_rss_mib():
    """Peak memory of the entire benchmark Python process, not one pipeline."""
    if resource is None:
        return None
    # ru_maxrss is KiB on Linux, bytes on macOS (document runner semantics).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss / (1024 if platform.system() == "Linux" else 1024 ** 2))


def measure(kind, seed, ticks):
    plant = PlantParams()
    controller = TimedController(LQRController(plant))
    if kind == "geometry":
        inner = GeoPipeline(seed=seed)
    elif kind == "evidence":
        inner = EvidencePipeline(Q=np.diag([1e-7, 1e-5, 1e-6, 1e-4]), seed=seed)
    else:
        raise ValueError(kind)
    pipeline = TimedPipeline(inner)
    result = run_episode(
        plant, controller, pipeline, n_ticks=ticks,
        s0=np.array([0.0, 0.0, 0.01, 0.0]),
        cam_seed=seed, stop_on_fail=False,
    )
    assert len(controller.timings_ns) == ticks
    assert len(pipeline.timings_ns) == ticks
    return {
        "pipeline": kind,
        "controller": describe_ns(controller.timings_ns),
        "estimator_update": describe_ns(pipeline.timings_ns),
        "simulated_ticks": len(result["s"]),
        "process_peak_rss_mib": process_peak_rss_mib(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--output", default="cpu-performance.json")
    args = parser.parse_args()
    if args.ticks <= 0:
        parser.error("--ticks must be positive")
    report = {
        "schema_version": 1,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_visible": os.cpu_count(),
        "method": "Python wall-clock method timings on GitHub-hosted CPU, no hardware I/O",
        "limits": (
            "Not a real-time guarantee, hardware benchmark, or detect-to-act latency measurement. "
            "Peak RSS covers the entire Python process, accumulates across both pipeline runs, "
            "and cannot be attributed to a single estimator."
        ),
        "results": [measure(kind, seed=107, ticks=args.ticks)
                    for kind in ("geometry", "evidence")],
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Wrote", args.output)
    for result in report["results"]:
        print(result["pipeline"], result["estimator_update"])


if __name__ == "__main__":
    main()
