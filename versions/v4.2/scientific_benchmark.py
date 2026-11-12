"""Pre-registered paired simulation benchmark (not physical hardware validation).

Only network-free geometry D and D-ev are supported here. Neural V4.1 and V4.2
comparisons require real, validated checkpoints and a separate registered run.
"""
import argparse
import json
import platform
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from aurasync.conditions import PRESETS
from aurasync.controller import LQRController
from aurasync.physics import PlantParams
from aurasync.safety import SafetyShield, ShieldedController
from aurasync.simulate import run_episode
from aurasync.v4.geo_pipeline import GeoPipeline
from aurasync.v41.pipeline import EvidencePipeline

CONDITIONS = ("nominal", "low_light", "dropped_events", "occlusion")
SEED_START = {"pilot": 2000, "validation": 3000, "heldout": 700000}
Q = np.diag([1e-7, 1e-5, 1e-6, 1e-4])


def seeds_for(split, count):
    if split not in SEED_START:
        raise ValueError("invalid evaluation split")
    if not (1 <= count <= (100 if split == "heldout" else 50)):
        raise ValueError("seed count out of registered bounds")
    return list(range(SEED_START[split], SEED_START[split] + count))


def paired_interval(differences, *, seed=461, n_boot=2000):
    """95% percentile interval; resample independent paired episodes (whole seeds)."""
    x = np.asarray(differences, float)
    if x.ndim != 1 or x.size < 1 or not np.isfinite(x).all():
        raise ValueError("finite 1-dimensional paired differences required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size=(n_boot, x.size))
    samples = x[indices].mean(axis=1)
    return [float(np.quantile(samples, q)) for q in (0.025, 0.975)]


def run_one(kind, sensor, seed, ticks):
    params = PlantParams()
    if kind == "D":
        pipe = GeoPipeline(seed=seed, plant=params)
    elif kind == "D-ev":
        pipe = EvidencePipeline(Q=Q.copy(), neural=None, seed=seed, plant=params)
    else:
        raise ValueError("Unregistered/unavailable pipeline")
    ctrl = ShieldedController(LQRController(params), SafetyShield(plant=params), pipe)
    kick_tick = ticks // 3
    snap_tick = ticks // 2
    result = run_episode(
        params, ctrl, pipe, n_ticks=ticks,
        s0=np.array([0.0, 0.0, 0.03, 0.0]),
        kicks=[(kick_tick, 0.3)],
        snap=(snap_tick, replace(params, M=params.M * 1.35)),
        cond=PRESETS[sensor], cam_seed=seed, aps_seed=seed + 100000,
        stop_on_fail=True,
    )
    actual = result["s"]
    estimated = result["s_hat"]
    final = actual[-1]
    survival = result["fail_tick"] is None and len(actual) == ticks
    safe_recovery = (
        survival and abs(final[2]) <= 0.1 and abs(final[0]) <= 0.7
        and abs(final[3]) <= 1.0
    )
    counts = ctrl.stats
    interventions = sum(counts[key] for key in ("modified", "soft_dropped", "infeasible"))
    return {
        "survival": bool(survival),
        "safe_recovery_proxy": bool(safe_recovery),
        "angle_rmse_rad": float(np.sqrt(np.mean((actual[:, 2] - estimated[:, 2]) ** 2))),
        "intervention_rate": float(interventions / max(1, counts["ticks"])),
        "false_intervention_rate": None,
        "cpu_cost_ms": None,
        "fail_tick": None if result["fail_tick"] is None else int(result["fail_tick"]),
        "ticks_observed": len(actual),
        "final_angle_rad": float(final[2]),
    }


def evaluate(split="pilot", count=2, ticks=40, freeze_commit=None):
    seeds = seeds_for(split, count)
    if ticks <= 0:
        raise ValueError("episode ticks must be positive")
    if split == "heldout":
        if count != 100 or ticks < 1000:
            raise ValueError("final holdout requires 100 seeds and >=1000 ticks")
        if not freeze_commit or len(freeze_commit) != 40:
            raise ValueError("heldout requires an explicit 40-character freeze commit")
        actual_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        if actual_sha != freeze_commit:
            raise ValueError("requested freeze commit does not match checked-out code")
    rows = []
    summary = []
    for condition in CONDITIONS:
        data = []
        for seed in seeds:
            a = run_one("D", condition, seed, ticks)
            b = run_one("D-ev", condition, seed, ticks)
            pair = {"seed": seed, "condition": condition, "D": a, "D-ev": b}
            rows.append(pair)
            data.append(pair)
        effects = {}
        for name, key in [
            ("survival", "survival"),
            ("safe_recovery_proxy", "safe_recovery_proxy"),
            ("angle_rmse_rad", "angle_rmse_rad"),
        ]:
            diffs = [float(p["D-ev"][key]) - float(p["D"][key]) for p in data]
            effects[name] = {
                "paired_mean_difference": float(np.mean(diffs)),
                "paired_ci95": paired_interval(diffs),
                "negative_seed_count": int(sum(d < 0 for d in diffs)),
                "zero_seed_count": int(sum(d == 0 for d in diffs)),
            }
        summary.append({"condition": condition, "count": len(data), "effects": effects})
    return {
        "schema_version": 1,
        "protocol": "docs/SCIENTIFIC_PROTOCOL.md",
        "split": split, "seed_start": seeds[0], "seeds": seeds,
        "ticks_requested": ticks,
        "python": platform.python_version(),
        "freeze_commit": freeze_commit,
        "methods": ["D", "D-ev"],
        "missing_methods": ["V4.1-neural", "V4.2-candidate"],
        "scenarios": {
            "conditions": list(CONDITIONS),
            "kick_tick": ticks // 3, "kick_theta_dot": 0.3,
            "snap_tick": ticks // 2, "mass_multiplier": 1.35,
            "controller": "LQR with SafetyShield",
        },
        "limitations": (
            "Pilot/validation: no scientific superiority or hardware guarantee. "
            "Safe recovery is a simulator proxy; false interventions and CPU cost "
            "are not measured here. Full 4-method comparison is unavailable."
        ),
        "episodes": rows, "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SEED_START), default="pilot")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--freeze-commit")
    parser.add_argument("--output", default="scientific-pilot.json")
    args = parser.parse_args()
    report = evaluate(args.split, args.seeds, args.ticks, args.freeze_commit)
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Wrote", args.output, "with", len(report["episodes"]), "paired episodes")
    for row in report["summary"]:
        print(row["condition"], row["effects"]["safe_recovery_proxy"])


if __name__ == "__main__":
    main()
