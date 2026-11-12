"""Reproducible, *small-scale* paired simulation audit.

The runner reports rather than assumes survival superiority. It does not establish
physical hardware latency, production safety, or statistical power. Validation and
held-out seeds are disjoint; tune only on validation data.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from aurasync.conditions import PRESETS
from aurasync.controller import LQRController
from aurasync.physics import PlantParams
from aurasync.simulate import run_episode
from aurasync.v4.geo_pipeline import GeoPipeline
from aurasync.v41.pipeline import EvidencePipeline


Q = np.diag([1e-7, 1e-5, 1e-6, 1e-4])
CONDITIONS = ("nominal", "low_light", "dropped_events", "occlusion")


def one_episode(kind, condition, seed, n_ticks):
    plant = PlantParams()
    controller = LQRController(plant)
    if kind == "geometry":
        pipeline = GeoPipeline(seed=seed)
    elif kind == "evidence":
        pipeline = EvidencePipeline(Q=Q.copy(), neural=None, seed=seed)
    else:
        raise ValueError(f"unknown pipeline: {kind}")
    result = run_episode(
        plant, controller, pipeline, n_ticks=n_ticks,
        s0=np.array([0.0, 0.0, 0.03, 0.0]),
        cond=PRESETS[condition], cam_seed=seed,
        stop_on_fail=True,
    )
    return dict(
        survived=result["fail_tick"] is None,
        fail_tick=None if result["fail_tick"] is None else int(result["fail_tick"]),
        ticks=int(len(result["s"])),
        angle_rmse_rad=float(np.sqrt(np.mean(
            (result["s"][:, 2] - result["s_hat"][:, 2]) ** 2
        ))),
        mean_event_count=float(np.mean(result["events"])),
    )


def paired_bootstrap_ci(differences, seed=991, n_boot=1000):
    """Percentile 95% CI for paired episode differences (exploratory)."""
    arr = np.asarray(differences, dtype=float)
    if arr.size == 0:
        raise ValueError("at least one paired sample required")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[draws].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate(seed_start=1000, n_seeds=4, n_ticks=40, split="validation"):
    if n_seeds <= 0 or n_ticks <= 0:
        raise ValueError("n_seeds and n_ticks must be positive")
    rows = []
    summary = []
    for condition in CONDITIONS:
        differences = []
        for seed in range(seed_start, seed_start + n_seeds):
            base = one_episode("geometry", condition, seed, n_ticks)
            candidate = one_episode("evidence", condition, seed, n_ticks)
            rows.append(dict(seed=seed, condition=condition,
                             geometry=base, evidence=candidate))
            differences.append(float(candidate["survived"]) - float(base["survived"]))
        summary.append(dict(
            condition=condition,
            paired_survival_difference=float(np.mean(differences)),
            exploratory_ci95=paired_bootstrap_ci(differences),
            total_seeds=n_seeds,
        ))
    return dict(
        schema_version=1, split=split,
        seed_start=seed_start, n_seeds=n_seeds, n_ticks=n_ticks,
        scientific_limits=(
            "Short simulation sanity audit only; no calibrated recovery superiority, "
            "hardware-in-loop validation, or sub-millisecond physical latency claim."
        ),
        episodes=rows, summary=summary,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "heldout"], default="validation")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--output", default="validation-report.json")
    args = parser.parse_args()
    first = 1000 if args.split == "validation" else 100000
    report = evaluate(seed_start=first, n_seeds=args.seeds, n_ticks=args.ticks, split=args.split)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in report["summary"]:
        print(row["condition"], "paired survival difference", row["paired_survival_difference"],
              "exploratory CI", row["exploratory_ci95"])
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
