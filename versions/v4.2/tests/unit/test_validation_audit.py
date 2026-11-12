"""Contract tests for the reproducible paired scientific audit."""

import numpy as np
import pytest

import validation_audit as audit


def test_paired_bootstrap_is_deterministic_and_reports_two_bounds():
    samples = [1.0, 0.0, -1.0, 1.0]
    first = audit.paired_bootstrap_ci(samples, seed=4, n_boot=200)
    second = audit.paired_bootstrap_ci(samples, seed=4, n_boot=200)
    assert first == second
    assert len(first) == 2
    assert -1 <= first[0] <= first[1] <= 1


def test_bootstrap_requires_nonempty_pairs():
    with pytest.raises(ValueError, match="at least one"):
        audit.paired_bootstrap_ci([])


def test_evaluate_preserves_pairing_and_distinct_seed_ranges(monkeypatch):
    def fake_episode(kind, condition, seed, n_ticks):
        survived = kind == "evidence"
        return dict(survived=survived, fail_tick=None if survived else 0,
                    ticks=n_ticks, angle_rmse_rad=0.0, mean_event_count=1.0)

    monkeypatch.setattr(audit, "one_episode", fake_episode)
    val = audit.evaluate(seed_start=1000, n_seeds=2, n_ticks=3, split="validation")
    held = audit.evaluate(seed_start=100000, n_seeds=2, n_ticks=3, split="heldout")

    assert len(val["episodes"]) == 8
    assert len(held["episodes"]) == 8
    assert not ({r["seed"] for r in val["episodes"]} &
                {r["seed"] for r in held["episodes"]})
    assert all(s["paired_survival_difference"] == 1.0 for s in val["summary"])
    assert all(s["exploratory_ci95"] == [1.0, 1.0] for s in val["summary"])
    assert "Short simulation" in val["scientific_limits"]


def test_evaluate_rejects_nonpositive_budgets():
    with pytest.raises(ValueError):
        audit.evaluate(n_seeds=0)
    with pytest.raises(ValueError):
        audit.evaluate(n_ticks=0)


def test_condition_list_contains_nominal_and_faulted_scenarios():
    assert audit.CONDITIONS == (
        "nominal", "low_light", "dropped_events", "occlusion"
    )
    assert np.isfinite(audit.Q).all()
