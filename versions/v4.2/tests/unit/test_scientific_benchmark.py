"""Fast contract checks for the pre-registered scientific benchmark."""

import numpy as np
import pytest

from scientific_benchmark import CONDITIONS, evaluate, paired_interval, seeds_for


def test_seed_splits_do_not_overlap():
    pilot = set(seeds_for("pilot", 8))
    validation = set(seeds_for("validation", 50))
    heldout = set(seeds_for("heldout", 100))
    assert not (pilot & validation or validation & heldout or pilot & heldout)


def test_seed_limits_reject_invalid_requests():
    with pytest.raises(ValueError):
        seeds_for("pilot", 0)
    with pytest.raises(ValueError):
        seeds_for("heldout", 101)
    with pytest.raises(ValueError):
        seeds_for("unknown", 2)


def test_bootstrap_interval_is_repeatable_and_zero_for_equal_pairs():
    assert paired_interval([0.0] * 10) == [0.0, 0.0]
    first = paired_interval([0.0, 1.0, -1.0, 1.0], seed=7)
    assert first == paired_interval([0.0, 1.0, -1.0, 1.0], seed=7)
    assert first[0] <= first[1]


def test_bootstrap_rejects_invalid_input():
    with pytest.raises(ValueError):
        paired_interval([])
    with pytest.raises(ValueError):
        paired_interval([np.inf])


def test_final_holdout_requires_frozen_complete_protocol():
    with pytest.raises(ValueError, match="100 seeds"):
        evaluate(split="heldout", count=1, ticks=10)
    with pytest.raises(ValueError, match="freeze commit"):
        evaluate(split="heldout", count=100, ticks=1000)
    with pytest.raises(ValueError, match="match checked-out"):
        evaluate(split="heldout", count=100, ticks=1000, freeze_commit="0" * 40)


def test_only_pre_registered_conditions_are_used():
    assert CONDITIONS == ("nominal", "low_light", "dropped_events", "occlusion")
