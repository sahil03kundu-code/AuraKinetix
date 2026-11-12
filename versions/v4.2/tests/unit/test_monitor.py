"""Focused tests for independent event-consistency monitoring."""

import numpy as np
import pytest

from aurasync.monitor import (
    EventConsistencyMonitor,
    FRESH2,
    monitor_trajectory,
)


def state():
    return np.array([0.0, 0.0, 0.0, 0.0])


def blank_surface():
    return np.zeros((4, 32, 96), dtype=np.float32)


def event_surface(region):
    """Build fresh positive events in a geometrically defined pole/fan region."""
    mon = EventConsistencyMonitor()
    s = state()
    wx, wy, scale = mon.r.wx, mon.r.wy, mon.r.S
    px, py = s[0], 0.04
    dx, dy = wx - px, wy - py
    fan = (
        (np.hypot(dx, dy) <= 2 * mon.l + 3 / scale)
        & (np.hypot(dx, dy) >= 0.06)
        & (np.abs(np.arctan2(dx, dy)) <= mon.fan)
    )
    pole_distance = np.abs(wx - px) * scale
    near = fan & (pole_distance <= mon.near_px)
    far = fan & (pole_distance > mon.far_px)
    surface = blank_surface()
    if region == "near":
        surface[0, near] = 1.0
    elif region == "far":
        surface[1, far] = 1.0
    elif region == "both":
        surface[0, fan] = 1.0
    else:
        raise ValueError(region)
    return surface


def test_monitor_starts_at_calibrated_prior_and_resets():
    mon = EventConsistencyMonitor(m0=0.3, ema=0.5)
    assert mon.m == 0.3
    assert mon.info_ticks == 0
    mon.update(state(), event_surface("far"))
    assert mon.info_ticks == 1
    mon.reset()
    assert mon.m == 0.3
    assert mon.info_ticks == 0


def test_quiet_surface_is_uninformative():
    mon = EventConsistencyMonitor()
    quiet = blank_surface()
    assert mon.raw_score(state(), quiet) is None
    assert mon.update(state(), quiet) == mon.m0
    assert mon.info_ticks == 0


def test_near_pole_events_are_consistent():
    mon = EventConsistencyMonitor()
    score = mon.raw_score(state(), event_surface("near"))
    assert score is not None
    assert score == pytest.approx(0.0, abs=1e-9)


def test_far_events_contradict_pole_estimate():
    mon = EventConsistencyMonitor()
    score = mon.raw_score(state(), event_surface("far"))
    assert score is not None
    assert score > 0.99


def test_both_regions_are_informative():
    mon = EventConsistencyMonitor()
    score = mon.raw_score(state(), event_surface("both"))
    assert score is not None
    assert 0.0 < score < 1.0


def test_ema_updates_only_on_informative_ticks():
    mon = EventConsistencyMonitor(m0=0.25, ema=0.5)
    score = mon.raw_score(state(), event_surface("far"))
    expected = 0.25 + 0.5 * (score - 0.25)
    actual = mon.update(state(), event_surface("far"))
    assert actual == pytest.approx(expected)
    assert mon.info_ticks == 1
    assert mon.update(state(), blank_surface()) == pytest.approx(expected)
    assert mon.info_ticks == 1


def test_inflation_is_inactive_below_threshold_and_grows_above_it():
    mon = EventConsistencyMonitor(m0=0.25, beta=0.5)
    assert mon.inflate(0.1) == pytest.approx(0.1)
    mon.m = 0.85
    expected = np.sqrt(0.1 ** 2 + (0.5 * (0.85 - 0.25)) ** 2)
    assert mon.inflate(0.1) == pytest.approx(expected)
    assert mon.inflate(0.1) > 0.1


def test_minimum_event_threshold_is_respected():
    mon = EventConsistencyMonitor(min_events=10000)
    assert mon.raw_score(state(), event_surface("both")) is None
    assert mon.info_ticks == 0


def test_monitor_trajectory_returns_one_score_per_tick():
    trajectory = np.stack([state(), state(), state()])
    surfaces = np.stack(
        [blank_surface(), event_surface("far"), blank_surface()]
    )
    output = monitor_trajectory(trajectory, surfaces, ema=0.5)
    assert output.shape == (3,)
    assert output[0] == pytest.approx(0.25)
    assert output[1] > output[0]
    assert output[2] == pytest.approx(output[1])


def test_freshness_threshold_ignores_stale_activity():
    mon = EventConsistencyMonitor()
    stale = event_surface("far")
    stale[stale > 0] = FRESH2 * 0.5
    assert mon.raw_score(state(), stale) is None
