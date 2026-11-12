"""Unit tests for geometry-based angle measurement from event surfaces."""

import numpy as np
import pytest

from aurasync.v4.geometry import EventGeometry, PIVOT_Y, TAU_FAST


def surface():
    return np.zeros((4, 32, 96), dtype=np.float32)


def line_surface(geometry, x_pivot=0.0, theta=0.0):
    """Synthetic fresh events within one pixel of a known pole segment."""
    out = surface()
    wx, wy = geometry.wx, geometry.wy
    y = wy - PIVOT_Y
    x_line = x_pivot + y * np.tan(theta)
    mask = (
        (y >= 0.08)
        & (y <= 2 * geometry.l - 0.02)
        & (np.abs(wx - x_line) <= 0.65 / geometry.S)
    )
    out[0][mask] = 1.0
    return out


def test_event_mask_combines_positive_and_negative_channels():
    geo = EventGeometry()
    data = surface()
    data[0, 3, 4] = 1.0
    data[1, 5, 6] = 1.0
    data[0, 7, 8] = geo.fresh * 0.5
    mask = geo.events(data)
    assert mask.dtype == np.bool_
    assert mask.sum() == 2
    assert mask[3, 4] and mask[5, 6]
    assert not mask[7, 8]


def test_freshness_threshold_matches_window():
    geo = EventGeometry(window=0.003)
    assert geo.fresh == pytest.approx(np.exp(-0.003 / TAU_FAST))


def test_empty_surface_produces_no_measurement():
    geo = EventGeometry()
    assert geo.measure(surface(), x_hint=0.0) is None


def test_too_few_events_produce_no_measurement():
    geo = EventGeometry(min_inliers=8)
    data = surface()
    data[0, 10, 48] = 1.0
    assert geo.measure(data, x_hint=0.0) is None


@pytest.mark.parametrize("theta", [0.0, np.radians(20.0)])
def test_straight_pole_measurement_is_finite_and_geometric(theta):
    geo = EventGeometry(
        min_inliers=5, min_span=0.2, min_snr=1.0, n_iter=300, seed=11
    )
    data = line_surface(geo, theta=theta)
    assert geo.events(data).sum() >= 5
    result = geo.measure(data, x_hint=0.0)
    assert result is not None
    assert np.isfinite(result["theta"])
    assert abs(result["theta"] - theta) < np.radians(6.0)
    assert np.isfinite(result["sigma_theta"])
    assert result["sigma_theta"] > 0
    assert result["n_inliers"] >= geo.min_inliers
    assert result["n_events"] >= result["n_inliers"]
    assert 0 <= result["quality"] <= 1
    assert result["span"] > 0
    assert result["snr"] >= 0


def test_hint_outside_event_region_returns_no_measurement():
    geo = EventGeometry(band_halfwidth=0.05)
    data = line_surface(geo)
    assert geo.measure(data, x_hint=2.0) is None
