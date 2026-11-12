"""Regression tests for the geometry-only closed-loop estimator (Baseline D)."""

from unittest.mock import Mock

import numpy as np

from aurasync.physics import rk4_step
from aurasync.v4.geo_pipeline import DT, GeoPipeline


def make_measurement(theta=0.0, x_pivot=0.0, valid=True):
    return dict(
        theta=theta,
        x_pivot=x_pivot,
        sigma_theta=0.01,
        n_inliers=25,
        valid=valid,
    )


def test_reset_copies_input_and_initializes_uncertainty():
    pipe = GeoPipeline(seed=3)
    initial = np.array([0.02, 0.0, 0.04, 0.0])
    pipe.reset(initial)
    assert pipe.estimate().shape == (4,)
    assert pipe.covariance().shape == (4, 4)
    assert np.all(np.diag(pipe.covariance()) > 0)
    assert pipe.n_fused == 0
    assert pipe.age_ticks() == 0
    initial[0] = 123.0
    assert pipe.estimate()[0] != 123.0


def test_no_measurement_coasts_on_physics_and_covariance_grows():
    pipe = GeoPipeline()
    initial = np.array([0.0, 0.0, 0.05, 0.0])
    pipe.reset(initial)
    before = pipe.covariance().copy()
    pipe.geo.measure = Mock(return_value=None)
    pipe.update(0, DT, initial, None, 0.0)
    np.testing.assert_allclose(
        pipe.estimate(), rk4_step(initial, 0.0, DT, pipe.p), atol=1e-12
    )
    assert pipe.n_fused == 0
    assert np.trace(pipe.covariance()) > np.trace(before)


def test_invalid_measurement_is_ignored():
    pipe = GeoPipeline()
    initial = np.array([0.0, 0.0, 0.03, 0.0])
    pipe.reset(initial)
    pipe.geo.measure = Mock(return_value=make_measurement(valid=False))
    pipe.update(0, DT, initial, None, 0.0)
    assert pipe.n_fused == 0
    np.testing.assert_allclose(
        pipe.estimate(), rk4_step(initial, 0.0, DT, pipe.p), atol=1e-12
    )


def test_valid_measurement_updates_geometry_estimate_and_covariance():
    pipe = GeoPipeline()
    initial = np.array([0.0, 0.0, 0.01, 0.0])
    pipe.reset(initial)
    prediction = rk4_step(initial, 0.0, DT, pipe.p)
    pipe.geo.measure = Mock(return_value=make_measurement(theta=0.02, x_pivot=0.03))
    pipe.update(0, DT, initial, None, 0.0)
    assert pipe.n_fused == 1
    assert pipe.estimate()[0] > prediction[0]
    assert pipe.estimate()[2] > prediction[2]
    assert np.isfinite(pipe.covariance()).all()
    np.testing.assert_allclose(pipe.covariance(), pipe.covariance().T, atol=1e-10)
    assert np.linalg.eigvalsh(pipe.covariance()).min() >= -1e-10


def test_large_angle_outlier_rejected_only_after_tracking_is_established():
    pipe = GeoPipeline(gate=0.01)
    initial = np.zeros(4)
    pipe.reset(initial)
    pipe.n_fused = 51
    pipe.geo.measure = Mock(return_value=make_measurement(theta=1.0))
    pipe.update(0, DT, initial, None, 0.0)
    assert pipe.n_fused == 51
    np.testing.assert_allclose(
        pipe.estimate(), rk4_step(initial, 0.0, DT, pipe.p), atol=1e-12
    )


def test_covariance_and_state_are_finite_after_repeated_updates():
    pipe = GeoPipeline()
    pipe.reset(np.array([0.0, 0.0, 0.01, 0.0]))
    pipe.geo.measure = Mock(return_value=make_measurement(theta=0.01))
    for k in range(5):
        pipe.update(k, (k + 1) * DT, np.zeros(4), None, 0.0)
    assert pipe.n_fused == 5
    assert np.isfinite(pipe.estimate()).all()
    assert np.isfinite(pipe.covariance()).all()
