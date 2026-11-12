"""Isolated regression checks for V4 anomaly detection and recovery helpers."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from aurasync.v4.pipeline import ConsistencyDetector, V4Pipeline


def state(theta=0.0, velocity=0.0):
    return np.array([0.0, velocity, theta, 0.0], dtype=float)


def covariance():
    return np.diag([0.01, 0.01, 0.001, 0.01])


def observation(theta=0.0, x_pivot=0.0, quality=0.9):
    return {
        "valid": True,
        "theta": theta,
        "x_pivot": x_pivot,
        "sigma_theta": 0.01,
        "n_inliers": 25,
        "quality": quality,
    }


def bare_pipeline():
    """Exercise numerical helpers without requiring a trained neural checkpoint."""
    pipeline = V4Pipeline.__new__(V4Pipeline)
    pipeline.s = np.zeros(4)
    pipeline.Sigma = covariance()
    pipeline.fuse_x = True
    pipeline.sigma_x_floor = 0.004
    pipeline.geo = SimpleNamespace(S=100.0)
    pipeline.buf = deque(maxlen=32)
    pipeline.confirm_n = 3
    pipeline.confirm_window = 12
    pipeline.confirm_spread = np.radians(3.0)
    pipeline.last_reinit = -10**9
    pipeline.det = ConsistencyDetector()
    pipeline.hidden_reset = "keep"
    pipeline.models = []
    pipeline.job = None
    pipeline.ev_log = []
    return pipeline


def test_detector_initial_state_and_reset():
    detector = ConsistencyDetector()
    assert detector.g == 0.0
    assert not detector.alarm
    assert detector.t_detect is None
    detector.g = 100.0
    detector.reset()
    assert detector.g == 0.0
    assert detector.reasons == []


def test_detector_detects_repeated_large_angle_disagreement():
    detector = ConsistencyDetector(cusum_h=30.0)
    measurement = observation(theta=0.5)
    result = None
    for k in range(4):
        result = detector.update(k, state(), covariance(), measurement, 10)
    assert result["alarm"]
    assert result["t_detect"] == 2
    assert result["d2"] > 9
    assert "angle_mismatch" in result["reasons"]
    assert detector.rel > 0


def test_detector_reports_sign_mismatch():
    detector = ConsistencyDetector()
    result = detector.update(
        0, state(theta=-0.2), covariance(), observation(theta=0.2), 10
    )
    assert "sign_mismatch" in result["reasons"]


def test_detector_reports_missing_events_and_stale_evidence():
    detector = ConsistencyDetector(stale_ticks=2, motion_speed=0.3)
    result = detector.update(3, state(velocity=0.8), covariance(), None, 0)
    assert "missing_events" in result["reasons"]
    assert "stale_evidence" in result["reasons"]


def test_detector_reports_noise_burst_after_baseline():
    detector = ConsistencyDetector(burst_factor=3.0)
    detector.update(0, state(), covariance(), None, 5)
    result = detector.update(1, state(), covariance(), None, 100)
    assert "noise_burst" in result["reasons"]


def test_kalman_event_fusion_moves_estimate_toward_valid_measurement():
    pipeline = bare_pipeline()
    before = pipeline.Sigma.copy()
    pipeline._fuse(observation(theta=0.03, x_pivot=0.05))
    assert pipeline.s[0] > 0
    assert pipeline.s[2] > 0
    assert pipeline.Sigma[0, 0] < before[0, 0]
    assert pipeline.Sigma[2, 2] < before[2, 2]
    np.testing.assert_allclose(pipeline.Sigma, pipeline.Sigma.T, atol=1e-12)


def test_theta_only_event_fusion_preserves_unobserved_pivot():
    pipeline = bare_pipeline()
    pipeline.fuse_x = False
    pipeline._fuse(observation(theta=0.02, x_pivot=0.5))
    assert pipeline.s[0] == pytest.approx(0.0)
    assert pipeline.s[2] > 0


def test_confirmation_requires_consistent_recent_measurements():
    pipeline = bare_pipeline()
    pipeline.buf.extend([
        (1, 0.20, 0.01, 0.02),
        (2, 0.21, 0.01, 0.02),
    ])
    assert not pipeline._confirmed(2)
    pipeline.buf.append((3, 0.19, 0.01, 0.02))
    assert pipeline._confirmed(3)
    pipeline.buf.append((4, 0.5, 0.01, 0.02))
    assert not pipeline._confirmed(4)
    assert not pipeline._confirmed(100)


def test_reinit_uses_corrobated_geometry_and_cancels_pending_job():
    pipeline = bare_pipeline()
    pipeline.job = ("pending",)
    pipeline.buf.extend([
        (1, 0.2, 0.02, 0.03),
        (2, 0.21, 0.02, 0.04),
        (3, 0.19, 0.02, 0.05),
    ])
    pipeline._reinit(3)
    assert pipeline.s[2] == pytest.approx(0.2)
    assert pipeline.s[0] == pytest.approx(0.04)
    assert pipeline.job is None
    assert pipeline.last_reinit == 3
    assert pipeline.det.g == 0
    assert pipeline.ev_log[-1]["action"] == "reinit_detail"


def test_position_only_neural_fusion_does_not_overwrite_velocity():
    pipeline = bare_pipeline()
    pipeline.neural_fusion = "kalman_pos"
    pipeline._fuse_neural(
        np.array([0.08, 10.0, 0.03, -10.0]),
        np.eye(4) * 0.001,
    )
    assert pipeline.s[0] > 0.0
    assert pipeline.s[2] > 0.0
    assert pipeline.s[1] == pytest.approx(0.0)
    assert pipeline.s[3] == pytest.approx(0.0)
    assert np.linalg.eigvalsh(pipeline.Sigma).min() >= -1e-10
