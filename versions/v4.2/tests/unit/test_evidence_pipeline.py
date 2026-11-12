"""Deterministic tests for V4.1 evidence-aware filtering and recovery."""

from unittest.mock import Mock

import numpy as np
import pytest

from aurasync.v41.pipeline import (
    CONTRADICTORY,
    CORRUPT,
    FUSED,
    QUIET,
    REINIT,
    UNOBSERVED,
    EvidencePipeline,
    STATE_NAMES,
)


def make_pipeline(**kwargs):
    pipe = EvidencePipeline(Q=np.diag([1e-7, 1e-5, 1e-6, 1e-4]), **kwargs)
    pipe.reset(np.zeros(4))
    return pipe


def empty_surface():
    return np.zeros((4, 32, 96), dtype=np.float32)


def update(pipe, k, surface=None):
    pipe.update(
        k, k * 0.001, np.zeros(4),
        empty_surface() if surface is None else surface, 0.0,
    )


def observation(theta=0.0, x=0.0, valid=True):
    return dict(
        valid=valid,
        theta=theta,
        x_pivot=x,
        sigma_theta=0.01,
        n_inliers=25,
        span=0.9,
    )


def test_state_names_and_reset_interface():
    pipe = make_pipeline(seed=17)
    assert STATE_NAMES == [
        "quiet", "unobserved", "fused", "contradictory", "corrupt", "reinit"
    ]
    assert pipe.state == QUIET
    assert pipe.age_ticks() == 0
    assert pipe.estimate().shape == (4,)
    assert pipe.covariance().shape == (4, 4)
    assert pipe.log["states"] == [0] * 6
    assert pipe.ev_base is None
    assert pipe.job is None


def test_empty_unobserved_scene_increases_uncertainty():
    pipe = make_pipeline()
    pipe.geo.measure = Mock(return_value=None)
    before = pipe.covariance().copy()
    update(pipe, 0)
    assert pipe.state == UNOBSERVED
    assert pipe.log["states"][UNOBSERVED] == 1
    assert pipe.unobs_run == 1
    assert pipe.covariance()[2, 2] > before[2, 2]


def test_live_quiet_scene_keeps_bounded_uncertainty():
    pipe = make_pipeline()
    pipe.geo.measure = Mock(return_value=None)
    pipe.geo.events = Mock(return_value=np.pad(
        np.array([[True]]), ((0, 31), (0, 95)), constant_values=False
    ))
    update(pipe, 0)
    assert pipe.state == QUIET
    assert pipe.log["states"][QUIET] == 1
    assert pipe.unobs_run == 0
    assert np.sqrt(pipe.covariance()[2, 2]) <= pipe.sig_quiet + 1e-12


def test_valid_geometry_measurement_is_fused():
    pipe = make_pipeline()
    pipe.geo.measure = Mock(return_value=observation(theta=0.01, x=0.02))
    update(pipe, 0)
    assert pipe.state == FUSED
    assert pipe.log["states"][FUSED] == 1
    assert pipe.estimate()[0] > 0.0
    assert pipe.estimate()[2] > 0.0
    assert len(pipe.buf) == 1
    assert len(pipe.cbuf) == 0
    np.testing.assert_allclose(pipe.covariance(), pipe.covariance().T, atol=1e-10)


def test_large_geometric_innovation_is_contradictory_without_immediate_reset():
    pipe = make_pipeline(confirm_n=4, gate=0.01)
    pipe.geo.measure = Mock(return_value=observation(theta=0.5))
    update(pipe, 0)
    assert pipe.state == CONTRADICTORY
    assert len(pipe.cbuf) == 1
    assert pipe.last_reinit < 0
    assert pipe.log["states"][CONTRADICTORY] == 1


def test_repeated_consistent_contradictions_trigger_reinitialization():
    pipe = make_pipeline(confirm_n=2, gate=0.01, cooldown=0)
    pipe.geo.measure = Mock(return_value=observation(theta=0.5, x=0.02))
    update(pipe, 0)
    assert pipe.state == CONTRADICTORY
    update(pipe, 1)
    assert pipe.state == REINIT
    assert pipe.estimate()[2] == pytest.approx(0.5)
    assert pipe.estimate()[0] == pytest.approx(0.02)
    assert pipe.last_reinit == 1
    assert len(pipe.cbuf) == 0


def test_unusable_event_burst_is_marked_corrupt():
    pipe = make_pipeline(burst_min=40)
    pipe.geo.measure = Mock(return_value=None)
    pipe.geo.events = Mock(side_effect=[
        np.zeros((32, 96), dtype=bool),
        np.ones((32, 96), dtype=bool),
    ])
    update(pipe, 0)
    update(pipe, 1)
    assert pipe.state == CORRUPT
    assert pipe.log["states"][CORRUPT] == 1
    assert pipe.covariance()[2, 2] > 0.0


def test_kalman_update_fuses_position_and_preserves_psd():
    pipe = make_pipeline()
    before = pipe.covariance().copy()
    pipe._kf([0, 2], np.array([0.03, 0.04]), np.diag([1e-4, 1e-4]))
    assert pipe.estimate()[0] > 0
    assert pipe.estimate()[2] > 0
    assert pipe.covariance()[0, 0] <= before[0, 0]
    assert np.linalg.eigvalsh(pipe.covariance()).min() >= -1e-10


def test_uncertainty_cap_and_widen_preserve_finite_covariance():
    pipe = make_pipeline()
    pipe.Sigma[2, 2] = 0.1
    pipe._cap(2, 0.02)
    assert pipe.Sigma[2, 2] == pytest.approx(0.02 ** 2)
    pipe._widen(2, 0.03)
    assert pipe.Sigma[2, 2] == pytest.approx(0.0304)
    assert np.isfinite(pipe.covariance()).all()


def test_confirmation_requires_temporal_consistency():
    pipe = make_pipeline(confirm_n=3, confirm_spread_deg=2.0)
    pipe.cbuf.extend([(1, 0.2, 0.01, 0.0), (2, 0.21, 0.01, 0.0)])
    assert not pipe._confirmed(2)
    pipe.cbuf.append((3, 0.22, 0.01, 0.0))
    assert pipe._confirmed(3)
    assert not pipe._confirmed(100)
