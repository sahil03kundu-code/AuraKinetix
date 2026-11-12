"""End-to-end simulation checks for independent geometry and V4.1 evidence pipelines.

These tests use actual camera rendering, event sampling, plant dynamics, and
pipeline updates. They do not assert scientific performance or hardware timing.
"""

from unittest.mock import Mock

import numpy as np
import pytest

# The slim Docker smoke-test image intentionally excludes PyTorch.
# Full integration CI installs CPU PyTorch and executes these tests.
pytest.importorskip("torch", reason="cross-pipeline simulation requires CPU PyTorch")

from aurasync.conditions import NOMINAL, PRESETS
from aurasync.physics import PlantParams
from aurasync.simulate import run_episode
from aurasync.v4.geo_pipeline import GeoPipeline
from aurasync.v41.pipeline import EvidencePipeline


def controller_stub():
    controller = Mock()
    controller.act.return_value = 0.0
    return controller


@pytest.mark.parametrize("sensor", ["nominal", "low_light"])
def test_geometry_baseline_runs_with_real_event_sensor(sensor):
    pipeline = GeoPipeline(seed=31)
    n_ticks = 4
    log = run_episode(
        PlantParams(), controller_stub(), pipeline, n_ticks=n_ticks,
        s0=np.array([0.0, 0.0, 0.01, 0.0]),
        cam_seed=17, cond=PRESETS[sensor],
        record=True, stop_on_fail=False,
    )

    assert log["s"].shape == (n_ticks, 4)
    assert log["s_hat"].shape == (n_ticks, 4)
    assert log["surf"].shape == (n_ticks, 4, 32, 96)
    assert log["events"].shape == (n_ticks,)
    assert np.isfinite(log["s_hat"]).all()
    assert np.isfinite(pipeline.covariance()).all()
    assert np.all(log["events"] >= 0)
    assert pipeline.n_fused >= 0
    assert log["fail_tick"] is None


@pytest.mark.parametrize("sensor", ["nominal", "low_light"])
def test_v41_evidence_pipeline_runs_with_real_event_sensor(sensor):
    pipeline = EvidencePipeline(
        Q=np.diag([1e-7, 1e-5, 1e-6, 1e-4]),
        neural=None, seed=31,
    )
    n_ticks = 4
    log = run_episode(
        PlantParams(), controller_stub(), pipeline, n_ticks=n_ticks,
        s0=np.array([0.0, 0.0, 0.01, 0.0]),
        cam_seed=17, cond=PRESETS[sensor],
        record=True, stop_on_fail=False,
    )

    assert log["s"].shape == (n_ticks, 4)
    assert log["s_hat"].shape == (n_ticks, 4)
    assert log["surf"].shape == (n_ticks, 4, 32, 96)
    assert len(pipeline.log["state_seq"]) == n_ticks
    assert sum(pipeline.log["states"]) == n_ticks
    assert pipeline.log["calls"] == 0  # no neural model configured
    assert np.isfinite(log["s_hat"]).all()
    assert np.isfinite(pipeline.covariance()).all()
    assert np.all(log["events"] >= 0)
    assert log["fail_tick"] is None


def test_v41_nominal_episode_is_deterministic_for_same_seeds():
    def episode():
        pipeline = EvidencePipeline(
            Q=np.diag([1e-7, 1e-5, 1e-6, 1e-4]),
            neural=None, seed=47,
        )
        result = run_episode(
            PlantParams(), controller_stub(), pipeline, n_ticks=4,
            s0=np.array([0.0, 0.0, 0.01, 0.0]),
            cam_seed=3, cond=NOMINAL, stop_on_fail=False,
        )
        return result, pipeline.log["state_seq"]

    first, first_states = episode()
    second, second_states = episode()
    np.testing.assert_array_equal(first["s"], second["s"])
    np.testing.assert_array_equal(first["s_hat"], second["s_hat"])
    np.testing.assert_array_equal(first["events"], second["events"])
    assert first_states == second_states
