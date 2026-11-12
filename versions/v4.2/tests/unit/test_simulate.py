"""Lightweight regression tests for closed-loop simulation and baseline pipelines."""

from unittest.mock import Mock

import numpy as np

from aurasync.physics import PlantParams
from aurasync.simulate import (
    DT,
    SUB,
    FramePipeline,
    OraclePipeline,
    run_episode,
)


def test_oracle_pipeline_copies_initial_and_updated_state():
    pipeline = OraclePipeline()
    initial = np.array([0.1, 0.0, 0.03, 0.0])
    pipeline.reset(initial)
    np.testing.assert_array_equal(pipeline.estimate(), initial)
    initial[0] = 99.0
    assert pipeline.estimate()[0] != 99.0

    current = np.array([0.2, 0.0, 0.04, 0.0])
    pipeline.update(0, 0.001, current, None, 0.0)
    np.testing.assert_array_equal(pipeline.estimate(), current)
    current[0] = 99.0
    assert pipeline.estimate()[0] != 99.0


def test_frame_pipeline_holds_result_until_latency_elapsed():
    pipeline = FramePipeline(fps=20, latency=0.003)
    initial = np.zeros(4)
    captured = np.array([0.2, 0.0, 0.1, 0.0])
    pipeline.reset(initial)
    for tick, t in enumerate((0.0, 0.001, 0.002)):
        pipeline.update(tick, t, captured, None, 0.0)
        np.testing.assert_array_equal(pipeline.estimate(), initial)
    pipeline.update(3, 0.003, captured, None, 0.0)
    np.testing.assert_array_equal(pipeline.estimate(), captured)


def test_frame_pipeline_capture_schedule_respects_fps():
    pipeline = FramePipeline(fps=10, latency=1.0)
    pipeline.reset(np.zeros(4))
    pipeline.update(0, 0.0, np.ones(4), None, 0.0)
    pipeline.update(1, 0.05, np.ones(4) * 2, None, 0.0)
    pipeline.update(2, 0.1, np.ones(4) * 3, None, 0.0)
    assert len(pipeline.pending) == 2
    assert pipeline.next_capture == 0.2


def test_closed_loop_episode_returns_finite_arrays_and_event_counts():
    controller = Mock()
    controller.act.return_value = 0.0
    ticks = 3
    log = run_episode(
        PlantParams(), controller, OraclePipeline(), n_ticks=ticks,
        s0=np.array([0.0, 0.0, 0.01, 0.0]), cam_seed=42,
        stop_on_fail=False, aps_fps=0,
    )
    assert log["s"].shape == (ticks, 4)
    assert log["s_hat"].shape == (ticks, 4)
    assert log["u"].shape == (ticks,)
    assert log["events"].shape == (ticks,)
    assert np.isfinite(log["s"]).all()
    assert np.isfinite(log["s_hat"]).all()
    assert np.all(log["events"] >= 0)
    np.testing.assert_array_equal(log["s"], log["s_hat"])
    assert log["fail_tick"] is None
    assert controller.act.call_count == ticks
    assert SUB == 5 and DT == 1e-3


def test_recorded_episode_captures_four_channel_surfaces():
    controller = Mock()
    controller.act.return_value = 0.0
    log = run_episode(
        PlantParams(), controller, OraclePipeline(), n_ticks=2,
        s0=np.zeros(4), record=True, cam_seed=7,
        stop_on_fail=False, aps_fps=0,
    )
    assert log["surf"].shape == (2, 4, 32, 96)
    assert log["surf"].dtype == np.float16


def test_exploration_control_is_clipped_to_plant_limits():
    controller = Mock()
    controller.act.return_value = 0.0
    params = PlantParams()
    log = run_episode(
        params, controller, OraclePipeline(), n_ticks=1,
        s0=np.zeros(4), exploration=lambda: params.u_max * 10,
        stop_on_fail=False, aps_fps=0,
    )
    np.testing.assert_allclose(log["u"], [params.u_max])
