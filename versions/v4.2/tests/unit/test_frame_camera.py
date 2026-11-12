"""Deterministic unit tests for frame-camera acquisition and inference interfaces."""

from collections import deque
from unittest.mock import Mock

import numpy as np
import torch

from aurasync.conditions import NOMINAL, SensorConditions
from aurasync.frame_camera import (
    BasicBlock,
    FrameEstimator,
    LearnedFramePipeline,
    N_FRAMES,
    blur_frame,
    cond_vector,
)


def model_stub(state=None):
    model = Mock()
    model.eval.return_value = model
    if state is None:
        state = [0.0, 0.0, 0.0, 0.0]
    model.return_value = torch.tensor([state], dtype=torch.float32)
    return model


def test_blur_frame_uses_latest_exposure_ticks():
    history = deque(
        [np.full((2, 3), value) for value in (1.0, 2.0, 4.0, 8.0)],
        maxlen=16,
    )
    np.testing.assert_allclose(blur_frame(history, 2), 6.0)
    np.testing.assert_allclose(blur_frame(history, 4), 3.75)


def test_conditioning_vector_has_expected_units_and_shape():
    result = cond_vector(fps=30, exposure=0.004)
    assert result.shape == (1, 2)
    assert result.dtype == torch.float32
    np.testing.assert_allclose(
        result.numpy(), [[(1.0 / 30.0) / 0.0333, 0.5]], rtol=1e-6
    )


def test_basic_block_identity_and_projection_shapes():
    identity = BasicBlock(8, 8).eval()
    projection = BasicBlock(8, 16, stride=2).eval()
    x = torch.zeros((1, 8, 16, 24))
    with torch.inference_mode():
        assert identity(x).shape == (1, 8, 16, 24)
        assert projection(x).shape == (1, 16, 8, 12)


def test_frame_estimator_output_shape_and_finiteness():
    torch.manual_seed(7)
    model = FrameEstimator(H=32, W=96, K=4).eval()
    images = torch.full((1, N_FRAMES, 32, 96), 0.5)
    with torch.inference_mode():
        output = model(images, cond_vector(fps=30, exposure=0.004))
    assert output.shape == (1, 4)
    assert torch.isfinite(output).all()


def test_pipeline_reset_copies_state_and_initializes_buffers():
    pipeline = LearnedFramePipeline(
        model_stub(), fps=30, latency=0.002, read_noise=0.0
    )
    initial = np.zeros(4)
    pipeline.reset(initial)
    assert len(pipeline.hist) == 16
    assert len(pipeline.frames) == 0
    assert len(pipeline.pending) == 0
    assert len(pipeline.u_hist) == 0
    np.testing.assert_array_equal(pipeline.estimate(), initial)
    initial[0] = 5.0
    assert pipeline.estimate()[0] == 0.0


def test_capture_stacks_three_frames_and_is_deterministic_without_noise():
    pipeline = LearnedFramePipeline(
        model_stub(), fps=30, latency=0.002, read_noise=0.0
    )
    pipeline.reset(np.zeros(4))
    stack = pipeline._capture()
    assert stack.shape == (N_FRAMES, 32, 96)
    assert stack.dtype == np.float32
    np.testing.assert_array_equal(stack[0], stack[1])
    np.testing.assert_array_equal(stack[1], stack[2])
    assert len(pipeline.frames) == N_FRAMES


def test_set_conditions_updates_exposure_and_nominal_restores_it():
    pipeline = LearnedFramePipeline(
        model_stub(), fps=30, latency=0.002, exposure=0.004
    )
    low = SensorConditions(name="low_light_test", brightness=0.5)
    pipeline.set_conditions(low)
    assert pipeline.exposure == 0.008
    assert pipeline.exp_ticks == 8
    assert pipeline.noise > 0.0
    pipeline.set_conditions(NOMINAL)
    assert pipeline.exposure == 0.004
    assert pipeline.exp_ticks == 4


def test_zero_latency_capture_applies_inference_in_zoh_mode():
    expected = [0.12, 0.0, 0.04, 0.0]
    model = model_stub(expected)
    pipeline = LearnedFramePipeline(
        model, fps=30, latency=0.0, mode="zoh", read_noise=0.0
    )
    pipeline.reset(np.zeros(4))
    pipeline.update(
        k=0, t=0.0, s_true=np.zeros(4), surface=None, u=0.0
    )
    np.testing.assert_allclose(pipeline.estimate(), expected, atol=1e-6)
    assert len(pipeline.pending) == 0
    assert len(pipeline.frames) == N_FRAMES
    model.assert_called_once()


def test_capture_applies_non_nominal_sensor_response():
    model = model_stub()
    pipeline = LearnedFramePipeline(
        model, fps=30, latency=0.002, read_noise=0.0
    )
    conditions = SensorConditions(name="dim", brightness=0.5)
    pipeline.set_conditions(conditions)
    pipeline.reset(np.zeros(4))
    assert pipeline.noise > 0.0  # Low light correctly enables read noise.
    pipeline.noise = 0.0  # Isolate deterministic sensor-response mapping.
    image = pipeline.hist[-1].copy()
    stack = pipeline._capture()
    expected = conditions.frame_sensor(image).astype(np.float32)
    np.testing.assert_allclose(stack[0], expected, atol=1e-6)


def test_capture_waits_for_latency_before_using_estimator():
    expected = [0.15, 0.0, 0.0, 0.0]
    model = model_stub(expected)
    pipeline = LearnedFramePipeline(
        model, fps=30, latency=0.003, mode="zoh", read_noise=0.0
    )
    pipeline.reset(np.zeros(4))
    for tick in range(3):
        pipeline.update(
            k=tick, t=tick * 0.001,
            s_true=np.zeros(4), surface=None, u=0.0,
        )
        model.assert_not_called()
    assert len(pipeline.pending) == 1
    pipeline.update(
        k=3, t=0.003, s_true=np.zeros(4), surface=None, u=0.0
    )
    model.assert_called_once()
    assert len(pipeline.pending) == 0
    np.testing.assert_allclose(pipeline.estimate(), expected, atol=1e-6)


def test_physics_mode_replays_controls_after_delayed_frame():
    from aurasync.physics import rk4_step

    measured = [0.1, 0.0, 0.04, 0.0]
    model = model_stub(measured)
    pipeline = LearnedFramePipeline(
        model, fps=30, latency=0.002, mode="physics", read_noise=0.0
    )
    pipeline.reset(np.zeros(4))
    controls = (0.0, 0.1, -0.1)
    for tick, u in enumerate(controls):
        pipeline.update(
            k=tick, t=tick * 0.001,
            s_true=np.zeros(4), surface=None, u=u,
        )
    expected = np.asarray(measured, dtype=np.float64)
    for u in controls[-2:]:
        expected = rk4_step(expected, u, 0.001, pipeline.p)
    np.testing.assert_allclose(
        pipeline.estimate(), expected, rtol=1e-6, atol=1e-6
    )
    model.assert_called_once()


def test_pipeline_uses_true_plant_length_for_rendering():
    from types import SimpleNamespace
    from unittest.mock import patch

    pipeline = LearnedFramePipeline(
        model_stub(), fps=30, latency=0.005, read_noise=0.0
    )
    pipeline.p_true = SimpleNamespace(l=0.4)
    with patch.object(pipeline.renderer, "render", wraps=pipeline.renderer.render) as render:
        pipeline.reset(np.zeros(4))
        assert render.call_args.args[1] == 0.4
        pipeline.update(
            k=0, t=0.0, s_true=np.zeros(4), surface=None, u=0.0
        )
        assert render.call_args.args[1] == 0.4
