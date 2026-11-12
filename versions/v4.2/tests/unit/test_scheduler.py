"""Unit tests for the event-triggered scheduler."""

import unittest
from unittest.mock import Mock

import numpy as np
import torch

from aurasync.scheduler import TriggeredEventPipeline


class TestSchedulerInitialization(unittest.TestCase):

    def test_reset_initializes_state_and_buffers(self):
        model = Mock()
        model.eval.return_value = model
        model.init_hidden.return_value = torch.zeros(1, 8)

        pipeline = TriggeredEventPipeline(
            models=model,
            mode="triggered",
            seed=42,
        )

        initial_state = np.array([0.1, 0.2, 0.15, -0.3])
        pipeline.reset(initial_state)

        np.testing.assert_allclose(
            pipeline.estimate(), initial_state
        )

        self.assertEqual(pipeline.covariance().shape, (4, 4))
        self.assertEqual(pipeline.age_ticks(), 0)
        self.assertEqual(pipeline.k, -1)
        self.assertIsNone(pipeline.job)
        self.assertEqual(len(pipeline.u_hist), 0)
        self.assertEqual(pipeline.log["calls"], 0)
        self.assertEqual(pipeline.log["misses"], 0)

        # Reset must copy the initial state.
        initial_state[0] = 999.0
        self.assertNotEqual(pipeline.estimate()[0], 999.0)


if __name__ == "__main__":
    unittest.main()

class TestSchedulerTriggers(unittest.TestCase):

    def setUp(self):
        model = Mock()
        model.eval.return_value = model
        model.init_hidden.return_value = torch.zeros(1, 8)

        self.pipeline = TriggeredEventPipeline(
            models=model,
            mode="triggered",
            max_gap=20,
            burst_thr=12,
        )
        self.pipeline.reset(np.zeros(4))
        self.pipeline.Sigma[:] = 0.0
        self.surface = np.zeros((2, 4, 4), dtype=np.float32)

    def test_greedy_mode(self):
        self.pipeline.mode = "greedy"
        self.assertEqual(self.pipeline._trigger(self.surface), "idle")

    def test_periodic_mode(self):
        self.pipeline.mode = "periodic"
        self.pipeline.k = 4
        self.assertIsNone(self.pipeline._trigger(self.surface))

        self.pipeline.k = 5
        self.assertEqual(self.pipeline._trigger(self.surface), "periodic")

    def test_maximum_gap(self):
        self.pipeline.k = 20
        self.assertEqual(self.pipeline._trigger(self.surface), "gap")

    def test_event_burst(self):
        self.surface[0, :, :] = 1.0
        self.assertEqual(self.pipeline._trigger(self.surface), "burst")

    def test_uncertainty_trigger(self):
        self.pipeline.Sigma[2, 2] = np.radians(2.0) ** 2
        self.assertEqual(self.pipeline._trigger(self.surface), "doubt")

    def test_safety_margin_trigger(self):
        self.pipeline.s[2] = np.radians(8.0)
        self.assertEqual(self.pipeline._trigger(self.surface), "margin")

    def test_quiet_conditions(self):
        self.assertIsNone(self.pipeline._trigger(self.surface))

class TestSchedulerUpdate(unittest.TestCase):

    def test_deadline_miss_preserves_physics_fallback(self):
        from unittest.mock import patch

        model = Mock()
        model.eval.return_value = model
        model.init_hidden.return_value = torch.zeros(1, 8)

        pipeline = TriggeredEventPipeline(
            models=model,
            mode="triggered",
            deadline=2,
            max_gap=100,
        )
        pipeline.reset(np.zeros(4))

        surface = np.zeros((2, 4, 4), dtype=np.float32)

        corrected_state = np.full(4, 999.0)
        corrected_cov = np.eye(4) * 0.01

        pipeline.job = (
            0, 10, corrected_state, corrected_cov, ["late_hidden"]
        )

        # Keep triggers disabled to isolate deadline handling.
        with patch.object(pipeline, "_trigger", return_value=None):
            pipeline.update(
                k=2,
                t=0.002,
                s_true=np.zeros(4),
                surface=surface,
                u=0.0,
            )

        self.assertEqual(pipeline.log["misses"], 1)
        self.assertIsNone(pipeline.job)
        self.assertFalse(pipeline.log["corrected"][-1])
        self.assertFalse(np.allclose(pipeline.estimate(), corrected_state))
        self.assertTrue(np.all(np.isfinite(pipeline.estimate())))

    def test_completed_correction_replays_controls(self):
        from unittest.mock import patch

        model = Mock()
        model.eval.return_value = model
        model.init_hidden.return_value = torch.zeros(1, 8)

        pipeline = TriggeredEventPipeline(
            models=model,
            mode="triggered",
            deadline=6,
            max_gap=100,
        )
        pipeline.reset(np.zeros(4))

        corrected_state = np.array([0.1, 0.2, 0.3, 0.4])
        corrected_cov = np.eye(4) * 0.01
        new_hidden = ["updated_hidden"]

        # Correction captured at tick 0, ready at tick 2.
        pipeline.job = (
            0, 2, corrected_state.copy(),
            corrected_cov.copy(),
            new_hidden,
        )

        pipeline.u_hist.append(0.25)
        surface = np.zeros((2, 4, 4), dtype=np.float32)

        with patch.object(pipeline, "_trigger", return_value=None):
            with patch.object(
                pipeline,
                "_propagate",
                side_effect=lambda s, cov, u: (
                    s + u,
                    cov + np.eye(4) * 0.001,
                ),
            ) as propagate:
                pipeline.update(
                    k=2,
                    t=0.002,
                    s_true=np.zeros(4),
                    surface=surface,
                    u=0.5,
                )

        # One normal prediction + two replayed controls.
        self.assertEqual(propagate.call_count, 3)

        expected_state = corrected_state + 0.25 + 0.5
        np.testing.assert_allclose(pipeline.estimate(), expected_state)

        self.assertEqual(pipeline.h, new_hidden)
        self.assertTrue(pipeline.log["corrected"][-1])
        self.assertIsNone(pipeline.job)
        self.assertEqual(pipeline.log["misses"], 0)

    def test_pending_job_stays_busy_without_new_correction(self):
        from unittest.mock import patch

        model = Mock()
        model.eval.return_value = model
        model.init_hidden.return_value = torch.zeros(1, 8)

        pipeline = TriggeredEventPipeline(
            models=model,
            mode="triggered",
            deadline=10,
            max_gap=100,
        )
        pipeline.reset(np.zeros(4))

        pipeline.job = (
            0,
            5,
            np.ones(4),
            np.eye(4) * 0.01,
            ["pending_hidden"],
        )

        surface = np.zeros((2, 4, 4), dtype=np.float32)

        with patch.object(pipeline, "_trigger") as trigger:
            with patch.object(pipeline, "_correct") as correct:
                pipeline.update(
                    k=2,
                    t=0.002,
                    s_true=np.zeros(4),
                    surface=surface,
                    u=0.0,
                )

        trigger.assert_not_called()
        correct.assert_not_called()

        self.assertEqual(pipeline.log["busy_ticks"], 1)
        self.assertEqual(pipeline.log["calls"], 0)
        self.assertFalse(pipeline.log["corrected"][-1])
        self.assertIsNotNone(pipeline.job)

    def test_trigger_starts_new_correction_job(self):
        from unittest.mock import patch

        model = Mock()
        model.eval.return_value = model
        model.init_hidden.return_value = torch.zeros(1, 8)

        pipeline = TriggeredEventPipeline(
            models=model,
            mode="triggered",
            latency_samples=(0.002,),
            max_gap=100,
            burst_thr=12,
        )
        pipeline.reset(np.zeros(4))

        surface = np.ones((2, 4, 4), dtype=np.float32)
        corrected_state = np.ones(4)
        corrected_cov = np.eye(4) * 0.01
        new_hidden = ["new_hidden"]

        with patch.object(
            pipeline,
            "_correct",
            return_value=(corrected_state, corrected_cov, new_hidden),
        ) as correct:
            pipeline.update(
                k=1,
                t=0.001,
                s_true=np.zeros(4),
                surface=surface,
                u=0.0,
            )

        correct.assert_called_once()
        self.assertEqual(pipeline.job[0], 1)
        self.assertEqual(pipeline.job[1], 3)
        self.assertEqual(pipeline.last_capture, 1)
        self.assertEqual(pipeline.log["calls"], 1)
        self.assertEqual(pipeline.log["triggers"]["burst"], 1)
        self.assertFalse(pipeline.log["corrected"][-1])

    def test_trigger_starts_new_correction_job(self):
        from unittest.mock import patch

        model = Mock()
        model.eval.return_value = model
        model.init_hidden.return_value = torch.zeros(1, 8)

        pipeline = TriggeredEventPipeline(
            models=model,
            mode="triggered",
            latency_samples=(0.002,),
            max_gap=100,
            burst_thr=12,
        )
        pipeline.reset(np.zeros(4))

        surface = np.ones((2, 4, 4), dtype=np.float32)
        corrected_state = np.ones(4)
        corrected_cov = np.eye(4) * 0.01
        new_hidden = ["new_hidden"]

        with patch.object(
            pipeline,
            "_correct",
            return_value=(corrected_state, corrected_cov, new_hidden),
        ) as correct:
            pipeline.update(
                k=1,
                t=0.001,
                s_true=np.zeros(4),
                surface=surface,
                u=0.0,
            )

        correct.assert_called_once()
        self.assertEqual(pipeline.job[0], 1)
        self.assertEqual(pipeline.job[1], 3)
        self.assertEqual(pipeline.last_capture, 1)
        self.assertEqual(pipeline.log["calls"], 1)
        self.assertEqual(pipeline.log["triggers"]["burst"], 1)
        self.assertFalse(pipeline.log["corrected"][-1])

    def test_dense_mode_applies_correction(self):
        from unittest.mock import patch

        model = Mock()
        model.eval.return_value = model
        model.init_hidden.return_value = torch.zeros(1, 8)

        pipeline = TriggeredEventPipeline(
            models=model,
            mode="dense",
        )
        pipeline.reset(np.zeros(4))

        corrected_state = np.array([0.1, 0.2, 0.3, 0.4])
        corrected_cov = np.eye(4) * 0.01
        new_hidden = ["dense_hidden"]

        surface = np.zeros((2, 4, 4), dtype=np.float32)

        with patch.object(
            pipeline,
            "_correct",
            return_value=(corrected_state, corrected_cov, new_hidden),
        ) as correct:
            pipeline.update(
                k=0,
                t=0.0,
                s_true=np.zeros(4),
                surface=surface,
                u=0.0,
            )

        correct.assert_called_once()
        np.testing.assert_allclose(
            pipeline.s, corrected_state
        )
        np.testing.assert_allclose(
            pipeline.Sigma, corrected_cov
        )

        self.assertEqual(pipeline.age_ticks(), 1)
        self.assertEqual(pipeline.log["calls"], 1)
        self.assertTrue(pipeline.log["corrected"][-1])

class TestSchedulerNeuralCorrection(unittest.TestCase):

    def test_single_model_correction(self):
        model = Mock()
        model.eval.return_value = model
        model.init_hidden.return_value = torch.zeros(1, 8)

        model.encode.return_value = (
            torch.zeros(1, 8),
            torch.zeros(1, 8),
        )

        predicted_state = np.array([0.1, 0.2, 0.3, 0.4])

        model.correct.return_value = (
            torch.from_numpy(predicted_state.astype(np.float32)).unsqueeze(0),
            torch.ones(1, 8),
            None,
            torch.zeros(1, 4),
        )

        pipeline = TriggeredEventPipeline(models=model)
        pipeline.reset(np.zeros(4))

        surface = np.zeros((4, 32, 96), dtype=np.float32)

        state, covariance, hidden = pipeline._correct(
            surface,
            predicted_state,
            gap=1,
        )

        np.testing.assert_allclose(
            state, predicted_state, atol=1e-6
        )

        self.assertEqual(covariance.shape, (4, 4))
        self.assertTrue(np.all(np.diag(covariance) > 0))
        self.assertEqual(len(hidden), 1)
        model.encode.assert_called_once()
        model.correct.assert_called_once()

class TestSchedulerPerformance(unittest.TestCase):

    def test_measure_correction_latency(self):
        from aurasync.scheduler import measure_correction_latency

        model = Mock()
        model.init_hidden.return_value = torch.zeros(1, 8)
        model.encode.return_value = (
            torch.zeros(1, 8),
            torch.zeros(1, 8),
        )
        model.correct.return_value = None

        timings = measure_correction_latency(
            model, n=3, warm=1
        )

        self.assertEqual(timings.shape, (3,))
        self.assertTrue(np.all(np.isfinite(timings)))
        self.assertTrue(np.all(timings >= 0))
        self.assertEqual(model.encode.call_count, 4)
        self.assertEqual(model.correct.call_count, 4)

    def test_physics_tick_cost(self):
        from aurasync.scheduler import physics_tick_cost

        timings = physics_tick_cost(n=3)

        self.assertEqual(timings.shape, (3,))
        self.assertTrue(np.all(np.isfinite(timings)))
        self.assertTrue(np.all(timings >= 0))
