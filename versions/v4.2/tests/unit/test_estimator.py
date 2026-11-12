"""Unit tests for the PyTorch estimator."""

import unittest
import numpy as np
import torch

from aurasync.physics import PlantParams, deriv, rk4_step
from aurasync.estimator import TorchCartPole


class TestTorchCartPole(unittest.TestCase):

    def setUp(self):
        self.params = PlantParams()
        self.model = TorchCartPole(
            M=self.params.M,
            m=self.params.m,
            l=self.params.l,
            g=self.params.g,
            b_cart=self.params.b_cart,
            b_pole=self.params.b_pole,
        )

        self.cases = [
            (np.zeros(4), 0.0),
            (np.array([0.1, 0.2, 0.15, -0.3]), 2.0),
            (np.array([-0.2, -0.4, -0.25, 0.5]), -5.0),
            (np.array([0.3, 0.1, 0.4, -1.0]), 10.0),
        ]

    def test_derivatives_match_numpy(self):
        for state, force in self.cases:
            with self.subTest(state=state.tolist(), force=force):
                expected = deriv(state, force, self.params)

                actual = self.model.deriv(
                    torch.tensor(state, dtype=torch.float64),
                    torch.tensor(force, dtype=torch.float64),
                ).detach().numpy()

                np.testing.assert_allclose(
                    actual, expected, rtol=1e-10, atol=1e-12
                )

    def test_rk4_matches_numpy(self):
        dt = 0.001

        for state, force in self.cases:
            with self.subTest(state=state.tolist(), force=force):
                expected = rk4_step(state, force, dt, self.params)

                actual = self.model.step(
                    torch.tensor(state, dtype=torch.float64),
                    torch.tensor(force, dtype=torch.float64),
                    dt,
                ).detach().numpy()

                np.testing.assert_allclose(
                    actual, expected, rtol=1e-10, atol=1e-12
                )



class TestEstimatorEncoders(unittest.TestCase):

    def test_spatial_softmax_output_and_gradients(self):
        from aurasync.estimator import SpatialSoftmaxKeypoints

        torch.manual_seed(42)
        model = SpatialSoftmaxKeypoints(
            cin=4, H=16, W=32, K=3
        )

        x = torch.randn(2, 4, 16, 32, requires_grad=True)
        output = model(x)

        self.assertEqual(output.shape, (2, 15))
        self.assertTrue(torch.isfinite(output).all().item())

        output.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_event_encoder_output_and_gradients(self):
        from aurasync.estimator import EventEncoder

        torch.manual_seed(42)
        model = EventEncoder(
            H=16, W=32, d=16, cin=4
        )

        x = torch.randn(2, 4, 16, 32, requires_grad=True)
        output = model(x)

        self.assertEqual(output.shape, (2, 16))
        self.assertTrue(torch.isfinite(output).all().item())

        output.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())


class TestEventObserver(unittest.TestCase):

    def test_hidden_state_initialization(self):
        from aurasync.estimator import EventObserver

        model = EventObserver(d=16, n_state=8)
        hidden = model.init_hidden(batch=2)

        self.assertEqual(hidden.shape, (2, 8))
        self.assertTrue(torch.all(hidden == 0).item())

    def test_zero_correction_matches_physics(self):
        from aurasync.estimator import EventObserver

        torch.manual_seed(42)
        model = EventObserver(d=16, n_state=8)

        state = torch.tensor(
            [[0.1, 0.2, 0.15, -0.3]],
            dtype=torch.float32
        )
        force = torch.tensor([2.0])
        features = torch.zeros(1, 16)
        hidden = model.init_hidden(batch=1)

        expected = model.physics.step(state, force, model.dt)

        actual, new_hidden, correction = model.cell(
            features, hidden, state, force
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            correction, torch.zeros_like(correction)
        )

        self.assertEqual(new_hidden.shape, (1, 8))
        self.assertTrue(torch.isfinite(new_hidden).all().item())


class TestEventObserverSequence(unittest.TestCase):

    def test_event_observer_sequence_forward(self):
        from aurasync.estimator import EventObserver

        torch.manual_seed(42)
        model = EventObserver(d=16, n_state=8)
        model.eval()

        # Small batch and short sequence for an 8 GB RAM laptop.
        surfaces = torch.zeros(1, 2, 4, 32, 96)
        forces = torch.tensor([[0.0, 2.0]])
        initial_state = torch.zeros(1, 4)

        with torch.no_grad():
            states, corrections = model(
                surfaces, forces, initial_state
            )

        self.assertEqual(states.shape, (1, 2, 4))
        self.assertEqual(corrections.shape, (1, 2, 4))

        self.assertTrue(torch.isfinite(states).all().item())
        self.assertTrue(torch.isfinite(corrections).all().item())

        # The correction head is initially zero.
        torch.testing.assert_close(
            corrections, torch.zeros_like(corrections)
        )

        # Both timesteps should follow physics-only predictions.
        expected_first = model.physics.step(
            initial_state, forces[:, 0], model.dt
        )
        expected_second = model.physics.step(
            expected_first, forces[:, 1], model.dt
        )

        torch.testing.assert_close(states[:, 0], expected_first)
        torch.testing.assert_close(states[:, 1], expected_second)


class TestProbEventObserver(unittest.TestCase):

    def test_initial_uncertainty_and_correction(self):
        from aurasync.estimator import ProbEventObserver

        torch.manual_seed(42)
        model = ProbEventObserver(d=16, n_state=8)

        state = torch.zeros(1, 4)
        features = torch.zeros(1, 16)
        activity = torch.zeros(1, 3)
        hidden = model.init_hidden(batch=1)
        gap = torch.tensor([1.0])

        with torch.no_grad():
            new_state, new_hidden, correction, logvar = (
                model.correct(
                    features, activity, hidden, state, gap
                )
            )

        self.assertEqual(new_state.shape, (1, 4))
        self.assertEqual(new_hidden.shape, (1, 8))
        self.assertEqual(correction.shape, (1, 4))
        self.assertEqual(logvar.shape, (1, 4))

        torch.testing.assert_close(new_state, state)
        torch.testing.assert_close(
            correction, torch.zeros_like(correction)
        )
        torch.testing.assert_close(
            logvar, torch.full_like(logvar, -3.0)
        )

        self.assertTrue(torch.isfinite(new_hidden).all().item())


class TestProbObserverSequence(unittest.TestCase):

    def test_skipped_observations_follow_physics(self):
        from aurasync.estimator import ProbEventObserver

        torch.manual_seed(42)
        model = ProbEventObserver(d=16, n_state=8)
        model.eval()

        surfaces = torch.zeros(1, 3, 4, 32, 96)
        forces = torch.tensor([[1.0, 2.0, 3.0]])
        initial_state = torch.zeros(1, 4)

        # Always skip intermediate corrections.
        skip_probability = torch.tensor([1.0])

        with torch.no_grad():
            states, corrections, logvar, mask = model(
                surfaces,
                forces,
                initial_state,
                skip_p=skip_probability,
                gen=torch.Generator().manual_seed(42),
            )

        self.assertEqual(states.shape, (1, 3, 4))
        self.assertEqual(corrections.shape, (1, 3, 4))
        self.assertEqual(logvar.shape, (1, 3, 4))
        self.assertEqual(mask.shape, (1, 3))

        self.assertEqual(mask.tolist(), [[False, False, True]])

        self.assertTrue(torch.isfinite(states).all().item())
        self.assertTrue(torch.isfinite(logvar).all().item())

        torch.testing.assert_close(
            corrections,
            torch.zeros_like(corrections),
        )

        expected = initial_state
        for k in range(3):
            expected = model.physics_step(
                expected, forces[:, k]
            )
            torch.testing.assert_close(
                states[:, k], expected
            )


class TestEstimatorLosses(unittest.TestCase):

    def test_physics_informed_loss_zero_residual(self):
        from aurasync.estimator import physics_informed_loss

        states = torch.zeros(1, 3, 4)
        corrections = torch.zeros_like(states)
        initial_state = torch.zeros(1, 4)

        loss, metrics = physics_informed_loss(
            states,
            states.clone(),
            corrections,
            initial_state,
            dt=0.001,
        )

        self.assertTrue(torch.isfinite(loss).item())
        torch.testing.assert_close(loss, torch.tensor(0.0))

        for key in ("data", "kin", "reg"):
            self.assertIn(key, metrics)
            self.assertAlmostEqual(metrics[key], 0.0)

    def test_physics_informed_loss_detects_errors(self):
        from aurasync.estimator import physics_informed_loss

        predicted = torch.zeros(1, 3, 4)
        target = torch.zeros_like(predicted)
        target[:, :, 2] = 0.2

        corrections = torch.ones_like(predicted) * 0.1
        initial_state = torch.zeros(1, 4)

        loss, metrics = physics_informed_loss(
            predicted,
            target,
            corrections,
            initial_state,
            dt=0.001,
        )

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(metrics["data"], 0.0)
        self.assertGreater(metrics["reg"], 0.0)


class TestProbObserverAPS(unittest.TestCase):

    def test_aps_channel_is_ignored_when_disabled(self):
        from aurasync.estimator import ProbEventObserver

        torch.manual_seed(42)
        model = ProbEventObserver(
            d=16, n_state=8, use_aps=False
        )
        model.eval()

        first = torch.zeros(1, 4, 32, 96)
        second = first.clone()

        # Change only the APS channel.
        second[:, 3] = 1.0

        with torch.no_grad():
            z_first, activity_first = model.encode(first)
            z_second, activity_second = model.encode(second)

        torch.testing.assert_close(z_first, z_second)
        torch.testing.assert_close(
            activity_first, activity_second
        )

        self.assertEqual(z_first.shape, (1, 16))
        self.assertEqual(activity_first.shape, (1, 3))


class TestGaussianNLL(unittest.TestCase):

    def test_gaussian_nll_error_and_mask(self):
        from aurasync.estimator import gaussian_nll

        target = torch.zeros(1, 3, 4)
        predicted = torch.zeros_like(target)
        logvar = torch.zeros_like(target)

        mask = torch.tensor([[True, True, False]])

        perfect_loss = gaussian_nll(
            predicted, target, logvar, mask
        )
        torch.testing.assert_close(
            perfect_loss, torch.tensor(0.0)
        )

        # Add an error at an included timestep.
        predicted[:, 0, 2] = 0.2

        error_loss = gaussian_nll(
            predicted, target, logvar, mask
        )

        self.assertTrue(torch.isfinite(error_loss).item())
        self.assertGreater(error_loss.item(), 0.0)

        # An error at an excluded timestep must not affect loss.
        predicted[:, 2, 2] = 100.0

        masked_loss = gaussian_nll(
            predicted, target, logvar, mask
        )

        torch.testing.assert_close(
            masked_loss, error_loss
        )


class TestProbObserverDense(unittest.TestCase):

    def test_dense_observations_are_all_processed(self):
        from aurasync.estimator import ProbEventObserver

        torch.manual_seed(42)
        model = ProbEventObserver(d=16, n_state=8)
        model.eval()

        surfaces = torch.zeros(1, 2, 4, 32, 96)
        forces = torch.tensor([[1.0, 2.0]])
        initial_state = torch.zeros(1, 4)

        with torch.no_grad():
            states, corrections, logvar, mask = model(
                surfaces, forces, initial_state
            )

        self.assertEqual(mask.tolist(), [[True, True]])
        self.assertEqual(states.shape, (1, 2, 4))
        self.assertEqual(corrections.shape, (1, 2, 4))
        self.assertEqual(logvar.shape, (1, 2, 4))

        self.assertTrue(torch.isfinite(states).all().item())
        self.assertTrue(torch.isfinite(logvar).all().item())


class TestEstimatorCheckpointing(unittest.TestCase):

    def test_large_batch_uses_gradient_checkpointing(self):
        from unittest.mock import patch
        import torch.utils.checkpoint as cp
        from aurasync.estimator import ProbEventObserver

        torch.manual_seed(42)
        model = ProbEventObserver(d=16, n_state=8)

        # 257 observations trigger the >256 checkpoint branch.
        # One timestep keeps the recurrent workload small.
        surfaces = torch.zeros(257, 1, 4, 32, 96)
        forces = torch.zeros(257, 1)
        initial_state = torch.zeros(257, 4)

        def lightweight_encode(ts):
            mean = ts.mean(dim=(1, 2, 3), keepdim=False)
            features = mean[:, None].expand(-1, 16)
            activity = ts[:, :3].mean((2, 3)) * 10.0
            return features, activity

        with torch.enable_grad():
            with patch.object(model, "encode",
                              side_effect=lightweight_encode):
                with patch.object(
                    cp, "checkpoint", wraps=cp.checkpoint
                ) as checkpoint_spy:
                    states, corrections, logvar, mask = model(
                        surfaces, forces, initial_state
                    )

        self.assertEqual(checkpoint_spy.call_count, 2)
        self.assertEqual(states.shape, (257, 1, 4))
        self.assertEqual(corrections.shape, (257, 1, 4))
        self.assertEqual(logvar.shape, (257, 1, 4))
        self.assertTrue(mask.all().item())
        self.assertTrue(torch.isfinite(states).all().item())

if __name__ == "__main__":
    unittest.main()
