"""Deterministic unit tests for the AuraKinetix physics engine."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aurasync.physics import PlantParams, deriv, rk4_step, failed, linearise


class TestPhysics(unittest.TestCase):

    def setUp(self):
        self.p = PlantParams()
        self.zero = np.zeros(4)

    def test_default_parameters_are_positive(self):
        self.assertGreater(self.p.M, 0)
        self.assertGreater(self.p.m, 0)
        self.assertGreater(self.p.l, 0)
        self.assertGreater(self.p.g, 0)
        self.assertGreater(self.p.u_max, 0)

    def test_upright_equilibrium_has_zero_derivative(self):
        result = deriv(self.zero, 0.0, self.p)
        np.testing.assert_allclose(result, self.zero, atol=1e-12)

    def test_rk4_preserves_upright_equilibrium(self):
        result = rk4_step(self.zero, 0.0, 0.001, self.p)
        np.testing.assert_allclose(result, self.zero, atol=1e-12)

    def test_failure_angle_boundary(self):
        safe = np.array([0.0, 0.0, 0.8, 0.0])
        unsafe = np.array([0.0, 0.0, 0.801, 0.0])

        self.assertFalse(failed(safe, self.p))
        self.assertTrue(failed(unsafe, self.p))

    def test_failure_track_boundary(self):
        safe = np.array([self.p.x_limit, 0.0, 0.0, 0.0])
        unsafe = np.array([self.p.x_limit + 0.001, 0.0, 0.0, 0.0])

        self.assertFalse(failed(safe, self.p))
        self.assertTrue(failed(unsafe, self.p))

    def test_linearisation_returns_finite_matrices(self):
        A, B = linearise(self.p, 0.001)

        self.assertEqual(A.shape, (4, 4))
        self.assertEqual(B.shape, (4, 1))
        self.assertTrue(np.isfinite(A).all())
        self.assertTrue(np.isfinite(B).all())


    def test_perturbed_parameters_stay_within_bounds(self):
        rng = np.random.default_rng(42)
        original = PlantParams()

        for _ in range(100):
            changed = original.perturbed(rng, frac=0.3)

            self.assertGreaterEqual(changed.M, original.M * 0.7)
            self.assertLessEqual(changed.M, original.M * 1.3)

            self.assertGreaterEqual(changed.m, original.m * 0.7)
            self.assertLessEqual(changed.m, original.m * 1.3)

            self.assertGreaterEqual(changed.l, original.l * 0.9)
            self.assertLessEqual(changed.l, original.l * 1.1)

            self.assertGreater(changed.M, 0)
            self.assertGreater(changed.m, 0)
            self.assertGreater(changed.l, 0)

        self.assertEqual(original, PlantParams())

if __name__ == "__main__":
    unittest.main()
