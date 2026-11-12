
"""Unit tests for AuraKinetix safety constraints."""

import unittest

from aurasync.safety import SafetyShield


class TestSafetyIntervals(unittest.TestCase):

    def test_positive_coefficient(self):
        lo, hi = SafetyShield._interval(0, 2, -4, 4)
        self.assertEqual((lo, hi), (-2, 2))

    def test_negative_coefficient(self):
        lo, hi = SafetyShield._interval(0, -2, -4, 4)
        self.assertEqual((lo, hi), (-2, 2))

    def test_zero_coefficient_feasible(self):
        lo, hi = SafetyShield._interval(1, 0, -2, 2)
        self.assertEqual((lo, hi), (-float("inf"), float("inf")))

    def test_zero_coefficient_infeasible(self):
        lo, hi = SafetyShield._interval(5, 0, -2, 2)
        self.assertEqual((lo, hi), (float("inf"), -float("inf")))


    def test_zero_force_passes_at_equilibrium(self):
        import numpy as np

        shield = SafetyShield()
        state = np.zeros(4)
        covariance = np.zeros((4, 4))

        force, mode = shield.filter(state, covariance, 0.0)

        self.assertAlmostEqual(force, 0.0)
        self.assertEqual(mode, "pass")

    def test_excessive_force_is_limited(self):
        import numpy as np

        shield = SafetyShield()
        state = np.zeros(4)
        covariance = np.zeros((4, 4))

        force, mode = shield.filter(state, covariance, 100.0)

        self.assertEqual(mode, "modified")
        self.assertLessEqual(abs(force), shield.u_max)
        self.assertLess(force, 100.0)

    def test_soft_constraints_dropped_during_recovery(self):
        import numpy as np

        shield = SafetyShield()
        state = np.array([0.0, 0.0, np.radians(25), 0.0])
        covariance = np.zeros((4, 4))

        force, mode = shield.filter(state, covariance, 0.0)

        self.assertEqual(mode, "soft_dropped")
        self.assertTrue(shield.soft_off)
        self.assertLessEqual(abs(force), shield.u_max)

    def test_infeasible_hard_constraints_use_clipped_nominal(self):
        import numpy as np

        shield = SafetyShield()
        state = np.array([0.0, 0.0, np.radians(40), 0.0])
        covariance = np.zeros((4, 4))
        nominal_force = 100.0

        force, mode = shield.filter(
            state, covariance, nominal_force
        )

        self.assertEqual(mode, "infeasible")
        self.assertAlmostEqual(force, shield.u_max)

if __name__ == "__main__":
    unittest.main()
