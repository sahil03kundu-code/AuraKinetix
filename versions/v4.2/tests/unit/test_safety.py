
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

    def test_stale_estimate_uses_fallback_controller(self):
        import numpy as np
        from unittest.mock import Mock, patch
        from aurasync.safety import ShieldedController

        shield = SafetyShield()

        pipeline = Mock()
        pipeline.age_ticks.return_value = 0
        pipeline.covariance.return_value = np.diag([
            0.0, 0.0, np.radians(10.0) ** 2, 0.0
        ])

        normal_lqr = Mock()
        normal_lqr.act.return_value = 3.0

        controller = ShieldedController(
            normal_lqr, shield, pipeline
        )

        state = np.array([
            0.0, 0.0, np.radians(5.0), 0.0
        ])

        with patch.object(
            shield.fallback_ctrl, "act", return_value=2.0
        ) as fallback:
            force = controller.act(state)

        fallback.assert_called_once()
        normal_lqr.act.assert_not_called()

        self.assertEqual(controller.stats["stale"], 1)
        self.assertEqual(controller.stats["ticks"], 1)
        self.assertTrue(np.isfinite(force))
        self.assertLessEqual(abs(force), shield.u_max)

    def test_delay_compensation_replays_previous_forces(self):
        import numpy as np
        from unittest.mock import Mock, patch
        from aurasync.safety import ShieldedController
        from aurasync.physics import rk4_step

        shield = SafetyShield()

        pipeline = Mock()
        pipeline.age_ticks.return_value = 2
        pipeline.covariance.return_value = np.zeros((4, 4))

        nominal_controller = Mock()
        nominal_controller.act.return_value = 0.0

        controller = ShieldedController(
            nominal_controller, shield, pipeline,
            enabled=False
        )

        previous_forces = [2.0, -1.0]
        controller.u_hist.extend(previous_forces)

        initial_state = np.array(
            [0.1, 0.2, np.radians(5.0), 0.1]
        )

        expected_state = initial_state.copy()
        for force in previous_forces:
            expected_state = rk4_step(
                expected_state, force, 1e-3, shield.p
            )

        with patch.object(
            shield.fallback_ctrl, "act", return_value=0.0
        ):
            controller.act(initial_state)

        actual_state = (
            nominal_controller.act.call_args.args[0]
        )

        np.testing.assert_allclose(
            actual_state, expected_state,
            rtol=1e-10, atol=1e-12
        )

        self.assertEqual(controller.stats["ticks"], 1)

if __name__ == "__main__":
    unittest.main()
