
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

    def test_nan_force_is_rejected(self):
        import numpy as np

        shield = SafetyShield()
        state = np.zeros(4)
        covariance = np.zeros((4, 4))

        with self.assertRaises(ValueError):
            shield.filter(
                state,
                covariance,
                float("nan")
            )

    def test_infinite_forces_are_rejected(self):
        import numpy as np

        shield = SafetyShield()
        state = np.zeros(4)
        covariance = np.zeros((4, 4))

        for invalid_force in [float("inf"), float("-inf")]:
            with self.subTest(force=invalid_force):
                with self.assertRaises(ValueError):
                    shield.filter(
                        state, covariance, invalid_force
                    )

    def test_invalid_state_and_covariance_shapes(self):
        import numpy as np

        shield = SafetyShield()
        valid_state = np.zeros(4)
        valid_covariance = np.zeros((4, 4))

        for length in [3, 5]:
            with self.subTest(state_length=length):
                with self.assertRaisesRegex(
                    ValueError, "state"
                ):
                    shield.filter(
                        np.zeros(length),
                        valid_covariance,
                        0.0
                    )

        for shape in [(3, 3), (5, 5)]:
            with self.subTest(covariance_shape=shape):
                with self.assertRaisesRegex(
                    ValueError, "covariance"
                ):
                    shield.filter(
                        valid_state,
                        np.zeros(shape),
                        0.0
                    )

    def test_nonfinite_state_and_covariance_rejected(self):
        import numpy as np

        valid_state = np.zeros(4)
        valid_covariance = np.zeros((4, 4))

        for bad_value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=bad_value, input="state"):
                state = valid_state.copy()
                state[2] = bad_value

                with self.assertRaisesRegex(ValueError, "state"):
                    SafetyShield().filter(
                        state, valid_covariance, 0.0
                    )

            with self.subTest(value=bad_value, input="covariance"):
                covariance = valid_covariance.copy()
                covariance[2, 2] = bad_value

                with self.assertRaisesRegex(ValueError, "covariance"):
                    SafetyShield().filter(
                        valid_state, covariance, 0.0
                    )

    def test_pipeline_fallback_age_and_covariance(self):
        import numpy as np
        from unittest.mock import Mock
        from aurasync.safety import ShieldedController

        class LegacyPipeline:
            delay_ticks = 3

        sigma = np.array([0.1, 0.2, 0.3, 0.4])

        controller = ShieldedController(
            Mock(), SafetyShield(), LegacyPipeline(),
            sigma_const=sigma
        )

        self.assertEqual(controller._age(), 3)
        np.testing.assert_allclose(
            controller._cov(),
            np.diag(sigma ** 2)
        )

    def test_soft_constraints_remain_disabled_during_recovery(self):
        import numpy as np

        shield = SafetyShield()
        covariance = np.zeros((4, 4))

        # Enter recovery mode with a large pole angle.
        dangerous = np.array([
            0.0, 0.0, np.radians(25), 0.0
        ])
        shield.filter(dangerous, covariance, 0.0)
        self.assertTrue(shield.soft_off)

        # Reduce the angle, but keep angular velocity too high
        # to satisfy the recovery/reset conditions.
        recovering = np.array([
            0.0, 0.0, 0.0, 1.0
        ])
        shield.filter(recovering, covariance, 0.0)

        self.assertTrue(shield.soft_off)

if __name__ == "__main__":
    unittest.main()
