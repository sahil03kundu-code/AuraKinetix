"""Unit tests for the LQR controller."""

import unittest
import numpy as np

from aurasync.controller import dlqr


class TestLQRController(unittest.TestCase):

    def test_dlqr_iteration_limit(self):
        A = np.array([[0.9]])
        B = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])

        # Force the solver to exhaust its iteration budget.
        K = dlqr(A, B, Q, R, iters=1, tol=0.0)

        # Independently calculate the gain after one iteration.
        P0 = Q.copy()
        K0 = np.linalg.solve(
            R + B.T @ P0 @ B,
            B.T @ P0 @ A
        )
        P1 = Q + A.T @ P0 @ (A - B @ K0)

        expected = np.linalg.solve(
            R + B.T @ P1 @ B,
            B.T @ P1 @ A
        )

        np.testing.assert_allclose(K, expected)
        self.assertEqual(K.shape, (1, 1))
        self.assertTrue(np.isfinite(K).all())


if __name__ == "__main__":
    unittest.main()
