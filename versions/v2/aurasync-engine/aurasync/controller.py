"""1 kHz state-feedback controller.

An infinite-horizon LQR is the unconstrained limit of linear MPC: it minimises
the same quadratic cost over an unbounded horizon and collapses to a single
gain matrix, so it costs a 4-element dot product per tick and runs comfortably
at 1 kHz even in Python. Actuator limits are enforced by saturation.
A constrained MPC can be dropped in through the same `act(state)` interface.
"""
import numpy as np
from .physics import PlantParams, linearise


def dlqr(A, B, Q, R, iters=20000, tol=1e-10):
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P_new = Q + A.T @ P @ (A - B @ K)
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


class LQRController:
    def __init__(self, p: PlantParams, dt=1e-3,
                 Q=(30.0, 2.0, 200.0, 5.0), R=0.02):
        A, B = linearise(p, dt)
        self.K = dlqr(A, B, np.diag(Q), np.array([[R]]))
        self.u_max = p.u_max

    def act(self, s):
        u = -float((self.K @ np.asarray(s))[0])
        return float(np.clip(u, -self.u_max, self.u_max))
