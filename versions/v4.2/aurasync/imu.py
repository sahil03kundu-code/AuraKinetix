"""Proprioceptive baseline (Upgrade D): IMU on the pole + cart wheel encoder + EKF.

This is what a controls engineer would build first for a cart-pole, so any
camera-based system has to be judged against it. No camera at all, so lighting,
HDR, occlusion and event-sensor faults do not touch it; model mismatch (unseen
plant parameters, mechanical snap) does, through the EKF's physics model.

Sensors at 1 kHz, mounted at the pivot:
  gyro         theta_dot + bias + noise      (sigma 0.005 rad/s, bias ~ N(0, 0.005 rad/s))
  accelerometer specific force in the pole frame (sigma 0.05 m/s^2 per axis):
                 f_along = xdd*sin(th) + g*cos(th),  f_perp = xdd*cos(th) - g*sin(th)
  cart encoder  x quantised to 0.1 mm
EKF state: [x, x_dot, theta, theta_dot, gyro_bias, cart_accel]; prediction through
the same nominal RK4 physics as every other pipeline. Cart acceleration is a
free random-walk state rather than a model prediction, so the two accelerometer
axes pin down tilt AND acceleration without trusting the plant model
(robust to unseen masses); Jacobians by finite differences.
The filter also carries a covariance, so it doubles as a filter-based
uncertainty baseline.
"""
from collections import deque
import numpy as np

from .physics import PlantParams, rk4_step, deriv

DT = 1e-3
G = 9.81


class IMUPipeline:
    name = "imu-ekf"
    needs_true_plant = True

    def __init__(self, delay_ticks=1, plant=None, gyro_sigma=0.005, acc_sigma=0.05,
                 enc_res=1e-4, bias_sigma=0.005):
        self.p = plant or PlantParams()
        self.delay_ticks = delay_ticks
        self.gs, self.as_, self.res, self.bs = gyro_sigma, acc_sigma, enc_res, bias_sigma
        self.Q = np.diag([1e-9, 1e-5, 1e-9, 1e-4, 1e-12, 25.0])
        self.R = np.diag([enc_res ** 2 / 12 + 1e-12, gyro_sigma ** 2, acc_sigma ** 2, acc_sigma ** 2])

    def reset(self, s0):
        seed = int(abs(np.asarray(s0)).sum() * 1e7) % (2 ** 32)
        self.rng = np.random.default_rng(seed)
        self.bias = self.rng.normal(0, self.bs)
        self.x = np.concatenate([np.asarray(s0, float), [0.0, 0.0]])
        self.P = np.diag([1e-6, 1e-4, 1e-6, 1e-4, self.bs ** 2, 10.0])
        self.buf = deque([np.asarray(s0, float)] * (self.delay_ticks + 1), maxlen=self.delay_ticks + 1)
        self.cov_buf = deque([self.P[:4, :4].copy()] * (self.delay_ticks + 1), maxlen=self.delay_ticks + 1)

    def _f(self, x, u):
        return np.concatenate([rk4_step(x[:4], u, DT, self.p), x[4:]])

    def _h(self, x, u, p=None):
        """Sensor model. With p given (simulation of the TRUE sensor) the cart
        acceleration comes from the true plant; inside the EKF it is the state x[5]."""
        xdd = deriv(x[:4], u, p)[1] if p is not None else x[5]
        th = x[2]
        return np.array([x[0], x[3] + x[4],
                         xdd * np.sin(th) + G * np.cos(th),
                         xdd * np.cos(th) - G * np.sin(th)])

    @staticmethod
    def _jac(f, x, eps=1e-6):
        f0 = f(x)
        J = np.zeros((len(f0), len(x)))
        for i in range(len(x)):
            d = np.zeros(len(x)); d[i] = eps
            J[:, i] = (f(x + d) - f(x - d)) / (2 * eps)
        return J

    def measure(self, s_true, u, p_true):
        """Simulated sensor readings from the TRUE state and TRUE plant."""
        z = self._h(np.concatenate([s_true, [self.bias, 0.0]]), u, p_true)
        z[0] = np.round(s_true[0] / self.res) * self.res
        z[1] += self.rng.normal(0, self.gs)
        z[2:] += self.rng.normal(0, self.as_, 2)
        return z

    def update(self, k, t, s_true, surface, u):
        z = self.measure(s_true, u, getattr(self, "p_true", None) or self.p)
        F = self._jac(lambda x: self._f(x, u), self.x)
        self.x = self._f(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        H = self._jac(lambda x: self._h(x, u), self.x)
        y = z - self._h(self.x, u)
        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T     # Joseph form
        self.buf.append(self.x[:4].copy()); self.cov_buf.append(self.P[:4, :4].copy())

    def estimate(self):
        return self.buf[0]

    def covariance(self):
        return self.cov_buf[0]
