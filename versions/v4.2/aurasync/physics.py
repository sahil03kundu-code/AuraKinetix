"""Kinetic plant: an inverted pendulum on a force-driven cart.

This is the "unstable kinetic system" AuraSync must protect. The pole is short
(0.5 m), so it falls fast: its natural time constant sqrt(l/g) is ~160 ms, which
means a 25-40 ms perception delay eats a large share of the recovery window.

State s = [x, x_dot, theta, theta_dot]; theta = 0 is upright, positive tilts right.
"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass
class PlantParams:
    M: float = 1.0        # cart mass [kg]
    m: float = 0.1        # pole mass [kg]
    l: float = 0.25       # pole half-length [m] (full pole = 0.5 m)
    g: float = 9.81
    b_cart: float = 0.1   # cart viscous friction
    b_pole: float = 0.002 # pivot friction
    u_max: float = 20.0   # actuator force limit [N]
    x_limit: float = 1.0  # track half-length [m]

    def perturbed(self, rng, frac=0.3):
        """Domain randomisation: scale masses / length by up to +-frac."""
        s = lambda: 1.0 + rng.uniform(-frac, frac)
        return replace(self, M=self.M * s(), m=self.m * s(), l=self.l * (1 + rng.uniform(-0.1, 0.1)))


def deriv(s, u, p: PlantParams):
    x, xd, th, thd = s
    sin, cos = np.sin(th), np.cos(th)
    total = p.M + p.m
    temp = (u + p.m * p.l * thd ** 2 * sin - p.b_cart * xd) / total
    thdd = (p.g * sin - cos * temp - p.b_pole * thd / (p.m * p.l)) / (
        p.l * (4.0 / 3.0 - p.m * cos ** 2 / total))
    xdd = temp - p.m * p.l * thdd * cos / total
    return np.array([xd, xdd, thd, thdd])


def rk4_step(s, u, dt, p: PlantParams):
    k1 = deriv(s, u, p)
    k2 = deriv(s + 0.5 * dt * k1, u, p)
    k3 = deriv(s + 0.5 * dt * k2, u, p)
    k4 = deriv(s + dt * k3, u, p)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def failed(s, p: PlantParams, max_angle=0.8):
    """Kinetic failure = pole past ~46 degrees or cart off the track."""
    return abs(s[2]) > max_angle or abs(s[0]) > p.x_limit


def linearise(p: PlantParams, dt, eps=1e-6):
    """Discrete-time (A, B) about the upright equilibrium via finite differences."""
    s0 = np.zeros(4)
    A = np.zeros((4, 4))
    for i in range(4):
        d = np.zeros(4); d[i] = eps
        A[:, i] = (rk4_step(s0 + d, 0.0, dt, p) - rk4_step(s0 - d, 0.0, dt, p)) / (2 * eps)
    B = ((rk4_step(s0, eps, dt, p) - rk4_step(s0, -eps, dt, p)) / (2 * eps)).reshape(4, 1)
    return A, B
