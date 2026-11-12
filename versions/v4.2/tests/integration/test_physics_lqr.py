
"""Integration tests for the physics engine and LQR controller."""

import numpy as np

from aurasync.physics import PlantParams, rk4_step
from aurasync.controller import LQRController


def test_lqr_physics_closed_loop():
    plant = PlantParams()
    controller = LQRController(plant)

    dt = 0.001
    initial_angle = np.radians(5.0)
    state = np.array([0.0, 0.0, initial_angle, 0.0])

    max_force = plant.u_max

    for _ in range(1000):
        force = controller.act(state)

        assert np.isfinite(force)
        assert abs(force) <= max_force

        state = rk4_step(state, force, dt, plant)

        assert np.all(np.isfinite(state))

    final_angle = abs(state[2])

    assert final_angle < abs(initial_angle), (
        f"LQR did not reduce pole angle: "
        f"initial={np.degrees(initial_angle):.2f}, "
        f"final={np.degrees(final_angle):.2f}"
    )
