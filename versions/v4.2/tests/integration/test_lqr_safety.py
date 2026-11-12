
"""Integration tests for LQR, SafetyShield, and plant physics."""

import numpy as np

from aurasync.physics import PlantParams, rk4_step
from aurasync.controller import LQRController
from aurasync.safety import SafetyShield


def test_lqr_safety_physics_closed_loop():
    plant = PlantParams()
    controller = LQRController(plant)
    shield = SafetyShield(plant)

    state = np.array([
        0.0, 0.0, np.radians(5.0), 0.0
    ])
    covariance = np.zeros((4, 4))

    initial_angle = abs(state[2])
    modes = set()

    for _ in range(1000):
        nominal_force = controller.act(state)

        safe_force, mode = shield.filter(
            state, covariance, nominal_force
        )

        modes.add(mode)

        assert np.isfinite(safe_force)
        assert abs(safe_force) <= plant.u_max

        state = rk4_step(
            state, safe_force, 0.001, plant
        )

        assert np.all(np.isfinite(state))

    assert abs(state[2]) < initial_angle
    assert "pass" in modes


def test_safety_shield_intervenes_on_excessive_command():
    plant = PlantParams()
    controller = LQRController(plant)
    shield = SafetyShield(plant)

    state = np.zeros(4)
    covariance = np.zeros((4, 4))

    nominal_force = controller.act(state) + 100.0

    safe_force, mode = shield.filter(
        state, covariance, nominal_force
    )

    assert mode == "modified"
    assert abs(safe_force) <= plant.u_max
    assert safe_force < nominal_force

    next_state = rk4_step(
        state, safe_force, 0.001, plant
    )

    assert np.all(np.isfinite(next_state))
