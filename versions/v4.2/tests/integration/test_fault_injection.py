"""Bounded closed-loop fault-injection checks using the real V4.2 simulation.

These are integration/sanity regressions, not claims of recovery performance or
sub-millisecond deployment latency. CPU PyTorch is installed by integration CI.
"""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("torch", reason="simulation imports PyTorch; slim Docker smoke image omits it")

from aurasync.conditions import PRESETS
from aurasync.controller import LQRController
from aurasync.physics import PlantParams, rk4_step
from aurasync.safety import SafetyShield, ShieldedController
from aurasync.simulate import DT, OraclePipeline, run_episode
from aurasync.v41.pipeline import EvidencePipeline


def no_force():
    ctrl = Mock()
    ctrl.act.return_value = 0.0
    return ctrl


def evidence(seed=8):
    return EvidencePipeline(
        Q=np.diag([1e-7, 1e-5, 1e-6, 1e-4]), neural=None, seed=seed,
    )


@pytest.mark.parametrize("fault", ["dropped_events", "occlusion", "hdr", "event_noise"])
def test_faulted_camera_conditions_keep_evidence_filter_numerically_stable(fault):
    estimator = evidence()
    log = run_episode(
        PlantParams(), no_force(), estimator, n_ticks=6,
        s0=np.array([0.0, 0.0, 0.02, 0.0]),
        cond=PRESETS[fault], cam_seed=19, stop_on_fail=False,
    )
    assert log["s"].shape == (6, 4)
    assert log["s_hat"].shape == (6, 4)
    assert len(estimator.log["state_seq"]) == 6
    assert sum(estimator.log["states"]) == 6
    assert estimator.log["calls"] == 0
    assert np.isfinite(log["s_hat"]).all()
    assert np.isfinite(estimator.covariance()).all()
    assert np.linalg.eigvalsh(estimator.covariance()).min() >= -1e-8
    assert np.all(log["events"] >= 0)


def test_kick_is_applied_before_first_plant_substep():
    plant = PlantParams()
    s0 = np.zeros(4)
    kick = 0.7
    log = run_episode(
        plant, no_force(), OraclePipeline(), n_ticks=1,
        s0=s0, kicks=[(0, kick)], cam_seed=11, stop_on_fail=False,
        aps_fps=0,
    )
    expected = s0.copy()
    expected[3] += kick
    for _ in range(5):
        expected = rk4_step(expected, 0.0, DT / 5, plant)
    np.testing.assert_allclose(log["s"][0], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(log["s"], log["s_hat"])


def test_snap_uses_replaced_physics_parameters_from_trigger_tick():
    initial_plant = PlantParams()
    new_plant = replace(initial_plant, l=0.40)
    s0 = np.array([0.0, 0.0, 0.12, 0.0])
    log = run_episode(
        initial_plant, no_force(), OraclePipeline(), n_ticks=2,
        s0=s0, snap=(1, new_plant), cam_seed=11,
        stop_on_fail=False, aps_fps=0,
    )
    expected = s0.copy()
    for plant in (initial_plant, new_plant):
        for _ in range(5):
            expected = rk4_step(expected, 0.0, DT / 5, plant)
    np.testing.assert_allclose(log["s"][-1], expected, rtol=1e-12, atol=1e-12)


def test_stop_on_fail_truncates_episode_at_first_failure():
    log = run_episode(
        PlantParams(), no_force(), OraclePipeline(), n_ticks=6,
        s0=np.array([0.0, 0.0, 0.9, 0.0]),
        stop_on_fail=True, cam_seed=3, aps_fps=0,
    )
    assert log["fail_tick"] == 0
    assert log["s"].shape == (1, 4)
    assert log["u"].shape == (1,)


def test_continue_after_fail_records_full_requested_horizon():
    log = run_episode(
        PlantParams(), no_force(), OraclePipeline(), n_ticks=4,
        s0=np.array([0.0, 0.0, 0.9, 0.0]),
        stop_on_fail=False, cam_seed=3, aps_fps=0,
    )
    assert log["fail_tick"] is not None
    assert log["s"].shape == (4, 4)
    assert log["events"].shape == (4,)


def test_shielded_lqr_controller_runs_inside_real_episode():
    plant = PlantParams()
    estimator = evidence()
    nominal_controller = LQRController(plant)
    shield = SafetyShield(plant=plant)
    controller = ShieldedController(nominal_controller, shield, estimator)
    result = run_episode(
        plant, controller, estimator, n_ticks=4,
        s0=np.array([0.0, 0.0, 0.01, 0.0]),
        cam_seed=4, stop_on_fail=False,
    )
    assert result["u"].shape == (4,)
    assert np.isfinite(result["u"]).all()
    assert np.all(np.abs(result["u"]) <= plant.u_max + 1e-12)
    assert controller.stats["ticks"] == 4
    assert sum(
        controller.stats[mode]
        for mode in ("pass_", "modified", "soft_dropped", "infeasible")
    ) == 4
    assert len(controller.times) == 4
