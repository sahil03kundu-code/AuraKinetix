"""Seeded simulation reproducibility across supported sensor faults.

Checks exact repeatability on the same runner; does not establish accuracy,
hardware latency, or cross-platform bitwise reproducibility.
"""
import numpy as np
import pytest

pytest.importorskip("torch", reason="V4.2 simulation requires CPU PyTorch")
from aurasync.conditions import PRESETS
from aurasync.physics import PlantParams
from aurasync.simulate import run_episode
from aurasync.v41.pipeline import EvidencePipeline


class ZeroController:
    def act(self, state):
        return 0.0


def episode(condition):
    pipeline = EvidencePipeline(
        Q=np.diag([1e-7, 1e-5, 1e-6, 1e-4]), neural=None, seed=23,
    )
    log = run_episode(
        PlantParams(), ZeroController(), pipeline, n_ticks=5,
        s0=np.array([0.0, 0.0, 0.02, 0.0]),
        cond=PRESETS[condition], cam_seed=37, aps_seed=43,
        kicks=[(2, 0.2)], stop_on_fail=False, record=True,
    )
    return log, list(pipeline.log["state_seq"])


@pytest.mark.parametrize(
    "condition",
    ["nominal", "low_light", "hdr", "event_noise",
     "thresh_mismatch", "dropped_events", "occlusion"],
)
def test_seeded_faulted_episode_reproduces_exact_outputs(condition):
    first, states_first = episode(condition)
    second, states_second = episode(condition)
    assert first["fail_tick"] == second["fail_tick"]
    assert states_first == states_second
    for key in ("s", "s_hat", "u", "events", "surf"):
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)
    assert first["s"].shape == (5, 4)
    assert first["surf"].shape[0] == 5
    assert np.isfinite(first["s_hat"]).all()
