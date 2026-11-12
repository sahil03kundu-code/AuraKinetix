"""Checks for operating condition transformations and sensor consistency."""

from dataclasses import FrozenInstanceError
from dataclasses import replace

import numpy as np
import pytest

from aurasync.conditions import (
    BASE_EXPOSURE, MAX_EXPOSURE, NOMINAL, PRESETS, READ_NOISE, SensorConditions,
)


def test_default_conditions_are_nominal_and_immutable():
    assert NOMINAL.is_nominal
    assert not PRESETS["low_light"].is_nominal
    with pytest.raises(FrozenInstanceError):
        NOMINAL.brightness = 0.5


def test_event_noise_and_photoreceptor_follow_brightness_model():
    nominal = SensorConditions()
    low = replace(nominal, name="dim", brightness=0.25, ev_noise_mult=2.0)
    np.testing.assert_allclose(nominal.ev_noise_rate, 0.3)
    np.testing.assert_allclose(nominal.photoreceptor_tau, 50e-6)
    np.testing.assert_allclose(low.ev_noise_rate, 0.3 * 2 / 0.5)
    np.testing.assert_allclose(low.photoreceptor_tau, 200e-6)


def test_exposure_scales_with_light_and_caps_at_maximum():
    assert NOMINAL.frame_exposure() == BASE_EXPOSURE
    dim = SensorConditions(name="dim", brightness=0.5)
    assert dim.frame_exposure() == MAX_EXPOSURE
    dark = SensorConditions(name="dark", brightness=0.05)
    assert dark.frame_exposure() == MAX_EXPOSURE


def test_frame_noise_scales_with_exposure_and_light():
    assert NOMINAL.frame_noise(BASE_EXPOSURE) == READ_NOISE
    dim = SensorConditions(name="dim", brightness=0.5)
    expected = READ_NOISE * np.sqrt(BASE_EXPOSURE / (0.5 * MAX_EXPOSURE))
    np.testing.assert_allclose(dim.frame_noise(MAX_EXPOSURE), expected)


def test_frame_sensor_normalizes_brightness_and_clips_saturation():
    cond = SensorConditions(name="dim", brightness=0.5)
    radiance = np.array([-0.1, 0.0, 0.125, 0.5, 1.0])
    np.testing.assert_allclose(
        cond.frame_sensor(radiance), [0.0, 0.0, 0.25, 1.0, 1.0]
    )


@pytest.mark.parametrize(
    "key",
    ["nominal", "low_light", "hdr", "event_noise", "thresh_mismatch", "dropped_events", "occlusion"],
)
def test_builtin_preset_definitions(key):
    cond = PRESETS[key]
    assert isinstance(cond, SensorConditions)
    assert cond.name == key
    assert cond.brightness > 0
    assert cond.hdr_gain > 0
