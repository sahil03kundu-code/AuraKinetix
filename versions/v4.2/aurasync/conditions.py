"""Operating conditions for the stress benchmark (Upgrade D).

One SensorConditions object describes the scene and both sensors, so the event
camera and the frame camera always see the same world. NOMINAL reproduces the
v2 simulation exactly.

Physical models (deliberately simple, documented so a reviewer can argue with them):
  brightness     scene illumination scale. Event pixels compute log(I + dark
                 current), so dim scenes compress contrast and fire fewer
                 events; background noise rises ~1/sqrt(b) and the photoreceptor
                 slows (time constant 50 us / b). Frame cameras auto-expose up to
                 8 ms (inside the estimator's training range), then add gain,
                 so read noise grows ~sqrt(4 ms / (b * exposure)).
  hdr_gain       a bright window behind the top of the pole (y > 0.30 m) is
                 hdr_gain times brighter than the room. Frame pixels exposed for
                 the room saturate there (clip at 1.0); event pixels are
                 logarithmic and keep the contrast.
  ev_noise_mult  multiplies event background activity (0.3 Hz/pixel nominal)
  thresh_sigma   per-pixel contrast-threshold mismatch (0.03 nominal)
  drop_prob      fraction of events lost in readout (bus saturation)
  occlusion      (t0, t1) or a tuple of such windows: a grey panel covers the upper pole
  x_offset_m, scale  camera miscalibration: the sensor images world x * scale + offset at the
                 pixel the estimators believe shows world x (V4 'miscalib' scenario)
"""
from dataclasses import dataclass, replace
import numpy as np

READ_NOISE = 0.02
BASE_EXPOSURE = 0.004
MAX_EXPOSURE = 0.008


@dataclass(frozen=True)
class SensorConditions:
    name: str = "nominal"
    brightness: float = 1.0
    hdr_gain: float = 1.0
    hdr_y: float = 0.30
    ev_noise_mult: float = 1.0
    thresh_sigma: float = 0.03
    drop_prob: float = 0.0
    occlusion: tuple = None
    occ_box: tuple = (-0.30, 0.30, 0.30, 0.62)   # world x0, x1, y0, y1 [m]
    occ_value: float = 0.5
    x_offset_m: float = 0.0       # camera mounted off-centre (estimators assume the nominal calibration)
    scale: float = 1.0            # focal-length / pixel-scale error

    @property
    def is_nominal(self):
        return self == NOMINAL

    # ---- event sensor ---------------------------------------------------------
    @property
    def ev_noise_rate(self):
        return 0.3 * self.ev_noise_mult / np.sqrt(self.brightness)

    @property
    def photoreceptor_tau(self):
        return 50e-6 / self.brightness

    # ---- frame sensor (also used for the event sensor's APS channel) ----------
    def frame_exposure(self, base=BASE_EXPOSURE):
        return float(min(base / self.brightness, max(base, MAX_EXPOSURE)))

    def frame_noise(self, exposure):
        return float(READ_NOISE * np.sqrt(BASE_EXPOSURE / (self.brightness * exposure)))

    def frame_sensor(self, radiance_avg):
        """Time-averaged scene radiance (already x brightness) -> normalised pixel values.
        Exposure + gain map a room-lit scene back to the nominal range; anything
        brighter than full scale saturates. Noise is added by the caller."""
        return np.clip(radiance_avg / self.brightness, 0.0, 1.0)


NOMINAL = SensorConditions()
PRESETS = {
    "nominal": NOMINAL,
    "low_light": SensorConditions("low_light", brightness=0.05),
    "hdr": SensorConditions("hdr", hdr_gain=40.0),
    "event_noise": SensorConditions("event_noise", ev_noise_mult=10.0),
    "thresh_mismatch": SensorConditions("thresh_mismatch", thresh_sigma=0.09),
    "dropped_events": SensorConditions("dropped_events", drop_prob=0.5),
    "occlusion": SensorConditions("occlusion", occlusion=(0.21, 0.31)),
}
