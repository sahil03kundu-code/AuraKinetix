# AuraSync V4 — Hardware-validation plan

None of the V4 results come from hardware. This plan sets out what would have to happen
before any V4 claim could be called real-world. The steps are in order, and each one has
a pass/fail gate.

## 0. Rig
- **Plant:** a cart-pole or rotary inverted pendulum. A Quanser-style unit or a custom
  belt-drive with a 0.25–0.30 m pole is enough.
- **Ground truth:** a high-resolution encoder on the pole joint (≥ 4096 CPR) and one on
  the cart. Used for evaluation only, never fed to the estimator. This mirrors the
  simulator's rule against using ground truth.
- **Event camera:** for example a Prophesee EVK4 / IMX636 or an iniVation DAVIS346. The
  DAVIS also gives the APS frames that the event+APS model expects.
- **Compute:** first the same class of CPU as the simulation measurements, then the
  intended embedded target.
- **Timing:**
  - One hardware clock shared by the camera trigger input, the encoder capture and the
    motor PWM.
  - An LED or IR flash in view, wired to the same clock, so photon-to-event delay can
    be measured.
- **Safety:** a mechanical end-stop, a current-limited drive, a hardware e-stop, and a
  pole catch for falls.

## 1. Latency is measured, not assumed
1. Photon → event timestamp, using the flash: distribution of p50, p99 and max.
2. Event → estimate available: software timestamps around the V4 pipeline.
3. Estimate → command → PWM edge: a GPIO toggle measured on a scope.
4. Report the end-to-end sensor-to-actuator distribution.

**Gate:** p99 end-to-end below the 6 ms deadline used in simulation. If it is above,
re-run the simulation with the measured latency distribution before going further.

## 2. Sensor model check (open loop, motor off)
1. Swing the pole by hand and with a pendulum release, under nominal light, low light
   (about 5 lux) and an HDR scene.
2. Compare EventGeometry θ against the encoder.

**Gate:**
- median error ≤ 1.5°;
- validity fraction within 15 points of the simulation figure;
- normalised-error coverage (|z| < 2) ≥ 0.85.

If it fails, record which simulator assumption broke: contrast thresholds, refractory
period, noise, or lens distortion.

## 3. Learned model transfer
1. Run the V3/V4 neural observer open loop on the recorded data.
2. Expect a sim-to-real gap. Fine-tune only on recordings from separate sessions, and
   keep a held-out recording day for testing.
3. Re-fit the temperature calibration on real validation data.

**Gate:** the neural observer is no worse than geometry alone; otherwise run V4 as
Baseline D (no network).

## 4. Detector and correction, open loop
1. Replay the recordings through V4 with the controller disconnected.
2. Inject the V4 scenarios in software: sign flip, confident offset, stream stall.
3. Add the physical scenarios: lens cap / occlusion, lamp flicker, and a 1–2 mm camera
   shift for miscalibration.

**Gate:**
- alarm precision and recall within 0.1 of simulation;
- no correction applied while the camera is occluded.

## 5. Closed loop, staged
1. **Small hits** from a solenoid tapper with repeatable impulse, calibrated with a load
   cell or the encoder. Start with the safety supervisor on and track limits at 50% of
   the mechanical range.
2. Raise hit strength in steps of 0.5 rad/s. Stop a step after 2 consecutive falls.
3. Run the same trial schedule for the encoder-LQR reference, Baseline D and V4.
   Randomise and interleave the order across configurations to average out drift.
4. Use at least 20 trials per hit level and configuration. Use paired statistics where
   the same tapper sequence can be replayed.

**Gate for any "V4 helps" claim:** the paired survival difference CI excludes 0 on
hardware. Simulation results are not counted here.

## 6. Stress on hardware
1. Low light, flicker, added event noise (vibration), a heavier or longer pole (the
   OOD plant), and camera miscalibration.
2. Report every failure with its log: events, estimate, alarm, action, command.

## 7. What would falsify V4
Any one of these would count against V4:
- the geometry/detector path adds more latency than it removes;
- corrections from real events are wrong often enough to cause falls that geometry-free
  V3 does not have;
- the benefit vanishes once the measured latency distribution is applied.

Each of these would be reported, not tuned away on test data.
