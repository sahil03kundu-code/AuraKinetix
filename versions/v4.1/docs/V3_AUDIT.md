# AuraSync-Engine V3: architecture audit (V4 Step 1)

Audited at git tag `v3` (commit 523a579). Everything below was checked by reading the code and running it,
not taken from the README.

## 1. Architecture and data flow

```
                 true plant (RK4, 5 kHz)  <-- hits, mechanical snap, unseen parameters
                        |  true state s(t), true params p
          +-------------+--------------------------------------------+
          | SceneRenderer (96x32 px, true pole length, conditions)    |
          +------+----------------------------+---------------------+
                 | radiance @ 5 kHz           | radiance
       EventCamera (log-intensity            LearnedFramePipeline's camera
       thresholds, noise, drops)              (exposure blur, saturation, noise)
                 |                            |
       TimeSurfaceEncoder (ON/OFF/slow)       FrameEstimator (CNN + keypoints)
       + 30 fps APS frame (same sensor model) |  + physics roll-forward
                 |  surface (4x32x96) @ 1 kHz |
       ProbEventObserver (physics step + CNN/keypoint/SSM correction, variance head)
       inside TriggeredEventPipeline (dense | greedy | periodic | triggered,
       background corrections with measured latency, deadline 6 ticks)
                 |  estimate s_hat, covariance Sigma  (+ optional EventConsistencyMonitor)
       ShieldedController: delay compensation -> stale check -> LQR -> CBF shield
                 |  force u (|u| <= 20 N), 1 kHz
                 +--> true plant
```

| Component | File | State / interface |
|---|---|---|
| Plant | `aurasync/physics.py` | `s = [x, x_dot, theta, theta_dot]`, `PlantParams`, `rk4_step`, `linearise` |
| Scene + event camera + time surface | `aurasync/event_camera.py` | `SceneRenderer.render(s, l, t)`, `EventCamera.step`, `TimeSurfaceEncoder.tensor` |
| Conditions | `aurasync/conditions.py` | `SensorConditions`, `PRESETS` |
| Neural observer | `aurasync/estimator.py` | `ProbEventObserver.encode / physics_step / correct(z, act, h, s_pred, gap)`; hidden SSM state `h` (64-d) is separate from the physical state |
| Scheduler | `aurasync/scheduler.py` | pipeline interface `reset(s0) / update(k, t, s_true, surface, u) / estimate() / covariance() / age_ticks()` |
| Frame baseline | `aurasync/frame_camera.py` | `LearnedFramePipeline` (same interface) |
| IMU baseline | `aurasync/imu.py` | `IMUPipeline` (EKF, same interface) |
| Monitor | `aurasync/monitor.py` | `EventConsistencyMonitor.update(s_hat, surface)` (diagnostic only in V3) |
| Controller + shield | `aurasync/controller.py`, `aurasync/safety.py` | `LQRController.act(s)`, `SafetyShield.filter`, `ShieldedController.act(est)` |
| Closed loop | `aurasync/simulate.py` | `run_episode(plant, controller, pipeline, ...)` |

## 2. Ground-truth leakage check

Every pipeline's `update()` receives `s_true`. Who reads it:

| Pipeline | Reads `s_true`? | Verdict |
|---|---|---|
| `TriggeredEventPipeline`, `EventPipeline` (event estimators) | No (signature only) | **Clean** |
| `EventConsistencyMonitor` | No | **Clean** |
| `LearnedFramePipeline` | Yes, to render the camera image | Sensor simulation living inside the pipeline object; the estimator network only sees pixels. Clean, but architecturally fragile |
| `IMUPipeline` | Yes (+ true plant), to synthesise sensor readings | Same pattern; the EKF only sees noisy readings. Clean |
| `OraclePipeline`, `FramePipeline` | Yes, by design | Labelled reference baselines, not estimators |

Leaks and issues found:

1. **All estimators start from the true initial state** (`pipeline.reset(s_true_0)`). Every method benefits equally and the hit
   arrives 200 ms later, so rankings are unlikely to change, but it is unrealistic and was undocumented. V4 adds an
   `init_error` scenario to measure it.
2. **Bug: the frame camera imaged the nominal 0.25 m pole even when the true pole was 0.29 m** (`ood_plant` condition, V2/V3),
   while the event camera imaged the true pole. This handed the frame baseline an in-distribution image in exactly the condition
   meant to be out of distribution. **Fixed in V4** (the camera now images the true plant); nominal results are bit-identical
   (checked on 5 trials) and the frame `ood_plant` runs are redone.
3. Calibration, shield and trigger tuning used validation data only (`data/val_*`, closed-loop seeds 300–315). Clean.
4. Benchmark metrics (`est_th_deg`, survival) use ground truth for scoring only. Clean.

## 3. Documentation vs code

| README (V3) says | Code / data actually shows |
|---|---|
| Event-triggered scheduling | Triggers fired whenever the worker was idle; identical to a fixed 2 ms schedule (already stated in README) |
| Unseen-parameter result "frame wins −19 pts" | Biased by leak 2 above; re-measured in V4 |
| Perfect initial state | Not mentioned (leak 1) |
| Monitor baseline m0 = 0.25 | It is the monitor's initial value surviving quiet validation stretches (stated) |
| Shield "uncertainty-aware" | Validation tuning chose kappa = 0 for event and frame, i.e. the margin is off in the reported runs |

## 4. Profile (this machine, 2 cores, 8 GB)

| Item | Value |
|---|---|
| Peak RSS per simulation process | 490–520 MB (PyTorch ~490 MB; models +20–30 MB) |
| Simulation CPU per simulated second | oracle 1.6 s, IMU 2.4 s, frame 120 fps 2.7 s, event triggered 3.2 s, event dense 4.8 s (dominated by the 5 kHz camera simulation) |
| Deployed latency (from `results/v3/latency.json`) | neural correction p50 1.64 / p99 3.20 / max 7.83 ms; frame CNN p50 5.33 ms; always-on tick p50 0.16 / p99 0.40 ms |

## 5. Preserved V3 assets

`git tag v3`; checkpoints `checkpoints/v3/*.pt`; raw results `results/v3/raw/`; calibration and tuning JSONs. V4 adds new modules
and wraps the existing interfaces rather than changing them.
