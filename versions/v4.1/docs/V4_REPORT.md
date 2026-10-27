# AuraSync V4: Detect → Correct → Recover — Technical report

All results come from simulation: a cart-pole with a simulated event camera. Nothing
here has been validated on hardware. See `docs/V4_HARDWARE_PLAN.md` for what that would
take.

## Abstract
V3 fell when its neural event observer became **confidently wrong**. On validation runs,
every fall came after a stretch of tilt error above 10° with low reported uncertainty,
usually with the wrong sign.

V4 adds:
- an independent event-geometry measurement, a RANSAC line fit on events less than 3 ms
  old;
- a CUSUM consistency detector;
- gated Kalman correction plus confirmed re-initialisation;
- uncertainty inflation;
- position-only fusion of the neural output;
- the V3 recovery supervisor, re-tuned on validation seeds.

On the test seeds, the proposed V4 improves strongly on every V3 baseline:
- **Nominal:** survival 96.4% vs 54.8–73.8%; safe recovery 62.5% vs 16.7–32.1%.
  The paired 95% CIs exclude zero.
- **Fault scenarios:** it fixes the sign-error and confident-error failures.
- **HDR:** 100% survival vs 0% for V3 and the frame camera.

However, **a network-free version of the same idea (Baseline D: physics + event
geometry + the same supervisor) matches or beats V4 in most conditions.**
- **Nominal:** safe recovery is 7.1 points lower for V4, CI [−13, −2]; survival is
  −2.7, CI [−7, +1].
- **Low light and dropped events:** D is directionally better.
- **Occlusion** is the one condition where V4 is clearly ahead: survival +25,
  CI [0, +50], with tilt error 2.2° vs 15.3°. With part of the pole hidden, the line fit
  loses its evidence, and the neural observer keeps tracking.
- **Compute:** at its default rate of about 450 neural calls/s, V4 needs about 1.5 CPU
  cores of compute per second of control on this machine, so it is not real-time on one
  core. D needs 0.25 cores.

The honest conclusion is that V4's gain over V3 comes from the detector and the
geometric correction. The V3 neural observer adds value only where the geometry is
blind.

## 1. Setup
- **Plant:** cart-pole, RK4 at 5 kHz, control at 1 kHz.
- **Disturbance:** a hit (angular-velocity impulse) at tick 200.
- **Episode:** 1000 ticks.
- **Survival** = pole stays below the fall angle and the cart stays on the track.
- **Safe recovery** = survival and peak tilt ≤ 12°.
- **Sensors:** ESIM-style event camera, 96×32 px, time surface plus a 30 fps APS
  channel. Frame camera and IMU/encoder for the baselines.
- **Seeds:**

  | Use | Seeds |
  |---|---|
  | Training (V3 checkpoints, unchanged) | event 0/1/2, frame 0/1/2 |
  | Development, design, tuning | validation seeds 300–315 only |
  | Test hits | 500+ |
  | Test snap | 800+ |

  The design was frozen (commit `28d23b7`) before any test run.
- **Trial counts:**
  - Nominal: 7 hit strengths (1–4 rad/s) × 16 trials = 112 per seed.
  - Stress conditions and new scenarios: 2 hit strengths × 8 trials = 16 per seed.
    Hits are 1.5 and 3 rad/s, or 4 and 5 for `fast`.
- **Statistics:**
  - Rates averaged over training seeds per trial.
  - Paired bootstrap 95% CI over trials, restricted to seeds both sides share.
  - Cohen's h for rates.
- **Compute limits** (2 CPU cores): 3 seeds for V4 ± supervisor in nominal; seed 0
  elsewhere.

## 2. Step 1–2: audit and reproduction
- **V3 reproduced exactly:** 6 configs × 6 trials, bit-identical
  (`results/v4/repro_v3.json`).
- **Audit findings** (`docs/V3_AUDIT.md`):
  - No estimator reads the true state. Only the sensor simulators do.
  - Two problems were fixed:
    - The perfect initial state was handed to estimators. The new `init_error`
      scenario tests this.
    - The frame camera imaged the nominal pole under `ood_plant`. This was fixed, and
      V3's frame configs were re-run for that condition.
- **Failure catalogue** (160 validation runs, `results/v4/dev_catalogue.json`):
  - V3 triggered: 38/80 falls.
  - V3 + shield: 64/80 falls.
  - **Every fall was preceded by confidently wrong ticks.**
  - 73–74 of 80 runs contained wrong-sign estimates.
  - Plots: `results/v4/fig_failure_cases.png`.

## 3. Step 3–5: detector and correction, open loop (validation)
160 recorded validation runs were replayed through V4 variants (`v4/eval_dev.py` →
`results/v4/dev_eval.json`).
- **How the replay works:** the trajectory is fixed, so only the estimate changes.
- **Scoring:** ground truth is used for scoring only, from tick 50 on.
- **Confident-wrong:** error above 5° while the error is more than 3σ.

**Detection** (the alarm against ticks where V3's error exceeds 5°):

| Recorded runs | Precision | Recall | Falls warned before the fall | Median lead |
|---|---|---|---|---|
| V3 triggered | 0.77 | 0.86 | 38/38 | 462 ms |
| V3 + shield | 0.94 | 0.91 | 64/64 | 462 ms |

Runs where an alarm fired: 76/80 for each.

**Correction and calibration** (V3 + shield runs; V3 triggered runs in brackets):

| Variant | Mean error | Ticks > 10° | Wrong sign | Confident-wrong | Within 1σ / 2σ |
|---|---|---|---|---|---|
| V3 (detect only) | 13.16° (7.95°) | 45.0% | 45.3% | 57.2% | 0.20 / 0.33 |
| + inflation only | 13.16° | 45.0% | 45.3% | 15.6% | 0.31 / 0.65 |
| + gated fusion | 1.56° | 1.8% | 0.9% | 5.0% | 0.60 / 0.82 |
| **final V4** | **1.24° (1.83°)** | **0.1% (1.1%)** | **0.2%** | **2.9%** | 0.61 / 0.84 |
| first design (overwrite) | 1.28° (1.43°) | 0.0% | 0.1% | 1.5% | 0.68 / 0.90 |

**What this shows:**
- **Correction:** it removes most of the confidently-wrong error. Gated fusion does
  nearly all of the work; re-initialisation fired 0–0.06 times per run.
- **Inflation:** it cuts the confident-wrong *rate* (57% → 16%) without reducing
  error at all.
- **Calibration:** still over-confident. 84% of ticks fall within 2σ, against 95% for
  a calibrated estimator.
- **The overwrite design looks best open-loop and is the worst in closed loop**
  (Section 6.2). Open-loop estimation error was not a reliable guide to control.

EventGeometry alone (validation):
- valid on 72.7% of ticks;
- median |error| 0.86°, p95 4.2°;
- |z| < 2 on 95% of ticks.

## 4. Step 6: design selection and tuning (validation only, 64 runs)

| Design | Survival | Safe | Tilt error |
|---|---|---|---|
| V3 triggered | 75.0% | 20.3% | 6.59° |
| D geometry | 100% | 39.1% | 1.59° |
| D geometry + fallback | 100% | 40.6% | 1.54° |
| V4, neural overwrite (first design) | 85.9% | 26.6% | 1.48° |
| V4, full-state Kalman | 89.1% | 21.9% | 1.52° |
| V4, position-only Kalman | 98.4% | 32.8% | 1.19° |
| **V4, position-only + supervisor (proposed)** | 98.4% | 40.6% | 1.38° |

- **Why position-only fusion:** on validation, the neural velocity outputs were
  confidently wrong after large hits (θ̇ error 117–139 °/s vs 49–89 °/s for
  geometry), so only position is fused.
- **D was already level with V4 on validation.** V4 was still taken forward as the
  proposed system because it is what the spec asks to evaluate. D was kept as the
  baseline that tests whether the network is needed.
- **Supervisor tuning** (16 validation runs, 12-point grid per family; selected setting
  at 100% survival and 75% safe):

  | Family | γ | κ | stale |
  |---|---|---|---|
  | V4 | 0.03 | 2 | 4° |
  | D | 0.05 | 2 | 8° |

## 5. Step 7: closed-loop test results
Full table: `results/v4/v4_report.json`. Figures: `results/v4/v4_nominal.png` and
`results/v4/v4_envelope.png`.

### 5.1 Nominal (112 trials per seed)

| System | Survival [95% CI] | Safe | Tilt error |
|---|---|---|---|
| F IMU + encoder (+shield) | 100 [100,100] | 57.1 (71.4) | 0.06° |
| D geometry, no supervisor | 100 [100,100] | 34.8 | 1.62° |
| D geometry + supervisor | 99.1 [97,100] | 69.6 | 1.59° |
| **V4 proposed (3 seeds)** | **96.4 [93,99]** | **62.5** | **1.22°** |
| E frame 120 fps (3 seeds) | 84.5 [79,90] | 45.8 | 1.41° |
| B V3 triggered (3 seeds) | 73.8 [68,79] | 16.7 | 4.96° |
| A V3 dense (3 seeds) | 70.2 [65,75] | 17.6 | 5.03° |
| C V3 + shield (3 seeds) | 54.8 [47,62] | 32.1 | 6.38° |

Paired differences, V4 minus the baseline (percentage points):

| Baseline | Survival [95% CI] | Cohen's h | Safe recovery [95% CI] |
|---|---|---|---|
| A | +26.2 [+21, +31] | — | +44.9 [+38, +52] |
| B | +22.6 [+18, +28] | — | +45.8 [+38, +53] |
| C | +41.7 [+34, +49] | 1.10 | +30.4 [+23, +37] |
| E 120 fps | +11.9 [+7, +17] | — | +16.7 [+10, +24] |
| D + supervisor | −2.7 [−7, +1] | — | **−7.1 [−13, −2]** |
| F (IMU + encoder) | −3.6 [−7, −1] | — | — |

The F row is the IMU baseline's +3.6 [+1, +7] advantage in survival, stated from V4's
side.

V4 also had track-limit violations in 3.6% of trials, against 0% for D.

### 5.2 Stress conditions (16 trials, V4 seed 0)
Survival (safe recovery) in %:

| Condition | V4 | D + supervisor | E 120 fps | A V3 dense |
|---|---|---|---|---|
| low light | 81 (31) | 100 (50) | 94 (52) | 79 (21) |
| HDR | **100 (62)** | 100 (94) | **0 (0)** | **0 (0)** |
| event noise | 100 (62) | 100 (88) | 94 (52) | 75 (10) |
| threshold mismatch | 100 (88) | 100 (88) | 94 (52) | 83 (10) |
| dropped events | 81 (19) | 100 (56) | 94 (52) | 75 (25) |
| occlusion | **69 (25)** | 44 (25) | 54 (23) | 27 (6) |
| OOD plant | 88 (56) | 100 (50) | 65 (31) | 38 (2) |
| snap (+80% cart mass) | 100 (69) | 100 (81) | 94 (46) | 73 (19) |
| fast (4–5 rad/s) | 50 (0) | 50 (0) | 25 (0) | 19 (0) |

- **Fast hits:** the oracle with perfect state survives only 81% and recovers safely in
  0%. These hits are at or beyond the controller's limit whatever the estimator.
- **Missing data:** Baseline C was not run under the stress conditions in V3. Those
  cells are missing, not zero.

### 5.3 New fault scenarios (16 trials, seed 0)
Survival (safe recovery) in %:

| Scenario | V4 | V4 without correction | D + supervisor | C V3 + shield | B V3 triggered | E 120 fps | F IMU |
|---|---|---|---|---|---|---|---|
| sign flip at +60 ms | **100 (75)** | 25 (19) | 94 (75) | 44 (25) | 69 (19) | 100 (50) | 100 (50) |
| confident 15° offset | **100 (69)** | 12 (6) | 100 (62) | 38 (25) | 50 (6) | 100 (50) | 100 (69) |
| 40 ms stale stream | 94 (56) | 12 (6) | 94 (75) | 50 (25) | 69 (19) | 94 (50) | 100 (56) |
| miscalibration | 100 (81) | 6 (6) | 100 (88) | 50 (25) | 88 (19) | 100 (50) | 100 (50) |
| initial-state error | 100 (81) | **0 (0)** | 100 (88) | 50 (19) | 81 (12) | 100 (50) | 100 (88) |
| combo (low light + 5× noise + OOD plant) | 31 (0) | 25 (12) | 25 (0) | 12 (0) | 6 (0) | 50 (19) | 100 (50) |

**Detection:**
- Sign flip: the alarm fired on 16/16 trials, median 4 ms after injection (max
  214 ms).
- Confident offset: 16/16 trials, median 4 ms (max 4 ms).

**V4 vs C (the V3 system V4 is meant to fix), survival differences:**

| Scenario | Survival difference [95% CI] |
|---|---|
| sign flip | +56 [+31, +81] |
| confident offset | +62 [+38, +88] |
| initial-state error | +50 [+25, +75] |
| miscalibration | +50 [+25, +75] |

**Combo scenario:**
- Geometry breaks down: D's tilt error is 68–78° because the line fit locks onto noise.
- V4's network keeps its error to 5.9°, but V4 still falls in 69% of trials.
- Only the IMU baseline survives.

### 5.4 Ablations (nominal, seed 0, paired against V4 seed 0)

| Removed | Survival [95% CI] | Safe [95% CI] | Verdict |
|---|---|---|---|
| state correction | **−90.2 [−96, −85]** | −58.9 [−68, −49] | essential; detection plus inflation alone is harmful (see §6) |
| supervisor (3 seeds) | +1.8 [0, +4] | **−36.3 [−44, −29]** | supervisor buys safe recovery, not survival |
| uncertainty feedback (inflation) | 0.0 [0, 0] | +5.4 [0, +11] | **no benefit shown**; slightly worse with it |
| triggered scheduling (dense instead) | −1.8 [−6, +3] | −3.6 [−12, +5] | no difference; triggered uses 438 vs 1000 calls/s |
| APS channel (event-only) | −2.7 [−7, +2] | 0.0 [−10, +10] | no difference in nominal. Under stress, event-only was higher in 6 of 9 cells and significantly better in safe recovery under dropped events (+31 [+12, +56]); worse only under fast hits (survival −12, n.s.) |
| position-only fusion (V3 overwrite instead) | **−71.4 [−79, −62]** | −47.3 [−57, −38] | overwrite design fails |
| budget 250 calls/s | −0.9 [−3, 0] | +2.7 [−4, +10] | no measurable cost |
| budget 100 calls/s | +0.9 [−2, +4] | 0.0 [−10, +10] | no measurable cost |

Signs are "ablated minus V4". The supervisor row uses 3 seeds; the rest use seed 0.

## 6. Failures and their causes

### 6.1 Detection plus inflation without correction makes things worse
Without correction, survival is 8% nominal and 0% under initial-state error. V3 with
its shield gets 55% and 50%.
- With nothing correcting the state, the detector stays in alarm. It raised an alarm on
  every one of 112/112 nominal trials, and inflation fired about 133 times per trial.
- The supervisor's stale fallback (a low-gain LQR) was then active on 46% of ticks, vs
  0.2% for full V4.
- The pole falls even at 1 rad/s hits: 25% survival, against 100% for V3 with its
  shield.

**Lesson:** telling the controller "you are unsure" helps only if something then
restores a usable estimate.

### 6.2 The first V4 design (neural overwrite)
It had the lowest tilt error of any event method (1.25°) and 27% survival.
- The network's velocity outputs were confidently wrong after hits.
- Overwriting the state threw away the geometry-informed velocities.
- 47% of trials ended on a track-limit violation.

This is the spec's warning case: better estimation, worse control. It is reported as a
failure.

### 6.2b Open-loop evaluation misled design choices
- **Open loop** (validation), the overwrite design had the best calibration of all
  variants (0.90 within 2σ) and the fewest error ticks.
- **Closed loop**, it was the worst V4 variant.
- **The reason:** replay holds the trajectory fixed, so it cannot show how wrong
  velocities change what the controller does next.
- **Why the final choice held up:** position-only fusion was chosen on *closed-loop*
  validation results (Section 4), not on open-loop error.

### 6.3 Uncertainty feedback (Module C) is not supported by the data
Removing inflation did not reduce survival in nominal conditions, and safe recovery
moved +5.4 [0, +11] the other way. It was not ablated under stress (compute limits),
so its effect there is unknown.

### 6.4 The neural observer is mostly redundant given geometry
- In 14 of the 15 non-nominal conditions, D with the supervisor was level with V4
  (within one trial of 16) or better.
- V4 was clearly better only under occlusion. In the combo scenario V4 tracked much
  better (5.9° vs 68° error), but control was poor for both.
- Confirmed re-initialisation fired rarely (0.08 per nominal trial). Gated fusion did
  almost all the correcting.

### 6.5 The supervisor costs survival on the OOD plant and fast hits
- OOD plant: survival −12.5 [−31, 0].
- Fast hits: −12.5 [−38, +12].
- The supervisor's barrier assumes nominal plant parameters. That is the likely cause,
  but it has not been verified.

### 6.6 Low light, dropped events and occlusion are V4's weak conditions
Survival is 69–81%. In these conditions, fresh events are scarce:
- geometry validity drops;
- the detector reports `missing_events` / `stale_evidence`;
- inflation rises (12.8 per trial in low light, vs 2.4 nominal).

### 6.7 Case studies on test trials
The figure is `results/v4/fig_test_cases.png`. Per-tick logs are in
`results/v4/test_cases/*.csv`.

**(a) Sign flip, hit 1.5 rad/s, test trial 6.**
- **V3 with shield:** the estimate drifts while ±2σ stays about ±5°, and the true tilt
  reaches −30°. This is the confidently-wrong failure, and the pole falls.
- **V4 on the same trial:** the flipped estimate is pulled back within a few ms by
  gated fusion, and the pole survives.

**(b) Occlusion, V4 fall.**
- Right after the hit (ticks 200–300), events from the pole are hidden. The estimate
  moves the wrong way while the true tilt rises to 30°.
- The detector score plateaus just under the alarm level (about 0.9), because no valid
  geometry arrives. The `missing_events` reason is logged but feeds neither the CUSUM
  nor the covariance, so σ stays small.
- By the time the pole is visible again and the correction arrives, the tilt is beyond
  recovery.

**Design flaw found:** missing evidence should raise uncertainty, but it doesn't.

**(c) Combo scenario, V4 fall.**
- At about 780 ms the estimate diverges from the truth.
- The alarm fires and σ is inflated, but no valid correction exists in low light plus
  5× noise. The controller follows the diverging estimate.
- The supervisor's stale threshold did not stop it in time.

## 7. Step 8: compute and latency (Module E)
Measured on 1 thread of an Intel Xeon @ 2.8 GHz, replaying 7,070 ticks of recorded
validation data (`results/v4/latency_v4.json`):

| Stage | p50 | p99 | max |
|---|---|---|---|
| event geometry | 0.28 ms | 0.58 ms | 2.9 ms |
| consistency detector | 0.009 ms | 0.042 ms | 0.13 ms |
| Kalman fusion | 0.075 ms | 0.17 ms | 0.89 ms |
| supervisor (CBF + fallback) | 0.39 ms | 0.76 ms | 1.9 ms |
| neural correction (CNN + SSM) | 2.07 ms | 3.42 ms | 7.3 ms |

CPU load in cores, i.e. CPU-seconds needed per second of control:

| Configuration | Neural calls/s | CPU (cores) |
|---|---|---|
| V4 default | 447 | **1.50** (0.54 always-on + 0.97 neural) |
| V4, budget 100/s | 99 | 0.58 |
| D geometry | 0 | 0.25 |

**What this means:**
- **Real-time feasibility:** the default V4 is not real-time on one core of this
  machine. Its two paths are designed to run in parallel, the neural path
  asynchronously on a second core.
- **What is not measured:** the 6 ms deadline-miss rate (0%) comes from the simulator,
  which samples latency from V3's measured distribution. It is not a live measurement.
- **Modelled latency:** sensor-to-actuator latency is modelled, not demonstrated. The
  model is 1 ms tick quantisation plus measured p99 compute:
  - event path p99 ≈ 2.6 ms;
  - neural path p99 ≈ 4.4 ms.

  It excludes sensor readout, transport, OS scheduling and actuator delay.

**A 1 kHz control rate is not a 1 ms latency.**

Memory: peak RSS 740 MB, with PyTorch loaded.

## 8. Limitations
- **Simulation only.** The event camera model, the plant and the noise are all
  simulated. See the hardware plan.
- **Sample size.** Stress cells use 16 trials and mostly one seed. Many CIs are
  ±25–40 points, and differences under about 20 points there are not established.
- **Baseline coverage.** Baseline C was not run under the V3 stress conditions.
  Uncertainty-feedback and scheduling ablations ran only in nominal conditions.
- **Supervisor.** Its model and parameters assume the nominal plant.
- **Not computed.** Two analyses the spec asks for were not done: calibration broken
  down by hit strength and condition, and the distribution of time from detection to a
  reliable corrected estimate. Only the median alarm delay for injected faults (4 ms)
  is reported.
- **No prior-art claim.** No comprehensive prior-art search was done. Nothing here
  claims V4 is novel, formally safe, or better than existing methods outside this
  benchmark.

## 9. Recommendations
1. Make the geometry+supervisor pipeline (D) the default: same or better control at a
   sixth of V4's compute.
2. Use the neural observer only as a fallback when geometry validity drops (occlusion,
   combo). Its value is in exactly the conditions where geometry is blind.
3. Remove or redesign uncertainty inflation, and never run it without correction.
   Make missing evidence (no valid geometry while the estimate predicts motion) raise
   σ and the detector score: the occlusion falls in §6.7(b) come from this gap.
4. Make the supervisor robust to plant uncertainty before claiming it helps on unseen
   plants.
5. Take the hardware plan's latency step first. Every timing number above is a model.
