# AuraSync V4.1: does selective neural assistance beat a geometry-first system?

All results are from simulation. Nothing was tested on hardware, no neural network was
retrained, and no new architecture was added.

## Answer
**No, not measurably.** Over the 272 paired held-out stress trials (all non-nominal
conditions), V4.1 (geometry first, with neural help on demand) compared with the same
pipeline without the network (D-ev):
- survival: **−1.8 points**, 95% CI [−7, +3];
- safe recovery: −0.4, CI [−6, +5].

On the unfamiliar plant, V4.1 was significantly *worse* (−14.6, CI [−27, −4]).

**What did help was the evidence fix (Step 3), with no network at all.** D-ev compared
with Baseline D, pooled over stress conditions:
- survival **+7.7**, CI [+2, +13];
- safe recovery **+9.2**, CI [+4, +14].

Occlusion improved the most: survival **+39**, CI [+23, +55], with tilt error falling from
15.0° to 1.6°.

V4.1 does beat the existing V4: pooled survival +7.7 [+2, +14] and safe recovery +10.7
[+5, +17], at 33–67 neural calls/s instead of about 440. But that gain comes from the
evidence fix, not from the network.

**Recommendation:** make D-ev the default estimator. Do not add neural assistance until
it can be shown to add information the event sensor itself is missing (Section 6).

## 1. Protocol
- **Step 1, audit:** `docs/V41_AUDIT.md` covers geometry validity, missing-evidence
  handling and supervisor behaviour.
- **Step 2, reproduction:** Baseline D and V4 were re-run on 60 stored test trials
  (nominal + occlusion). **All bit-identical** (`results/v41/repro_v4.json`).
- **Design and tuning:** validation seeds 300–331 only.
  - geometry calibration: `results/v41/geometry_calibration.json`;
  - supervisor grid (same grid and objective as V3/V4): `results/v41/tuning_v41.json`;
  - design rounds: `results/v41/val/`.
  - The design was frozen in commit `3be1670` **before** the test run.
- **Final test:** **fresh seeds 4000+**, never used by V1–V4 or by the V4.1 design.
  - Nominal: 7 hits × 12 = 84 trials.
  - Occlusion and combo: 2 hits × 32 = 64.
  - Dropped events, low light, OOD plant: 2 hits × 24 = 48.
  - All five systems saw identical trials, so statistics are paired bootstrap 95% CIs.
- **Metrics:**

  | Metric | Definition |
  |---|---|
  | Survival | pole stays up and cart stays on the track |
  | Safe recovery | survival with peak ≤ 12° |
  | Angle / velocity error | mean \|θ̂ − θ\| and mean \|θ̂̇ − θ̇\| |
  | Track violation | the cart reached the track end |
  | False intervention | % of ticks where the supervisor modified the command or used its stale fallback, although the same supervisor given the true state (no uncertainty) would have passed it |
  | Neural calls/s | — |
  | CPU | measured on an idle core (Section 5) |

## 2. What V4.1 changed (Steps 3–4)
These responses come from the audit. The code is `aurasync/v41/pipeline.py`, with tests
in `tests/test_v41.py` (9 passing).

| Evidence state (per tick) | Detected by | Response |
|---|---|---|
| **quiet** | sensor alive, < 4 fresh events in the pole region, estimate predicts no motion | small process noise; σθ capped at 3°, σθ̇ at 0.5 rad/s |
| **unobserved** | activity in the pole region but no usable fit, *or* motion predicted but no fit | full process noise: absence of evidence is not treated as stillness |
| **fused** | valid fit inside the gate (d² ≤ 9) | Kalman update. Measurement σ inflated ×1.4, or ×1.6 for low-support fits (validation calibration showed σ was over-confident) |
| **contradictory** | valid fit outside the gate | re-initialise after 4 consistent contradicting fits in 12 ms; meanwhile bounded covariance matching, so the estimate can never be locked out (D's gate was permanent) |
| **corrupt** | event burst > 4× baseline, or a fit inconsistent with the fits of the last 5 ms by > 15° | reject; widen σθ |

Neural assistance (V4.1 only):
- **When it is requested:** geometry has been unobserved for 3 or more ticks, a
  contradiction is unresolved, evidence is corrupt, or σθ > 3°. There is also a check-in
  at least every 50 ms.
- **Budget:** at most one call per 10 ms (≤ 100/s), with the 6 ms deadline.
- **How the answer is accepted:** through a gate (d² ≤ 9), for **positions only**
  (x, θ). Velocities stay with the physics + geometry filter, as the unit tests check.

## 3. Held-out results

![results](../results/v41/v41_results.png)

| Condition | System | Survival [95% CI] | Safe | θ err | θ̇ err (°/s) | Track viol. | False interv. | Neural calls/s |
|---|---|---|---|---|---|---|---|---|
| **nominal** (84) | D | 98.8 [96, 100] | 65.5 | 1.51° | 84 | 1.2% | 38.5% | 0 |
| | D-ev | **100 [100, 100]** | 67.9 | 1.44° | 73 | 0% | 15.0% | 0 |
| | V4 | 95.2 [90, 99] | 63.1 | 1.30° | 65 | 4.8% | 19.8% | 442 |
| | V4 @100/s | 97.6 [94, 100] | 64.3 | 1.33° | 72 | 2.4% | 26.0% | 99 |
| | **V4.1** | **100 [100, 100]** | **70.2** | 1.35° | 70 | 0% | 14.9% | 33 |
| **occlusion** (64) | D | 45.3 [33, 58] | 12.5 | 14.96° | 220 | 7.8% | 34.5% | 0 |
| | **D-ev** | **84.4 [75, 92]** | **21.9** | **1.64°** | 61 | 7.8% | 4.4% | 0 |
| | V4 | 71.9 [61, 83] | 20.3 | 2.43° | 72 | 15.6% | 15.7% | 445 |
| | V4 @100/s | 60.9 [48, 73] | 18.8 | 3.09° | 85 | 15.6% | 22.6% | 99 |
| | V4.1 | 76.6 [66, 86] | 20.3 | 1.72° | 60 | 15.6% | 4.7% | 36 |
| **dropped events** (48) | D | 100 | 33.3 | 2.31° | 91 | 0% | 38.9% | 0 |
| | **D-ev** | **100** | **45.8** | 2.09° | 76 | 0% | 15.5% | 0 |
| | V4 | 87.5 [77, 96] | 33.3 | 1.95° | 67 | 8.3% | 21.5% | 440 |
| | V4 @100/s | 100 | 25.0 | 1.96° | 80 | 0% | 27.8% | 99 |
| | V4.1 | 100 | 43.8 | 1.93° | 69 | 0% | 17.6% | 48 |
| **low light** (48) | D | 97.9 [94, 100] | 41.7 | 2.57° | 95 | 0% | 40.6% | 0 |
| | **D-ev** | 97.9 [94, 100] | **62.5** | 2.43° | 76 | 0% | 14.6% | 0 |
| | V4 | 85.4 [75, 94] | 22.9 | 2.28° | 66 | 6.2% | 22.1% | 444 |
| | V4 @100/s | 97.9 [94, 100] | 37.5 | 1.99° | 78 | 2.1% | 28.6% | 99 |
| | V4.1 | **100 [100, 100]** | 58.3 | 1.86° | 68 | 0% | 15.9% | 56 |
| **combo** (64) | D | 34.4 [23, 45] | 3.1 | 44.6° | 374 | 25.0% | 33.6% | 0 |
| | D-ev | 31.2 [20, 42] | 7.8 | 35.4° | 303 | 21.9% | 39.2% | 0 |
| | V4 | 25.0 [14, 36] | 1.6 | 19.0° | 197 | 34.4% | 21.0% | 449 |
| | V4 @100/s | 39.1 [27, 50] | 1.6 | 14.0° | 161 | 23.4% | 31.7% | 99 |
| | V4.1 | **40.6 [30, 53]** | **12.5** | 18.5° | 204 | 15.6% | 29.4% | 67 |
| **OOD plant** (48) | **D** | **93.8 [85, 100]** | 50.0 | 1.26° | 65 | 6.2% | 36.8% | 0 |
| | D-ev | 89.6 [81, 98] | 50.0 | 1.48° | 57 | 8.3% | 12.4% | 0 |
| | V4 | 85.4 [75, 96] | 50.0 | 1.04° | 52 | 14.6% | 19.1% | 444 |
| | V4 @100/s | 87.5 [77, 96] | 41.7 | 1.07° | 57 | 12.5% | 25.2% | 99 |
| | V4.1 | 75.0 [62, 88] | 50.0 | 1.06° | 52 | 25.0% | 11.4% | 33 |

### Paired differences in survival (safe recovery), percentage points [95% CI]

| Condition | V4.1 − D-ev (does the network help?) | D-ev − D (the evidence fix) | V4.1 − V4 |
|---|---|---|---|
| nominal | 0 [0, 0] (+2.4 [−2, +7]) | +1.2 [0, +4] (+2.4 [−5, +10]) | **+4.8 [+1, +10]** (+7.1 [−2, +17]) |
| occlusion | −7.8 [−20, +3] (−1.6) | **+39.1 [+23, +55]** (+9.4 [−2, +22]) | +4.7 [−9, +19] (0.0) |
| dropped events | 0 [0, 0] (−2.1 [−17, +12]) | 0 (+12.5 [0, +27]) | **+12.5 [+4, +23]** (+10.4) |
| low light | +2.1 [0, +6] (−4.2 [−19, +10]) | 0 (**+20.8 [+6, +35]**) | **+14.6 [+6, +25]** (**+35.4 [+19, +52]**) |
| combo | +9.4 [−5, +23] (+4.7 [−5, +14]) | −3.1 [−16, +9] (+4.7) | **+15.6 [+2, +30]** (**+10.9 [+2, +20]**) |
| OOD plant | **−14.6 [−27, −4]** (0) | −4.2 [−12, +4] (0) | −10.4 [−25, +4] (0) |
| **pooled stress (272)** | **−1.8 [−7, +3]** (−0.4 [−6, +5]) | **+7.7 [+2, +13]** (**+9.2 [+4, +14]**) | **+7.7 [+2, +14]** (**+10.7 [+5, +17]**) |

## 4. Interpretation
1. **The neural assistant did not add measurable value.**
   - Pooled over stress conditions, survival and safe recovery are level with the
     network-free D-ev.
   - It lowered angle error in some conditions (low light 1.86° vs 2.43°, combo 18.5° vs
     35.4°), but control did not improve accordingly. Following the spec's rule, better
     estimation without better recovery is **not** a control improvement.
   - The single significant difference is negative: the unfamiliar plant (Section 6c).
2. **The evidence fix is the real improvement.**
   - Bounded quiet-time uncertainty, calibrated measurement noise, and no permanent
     lock-out took occlusion survival from 45% to 84%.
   - False interventions fell from 35–41% of ticks to 4–16% (all conditions except combo,
     where D-ev stays at 39%). Most of D's were the stale
     fallback firing during quiet periods, when σθ had grown to about 100°.
3. **Fewer neural calls did not hurt.**
   - V4 at 100 calls/s matched or beat V4 at about 440 calls/s on survival everywhere
     except occlusion (−11 points).
   - V4.1 at 33–67 calls/s beat both.

## 5. Compute: measured on one idle core (`results/v41/latency_v41.json`)
Each figure is estimator plus supervisor per 1 ms tick, replaying 7,070 ticks of recorded
validation data. Neural calls are computed inline here; in deployment they would run on a
second core.

| System | Tick p50 | Tick p99 | Tick max | CPU cores | Neural calls/s (replay) |
|---|---|---|---|---|---|
| D | 0.68 ms | 1.30 ms | 3.8 ms | 0.66 | 0 |
| D-ev | 0.76 ms | 1.42 ms | 4.9 ms | 0.73 | 0 |
| V4.1 | 0.78 ms | 3.51 ms | 9.0 ms | 0.86 | 32 |
| V4 @100/s | 0.76 ms | 4.02 ms | 8.5 ms | 1.01 | 99 |
| V4 | 1.03 ms | 4.82 ms | 11.0 ms | 1.83 | 447 |

- **Over budget at the tail:** even D-ev's p99 (1.42 ms) exceeds a 1 ms tick in Python.
  None of these figures is a sensor-to-actuator latency, and a 1 kHz loop rate is not a
  demonstrated 1 ms latency.
- **Deadline misses:** 0 in all runs. This is simulated: the simulator draws latency
  from V3's measured distribution rather than timing the network live.

## 6. Failure cases
The figure is `results/v41/fig_v41_cases.png`; per-tick logs are in
`results/v41/cases/*.csv`.

**(a) V4.1 under occlusion: a fall where D-ev survived** (hit 1.5 rad/s, seed 4008).
- During the occlusion (ticks 210–310) V4.1 correctly entered *unobserved*, its
  ±2σ band widened to cover the truth, and it called the network every 10 ms.
- **The network was as blind as the geometry**, because it reads the same occluded time
  surface. Its answers stayed near 0° while the true tilt rose to 20°.
- Because the band was wide, the gate accepted them (rejections averaged 0.1 per trial in
  every condition). So **wrong neural answers pulled the estimate the wrong way and
  narrowed σ**.
- The estimate snapped back when the pole reappeared (tick 310), but the pole was already
  at 30°, and the cart later ran off the track.

**Lesson:** a gate centred on a deliberately wide prior cannot reject a wrong helper.
The helper must bring independent information, or need corroboration.

**(b) Unfamiliar plant: track-end falls** (all 12 of V4.1's falls were after 3 rad/s
hits).
- V4.1's estimates were the most accurate (0.86° angle, 7.8 mm cart position), and it
  still fell.
- A post-hoc diagnostic on the same trials without the supervisor gave **100% survival
  for D, D-ev and V4.1** (`results/v41/diag_ood_nosup.json`).
- The falls therefore come from the supervisor, whose braking-distance and barrier
  models assume the nominal plant. This repeats V4 report §6.5.
- Why V4.1 was hit harder than D-ev is not established; it may be chance in how the
  trajectories meet the track limit.
- The trace also shows a **second flaw**: at the hit, real fast motion produced an event
  burst that the detector labelled *corrupt* (2–4% of ticks in every condition).
  Measurements were rejected for about 100 ms, and the estimate lagged (6° vs 12°).

**(c) Combo: the scene is unobservable.**
- With low light and 5× noise, geometry was unusable 99.7% of the time, and the pole
  drifted to −45° even before the hit.
- The network, reading the same sensor, could not recover it either.
- Only the IMU baseline survives combo, in earlier versions.

## 7. Limitations
- **Configurations.** One plant, one sensor model, and one neural seed (s0) for V4/V4.1.
- **Supervisor.** It was re-tuned only on nominal validation runs. It is the cause of
  the OOD-plant falls for every estimator.
- **CPU timing.** Per-tick CPU in the benchmark rows (`est_cpu_ms_*`) was measured with
  2 workers sharing 2 cores and is only indicative. Section 5 is the reference.
- **No hardware.** Nothing here is validated on real hardware.

## 8. Before proposing any new architecture
As the milestone asked, this report stops here. The measured gaps point at three things
that are not new architecture:
1. Separate real motion from noise in the burst/corruption test.
2. Do not accept a helper's answer that is drawn from the same blinded sensor without
   corroboration.
3. Make the supervisor robust to plant uncertainty.

Whether neural assistance is worth having at all should be re-asked only once there is a
case where the event sensor holds information the geometry cannot extract.
