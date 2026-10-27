# AuraSync V4 — Design: Detect → Correct → Recover

Status: simulation only. Everything here was built and measured in the cart-pole simulator
in this repository. Nothing has been run on hardware.

## 1. The problem V4 addresses (from the V3 audit, Step 1–2)

On validation runs (seeds 300–315; hits of 1.5–4 rad/s; 160 runs), every fall of the
V3 event observer came after a stretch of **confidently wrong** tilt estimates:
- error > 10° while the observer reported σ < 3°;
- often with the **wrong sign**: 73 of 80 unshielded runs contained wrong-sign ticks.

V3's uncertainty head did not flag these stretches, so the shield trusted them.
Details: `docs/V3_AUDIT.md` and `results/v4/dev_catalogue.json`.

Analogy: a ship's navigator (the neural observer) dead-reckons from his instruments.
After a big wave he is sure of his position, and wrong. V4 adds a lookout who takes
direct bearings on the mast (event geometry), a rule for when to believe the lookout
(the gated detector), and a way to tell the captain (the controller and supervisor)
when nobody is sure.

## 2. Architecture

```
events ─► time surface ─┬─► V3 neural observer (CNN + SSM, triggered, async) ──► neural estimate + σ
                        │                                                         │ (Kalman, x and θ only)
                        └─► Module A: EventGeometry (RANSAC line fit, < 3 ms old)  ▼
                                   │  θ_m, x_m, σ_m, quality      ┌──────── fused state ŝ, Σ ───────┐
                                   ▼                              │                                │
                            ConsistencyDetector (CUSUM on d²) ─► Module B: fuse | re-init | none    │
                                   │ alarm, reasons               Module C: inflate Σ when alarmed  │
                                   ▼                                        but not correctable     │
                          Module D: ShieldedController (LQR + CBF supervisor, stale fallback) ◄─────┘
                          Module E: trigger policy + optional hard budget (≥ min_gap ticks between neural calls)
```

Code: `aurasync/v4/geometry.py`, `aurasync/v4/pipeline.py`, `aurasync/v4/geo_pipeline.py`
(Baseline D), `benchmark_v4.py`.

## 3. Modules

### A. Detection — `EventGeometry` + `ConsistencyDetector`
- **Independent measurement.** A RANSAC line fit x = x_p + (y − 0.04)·tan θ runs on events
  younger than 3 ms, in a band around the current cart estimate. It is vectorised
  (60 hypotheses) and fixed-seed.
- **Measurement outputs:** θ_m, pivot x_m, σ_θ (from residual spread and inlier count,
  floor 0.4°), span, SNR and quality. The fit is invalid when the span, SNR or inlier
  count is too small.
- **Dev accuracy** (validation, open-loop):
  - valid on 72.7% of ticks;
  - median |error| 0.86°, p95 4.2°;
  - 95% of normalised errors within ±2σ;
  - cost p50 0.20 ms, p99 0.48 ms.
- **Score.** d² = (θ_m − θ̂)² / (Σθθ + σ_m²). A CUSUM accumulates
  g ← max(0, g + min(d², 16) − 4) and raises the alarm at g > 40.
  - A single outlier can add at most 12, so the alarm needs at least 4 strong
    disagreements in a row (unit-tested).
- **Outputs:** alarm and timestamp, score g/h, reliability (EMA of quality) and reasons:
  `angle_mismatch`, `sign_mismatch`, `missing_events` (estimate says motion, sensor sees
  none), `stale_evidence`, `noise_burst`.

### B. Correction
1. **Gated Kalman fusion** when d² ≤ 9: θ and pivot x are fused as measurements, and
   the velocities are corrected through the cross-covariance.
2. **Confirmed re-initialisation** when the alarm is up and the measurement is outside
   the gate:
   - needs ≥ 4 valid measurements within 12 ms, spread ≤ 3°, and a 40 ms cooldown;
   - θ is set to their median, θ̇ to their slope, x to the median pivot;
   - the SSM memory is zeroed and any in-flight neural job is cancelled.
3. **Neural fusion (`kalman_pos`).** The neural estimate is treated as a measurement of
   x and θ only, with its calibrated variance.
   - The first design overwrote the state with the neural output, as V3 does. On
     validation its velocity outputs were confidently wrong after big hits (θ̇ error
     117–139 °/s vs 49–89 °/s for geometry alone). Overwriting also threw away the
     cross-covariance.
   - Both the overwrite design (`v4-ow`) and a full-state Kalman variant (`v4-kf`) were
     kept as reported ablations.
4. **No correction from bad evidence.** Invalid fits and noise bursts never correct.

### C. Calibration
Σ is the posterior of the fusion, so it narrows only when evidence agrees. If the
detector alarms but nothing passes the gate or confirmation, Σθθ is inflated to (10°)²
and Σθ̇θ̇ to at least 4. The supervisor and stale fallback then react to that higher
uncertainty.

### D. Recovery-aware control
This is the V3 `ShieldedController`:
- LQR plus a CBF supervisor on ψ = θ + 0.1θ̇, a braking-distance cart barrier and
  delay compensation;
- an uncertainty-aware margin κσ and a stale-evidence fallback.

Its parameters were re-tuned for V4 on validation seeds only (`results/v4/tuning_v4.json`):
- V4: γ 0.03, κ 2, stale 4°;
- geometry baseline: γ 0.05, κ 2, stale 8°.

### E. Deadline-aware computation
- The V3 trigger policy (event burst, doubt, safety margin, max gap) and the 6 ms
  deadline are kept.
- Optional hard budget `min_gap`: at most one neural call per min_gap ticks
  (`-b250`, `-b100`).
- When a budget is set, a residual trigger (d² > gate) is also on, so calls go where
  evidence disagrees.
- Geometry and detection run every tick; their cost is measured in the latency report.

## 4. Things that were tried and changed (all on validation data)

| Change | Why |
|---|---|
| CUSUM cap 100 → 16 | unit test: one outlier could raise the alarm |
| neural overwrite → `kalman_pos` | overwrite: 85.9% validation survival; kp: 98.4% (64 runs) |
| `_finish_job` branch `== "kalman"` → `startswith` | bug: kp silently used overwrite; affected candidates re-run |
| residual trigger default on → only with a budget | would have changed the validated design |
| frame camera pole length under `ood_plant` | leak/bug found in audit: imaged nominal length; fixed, V3 frame configs re-run |

## 5. Integrity rules applied
- **No ground truth inside estimators.** The pipeline receives `s_true=None` in replay
  tests. Only sensor simulators read the true state.
- **Fixed before testing.** All design choices and tuning used validation seeds
  300–315. Test seeds were run once, after the design was frozen (commit `28d23b7`).
- **Timing.** A 1 kHz control loop is not a demonstrated 1 ms sensor-to-actuator
  latency; see `docs/V4_LATENCY.md`.
