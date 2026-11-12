# V4.1 Step 1: Audit of the V4 modules

This audit reads the code and traces validation runs (seeds 300–311) to find out three
things:
- how geometry validity is decided;
- what happens when evidence goes missing;
- how the controller reacts to an unreliable estimate.

Probes: `v41/probe_evidence.py` and `v41/calib_geometry.py`, written to
`results/v41/geometry_calibration.json`.

## 1. Event geometry (`aurasync/v4/geometry.py`)

**Input:** fresh events (younger than 3 ms) in a band of ±0.6 m around the current
pivot estimate.

**Fit:**
- RANSAC with 60 hypotheses, then least-squares on the inliers.
- `measure()` returns `None` when there are fewer than 8 events in the band, no usable
  hypothesis, or fewer than 8 inliers.

**Validity:** a fit is `valid` when `span ≥ 0.35` (fraction of the pole's length covered)
and `snr ≥ 4`. `quality` is computed but **not used** for gating; it only feeds a
reliability EMA in V4.

**Reported σθ:** inlier scatter over the vertical lever arm, floored at 0.4°.

On validation, the reported σθ is **over-confident**:

| Fit support | Within 2σ | Inflation needed for 95% |
|---|---|---|
| all valid fits | 89% | — |
| span 0.35–0.6, < 12 inliers | 78–81% | ×1.5 (p95 error 10–15°) |
| other bins | — | ×1.2–1.6 |

**When no valid fit occurs:**
- **The pole is still.** Before the hit there were 0% valid ticks and about 1 fresh
  event in the pole region per tick: a still pole fires no events.
- **Occlusion.** Over ticks 200–320, only 19–33% of ticks had a valid fit, although the
  visible lower pole fires about 20 events per tick. Fits there sit at the minimum span
  with 8–9 inliers, and their errors run from 5° to 20°.

## 2. Baseline D (`aurasync/v4/geo_pipeline.py`)

**Prediction:** physics each tick with the V3 process noise, Q = base × 10⁴, which adds
σθ̇² = 1 (rad/s)² per tick.

**Missing evidence:** there is no handling at all. With no valid fit, D propagates with
the full Q, so in a quiet scene **σθ reaches about 100°**. The trace for validation seed
300 shows the consequences:
- The supervisor's stale fallback (σθ > 8°) is active through the whole quiet period
  before the hit.
- The first fits at motion onset are noisy: at ticks 204 and 210 they read +14° and −21°
  while the truth is 0.1–0.6°. Because σθ is so large, D accepts them almost fully.

**Outlier rejection:** the gate (d² > 9) only starts after 50 successful fusions, and
then applies **forever**. There is no re-initialisation, so a diverged estimate can lock
out correct measurements. This is the likely cause of the 68–78° errors in `combo`.

**Velocities:** inferred only through the dynamics and cross-covariance.

## 3. V4 (`aurasync/v4/pipeline.py`)

**Neural observer:** runs whenever its worker is free, at about 440 calls/s. V3
triggers: event burst, doubt, margin, and a maximum gap of 10 ticks.

**ConsistencyDetector:** a CUSUM on d², updated **only on ticks with a valid fit**. The
`missing_events` and `stale_evidence` reasons are logged, but they change neither the
CUSUM nor the covariance (see case (b) of the V4 report).

**Correction:**
- gated fusion of θ and pivot x (d² ≤ 9);
- confirmed re-initialisation (4 fits in 12 ms, spread ≤ 3°, 40 ms cooldown), using the
  full fit buffer;
- position-only neural fusion (`kalman_pos`), which is **not gated**.

**Inflation:** Σθθ is raised to (10°)² only when the alarm is up and no correction was
possible.

## 4. Recovery supervisor (`aurasync/safety.py`)

- **Delay compensation**, then a **stale check**: if σθ > `stale_deg` (V4 4°, D 8°), the
  nominal command comes from a low-gain fallback LQR.
- **CBF filter**, tightened by κσ:
  - hard limits: track end (braking distance) and |ψ| ≤ 35°;
  - soft target: |ψ| ≤ 10°, gated by reachability and latched.
- **Its only signal of reliability is Σ.** An estimator that reports a small Σ while
  wrong, or a huge Σ while right, steers the supervisor wrongly. Both happen in D: small
  Σ during occlusion fits, huge Σ (about 100°) while quiet.

## 5. What V4.1 must change (derived from the above)

1. **Calibrate measurement noise by support:** ×1.4 generally, ×1.6 for span < 0.6 or
   fewer than 12 inliers.
2. **Classify the evidence each tick** and size the process noise accordingly:
   - **quiet** (sensor alive, no pole activity, no predicted motion): bounded
     uncertainty;
   - **unobserved** (pole-region activity without a usable fit, or motion predicted but
     no fit): full process noise, never "stationary";
   - **contradictory** (a valid fit outside the gate): corroborate, then re-initialise;
     meanwhile covariance matching so it cannot lock out;
   - **corrupt** (noise burst, or fits inconsistent with each other): reject, widen.
3. **Call the network only when geometry is inadequate,** within a budget and with a
   periodic check-in. Gate its answer, and accept positions only.
