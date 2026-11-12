# AuraSync V4 — Compute and latency report (Module E)

## Method
- **Setup:** measured with `v4/latency_v4.py` and `results/v4/latency_v4.json`.
- **Machine:** Intel Xeon @ 2.8 GHz, 1 PyTorch thread, otherwise idle.
- **Input:** recorded validation sensor data (7,070 ticks), replayed through V4
  (`v4+sup-s0`) with no ground truth.
- **Timing:** each stage is timed with `perf_counter`.

**What is not measured:** sensor readout, transport (USB/MIPI), OS scheduling jitter
under load, and actuator delay. No real sensor or actuator exists here.

## Per-stage compute (ms)

| Stage | p50 | p90 | p99 | max | runs |
|---|---|---|---|---|---|
| event geometry (RANSAC) | 0.283 | 0.433 | 0.582 | 2.90 | every tick |
| consistency detector | 0.009 | 0.012 | 0.042 | 0.13 | every tick |
| Kalman fusion | 0.075 | 0.113 | 0.172 | 0.89 | 69% of ticks |
| supervisor (CBF + delay comp. + fallback) | 0.387 | 0.569 | 0.755 | 1.86 | every tick |
| neural correction (CNN + SSM) | 2.07 | 2.53 | 3.42 | 7.33 | 447 per second |
| whole tick, neural inline | 0.57 | 3.11 | 4.50 | 7.55 | every tick |

## CPU load in cores (CPU-seconds needed per second of control)

| Configuration | Neural calls/s | Cores |
|---|---|---|
| V4 default | 447 | **1.50**: 0.54 always-on + 0.97 neural |
| V4, budget 100/s | 99 | 0.58 |
| Baseline D (geometry, no network) | 0 | 0.25 |

- **The default V4 is not real-time on one core of this CPU.** It needs the
  asynchronous neural path on a second core, as designed.
- With a 100/s budget it fits on one core, and the closed-loop results showed no
  measurable cost from the budget.

## Deadline misses
The benchmark's 0% miss rate against the 6 ms deadline is a **simulation result**. The
simulator draws each neural job's latency from V3's measured latency distribution. It
does not time the job live. A live measurement on the target hardware is step 1 of
the hardware plan.

## Modelled sensor-to-actuator latency (model, not measurement)
The model is the worst-case 1 ms tick quantisation plus the measured p99 compute:
- **event path** (geometry + detector + fusion + supervisor): about 2.6 ms;
- **neural path:** about 4.4 ms.

A configured 1 kHz control rate is **not** a demonstrated 1 ms sensor-to-actuator
latency.

## Memory
Peak RSS is 740 MB, mostly PyTorch and the loaded models.
