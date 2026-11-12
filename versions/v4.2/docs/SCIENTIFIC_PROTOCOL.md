# Pre-registered scientific evaluation protocol (V4.2)

**Status: exploratory protocol, NOT a scientific result or AWS deployment approval.**
Protocol version: 1. All thresholds below are declared *before* examining the final
held-out results. Changes require a new protocol version and a clear audit trail.

## Hypotheses and comparisons

- **D**: independent geometric Kalman baseline (`GeoPipeline`).
- **D-ev**: network-free evidence-aware geometry pipeline (`EvidencePipeline(neural=None)`).
- **V4.1 neural-assisted**: required comparator only when a validated checkpoint,
  inference configuration, and dependency manifest are present.
- **V4.2**: required comparator only when an actual V4.2 candidate implementation
  and the corresponding frozen checkpoint exist and pass contract tests.

The automated pilot *currently covers only D and D-ev*. A green CI status does not
imply V4.1 neural-assisted or V4.2 testing occurred.

Primary hypothesis (not presumed true): D-ev improves **paired safe-recovery rate**
relative to D without materially worsening failure rate or estimate error.
Other comparisons are unverified until their implementation/checkpoints exist.

## Data separation

Fixed pilot seeds: 2000 to 2007.
Validation seeds: 3000 to 3049 (use only to select/tune thresholds).
**Untouched final holdout:** seeds 700000 to 700099; must not be executed or
examined until the candidate, protocol and commit SHA are frozen. The earlier
30-tick `validation_audit.py` used heldout seeds beginning at 100000; these are
already exposed and **must not be reused** for the final holdout.

The audit entrypoint refuses to run the final holdout without an explicit
`--freeze-commit` marker. Include the exact Git commit in the report and do not
change code after opening the held-out gate. Run pilot/validation first.

## Faults and scenarios

Fixed sensor conditions: nominal, low_light, dropped_events, occlusion.
The pilot uses a fixed angle of 0.03 rad and a pole angular-velocity impulse of
0.3 rad/s after one-third of the horizon. A 35% cart mass increase at half the
horizon models a plant mismatch. The test runs a nominal LQR with a safety shield,
identically configured for D and D-ev, and the same scenario/seed.

Standard full-horizon proposal: >= 1000 simulation ticks per episode (1 second);
100 paired held-out seeds across four conditions. Confirm computational budget and
checkpoint availability before the full run. The pilot CI uses much shorter
episodes to test experiment machinery, not scientific power.

## Outcome definitions

- **Survival:** no simulator failure before the entire requested horizon.
- **Safe recovery (operational proxy):** survived, and final pole angle <= 0.1 rad,
  cart position <= 0.7 m, and final absolute pole angular velocity <= 1 rad/s.
  This proxy does **not** verify human/physical safety.
- **Angle RMSE:** square-root of mean squared estimate-vs-truth angle error (radians).
- **Intervention rate:** proportion of controller ticks for which the shield reports
  `modified`, `soft_dropped`, or `infeasible`.
- **False intervention rate:** NOT MEASURED. A valid truth-based reference policy and
  causal criterion must be specified; do not relabel interventions as false ones.
- **Compute cost:** NOT MEASURED in this statistical script. Attach the independent
  `performance_audit.py` runner timing report, never conflate with hardware latency.
- **Effect size:** paired D-ev minus D for survival, safe recovery, and angle RMSE.
- **Uncertainty:** resample whole paired seeds (never individual ticks) using a
  fixed-seed percentile bootstrap; report interval and raw per-seed outcomes,
  including negative differences. Low-N pilot intervals are exploratory.

## Stop rules and integrity

Keep failure episodes and negative results, including if the new method regresses.
Never tune on final held-out seeds; report exact source commit, Python version,
condition, plant change, controller/shield configuration, seed and tick count.
All physical recovery and <1 ms detect-to-act claims require separate instrumented
real-hardware evidence. Existing simulations do not establish either claim.

**AWS gate remains NOT READY** until model-based comparators, long-horizon held-out
runs, risk review, dependency lock policy, security, observability, release signoff,
and owner-approved AWS billing safeguards are complete.
