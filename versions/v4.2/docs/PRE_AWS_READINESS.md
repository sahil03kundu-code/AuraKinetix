# Pre-AWS engineering readiness gate (V4.2)

**State: NOT YET READY.** This checklist is a gate, not a claim of production readiness or hardware results. The purpose is to finish reproducible, cost-free GitHub-hosted preparation before asking the owner to sign in to AWS.

## Scope and evidence

- No AWS credentials, account access, resources, purchases, or billing changes in this stage.
- The four existing CI jobs (legacy V4.1 tests, V4.2 unit, V4.2 integration, slim Docker) must be green **on the same main commit**.
- The new Pre-AWS Validation workflow must be green on the same commit and upload two machine-readable JSON artifacts.
- The validation audit compares **geometry baseline** with **V4.1 network-free evidence pipeline**, using paired seeds. These are *not* the full neural-assisted AuraSync V4.2 system and cannot establish its superiority.
- Train/tune only on validation seeds; held-out seeds are reserved for a final evaluation. In this lightweight CI, the short runs merely check reproducible infrastructure and do not constitute a statistically powered scientific test.

## Required work before claiming AWS readiness

- [ ] Scientific: pre-register comparisons (D, D-ev, V4.1, V4.2), hypotheses, seed sets, metrics; freeze code; run full-length held-out runs; report survival, safe recovery, estimation error, false intervention, CPU cost, paired intervals, and negative outcomes. Do not claim a result from short CI smoke runs.
- [ ] Reliability: validate disturbance/kick/plant snap, camera noise, event bursts, dropouts, occlusion, delayed evidence, invalid/nonfinite inputs, and meaningful recovery behavior over longer simulations; include failure cases.
- [ ] Performance: report observed distributions of per-tick CPU time, model-inference latency and memory at a documented runner environment. **Never infer sub-millisecond hardware recovery** from Python simulation or GitHub runners.
- [ ] Security: least-privilege GitHub Actions permissions, dependency/source audit, pinned or documented dependencies, secret-leak safeguards, no untrusted PR execution with release credentials.
- [ ] Reproducibility and compatibility: documented Python/OS matrix, CPU-only CI, explicit package versions/lock policy, fixed random seeds, archived report artifacts and environment versions.
- [ ] Observability: readable test results, machine-readable experiments, clear failing-run triage instructions and coverage change reporting.
- [ ] Release: validate tag-to-directory logic (currently `versions/${TAG%.*}`, which may not resolve to an existing directory), artifact exclusions, release asset correctness, and non-publishing dry-run before any release.
- [ ] Final gate: all required workflows green for the chosen commit, critical issues resolved, reproducibility records archived, and handoff documented.

## Current small-scale audit

Run from `versions/v4.2` with CPU PyTorch installed:

```sh
python validation_audit.py --split validation --seeds 3 --ticks 30 --output audit-validation.json
python validation_audit.py --split heldout --seeds 3 --ticks 30 --output audit-heldout.json
```

Outputs contain per-seed paired results and **exploratory** paired bootstrap intervals. The seed ranges are separate; small sample counts and very short episodes provide no evidence that recovery is superior. These reports should inform whether to run a proper high-power benchmark, not whether to deploy physically.

## AWS handoff (future, zero-spend planning)

When the gate is signed off, announce **AWS PHASE REACHED — READY TO START** and obtain explicit owner approval for an AWS deployment approach, region, account/billing safeguards, hard cost cap, and IAM authentication. Default to read-only inspection, budgets and cost alarms before deployment. Do **not** create infrastructure or spend money without approval.
