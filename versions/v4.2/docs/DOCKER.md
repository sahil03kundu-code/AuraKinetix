# AuraKinetix V4.2 - Docker guide

Status legend: **PASS** executed and succeeded, **NOT RUN** not executed, **BLOCKED** a prerequisite is missing.

## 1. What this image is (audit summary)
AuraKinetix is a Python simulation and research codebase, not a service. There is no server, port or health endpoint,
so none was added. The "entry points" are standalone scripts (`benchmark_v41.py`, `benchmark_v4.py`, `v41/validate.py`,
`train_v3.py`, `analyze_v41.py`, ...) and the test suites. The image is therefore a **job/test runner**.

| Finding | Detail |
|---|---|
| Runtime deps | numpy, scipy, matplotlib (Agg backend); CPU-only torch for `aurasync/estimator.py`, `scheduler.py`, `v4/pipeline.py`, `v41/pipeline.py` and training/benchmark scripts. scikit-learn is imported lazily in `eval_uncertainty.py` only (not installed). |
| Test deps | pytest, pytest-cov (new suites `tests/unit`, `tests/integration` need only numpy/scipy). Original `tests/test_v4.py`, `tests/test_v41.py` need torch. |
| Env vars read by scripts | `PROCS`, `TRIALS`, `STRESS_TRIALS`, `WHICH`, `CONFIGS`, `SEED0`, `NFILES`, `EPS`, `RECAL`, `ONLY`, `ROUND`, `STAGE`, `KICK` |
| Paths written (relative to the working dir) | `results/`, `checkpoints/`, `data/` |
| Not in git / not in the image | `data/` (gitignored, ~3.4 GB), V4.2 `checkpoints/` and `results/` (untracked) |
| Existing Docker config | none |

## 2. Architecture
`Dockerfile` (context = `versions/v4.2`), three stages:
1. `builder` - venv with pinned-range deps (`requirements-docker.txt`) and, by default, CPU torch from the PyTorch CPU index. Build tools and pip cache stay out of the final image (BuildKit cache mount).
2. `runtime` - `python:3.11-slim-bookworm` (same minor as CI), venv copied in, non-root user `aura` (uid 10001), `ENTRYPOINT ["python"]`, default command = import smoke test.
3. `test` - runtime + pytest/pytest-cov; default command runs `tests/unit` and `tests/integration`, JUnit XML to `/reports`.

Design choices: no checkpoints/data/results baked in (mounted instead); `OMP_NUM_THREADS=1` to match how the project measures; `MPLBACKEND=Agg`; run with `--init` (compose: `init: true`) so SIGTERM reaches Python.
The multi-stage split mainly keeps pip/build residue out; its size benefit has **not been measured**.

Build args: `INSTALL_TORCH=1` (default) or `0` for the light image used by the new unit/integration suites; `PYTHON_VERSION=3.11`.
Resources: tests were light in a local venv (3.1 s for 23 tests, no torch). Benchmarks are heavy (about 2 h on 2 cores for `benchmark_v4.py` per the version README). Compose defaults: 2 CPUs, 3 GB RAM, sized for an 8 GB machine.

## 3. Windows PowerShell commands
Run from the repository root (`AuraKinetix`). Docker Desktop must be running.

```powershell
# Build (light image: no torch)
docker build -f versions/v4.2/Dockerfile --target runtime --build-arg INSTALL_TORCH=0 -t aurakinetix:v4.2-light versions/v4.2
# Build (full image with CPU torch)
docker build -f versions/v4.2/Dockerfile --target runtime -t aurakinetix:v4.2 versions/v4.2
# Build test image (full, with torch; add --build-arg INSTALL_TORCH=0 for a light one that runs only tests/unit and tests/integration)
docker build -f versions/v4.2/Dockerfile --target test -t aurakinetix:v4.2-test versions/v4.2

# Smoke test (default command)
docker run --rm --init aurakinetix:v4.2-light

# Unit + integration tests, JUnit XML exported to .\reports
New-Item -ItemType Directory -Force reports | Out-Null
docker run --rm --init --cpus 2 --memory 3g -v "${PWD}\reports:/reports" aurakinetix:v4.2-test
"exit code: $LASTEXITCODE"          # must be 0 on success, nonzero if any test fails

# Original V4/V4.1 suites (need torch; checkpoint tests skip unless mounted)
docker run --rm --init aurakinetix:v4.2-test -m unittest discover -s tests -v

# With checkpoints mounted read-only so TestBudget can run
docker run --rm --init -v "${PWD}\versions\v4.2\checkpoints:/app/checkpoints:ro" aurakinetix:v4.2-test -m unittest discover -s tests -v

# A benchmark, results written to a host folder (resumable scripts keep their raw files there)
docker run --rm --init --cpus 2 --memory 3g -e PROCS=2 `
  -v "${PWD}\versions\v4.2\checkpoints:/app/checkpoints:ro" `
  -v "${PWD}\versions\v4.2\data:/app/data" -v "${PWD}\versions\v4.2\results:/app/results" `
  aurakinetix:v4.2 benchmark_v41.py
```
Compose equivalent (from `versions\v4.2`): `docker compose run --rm tests`, `docker compose run --rm sim benchmark_v41.py`.

## 4. Verification report (this session)
| # | Check | Status | Evidence / note |
|---|---|---|---|
| 1 | Docker image builds | **BLOCKED** | `docker` not installed in the authoring environment |
| 2 | Container starts and runs default command | **BLOCKED** | same |
| 3 | Python imports (core modules) | **PASS** (outside Docker) | clean venv, Python 3.10.12, numpy 2.2.6, scipy 1.15.3: `core import OK`; exact CMD string exited 0. Not Python 3.11. |
| 4 | Unit tests | **PASS** (outside Docker) | 23 passed in 3.11 s with the `tests/unit tests/integration` command, on a copy filtered like `.dockerignore` |
| 5 | Integration tests | **PASS** (outside Docker) | included in the 23 |
| 6 | Invalid-input tests | **PASS** (outside Docker) | the unit suite includes NaN, infinite and bad-shape cases; all passed |
| 7 | Nonzero exit propagates | **PASS** (pytest only) | a deliberately failing test exited 1; Docker passthrough of the exit code **NOT RUN** |
| 8 | Runs without checkpoints | **PASS** (outside Docker) | the new suites need none; original checkpoint tests skip by design |
| 9 | Artifact export | **NOT RUN** | JUnit XML written to a host dir outside Docker only |
| 10 | Resource limits | **NOT RUN** | compose limits unverified |
| 11 | `compose.yaml` valid | **PASS** (YAML parse only) | `docker compose config` NOT RUN |
| 12 | torch-based tests (`test_v4.py`, `test_v41.py`) | **NOT RUN** | torch not installed in the check venv |
| 13 | Dockerfile lint (hadolint) | **NOT RUN** | tool unavailable |
| 14 | Vulnerability scan | **NOT RUN** | |

Dockerization is **not complete** under the success criteria until rows 1, 2, 4-7 pass inside Docker on your machine.

## 5. CI/CD recommendations (not created; needs approval)
The repo already has `tests.yml` (V4.1), `unit-tests.yml`, `integration-tests.yml`, `release.yml`. Don't duplicate the pytest jobs. Add one `docker.yml` (PR + main) with:
1. `hadolint` on `versions/v4.2/Dockerfile`.
2. `docker/build-push-action` with `push: false`, `load: true`, GHA layer cache, tag `aurakinetix:${{ github.sha }}`.
3. Run the smoke command and the `test` target from that image; upload the JUnit XML.
4. `aquasecurity/trivy-action` on the image (fail on HIGH/CRITICAL with a reviewed ignore list) and `pip-audit -r requirements-docker.txt`.
5. SHA-pin the actions, `permissions: contents: read`. Provenance (`provenance: true`, `actions/attest-build-provenance`) only once pushing to a registry is approved.
No image is published and no release is created by anything in this folder.

## 6. AWS migration path (design only; nothing created)
Workload: batch jobs (tests: seconds to minutes; benchmarks: hours), CPU only, no inbound traffic.
| Option | Fit | Notes |
|---|---|---|
| AWS Batch on Fargate (or ECS `RunTask`) | **Best fit for benchmarks** | pay only while the job runs; resumable scripts tolerate interruption; no servers to manage |
| CodeBuild (running the `test` image) | Good for test gates | simple, per-minute billing; or just keep GitHub Actions |
| ECS/Fargate *service* | Poor fit | continuous service with nothing to serve |
| EC2 (spot) | Cheapest for long benchmarks | more ops work; keep for heavy nightly runs |
Pieces: ECR (immutable tags = git SHA, scan on push); S3 for data/checkpoints/results (pull at start, sync outputs at end; avoids Fargate's limited ephemeral disk); CloudWatch Logs (log retention 14-30 days) and a failure alarm; GitHub OIDC role scoped to one repo/branch/environment (no long-lived keys); AWS Budgets alert. Pricing is intentionally not quoted: use the AWS Pricing Calculator with the measured job duration.

## 7. Troubleshooting
- `docker` not recognised: start Docker Desktop (WSL 2 backend) and reopen PowerShell.
- Build runs out of memory/slow: Docker Desktop > Settings > Resources; 8 GB host: give Docker 4 GB.
- Torch download fails: the CPU wheel index needs network access; retry or use `--build-arg INSTALL_TORCH=0`.
- `ModuleNotFoundError: torch` in tests: you built the light image; rebuild without `INSTALL_TORCH=0`.
- Tests skipped: checkpoint/data tests skip unless `checkpoints/` (and `data/v4dev`) are mounted.
- Permission denied writing results: the container user is uid 10001; on Windows bind mounts this is normally fine, on Linux `chmod`/`chown` the host folder.
- Ctrl+C does nothing: add `--init`.
- Mounted path wrong: PowerShell needs `${PWD}` with quotes; use backticks for line continuation.
