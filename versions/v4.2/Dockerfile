# syntax=docker/dockerfile:1
# AuraKinetix V4.2 - simulation / test job image (CPU only).
# Build context: versions/v4.2   (see docs/DOCKER.md)
#
#   target "runtime": numpy, scipy, matplotlib (+ CPU torch if INSTALL_TORCH=1)
#   target "test"   : runtime + pytest, pytest-cov
#
# Data, checkpoints and results are NOT copied in; mount them (see compose.yaml).

ARG PYTHON_VERSION=3.11

# ---- builder: resolves and installs dependencies into a venv ----------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG INSTALL_TORCH=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_COMPILE=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-docker.txt \
 && if [ "$INSTALL_TORCH" = "1" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"; \
    fi

# ---- runtime ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
COPY --from=builder /opt/venv /opt/venv

# Non-root user; the working tree and the mount points are owned by it.
RUN useradd --create-home --uid 10001 aura \
 && mkdir -p /app /reports /app/data /app/checkpoints /app/results \
 && chown -R aura:aura /app /reports
WORKDIR /app
COPY --chown=aura:aura aurasync ./aurasync
COPY --chown=aura:aura v4 ./v4
COPY --chown=aura:aura v41 ./v41
COPY --chown=aura:aura tests ./tests
COPY --chown=aura:aura *.py *.sh ./
USER aura

# Default command: import smoke test only (no simulation, no checkpoints needed).
# Run `docker run --init ...` (or compose `init: true`) so SIGTERM/SIGINT reach Python.
ENTRYPOINT ["python"]
CMD ["-c", "import importlib.util as u, aurasync.physics, aurasync.controller, aurasync.safety, aurasync.v4.geometry; print('aurasync core import OK'); u.find_spec('torch') and __import__('aurasync.v41.pipeline') and print('aurasync v41 (torch) import OK')"]

# ---- test: adds pytest tooling; reports go to /reports -----------------------------------
FROM runtime AS test
USER root
RUN pip install --no-cache-dir "pytest>=8,<9" "pytest-cov>=5,<7"
USER aura
CMD ["-m", "pytest", "tests/unit", "tests/integration", "-v", "-p", "no:cacheprovider", "--confcutdir=tests", "--junitxml=/reports/test-results.xml"]
