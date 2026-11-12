# Python dependency audit and reproducibility (pre-AWS)

This project uses **CPython 3.11** on GitHub-hosted Ubuntu 24.04 for the
dependency audit. Its scientific runtime manifests are in
`versions/v4.2/requirements-docker.txt`; CPU-only PyTorch is installed
from the official PyTorch CPU index. The runtime requirements currently use
version **ranges** for runtime images and an **exactly version-pinned** CI audit snapshot at `requirements-ci-py311-ubuntu24.lock`. The snapshot is not hash-pinned or an immutable cross-platform lockfile. Do not mistake repeatable
seed choices for a byte-for-byte reproducible dependency environment.

## Automated evidence

`.github/workflows/dependency-security.yml` performs:

1. Install the V4.2 runtime dependency ranges and CPU-only torch.
2. Check dependency consistency with `pip check`.
3. Record actual package versions (`dependency-environment.json` and
   `dependency-environment.freeze.txt`).
4. Generate a CycloneDX JSON SBOM for the installed Python environment.
5. Use `pip-audit` to check installed packages against known vulnerability
   advisories; a detected advisory fails the CI job.
6. Upload available manifests as GitHub Actions artifacts, including when the
   audit fails. This is intentional: audit results must not be silently suppressed.

## Security and reproducibility limits

- Results reflect the package versions and advisories visible at CI run time.
  New advisories may appear later.
- This does not scan Git history, vendored binaries, GitHub Action supply
  chain, operating-system CVEs or cloud IAM roles.
- Generated SBOM includes *installed audit tools*, as well as runtime libraries;
  it is an environment SBOM, not a minimal production-only SBOM.
- The reviewed `requirements-ci-py311-ubuntu24.lock` is checked against the `pip freeze --all` output in the security audit. It captures exact versions for Python 3.11 on Ubuntu 24.04, including the audit tooling and CPU-only torch. It is **not hash-pinned** and other jobs/images still use range-based dependencies. A full release build lock policy remains outstanding.
- Dependency scanning must not be treated as security proof or deployment approval.
- CI workflow does not deploy to AWS or use any AWS credentials.

Before AWS deployment, review the audit artifact, address known vulnerabilities,
resolve and validate a hash-pinned, production-specific lock policy, and repeat the scan for the selected
release commit. No paid cloud infrastructure is required for this work.
