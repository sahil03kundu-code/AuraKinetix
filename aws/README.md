# AWS Phase 1 — safe identity preflight (NO DEPLOYMENT)

**Status: Prepared in GitHub only. Not deployed or authenticated to AWS.**

This provides a minimal and auditable first AWS integration for
`sahil03kundu-code/AuraKinetix`. It uses short-lived GitHub Actions OIDC
tokens instead of persistent access keys or secrets. It makes **no claims**
about real-hardware sub-millisecond recovery, V4.2 candidate superiority, or
production readiness.

## What exists in the repository

- `aws/oidc-readonly-preflight.json` — CloudFormation defines only an IAM
  GitHub OIDC identity provider and a role with **no attached permissions**.
  The trust requires `sts.amazonaws.com` audience and this repository's
  `main` branch. Both repository and owner are parameterized with exact
  defaults; do not change them to wildcards.
- `.github/workflows/aws-iac-preflight.yml` — lints the template and tests
  least-privilege contracts **without AWS credentials**.
- `.github/workflows/aws-identity-probe.yml` — **manual dispatch only**;
  requires GitHub variable `AWS_READONLY_ROLE_ARN` and checks only STS caller
  identity. Never deploys or creates resources. Workflow's OIDC permission
  is scoped to this job's token, not an AWS identity policy.

## AWS prerequisites before manually creating an IAM role

1. Confirm Free-plan status and remaining credit from AWS Billing again,
   including whether any service is restricted on the Free plan.
2. Pick and record account/region, account-root MFA, and a separate everyday
   IAM Identity Center or equivalent admin identity. **Do not post account
   numbers, passwords, keys, or MFA codes in chat**.
3. Confirm cost monitoring and billing alerts (including credit consumption)
   before enabling future AWS services. A $0 billing balance alone is not a
   hard spending limit; credits may be consumed by a running resource.
4. Check IAM Identity Providers to avoid creating a duplicate
   `token.actions.githubusercontent.com` provider. The template as written
   **creates a new provider**, so DO NOT deploy it if this provider already
   exists. Review/import/revise the stack instead.
5. Manually review the CloudFormation change set. Creating IAM resources
   requires `CAPABILITY_IAM`. Do not deploy anything without explicit
   review and approval. IAM roles/identity provider are not application
   compute infrastructure.

Once a reviewed role exists, set the role ARN as the GitHub repository
**variable** `AWS_READONLY_ROLE_ARN` (not an AWS key and not a secret).
Trigger the identity workflow on `main` manually, and inspect the GitHub
Actions outcome. It is gated to that exact repository and branch.

## Deployment plan — NOT yet executed

First useful AWS stage after identity: private, encrypted, versioned S3 release
artifact storage **only if approved and account eligibility/cost reviewed**.
Then an explicit manually approved pipeline to upload a verified source
artifact with very narrow S3 object permissions. That should be followed by
a sandbox deployment test plan; no automatic EC2/ECS/ECR/CodeBuild deployment,
no public bucket, and no privileged GitHub role. Artifacts and dependency
SBOMs are not production-ready workloads by themselves.

The AWS account is **not connected directly through a tool in this session**.
A screenshot of AWS console billing previously showed a Free-plan account
with promotional credits, but that is not live access. GitHub CI cannot access
the account until an approved OIDC role and repository variable exist.

## Current scientific release blockers

See `versions/v4.2/docs/PRE_AWS_READINESS.md` and issue #21. The full
four-method held-out comparison, validated neural checkpoints, long-horizon
fault evidence, and production release approval remain separate blockers.
Do not equate an AWS OIDC authentication success with production readiness.
