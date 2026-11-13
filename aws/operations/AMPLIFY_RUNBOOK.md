# AuraKinetix — AWS Amplify operations and rollback runbook

## Scope / deployed environment

- Static public research showcase (not the simulation engine)
- Repository: `sahil03kundu-code/AuraKinetix`, branch `main`
- Live HTTPS endpoint: https://main.d2aek34w544oam.amplifyapp.com/
- App ID visible in AWS Console: `d2aek34w544oam`
- Hosting region from AWS Console: `ap-southeast-2`
- Build: `amplify.yml` copies `aws/demo-site/index.html` to `dist/index.html`
- No AWS API credentials are required for the public-site HTTP monitor.

## Detection and verification

`.github/workflows/aws-public-site-monitor.yml` runs at approximately
02:23 and 14:23 UTC each day, and on manual dispatch. GitHub scheduled jobs
can be delayed or missed. The workflow performs a read-only HTTPS GET to the
approved host, checks HTTP 200, text/html and expected content markers, and
retries once. A downloadable JSON report is uploaded even after a failure.

**Do not interpret passing status as proof of 100% uptime, successful AWS
backend operation or proof of event-camera simulation accuracy.** This is a
sampled static-website check only. GitHub Actions run failures are visible at:
https://github.com/sahil03kundu-code/AuraKinetix/actions
Check your repository notification settings separately if email alerts are
required; workflow failure alone does not prove email delivery.

## Manual incident procedure

1. Open the live HTTPS page in a private browser window and retry one time.
   Confirm whether the page is unavailable, displays a 404/403, or contains
   incorrect/old content.
2. Open AWS Amplify Console in region `ap-southeast-2`:
   https://ap-southeast-2.console.aws.amazon.com/amplify/home?region=ap-southeast-2
   Select AuraKinetix, then the `main` branch, and read the latest build
   / deploy logs, app status and deployment history.
3. Check GitHub `main` commit, `amplify.yml`, and
   `aws/demo-site/index.html`. Determine whether failure is build-time,
   hosting-level, DNS/TLS, or an unexpected content change.
4. Check AWS Billing / Budgets for unexpected cost consumption. The verified
   budget alert is configured for actual spend above $0.01, but reporting can
   lag; it is **not** a hard cap.
5. If a recent commit introduced the error, prefer reverting that GitHub
   change through a reviewed pull request. The connected Amplify branch can
   then redeploy the corrected source. Alternatively, if Amplify offers a
   supported redeploy of a **known-good** version, inspect that option and its
   related billable build usage before running it.
6. Only after explicitly checking the target commit, invoke the permitted
   Amplify redeploy/rollback process in the AWS Console. **No automatic
   production rollback** is configured, and monitoring jobs never mutate AWS.
7. Verify the public HTTPS URL again and inspect GitHub Actions' next
   public-site check. Record incident start, detection time, suspected cause,
   failed deployment, recovery deployment, verification and cost impact.

## Safe release checklist

- [ ] Latest `main` CI checks pass, including static demo preflight
- [ ] Proposed copy/links/caveats reviewed for accuracy
- [ ] No AWS credentials or secrets appear in static HTML
- [ ] `amplify.yml` has `dist` as build artifact directory
- [ ] No backend build or unneeded branch preview enabled
- [ ] Latest known-good Git SHA and last known-good Amplify deployment noted
- [ ] Budget status reviewed; hosting charges/credits understood
- [ ] Public-site smoke test manually rerun after deployment, if needed

## Unfinished enhancements

Actual IAM/OIDC role authentication is separate and remains unverified.
There is no protected environment with deployment approvals, automated
rollback, alert fanout, private observability, or agreed production SLO.
The research code remains a simulation engine, not a deployed AWS inference
or physical control service.
