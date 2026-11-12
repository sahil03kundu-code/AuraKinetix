# AuraKinetix public AWS showcase

This folder contains a dependency-free, responsive **static research showcase**.
It does **not** run the simulation in the browser, require API tokens, call AWS
services, or claim real-hardware validation.

## Published status

**Not deployed to AWS yet. There is no public URL until AWS Amplify Hosting
completes an approved deployment.** The site exists in the GitHub repository
and passes offline tests in GitHub Actions.

## Hosting via AWS Amplify (manual, cost-reviewed)

1. Open the AWS Amplify Hosting console: https://console.aws.amazon.com/amplify/home
2. Confirm account billing, AWS Free plan eligibility, promotional credits,
   and whether any hosting usage will consume credits or cause charges. Cloud
   credits are **not** a spending cap. Do not upgrade or create infrastructure
   solely for this demo without reviewing the costs.
3. Choose to create a new app, and connect the GitHub repository
   `sahil03kundu-code/AuraKinetix` on branch `main`.
4. The repository-root `amplify.yml` copies only
   `aws/demo-site/index.html` into `dist`; the published artifact is
   `dist/index.html`. Check the build settings before pressing Deploy.
5. Turn off unwanted branch previews or extra build environments. No backend
   environment, EC2, Lambda, S3, RDS, or custom domain is required for this
   static showcase.
6. After an approved successful Amplify build, AWS will show an **actual**
   public default `amplifyapp.com` URL. Share that issued URL; don't guess it.
   Verify the homepage, GitHub links, visible safety caveat, and mobile layout.
7. To stop ongoing hosting usage, delete/disable the Amplify app after
   saving the link and records as needed; consult AWS billing for actual cost
   and any credit usage.

This public website is independent of the separate OIDC-based GitHub STS
identity preflight. Connecting a repository to Amplify does not prove GitHub
Actions' OIDC trust role is configured.

## Local preview without AWS

Open `index.html` in a browser. No package installation or network service
is required except for opening links to GitHub. GitHub Actions stores the
reviewed static HTML as a downloadable artifact before any cloud deployment.

## Accuracy policy

The site deliberately makes no claims of physical sub-millisecond recovery,
real-time operational safety, production readiness or neural V4.2 superiority.
Scientific gaps are tracked in `versions/v4.2/docs/SCIENTIFIC_PROTOCOL.md`
and issue #21.
