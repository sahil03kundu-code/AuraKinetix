"""Verify publicly hosted Amplify deployment-manifest.json against live HTML.

Read-only and account-key-free. A matching manifest is evidence of deployed
static content consistency, not independent cryptographic attestation of AWS.
"""
import argparse
import json
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from tools.build_amplify_manifest import validate_manifest

SITE = "https://main.d2aek34w544oam.amplifyapp.com"
HOST = "main.d2aek34w544oam.amplifyapp.com"
LIMIT = 512_000


def safe_fetch(path, *, opener=urlopen):
    if path not in ("/", "/deployment-manifest.json"):
        raise ValueError("Unapproved path")
    req = Request(SITE + path, headers={"Cache-Control": "no-cache", "User-Agent": "AuraKinetix-provenance-verifier/1.0"})
    with opener(req, timeout=12) as response:
        target = urlsplit(response.geturl())
        if target.scheme != "https" or target.hostname != HOST:
            raise ValueError("Redirected to an unapproved host")
        if response.getcode() != 200:
            raise ValueError("Non-success HTTP status")
        body = response.read(LIMIT + 1)
    if len(body) > LIMIT:
        raise ValueError("Unexpectedly large response")
    return body


def probe(*, opener=urlopen):
    manifest = json.loads(safe_fetch("/deployment-manifest.json", opener=opener))
    html = safe_fetch("/", opener=opener)
    validate_manifest(manifest, html)
    return {
        "ok": True,
        "git_commit": manifest["git_commit"],
        "artifact_sha256": manifest["artifact_sha256"],
        "explanation": "Live public HTML matches public build manifest; AWS account identity and build logs are not independently attested.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", help="Optional full expected Git commit SHA; omit to inspect deployed version")
    args = parser.parse_args()
    report = probe()
    if args.expected_commit:
        if report["git_commit"] != args.expected_commit:
            raise SystemExit("Live deployed Git commit does not equal expected commit; check Amplify build/deploy history")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
