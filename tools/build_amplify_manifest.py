"""Generate and validate the build provenance for the static AWS Amplify demo.

No AWS credentials, IAM calls, or deployment APIs required. The manifest is
intentionally public: no environment variables, account IDs or tokens stored.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
REPO = "sahil03kundu-code/AuraKinetix"
SOURCE = "aws/demo-site/index.html"
SCHEMA = 1


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build_manifest(html, commit_sha):
    if not SHA_PATTERN.fullmatch(commit_sha):
        raise ValueError("Invalid full Git commit SHA")
    if b"Sense sooner." not in html or b"Recover smarter." not in html:
        raise ValueError("Expected public showcase content not found")
    return {
        "schema_version": SCHEMA,
        "repository": REPO,
        "git_commit": commit_sha,
        "artifact": "index.html",
        "artifact_sha256": sha256(html),
        "source_path": SOURCE,
        "source_only_showcase": True,
        "limitations": "Build-artifact source identity only; not independently verified AWS deployment identity or simulation evidence.",
    }


def validate_manifest(manifest, html):
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be an object")
    if set(manifest) != {
        "schema_version", "repository", "git_commit", "artifact", "artifact_sha256",
        "source_path", "source_only_showcase", "limitations",
    }:
        raise ValueError("Unexpected manifest schema")
    if manifest["schema_version"] != SCHEMA or manifest["repository"] != REPO:
        raise ValueError("Unexpected repository or schema")
    if manifest["source_path"] != SOURCE or manifest["artifact"] != "index.html":
        raise ValueError("Unexpected source or artifact")
    if manifest["source_only_showcase"] is not True:
        raise ValueError("Unsupported artifact type")
    if not SHA_PATTERN.fullmatch(manifest["git_commit"]):
        raise ValueError("Commit SHA is invalid")
    expected = build_manifest(html, manifest["git_commit"])
    if manifest["artifact_sha256"] != expected["artifact_sha256"]:
        raise ValueError("Content hash mismatch")
    return True


def write_manifest(site_path, output_path, commit_sha):
    html = Path(site_path).read_bytes()
    manifest = build_manifest(html, commit_sha)
    validate_manifest(manifest, html)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", default="dist/index.html")
    parser.add_argument("--output", default="dist/deployment-manifest.json")
    parser.add_argument("--commit")
    args = parser.parse_args()
    commit = args.commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = write_manifest(args.html, args.output, commit)
    print("Generated static showcase provenance for Git commit", manifest["git_commit"])


if __name__ == "__main__":
    main()
