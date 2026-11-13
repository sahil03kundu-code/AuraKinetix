"""Offline, byte-for-byte rehearsal of restoring a known-good static site artifact.

This never invokes AWS APIs or alters the live Amplify deployment.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

MAX_SIZE = 512_000
REQUIRED = (
    b"<html lang=\"en\">",
    b"Sense sooner.",
    b"Recover smarter.",
    b"This is a public project showcase, not a live control system.",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_html(data):
    if not data or len(data) > MAX_SIZE:
        raise ValueError("HTML is empty or too large")
    if any(marker not in data for marker in REQUIRED):
        raise ValueError("Static showcase content missing required markers")


def create_snapshot(source, snapshot_dir):
    """Preserve an approved static site as immutable-content+SHA256 manifest."""
    source = Path(source)
    data = source.read_bytes()
    check_html(data)
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "index.html").write_bytes(data)
    manifest = {
        "schema_version": 1,
        "artifact_name": "index.html",
        "sha256": digest(data),
        "bytes": len(data),
        "provenance": "local CI snapshot, not confirmed live Amplify build",
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_snapshot(snapshot_dir):
    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("artifact_name") != "index.html":
        raise ValueError("Invalid snapshot manifest")
    if set(manifest) != {"schema_version", "artifact_name", "sha256", "bytes", "provenance"}:
        raise ValueError("Unexpected manifest content")
    data = (snapshot_dir / "index.html").read_bytes()
    check_html(data)
    if digest(data) != manifest["sha256"] or len(data) != manifest["bytes"]:
        raise ValueError("Snapshot content does not match manifest")
    return manifest


def restore_snapshot(snapshot_dir, destination):
    """Verify snapshot *before* writing; never call an AWS deployment API."""
    manifest = verify_snapshot(snapshot_dir)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (Path(snapshot_dir) / "index.html").read_bytes()
    destination.write_bytes(data)
    if digest(destination.read_bytes()) != manifest["sha256"]:
        raise ValueError("Restoration differs from known-good snapshot")
    return manifest


def rehearsal(source):
    """Verify corrupted candidate is rejected; restore verified snapshot."""
    source = Path(source)
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        snapshot_dir = root / "known-good"
        manifest = create_snapshot(source, snapshot_dir)
        candidate = root / "candidate" / "index.html"
        candidate.parent.mkdir()
        candidate.write_text("<html>broken candidate</html>", encoding="utf-8")
        rejected = False
        try:
            check_html(candidate.read_bytes())
        except ValueError:
            rejected = True
        if not rejected:
            raise AssertionError("Broken candidate must fail preflight")
        restore_snapshot(snapshot_dir, candidate)
        if candidate.read_bytes() != source.read_bytes():
            raise AssertionError("Rehearsal did not restore identical bytes")
        return {
            "schema_version": 1,
            "status": "pass",
            "negative_candidate_rejected": True,
            "restoration_identical": True,
            "sha256": manifest["sha256"],
            "limitations": "Offline artifact restoration only; does not test AWS Amplify rollback or live availability.",
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="aws/demo-site/index.html")
    p.add_argument("--report", default="rollback-rehearsal.json")
    args = p.parse_args()
    result = rehearsal(args.source)
    Path(args.report).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Offline rollback rehearsal:", result["status"])


if __name__ == "__main__":
    main()
