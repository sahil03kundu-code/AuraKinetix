"""Conservative, dependency-free source security preflight for pre-AWS CI.

This is a guardrail, not a substitute for secret-scanning services, dependency
vulnerability auditing, IAM review or manual threat modelling.
"""
import argparse
import json
import re
from pathlib import Path

SCANNED_SUFFIXES = {".py", ".sh", ".ps1", ".md", ".txt", ".yaml", ".yml", ".json",
                    ".toml", ".ini", ".cfg", ".env"}
IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
                ".mypy_cache", ".tox", "dist", "build"}
# Construct sensitive marker strings in parts so tests and this script don't
# accidentally resemble a real credential in source scans.
AWS_KEY_PATTERN = re.compile(r"(?<![A-Z0-9])" + "AK" + "IA" + r"[0-9A-Z]{16}(?![A-Z0-9])")
GITHUB_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}")
FINE_GITHUB_TOKEN_PATTERN = re.compile("github" + r"_pat_[A-Za-z0-9_]{30,}")
PRIVATE_KEY_PATTERN = re.compile("-----BEGIN " + "(?:RSA |EC |OPENSSH |DSA )?" +
                                 "PRIVATE" + " KEY-----")
AWS_SECRET_PATTERN = re.compile(
    r"(?i)\baws_secret_access_key\s*[:=]\s*['\"]?([A-Za-z0-9/+]{40})\b"
)
CREDENTIAL_PATTERNS = {
    "aws_access_key": AWS_KEY_PATTERN,
    "github_token": GITHUB_TOKEN_PATTERN,
    "github_fine_grained_token": FINE_GITHUB_TOKEN_PATTERN,
    "private_key": PRIVATE_KEY_PATTERN,
    "aws_secret_access_key": AWS_SECRET_PATTERN,
}
DANGEROUS_WORKFLOW_PATTERNS = {
    "pull_request_target": re.compile(r"(?m)^\s*pull_request_target\s*:"),
    "workflow_run": re.compile(r"(?m)^\s*workflow_run\s*:"),
    "write_all": re.compile(r"(?m)^\s*permissions\s*:\s*write-all\s*(?:#.*)?$"),
}


def scan_text(path, text):
    """Return only finding names and locations; never print secret material."""
    findings = []
    for name, pattern in CREDENTIAL_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({
                "file": str(path),
                "line": text.count("\n", 0, match.start()) + 1,
                "rule": name,
            })
    if str(path).replace("\\", "/").startswith(".github/workflows/"):
        for name, pattern in DANGEROUS_WORKFLOW_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({
                    "file": str(path),
                    "line": text.count("\n", 0, match.start()) + 1,
                    "rule": "workflow_" + name,
                })
    return findings


def scan_repository(root):
    root = Path(root).resolve()
    findings = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root)
        if path.name == ".env" or path.name.startswith(".env.") and path.name not in (".env.example", ".env.sample"):
            findings.append({"file":relative.as_posix(), "line": 1, "rule": "committed_env"})
        if path.suffix.lower() not in SCANNED_SUFFIXES and path.name not in (".env",):
            continue
        # Skip binary / unusually large generated files. They need their own inventory.
        if path.stat().st_size > 2_000_000:
            continue
        try:
            data = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            continue
        scanned += 1
        findings.extend(scan_text(relative.as_posix(), data))
    return {"schema_version": 1, "files_scanned": scanned,
            "status": "fail" if findings else "pass", "findings": findings,
            "limitations": "Pattern-based guardrail; not a dependency audit or proof that no secrets exist."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="security-preflight.json")
    args = parser.parse_args()
    report = scan_repository(args.root)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Security preflight:", report["status"], "-", report["files_scanned"], "files")
    for item in report["findings"]:
        print(f'{item["file"]}:{item["line"]} [{item["rule"]}]')
    if report["findings"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
