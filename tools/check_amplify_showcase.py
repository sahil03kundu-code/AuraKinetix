"""Read-only external availability check for the public AWS Amplify showcase.

An HTTP/content probe is not an end-to-end AWS deployment or simulation test.
Uses only stdlib and does not access AWS account APIs or credentials.
"""
import argparse
import json
import ssl
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from pathlib import Path

URL = "https://main.d2aek34w544oam.amplifyapp.com/"
EXPECTED_HOST = "main.d2aek34w544oam.amplifyapp.com"
EXPECTED_MARKERS = (
    "AuraKinetix",
    "Sense sooner.",
    "Recover smarter.",
    "This is a public project showcase, not a live control system.",
)
MAX_READ_BYTES = 512_000


def fetch_once(url=URL, opener=urlopen, timeout=12):
    """Never log page bodies; report only availability and expected markers."""
    if urlsplit(url).scheme != "https" or urlsplit(url).hostname != EXPECTED_HOST:
        raise ValueError("Only the approved AuraKinetix HTTPS endpoint may be probed")
    request = Request(url, headers={"User-Agent": "AuraKinetix-GitHub-CI-Availability/1.0",
                                     "Cache-Control": "no-cache"})
    begin = time.monotonic()
    with opener(request, timeout=timeout) as response:
        effective_url = response.geturl()
        if urlsplit(effective_url).scheme != "https" or urlsplit(effective_url).hostname != EXPECTED_HOST:
            raise ValueError("Unexpected redirect away from the approved host")
        status = response.getcode()
        data = response.read(MAX_READ_BYTES + 1)
        content_type = response.headers.get("Content-Type", "")
    elapsed_ms = round((time.monotonic() - begin) * 1000, 2)
    if len(data) > MAX_READ_BYTES:
        raise ValueError("Response exceeds the size limit")
    if status != 200:
        raise ValueError(f"Unexpected status {status}")
    if "text/html" not in content_type.lower():
        raise ValueError("Expected HTML content-type")
    body = data.decode("utf-8", errors="replace")
    missing = [marker for marker in EXPECTED_MARKERS if marker not in body]
    return {
        "status_code": status, "duration_ms": elapsed_ms,
        "content_type": content_type.split(";")[0], "bytes": len(data),
        "required_markers_found": not missing,
        "missing_marker_count": len(missing),
        "ok": not missing,
    }


def run_probe(*, attempts=2, delay=4, opener=urlopen, sleep=time.sleep):
    if not 1 <= attempts <= 3:
        raise ValueError("Attempts must be 1..3")
    report = {
        "schema_version": 1,
        "target": URL,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "read_only_public_http_content_probe",
        "attempts": [],
        "limitations": "Sampled external availability only; not an uptime SLA, AWS API verification, backend check, or simulation performance evidence.",
    }
    for n in range(attempts):
        try:
            sample = fetch_once(opener=opener)
            report["attempts"].append(sample)
            if sample["ok"]:
                report["ok"] = True
                return report
        except (ValueError, HTTPError, URLError, OSError, TimeoutError, ssl.SSLError) as exc:
            report["attempts"].append({
                "ok": False, "error_type": type(exc).__name__,
                "message": "Public site check failed (response details deliberately omitted)"
            })
        if n + 1 < attempts:
            sleep(delay)
    report["ok"] = False
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="amplify-public-probe.json")
    p.add_argument("--attempts", type=int, default=2)
    args = p.parse_args()
    result = run_probe(attempts=args.attempts)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"AuraKinetix public HTTP check: {'PASS' if result['ok'] else 'FAIL'}")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
