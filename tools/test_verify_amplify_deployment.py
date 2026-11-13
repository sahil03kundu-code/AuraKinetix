"""Network-free verification tests for public hosted provenance."""
import json
import unittest

from tools.build_amplify_manifest import build_manifest
from tools.verify_amplify_deployment import SITE, probe, safe_fetch

HTML = b'<!doctype html><html><body>Sense sooner. Recover smarter.</body></html>'
SHA = "a" * 40


class Response:
    def __init__(self, data, url, status=200):
        self.data, self.url, self.status = data, url, status
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def geturl(self):
        return self.url
    def getcode(self):
        return self.status
    def read(self, limit):
        return self.data[:limit]


def opener_factory(html=HTML, manifest=None):
    manifest = manifest if manifest is not None else build_manifest(html, SHA)
    def opener(request, timeout):
        path = request.full_url.removeprefix(SITE)
        if path == "/deployment-manifest.json":
            return Response(json.dumps(manifest).encode(), request.full_url)
        return Response(html, request.full_url)
    return opener


class PublicProvenanceTests(unittest.TestCase):
    def test_checks_live_html_hash(self):
        out = probe(opener=opener_factory())
        self.assertTrue(out["ok"])
        self.assertEqual(out["git_commit"], SHA)

    def test_rejects_tampered_live_html(self):
        manifest = build_manifest(HTML, SHA)
        with self.assertRaisesRegex(ValueError, "Content hash mismatch"):
            probe(opener=opener_factory(html=HTML+b"changed", manifest=manifest))

    def test_rejects_redirect_away_from_host(self):
        def wrong(request, timeout):
            return Response(HTML, "https://example.org/")
        with self.assertRaisesRegex(ValueError, "unapproved host"):
            safe_fetch("/", opener=wrong)

    def test_does_not_request_unapproved_paths(self):
        with self.assertRaises(ValueError):
            safe_fetch("/secret", opener=opener_factory())

    def test_non_200_fails(self):
        def bad(request, timeout):
            return Response(HTML, request.full_url, 404)
        with self.assertRaisesRegex(ValueError, "HTTP"):
            safe_fetch("/", opener=bad)


if __name__ == "__main__":
    unittest.main()
