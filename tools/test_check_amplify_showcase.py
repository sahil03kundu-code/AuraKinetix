"""Network-free tests for the public Amplify HTTP monitoring contract."""
import io
import unittest
from unittest.mock import patch

from tools.check_amplify_showcase import EXPECTED_MARKERS, URL, fetch_once, run_probe


class FakeResponse:
    def __init__(self, body=None, status=200, url=URL, content_type="text/html; charset=utf-8"):
        self.body = (body if body is not None else " ".join(EXPECTED_MARKERS)).encode()
        self.status = status
        self.url = url
        self.headers = {"Content-Type": content_type}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def geturl(self):
        return self.url
    def getcode(self):
        return self.status
    def read(self, limit):
        return self.body[:limit]


def opener(response):
    return lambda request, timeout: response


class AmplifyPublicProbeTests(unittest.TestCase):
    def test_success_checks_content_and_host(self):
        output = fetch_once(opener=opener(FakeResponse()))
        self.assertTrue(output["ok"])
        self.assertEqual(output["status_code"], 200)
        self.assertEqual(output["missing_marker_count"], 0)
        self.assertGreaterEqual(output["duration_ms"], 0)

    def test_missing_content_fails_and_does_not_echo_site_body(self):
        body = "Internal data that must not be in the report"
        output = fetch_once(opener=opener(FakeResponse(body=body)))
        self.assertFalse(output["ok"])
        self.assertGreater(output["missing_marker_count"], 0)
        self.assertNotIn(body, str(output))

    def test_wrong_content_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "HTML"):
            fetch_once(opener=opener(FakeResponse(content_type="application/json")))

    def test_redirect_to_other_host_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "redirect"):
            fetch_once(opener=opener(FakeResponse(url="https://example.com/")))

    def test_enforces_endpoint_allowlist(self):
        with self.assertRaisesRegex(ValueError, "approved"):
            fetch_once(url="http://main.d2aek34w544oam.amplifyapp.com/", opener=opener(FakeResponse()))
        with self.assertRaisesRegex(ValueError, "approved"):
            fetch_once(url="https://untrusted.example/", opener=opener(FakeResponse()))

    def test_retries_and_reports_success_without_leaking_response(self):
        failures = iter([FakeResponse(body="error"), FakeResponse()])
        def fetch(request, timeout):
            return next(failures)
        result = run_probe(attempts=2, delay=0, opener=fetch, sleep=lambda _: None)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertFalse(result["attempts"][0]["ok"])

    def test_both_failed_results_are_not_reported_as_success(self):
        result = run_probe(
            attempts=2, delay=0,
            opener=opener(FakeResponse(body="unexpected")), sleep=lambda _: None,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["attempts"]), 2)

    def test_invalid_attempts_rejected(self):
        with self.assertRaises(ValueError):
            run_probe(attempts=0)

    def test_no_live_network_is_ever_needed_for_tests(self):
        with patch("tools.check_amplify_showcase.urlopen", side_effect=AssertionError("network access")):
            self.assertTrue(fetch_once(opener=opener(FakeResponse()))["ok"])


if __name__ == "__main__":
    unittest.main()
