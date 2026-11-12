"""Unit tests for pre-AWS source guardrails, using synthetic non-secret fixtures."""
import tempfile
import unittest
from pathlib import Path

from tools.security_preflight import scan_repository, scan_text


class SecurityPreflightTests(unittest.TestCase):
    def test_detects_credential_pattern_without_echoing_secret(self):
        fake = "AK" + "IA" + "A" * 16
        findings = scan_text("example.py", "ACCESS=" + fake + "\n")
        self.assertEqual(findings[0]["rule"], "aws_access_key")
        self.assertNotIn(fake, str(findings))

    def test_detects_private_key_header(self):
        fake = "-----BEGIN " + "PRIVATE" + " KEY-----"
        findings = scan_text("example.pem", fake)
        self.assertEqual(findings[0]["rule"], "private_key")

    def test_blocks_untrusted_workflow_privilege_escalation(self):
        test = "name: wrong\non:\n  pull_request_target:\npermissions: write-all\n"
        finding_names = {f["rule"] for f in scan_text(".github/workflows/test.yml", test)}
        self.assertIn("workflow_pull_request_target", finding_names)
        self.assertIn("workflow_write_all", finding_names)

    def test_safe_workflow_passes(self):
        sample = "on:\n  pull_request:\npermissions:\n  contents: read\n"
        self.assertEqual(scan_text(".github/workflows/test.yml", sample), [])

    def test_env_is_rejected_but_example_is_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".env").write_text("EXAMPLE=not-a-real-secret", encoding="utf-8")
            (root / ".env.example").write_text("EXAMPLE=placeholder", encoding="utf-8")
            report = scan_repository(root)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any(f["rule"] == "committed_env" for f in report["findings"]))
            (root / ".env").unlink()
            self.assertEqual(scan_repository(root)["status"], "pass")

    def test_ignores_generated_and_git_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".git").mkdir()
            (root / ".git" / "secret.py").write_text("AK" + "IA" + "A" * 16)
            (root / "visible.py").write_text("print('hello')", encoding="utf-8")
            report = scan_repository(root)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["files_scanned"], 1)


if __name__ == "__main__":
    unittest.main()
