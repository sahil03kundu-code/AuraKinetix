"""Offline tests for exact-commit AWS Amplify build manifest."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_amplify_manifest import build_manifest, validate_manifest, write_manifest


class AmplifyManifestTests(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 40
        self.html = b'<!doctype html><html><body>Sense sooner. Recover smarter.</body></html>'

    def test_build_and_validate(self):
        m = build_manifest(self.html, self.sha)
        self.assertEqual(m["git_commit"], self.sha)
        self.assertEqual(len(m["artifact_sha256"]), 64)
        self.assertTrue(validate_manifest(m, self.html))

    def test_reject_invalid_commit(self):
        with self.assertRaisesRegex(ValueError, "commit"):
            build_manifest(self.html, "main")

    def test_reject_missing_showcase(self):
        with self.assertRaisesRegex(ValueError, "showcase"):
            build_manifest(b"wrong contents", self.sha)

    def test_tampered_artifact_must_fail(self):
        manifest = build_manifest(self.html, self.sha)
        with self.assertRaisesRegex(ValueError, "Content hash"):
            validate_manifest(manifest, self.html + b"changed")

    def test_reject_unexpected_manifest_fields(self):
        manifest = build_manifest(self.html, self.sha)
        manifest["secret"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_manifest(manifest, self.html)

    def test_generate_manifest_file_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = root / "dist/index.html"
            html.parent.mkdir()
            html.write_bytes(self.html)
            manifest_path = root / "dist/deployment-manifest.json"
            written = write_manifest(html, manifest_path, self.sha)
            data = json.loads(manifest_path.read_text())
            self.assertEqual(data, written)
            self.assertTrue(validate_manifest(data, html.read_bytes()))
            self.assertEqual(list(root.rglob("*.json")), [manifest_path])

    def test_no_sensitive_aws_account_info_is_in_manifest(self):
        m = build_manifest(self.html, self.sha)
        encoded = json.dumps(m).lower()
        self.assertNotIn("aws_access_key", encoded)
        self.assertNotIn("account_id", encoded)
        self.assertNotIn("secret", encoded)


if __name__ == "__main__":
    unittest.main()
