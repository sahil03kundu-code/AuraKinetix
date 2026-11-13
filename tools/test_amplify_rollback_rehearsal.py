"""Tests for offline Amplify static-artifact rollback rehearsal."""
import tempfile
import unittest
from pathlib import Path

from tools.amplify_rollback_rehearsal import (
    REQUIRED, check_html, create_snapshot, rehearsal, restore_snapshot, verify_snapshot,
)


class RollbackRehearsalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.html"
        self.source.write_bytes(b"\n".join(REQUIRED) + b"\n")

    def test_round_trip_identical(self):
        snapshot = self.root / "snapshot"
        manifest = create_snapshot(self.source, snapshot)
        self.assertEqual(verify_snapshot(snapshot), manifest)
        destination = self.root / "restored/index.html"
        result = restore_snapshot(snapshot, destination)
        self.assertEqual(destination.read_bytes(), self.source.read_bytes())
        self.assertEqual(result["sha256"], manifest["sha256"])

    def test_corrupted_known_good_refused(self):
        snapshot = self.root / "snapshot"
        create_snapshot(self.source, snapshot)
        (snapshot / "index.html").write_bytes(self.source.read_bytes() + b"tampering")
        destination = self.root / "dist/index.html"
        with self.assertRaisesRegex(ValueError, "manifest"):
            restore_snapshot(snapshot, destination)
        self.assertFalse(destination.exists())

    def test_broken_new_candidate_rejected(self):
        with self.assertRaisesRegex(ValueError, "required markers"):
            check_html(b"<html>broken</html>")
        with self.assertRaisesRegex(ValueError, "empty or too large"):
            check_html(b"")

    def test_rehearsal_reports_offline_only(self):
        report = rehearsal(self.source)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["negative_candidate_rejected"])
        self.assertTrue(report["restoration_identical"])
        self.assertIn("Offline artifact", report["limitations"])

    def test_missing_or_wrong_manifest_rejected(self):
        snapshot = self.root / "snapshot"
        create_snapshot(self.source, snapshot)
        (snapshot / "manifest.json").write_text('{"schema_version": 2}')
        with self.assertRaises(ValueError):
            verify_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
