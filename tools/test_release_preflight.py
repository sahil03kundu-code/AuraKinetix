"""No-network release packaging and safety regression tests."""
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from tools.release_preflight import package, resolve_version


class ReleasePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "versions" / "v4.2"
        (self.folder / "aurasync").mkdir(parents=True)
        (self.folder / "aurasync" / "__init__.py").write_text("# aura\n")
        (self.folder / "aurasync" / "physics.py").write_text("VALUE = 42\n")
        (self.folder / "checkpoints").mkdir()
        (self.folder / "checkpoints" / "model.py").write_text("excluded")
        (self.folder / "data").mkdir()
        (self.folder / "data" / "capture.txt").write_text("excluded")
        (self.folder / ".env").write_text("SECRET=hidden")
        (self.folder / "README.md").write_text("release notes")

    def test_exact_release_tag_maps_to_correct_version(self):
        self.assertEqual(resolve_version(self.root, "v4.2.0"), self.folder)
        for bad in ("v4.2", "v4.2.0-rc1", "main", "../../x", "v4.2.0/"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_version(self.root, bad)

    def test_zip_excludes_checkpoints_data_and_hidden_files(self):
        output = self.root / "dist" / "release.zip"
        info = package(self.root, "v4.2.0", output)
        self.assertTrue(output.exists())
        self.assertEqual(len(info["sha256"]), 64)
        with ZipFile(output) as archive:
            names = archive.namelist()
            self.assertIn("versions/v4.2/aurasync/physics.py", names)
            self.assertIn("versions/v4.2/README.md", names)
            self.assertNotIn("versions/v4.2/checkpoints/model.py", names)
            self.assertNotIn("versions/v4.2/data/capture.txt", names)
            self.assertNotIn("versions/v4.2/.env", names)

    def test_archives_are_byte_identical_across_repeated_builds(self):
        one = package(self.root, "v4.2.0", self.root / "first.zip")
        two = package(self.root, "v4.2.0", self.root / "second.zip")
        self.assertEqual(one["sha256"], two["sha256"])

    def test_missing_version_fails_without_creating_zip(self):
        output = self.root / "missing.zip"
        with self.assertRaises(ValueError):
            package(self.root, "v3.1.0", output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
