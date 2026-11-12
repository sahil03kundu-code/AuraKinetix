"""Offline release archive integrity regression tests."""
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from tools.release_preflight import package
from tools.verify_release_archive import verify


class ReleaseArchiveVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "versions/v4.2/aurasync"
        source.mkdir(parents=True)
        (source / "__init__.py").write_text("# package\n")
        (source / "physics.py").write_text("x = 42\n")
        (self.root / "versions/v4.2/README.md").write_text("Readme")
        self.zipfile = self.root / "dist/release.zip"
        package(self.root, "v4.2.0", self.zipfile)

    def test_valid_source_only_zip_passes(self):
        result = verify(self.zipfile, "v4.2.0")
        self.assertEqual(len(result["sha256"]), 64)
        self.assertEqual(result["entries"], 3)
        self.assertTrue(result["source_only"])

    def test_wrong_tag_rejected(self):
        with self.assertRaises(ValueError):
            verify(self.zipfile, "v4.1.0")
        with self.assertRaises(ValueError):
            verify(self.zipfile, "v4.2")

    def test_traversal_and_hidden_secrets_rejected(self):
        for path in (
            "versions/v4.2/../credentials.txt",
            "versions/v4.2/.env",
            "versions/v4.2/checkpoints/model.py",
            "versions/v4.2/aurasync/private.pem",
        ):
            with self.subTest(path=path):
                with ZipFile(self.zipfile, "a") as z:
                    z.writestr(path, "unsafe")
                with self.assertRaises(ValueError):
                    verify(self.zipfile, "v4.2.0")
                self.zipfile.unlink()
                package(self.root, "v4.2.0", self.zipfile)

    def test_duplicate_path_rejected(self):
        with ZipFile(self.zipfile, "a") as z:
            z.writestr("versions/v4.2/aurasync/physics.py", "duplicate")
        with self.assertRaises(ValueError):
            verify(self.zipfile, "v4.2.0")

    def test_broken_archive_rejected(self):
        self.zipfile.write_bytes(b"not a zip")
        with self.assertRaises(ValueError):
            verify(self.zipfile, "v4.2.0")


if __name__ == "__main__":
    unittest.main()
