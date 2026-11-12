"""Verify an AuraKinetix release archive and generate a SHA-256 sidecar.

This checks archive structure and source-only policy without requiring ML
dependencies, credentials, AWS, or publishing a release.
"""
import argparse
import hashlib
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from tools.release_preflight import (
    BLOCKED_DIRS, BLOCKED_NAMES, MAX_FILE_BYTES, SOURCE_SUFFIXES,
    VALID_TAG,
)


def verify(archive_path, tag):
    archive_path = Path(archive_path)
    if not VALID_TAG.fullmatch(tag):
        raise ValueError("invalid release tag")
    match = VALID_TAG.fullmatch(tag)
    expected_root = f"versions/v{match.group('major')}.{match.group('minor')}/"
    try:
        with ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("corrupt archive entry")
            entries = archive.infolist()
            if not entries:
                raise ValueError("release archive is empty")
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise ValueError("duplicate paths in release archive")
            package_init = False
            for entry in entries:
                name = entry.filename
                parts = name.split("/")
                if (
                    not name.startswith(expected_root)
                    or entry.is_dir()
                    or len(parts) < 3
                    or any(part in ("", ".", "..") for part in parts)
                    or name.startswith("/")
                    or "\\" in name
                ):
                    raise ValueError("unsafe release archive path")
                if any(p.lower() in BLOCKED_DIRS or p.startswith(".") for p in parts[2:-1]):
                    raise ValueError("blocked release directory")
                filename = parts[-1]
                if (
                    filename.lower() in BLOCKED_NAMES
                    or filename.startswith(".")
                    or Path(filename).suffix.lower() not in SOURCE_SUFFIXES
                    or entry.file_size > MAX_FILE_BYTES
                ):
                    raise ValueError("blocked release file")
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("symlink entry prohibited")
                if name == expected_root + "aurasync/__init__.py":
                    package_init = True
            if not package_init:
                raise ValueError("missing aurasync source package")
    except BadZipFile as error:
        raise ValueError("invalid release zip") from error
    return {
        "tag": tag,
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "entries": len(names),
        "source_only": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checksum-output", required=True)
    args = parser.parse_args()
    result = verify(args.archive, args.tag)
    output = Path(args.checksum_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'{result["sha256"]}  {Path(args.archive).name}\n', encoding="utf-8"
    )
    print("Verified source-only release:", result)


if __name__ == "__main__":
    main()
