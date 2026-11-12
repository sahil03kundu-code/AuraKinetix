"""Make a deterministic, source-only AuraKinetix release archive.

No checkpoint weights, training data, result directories, credentials or
hidden files are allowed. This runs without AWS or installed ML dependencies.
"""
import argparse
import hashlib
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


VALID_TAG = re.compile(r"^v(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)$")
SOURCE_SUFFIXES = {".py", ".md", ".txt", ".sh", ".yaml", ".yml", ".toml"}
BLOCKED_DIRS = {
    "checkpoints", "results", "results_phase1", "data", "datasets",
    ".git", ".venv", "venv", "__pycache__", "wandb", "logs", "cache",
}
BLOCKED_NAMES = {".env", "id_rsa", "id_ed25519", "credentials", "secrets"}
MAX_FILE_BYTES = 2 * 1024 * 1024


def resolve_version(root, tag):
    match = VALID_TAG.fullmatch(tag)
    if not match:
        raise ValueError("release tags must have exact form vMAJOR.MINOR.PATCH")
    name = f"v{match.group('major')}.{match.group('minor')}"
    version_dir = root / "versions" / name
    if not version_dir.is_dir():
        raise ValueError(f"version directory not found: {version_dir}")
    return version_dir


def allowed_files(root, version_dir):
    files = []
    for path in sorted(version_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks not permitted in release: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(version_dir)
        if any(part.lower() in BLOCKED_DIRS or part.startswith(".") for part in relative.parts[:-1]):
            continue
        if (path.name.lower() in BLOCKED_NAMES or path.name.startswith(".")
                or path.suffix.lower() not in SOURCE_SUFFIXES):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"oversized source file needs review: {path}")
        files.append(path)
    if not files or not any(p.name == "__init__.py" for p in files):
        raise ValueError("release has no verified Python source package")
    return files


def package(root, tag, output):
    root = Path(root).resolve()
    version_dir = resolve_version(root, tag)
    output = Path(output).resolve()
    files = allowed_files(root, version_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            member = path.relative_to(root).as_posix()
            entry = ZipInfo(member, date_time=(2020, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, path.read_bytes(), compress_type=ZIP_DEFLATED)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return dict(path=str(output), entries=len(files), sha256=digest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(package(args.root, args.tag, args.output))


if __name__ == "__main__":
    main()
