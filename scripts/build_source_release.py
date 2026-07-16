"""Build a deterministic source ZIP for a tagged release.

Historical refs may contain Git LFS pointers; those are expanded when present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
VERSION_RE = re.compile(rb'^__version__\s*=\s*["\']([^"\']+)["\']\s*$', re.MULTILINE)
LFS_OID_RE = re.compile(rb"^oid sha256:([0-9a-f]{64})$", re.MULTILINE)
LFS_SIZE_RE = re.compile(rb"^size ([0-9]+)$", re.MULTILINE)


@dataclass(frozen=True)
class TreeEntry:
    mode: str
    object_id: str
    path: str


def _git(*args: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def _tree_entries(ref: str) -> list[TreeEntry]:
    raw = _git("ls-tree", "-r", "-z", "--full-tree", ref)
    entries: list[TreeEntry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type != "blob":
            continue
        entries.append(
            TreeEntry(
                mode=mode,
                object_id=object_id,
                path=raw_path.decode("utf-8", errors="surrogateescape"),
            )
        )
    return sorted(entries, key=lambda item: item.path)


def _parse_lfs_pointer(data: bytes) -> tuple[str, int] | None:
    if not data.startswith(LFS_POINTER_PREFIX):
        return None
    oid_match = LFS_OID_RE.search(data)
    size_match = LFS_SIZE_RE.search(data)
    if oid_match is None or size_match is None:
        raise RuntimeError("Malformed Git LFS pointer")
    return oid_match.group(1).decode("ascii"), int(size_match.group(1))


def _expand_lfs(path: str, pointer: bytes) -> bytes:
    expected = _parse_lfs_pointer(pointer)
    if expected is None:
        return pointer
    expected_oid, expected_size = expected
    data = _git("lfs", "smudge", "--", path, input_bytes=pointer)
    actual_oid = hashlib.sha256(data).hexdigest()
    if len(data) != expected_size or actual_oid != expected_oid:
        raise RuntimeError(
            f"Git LFS object verification failed for {path}: "
            f"expected sha256={expected_oid} size={expected_size}, "
            f"got sha256={actual_oid} size={len(data)}"
        )
    return data


def _version_from_blob(data: bytes) -> str:
    match = VERSION_RE.search(data)
    if match is None:
        raise RuntimeError("project_version.py does not contain __version__")
    return match.group(1).decode("ascii")


def _zip_datetime(epoch: int) -> tuple[int, int, int, int, int, int]:
    # ZIP timestamps cannot represent dates earlier than 1980.
    return time.gmtime(max(epoch, 315532800))[:6]


def _write_entry(
    archive: zipfile.ZipFile,
    archive_path: str,
    data: bytes,
    *,
    timestamp: tuple[int, int, int, int, int, int],
    executable: bool = False,
) -> None:
    info = zipfile.ZipInfo(archive_path, date_time=timestamp)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    permissions = 0o100755 if executable else 0o100644
    info.external_attr = permissions << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_release(ref: str, output_dir: Path) -> tuple[Path, Path, str]:
    commit = _git("rev-parse", f"{ref}^{{commit}}").decode("ascii").strip()
    commit_epoch = int(_git("show", "-s", "--format=%ct", commit).decode("ascii").strip())
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", commit_epoch))
    timestamp = _zip_datetime(epoch)
    entries = _tree_entries(commit)

    version_entry = next((item for item in entries if item.path == "project_version.py"), None)
    if version_entry is None:
        raise RuntimeError("project_version.py is missing from the selected ref")
    version = _version_from_blob(_git("cat-file", "blob", version_entry.object_id))

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"renpy-translation-lab-{version}-source.zip"
    archive_path = output_dir / archive_name
    checksum_path = output_dir / f"{archive_name}.sha256"
    archive_root = f"renpy-translation-lab-{version}"

    metadata = {
        "archive_format": 1,
        "commit": commit,
        "source_ref": ref,
        "version": version,
    }

    with zipfile.ZipFile(archive_path, "w") as archive:
        for entry in entries:
            pointer_or_data = _git("cat-file", "blob", entry.object_id)
            data = _expand_lfs(entry.path, pointer_or_data)
            _write_entry(
                archive,
                f"{archive_root}/{entry.path}",
                data,
                timestamp=timestamp,
                executable=entry.mode == "100755",
            )
        _write_entry(
            archive,
            f"{archive_root}/RELEASE-METADATA.json",
            (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            timestamp=timestamp,
        )

    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    checksum_path.write_text(f"{digest}  {archive_name}\n", encoding="ascii", newline="\n")
    return archive_path, checksum_path, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic source release ZIP.")
    parser.add_argument("--ref", default="HEAD", help="Git ref to package (default: HEAD).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dist" / "release",
        help="Directory for the ZIP and checksum.",
    )
    args = parser.parse_args(argv)
    archive_path, checksum_path, digest = build_release(args.ref, args.output_dir)
    print(f"Source archive: {archive_path}")
    print(f"Checksum file: {checksum_path}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
