"""Atomic file writes for translation writeback and batch artifacts.

Pattern matches the RAG store: write to a same-directory temporary file,
flush + fsync, then ``os.replace`` so readers never observe a truncated file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Callable, Iterable, TextIO


def file_sha256(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str, *, encoding: str = "utf-8") -> str:
    return hashlib.sha256(text.encode(encoding)).hexdigest()


def atomic_write(
    path: str | os.PathLike[str],
    writer: Callable[[TextIO], None],
    *,
    encoding: str = "utf-8",
    newline: str | None = "\n",
) -> None:
    """Write text via *writer* and replace *path* atomically.

    Default ``newline='\\n'`` keeps JSON/JSONL bytes stable across platforms so
    checksums match the in-memory content used when hashing downloads.
    """
    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        # Close via context manager before os.replace (required on Windows).
        with os.fdopen(fd, "w", encoding=encoding, newline=newline) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = "\n",
) -> None:
    def write(handle: TextIO) -> None:
        handle.write(text)

    atomic_write(path, write, encoding=encoding, newline=newline)


def atomic_write_lines(
    path: str | os.PathLike[str],
    lines: Iterable[str],
    *,
    encoding: str = "utf-8",
    newline: str | None = "\n",
) -> None:
    def write(handle: TextIO) -> None:
        handle.writelines(lines)

    atomic_write(path, write, encoding=encoding, newline=newline)


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Any,
    *,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: int | None = 2,
    newline: str | None = "\n",
    **dump_kwargs: Any,
) -> None:
    def write(handle: TextIO) -> None:
        json.dump(
            payload,
            handle,
            ensure_ascii=ensure_ascii,
            indent=indent,
            **dump_kwargs,
        )

    atomic_write(path, write, encoding=encoding, newline=newline)


def atomic_write_jsonl(
    path: str | os.PathLike[str],
    rows: Iterable[Any],
    *,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    newline: str | None = "\n",
) -> None:
    def write(handle: TextIO) -> None:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=ensure_ascii) + "\n")

    atomic_write(path, write, encoding=encoding, newline=newline)


def is_complete_jsonl(path: str | os.PathLike[str], *, encoding: str = "utf-8") -> bool:
    """Return True when *path* is a non-empty JSONL file with parseable lines."""
    target = os.fspath(path)
    if not os.path.isfile(target) or os.path.getsize(target) <= 0:
        return False
    try:
        with open(target, "r", encoding=encoding) as handle:
            saw_line = False
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                saw_line = True
                json.loads(line)
            return saw_line
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def result_artifact_is_complete(
    path: str | os.PathLike[str],
    expected_sha256: str | None = None,
    *,
    encoding: str = "utf-8",
) -> bool:
    """Validate a downloaded results.jsonl artifact.

    When *expected_sha256* is known (from a previous successful download), require
    an exact content match. Otherwise require non-empty, parseable JSONL so a
    truncated mid-write file is not treated as already downloaded.
    """
    target = os.fspath(path)
    if not os.path.isfile(target) or os.path.getsize(target) <= 0:
        return False
    if expected_sha256:
        try:
            return file_sha256(target) == str(expected_sha256).strip().lower()
        except OSError:
            return False
    return is_complete_jsonl(target, encoding=encoding)
