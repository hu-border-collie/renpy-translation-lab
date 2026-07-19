"""Atomic file writes for translation writeback and batch artifacts.

Pattern matches the RAG store: write to a same-directory temporary file,
flush + fsync, then ``os.replace`` so readers never observe a truncated file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
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
    target_mode = None
    try:
        target_mode = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        pass
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
        if target_mode is not None:
            os.chmod(tmp_path, target_mode)
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


class AtomicWriteTransactionError(RuntimeError):
    """Raised when a multi-file atomic-write transaction cannot be recovered."""


def _remove_if_present(path: str) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _stage_lines(
    path: str,
    lines: Iterable[str],
    *,
    encoding: str,
    newline: str | None,
) -> str:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    target_mode = None
    try:
        target_mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        pass
    fd, staged_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".txn.tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline=newline) as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        if target_mode is not None:
            os.chmod(staged_path, target_mode)
        return staged_path
    except Exception:
        _remove_if_present(staged_path)
        raise


def _backup_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    directory = os.path.dirname(path) or "."
    fd, backup_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".txn.bak",
        dir=directory,
    )
    os.close(fd)
    try:
        shutil.copy2(path, backup_path)
        with open(backup_path, "rb+") as handle:
            os.fsync(handle.fileno())
        return backup_path
    except Exception:
        _remove_if_present(backup_path)
        raise


def _cleanup_transaction_entries(entries: Iterable[dict[str, Any]]) -> None:
    for entry in entries:
        _remove_if_present(str(entry.get("staged_path") or ""))
        _remove_if_present(str(entry.get("backup_path") or ""))


def _validate_transaction_journal(
    payload: Any,
    journal: str,
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise AtomicWriteTransactionError(
            f"Invalid writeback transaction journal: {journal}"
        )

    state = payload.get("state")
    entries = payload.get("entries")
    if state not in {"prepared", "committed"} or not isinstance(entries, list):
        raise AtomicWriteTransactionError(
            f"Invalid writeback transaction journal: {journal}"
        )

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AtomicWriteTransactionError(
                f"Invalid entry {index} in writeback transaction journal: {journal}"
            )
        target = entry.get("target")
        staged_path = entry.get("staged_path")
        backup_path = entry.get("backup_path")
        existed = entry.get("existed")
        if (
            not isinstance(target, str)
            or not target
            or not isinstance(staged_path, str)
            or not staged_path
            or not isinstance(backup_path, str)
            or not isinstance(existed, bool)
            or (existed and not backup_path)
        ):
            raise AtomicWriteTransactionError(
                f"Invalid entry {index} in writeback transaction journal: {journal}"
            )

    return state, entries


def _restore_backup_copy(backup_path: str, target: str) -> None:
    directory = os.path.dirname(target) or "."
    fd, restore_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.",
        suffix=".txn.restore",
        dir=directory,
    )
    os.close(fd)
    try:
        shutil.copy2(backup_path, restore_path)
        with open(restore_path, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(restore_path, target)
    except Exception:
        _remove_if_present(restore_path)
        raise


def recover_atomic_write_transaction(
    journal_path: str | os.PathLike[str],
) -> bool:
    """Recover an interrupted multi-file write transaction.

    Prepared transactions are rolled back. Committed transactions only need
    leftover temporary files removed. Returns True when a journal was found.
    """
    journal = os.path.abspath(os.fspath(journal_path))
    if not os.path.isfile(journal):
        return False
    try:
        with open(journal, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AtomicWriteTransactionError(
            f"Could not read writeback transaction journal {journal}: {exc}"
        ) from exc

    state, entries = _validate_transaction_journal(payload, journal)

    if state == "prepared":
        for entry in reversed(entries):
            target = entry["target"]
            staged_path = entry["staged_path"]
            backup_path = entry["backup_path"]
            existed = entry["existed"]
            # os.replace consumes the staged path. Its absence therefore means
            # this target was already committed before interruption.
            if staged_path and os.path.exists(staged_path):
                continue
            if existed:
                if not backup_path or not os.path.isfile(backup_path):
                    raise AtomicWriteTransactionError(
                        f"Missing rollback backup for {target}: {backup_path or '(none)'}"
                    )
                _restore_backup_copy(backup_path, target)
            else:
                _remove_if_present(target)

    _cleanup_transaction_entries(entries)
    _remove_if_present(journal)
    return True


def atomic_write_many_lines(
    writes: Iterable[tuple[str | os.PathLike[str], Iterable[str]]],
    *,
    journal_path: str | os.PathLike[str],
    encoding: str = "utf-8",
    newline: str | None = "\n",
) -> None:
    """Replace multiple text files as one recoverable writeback transaction.

    All target contents are staged and backed up before the first replacement.
    A replacement failure rolls back every already-replaced target. A surviving
    ``prepared`` journal is likewise rolled back on the next invocation.
    """
    journal = os.path.abspath(os.fspath(journal_path))
    recover_atomic_write_transaction(journal)
    normalized_writes = [
        (os.path.abspath(os.fspath(path)), lines)
        for path, lines in writes
    ]
    if not normalized_writes:
        return

    entries: list[dict[str, Any]] = []
    journal_written = False
    try:
        for target, lines in normalized_writes:
            existed = os.path.exists(target)
            staged_path = _stage_lines(
                target,
                lines,
                encoding=encoding,
                newline=newline,
            )
            try:
                backup_path = _backup_file(target)
            except Exception:
                _remove_if_present(staged_path)
                raise
            entries.append(
                {
                    "target": target,
                    "staged_path": staged_path,
                    "backup_path": backup_path,
                    "existed": existed,
                }
            )

        payload = {"version": 1, "state": "prepared", "entries": entries}
        atomic_write_json(journal, payload, ensure_ascii=False, indent=2)
        journal_written = True

        for entry in entries:
            os.replace(entry["staged_path"], entry["target"])

        payload["state"] = "committed"
        atomic_write_json(journal, payload, ensure_ascii=False, indent=2)
    except Exception:
        if journal_written:
            recover_atomic_write_transaction(journal)
        else:
            _cleanup_transaction_entries(entries)
        raise

    _cleanup_transaction_entries(entries)
    _remove_if_present(journal)


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
