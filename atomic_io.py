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
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterable, TextIO


def file_sha256(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str, *, encoding: str = "utf-8") -> str:
    return hashlib.sha256(text.encode(encoding)).hexdigest()


class AtomicFileLockTimeoutError(TimeoutError):
    """Raised when a same-directory exclusive file lock cannot be acquired."""


@contextmanager
def exclusive_file_lock(
    lock_path: str | os.PathLike[str],
    *,
    timeout: float = 30.0,
    poll_interval: float = 0.01,
    stale_after: float = 300.0,
):
    """Serialize cooperating writers with an exclusive same-directory lock file.

    The lock uses atomic ``O_EXCL`` creation so it works on Windows and POSIX.
    A token prevents one owner from deleting a replacement lock, and abandoned
    regular lock files are recovered after ``stale_after`` seconds.
    """
    target = os.path.abspath(os.fspath(lock_path))
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    token = uuid.uuid4().hex
    owner = {
        "pid": os.getpid(),
        "token": token,
        "created_at": time.time(),
    }
    deadline = time.monotonic() + max(0.0, float(timeout))
    delay = max(0.001, float(poll_interval))

    while True:
        try:
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                lock_stat = os.lstat(target)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(lock_stat.st_mode):
                raise OSError(f"File lock path is not a regular file: {target}")
            age = max(0.0, time.time() - lock_stat.st_mtime)
            if stale_after >= 0 and age >= float(stale_after):
                try:
                    os.unlink(target)
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise AtomicFileLockTimeoutError(
                    f"Timed out waiting for file lock: {target}"
                )
            time.sleep(delay)
            continue

        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(owner, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.unlink(target)
            except OSError:
                pass
            raise
        break

    try:
        yield owner
    finally:
        try:
            with open(target, "r", encoding="utf-8") as handle:
                current = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            current = {}
        if isinstance(current, dict) and current.get("token") == token:
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass


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


class AtomicWritePreimageConflict(AtomicWriteTransactionError):
    """Raised when a guarded target changed outside the pending transaction."""


# Sentinel used by ``expected_preimages``: a mapping value of ``None`` means
# "the target must not exist"; a SHA-256 string means "the target must match".
_EXPECTED_NOT_CHECKED = object()


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


def _stage_bytes(
    path: str,
    content: bytes,
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
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
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


def _normalize_preimage_hashes(
    expected_preimages: dict[str | os.PathLike[str], str | None] | None,
) -> dict[str, str | None]:
    """Normalize guarded target paths and validate SHA-256 expectations."""

    normalized: dict[str, str | None] = {}
    for raw_path, raw_hash in (expected_preimages or {}).items():
        path = os.path.abspath(os.fspath(raw_path))
        if raw_hash is None:
            normalized[path] = None
            continue
        value = str(raw_hash).strip().lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(
                f"Expected target preimage must be a SHA-256 hex digest or None: {raw_path!r}"
            )
        normalized[path] = value
    return normalized


def _preimage_digest(path: str, backup_path: str, existed: bool) -> str | None:
    if not existed:
        return None
    return file_sha256(backup_path) if backup_path else None


def _make_transaction_entry(
    target: str,
    staged_path: str,
    backup_path: str,
    existed: bool,
    *,
    staged_sha256: str,
    target_preimage_sha256: str | None,
) -> dict[str, Any]:
    return {
        "target": target,
        "staged_path": staged_path,
        "backup_path": backup_path,
        "existed": existed,
        "staged_sha256": staged_sha256,
        "target_preimage_sha256": target_preimage_sha256,
    }


def _verify_target_before_replace(entry: dict[str, Any]) -> None:
    """Fail closed if a target changed after staging/backup."""

    target = entry["target"]
    preimage = entry.get("target_preimage_sha256")
    if entry["existed"]:
        try:
            current = file_sha256(target)
        except OSError as exc:
            raise AtomicWritePreimageConflict(
                f"Target disappeared before transaction replace: {target}"
            ) from exc
        if preimage and current != preimage:
            raise AtomicWritePreimageConflict(
                f"Target changed outside the transaction before replace: {target}"
            )
    elif os.path.lexists(target):
        raise AtomicWritePreimageConflict(
            f"Target was created outside the transaction before replace: {target}"
        )


def _verify_prepared_target_state(entry: dict[str, Any]) -> None:
    """Preflight one prepared entry without mutating any target."""

    staged_sha256 = entry.get("staged_sha256")
    if not staged_sha256:
        # Legacy journal: no content guard available; callers that require
        # strict recovery must reject rather than guess when hashes are absent.
        return
    target = entry["target"]
    staged_path = entry["staged_path"]
    preimage = entry.get("target_preimage_sha256")
    # ``os.replace`` consumes the staged path. Its absence means the replace
    # had already happened before the interruption.
    if not os.path.exists(staged_path):
        try:
            current = file_sha256(target)
        except OSError as exc:
            raise AtomicWriteTransactionError(
                f"Committed target is missing during recovery: {target}"
            ) from exc
        if current != staged_sha256:
            raise AtomicWritePreimageConflict(
                "Committed target changed outside the pending transaction; "
                f"refusing to overwrite it: {target}"
            )
        return
    if entry["existed"]:
        if preimage:
            try:
                current = file_sha256(target)
            except OSError as exc:
                raise AtomicWriteTransactionError(
                    f"Uncommitted target is missing during recovery: {target}"
                ) from exc
            if current != preimage:
                raise AtomicWritePreimageConflict(
                    "Uncommitted target changed outside the pending transaction; "
                    f"refusing to overwrite it: {target}"
                )
    elif os.path.lexists(target):
        raise AtomicWritePreimageConflict(
            "Uncommitted target was created outside the pending transaction; "
            f"refusing to remove it: {target}"
        )


def _cleanup_committed_transaction(
    entries: Iterable[dict[str, Any]],
    journal: str,
) -> None:
    """Best-effort cleanup after a committed journal was persisted.

    The commit is already durable at this point. Leftover temporary files or a
    committed journal are harmless: the next recovery pass will remove them
    without touching committed targets.
    """

    try:
        _cleanup_transaction_entries(entries)
    except Exception:
        pass
    try:
        _remove_if_present(journal)
    except Exception:
        pass


def _run_many_transaction(
    entries: list[dict[str, Any]],
    *,
    journal: str,
    transaction_kind: str,
    metadata: dict[str, Any] | None,
    post_commit_validator: Callable[[], None] | None,
) -> None:
    """Persist one prepared -> committed journal around already-staged entries."""

    payload: dict[str, Any] = {
        "version": 1,
        "transaction_kind": transaction_kind,
        "state": "prepared",
        "entries": entries,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    journal_written = False
    committed = False
    try:
        atomic_write_json(journal, payload, ensure_ascii=False, indent=2)
        journal_written = True

        for entry in entries:
            _verify_target_before_replace(entry)
            os.replace(entry["staged_path"], entry["target"])
        if post_commit_validator is not None:
            post_commit_validator()

        payload["state"] = "committed"
        atomic_write_json(journal, payload, ensure_ascii=False, indent=2)
        committed = True
    except Exception:
        if committed:
            _cleanup_committed_transaction(entries, journal)
            raise
        if journal_written or os.path.lexists(journal):
            try:
                recover_atomic_write_transaction(
                    journal,
                    expected_transaction_kind=transaction_kind,
                    verify_targets=True,
                )
            except Exception as recovery_exc:
                # Leave the journal in place for manual recovery; never claim
                # that a partial transaction was rolled back.
                raise AtomicWriteTransactionError(
                    f"Transaction {transaction_kind!r} failed and could not be "
                    f"safely recovered: {recovery_exc}"
                ) from recovery_exc
        else:
            _cleanup_transaction_entries(entries)
        raise

    _cleanup_committed_transaction(entries, journal)


def _validate_transaction_journal(
    payload: Any,
    journal: str,
) -> tuple[str, str, list[dict[str, Any]]]:
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
    transaction_kind = payload.get("transaction_kind")
    if transaction_kind is None:
        # Journals written before transaction identities were introduced are
        # still recoverable.  The dedicated revision filename is the only
        # legacy signal available; an export journal without an identity must
        # remain untrusted and will be rejected by its export caller.
        transaction_kind = (
            "revision"
            if os.path.basename(journal) == ".revision_writeback_transaction.json"
            else "apply"
        )
    if not isinstance(transaction_kind, str) or not transaction_kind.strip():
        raise AtomicWriteTransactionError(
            f"Invalid transaction identity in writeback journal: {journal}"
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
        staged_sha256 = entry.get("staged_sha256")
        if staged_sha256 is not None and (
            not isinstance(staged_sha256, str)
            or len(staged_sha256) != 64
            or any(char not in "0123456789abcdef" for char in staged_sha256)
        ):
            raise AtomicWriteTransactionError(
                f"Invalid staged digest in entry {index} of writeback journal: {journal}"
            )
        preimage_sha256 = entry.get("target_preimage_sha256")
        if preimage_sha256 is not None and (
            not isinstance(preimage_sha256, str)
            or len(preimage_sha256) != 64
            or any(char not in "0123456789abcdef" for char in preimage_sha256)
        ):
            raise AtomicWriteTransactionError(
                f"Invalid preimage digest in entry {index} of writeback journal: {journal}"
            )

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise AtomicWriteTransactionError(
            f"Invalid metadata in writeback transaction journal: {journal}"
        )

    return state, transaction_kind, entries


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


def read_atomic_write_transaction_journal(
    journal_path: str | os.PathLike[str],
    *,
    expected_transaction_kind: str | None = None,
) -> dict[str, Any] | None:
    """Read and validate a writeback journal without mutating its targets."""

    journal = os.path.abspath(os.fspath(journal_path))
    if not os.path.lexists(journal):
        return None
    if os.path.islink(journal) or not os.path.isfile(journal):
        raise AtomicWriteTransactionError(
            f"Writeback transaction journal is not a regular file: {journal}"
        )
    try:
        with open(journal, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AtomicWriteTransactionError(
            f"Could not read writeback transaction journal {journal}: {exc}"
        ) from exc

    _state, transaction_kind, _entries = _validate_transaction_journal(payload, journal)
    if (
        expected_transaction_kind is not None
        and transaction_kind != expected_transaction_kind
    ):
        raise AtomicWriteTransactionError(
            "Writeback transaction identity does not match the requested "
            f"recovery kind: expected {expected_transaction_kind!r}, "
            f"got {transaction_kind!r} ({journal})."
        )
    return payload


def recover_atomic_write_transaction(
    journal_path: str | os.PathLike[str],
    *,
    expected_transaction_kind: str | None = None,
    verify_targets: bool = False,
) -> bool:
    """Recover an interrupted multi-file write transaction.

    Prepared transactions are rolled back. Committed transactions only need
    leftover temporary files removed. Returns True when a journal was found.

    When ``verify_targets`` is true, prepared entries that carry content
    digests are preflighted first: a target that changed outside the pending
    transaction makes recovery fail without overwriting that change. Entries
    from legacy journals without digests fall back to the historical rollback
    behavior.
    """

    journal = os.path.abspath(os.fspath(journal_path))
    payload = read_atomic_write_transaction_journal(
        journal,
        expected_transaction_kind=expected_transaction_kind,
    )
    if payload is None:
        return False
    state = str(payload["state"])
    entries = list(payload["entries"])

    if state == "prepared":
        if verify_targets:
            # Validate every target before touching any of them. A blocked
            # recovery must leave the journal and all visible state in place.
            for entry in entries:
                _verify_prepared_target_state(entry)
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


def _guarded_transaction_entry(
    *,
    target: str,
    staged_path: str,
    backup_path: str,
    existed: bool,
    expected_preimages: dict[str, str | None],
) -> dict[str, Any]:
    """Record actual content digests and enforce expected preimages."""

    actual_preimage = _preimage_digest(target, backup_path, existed)
    expected = expected_preimages.get(target, _EXPECTED_NOT_CHECKED)
    if expected is not _EXPECTED_NOT_CHECKED and actual_preimage != expected:
        _remove_if_present(staged_path)
        _remove_if_present(backup_path)
        expected_text = "(absent)" if expected is None else str(expected)
        actual_text = "(absent)" if actual_preimage is None else str(actual_preimage)
        raise AtomicWritePreimageConflict(
            f"Target changed outside the pending transaction before staging: {target} "
            f"(expected {expected_text}, found {actual_text})"
        )
    return _make_transaction_entry(
        target,
        staged_path,
        backup_path,
        existed,
        staged_sha256=file_sha256(staged_path),
        target_preimage_sha256=actual_preimage,
    )


def atomic_write_many_lines(
    writes: Iterable[tuple[str | os.PathLike[str], Iterable[str]]],
    *,
    journal_path: str | os.PathLike[str],
    encoding: str = "utf-8",
    newline: str | None = "\n",
    transaction_kind: str = "apply",
    expected_preimages: dict[str | os.PathLike[str], str | None] | None = None,
    metadata: dict[str, Any] | None = None,
    post_commit_validator: Callable[[], None] | None = None,
) -> None:
    """Replace multiple text files as one recoverable writeback transaction.

    All target contents are staged and backed up before the first replacement.
    A replacement failure rolls back every already-replaced target. A surviving
    ``prepared`` journal is likewise rolled back on the next invocation.

    ``expected_preimages`` optionally binds each target to its expected
    pre-transaction SHA-256 (or ``None`` for "must be absent"). Mismatches fail
    before any target is replaced.
    """

    journal = os.path.abspath(os.fspath(journal_path))
    recover_atomic_write_transaction(
        journal,
        expected_transaction_kind=transaction_kind,
        verify_targets=True,
    )
    normalized_writes = [
        (os.path.abspath(os.fspath(path)), lines)
        for path, lines in writes
    ]
    if not normalized_writes:
        return
    expected_hashes = _normalize_preimage_hashes(expected_preimages)

    entries: list[dict[str, Any]] = []
    try:
        for target, lines in normalized_writes:
            existed = os.path.exists(target)
            staged_path = _stage_lines(
                target,
                lines,
                encoding=encoding,
                newline=newline,
            )
            backup_path = ""
            try:
                backup_path = _backup_file(target)
                entries.append(
                    _guarded_transaction_entry(
                        target=target,
                        staged_path=staged_path,
                        backup_path=backup_path,
                        existed=existed,
                        expected_preimages=expected_hashes,
                    )
                )
            except Exception:
                _remove_if_present(staged_path)
                _remove_if_present(backup_path)
                raise
    except Exception:
        _cleanup_transaction_entries(entries)
        raise

    _run_many_transaction(
        entries,
        journal=journal,
        transaction_kind=transaction_kind,
        metadata=metadata,
        post_commit_validator=post_commit_validator,
    )


def atomic_write_many_bytes(
    writes: Iterable[tuple[str | os.PathLike[str], bytes]],
    *,
    journal_path: str | os.PathLike[str],
    transaction_kind: str = "apply",
    expected_preimages: dict[str | os.PathLike[str], str | None] | None = None,
    metadata: dict[str, Any] | None = None,
    post_commit_validator: Callable[[], None] | None = None,
) -> None:
    """Replace multiple byte files in one recoverable transaction.

    Exported localization files are compared and persisted as bytes so a
    UTF-8 BOM and the source newline/encoding contract cannot be lost while
    staging a complete file tree.
    """

    journal = os.path.abspath(os.fspath(journal_path))
    recover_atomic_write_transaction(
        journal,
        expected_transaction_kind=transaction_kind,
        verify_targets=True,
    )
    normalized_writes = [
        (os.path.abspath(os.fspath(path)), bytes(content))
        for path, content in writes
    ]
    if not normalized_writes:
        return
    expected_hashes = _normalize_preimage_hashes(expected_preimages)

    entries: list[dict[str, Any]] = []
    try:
        for target, content in normalized_writes:
            existed = os.path.exists(target)
            staged_path = _stage_bytes(target, content)
            backup_path = ""
            try:
                backup_path = _backup_file(target)
                entries.append(
                    _guarded_transaction_entry(
                        target=target,
                        staged_path=staged_path,
                        backup_path=backup_path,
                        existed=existed,
                        expected_preimages=expected_hashes,
                    )
                )
            except Exception:
                _remove_if_present(staged_path)
                _remove_if_present(backup_path)
                raise
    except Exception:
        _cleanup_transaction_entries(entries)
        raise

    _run_many_transaction(
        entries,
        journal=journal,
        transaction_kind=transaction_kind,
        metadata=metadata,
        post_commit_validator=post_commit_validator,
    )


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
