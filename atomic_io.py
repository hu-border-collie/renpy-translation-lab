"""Atomic file writes for translation writeback and batch artifacts.

Pattern matches the RAG store: write to a same-directory temporary file,
flush + fsync, then ``os.replace`` so readers never observe a truncated file.
"""

from __future__ import annotations

import errno
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


class AtomicFileLockUnavailableError(AtomicFileLockTimeoutError):
    """Raised when the filesystem cannot provide kernel-backed locks.

    Subclasses :class:`AtomicFileLockTimeoutError` so callers that already map
    a lock failure to their retryable "busy" result keep working on
    filesystems where ``flock`` / ``LockFile`` is not supported.
    """


LATEST_MANIFEST_LOCK_TIMEOUT = 30.0

_OS_LOCK_PROTOCOL = "os_lock_v1"
# Windows ``msvcrt.locking`` locks a byte range from the current position and
# may lock past EOF.  Locking a byte far beyond the small owner record keeps
# the file readable by diagnostics while the lock is held.
_OS_LOCK_BYTE_OFFSET = 1 << 20
# Owner records are rewritten in place with space padding instead of
# truncating the file: JSON parsers accept trailing whitespace, and avoiding
# SetEndOfFile under an active byte-range lock keeps the Windows path boring.
_OS_LOCK_OWNER_BYTES = 512
_OS_LOCK_OWNER_MAX_BYTES = 1 << 20
_LOCK_CONTENTION_ERRNOS = frozenset(
    value
    for value in (
        errno.EACCES,
        errno.EAGAIN,
        errno.EDEADLK,
        getattr(errno, "EDEADLOCK", None),
    )
    if value is not None
)
# Filesystems without kernel lock support (some NFS/FUSE/network mounts).
_LOCK_UNSUPPORTED_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "ENOLCK", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)


def _read_lock_owner(lock_path: str) -> dict[str, Any] | None:
    """Read the lock owner JSON; malformed/partially written files are unknown."""

    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _acquire_kernel_lock(fd: int) -> None:
    """Take the process-level exclusive lock or raise on contention."""

    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _OS_LOCK_BYTE_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_kernel_lock(fd: int) -> None:
    """Release the process-level exclusive lock.

    Closing the descriptor releases the lock as well, so callers treat a
    failure here as best effort.
    """

    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _OS_LOCK_BYTE_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def _is_lock_contention(exc: OSError) -> bool:
    """Return whether ``exc`` means another process currently holds the lock."""

    if exc.errno in _LOCK_CONTENTION_ERRNOS:
        return True
    # Windows ``msvcrt.locking`` can surface the underlying sharing/lock
    # violation codes instead of a mapped errno.
    return getattr(exc, "winerror", None) in (32, 33)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while recording file lock owner")
        view = view[written:]


def _lock_file_accepts_owner_record(fd: int) -> bool:
    """Return whether the existing lock file may be rewritten.

    The owner record is diagnostic only, so a pre-existing file that does not
    look like one of our lock files must never be overwritten or truncated.
    Empty files and legacy/current owner records (``pid`` plus ``token``) are
    accepted; anything else keeps its bytes and is only used for the kernel
    lock.
    """

    size = os.fstat(fd).st_size
    if size == 0:
        return True
    if size > _OS_LOCK_OWNER_MAX_BYTES:
        return False
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, size)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    pid = payload.get("pid")
    token = payload.get("token")
    return (
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and isinstance(token, str)
        and bool(token)
    )


@contextmanager
def exclusive_file_lock(
    lock_path: str | os.PathLike[str],
    *,
    timeout: float = 30.0,
    poll_interval: float = 0.01,
):
    """Serialize cooperating writers with a kernel-backed exclusive lock file.

    The lock is held by the operating system (``flock`` on POSIX and
    ``msvcrt.locking`` on Windows), so it is released automatically when the
    owner process exits or crashes.  Age-based stale-lock preemption is no
    longer needed and is no longer performed, which removes the
    read-then-unlink race between two preemptors (#474).

    The lock file is intentionally persistent: releasing the lock never
    unlinks it, because another waiter may already hold a handle to the same
    inode.  Never delete a lock file while writers may be active; a leftover
    file is harmless because only the kernel lock grants mutual exclusion.

    Symlinks and other non-regular files at the lock path are rejected, and a
    pre-existing file that does not look like one of our lock records is used
    for the kernel lock without being rewritten or truncated.
    """

    target = os.path.abspath(os.fspath(lock_path))
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(timeout))
    delay = max(0.001, float(poll_interval))

    try:
        link_stat = os.lstat(target)
    except FileNotFoundError:
        link_stat = None
    if link_stat is not None and not stat.S_ISREG(link_stat.st_mode):
        raise OSError(f"File lock path is not a regular file: {target}")

    try:
        fd = os.open(
            target,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except PermissionError as exc:
        # A lock file created by another user (or a read-only directory) is
        # reported as lock contention rather than a bare permission error so
        # callers keep their documented timeout semantics.
        if os.path.exists(target):
            raise AtomicFileLockTimeoutError(
                f"Timed out waiting for file lock: {target}"
            ) from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"File lock path is not a regular file: {target}")
    except Exception:
        os.close(fd)
        raise
    try:
        while True:
            try:
                _acquire_kernel_lock(fd)
            except OSError as exc:
                if exc.errno in _LOCK_UNSUPPORTED_ERRNOS:
                    raise AtomicFileLockUnavailableError(
                        f"File lock is not supported by this filesystem: {target}"
                    ) from exc
                if not _is_lock_contention(exc):
                    raise
                if time.monotonic() >= deadline:
                    raise AtomicFileLockTimeoutError(
                        f"Timed out waiting for file lock: {target}"
                    ) from exc
                time.sleep(delay)
                continue
            break

        owner = {
            "pid": os.getpid(),
            "token": uuid.uuid4().hex,
            "created_at": time.time(),
            "lock_protocol": _OS_LOCK_PROTOCOL,
        }
        if _lock_file_accepts_owner_record(fd):
            try:
                payload = json.dumps(owner, ensure_ascii=False).encode("utf-8")
                existing_size = os.fstat(fd).st_size
                target_size = max(_OS_LOCK_OWNER_BYTES, existing_size)
                if len(payload) > target_size:
                    raise OSError("file lock owner record exceeds reserved size")
                os.lseek(fd, 0, os.SEEK_SET)
                _write_all(fd, payload + b" " * (target_size - len(payload)))
                os.fsync(fd)
            except Exception:
                try:
                    _release_kernel_lock(fd)
                except OSError:
                    pass
                raise

        try:
            yield owner
        finally:
            try:
                _release_kernel_lock(fd)
            except OSError:
                pass
    finally:
        os.close(fd)


@contextmanager
def _latest_manifest_file_lock(
    lock_path: str | os.PathLike[str],
    *,
    timeout: float = LATEST_MANIFEST_LOCK_TIMEOUT,
    poll_interval: float = 0.01,
):
    """Non-preempting file-existence lock for the latest-manifest service.

    This preserves the #422 contract for the latest cursor: a leftover lock
    file (dead owner, corrupt record, or unknown age) never triggers automatic
    deletion, and waiters surface ``AtomicFileLockTimeoutError`` until an
    operator removes ``<latest>.lock`` after confirming that no writer is
    active.  Every other writer uses the kernel-backed
    :func:`exclusive_file_lock`, so the read-then-unlink race no longer exists
    anywhere in the codebase.
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
            if time.monotonic() >= deadline:
                raise AtomicFileLockTimeoutError(
                    f"Timed out waiting for file lock: {target}"
                ) from None
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


def latest_manifest_lock_path(latest_manifest_path: str | os.PathLike[str]) -> str:
    """Return the lock file shared by every latest-manifest writer."""

    return f"{os.path.abspath(os.fspath(latest_manifest_path))}.lock"


def _normalized_latest_cursor(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return os.path.normcase(os.path.realpath(os.path.abspath(text)))


def write_latest_manifest_locked(
    latest_manifest_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    timeout: float = LATEST_MANIFEST_LOCK_TIMEOUT,
) -> None:
    """Physically write the latest cursor under its shared writer lock."""

    latest = os.path.abspath(os.fspath(latest_manifest_path))
    # The latest cursor keeps the #422 manual-recovery contract: the
    # file-existence lock never preempts, so an abandoned <latest>.lock only
    # produces AtomicFileLockTimeoutError until an operator removes it after
    # confirming that no writer is active.
    with _latest_manifest_file_lock(
        latest_manifest_lock_path(latest),
        timeout=timeout,
    ):
        atomic_write_text(latest, str(manifest_path))


def compare_and_swap_latest_manifest_locked(
    latest_manifest_path: str | os.PathLike[str],
    expected: object,
    target: object,
    *,
    timeout: float = LATEST_MANIFEST_LOCK_TIMEOUT,
) -> dict[str, Any]:
    """Compare-and-swap the latest cursor under the shared writer lock."""

    latest = os.path.abspath(os.fspath(latest_manifest_path))
    expected_text = str(expected or "").strip()
    target_text = str(target or "").strip()
    # See write_latest_manifest_locked: latest locks are never auto-preempted.
    with _latest_manifest_file_lock(
        latest_manifest_lock_path(latest),
        timeout=timeout,
    ):
        current = ""
        try:
            with open(latest, "r", encoding="utf-8") as handle:
                current = handle.read().strip()
        except OSError:
            current = ""
        record = {
            "status": "skipped",
            "expected": expected_text,
            "target": target_text,
            "current": current,
            "message": "",
        }
        if target_text and _normalized_latest_cursor(current) == _normalized_latest_cursor(
            target_text
        ):
            record["status"] = "already_advanced"
            record["message"] = (
                "latest manifest cursor already points at the planned target"
            )
            return record
        if _normalized_latest_cursor(current) == _normalized_latest_cursor(
            expected_text
        ):
            atomic_write_text(latest, target_text)
            written = ""
            try:
                with open(latest, "r", encoding="utf-8") as handle:
                    written = handle.read().strip()
            except OSError:
                written = ""
            if _normalized_latest_cursor(written) != _normalized_latest_cursor(
                target_text
            ):
                raise RuntimeError(
                    "latest manifest cursor verification failed "
                    f"({written!r} != {target_text!r})"
                )
            record["status"] = "advanced"
            record["current"] = written
            record["message"] = "latest manifest cursor advanced to the planned target"
            return record
        record["status"] = "retained_newer"
        record["current"] = current
        record["message"] = (
            "latest manifest cursor was advanced by another operation; "
            "kept the newer value instead of overwriting it"
        )
        return record


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


def _classify_prepared_entry(entry: dict[str, Any]) -> str:
    """Classify one prepared journal entry without mutating any target.

    Returns one of ``legacy`` (no content digest), ``uncommitted``,
    ``committed`` or ``rolled_back``.  Any state that matches neither the
    staged nor the preimage bytes is an external modification and raises
    :class:`AtomicWritePreimageConflict`.
    """

    staged_sha256 = entry.get("staged_sha256")
    if not staged_sha256:
        return "legacy"
    target = entry["target"]
    staged_path = entry["staged_path"]
    preimage = entry.get("target_preimage_sha256")
    recorded = entry.get("rollback_state")
    staged_exists = os.path.exists(staged_path)
    target_exists = os.path.lexists(target)
    if target_exists and (os.path.islink(target) or not os.path.isfile(target)):
        raise AtomicWritePreimageConflict(
            f"Recovery target is not a regular file: {target}"
        )
    current = file_sha256(target) if target_exists else None
    expected_rolled_back = preimage if entry["existed"] else None

    if recorded == "rolled_back":
        if current == expected_rolled_back:
            return "rolled_back"
        raise AtomicWritePreimageConflict(
            "Previously rolled-back target changed outside the pending "
            f"transaction; refusing to continue recovery: {target}"
        )

    if staged_exists:
        # The staged file still exists, so this target was not replaced.
        if entry["existed"]:
            if preimage is None:
                raise AtomicWriteTransactionError(
                    f"Prepared entry has no preimage digest: {target}"
                )
            if current == preimage:
                return "uncommitted"
            raise AtomicWritePreimageConflict(
                "Uncommitted target changed outside the pending transaction; "
                f"refusing to roll it back: {target}"
            )
        if current is None:
            return "uncommitted"
        raise AtomicWritePreimageConflict(
            "Uncommitted target was created outside the pending transaction; "
            f"refusing to remove it: {target}"
        )

    # The staged file is gone: the replace for this target already happened, or
    # an earlier rollback already restored it.
    if entry["existed"]:
        if current == staged_sha256:
            return "committed"
        if preimage is not None and current == preimage:
            return "rolled_back"
        raise AtomicWritePreimageConflict(
            "Committed target is missing or changed outside the pending "
            f"transaction; refusing to overwrite it: {target}"
        )
    if current is None:
        return "rolled_back"
    if current == staged_sha256:
        return "committed"
    raise AtomicWritePreimageConflict(
        "Target was created outside the pending transaction after a new-file "
        f"commit; refusing to remove it: {target}"
    )


def _verify_prepared_target_state(entry: dict[str, Any]) -> None:
    """Preflight one prepared entry without mutating any target."""

    _classify_prepared_entry(entry)


def _journal_entries_are_strict(entries: Iterable[dict[str, Any]]) -> bool:
    """Return True when every entry carries content digests for strict recovery."""

    entry_list = list(entries)
    return bool(entry_list) and all(entry.get("staged_sha256") for entry in entry_list)


def _write_transaction_journal(journal: str, payload: dict[str, Any]) -> None:
    atomic_write_json(journal, payload, ensure_ascii=False, indent=2)


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
    if state not in {"prepared", "committed", "rolling_back", "rolled_back"} or not isinstance(entries, list):
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
        rollback_state = entry.get("rollback_state")
        if rollback_state is not None and rollback_state not in {"pending", "rolled_back"}:
            raise AtomicWriteTransactionError(
                f"Invalid rollback state in entry {index} of writeback journal: {journal}"
            )

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise AtomicWriteTransactionError(
            f"Invalid metadata in writeback transaction journal: {journal}"
        )

    recovery = payload.get("recovery")
    if recovery is not None and not isinstance(recovery, dict):
        raise AtomicWriteTransactionError(
            f"Invalid recovery phase in writeback transaction journal: {journal}"
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

    Prepared transactions are rolled back. Committed or already-rolled-back
    transactions only need leftover temporary files removed. Returns True when
    a journal was found.

    Strict recovery (new journals with content digests) persists the rollback
    phase in the journal before mutating targets and records each entry as
    ``rolled_back`` as it is restored.  A recovery pass interrupted by injected
    failures or a process stop is therefore replayable: already-restored
    entries are recognized as a legal state instead of being mistaken for
    external modifications.  A target that matches neither the staged bytes
    nor the preimage bytes still fails closed and is never overwritten.

    Legacy journals without content digests fall back to the historical
    rollback behavior.
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

    strict_entries = _journal_entries_are_strict(entries)
    use_rollback_phase = state == "rolling_back" or (
        state == "prepared" and verify_targets and strict_entries
    )
    if use_rollback_phase:
        if state == "prepared":
            # Preflight all targets and persist the rollback phase before any
            # mutation, so a crash during rollback is replayable.
            classifications = [
                _classify_prepared_entry(entry) for entry in entries
            ]
            for entry, classification in zip(entries, classifications):
                entry["rollback_state"] = (
                    "rolled_back"
                    if classification in {"uncommitted", "rolled_back"}
                    else "pending"
                )
            payload["state"] = "rolling_back"
            recovery = payload.get("recovery")
            if not isinstance(recovery, dict):
                recovery = {}
            recovery["started_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            payload["recovery"] = recovery
            _write_transaction_journal(journal, payload)
        else:
            # Resume: repair any marker whose target was restored before the
            # process stopped while that entry's journal update was pending.
            repaired = False
            for entry in entries:
                classification = _classify_prepared_entry(entry)
                if classification in {"uncommitted", "rolled_back"} and (
                    entry.get("rollback_state") != "rolled_back"
                ):
                    entry["rollback_state"] = "rolled_back"
                    repaired = True
            if repaired:
                _write_transaction_journal(journal, payload)

        for entry in reversed(entries):
            if entry.get("rollback_state") == "rolled_back":
                continue
            classification = _classify_prepared_entry(entry)
            if classification in {"uncommitted", "rolled_back"}:
                entry["rollback_state"] = "rolled_back"
                continue
            if classification != "committed":
                raise AtomicWriteTransactionError(
                    "Recovery cannot classify transaction target: "
                    f"{entry.get('target', '')}"
                )
            target = entry["target"]
            backup_path = entry["backup_path"]
            if entry["existed"]:
                if not backup_path or not os.path.isfile(backup_path):
                    raise AtomicWriteTransactionError(
                        f"Missing rollback backup for {target}: {backup_path or '(none)'}"
                    )
                _restore_backup_copy(backup_path, target)
                expected_preimage = entry.get("target_preimage_sha256")
                if expected_preimage and file_sha256(target) != expected_preimage:
                    raise AtomicWritePreimageConflict(
                        "Rollback restored bytes that do not match the recorded "
                        f"preimage: {target}"
                    )
            else:
                _remove_if_present(target)
                if os.path.lexists(target):
                    raise AtomicWritePreimageConflict(
                        f"Rollback could not remove newly committed target: {target}"
                    )
            entry["rollback_state"] = "rolled_back"
            _write_transaction_journal(journal, payload)

        # Mark the whole journal as rolled back before removing temporary files
        # and the journal itself. A later pass recognizes this terminal phase.
        payload["state"] = "rolled_back"
        recovery = payload.get("recovery")
        if not isinstance(recovery, dict):
            recovery = {}
        recovery["completed_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        payload["recovery"] = recovery
        _write_transaction_journal(journal, payload)
    elif state == "prepared":
        # Legacy rollback path (no content digests): preserve historical
        # behavior for journals written before strict target guards existed.
        if verify_targets:
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
