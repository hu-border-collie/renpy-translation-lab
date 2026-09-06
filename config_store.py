"""Shared raw JSON persistence for GUI settings and offline migration."""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"Failed to read {description}: {path}") from exc
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return data

def write_json_object(path: Path, data: dict[str, Any]) -> None:
    """Keep GUI serialization semantics while sharing the migration write lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with config_write_lock(path):
        _write_json_object(path, data)


def _write_json_object(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_mode = path.stat().st_mode & 0o777 if path.exists() else None
        target_mode = existing_mode
        if target_mode is None and path.name == "api_keys.json":
            target_mode = 0o600
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        if target_mode is None:
            tmp.write_text(payload, encoding="utf-8")
        else:
            if tmp.exists():
                tmp.unlink()
            fd: int | None = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, target_mode)
            try:
                assert fd is not None
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    fd = None
                    handle.write(payload)
            finally:
                if fd is not None:
                    os.close(fd)
            os.chmod(tmp, target_mode)
        os.replace(tmp, path)
    except OSError as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise ValueError(f"Failed to write JSON file: {path}") from exc


@contextmanager
def config_write_lock(path: Path):
    """Serialize cooperating settings/migration writers; never steal a stale lock.

    A process killed while holding the lock requires explicit operator cleanup.
    External editors do not honor this lock; migration also compares source bytes
    immediately before replacement.
    """
    lock = path.with_name(path.name + ".write-lock")
    try:
        handle = lock.open("xb")
    except FileExistsError as exc:
        raise ValueError("Configuration is locked; inspect active writers before removing the lock") from exc
    try:
        handle.write(str(os.getpid()).encode("ascii"))
        handle.close()
        yield
    finally:
        handle.close()
        lock.unlink()


def _copy_access_mode(source: Path, target: Path) -> None:
    """Preserve POSIX mode or the Windows source DACL before writing data."""
    if os.name != "nt":
        os.chmod(target, source.stat().st_mode & 0o777)
        return
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL("advapi32", use_last_error=True)
    get_security = api.GetFileSecurityW
    get_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                             wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    get_security.restype = wintypes.BOOL
    set_security = api.SetFileSecurityW
    set_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    set_security.restype = wintypes.BOOL
    needed = wintypes.DWORD()
    get_security(str(source), 4, None, 0, ctypes.byref(needed))
    if not needed.value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = ctypes.create_string_buffer(needed.value)
    if not get_security(str(source), 4, descriptor, needed, ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    # Protect the copied DACL from new inherited allow entries at its destination.
    if not set_security(str(target), 4 | 0x80000000, descriptor):
        raise ctypes.WinError(ctypes.get_last_error())


def write_private_artifact(path: Path, data: bytes, *, permissions_from: Path) -> None:
    """Create an exclusive backup/report without exposing its bytes to broader ACLs."""
    with path.open("xb") as handle:
        try:
            _copy_access_mode(permissions_from, path)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            path.unlink(missing_ok=True)
            raise


def replace_config_bytes(path: Path, data: bytes, *, expected: bytes) -> None:
    """Flush and atomically replace a config under the caller's write lock.

    Refuse symlinks and a changed source; temporary files inherit the original
    access restrictions. A failed replacement leaves the original intact.
    """
    if path.is_symlink() or path.read_bytes() != expected:
        raise ValueError("Configuration changed since preview")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            _copy_access_mode(path, temp)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError("Configuration changed since preview")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
