"""Safe export storage for Batch writeback results.

P1 (``export_only``) owns only the destination tree, receipt, and export
transaction.  P2 (``apply_and_export``) additionally commits the workspace
writeback targets and the export payloads in one recoverable transaction, then
leaves a receipt that the Batch workflow uses to advance apply state
idempotently.  This module never reads a manifest; the Batch workflow supplies
the already revalidated rendered bytes and recovery plan.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Iterable, Mapping

from atomic_io import (
    AtomicWritePreimageConflict,
    AtomicWriteTransactionError,
    atomic_write_many_bytes,
    file_sha256,
    read_atomic_write_transaction_journal,
    recover_atomic_write_transaction,
)


EXPORT_RECORD_SCHEMA_VERSION = 1
EXPORT_TRANSACTION_KIND = "export_only"
APPLY_EXPORT_TRANSACTION_KIND = "apply_export"
EXPORT_ONLY_RECORD_FILE = "export_only_record.json"
APPLY_EXPORT_RECORD_FILE = "apply_export_record.json"
APPLY_EXPORT_MODE = "apply-export"
EXPORT_ONLY_MODE = "export-only"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


class ExportTransactionError(ValueError):
    """Base class for unsafe export destination/receipt failures."""

    mode = "export"

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class ExportOnlyError(ExportTransactionError):
    """Raised when an export-only destination or receipt is unsafe to use."""

    mode = EXPORT_ONLY_MODE


class ApplyExportError(ExportTransactionError):
    """Raised when an apply + export destination or receipt is unsafe to use."""

    mode = APPLY_EXPORT_MODE


def _remap_export_only_error(exc: ExportOnlyError) -> ApplyExportError:
    reason_code = str(exc.reason_code or "export_only.failed")
    if reason_code.startswith("export_only."):
        reason_code = "apply_export." + reason_code[len("export_only.") :]
    message = str(exc)
    message = message.replace("--export-only", "--export-dir")
    message = message.replace("Export-only", "Apply-export")
    message = message.replace("export-only", "apply-export")
    return ApplyExportError(reason_code, message, details=exc.details)


@contextmanager
def _translate_export_only_errors() -> Iterator[None]:
    """Present shared P1 path failures as P2 reason codes at the P2 boundary."""

    try:
        yield
    except ApplyExportError:
        raise
    except ExportOnlyError as exc:
        raise _remap_export_only_error(exc) from exc


@dataclass(frozen=True)
class ExportRoot:
    """Validated destination information for one export request."""

    requested_path: str
    canonical_path: str
    record_path: str


def _digest_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _absolute_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.normpath(os.fspath(path)))


def _canonical_path(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(_absolute_path(path))


def _normalized_path(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(_canonical_path(path))


def _is_link_or_reparse(path: str) -> bool:
    try:
        if os.path.islink(path):
            return True
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ExportOnlyError(
            "export_only.path_unreadable",
            f"Cannot inspect export path component: {path} ({exc}).",
            details={"path": path},
        ) from exc
    return bool(
        int(getattr(info, "st_file_attributes", 0))
        & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _assert_no_link_components(
    path: str,
    field_name: str,
    *,
    allow_final_file: bool = False,
) -> None:
    """Reject links/junctions in the existing prefix of *path*."""

    candidate = _absolute_path(path)
    probe = candidate
    while not os.path.lexists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    if not os.path.isdir(probe):
        if _is_link_or_reparse(probe):
            raise ExportOnlyError(
                "export_only.path_link",
                f"{field_name} may not contain a symbolic link or junction: {probe}.",
                details={"field": field_name, "path": probe},
            )
        if not (allow_final_file and probe == candidate):
            raise ExportOnlyError(
                "export_only.path_invalid",
                f"{field_name} has a non-directory parent: {probe}.",
                details={"field": field_name, "path": path},
            )
        probe = os.path.dirname(probe)
        if not probe or not os.path.isdir(probe):
            raise ExportOnlyError(
                "export_only.path_invalid",
                f"{field_name} has a non-directory parent: {probe}.",
                details={"field": field_name, "path": path},
            )

    while True:
        if _is_link_or_reparse(probe):
            raise ExportOnlyError(
                "export_only.path_link",
                f"{field_name} may not contain a symbolic link or junction: {probe}.",
                details={"field": field_name, "path": probe},
            )
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent


def _is_within(base: str, candidate: str) -> bool:
    try:
        return os.path.commonpath([_normalized_path(base), _normalized_path(candidate)]) == _normalized_path(base)
    except ValueError:
        return False


def _paths_overlap(first: str, second: str) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _reject_parent_segments(path: str, field_name: str) -> None:
    parts = path.replace("\\", "/").split("/")
    if any(part == ".." for part in parts):
        raise ExportOnlyError(
            "export_only.path_escape",
            f"{field_name} may not contain '..' segments: {path}.",
            details={"field": field_name, "path": path},
        )


def _normalize_relative_path(value: str, field_name: str) -> str:
    raw = str(value or "")
    if not raw:
        raise ExportOnlyError(
            "export_only.path_invalid",
            f"{field_name} is empty.",
            details={"field": field_name},
        )
    normalized = raw.replace("\\", "/")
    _reject_parent_segments(normalized, field_name)
    if normalized.startswith("/"):
        raise ExportOnlyError(
            "export_only.path_escape",
            f"{field_name} must be relative: {value!r}.",
            details={"field": field_name, "path": raw},
        )
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ExportOnlyError(
            "export_only.path_escape",
            f"{field_name} must be relative: {value!r}.",
            details={"field": field_name, "path": raw},
        )
    parts = normalized.split("/")
    if any(not part or part == "." for part in parts):
        raise ExportOnlyError(
            "export_only.path_invalid",
            f"{field_name} is not normalized: {value!r}.",
            details={"field": field_name, "path": raw},
        )
    return "/".join(parts)


def validate_export_root(
    target_path: str,
    *,
    game_root: str,
    package_dir: str,
    record_filename: str = EXPORT_ONLY_RECORD_FILE,
) -> ExportRoot:
    """Validate an export root without creating it or inspecting payloads."""

    raw = str(target_path or "").strip()
    if not raw:
        raise ExportOnlyError(
            "export_only.path_required",
            "--export-only requires a non-empty destination directory.",
            details={"option": "--export-only"},
        )
    _reject_parent_segments(raw, "export-only destination")
    if not str(game_root or "").strip():
        raise ExportOnlyError(
            "export_only.game_root_missing",
            "Manifest has no bound game root; export-only is refused.",
            details={"field": "base_dir"},
        )
    if not str(package_dir or "").strip():
        raise ExportOnlyError(
            "export_only.package_root_missing",
            "Manifest has no task artifact root; export-only is refused.",
            details={"field": "_package_dir"},
        )

    _assert_no_link_components(_absolute_path(game_root), "manifest game root")
    _assert_no_link_components(_absolute_path(package_dir), "task artifact root")
    requested = _absolute_path(raw)
    _assert_no_link_components(requested, "export-only destination")
    if os.path.lexists(requested) and not os.path.isdir(requested):
        raise ExportOnlyError(
            "export_only.path_invalid",
            f"Export destination is not a directory: {requested}.",
            details={"path": requested},
        )

    game_root_abs = _canonical_path(game_root)
    package_dir_abs = _canonical_path(package_dir)
    destination_abs = _canonical_path(requested)
    if _paths_overlap(destination_abs, game_root_abs):
        raise ExportOnlyError(
            "export_only.path_overlaps_game_root",
            "Export destination may not overlap the manifest-bound game root.",
            details={
                "export_root": destination_abs,
                "game_root": game_root_abs,
            },
        )
    if _paths_overlap(destination_abs, package_dir_abs):
        raise ExportOnlyError(
            "export_only.path_overlaps_package_root",
            "Export destination may not overlap the task artifact root.",
            details={
                "export_root": destination_abs,
                "package_root": package_dir_abs,
            },
        )

    return ExportRoot(
        requested_path=requested,
        canonical_path=destination_abs,
        record_path=os.path.join(package_dir_abs, record_filename),
    )


def validate_apply_export_root(
    target_path: str,
    *,
    game_root: str,
    package_dir: str,
) -> ExportRoot:
    """Validate a P2 apply+export destination and its dedicated receipt path."""

    try:
        with _translate_export_only_errors():
            return validate_export_root(
                target_path,
                game_root=game_root,
                package_dir=package_dir,
                record_filename=APPLY_EXPORT_RECORD_FILE,
            )
    except ApplyExportError:
        raise


def _source_relative_path(game_root: str, source_path: str) -> str:
    source_abs = _canonical_path(source_path)
    game_root_abs = _canonical_path(game_root)
    if not _is_within(game_root_abs, source_abs) or _normalized_path(game_root_abs) == _normalized_path(source_abs):
        raise ExportOnlyError(
            "export_only.source_path_escape",
            f"Validated source path is outside the manifest game root: {source_path}.",
            details={"source_path": source_path, "game_root": game_root_abs},
        )
    relative = os.path.relpath(source_abs, game_root_abs).replace("\\", "/")
    return _normalize_relative_path(relative, "export relative path")


def _destination_path(root: ExportRoot, relative_path: str) -> str:
    candidate = _absolute_path(
        os.path.join(root.requested_path, *relative_path.split("/"))
    )
    if not _is_within(root.canonical_path, candidate):
        raise ExportOnlyError(
            "export_only.destination_escape",
            f"Export file escapes the destination root: {relative_path}.",
            details={"relative_path": relative_path, "export_root": root.canonical_path},
        )
    _assert_no_link_components(os.path.dirname(candidate), "export file parent")
    return candidate


def _scan_tree(root: ExportRoot) -> dict[str, Any] | None:
    if not os.path.lexists(root.requested_path):
        return None
    if _is_link_or_reparse(root.requested_path):
        raise ExportOnlyError(
            "export_only.path_link",
            "Export destination may not be a symbolic link or junction.",
            details={"export_root": root.requested_path},
        )
    if not os.path.isdir(root.requested_path):
        raise ExportOnlyError(
            "export_only.path_invalid",
            f"Export destination is not a directory: {root.requested_path}.",
            details={"export_root": root.requested_path},
        )

    files: dict[str, dict[str, Any]] = {}
    directories: set[str] = set()
    for current, dirnames, filenames in os.walk(
        root.requested_path,
        topdown=True,
        followlinks=False,
    ):
        for dirname in list(dirnames):
            directory = os.path.join(current, dirname)
            if _is_link_or_reparse(directory):
                raise ExportOnlyError(
                    "export_only.path_link",
                    f"Export tree contains a symbolic link or junction: {directory}.",
                    details={"path": directory},
                )
            rel_dir = os.path.relpath(directory, root.requested_path).replace("\\", "/")
            directories.add(_normalize_relative_path(rel_dir, "export directory path"))
        for filename in filenames:
            file_path = os.path.join(current, filename)
            if _is_link_or_reparse(file_path) or not os.path.isfile(file_path):
                raise ExportOnlyError(
                    "export_only.path_link",
                    f"Export tree contains an unsafe non-regular file: {file_path}.",
                    details={"path": file_path},
                )
            rel_file = os.path.relpath(file_path, root.requested_path).replace("\\", "/")
            rel_file = _normalize_relative_path(rel_file, "export file path")
            content = Path(file_path).read_bytes()
            files[rel_file] = {
                "sha256": _sha256(content),
                "size": len(content),
            }
    return {"files": files, "directories": sorted(directories)}


def _prune_empty_recovery_directories(root: ExportRoot) -> None:
    """Remove only empty directories left by a recovered staged export."""

    if not os.path.isdir(root.requested_path):
        return
    for current, dirnames, _filenames in os.walk(
        root.requested_path,
        topdown=False,
        followlinks=False,
    ):
        for dirname in dirnames:
            directory = os.path.join(current, dirname)
            if _is_link_or_reparse(directory):
                continue
            try:
                if not os.listdir(directory):
                    os.rmdir(directory)
            except OSError:
                continue


def prune_recovered_export_root(root: ExportRoot) -> None:
    """Remove empty directories left after recovering this export root."""

    _prune_empty_recovery_directories(root)


def recover_export_only_transaction(
    root: ExportRoot,
    journal_path: str,
) -> bool:
    """Recover an export journal after checking its targets are in scope."""

    if not os.path.lexists(journal_path):
        return False
    if _is_link_or_reparse(journal_path) or not os.path.isfile(journal_path):
        raise ExportOnlyError(
            "export_only.recovery_invalid",
            f"Export transaction journal is not a regular file: {journal_path}.",
            details={"journal_path": journal_path},
        )
    try:
        with open(journal_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExportOnlyError(
            "export_only.recovery_invalid",
            f"Export transaction journal is unreadable: {journal_path} ({exc}).",
            details={"journal_path": journal_path},
        ) from exc

    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ExportOnlyError(
            "export_only.recovery_invalid",
            f"Export transaction journal has no valid entries: {journal_path}.",
            details={"journal_path": journal_path},
        )

    root_norm = _normalized_path(root.canonical_path)
    record_norm = _normalized_path(root.record_path)
    for entry in entries:
        if not isinstance(entry, dict):
            raise ExportOnlyError(
                "export_only.recovery_invalid",
                f"Export transaction journal contains an invalid entry: {journal_path}.",
                details={"journal_path": journal_path},
            )
        target = str(entry.get("target") or "")
        if not target:
            raise ExportOnlyError(
                "export_only.recovery_invalid",
                f"Export transaction journal contains an empty target: {journal_path}.",
                details={"journal_path": journal_path},
            )
        target_norm = _normalized_path(target)
        if target_norm != record_norm and not (
            target_norm != root_norm and _is_within(root.canonical_path, target)
        ):
            raise ExportOnlyError(
                "export_only.recovery_escape",
                f"Export transaction target is outside the managed export scope: {target}.",
                details={"target": target, "journal_path": journal_path},
            )
        if os.path.lexists(target) and _is_link_or_reparse(target):
            raise ExportOnlyError(
                "export_only.recovery_link",
                f"Export transaction target is a symbolic link or junction: {target}.",
                details={"target": target, "journal_path": journal_path},
            )
        target_directory = _normalized_path(os.path.dirname(_absolute_path(target)))
        _assert_no_link_components(os.path.dirname(_absolute_path(target)), "export recovery parent")
        for temporary_key in ("staged_path", "backup_path"):
            temporary = str(entry.get(temporary_key) or "")
            if not temporary:
                continue
            if _normalized_path(os.path.dirname(_absolute_path(temporary))) != target_directory:
                raise ExportOnlyError(
                    "export_only.recovery_escape",
                    f"Export transaction temporary path is outside its target directory: {temporary}.",
                    details={"target": target, "temporary_path": temporary},
                )
            if os.path.lexists(temporary) and _is_link_or_reparse(temporary):
                raise ExportOnlyError(
                    "export_only.recovery_link",
                    f"Export transaction temporary path is a symbolic link or junction: {temporary}.",
                    details={"temporary_path": temporary},
                )

    try:
        return recover_atomic_write_transaction(
            journal_path,
            expected_transaction_kind=EXPORT_TRANSACTION_KIND,
        )
    except Exception as exc:
        if isinstance(exc, ExportOnlyError):
            raise
        raise ExportOnlyError(
            "export_only.recovery_invalid",
            f"Export transaction recovery failed: {exc}",
            details={"journal_path": journal_path},
        ) from exc


def _hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _load_record(
    record_path: str,
    *,
    transaction_kind: str = EXPORT_TRANSACTION_KIND,
) -> dict[str, Any]:
    if not os.path.lexists(record_path):
        return {
            "schema_version": EXPORT_RECORD_SCHEMA_VERSION,
            "transaction_kind": transaction_kind,
            "exports": [],
        }
    if _is_link_or_reparse(record_path):
        raise ExportOnlyError(
            "export_only.record_invalid",
            f"Export record may not be a symbolic link or junction: {record_path}.",
            details={"record_path": record_path},
        )
    try:
        with open(record_path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExportOnlyError(
            "export_only.record_invalid",
            f"Export record is unreadable or invalid: {record_path} ({exc}).",
            details={"record_path": record_path},
        ) from exc
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != EXPORT_RECORD_SCHEMA_VERSION
        or record.get("transaction_kind") != transaction_kind
        or not isinstance(record.get("exports"), list)
    ):
        raise ExportOnlyError(
            "export_only.record_invalid",
            f"Export record has an unsupported shape: {record_path}.",
            details={"record_path": record_path},
        )
    return record


def _validate_record_entry(
    entry: Any,
    *,
    transaction_kind: str = EXPORT_TRANSACTION_KIND,
    operation_prefix: str = "export-only",
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Export record contains a non-object export entry.",
        )
    export_root = entry.get("export_root")
    request_fingerprint = entry.get("request_fingerprint")
    operation_identity = entry.get("operation_identity")
    exported_at = entry.get("exported_at")
    files = entry.get("files")
    directories = entry.get("directories")
    if (
        not isinstance(export_root, str)
        or not export_root
        or not os.path.isabs(export_root)
        or not _hex64(request_fingerprint)
        or not isinstance(operation_identity, str)
        or not operation_identity
        or not isinstance(exported_at, str)
        or not exported_at
        or not isinstance(files, list)
        or not isinstance(directories, list)
        or entry.get("transaction_kind") != transaction_kind
    ):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Export record entry has an unsupported shape.",
        )
    if operation_identity != f"{operation_prefix}:{request_fingerprint}":
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Export record operation identity does not match its request fingerprint.",
            details={"export_root": export_root},
        )
    normalized_files: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ExportOnlyError(
                "export_only.record_invalid",
                "Export record contains a non-object file entry.",
            )
        relative_path = _normalize_relative_path(
            str(item.get("relative_path") or ""),
            "export record file path",
        )
        source_sha256 = item.get("source_sha256")
        output_sha256 = item.get("output_sha256")
        size = item.get("size")
        if (
            relative_path in seen_files
            or not _hex64(source_sha256)
            or not _hex64(output_sha256)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or output_sha256 == source_sha256
        ):
            raise ExportOnlyError(
                "export_only.record_invalid",
                "Export record contains an invalid file entry.",
                details={"relative_path": relative_path},
            )
        seen_files.add(relative_path)
        normalized_files.append(
            {
                "relative_path": relative_path,
                "source_sha256": source_sha256,
                "output_sha256": output_sha256,
                "size": size,
            }
        )

    normalized_directories: list[str] = []
    seen_directories: set[str] = set()
    for value in directories:
        directory = _normalize_relative_path(str(value or ""), "export record directory")
        if directory in seen_directories:
            raise ExportOnlyError(
                "export_only.record_invalid",
                "Export record contains duplicate directory entries.",
                details={"relative_path": directory},
            )
        seen_directories.add(directory)
        normalized_directories.append(directory)

    normalized_entry = dict(entry)
    normalized_entry["files"] = sorted(
        normalized_files,
        key=lambda item: item["relative_path"],
    )
    normalized_entry["directories"] = sorted(normalized_directories)
    if normalized_entry["directories"] != _managed_directories(
        [item["relative_path"] for item in normalized_entry["files"]]
    ):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Export record directory set does not match its managed files.",
            details={"export_root": export_root},
        )
    return normalized_entry


def _managed_directories(relative_paths: Iterable[str]) -> list[str]:
    directories: set[str] = set()
    for relative_path in relative_paths:
        parts = relative_path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    return sorted(directories)


def _record_file_entries(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for payload in payloads:
        content = bytes(payload["content"])
        entries.append(
            {
                "relative_path": payload["relative_path"],
                "source_sha256": str(payload["source_sha256"]),
                "output_sha256": _sha256(content),
                "size": len(content),
            }
        )
    return sorted(entries, key=lambda item: item["relative_path"])


def _workspace_record_entries(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for payload in payloads:
        content = bytes(payload["content"])
        entries.append(
            {
                "relative_path": payload["relative_path"],
                "file_key": payload["file_key"],
                "target_path": payload["target"],
                "source_sha256": str(payload["source_sha256"]),
                "output_sha256": _sha256(content),
                "size": len(content),
            }
        )
    return sorted(entries, key=lambda item: item["relative_path"])


def _normalize_workspace_record_entry(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Export record contains a non-object workspace file entry.",
        )
    relative_path = _normalize_relative_path(
        str(item.get("relative_path") or ""),
        "apply-export workspace file path",
    )
    file_key = _normalize_relative_path(
        str(item.get("file_key") or ""),
        "apply-export workspace file key",
    )
    target_path = item.get("target_path")
    source_sha256 = item.get("source_sha256")
    output_sha256 = item.get("output_sha256")
    size = item.get("size")
    if (
        not isinstance(target_path, str)
        or not target_path
        or not os.path.isabs(target_path)
        or not _hex64(source_sha256)
        or not _hex64(output_sha256)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
    ):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Export record contains an invalid workspace file entry.",
            details={"relative_path": relative_path},
        )
    return {
        "relative_path": relative_path,
        "file_key": file_key,
        "target_path": target_path,
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "size": size,
    }


def _normalize_apply_export_state_advancement(
    payload: Any,
    *,
    export_root: str,
    apply_identity: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export record has no state advancement plan.",
        )
    status = payload.get("status")
    manifest_path = payload.get("manifest_path")
    if payload.get("export_root") != export_root or payload.get("apply_identity") != apply_identity:
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export state plan does not match its export receipt.",
        )
    if status not in {"pending", "complete"}:
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export state plan has an invalid status.",
            details={"status": status},
        )
    if not isinstance(manifest_path, str) or not manifest_path or not os.path.isabs(manifest_path):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export state plan has an invalid manifest path.",
        )
    apply_summary = payload.get("apply_summary")
    if not isinstance(apply_summary, dict):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export state plan has no apply summary.",
        )
    progress = payload.get("progress")
    rag_jobs = payload.get("rag_jobs")
    if not isinstance(progress, list) or not isinstance(rag_jobs, list):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export state plan has invalid progress or RAG jobs.",
        )
    normalized_progress: list[dict[str, Any]] = []
    seen_file_keys: set[str] = set()
    for item in progress:
        if not isinstance(item, dict):
            raise ExportOnlyError(
                "export_only.record_invalid",
                "Apply-export progress entry must be an object.",
            )
        file_key = str(item.get("file_key") or "")
        if not file_key or file_key in seen_file_keys:
            raise ExportOnlyError(
                "export_only.record_invalid",
                "Apply-export progress entry has an invalid or duplicate file key.",
            )
        line_numbers = item.get("line_numbers")
        if not isinstance(line_numbers, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in line_numbers
        ):
            raise ExportOnlyError(
                "export_only.record_invalid",
                "Apply-export progress entry has invalid line numbers.",
                details={"file_key": file_key},
            )
        seen_file_keys.add(file_key)
        normalized_progress.append(
            {"file_key": file_key, "line_numbers": sorted(set(line_numbers))}
        )
    normalized_rag_jobs: list[dict[str, Any]] = []
    for item in rag_jobs:
        if not isinstance(item, dict):
            raise ExportOnlyError(
                "export_only.record_invalid",
                "Apply-export RAG job must be an object.",
            )
        file_rel_path = _normalize_relative_path(
            str(item.get("file_rel_path") or ""),
            "apply-export RAG file path",
        )
        normalized_rag_jobs.append({"file_rel_path": file_rel_path})
    latest_target = payload.get("latest_target")
    if not isinstance(latest_target, str):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export state plan has an invalid latest target.",
        )
    if latest_target and not os.path.isabs(latest_target):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export latest target must be absolute when present.",
        )
    for field in ("next_split_manifest_path", "applied_at", "quality_state", "updated_at"):
        value = payload.get(field)
        if not isinstance(value, str):
            raise ExportOnlyError(
                "export_only.record_invalid",
                f"Apply-export state plan has an invalid {field}.",
            )
    if not isinstance(payload.get("should_update_latest"), bool):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export state plan has an invalid latest-update flag.",
        )
    last_error = payload.get("last_error", "")
    if not isinstance(last_error, str):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export state plan has an invalid last_error.",
        )
    normalized = dict(payload)
    normalized["progress"] = normalized_progress
    normalized["rag_jobs"] = normalized_rag_jobs
    return normalized


def _validate_apply_export_entry(entry: Any) -> dict[str, Any]:
    normalized = _validate_record_entry(
        entry,
        transaction_kind=APPLY_EXPORT_TRANSACTION_KIND,
        operation_prefix="apply-export",
    )
    if normalized.get("mode") != APPLY_EXPORT_MODE:
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export record entry has an unsupported mode.",
            details={"export_root": normalized.get("export_root", "")},
        )
    apply_identity = normalized.get("apply_identity")
    if not _hex64(apply_identity):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export record entry has an invalid apply identity.",
            details={"export_root": normalized.get("export_root", "")},
        )
    raw_workspace_files = normalized.get("workspace_files")
    if not isinstance(raw_workspace_files, list):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Apply-export record entry has no workspace file list.",
        )
    workspace_files: list[dict[str, Any]] = []
    seen_workspace_files: set[str] = set()
    for item in raw_workspace_files:
        workspace_file = _normalize_workspace_record_entry(item)
        if workspace_file["relative_path"] in seen_workspace_files:
            raise ExportOnlyError(
                "export_only.record_invalid",
                "Apply-export record contains duplicate workspace file entries.",
                details={"relative_path": workspace_file["relative_path"]},
            )
        seen_workspace_files.add(workspace_file["relative_path"])
        workspace_files.append(workspace_file)
    normalized["workspace_files"] = sorted(
        workspace_files,
        key=lambda item: item["relative_path"],
    )
    normalized["state_advancement"] = _normalize_apply_export_state_advancement(
        normalized.get("state_advancement"),
        export_root=normalized["export_root"],
        apply_identity=apply_identity,
    )
    return normalized


def _load_apply_export_record(record_path: str) -> dict[str, Any]:
    try:
        with _translate_export_only_errors():
            return _load_record(
                record_path,
                transaction_kind=APPLY_EXPORT_TRANSACTION_KIND,
            )
    except ApplyExportError:
        raise


def _find_export_record(
    record: dict[str, Any],
    root: ExportRoot,
    *,
    transaction_kind: str = EXPORT_TRANSACTION_KIND,
    operation_prefix: str = "export-only",
) -> dict[str, Any] | None:
    for raw_entry in record["exports"]:
        entry = _validate_record_entry(
            raw_entry,
            transaction_kind=transaction_kind,
            operation_prefix=operation_prefix,
        )
        stored_root = entry["export_root"]
        if _normalized_path(stored_root) == _normalized_path(root.canonical_path):
            return entry
    return None


def _find_apply_export_record_by_root(
    record: dict[str, Any],
    root: ExportRoot,
) -> dict[str, Any] | None:
    with _translate_export_only_errors():
        return _find_export_record(
            record,
            root,
            transaction_kind=APPLY_EXPORT_TRANSACTION_KIND,
            operation_prefix="apply-export",
        )


def _find_apply_export_record_by_identity(
    record: dict[str, Any],
    apply_identity: str,
) -> dict[str, Any] | None:
    with _translate_export_only_errors():
        for raw_entry in record["exports"]:
            entry = _validate_apply_export_entry(raw_entry)
            if entry["apply_identity"] == apply_identity:
                return entry
    return None


def _tree_matches_entry(tree: dict[str, Any] | None, entry: dict[str, Any]) -> bool:
    if tree is None:
        return False
    expected_files = {
        item["relative_path"]: {
            "sha256": item["output_sha256"],
            "size": item["size"],
        }
        for item in entry["files"]
    }
    expected_directories = entry["directories"]
    return tree.get("files") == expected_files and tree.get("directories") == expected_directories


def _workspace_entries_match(
    recorded: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> bool:
    def key(item: dict[str, Any]) -> tuple[str, str, str, str, int]:
        return (
            item["relative_path"],
            item["file_key"],
            item["source_sha256"],
            item["output_sha256"],
            item["size"],
        )

    return sorted(map(key, recorded)) == sorted(map(key, expected))


def _prepare_export_payloads(
    root: ExportRoot,
    *,
    game_root: str,
    payloads: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate P1/P2 export payload paths and source-byte preimages."""

    game_root_abs = _canonical_path(game_root)
    if not os.path.isdir(game_root_abs):
        raise ExportOnlyError(
            "export_only.game_root_invalid",
            f"Manifest game root is not a directory: {game_root_abs}.",
            details={"game_root": game_root_abs},
        )

    prepared_payloads: list[dict[str, Any]] = []
    seen_relative: set[str] = set()
    seen_destinations: set[str] = set()
    for raw_payload in payloads:
        payload = dict(raw_payload)
        source_path = str(payload.get("source_path") or "")
        source_bytes = payload.get("source_bytes")
        output_bytes = payload.get("content")
        if not source_path or not isinstance(source_bytes, (bytes, bytearray)):
            raise ExportOnlyError(
                "export_only.payload_invalid",
                "Export payload is missing a validated source path or source bytes.",
            )
        if not isinstance(output_bytes, (bytes, bytearray)):
            raise ExportOnlyError(
                "export_only.payload_invalid",
                "Export payload is missing rendered bytes.",
            )
        _reject_parent_segments(source_path, "validated source path")
        source_path_abs = _absolute_path(source_path)
        _assert_no_link_components(
            source_path_abs,
            "validated source path",
            allow_final_file=True,
        )
        source_abs = _canonical_path(source_path_abs)
        relative_path = _source_relative_path(game_root_abs, source_abs)
        supplied_relative = payload.get("relative_path")
        if supplied_relative is not None and _normalize_relative_path(
            str(supplied_relative), "export relative path"
        ) != relative_path:
            raise ExportOnlyError(
                "export_only.path_mapping_mismatch",
                f"Export relative path does not match the validated source path: {source_path}.",
                details={"source_path": source_abs, "relative_path": relative_path},
            )
        if relative_path in seen_relative:
            raise ExportOnlyError(
                "export_only.duplicate_target",
                f"Multiple source files map to the same export path: {relative_path}.",
                details={"relative_path": relative_path},
            )
        destination = _destination_path(root, relative_path)
        destination_key = _normalized_path(destination)
        if destination_key in seen_destinations:
            raise ExportOnlyError(
                "export_only.duplicate_target",
                f"Multiple source files map to the same export target: {destination}.",
                details={"path": destination},
            )
        seen_relative.add(relative_path)
        seen_destinations.add(destination_key)

        source_content = bytes(source_bytes)
        try:
            current_source = Path(source_abs).read_bytes()
        except OSError as exc:
            raise ExportOnlyError(
                "export_only.source_unreadable",
                f"Validated source file cannot be read: {source_abs} ({exc}).",
                details={"source_path": source_abs},
            ) from exc
        if current_source != source_content:
            raise ExportOnlyError(
                "export_only.source_changed",
                f"Source changed before export: {source_abs}.",
                details={"source_path": source_abs},
            )

        output_content = bytes(output_bytes)
        if output_content == source_content:
            # The caller normally filters these after rendering; keep the
            # storage boundary fail-safe so a direct caller cannot turn a
            # byte-identical result into an exported payload.
            continue

        prepared_payloads.append(
            {
                "source_path": source_abs,
                "source_sha256": _sha256(source_content),
                "source_bytes": source_content,
                "relative_path": relative_path,
                "content": output_content,
                "destination": destination,
            }
        )

    prepared_payloads.sort(key=lambda item: item["relative_path"])
    return prepared_payloads


def export_only(
    root: ExportRoot,
    *,
    game_root: str,
    package_dir: str,
    payloads: Iterable[Mapping[str, Any]],
    request_payload: Mapping[str, Any],
    journal_path: str,
    recovery_state: str = "none",
) -> dict[str, Any]:
    """Export changed complete files and persist an idempotent receipt.

    ``payloads`` must contain bytes produced after the caller's check, source,
    and adapter revalidation.  The destination is created only after all path,
    receipt, and source-byte checks pass.
    """

    del package_dir  # The validated root already carries the record location.
    prepared_payloads = _prepare_export_payloads(
        root,
        game_root=game_root,
        payloads=payloads,
    )
    file_entries = _record_file_entries(prepared_payloads)
    request_document = {
        "schema_version": EXPORT_RECORD_SCHEMA_VERSION,
        "mode": "export-only",
        "request": dict(request_payload),
        "files": file_entries,
    }
    request_fingerprint = _digest_json(request_document)
    operation_identity = f"export-only:{request_fingerprint}"

    record = _load_record(root.record_path)
    existing_entry = _find_export_record(record, root)
    if recovery_state == "recovered" and existing_entry is None:
        _prune_empty_recovery_directories(root)
    current_tree = _scan_tree(root)
    expected_directories = _managed_directories(
        [entry["relative_path"] for entry in file_entries]
    )

    if existing_entry is not None:
        if existing_entry.get("request_fingerprint") != request_fingerprint:
            raise ExportOnlyError(
                "export_only.record_conflict",
                "Export destination is bound to a different request; use a new directory.",
                details={
                    "export_root": root.canonical_path,
                    "expected_request_fingerprint": request_fingerprint,
                    "record_request_fingerprint": existing_entry.get("request_fingerprint", ""),
                },
            )
        if not _tree_matches_entry(current_tree, existing_entry):
            raise ExportOnlyError(
                "export_only.tree_conflict",
                "Managed export files are missing, modified, or contain extra entries; use a new directory.",
                details={"export_root": root.canonical_path},
            )
        status = "no-op" if not file_entries else "exported"
        return {
            "mode": "export-only",
            "operation": EXPORT_TRANSACTION_KIND,
            "status": status,
            "export_root": root.canonical_path,
            "record_path": root.record_path,
            "export_record_path": root.record_path,
            "exported_files": len(file_entries),
            "payload_files": len(file_entries),
            "applied_files": 0,
            "actual_applied_files": 0,
            "applied_lines": 0,
            "idempotent": True,
            "request_fingerprint": request_fingerprint,
            "operation_identity": operation_identity,
            "recovery_state": "already_exported",
            "files": file_entries,
        }

    if current_tree is not None and (
        current_tree.get("files") or current_tree.get("directories")
    ):
        raise ExportOnlyError(
            "export_only.destination_conflict",
            "First export requires a missing or empty destination directory; use a new directory.",
            details={"export_root": root.canonical_path},
        )

    # No destination tree or receipt is created until every validation above,
    # including the final source-byte check, has completed.
    root_existed = os.path.lexists(root.requested_path)
    root_was_empty = current_tree is None or (
        not current_tree.get("files") and not current_tree.get("directories")
    )
    try:
        os.makedirs(root.requested_path, exist_ok=True)
    except OSError as exc:
        recovery_required = os.path.lexists(journal_path)
        if root_was_empty and not root_existed:
            try:
                _prune_empty_recovery_directories(root)
            except (ExportOnlyError, OSError):
                pass
            try:
                os.rmdir(root.requested_path)
            except OSError:
                pass
        reason_code = (
            "export_only.recovery_required"
            if recovery_required
            else "export_only.commit_failed"
        )
        message = (
            "Export destination setup failed and requires recovery before retry."
            if recovery_required
            else "Export destination could not be created."
        )
        raise ExportOnlyError(
            reason_code,
            f"{message} ({exc})",
            details={
                "journal_path": journal_path,
                "export_root": root.canonical_path,
                "recovery_state": (
                    "recovery_required" if recovery_required else "failed"
                ),
            },
        ) from exc
    expected_tree = {
        "files": {
            entry["relative_path"]: {
                "sha256": entry["output_sha256"],
                "size": entry["size"],
            }
            for entry in file_entries
        },
        "directories": expected_directories,
    }
    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "transaction_kind": EXPORT_TRANSACTION_KIND,
        "export_root": root.canonical_path,
        "request_fingerprint": request_fingerprint,
        "operation_identity": operation_identity,
        "exported_at": exported_at,
        "files": file_entries,
        "directories": expected_directories,
    }
    record["exports"] = [
        item
        for item in record["exports"]
        if isinstance(item, dict)
        and _normalized_path(str(item.get("export_root") or ""))
        != _normalized_path(root.canonical_path)
    ]
    record["exports"].append(entry)
    record_bytes = json.dumps(
        record,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    # The destination files and the receipt share one transaction journal.
    # This closes the interruption window between a successful tree commit and
    # receipt creation: recovery can roll back or complete both sides without
    # mistaking an unreceipted partial tree for a user-owned conflict.
    try:
        atomic_write_many_bytes(
            [
                *(
                    (payload["destination"], payload["content"])
                    for payload in prepared_payloads
                ),
                (root.record_path, record_bytes),
            ],
            journal_path=journal_path,
            transaction_kind=EXPORT_TRANSACTION_KIND,
        )
    except Exception as exc:
        recovery_required = os.path.lexists(journal_path)
        if root_was_empty:
            try:
                _prune_empty_recovery_directories(root)
            except (ExportOnlyError, OSError):
                # Cleanup diagnostics must not replace the original commit
                # failure or change the recovery classification.
                pass
            if not root_existed:
                try:
                    os.rmdir(root.requested_path)
                except OSError:
                    pass
        if isinstance(exc, ExportOnlyError):
            raise
        reason_code = (
            "export_only.recovery_required"
            if recovery_required
            else "export_only.commit_failed"
        )
        message = (
            "Export transaction failed and requires recovery before retry."
            if recovery_required
            else "Export transaction failed before producing a complete export."
        )
        raise ExportOnlyError(
            reason_code,
            f"{message} ({exc})",
            details={
                "journal_path": journal_path,
                "export_root": root.canonical_path,
                "recovery_state": (
                    "recovery_required" if recovery_required else "failed"
                ),
            },
        ) from exc

    written_tree = _scan_tree(root)
    if written_tree != expected_tree:
        raise ExportOnlyError(
            "export_only.tree_conflict",
            "Export tree changed during commit; the result is not accepted as a complete export.",
            details={"export_root": root.canonical_path},
        )

    status = "no-op" if not file_entries else "exported"
    return {
        "mode": "export-only",
        "operation": EXPORT_TRANSACTION_KIND,
        "status": status,
        "export_root": root.canonical_path,
        "record_path": root.record_path,
        "export_record_path": root.record_path,
        "exported_files": len(file_entries),
        "payload_files": len(file_entries),
        "applied_files": 0,
        "actual_applied_files": 0,
        "applied_lines": 0,
        "idempotent": False,
        "request_fingerprint": request_fingerprint,
        "operation_identity": operation_identity,
        "recovery_state": recovery_state,
        "files": file_entries,
    }


def read_apply_export_record(record_path: str) -> dict[str, Any]:
    """Read and validate a P2 apply-export record."""

    return _load_apply_export_record(record_path)


def normalize_apply_export_entry(entry: Any) -> dict[str, Any]:
    """Validate and normalize one P2 apply-export receipt entry."""

    try:
        with _translate_export_only_errors():
            return _validate_apply_export_entry(entry)
    except ApplyExportError:
        raise


def iter_apply_export_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every validated receipt entry in *record*."""

    exports = record.get("exports") if isinstance(record, dict) else None
    if not isinstance(exports, list):
        raise ApplyExportError(
            "apply_export.record_invalid",
            "Apply-export record has no export entry list.",
            details={"record_path": ""},
        )
    return [normalize_apply_export_entry(entry) for entry in exports]


def find_apply_export_entry_by_identity(
    record: dict[str, Any],
    apply_identity: str,
) -> dict[str, Any] | None:
    """Find one P2 receipt entry by its stable apply identity."""

    try:
        with _translate_export_only_errors():
            return _find_apply_export_record_by_identity(record, apply_identity)
    except ApplyExportError:
        raise


def load_apply_export_transaction_metadata(
    journal_path: str,
) -> dict[str, Any] | None:
    """Read the scope metadata of a pending P2 file transaction."""

    if not os.path.lexists(journal_path):
        return None
    if _is_link_or_reparse(journal_path) or not os.path.isfile(journal_path):
        raise ApplyExportError(
            "apply_export.recovery_invalid",
            f"Apply-export transaction journal is not a regular file: {journal_path}.",
            details={"journal_path": journal_path},
        )
    try:
        payload = read_atomic_write_transaction_journal(
            journal_path,
            expected_transaction_kind=APPLY_EXPORT_TRANSACTION_KIND,
        )
    except AtomicWriteTransactionError as exc:
        raise ApplyExportError(
            "apply_export.recovery_invalid",
            f"Apply-export transaction journal is invalid: {exc}",
            details={"journal_path": journal_path},
        ) from exc
    if payload is None:
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("mode") != APPLY_EXPORT_MODE:
        raise ApplyExportError(
            "apply_export.recovery_invalid",
            "Apply-export transaction journal has no usable scope metadata.",
            details={"journal_path": journal_path},
        )
    return dict(metadata)


def _verify_committed_receipt_entry(
    root: ExportRoot,
    entry: Mapping[str, Any],
    *,
    game_root: str,
    workspace_root: str,
    tree_conflict_code: str,
    workspace_conflict_code: str,
) -> dict[str, Any]:
    """Verify a committed receipt against both physical output trees."""

    normalized = normalize_apply_export_entry(dict(entry))
    game_root_norm = _normalized_path(game_root)
    workspace_root_norm = _normalized_path(workspace_root)
    with _translate_export_only_errors():
        try:
            tree = _scan_tree(root)
        except ExportOnlyError as exc:
            raise ApplyExportError(
                tree_conflict_code,
                f"Committed export tree cannot be verified: {exc}",
                details={"export_root": root.canonical_path},
            ) from exc
        if not _tree_matches_entry(tree, normalized):
            raise ApplyExportError(
                tree_conflict_code,
                "Committed export tree was modified after the transaction; "
                "refusing to advance apply state.",
                details={"export_root": root.canonical_path},
            )
        for workspace_file in normalized["workspace_files"]:
            target = workspace_file["target_path"]
            target_norm = _normalized_path(target)
            if not (
                _is_within(game_root_norm, target_norm)
                and _is_within(workspace_root_norm, target_norm)
            ):
                raise ApplyExportError(
                    workspace_conflict_code,
                    f"Committed workspace target is outside the managed scope: {target}.",
                    details={"target_path": target},
                )
            try:
                current_sha256 = file_sha256(target)
            except OSError as exc:
                raise ApplyExportError(
                    workspace_conflict_code,
                    f"Committed workspace file cannot be read: {target} ({exc}).",
                    details={"target_path": target},
                ) from exc
            if current_sha256 != workspace_file["output_sha256"]:
                raise ApplyExportError(
                    workspace_conflict_code,
                    f"Committed workspace file was modified after the transaction: {target}.",
                    details={"target_path": target},
                )
    return normalized


def verify_apply_export_entry(
    root: ExportRoot,
    entry: Mapping[str, Any],
    *,
    game_root: str,
    workspace_root: str,
) -> None:
    """Verify that a committed P2 receipt still matches both output trees."""

    _verify_committed_receipt_entry(
        root,
        entry,
        game_root=game_root,
        workspace_root=workspace_root,
        tree_conflict_code="apply_export.state_conflict",
        workspace_conflict_code="apply_export.state_conflict",
    )


def _prepare_workspace_payloads(
    *,
    workspace_root: str,
    game_root: str,
    payloads: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate P2 workspace writeback targets and source-byte preimages."""

    workspace_root_abs = _canonical_path(workspace_root)
    game_root_abs = _canonical_path(game_root)
    if not os.path.isdir(workspace_root_abs):
        raise ApplyExportError(
            "apply_export.workspace_root_invalid",
            f"Manifest TL root is not a directory: {workspace_root_abs}.",
            details={"workspace_root": workspace_root_abs},
        )
    if not os.path.isdir(game_root_abs):
        raise ApplyExportError(
            "apply_export.game_root_invalid",
            f"Manifest game root is not a directory: {game_root_abs}.",
            details={"game_root": game_root_abs},
        )

    prepared: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for raw_payload in payloads:
        payload = dict(raw_payload)
        target_value = str(payload.get("target_path") or payload.get("source_path") or "")
        source_value = str(payload.get("source_path") or target_value)
        source_bytes = payload.get("source_bytes")
        output_bytes = payload.get("content")
        if not target_value or not isinstance(source_bytes, (bytes, bytearray)):
            raise ApplyExportError(
                "apply_export.payload_invalid",
                "Workspace payload is missing a validated target path or source bytes.",
            )
        if not isinstance(output_bytes, (bytes, bytearray)):
            raise ApplyExportError(
                "apply_export.payload_invalid",
                "Workspace payload is missing rendered bytes.",
            )
        _reject_parent_segments(target_value, "validated workspace path")
        target_abs = _absolute_path(target_value)
        _assert_no_link_components(
            target_abs,
            "validated workspace path",
            allow_final_file=True,
        )
        target = _canonical_path(target_abs)
        if not _is_within(workspace_root_abs, target) or _normalized_path(
            workspace_root_abs
        ) == _normalized_path(target):
            raise ApplyExportError(
                "apply_export.workspace_path_escape",
                f"Workspace target is outside the manifest TL root: {target_value}.",
                details={"target_path": target, "workspace_root": workspace_root_abs},
            )
        if not _is_within(game_root_abs, target):
            raise ApplyExportError(
                "apply_export.workspace_path_escape",
                f"Workspace target is outside the manifest game root: {target_value}.",
                details={"target_path": target, "game_root": game_root_abs},
            )
        source_abs = _canonical_path(_absolute_path(source_value)) if source_value else target
        if _normalized_path(source_abs) != _normalized_path(target):
            raise ApplyExportError(
                "apply_export.workspace_mapping_mismatch",
                f"Workspace target does not match the validated source path: {target_value}.",
                details={"target_path": target, "source_path": source_abs},
            )
        if _normalized_path(target) in seen_targets:
            raise ApplyExportError(
                "apply_export.duplicate_target",
                f"Multiple workspace payloads map to the same target: {target}.",
                details={"target_path": target},
            )
        seen_targets.add(_normalized_path(target))

        relative_path = os.path.relpath(target, workspace_root_abs).replace("\\", "/")
        relative_path = _normalize_relative_path(relative_path, "workspace relative path")
        file_key = _normalize_relative_path(
            str(payload.get("file_key") or relative_path),
            "workspace file key",
        )
        source_content = bytes(source_bytes)
        try:
            current_source = Path(target).read_bytes()
        except OSError as exc:
            raise ApplyExportError(
                "apply_export.source_unreadable",
                f"Validated workspace file cannot be read: {target} ({exc}).",
                details={"target_path": target},
            ) from exc
        if current_source != source_content:
            raise ApplyExportError(
                "apply_export.source_changed",
                f"Workspace source changed before apply-export: {target}.",
                details={"target_path": target},
            )

        prepared.append(
            {
                "target": target,
                "source_path": target,
                "source_sha256": _sha256(source_content),
                "source_bytes": source_content,
                "relative_path": relative_path,
                "file_key": file_key,
                "content": bytes(output_bytes),
            }
        )
    prepared.sort(key=lambda item: item["relative_path"])
    return prepared


def _prune_failed_apply_export_root(
    root: ExportRoot,
    *,
    root_existed: bool,
    root_was_empty: bool,
) -> None:
    """Best-effort removal of directories created by a failed first export."""

    if not root_was_empty:
        return
    try:
        _prune_empty_recovery_directories(root)
    except (ExportOnlyError, OSError):
        pass
    if not root_existed:
        try:
            os.rmdir(root.requested_path)
        except OSError:
            pass


def _idempotent_apply_export_summary(
    existing_entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the stable summary for a receipt-matched idempotent replay."""

    existing = normalize_apply_export_entry(dict(existing_entry))
    state = existing["state_advancement"]
    applied_lines = int((state.get("apply_summary") or {}).get("applied_lines") or 0)
    status = (
        "no-op"
        if not existing["workspace_files"] and not existing["files"]
        else "applied_and_exported"
    )
    summary: dict[str, Any] = {
        "mode": APPLY_EXPORT_MODE,
        "operation": APPLY_EXPORT_TRANSACTION_KIND,
        "status": status,
        "export_root": existing["export_root"],
        "record_path": state.get("record_path", ""),
        "export_record_path": state.get("record_path", ""),
        "exported_files": len(existing["files"]),
        "payload_files": len(existing["files"]),
        "applied_files": 0,
        "actual_applied_files": 0,
        "applied_lines": applied_lines,
        "workspace_files_count": len(existing["workspace_files"]),
        "idempotent": True,
        "request_fingerprint": existing["request_fingerprint"],
        "apply_identity": existing["apply_identity"],
        "operation_identity": existing["operation_identity"],
        "recovery_state": "already_committed",
        "exported_at": existing["exported_at"],
        "files": existing["files"],
        "workspace_files": existing["workspace_files"],
        "state_advancement": state,
        "record_entry": existing,
    }
    return summary


def _apply_and_export_impl(
    root: ExportRoot,
    *,
    game_root: str,
    workspace_root: str,
    package_dir: str,
    workspace_payloads: Iterable[Mapping[str, Any]],
    export_payloads: Iterable[Mapping[str, Any]],
    request_payload: Mapping[str, Any],
    apply_identity: str,
    state_advancement: Mapping[str, Any],
    journal_path: str,
    recovery_state: str,
) -> dict[str, Any]:
    """Implement the P2 transaction body; P2 callers wrap error translation."""

    if not _hex64(apply_identity):
        raise ApplyExportError(
            "apply_export.identity_invalid",
            "Apply-export identity must be a SHA-256 hex digest.",
        )
    game_root_abs = _canonical_path(game_root)
    if not os.path.isdir(game_root_abs):
        raise ApplyExportError(
            "apply_export.game_root_invalid",
            f"Manifest game root is not a directory: {game_root_abs}.",
            details={"game_root": game_root_abs},
        )
    workspace_root_abs = _canonical_path(workspace_root)
    if not os.path.isdir(workspace_root_abs):
        raise ApplyExportError(
            "apply_export.workspace_root_invalid",
            f"Manifest TL root is not a directory: {workspace_root_abs}.",
            details={"workspace_root": workspace_root_abs},
        )
    package_dir_abs = _canonical_path(package_dir)
    if not os.path.isdir(package_dir_abs):
        raise ApplyExportError(
            "apply_export.package_root_invalid",
            f"Manifest task artifact root is not a directory: {package_dir_abs}.",
            details={"package_root": package_dir_abs},
        )

    # Receipt replay is checked before payload preimages: after a successful
    # commit the workspace sources already contain the rendered output, so a
    # repeat call must recognize the immutable receipt instead of re-reading
    # the pre-commit source bytes.
    record = _load_apply_export_record(root.record_path)
    existing_same_root = _find_apply_export_record_by_root(record, root)
    existing_same_identity = _find_apply_export_record_by_identity(
        record,
        apply_identity,
    )
    if existing_same_root is not None:
        if existing_same_root["apply_identity"] != apply_identity:
            raise ApplyExportError(
                "apply_export.record_conflict",
                "Export destination is bound to a different apply identity; use a new directory.",
                details={
                    "export_root": root.canonical_path,
                    "record_apply_identity": existing_same_root["apply_identity"],
                    "requested_apply_identity": apply_identity,
                },
            )
        _verify_committed_receipt_entry(
            root,
            existing_same_root,
            game_root=game_root_abs,
            workspace_root=workspace_root_abs,
            tree_conflict_code="apply_export.tree_conflict",
            workspace_conflict_code="apply_export.workspace_conflict",
        )
        return _idempotent_apply_export_summary(existing_same_root)
    if existing_same_identity is not None:
        raise ApplyExportError(
            "apply_export.record_conflict",
            "This apply identity is already bound to another export directory.",
            details={
                "apply_identity": apply_identity,
                "record_export_root": existing_same_identity["export_root"],
                "requested_export_root": root.canonical_path,
            },
        )

    prepared_workspace = _prepare_workspace_payloads(
        workspace_root=workspace_root_abs,
        game_root=game_root_abs,
        payloads=workspace_payloads,
    )
    prepared_export = _prepare_export_payloads(
        root,
        game_root=game_root_abs,
        payloads=export_payloads,
    )

    # The two sides must be projections of the same validated render output.
    workspace_by_source = {
        _normalized_path(item["source_path"]): item for item in prepared_workspace
    }
    for export_item in prepared_export:
        workspace_item = workspace_by_source.get(
            _normalized_path(export_item["source_path"])
        )
        if workspace_item is None or bytes(workspace_item["content"]) != bytes(
            export_item["content"]
        ):
            raise ApplyExportError(
                "apply_export.sides_mismatch",
                "Workspace and export outputs are not byte-identical for "
                f"{export_item['relative_path']}.",
                details={"relative_path": export_item["relative_path"]},
            )

    export_entries = _record_file_entries(prepared_export)
    workspace_entries = _workspace_record_entries(prepared_workspace)
    request_document = {
        "schema_version": EXPORT_RECORD_SCHEMA_VERSION,
        "mode": APPLY_EXPORT_MODE,
        "request": dict(request_payload),
        "files": export_entries,
        "workspace_files": [
            {key: value for key, value in item.items() if key != "target_path"}
            for item in workspace_entries
        ],
    }
    request_fingerprint = _digest_json(request_document)
    operation_identity = f"apply-export:{request_fingerprint}"

    plan = dict(state_advancement or {})
    apply_summary = plan.get("apply_summary")
    manifest_path = plan.get("manifest_path")
    if not isinstance(apply_summary, dict):
        raise ApplyExportError(
            "apply_export.state_plan_invalid",
            "Apply-export state advancement requires an apply summary.",
        )
    if not isinstance(manifest_path, str) or not manifest_path or not os.path.isabs(
        manifest_path
    ):
        raise ApplyExportError(
            "apply_export.state_plan_invalid",
            "Apply-export state advancement requires an absolute manifest path.",
        )
    raw_applied_lines = apply_summary.get("applied_lines", 0)
    if not isinstance(raw_applied_lines, int) or isinstance(raw_applied_lines, bool):
        raise ApplyExportError(
            "apply_export.state_plan_invalid",
            "Apply-export apply summary has invalid applied line counts.",
        )
    plan_applied_lines = max(0, raw_applied_lines)

    try:
        current_tree = _scan_tree(root)
    except ExportOnlyError as exc:
        raise ApplyExportError(
            "apply_export.destination_conflict",
            f"Export destination cannot be inspected: {exc}",
            details={"export_root": root.canonical_path},
        ) from exc
    if current_tree is not None and (
        current_tree.get("files") or current_tree.get("directories")
    ):
        raise ApplyExportError(
            "apply_export.destination_conflict",
            "First apply-export requires a missing or empty destination directory; "
            "use a new directory.",
            details={"export_root": root.canonical_path},
        )

    # Nothing below this point writes until every path, receipt, preimage, and
    # render-byte check above has completed.
    root_existed = os.path.lexists(root.requested_path)
    root_was_empty = current_tree is None or (
        not current_tree.get("files") and not current_tree.get("directories")
    )
    try:
        os.makedirs(root.requested_path, exist_ok=True)
    except OSError as exc:
        recovery_required = os.path.lexists(journal_path)
        _prune_failed_apply_export_root(
            root,
            root_existed=root_existed,
            root_was_empty=root_was_empty,
        )
        raise ApplyExportError(
            "apply_export.recovery_required"
            if recovery_required
            else "apply_export.commit_failed",
            f"Export destination could not be created ({exc}).",
            details={
                "journal_path": journal_path,
                "export_root": root.canonical_path,
                "recovery_state": (
                    "recovery_required" if recovery_required else "failed"
                ),
            },
        ) from exc

    expected_directories = _managed_directories(
        [entry["relative_path"] for entry in export_entries]
    )
    expected_tree = {
        "files": {
            entry["relative_path"]: {
                "sha256": entry["output_sha256"],
                "size": entry["size"],
            }
            for entry in export_entries
        },
        "directories": expected_directories,
    }
    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = (
        "no-op"
        if not prepared_workspace and not prepared_export
        else "applied_and_exported"
    )
    summary: dict[str, Any] = {
        "mode": APPLY_EXPORT_MODE,
        "operation": APPLY_EXPORT_TRANSACTION_KIND,
        "status": status,
        "export_root": root.canonical_path,
        "record_path": root.record_path,
        "export_record_path": root.record_path,
        "exported_files": len(export_entries),
        "payload_files": len(export_entries),
        "applied_files": len(workspace_entries),
        "actual_applied_files": len(workspace_entries),
        "applied_lines": plan_applied_lines,
        "workspace_files_count": len(workspace_entries),
        "idempotent": False,
        "request_fingerprint": request_fingerprint,
        "apply_identity": apply_identity,
        "operation_identity": operation_identity,
        "recovery_state": recovery_state,
        "exported_at": exported_at,
        "files": export_entries,
        "workspace_files": workspace_entries,
    }
    plan.update(
        {
            "status": "pending",
            "apply_identity": apply_identity,
            "export_root": root.canonical_path,
            "record_path": root.record_path,
            "manifest_path": manifest_path,
            "updated_at": exported_at,
        }
    )
    plan["export_summary"] = dict(summary)
    summary["state_advancement"] = plan
    record_entry = {
        "mode": APPLY_EXPORT_MODE,
        "transaction_kind": APPLY_EXPORT_TRANSACTION_KIND,
        "export_root": root.canonical_path,
        "request_fingerprint": request_fingerprint,
        "apply_identity": apply_identity,
        "operation_identity": operation_identity,
        "exported_at": exported_at,
        "files": export_entries,
        "directories": expected_directories,
        "workspace_files": workspace_entries,
        "state_advancement": plan,
    }
    record["exports"] = [
        item
        for item in record["exports"]
        if not (
            isinstance(item, dict)
            and _normalized_path(str(item.get("export_root") or ""))
            == _normalized_path(root.canonical_path)
        )
    ]
    record["exports"].append(record_entry)
    record_bytes = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")

    expected_preimages: dict[str, str | None] = {}
    for workspace_item in prepared_workspace:
        expected_preimages[workspace_item["target"]] = workspace_item["source_sha256"]
    for export_item in prepared_export:
        expected_preimages[export_item["destination"]] = None
    record_existed = os.path.lexists(root.record_path)
    expected_preimages[root.record_path] = (
        file_sha256(root.record_path) if record_existed else None
    )

    writes = [
        *(
            (workspace_item["target"], workspace_item["content"])
            for workspace_item in prepared_workspace
        ),
        *(
            (export_item["destination"], export_item["content"])
            for export_item in prepared_export
        ),
        (root.record_path, record_bytes),
    ]

    def validate_committed_outputs() -> None:
        for workspace_item in prepared_workspace:
            target = workspace_item["target"]
            try:
                current_sha256 = file_sha256(target)
            except OSError as exc:
                raise ApplyExportError(
                    "apply_export.verification_failed",
                    f"Workspace file disappeared during commit: {target} ({exc}).",
                    details={"target_path": target},
                ) from exc
            if current_sha256 != _sha256(bytes(workspace_item["content"])):
                raise ApplyExportError(
                    "apply_export.verification_failed",
                    f"Workspace file bytes do not match the validated render output: {target}.",
                    details={"target_path": target},
                )
        for export_item in prepared_export:
            destination = export_item["destination"]
            try:
                current_sha256 = file_sha256(destination)
            except OSError as exc:
                raise ApplyExportError(
                    "apply_export.verification_failed",
                    f"Export file disappeared during commit: {destination} ({exc}).",
                    details={"destination": destination},
                ) from exc
            if current_sha256 != _sha256(bytes(export_item["content"])):
                raise ApplyExportError(
                    "apply_export.verification_failed",
                    f"Export file bytes do not match the validated render output: {destination}.",
                    details={"destination": destination},
                )
        if _scan_tree(root) != expected_tree:
            raise ApplyExportError(
                "apply_export.verification_failed",
                "Export tree does not match the validated payload set.",
                details={"export_root": root.canonical_path},
            )

    metadata = {
        "mode": APPLY_EXPORT_MODE,
        "export_root": root.canonical_path,
        "workspace_root": workspace_root_abs,
        "game_root": game_root_abs,
        "package_dir": package_dir_abs,
        "record_path": root.record_path,
        "journal_path": os.path.abspath(journal_path),
        "apply_identity": apply_identity,
        "request_fingerprint": request_fingerprint,
        "operation_identity": operation_identity,
    }
    try:
        atomic_write_many_bytes(
            writes,
            journal_path=journal_path,
            transaction_kind=APPLY_EXPORT_TRANSACTION_KIND,
            expected_preimages=expected_preimages,
            metadata=metadata,
            post_commit_validator=validate_committed_outputs,
        )
    except ApplyExportError as exc:
        _prune_failed_apply_export_root(
            root,
            root_existed=root_existed,
            root_was_empty=root_was_empty,
        )
        if os.path.lexists(journal_path):
            raise ApplyExportError(
                "apply_export.recovery_required",
                f"Apply-export transaction failed and requires recovery: {exc}",
                details={
                    "journal_path": journal_path,
                    "export_root": root.canonical_path,
                    "record_path": root.record_path,
                    "recovery_state": "recovery_required",
                    "cause": str(exc),
                },
            ) from exc
        raise
    except AtomicWritePreimageConflict as exc:
        _prune_failed_apply_export_root(
            root,
            root_existed=root_existed,
            root_was_empty=root_was_empty,
        )
        raise ApplyExportError(
            "apply_export.preimage_conflict",
            f"Workspace or export target changed outside the transaction: {exc}",
            details={
                "journal_path": journal_path,
                "export_root": root.canonical_path,
                "recovery_state": (
                    "recovery_required"
                    if os.path.lexists(journal_path)
                    else "failed"
                ),
            },
        ) from exc
    except ExportTransactionError as exc:
        _prune_failed_apply_export_root(
            root,
            root_existed=root_existed,
            root_was_empty=root_was_empty,
        )
        if os.path.lexists(journal_path):
            raise ApplyExportError(
                "apply_export.recovery_required",
                f"Apply-export transaction failed and requires recovery: {exc}",
                details={
                    "journal_path": journal_path,
                    "export_root": root.canonical_path,
                    "record_path": root.record_path,
                    "recovery_state": "recovery_required",
                    "cause": str(exc),
                },
            ) from exc
        raise
    except Exception as exc:
        recovery_required = os.path.lexists(journal_path)
        _prune_failed_apply_export_root(
            root,
            root_existed=root_existed,
            root_was_empty=root_was_empty,
        )
        reason_code = (
            "apply_export.recovery_required"
            if recovery_required
            else "apply_export.commit_failed"
        )
        message = (
            "Apply-export transaction failed and requires recovery before retry."
            if recovery_required
            else "Apply-export transaction failed before producing complete output."
        )
        raise ApplyExportError(
            reason_code,
            f"{message} ({exc})",
            details={
                "journal_path": journal_path,
                "export_root": root.canonical_path,
                "record_path": root.record_path,
                "recovery_state": (
                    "recovery_required" if recovery_required else "failed"
                ),
            },
        ) from exc

    written_tree = _scan_tree(root)
    if not _tree_matches_entry(written_tree, record_entry):
        raise ApplyExportError(
            "apply_export.verification_failed",
            "Export tree changed during commit; the result is not accepted as complete.",
            details={
                "export_root": root.canonical_path,
                "recovery_state": "state_conflict",
            },
        )
    summary["record_entry"] = record_entry
    return summary


def apply_and_export(
    root: ExportRoot,
    *,
    game_root: str,
    workspace_root: str,
    package_dir: str,
    workspace_payloads: Iterable[Mapping[str, Any]],
    export_payloads: Iterable[Mapping[str, Any]],
    request_payload: Mapping[str, Any],
    apply_identity: str,
    state_advancement: Mapping[str, Any],
    journal_path: str,
    recovery_state: str = "none",
) -> dict[str, Any]:
    """Commit validated workspace and export outputs in one recoverable transaction.

    ``workspace_payloads`` and ``export_payloads`` must be projections of the
    same validated render result.  The receipt stores a pending state
    advancement plan that the Batch workflow replays idempotently after the
    file transaction commits.
    """

    try:
        with _translate_export_only_errors():
            return _apply_and_export_impl(
                root,
                game_root=game_root,
                workspace_root=workspace_root,
                package_dir=package_dir,
                workspace_payloads=workspace_payloads,
                export_payloads=export_payloads,
                request_payload=request_payload,
                apply_identity=apply_identity,
                state_advancement=state_advancement,
                journal_path=journal_path,
                recovery_state=recovery_state,
            )
    except ApplyExportError:
        raise


def recover_apply_export_transaction(
    root: ExportRoot,
    journal_path: str,
    *,
    workspace_root: str,
    game_root: str,
    package_dir: str,
) -> bool:
    """Recover an interrupted P2 transaction after validating managed scopes."""

    if not os.path.lexists(journal_path):
        return False
    if _is_link_or_reparse(journal_path) or not os.path.isfile(journal_path):
        raise ApplyExportError(
            "apply_export.recovery_invalid",
            f"Apply-export transaction journal is not a regular file: {journal_path}.",
            details={"journal_path": journal_path},
        )
    try:
        payload = read_atomic_write_transaction_journal(
            journal_path,
            expected_transaction_kind=APPLY_EXPORT_TRANSACTION_KIND,
        )
    except AtomicWriteTransactionError as exc:
        raise ApplyExportError(
            "apply_export.recovery_invalid",
            f"Apply-export transaction journal is invalid: {exc}",
            details={"journal_path": journal_path},
        ) from exc
    if payload is None:
        return False

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ApplyExportError(
            "apply_export.recovery_invalid",
            "Apply-export transaction journal has no valid entries.",
            details={"journal_path": journal_path},
        )
    record_norm = _normalized_path(root.record_path)
    export_root_norm = _normalized_path(root.canonical_path)
    workspace_root_norm = _normalized_path(workspace_root)
    game_root_norm = _normalized_path(game_root)
    package_dir_norm = _normalized_path(package_dir)
    if not _is_within(package_dir_norm, record_norm):
        raise ApplyExportError(
            "apply_export.recovery_escape",
            "Apply-export receipt path is outside the task artifact root.",
            details={"record_path": root.record_path, "package_root": package_dir},
        )
    for entry in entries:
        if not isinstance(entry, dict):
            raise ApplyExportError(
                "apply_export.recovery_invalid",
                "Apply-export transaction journal contains a non-object entry.",
                details={"journal_path": journal_path},
            )
        target = str(entry.get("target") or "")
        if not target or not os.path.isabs(target):
            raise ApplyExportError(
                "apply_export.recovery_invalid",
                "Apply-export transaction journal contains an invalid target.",
                details={"journal_path": journal_path},
            )
        target_norm = _normalized_path(target)
        in_export_scope = target_norm == export_root_norm or (
            target_norm != export_root_norm
            and _is_within(export_root_norm, target_norm)
        )
        in_workspace_scope = _is_within(game_root_norm, target_norm) and _is_within(
            workspace_root_norm,
            target_norm,
        )
        if not (target_norm == record_norm or in_export_scope or in_workspace_scope):
            raise ApplyExportError(
                "apply_export.recovery_escape",
                f"Apply-export transaction target is outside the managed scope: {target}.",
                details={"target": target, "journal_path": journal_path},
            )
        if os.path.lexists(target) and _is_link_or_reparse(target):
            raise ApplyExportError(
                "apply_export.recovery_link",
                f"Apply-export transaction target is a symbolic link or junction: {target}.",
                details={"target": target, "journal_path": journal_path},
            )
        target_directory = _normalized_path(os.path.dirname(_absolute_path(target)))
        _assert_no_link_components(
            os.path.dirname(_absolute_path(target)),
            "apply-export recovery parent",
        )
        for temporary_key in ("staged_path", "backup_path"):
            temporary = str(entry.get(temporary_key) or "")
            if not temporary:
                continue
            if (
                _normalized_path(os.path.dirname(_absolute_path(temporary)))
                != target_directory
            ):
                raise ApplyExportError(
                    "apply_export.recovery_escape",
                    f"Apply-export temporary path is outside its target directory: {temporary}.",
                    details={"target": target, "temporary_path": temporary},
                )
            if os.path.lexists(temporary) and _is_link_or_reparse(temporary):
                raise ApplyExportError(
                    "apply_export.recovery_link",
                    f"Apply-export temporary path is a symbolic link or junction: {temporary}.",
                    details={"temporary_path": temporary},
                )

    try:
        return recover_atomic_write_transaction(
            journal_path,
            expected_transaction_kind=APPLY_EXPORT_TRANSACTION_KIND,
            verify_targets=True,
        )
    except (AtomicWriteTransactionError, AtomicWritePreimageConflict) as exc:
        raise ApplyExportError(
            "apply_export.recovery_required",
            f"Apply-export transaction requires manual recovery: {exc}",
            details={
                "journal_path": journal_path,
                "export_root": root.canonical_path,
                "record_path": root.record_path,
                "recovery_state": "recovery_required",
            },
        ) from exc
    except Exception as exc:
        raise ApplyExportError(
            "apply_export.recovery_required",
            f"Apply-export transaction recovery failed: {exc}",
            details={
                "journal_path": journal_path,
                "export_root": root.canonical_path,
                "record_path": root.record_path,
                "recovery_state": "recovery_required",
            },
        ) from exc
