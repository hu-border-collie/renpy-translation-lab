"""Safe P1 export-only storage for Batch writeback results.

This module owns only the destination tree, receipt, and export transaction.
It never reads a manifest, changes a game file, or advances translation state;
the Batch workflow supplies the already revalidated rendered bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from atomic_io import atomic_write_many_bytes, recover_atomic_write_transaction


EXPORT_RECORD_SCHEMA_VERSION = 1
EXPORT_TRANSACTION_KIND = "export_only"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


class ExportOnlyError(ValueError):
    """Raised when an export-only destination or receipt is unsafe to use."""

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
        record_path=os.path.join(package_dir_abs, "export_only_record.json"),
    )


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


def _load_record(record_path: str) -> dict[str, Any]:
    if not os.path.lexists(record_path):
        return {
            "schema_version": EXPORT_RECORD_SCHEMA_VERSION,
            "transaction_kind": EXPORT_TRANSACTION_KIND,
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
        or record.get("transaction_kind") != EXPORT_TRANSACTION_KIND
        or not isinstance(record.get("exports"), list)
    ):
        raise ExportOnlyError(
            "export_only.record_invalid",
            f"Export record has an unsupported shape: {record_path}.",
            details={"record_path": record_path},
        )
    return record


def _validate_record_entry(entry: Any) -> dict[str, Any]:
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
        or not isinstance(request_fingerprint, str)
        or len(request_fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in request_fingerprint)
        or not isinstance(operation_identity, str)
        or not operation_identity
        or not isinstance(exported_at, str)
        or not exported_at
        or not isinstance(files, list)
        or not isinstance(directories, list)
        or entry.get("transaction_kind") != EXPORT_TRANSACTION_KIND
    ):
        raise ExportOnlyError(
            "export_only.record_invalid",
            "Export record entry has an unsupported shape.",
        )
    if operation_identity != f"export-only:{request_fingerprint}":
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
            or not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_sha256)
            or not isinstance(output_sha256, str)
            or len(output_sha256) != 64
            or any(char not in "0123456789abcdef" for char in output_sha256)
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


def _find_export_record(record: dict[str, Any], root: ExportRoot) -> dict[str, Any] | None:
    for raw_entry in record["exports"]:
        entry = _validate_record_entry(raw_entry)
        stored_root = entry["export_root"]
        if _normalized_path(stored_root) == _normalized_path(root.canonical_path):
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
    os.makedirs(root.requested_path, exist_ok=True)
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
        if root_was_empty:
            _prune_empty_recovery_directories(root)
            if not root_existed:
                try:
                    os.rmdir(root.requested_path)
                except OSError:
                    pass
        if isinstance(exc, ExportOnlyError):
            raise
        recovery_required = os.path.lexists(journal_path)
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
