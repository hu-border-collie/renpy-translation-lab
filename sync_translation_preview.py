"""Preview artifacts and guarded apply for synchronous translation."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from atomic_io import atomic_write_json, atomic_write_text, file_sha256, sha256_text


SCHEMA = "sync_translation_preview"
VERSION = 1


def _canonical(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def _safe_relative_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"Unsafe sync preview relative path: {value!r}")
    return path.as_posix()


def _inside(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _artifact_path(package_dir: Path, value: Any) -> Path:
    relative = _safe_relative_path(value)
    candidate = package_dir.joinpath(*PurePosixPath(relative).parts)
    if not _inside(package_dir, candidate):
        raise ValueError(f"Sync preview artifact escapes its package: {value!r}")
    return candidate


def _fingerprint_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest.get("schema"),
        "version": manifest.get("version"),
        "created_at": manifest.get("created_at"),
        "project_root": manifest.get("project_root"),
        "tl_dir": manifest.get("tl_dir"),
        "report_path": manifest.get("report_path"),
        "report_sha256": manifest.get("report_sha256"),
        "summary": manifest.get("summary"),
        "files": manifest.get("files"),
    }


def _fingerprint(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(
        _fingerprint_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_sync_preview(
    *,
    log_dir: str | os.PathLike[str],
    project_root: str | os.PathLike[str],
    tl_dir: str | os.PathLike[str],
    files: Iterable[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Persist source/proposed snapshots, a unified diff, and a bound manifest."""
    created = datetime.now(timezone.utc)
    run_name = created.strftime("%Y%m%dT%H%M%S.%fZ")
    package_dir = Path(log_dir) / "sync_runs" / run_name
    package_dir.mkdir(parents=True, exist_ok=False)

    entries: list[dict[str, Any]] = []
    report_lines = [
        "# Synchronous translation preview\n\n",
        f"Created: {created.isoformat()}\n\n",
    ]
    total_items = 0

    for raw in files:
        relative_path = _safe_relative_path(raw.get("relative_path"))
        source_text = str(raw.get("source_text", ""))
        preview_text = str(raw.get("preview_text", ""))
        if source_text == preview_text:
            continue

        source_rel = f"source/{relative_path}"
        preview_rel = f"preview/{relative_path}"
        source_path = _artifact_path(package_dir, source_rel)
        preview_path = _artifact_path(package_dir, preview_rel)
        atomic_write_text(source_path, source_text, encoding="utf-8")
        atomic_write_text(preview_path, preview_text, encoding="utf-8")

        progress_entries = [str(item) for item in raw.get("progress_entries") or []]
        translated_items = int(raw.get("translated_items") or len(progress_entries))
        total_items += translated_items
        entries.append(
            {
                "relative_path": relative_path,
                "source_snapshot_path": source_rel,
                "preview_path": preview_rel,
                "source_sha256": str(raw.get("source_sha256") or sha256_text(source_text)),
                "source_snapshot_sha256": sha256_text(source_text),
                "preview_sha256": sha256_text(preview_text),
                "progress_entries": progress_entries,
                "translated_items": translated_items,
            }
        )
        report_lines.extend(
            difflib.unified_diff(
                source_text.splitlines(keepends=True),
                preview_text.splitlines(keepends=True),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
            )
        )
        if report_lines and not report_lines[-1].endswith("\n"):
            report_lines[-1] += "\n"
        report_lines.append("\n")

    report_path = package_dir / "preview.diff"
    atomic_write_text(report_path, "".join(report_lines), encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "state": "preview_ready",
        "created_at": created.isoformat(),
        "project_root": os.path.realpath(os.path.abspath(os.fspath(project_root))),
        "tl_dir": os.path.realpath(os.path.abspath(os.fspath(tl_dir))),
        "report_path": "preview.diff",
        "report_sha256": file_sha256(report_path),
        "summary": {
            "files_changed": len(entries),
            "translated_items": total_items,
        },
        "files": entries,
    }
    manifest["preview_fingerprint"] = _fingerprint(manifest)
    manifest_path = package_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest, ensure_ascii=False, indent=2)
    return str(manifest_path), manifest


def load_sync_preview(manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read sync preview manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Sync preview manifest must be a JSON object.")
    if manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
        raise ValueError("Unsupported sync preview manifest schema or version.")
    if not isinstance(manifest.get("files"), list):
        raise ValueError("Sync preview manifest files must be a list.")
    if manifest.get("preview_fingerprint") != _fingerprint(manifest):
        raise ValueError("Sync preview manifest changed after preview generation.")
    manifest["_manifest_path"] = str(path)
    return manifest


def prepare_sync_preview_apply(
    manifest_path: str | os.PathLike[str],
    *,
    active_project_root: str | os.PathLike[str],
    active_tl_dir: str | os.PathLike[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate every source and artifact before the first project write."""
    manifest = load_sync_preview(manifest_path)
    if manifest.get("state") == "applied":
        raise ValueError("Sync preview has already been applied.")
    if _canonical(manifest.get("project_root", "")) != _canonical(active_project_root):
        raise ValueError("Sync preview belongs to a different project.")
    if _canonical(manifest.get("tl_dir", "")) != _canonical(active_tl_dir):
        raise ValueError("Sync preview belongs to a different translation directory.")

    package_dir = Path(manifest["_manifest_path"]).parent
    report_path = _artifact_path(package_dir, manifest.get("report_path"))
    if not report_path.is_file() or file_sha256(report_path) != manifest.get("report_sha256"):
        raise ValueError("Sync preview diff report changed after preview generation.")
    target_root = Path(active_tl_dir).resolve()
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Sync preview file entry must be an object.")
        relative_path = _safe_relative_path(entry.get("relative_path"))
        if relative_path in seen:
            raise ValueError(f"Duplicate sync preview path: {relative_path}")
        seen.add(relative_path)
        target = target_root.joinpath(*PurePosixPath(relative_path).parts)
        if not _inside(target_root, target):
            raise ValueError(f"Sync preview target escapes TL_DIR: {relative_path}")
        source_snapshot = _artifact_path(package_dir, entry.get("source_snapshot_path"))
        preview_path = _artifact_path(package_dir, entry.get("preview_path"))
        if not target.is_file() or not source_snapshot.is_file() or not preview_path.is_file():
            raise ValueError(f"Sync preview file is missing: {relative_path}")
        if file_sha256(source_snapshot) != entry.get("source_snapshot_sha256"):
            raise ValueError(f"Sync preview source snapshot changed: {relative_path}")
        if file_sha256(preview_path) != entry.get("preview_sha256"):
            raise ValueError(f"Sync preview proposed file changed: {relative_path}")
        current_sha = file_sha256(target)
        source_sha = str(entry.get("source_sha256") or "")
        preview_sha = str(entry.get("preview_sha256") or "")
        if current_sha not in {source_sha, preview_sha}:
            raise ValueError(f"Source changed after sync preview: {relative_path}")
        preview_text = preview_path.read_text(encoding="utf-8")
        prepared.append(
            {
                "entry": entry,
                "target": target,
                "preview_text": preview_text,
                "already_applied": current_sha == preview_sha,
            }
        )
    return manifest, prepared


def apply_sync_preview(
    manifest_path: str | os.PathLike[str],
    *,
    active_project_root: str | os.PathLike[str],
    active_tl_dir: str | os.PathLike[str],
    on_file_applied: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    manifest, prepared = prepare_sync_preview_apply(
        manifest_path,
        active_project_root=active_project_root,
        active_tl_dir=active_tl_dir,
    )
    applied_paths: list[str] = []
    try:
        for item in prepared:
            entry = item["entry"]
            if not item["already_applied"]:
                atomic_write_text(item["target"], item["preview_text"], encoding="utf-8")
            applied_paths.append(entry["relative_path"])
            if on_file_applied is not None:
                on_file_applied(entry)
    except Exception as exc:
        manifest["state"] = "apply_failed"
        manifest["last_apply_failure"] = str(exc)
        manifest["partially_applied_files"] = applied_paths
        atomic_write_json(manifest["_manifest_path"], _public_manifest(manifest), ensure_ascii=False, indent=2)
        raise

    manifest["state"] = "applied"
    manifest["applied_at"] = datetime.now(timezone.utc).isoformat()
    manifest["applied_files"] = applied_paths
    manifest.pop("last_apply_failure", None)
    manifest.pop("partially_applied_files", None)
    atomic_write_json(manifest["_manifest_path"], _public_manifest(manifest), ensure_ascii=False, indent=2)
    return manifest


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if not key.startswith("_")}
