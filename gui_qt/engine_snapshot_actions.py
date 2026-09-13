"""Pure read-only loaders for the GUI engine/snapshot/reuse dialog (#424 P6).

All matching, reconciliation and reuse semantics stay in ``engine_adapters``;
this module only loads existing artifacts and shapes them for presentation.
It deliberately has no PySide6 import so the contract is CLI-testable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

MAX_DIFF_ITEMS = 500
MAX_REUSE_CANDIDATES = 500
EXCERPT_LIMIT = 160


class EngineSnapshotActionError(ValueError):
    """Raised when a snapshot/reuse artifact cannot be presented read-only."""


def snapshot_locator_text(locator: Any) -> str:
    """Return ``file:line`` for one occurrence locator mapping."""
    payload = locator.get("locator") if isinstance(locator, Mapping) else None
    if not isinstance(payload, Mapping):
        payload = locator if isinstance(locator, Mapping) else {}
    file_rel_path = str(payload.get("file_rel_path") or payload.get("path") or "")
    line = None
    for key in ("line", "line_hint", "line_number", "ordinal"):
        value = payload.get(key)
        if value is not None:
            line = str(value)
            break
    if not file_rel_path:
        return "(locator unavailable)"
    return f"{file_rel_path}:{line}" if line else file_rel_path


def excerpt_text(value: Any, limit: int = EXCERPT_LIMIT) -> str:
    text = str(value or "").replace("\r", "").replace("\n", "\\n").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _read_snapshot_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _snapshot_summary(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    coverage = manifest.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    game_version = manifest.get("game_version")
    game_version = game_version if isinstance(game_version, Mapping) else {}
    try:
        mtime = os.path.getmtime(manifest_path)
    except OSError:
        mtime = 0.0
    return {
        "name": manifest_path.parent.name,
        "path": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "mtime": float(mtime),
        "kind": str(manifest.get("kind") or ""),
        "project_snapshot_schema_version": int(
            manifest.get("project_snapshot_schema_version") or 0
        ),
        "engine": str(manifest.get("engine") or ""),
        "adapter_version": str(manifest.get("adapter_version") or ""),
        "localization_mode": str(manifest.get("localization_mode") or ""),
        "target_language": str(manifest.get("target_language") or ""),
        "version_id": str(game_version.get("version_id") or manifest_path.parent.name),
        "version_label": str(game_version.get("label") or ""),
        "source_revision": str(game_version.get("source_revision") or ""),
        "source_fingerprint": str(manifest.get("source_fingerprint") or ""),
        "project_snapshot_fingerprint": str(
            manifest.get("project_snapshot_fingerprint") or ""
        ),
        "generated_at": str(manifest.get("generated_at") or ""),
        "snapshot_digest": str(manifest.get("snapshot_digest") or ""),
        "occurrence_count": int(manifest.get("occurrence_count") or 0),
        "coverage_status": str(coverage.get("coverage_status") or ""),
        "review_status": str(coverage.get("review_status") or ""),
        "review_policy": str(coverage.get("review_policy") or ""),
        "review_policy_satisfied": bool(coverage.get("review_policy_satisfied")),
        "unresolved_findings": int(coverage.get("unresolved_findings") or 0),
    }


def discover_snapshot_manifests(snapshot_root: str) -> list[dict[str, Any]]:
    """Return valid snapshot manifest summaries under *snapshot_root*."""
    root = Path(str(snapshot_root or ""))
    if not root.is_dir():
        return []
    from engine_adapters import versioning

    snapshots: list[dict[str, Any]] = []
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError:
        return []
    for child in children:
        if not child.is_dir():
            continue
        manifest_path = child / versioning.DEFAULT_SNAPSHOT_FILENAME
        if not manifest_path.is_file():
            continue
        manifest = _read_snapshot_manifest(manifest_path)
        if manifest is None:
            continue
        snapshots.append(_snapshot_summary(manifest_path, manifest))
    snapshots.sort(key=lambda item: (item["mtime"], item["name"]), reverse=True)
    return snapshots


def collect_engine_snapshot_overview(
    *,
    game_root: str = "",
    snapshot_root: str = "",
) -> dict[str, Any]:
    """Return current adapter capabilities plus discovered snapshot summaries."""
    import gemini_translate_batch as batch_mod
    from engine_adapters.renpy import RenPyAdapter

    adapter = RenPyAdapter(legacy_module=batch_mod.legacy)
    capabilities = adapter.capabilities()
    root = str(snapshot_root or batch_mod.PROJECT_SNAPSHOTS_DIR or "")
    snapshots = discover_snapshot_manifests(root)
    review_path = ""
    if str(game_root or "").strip():
        try:
            review_path = str(
                batch_mod.resolve_coverage_review_path(str(game_root)) or ""
            )
        except Exception:
            review_path = ""
    return {
        "engine": adapter.engine,
        "adapter_version": adapter.adapter_version,
        "protocol_version": int(adapter.protocol_version or 0),
        "locator_schema_version": int(adapter.locator_schema_version or 0),
        "behavior_digest": str(adapter.behavior_digest() or ""),
        "capabilities": capabilities.to_dict(),
        "snapshot_root": root,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "latest_snapshot": snapshots[0] if snapshots else None,
        "coverage_review_path": review_path,
        "game_root": str(game_root or ""),
    }


def reconcile_snapshots(
    base_path: str,
    target_path: str,
    *,
    item_limit: int = MAX_DIFF_ITEMS,
) -> dict[str, Any]:
    """Load two saved snapshots and return a presentation-formatted diff."""
    base_source = str(base_path or "").strip()
    target_source = str(target_path or "").strip()
    if not base_source or not target_source:
        raise EngineSnapshotActionError("base 与 target 快照路径不能为空。")
    from engine_adapters import versioning

    try:
        base = versioning.load_project_snapshot(base_source)
        target = versioning.load_project_snapshot(target_source)
        report = versioning.reconcile_project_snapshots(base, target)
    except versioning.VersioningArtifactError as exc:
        raise EngineSnapshotActionError(str(exc)) from exc

    base_by_id = {item.occurrence_id: item for item in base.occurrences}
    target_by_id = {item.occurrence_id: item for item in target.occurrences}
    limit = max(0, int(item_limit or 0))
    items: list[dict[str, Any]] = []
    for item in report.items[:limit]:
        base_occurrence = base_by_id.get(item.base_occurrence_id)
        target_occurrence = target_by_id.get(item.target_occurrence_id)
        candidate_occurrences = [
            target_by_id[occurrence_id]
            for occurrence_id in item.candidate_target_occurrence_ids
            if occurrence_id in target_by_id
        ]
        items.append(
            {
                "item_id": str(item.item_id),
                "disposition": str(item.disposition),
                "match_kind": str(item.match_kind),
                "confidence": float(item.confidence or 0.0),
                "evidence": dict(item.evidence or {}),
                "base_occurrence_id": str(item.base_occurrence_id or ""),
                "target_occurrence_id": str(item.target_occurrence_id or ""),
                "base_locator": (
                    snapshot_locator_text(base_occurrence.locator)
                    if base_occurrence is not None
                    else ""
                ),
                "target_locator": (
                    snapshot_locator_text(target_occurrence.locator)
                    if target_occurrence is not None
                    else ""
                ),
                "candidate_locators": [
                    snapshot_locator_text(occurrence.locator)
                    for occurrence in candidate_occurrences
                ],
                "base_source": (
                    excerpt_text(base_occurrence.source_text)
                    if base_occurrence is not None
                    else ""
                ),
                "target_source": (
                    excerpt_text(target_occurrence.source_text)
                    if target_occurrence is not None
                    else ""
                ),
                "ambiguous": str(item.disposition) in {"ambiguous", "ambiguous_target"}
                or len(item.candidate_target_occurrence_ids) > 1,
            }
        )
    return {
        "base_path": base_source,
        "target_path": target_source,
        "base_version_id": str(report.base_version_id or ""),
        "target_version_id": str(report.target_version_id or ""),
        "status": str(report.status or ""),
        "summary": dict(report.summary or {}),
        "coverage_changes": dict(report.coverage_changes or {}),
        "reconciliation_digest": str(report.reconciliation_digest or ""),
        "item_count": len(report.items),
        "item_limit": limit,
        "items": items,
    }


def load_reuse_candidates_overview(
    path: str,
    *,
    candidate_limit: int = MAX_REUSE_CANDIDATES,
) -> dict[str, Any]:
    """Load one existing reuse package/report for read-only presentation."""
    source = str(path or "").strip()
    if not source:
        raise EngineSnapshotActionError("复用候选包路径不能为空。")
    from engine_adapters import reuse as engine_reuse

    try:
        candidate_set = engine_reuse.load_reuse_candidates(source)
    except engine_reuse.VersioningArtifactError as exc:
        raise EngineSnapshotActionError(str(exc)) from exc

    supplied = Path(source)
    report_path = (
        supplied / engine_reuse.DEFAULT_REUSE_REPORT_FILENAME
        if supplied.is_dir()
        else supplied
    )
    review_path = str(report_path.parent / engine_reuse.DEFAULT_REUSE_REVIEW_FILENAME)
    limit = max(0, int(candidate_limit or 0))
    candidates: list[dict[str, Any]] = []
    for candidate in candidate_set.candidates[:limit]:
        candidates.append(
            {
                "candidate_id": str(candidate.candidate_id),
                "reuse_class": str(candidate.reuse_class),
                "status": str(candidate.status),
                "confidence": float(candidate.confidence or 0.0),
                "reference_only": bool(candidate.reference_only),
                "has_translation_record": bool(candidate.has_translation_record),
                "reference_origin": str(candidate.reference_origin or ""),
                "base_version_id": str(candidate.base_version_id or ""),
                "target_version_id": str(candidate.target_version_id or ""),
                "base_occurrence_id": str(candidate.base_occurrence_id or ""),
                "target_occurrence_id": str(candidate.target_occurrence_id or ""),
                "candidate_target_occurrence_ids": list(
                    candidate.candidate_target_occurrence_ids or ()
                ),
                "reference_translation": excerpt_text(candidate.reference_translation),
                "effective_translation": excerpt_text(candidate.effective_translation),
                "evidence": dict(candidate.evidence or {}),
                "decision": dict(candidate.decision or {}),
                "audit_count": len(candidate.audit or ()),
            }
        )
    return {
        "path": str(report_path),
        "review_path": review_path,
        "review_exists": Path(review_path).is_file(),
        "status": str(candidate_set.status or ""),
        "stale_reasons": [str(item) for item in candidate_set.stale_reasons or ()],
        "summary": dict(candidate_set.summary or {}),
        "base_version_id": str(candidate_set.base_version_id or ""),
        "target_version_id": str(candidate_set.target_version_id or ""),
        "reconciliation_digest": str(candidate_set.reconciliation_digest or ""),
        "candidate_set_digest": str(candidate_set.candidate_set_digest or ""),
        "candidate_count": len(candidate_set.candidates),
        "candidate_limit": limit,
        "candidates": candidates,
    }
