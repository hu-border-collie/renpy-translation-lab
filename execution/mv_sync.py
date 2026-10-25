"""Read-only MV bridge into the shared plan, durable results and writer.

All event interpretation and translation validation remain in the Adapter.
This module binds live source/target material to product artifacts and verifies
the real independent review record rather than manufacturing an approval.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import json
from pathlib import Path
from typing import Any, Mapping

from engine_adapters.contracts import InventoryPolicy, ProjectDiscoveryRequest, ValidatedTranslation
from engine_adapters.coverage import (build_coverage_report, default_coverage_review_path,
                                     digest_json, load_review_record, validate_review_record)
from engine_adapters.project_config import read_mv_project_config
from engine_adapters.rpgmaker_mv import RpgMakerMvAdapter
from engine_adapters.writeback import source_snapshot_fingerprint
from quality_review import translation_core, translation_plan


@dataclass(frozen=True)
class MvSyncSnapshot:
    """One immutable inventory, review binding and two material snapshots."""
    project: Any
    inventory: Any
    report: Any
    occurrences: tuple
    adapter: RpgMakerMvAdapter
    review_error: str = ""
    config_digest: str = ""
    include_files: tuple[str, ...] = ()
    include_prefixes: tuple[str, ...] = ()

    @cached_property
    def selected_occurrences(self):
        return tuple(row for row in self.occurrences if (
            (not self.include_files and not self.include_prefixes)
            or row.unit.file_rel_path in self.include_files
            or any(row.unit.file_rel_path.startswith(prefix) for prefix in self.include_prefixes)))

    @cached_property
    def pending_tasks_by_file(self):
        tasks: dict[str, list] = {}
        target_json = {}
        for row in self.selected_occurrences:
            mirrors_match = True
            mirrors = row.locator.locator.get('mirror_paths') or []
            if mirrors:
                relative = row.unit.file_rel_path
                if relative not in target_json:
                    doc = next(doc for doc in self.project.target_documents if doc.file_rel_path == relative)
                    target_json[relative] = json.loads(doc.content.decode('utf-8-sig'))
                for path in mirrors:
                    value = target_json[relative]
                    for component in path:
                        value = value[component]
                    mirrors_match = mirrors_match and value == row.unit.current_translation
            if (row.unit.current_translation.strip()
                    and row.unit.current_translation != row.unit.source_text
                    and mirrors_match
                    and self.adapter.validate_translation(row, row.unit.current_translation).status == 'pass'):
                continue
            tasks.setdefault(row.unit.file_rel_path, []).append(
                translation_core.legacy_item_from_unit(row.unit))
        return tasks

    @property
    def pending_task_count(self):
        return sum(len(items) for items in self.pending_tasks_by_file.values())


def build_snapshot(config: Mapping[str, Any], config_file: str | Path, *,
                   language: str = "schinese", require_review: bool = True) -> MvSyncSnapshot:
    """Audit the full material set before filtering requested narrative files.

    Unsupported structures remain visible as coverage attention. A current
    independent review authorizes this limited narrative scope; it never
    turns that report into a whole-game coverage or runtime acceptance claim.
    """
    settings = read_mv_project_config(config, config_file)
    adapter = RpgMakerMvAdapter(settings.window)
    project = adapter.discover_project(ProjectDiscoveryRequest(
        str(settings.source_root), str(settings.target_root), language))
    inventory = adapter.inventory_candidates(project, InventoryPolicy())
    report = build_coverage_report(project, inventory, adapter.audit_extraction(project, inventory),
                                   adapter_behavior_digest=adapter.behavior_digest())
    project = replace(project, coverage_digest=report.coverage_digest)
    review_error = ""
    try:
        record = load_review_record(default_coverage_review_path(settings.target_root))
        review = validate_review_record(record, report, inventory)
        project = replace(project, coverage_review_digest=review.coverage_review_digest)
        if not review.policy_satisfied:
            raise ValueError(f"MV scope review is {review.effective_status}; an independent current review is required.")
    except (OSError, ValueError) as exc:
        review_error = str(exc)
        if require_review:
            raise ValueError(f"MV scope review missing, invalid or stale: {exc}") from exc
    if report.coverage_status == "block":
        raise ValueError("MV narrative coverage is blocked; inspect coverage-status.")
    selected = [row.candidate_id for row in inventory.candidates if row.classification == "translatable"]
    occurrences = tuple(adapter.extract_occurrences(project, inventory, selected))
    def filters(key):
        values = config.get(key) or []
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f'MV {key} must be a list of relative paths.')
        normalized = tuple(sorted({value.strip().replace('\\', '/') for value in values if value.strip()}))
        if any(Path(value).is_absolute() or '..' in Path(value).parts or ':' in value for value in normalized):
            raise ValueError(f'MV {key} must stay within the content root.')
        return normalized
    include_files, include_prefixes = filters('include_files'), filters('include_prefixes')
    narrative_files = {doc.file_rel_path for doc in project.source_documents
                       if adapter._narrative_file(doc.file_rel_path)}
    if (set(include_files) - narrative_files
            or any(not any(path.startswith(prefix) for path in narrative_files) for prefix in include_prefixes)):
        raise ValueError('MV include filters must select existing narrative data files.')
    return MvSyncSnapshot(project, inventory, report, occurrences, adapter, review_error, digest_json(config),
                          include_files, include_prefixes)


def material_identity(snapshot: MvSyncSnapshot) -> dict:
    """Return non-secret digests and roots bound into every MV durable plan."""
    project = snapshot.project
    return {
        "engine": project.engine,
        "adapter_version": project.adapter_version,
        "source_root": project.project_root,
        "target_root": project.localization_root,
        "support_window_digest": project.support_window_digest,
        "coverage_digest": project.coverage_digest,
        "coverage_review_digest": project.coverage_review_digest,
        "config_digest": snapshot.config_digest,
        "source_files": {doc.file_rel_path: doc.sha256 for doc in project.source_documents},
        "target_files": {doc.file_rel_path: doc.sha256 for doc in project.target_documents},
    }


def source_identity(snapshot: MvSyncSnapshot) -> translation_plan.SourceIdentity:
    """Extend the shared source identity only for separate MV source/target copies."""
    project = snapshot.project
    return translation_plan.SourceIdentity(
        engine=project.engine, adapter_version=project.adapter_version,
        project_identity_digest=project.project_snapshot_fingerprint,
        source_snapshot_fingerprint=source_snapshot_fingerprint(project.source_documents),
        file_digests={doc.file_rel_path: doc.sha256 for doc in project.source_documents},
        material_identity=material_identity(snapshot))


def live_material_identity(snapshot: MvSyncSnapshot, config: Mapping[str, Any], config_file: str | Path) -> dict:
    """Hash current material before each dispatch without reinterpreting events."""
    settings = read_mv_project_config(config, config_file)
    current = material_identity(snapshot)
    current.update(source_root=str(settings.source_root), target_root=str(settings.target_root),
                   support_window_digest=settings.window.digest, config_digest=digest_json(config))
    review = validate_review_record(load_review_record(default_coverage_review_path(settings.target_root)),
                                    snapshot.report, snapshot.inventory)
    current['coverage_review_digest'] = review.coverage_review_digest
    for key, target in (('source_files', False), ('target_files', True)):
        current[key] = {doc.file_rel_path: doc.sha256 for doc in snapshot.adapter.read_live_documents(
            snapshot.project, target=target)}
    return current


def require_material_fresh(expected: Mapping[str, Any], current: Mapping[str, Any], *,
                           applied_files: Mapping[str, str] | None = None) -> None:
    """Allow only exact T0 material, or the checked bytes of an idempotent apply.

    Coverage identity includes the target snapshot, so after apply its old review
    is verified against the retained T0 snapshot by the caller. Everything else,
    including unmodified data, JS and the path set, still has to match exactly.
    """
    for key in ("source_root", "target_root", "support_window_digest", "source_files",
                "config_digest", "coverage_review_digest", "engine", "adapter_version"):
        if expected.get(key) != current.get(key):
            raise ValueError(f"MV frozen material changed: {key}.")
    before, live = dict(expected.get("target_files") or {}), dict(current.get("target_files") or {})
    if before.keys() != live.keys():
        raise ValueError("MV frozen target file set changed.")
    for path, digest in live.items():
        if digest not in {before[path], (applied_files or {}).get(path, "")}:
            raise ValueError(f"MV frozen target changed: {path}.")
    if not applied_files:
        for key in ("coverage_digest", "coverage_review_digest"):
            if not expected.get(key) or expected.get(key) != current.get(key):
                raise ValueError(f"MV frozen review changed: {key}.")


def build_plan(snapshot: MvSyncSnapshot, replacements: Mapping[str, Any]):
    by_id = {row.unit.id: row for row in snapshot.occurrences}
    validated = []
    for path, lines in replacements.items():
        for actions in lines.values():
            for action in actions:
                occurrence = by_id.get(str(action[6]))
                if (occurrence is None or occurrence.unit.file_rel_path != path
                        or action[5] != occurrence.unit.text):
                    raise ValueError("MV result occurrence does not match the frozen source.")
                translated = action[2]
                validated.append(ValidatedTranslation(occurrence, translated,
                    snapshot.adapter.validate_translation(occurrence, translated)))
    return snapshot.adapter.build_writeback_plan(
        snapshot.project, validated, snapshot.project.source_documents,
        live_targets=snapshot.project.target_documents)
