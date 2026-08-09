"""Versioned project snapshots and read-only cross-version reconciliation.

P3 persists engine-neutral occurrence evidence from an adapter scan without
turning RAG, manifests, or translated game files into a history database.  The
artifacts are portable JSON/JSONL packages.  Reconciliation only reports
candidate relationships; it never confirms lineage or emits writeback plans.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import unicodedata

from atomic_io import atomic_write_json, atomic_write_lines

from .contracts import Occurrence
from .coverage import (
    CoverageReport,
    build_review_template,
    digest_json,
    stable_json_dumps,
    validate_review_record,
)


GAME_VERSION_SCHEMA_VERSION = 1
UNIT_OCCURRENCE_SCHEMA_VERSION = 1
PROJECT_SNAPSHOT_SCHEMA_VERSION = 1
PROJECT_SNAPSHOT_DIGEST_SCHEMA_VERSION = 1
RECONCILIATION_ITEM_SCHEMA_VERSION = 1
RECONCILIATION_SCHEMA_VERSION = 1
RECONCILIATION_DIGEST_SCHEMA_VERSION = 1

PROJECT_SNAPSHOT_KIND = "project_snapshot"
RECONCILIATION_KIND = "project_snapshot_reconciliation"
DEFAULT_OCCURRENCES_FILENAME = "unit_occurrences.jsonl"
DEFAULT_SNAPSHOT_FILENAME = "project_snapshot.json"
DEFAULT_RECONCILIATION_ITEMS_FILENAME = "reconciliation_items.jsonl"
DEFAULT_RECONCILIATION_FILENAME = "reconciliation_report.json"
SOURCE_MODIFIED_MIN_SIMILARITY = 0.82
SOURCE_MODIFIED_MIN_MARGIN = 5.0
MAX_EXACT_GROUP_PAIR_CANDIDATES = 128
MAX_FUZZY_PAIR_CANDIDATES = 64
MAX_NGRAM_POSTING = 512
MAX_NGRAM_KEYS_PER_SOURCE = 16


class VersioningArtifactError(ValueError):
    """Raised when a snapshot or reconciliation artifact is invalid or stale."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text).strip()


def _normalize_path(value: Any) -> str:
    normalized = str(value or "").replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("/") or (
        len(normalized) >= 2
        and normalized[0].isalpha()
        and normalized[1] == ":"
    ):
        raise VersioningArtifactError(f"Invalid relative path: {value!r}")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise VersioningArtifactError(f"Invalid relative path: {value!r}")
    return "/".join(parts)


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VersioningArtifactError(f"{field_name} must be an object.")
    return dict(value)


def _sequence(value: Any, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise VersioningArtifactError(f"{field_name} must be a list.")
    return list(value)


def _required_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise VersioningArtifactError(f"{field_name} must not be empty.")
    return text


def _int_value(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise VersioningArtifactError(f"{field_name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise VersioningArtifactError(f"{field_name} must be an integer.") from exc
    if parsed < minimum:
        raise VersioningArtifactError(f"{field_name} must be >= {minimum}.")
    return parsed


def _float_value(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise VersioningArtifactError(f"{field_name} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VersioningArtifactError(
            f"{field_name} must be a finite number."
        ) from exc
    if not math.isfinite(parsed):
        raise VersioningArtifactError(f"{field_name} must be a finite number.")
    return parsed


def _artifact_path(package_dir: Path, relative_value: Any, *, field_name: str) -> Path:
    relative = _normalize_path(relative_value)
    root = package_dir.resolve()
    candidate = (root / Path(relative)).resolve()
    try:
        common = os.path.commonpath((os.fspath(root), os.fspath(candidate)))
    except ValueError as exc:
        raise VersioningArtifactError(f"{field_name} escapes the package directory.") from exc
    if os.path.normcase(common) != os.path.normcase(os.fspath(root)):
        raise VersioningArtifactError(f"{field_name} escapes the package directory.")
    return candidate


@dataclass(frozen=True)
class GameVersion:
    """User/project supplied identity for one scanned game version."""

    version_id: str
    label: str = ""
    source_revision: str = ""
    metadata: Mapping[str, Any] | None = None
    game_version_schema_version: int = GAME_VERSION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_version_schema_version": self.game_version_schema_version,
            "version_id": self.version_id,
            "label": self.label,
            "source_revision": self.source_revision,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GameVersion:
        payload = _mapping(value, field_name="game_version")
        schema = _int_value(
            payload.get("game_version_schema_version"),
            field_name="game_version_schema_version",
            minimum=1,
        )
        if schema != GAME_VERSION_SCHEMA_VERSION:
            raise VersioningArtifactError("Unsupported game version schema version.")
        return cls(
            version_id=_required_text(payload.get("version_id"), field_name="version_id"),
            label=str(payload.get("label") or ""),
            source_revision=str(payload.get("source_revision") or ""),
            metadata=_mapping(payload.get("metadata") or {}, field_name="game_version.metadata"),
            game_version_schema_version=schema,
        )


@dataclass(frozen=True)
class CoverageBinding:
    """Coverage and independent-review evidence frozen into a snapshot."""

    coverage_digest: str
    coverage_status: str
    coverage_schema_version: int
    inventory_digest: str
    source_fingerprint: str
    candidate_count: int
    classification_counts: Mapping[str, int]
    review_digest: str
    review_status: str
    review_policy: str
    review_policy_satisfied: bool
    unresolved_findings: int
    dependency_digest: str

    def stable_payload(self) -> dict[str, Any]:
        return {
            "coverage_digest": self.coverage_digest,
            "coverage_status": self.coverage_status,
            "coverage_schema_version": self.coverage_schema_version,
            "inventory_digest": self.inventory_digest,
            "source_fingerprint": self.source_fingerprint,
            "candidate_count": self.candidate_count,
            "classification_counts": dict(sorted(self.classification_counts.items())),
            "review_digest": self.review_digest,
            "review_status": self.review_status,
            "review_policy": self.review_policy,
            "review_policy_satisfied": self.review_policy_satisfied,
            "unresolved_findings": self.unresolved_findings,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.stable_payload(), "dependency_digest": self.dependency_digest}

    @classmethod
    def create(
        cls,
        *,
        coverage_digest: str,
        coverage_status: str,
        coverage_schema_version: int,
        inventory_digest: str,
        source_fingerprint: str,
        candidate_count: int,
        classification_counts: Mapping[str, int],
        review_digest: str,
        review_status: str,
        review_policy: str,
        review_policy_satisfied: bool,
        unresolved_findings: int,
    ) -> CoverageBinding:
        normalized_coverage_status = _required_text(
            coverage_status,
            field_name="coverage.coverage_status",
        )
        if normalized_coverage_status not in {"ready", "attention", "block", "stale"}:
            raise VersioningArtifactError(
                f"Unsupported coverage status: {normalized_coverage_status}"
            )
        normalized_review_status = _required_text(
            review_status,
            field_name="coverage.review_status",
        )
        if normalized_review_status not in {
            "pending",
            "agent_reviewed",
            "human_reviewed",
            "changes_requested",
            "stale",
        }:
            raise VersioningArtifactError(
                f"Unsupported coverage review status: {normalized_review_status}"
            )
        normalized_review_policy = _required_text(
            review_policy,
            field_name="coverage.review_policy",
        )
        if normalized_review_policy not in {"agent_or_human", "human_required"}:
            raise VersioningArtifactError(
                f"Unsupported coverage review policy: {normalized_review_policy}"
            )
        normalized_counts = {
            str(key): _int_value(
                count,
                field_name=f"coverage.classification_counts.{key}",
            )
            for key, count in classification_counts.items()
        }
        normalized_candidate_count = _int_value(
            candidate_count,
            field_name="coverage.candidate_count",
        )
        if sum(normalized_counts.values()) != normalized_candidate_count:
            raise VersioningArtifactError(
                "Coverage classification counts do not equal candidate_count."
            )
        normalized_unresolved = _int_value(
            unresolved_findings,
            field_name="coverage.unresolved_findings",
        )
        normalized_policy_satisfied = bool(review_policy_satisfied)
        if normalized_policy_satisfied and (
            normalized_review_status not in {"agent_reviewed", "human_reviewed"}
            or normalized_unresolved
            or (
                normalized_review_policy == "human_required"
                and normalized_review_status != "human_reviewed"
            )
        ):
            raise VersioningArtifactError(
                "Satisfied review policy requires a completed review with no findings."
            )
        provisional = cls(
            coverage_digest=_required_text(
                coverage_digest,
                field_name="coverage.coverage_digest",
            ),
            coverage_status=normalized_coverage_status,
            coverage_schema_version=_int_value(
                coverage_schema_version,
                field_name="coverage.coverage_schema_version",
                minimum=1,
            ),
            inventory_digest=_required_text(
                inventory_digest,
                field_name="coverage.inventory_digest",
            ),
            source_fingerprint=_required_text(
                source_fingerprint,
                field_name="coverage.source_fingerprint",
            ),
            candidate_count=normalized_candidate_count,
            classification_counts=normalized_counts,
            review_digest=_required_text(
                review_digest,
                field_name="coverage.review_digest",
            ),
            review_status=normalized_review_status,
            review_policy=normalized_review_policy,
            review_policy_satisfied=normalized_policy_satisfied,
            unresolved_findings=normalized_unresolved,
            dependency_digest="",
        )
        return replace(
            provisional,
            dependency_digest=digest_json(provisional.stable_payload()),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CoverageBinding:
        payload = _mapping(value, field_name="coverage")
        binding = cls.create(
            coverage_digest=str(payload.get("coverage_digest") or ""),
            coverage_status=str(payload.get("coverage_status") or ""),
            coverage_schema_version=_int_value(
                payload.get("coverage_schema_version"),
                field_name="coverage.coverage_schema_version",
                minimum=1,
            ),
            inventory_digest=str(payload.get("inventory_digest") or ""),
            source_fingerprint=str(payload.get("source_fingerprint") or ""),
            candidate_count=_int_value(
                payload.get("candidate_count"),
                field_name="coverage.candidate_count",
            ),
            classification_counts=_mapping(
                payload.get("classification_counts") or {},
                field_name="coverage.classification_counts",
            ),
            review_digest=str(payload.get("review_digest") or ""),
            review_status=str(payload.get("review_status") or ""),
            review_policy=str(payload.get("review_policy") or ""),
            review_policy_satisfied=bool(payload.get("review_policy_satisfied")),
            unresolved_findings=_int_value(
                payload.get("unresolved_findings"),
                field_name="coverage.unresolved_findings",
            ),
        )
        if str(payload.get("dependency_digest") or "") != binding.dependency_digest:
            raise VersioningArtifactError("Coverage dependency digest does not match its payload.")
        return binding


@dataclass(frozen=True)
class UnitOccurrenceRecord:
    """Portable, source-only version occurrence; it is not a translation record."""

    occurrence_id: str
    engine: str
    project_snapshot_fingerprint: str
    content_fingerprint: str
    content_fingerprint_schema_version: int
    candidate_id: str
    lineage_id: str
    locator: Mapping[str, Any]
    unit_id: str
    mode: str
    source_text: str
    speaker_id: str
    speaker_name: str
    file_rel_path: str
    line_number: int
    context_before: str
    context_after: str
    occurrence_digest: str
    unit_occurrence_schema_version: int = UNIT_OCCURRENCE_SCHEMA_VERSION

    def stable_payload(self) -> dict[str, Any]:
        return {
            "unit_occurrence_schema_version": self.unit_occurrence_schema_version,
            "occurrence_id": self.occurrence_id,
            "engine": self.engine,
            "project_snapshot_fingerprint": self.project_snapshot_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "content_fingerprint_schema_version": self.content_fingerprint_schema_version,
            "candidate_id": self.candidate_id,
            "lineage_id": self.lineage_id,
            "locator": dict(self.locator),
            "translation_unit": {
                "unit_id": self.unit_id,
                "mode": self.mode,
                "source_text": self.source_text,
                "speaker_id": self.speaker_id,
                "speaker_name": self.speaker_name,
                "file_rel_path": self.file_rel_path,
                "line_number": self.line_number,
            },
            "context": {
                "before": self.context_before,
                "after": self.context_after,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.stable_payload(), "occurrence_digest": self.occurrence_digest}

    @classmethod
    def create(
        cls,
        *,
        occurrence_id: str,
        engine: str,
        project_snapshot_fingerprint: str,
        content_fingerprint: str,
        content_fingerprint_schema_version: int,
        candidate_id: str,
        lineage_id: str,
        locator: Mapping[str, Any],
        unit_id: str,
        mode: str,
        source_text: str,
        speaker_id: str,
        speaker_name: str,
        file_rel_path: str,
        line_number: int,
        context_before: str,
        context_after: str,
    ) -> UnitOccurrenceRecord:
        normalized_locator = _mapping(locator, field_name="occurrence.locator")
        locator_engine = str(normalized_locator.get("engine") or "")
        normalized_engine = _required_text(engine, field_name="occurrence.engine")
        if locator_engine and locator_engine != normalized_engine:
            raise VersioningArtifactError(
                "Occurrence locator engine does not match occurrence engine."
            )
        provisional = cls(
            occurrence_id=_required_text(
                occurrence_id,
                field_name="occurrence.occurrence_id",
            ),
            engine=normalized_engine,
            project_snapshot_fingerprint=_required_text(
                project_snapshot_fingerprint,
                field_name="occurrence.project_snapshot_fingerprint",
            ),
            content_fingerprint=_required_text(
                content_fingerprint,
                field_name="occurrence.content_fingerprint",
            ),
            content_fingerprint_schema_version=_int_value(
                content_fingerprint_schema_version,
                field_name="occurrence.content_fingerprint_schema_version",
                minimum=1,
            ),
            candidate_id=_required_text(
                candidate_id,
                field_name="occurrence.candidate_id",
            ),
            lineage_id=str(lineage_id or "").strip(),
            locator=normalized_locator,
            unit_id=_required_text(unit_id, field_name="occurrence.translation_unit.unit_id"),
            mode=_required_text(mode, field_name="occurrence.translation_unit.mode"),
            source_text=str(source_text or ""),
            speaker_id=str(speaker_id or ""),
            speaker_name=str(speaker_name or ""),
            file_rel_path=_normalize_path(file_rel_path),
            line_number=_int_value(
                line_number,
                field_name="occurrence.translation_unit.line_number",
            ),
            context_before=str(context_before or ""),
            context_after=str(context_after or ""),
            occurrence_digest="",
        )
        return replace(
            provisional,
            occurrence_digest=digest_json(provisional.stable_payload()),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> UnitOccurrenceRecord:
        payload = _mapping(value, field_name="unit_occurrence")
        schema = _int_value(
            payload.get("unit_occurrence_schema_version"),
            field_name="unit_occurrence_schema_version",
            minimum=1,
        )
        if schema != UNIT_OCCURRENCE_SCHEMA_VERSION:
            raise VersioningArtifactError("Unsupported unit occurrence schema version.")
        unit = _mapping(payload.get("translation_unit"), field_name="translation_unit")
        context = _mapping(payload.get("context") or {}, field_name="context")
        record = cls.create(
            occurrence_id=str(payload.get("occurrence_id") or ""),
            engine=str(payload.get("engine") or ""),
            project_snapshot_fingerprint=str(
                payload.get("project_snapshot_fingerprint") or ""
            ),
            content_fingerprint=str(payload.get("content_fingerprint") or ""),
            content_fingerprint_schema_version=_int_value(
                payload.get("content_fingerprint_schema_version"),
                field_name="content_fingerprint_schema_version",
                minimum=1,
            ),
            candidate_id=str(payload.get("candidate_id") or ""),
            lineage_id=payload.get("lineage_id") or "",
            locator=_mapping(payload.get("locator"), field_name="locator"),
            unit_id=str(unit.get("unit_id") or ""),
            mode=str(unit.get("mode") or ""),
            source_text=unit.get("source_text") or "",
            speaker_id=unit.get("speaker_id") or "",
            speaker_name=unit.get("speaker_name") or "",
            file_rel_path=str(unit.get("file_rel_path") or ""),
            line_number=_int_value(
                unit.get("line_number"),
                field_name="translation_unit.line_number",
            ),
            context_before=context.get("before") or "",
            context_after=context.get("after") or "",
        )
        if str(payload.get("occurrence_digest") or "") != record.occurrence_digest:
            raise VersioningArtifactError(
                f"Occurrence digest does not match: {record.occurrence_id}"
            )
        return record


def _occurrence_sort_key(occurrence: Occurrence) -> tuple[Any, ...]:
    unit = occurrence.unit
    return (
        _normalize_path(unit.file_rel_path),
        int(unit.display_line_number or 0),
        int(unit.start or 0),
        occurrence.occurrence_id,
    )


def build_unit_occurrence_records(
    occurrences: Sequence[Occurrence],
    *,
    lineage_by_occurrence: Mapping[str, str] | None = None,
) -> tuple[UnitOccurrenceRecord, ...]:
    """Convert adapter occurrences into source-only, context-linked records."""

    ordered = sorted(
        ((_occurrence_sort_key(occurrence), occurrence) for occurrence in occurrences),
        key=lambda item: item[0],
    )
    lineage = dict(lineage_by_occurrence or {})
    records: list[UnitOccurrenceRecord] = []
    for index, (sort_key, occurrence) in enumerate(ordered):
        unit = occurrence.unit
        current_path = str(sort_key[0])
        before = ""
        after = ""
        if index > 0 and str(ordered[index - 1][0][0]) == current_path:
            before = ordered[index - 1][1].unit.source_text
        if index + 1 < len(ordered) and str(ordered[index + 1][0][0]) == current_path:
            after = ordered[index + 1][1].unit.source_text
        records.append(
            UnitOccurrenceRecord.create(
                occurrence_id=occurrence.occurrence_id,
                engine=occurrence.engine,
                project_snapshot_fingerprint=occurrence.project_snapshot_fingerprint,
                content_fingerprint=occurrence.content_fingerprint,
                content_fingerprint_schema_version=(
                    occurrence.content_fingerprint_schema_version
                ),
                candidate_id=occurrence.candidate_id,
                lineage_id=lineage.get(occurrence.occurrence_id, ""),
                locator=occurrence.locator.to_dict(),
                unit_id=unit.id,
                mode=unit.mode,
                source_text=unit.source_text,
                speaker_id=unit.speaker_id,
                speaker_name=unit.speaker_name,
                file_rel_path=unit.file_rel_path,
                line_number=unit.display_line_number,
                context_before=before,
                context_after=after,
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class ProjectSnapshot:
    """Validated in-memory form of a portable project snapshot package."""

    game_version: GameVersion
    engine: str
    adapter_version: str
    localization_mode: str
    target_language: str
    source_fingerprint: str
    project_snapshot_fingerprint: str
    source_files: tuple[Mapping[str, Any], ...]
    coverage: CoverageBinding
    occurrences: tuple[UnitOccurrenceRecord, ...]
    generated_at: str
    snapshot_digest: str
    project_snapshot_schema_version: int = PROJECT_SNAPSHOT_SCHEMA_VERSION
    project_snapshot_digest_schema_version: int = PROJECT_SNAPSHOT_DIGEST_SCHEMA_VERSION

    def stable_payload(self) -> dict[str, Any]:
        return {
            "project_snapshot_schema_version": self.project_snapshot_schema_version,
            "project_snapshot_digest_schema_version": (
                self.project_snapshot_digest_schema_version
            ),
            "game_version": self.game_version.to_dict(),
            "engine": self.engine,
            "adapter_version": self.adapter_version,
            "localization_mode": self.localization_mode,
            "target_language": self.target_language,
            "source_fingerprint": self.source_fingerprint,
            "project_snapshot_fingerprint": self.project_snapshot_fingerprint,
            "source_files": [dict(item) for item in self.source_files],
            "coverage": self.coverage.to_dict(),
            "occurrence_digests": [item.occurrence_digest for item in self.occurrences],
        }

    def to_manifest(
        self,
        *,
        occurrences_file: str = DEFAULT_OCCURRENCES_FILENAME,
    ) -> dict[str, Any]:
        return {
            "kind": PROJECT_SNAPSHOT_KIND,
            "project_snapshot_schema_version": self.project_snapshot_schema_version,
            **self.stable_payload(),
            "generated_at": self.generated_at,
            "occurrence_count": len(self.occurrences),
            "paths": {"occurrences": _normalize_path(occurrences_file)},
            "snapshot_digest": self.snapshot_digest,
        }


def _normalized_source_files(
    source_files: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in source_files:
        item = _mapping(raw, field_name="source_files item")
        rel_path = _normalize_path(item.get("file_rel_path"))
        if rel_path in seen:
            raise VersioningArtifactError(f"Duplicate source file: {rel_path}")
        seen.add(rel_path)
        normalized.append(
            {
                "file_rel_path": rel_path,
                "size": _int_value(item.get("size"), field_name=f"{rel_path}.size"),
                "sha256": _required_text(
                    item.get("sha256"),
                    field_name=f"{rel_path}.sha256",
                ),
            }
        )
    return tuple(sorted(normalized, key=lambda item: item["file_rel_path"]))


def create_project_snapshot(
    game_version: GameVersion,
    *,
    engine: str,
    adapter_version: str,
    localization_mode: str,
    target_language: str,
    source_fingerprint: str,
    project_snapshot_fingerprint: str,
    source_files: Sequence[Mapping[str, Any]],
    coverage: CoverageBinding,
    occurrences: Sequence[Occurrence],
    lineage_by_occurrence: Mapping[str, str] | None = None,
    generated_at: str | None = None,
) -> ProjectSnapshot:
    """Create and validate a deterministic snapshot from adapter occurrences."""

    version = GameVersion.from_dict(game_version.to_dict())
    normalized_engine = _required_text(engine, field_name="engine")
    normalized_project_fingerprint = _required_text(
        project_snapshot_fingerprint,
        field_name="project_snapshot_fingerprint",
    )
    if coverage.source_fingerprint != source_fingerprint:
        raise VersioningArtifactError(
            "Coverage source fingerprint does not match the project snapshot."
        )
    for occurrence in occurrences:
        if occurrence.engine != normalized_engine:
            raise VersioningArtifactError(
                f"Occurrence engine does not match project engine: {occurrence.occurrence_id}"
            )
        if occurrence.project_snapshot_fingerprint != normalized_project_fingerprint:
            raise VersioningArtifactError(
                "Occurrence project snapshot fingerprint does not match: "
                f"{occurrence.occurrence_id}"
            )
    records = build_unit_occurrence_records(
        occurrences,
        lineage_by_occurrence=lineage_by_occurrence,
    )
    provisional = ProjectSnapshot(
        game_version=version,
        engine=normalized_engine,
        adapter_version=_required_text(adapter_version, field_name="adapter_version"),
        localization_mode=_required_text(
            localization_mode,
            field_name="localization_mode",
        ),
        target_language=str(target_language or ""),
        source_fingerprint=_required_text(
            source_fingerprint,
            field_name="source_fingerprint",
        ),
        project_snapshot_fingerprint=normalized_project_fingerprint,
        source_files=_normalized_source_files(source_files),
        coverage=coverage,
        occurrences=records,
        generated_at=generated_at or _utc_now(),
        snapshot_digest="",
    )
    snapshot = replace(
        provisional,
        snapshot_digest=digest_json(provisional.stable_payload()),
    )
    validate_project_snapshot(snapshot)
    return snapshot


def build_project_snapshot(
    translation_snapshot: Any,
    game_version: GameVersion,
    *,
    coverage_review: Mapping[str, Any] | None = None,
    lineage_by_occurrence: Mapping[str, str] | None = None,
    generated_at: str | None = None,
) -> ProjectSnapshot:
    """Build a P3 snapshot from the existing adapter scan and coverage report."""

    report: CoverageReport = translation_snapshot.report
    review_record = dict(coverage_review or build_review_template(
        report,
        review_policy=translation_snapshot.review_policy,
    ))
    review = validate_review_record(
        review_record,
        report,
        translation_snapshot.inventory,
    )
    coverage = CoverageBinding.create(
        coverage_digest=report.coverage_digest,
        coverage_status=report.coverage_status,
        coverage_schema_version=report.coverage_schema_version,
        inventory_digest=report.inventory_digest,
        source_fingerprint=report.source_fingerprint,
        candidate_count=report.candidate_count,
        classification_counts=report.classification_counts,
        review_digest=review.coverage_review_digest,
        review_status=review.effective_status,
        review_policy=review.review_policy,
        review_policy_satisfied=review.policy_satisfied,
        unresolved_findings=review.unresolved_findings,
    )
    project = translation_snapshot.project
    return create_project_snapshot(
        game_version,
        engine=project.engine,
        adapter_version=project.adapter_version,
        localization_mode=project.localization_mode.value,
        target_language=project.target_language,
        source_fingerprint=project.source_fingerprint,
        project_snapshot_fingerprint=project.project_snapshot_fingerprint,
        source_files=[document.manifest_entry() for document in project.source_documents],
        coverage=coverage,
        occurrences=translation_snapshot.occurrences,
        lineage_by_occurrence=lineage_by_occurrence,
        generated_at=generated_at,
    )


def validate_project_snapshot(snapshot: ProjectSnapshot) -> None:
    """Fail closed when snapshot schemas, identities, or digests disagree."""

    if snapshot.project_snapshot_schema_version != PROJECT_SNAPSHOT_SCHEMA_VERSION:
        raise VersioningArtifactError("Unsupported project snapshot schema version.")
    if (
        snapshot.project_snapshot_digest_schema_version
        != PROJECT_SNAPSHOT_DIGEST_SCHEMA_VERSION
    ):
        raise VersioningArtifactError("Unsupported project snapshot digest schema version.")
    if snapshot.coverage.source_fingerprint != snapshot.source_fingerprint:
        raise VersioningArtifactError("Snapshot coverage source fingerprint does not match.")
    occurrence_ids: set[str] = set()
    candidate_ids: set[str] = set()
    locator_keys: set[str] = set()
    lineage_ids: set[str] = set()
    for occurrence in snapshot.occurrences:
        if occurrence.occurrence_id in occurrence_ids:
            raise VersioningArtifactError(
                f"Duplicate occurrence_id: {occurrence.occurrence_id}"
            )
        if occurrence.candidate_id in candidate_ids:
            raise VersioningArtifactError(
                f"Duplicate occurrence candidate_id: {occurrence.candidate_id}"
            )
        occurrence_ids.add(occurrence.occurrence_id)
        candidate_ids.add(occurrence.candidate_id)
        locator_key = stable_json_dumps(occurrence.locator)
        if locator_key in locator_keys:
            raise VersioningArtifactError(
                f"Duplicate occurrence locator: {occurrence.occurrence_id}"
            )
        locator_keys.add(locator_key)
        if occurrence.lineage_id:
            if occurrence.lineage_id in lineage_ids:
                raise VersioningArtifactError(
                    f"Duplicate confirmed lineage_id: {occurrence.lineage_id}"
                )
            lineage_ids.add(occurrence.lineage_id)
        if occurrence.engine != snapshot.engine:
            raise VersioningArtifactError(
                f"Occurrence engine mismatch: {occurrence.occurrence_id}"
            )
        if (
            occurrence.project_snapshot_fingerprint
            != snapshot.project_snapshot_fingerprint
        ):
            raise VersioningArtifactError(
                f"Occurrence project fingerprint mismatch: {occurrence.occurrence_id}"
            )
        if digest_json(occurrence.stable_payload()) != occurrence.occurrence_digest:
            raise VersioningArtifactError(
                f"Occurrence digest mismatch: {occurrence.occurrence_id}"
            )
    expected = digest_json(snapshot.stable_payload())
    if expected != snapshot.snapshot_digest:
        raise VersioningArtifactError("Project snapshot digest does not match its payload.")


@dataclass(frozen=True)
class SnapshotPackagePaths:
    package_dir: str
    snapshot_path: str
    occurrences_path: str


def export_project_snapshot(
    snapshot: ProjectSnapshot,
    output_dir: str | os.PathLike[str],
) -> SnapshotPackagePaths:
    """Atomically export a validated snapshot package without touching game files."""

    validate_project_snapshot(snapshot)
    package_dir = Path(output_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    occurrences_path = package_dir / DEFAULT_OCCURRENCES_FILENAME
    snapshot_path = package_dir / DEFAULT_SNAPSHOT_FILENAME
    atomic_write_lines(
        occurrences_path,
        (
            stable_json_dumps(item.to_dict()) + "\n"
            for item in snapshot.occurrences
        ),
        encoding="utf-8",
    )
    atomic_write_json(
        snapshot_path,
        snapshot.to_manifest(occurrences_file=DEFAULT_OCCURRENCES_FILENAME),
        ensure_ascii=False,
        indent=2,
    )
    return SnapshotPackagePaths(
        package_dir=str(package_dir.resolve()),
        snapshot_path=str(snapshot_path.resolve()),
        occurrences_path=str(occurrences_path.resolve()),
    )


def _read_json_object(path: Path, *, artifact_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VersioningArtifactError(f"Could not read {artifact_name}: {exc}") from exc
    if not isinstance(value, dict):
        raise VersioningArtifactError(f"{artifact_name} must be a JSON object.")
    return value


def _read_jsonl_objects(path: Path, *, artifact_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise VersioningArtifactError(
                        f"Invalid {artifact_name} JSON at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise VersioningArtifactError(
                        f"{artifact_name} line {line_number} must be an object."
                    )
                rows.append(value)
    except (OSError, UnicodeError) as exc:
        raise VersioningArtifactError(f"Could not read {artifact_name}: {exc}") from exc
    return rows


def load_project_snapshot(path: str | os.PathLike[str]) -> ProjectSnapshot:
    """Load and fully validate a project snapshot directory or manifest path."""

    supplied = Path(path)
    snapshot_path = supplied / DEFAULT_SNAPSHOT_FILENAME if supplied.is_dir() else supplied
    manifest = _read_json_object(snapshot_path, artifact_name="project snapshot")
    if str(manifest.get("kind") or "") != PROJECT_SNAPSHOT_KIND:
        raise VersioningArtifactError("Not a project snapshot artifact.")
    schema = _int_value(
        manifest.get("project_snapshot_schema_version"),
        field_name="project_snapshot_schema_version",
        minimum=1,
    )
    if schema != PROJECT_SNAPSHOT_SCHEMA_VERSION:
        raise VersioningArtifactError("Unsupported project snapshot schema version.")
    digest_schema = _int_value(
        manifest.get("project_snapshot_digest_schema_version"),
        field_name="project_snapshot_digest_schema_version",
        minimum=1,
    )
    if digest_schema != PROJECT_SNAPSHOT_DIGEST_SCHEMA_VERSION:
        raise VersioningArtifactError("Unsupported project snapshot digest schema version.")
    paths = _mapping(manifest.get("paths"), field_name="paths")
    occurrences_path = _artifact_path(
        snapshot_path.parent,
        paths.get("occurrences"),
        field_name="paths.occurrences",
    )
    rows = _read_jsonl_objects(
        occurrences_path,
        artifact_name="unit occurrences",
    )
    occurrences = tuple(UnitOccurrenceRecord.from_dict(row) for row in rows)
    expected_count = _int_value(
        manifest.get("occurrence_count"),
        field_name="occurrence_count",
    )
    if expected_count != len(occurrences):
        raise VersioningArtifactError("Project snapshot occurrence count does not match JSONL.")
    snapshot = ProjectSnapshot(
        game_version=GameVersion.from_dict(
            _mapping(manifest.get("game_version"), field_name="game_version")
        ),
        engine=_required_text(manifest.get("engine"), field_name="engine"),
        adapter_version=_required_text(
            manifest.get("adapter_version"),
            field_name="adapter_version",
        ),
        localization_mode=_required_text(
            manifest.get("localization_mode"),
            field_name="localization_mode",
        ),
        target_language=str(manifest.get("target_language") or ""),
        source_fingerprint=_required_text(
            manifest.get("source_fingerprint"),
            field_name="source_fingerprint",
        ),
        project_snapshot_fingerprint=_required_text(
            manifest.get("project_snapshot_fingerprint"),
            field_name="project_snapshot_fingerprint",
        ),
        source_files=_normalized_source_files(
            _sequence(manifest.get("source_files"), field_name="source_files")
        ),
        coverage=CoverageBinding.from_dict(
            _mapping(manifest.get("coverage"), field_name="coverage")
        ),
        occurrences=occurrences,
        generated_at=str(manifest.get("generated_at") or ""),
        snapshot_digest=_required_text(
            manifest.get("snapshot_digest"),
            field_name="snapshot_digest",
        ),
        project_snapshot_schema_version=schema,
        project_snapshot_digest_schema_version=digest_schema,
    )
    validate_project_snapshot(snapshot)
    return snapshot


@dataclass(frozen=True)
class _MatchEvidence:
    base_id: str
    target_id: str
    match_kind: str
    rank: int
    score: float
    confidence: float
    evidence: Mapping[str, Any]


def _speaker_key(item: UnitOccurrenceRecord) -> str:
    return _normalize_text(item.speaker_id or item.speaker_name)


def _context_matches(
    base: UnitOccurrenceRecord,
    target: UnitOccurrenceRecord,
) -> tuple[bool, bool]:
    base_source = _normalize_text(base.source_text)
    target_source = _normalize_text(target.source_text)
    before = _normalize_text(base.context_before)
    after = _normalize_text(base.context_after)
    target_before = _normalize_text(target.context_before)
    target_after = _normalize_text(target.context_after)
    return (
        bool(before)
        and before not in {base_source, target_source}
        and before == target_before,
        bool(after)
        and after not in {base_source, target_source}
        and after == target_after,
    )


def _pair_evidence(
    base: UnitOccurrenceRecord,
    target: UnitOccurrenceRecord,
) -> _MatchEvidence | None:
    if base.engine != target.engine:
        return None
    base_source = _normalize_text(base.source_text)
    target_source = _normalize_text(target.source_text)
    speaker_match = bool(_speaker_key(base)) and _speaker_key(base) == _speaker_key(target)
    before_match, after_match = _context_matches(base, target)
    common = {
        "source_equal": base_source == target_source,
        "speaker_equal": speaker_match,
        "context_before_equal": before_match,
        "context_after_equal": after_match,
        "file_moved": base.file_rel_path != target.file_rel_path,
        "line_changed": base.line_number != target.line_number,
    }
    if base.lineage_id and base.lineage_id == target.lineage_id:
        return _MatchEvidence(
            base.occurrence_id,
            target.occurrence_id,
            "confirmed_lineage",
            600,
            600.0,
            1.0,
            {**common, "lineage_id": base.lineage_id},
        )
    if stable_json_dumps(base.locator) == stable_json_dumps(target.locator):
        return _MatchEvidence(
            base.occurrence_id,
            target.occurrence_id,
            "locator_exact",
            500,
            500.0,
            1.0,
            common,
        )
    if base.content_fingerprint == target.content_fingerprint:
        return _MatchEvidence(
            base.occurrence_id,
            target.occurrence_id,
            "content_exact",
            450,
            450.0,
            1.0,
            common,
        )
    if base_source and base_source == target_source:
        score = 300.0
        score += 30.0 if speaker_match else 0.0
        score += 12.0 if before_match else 0.0
        score += 12.0 if after_match else 0.0
        score += 4.0 if base.mode == target.mode else 0.0
        return _MatchEvidence(
            base.occurrence_id,
            target.occurrence_id,
            "moved_exact",
            300,
            score,
            0.98 if speaker_match or before_match or after_match else 0.93,
            common,
        )
    if not base_source or not target_source or max(len(base_source), len(target_source)) < 4:
        return None
    similarity = SequenceMatcher(None, base_source, target_source, autojunk=False).ratio()
    if similarity < SOURCE_MODIFIED_MIN_SIMILARITY:
        return None
    score = 200.0 + similarity * 100.0
    score += 8.0 if speaker_match else 0.0
    score += 4.0 if before_match else 0.0
    score += 4.0 if after_match else 0.0
    confidence = min(
        0.99,
        similarity * 0.88
        + (0.05 if speaker_match else 0.0)
        + (0.025 if before_match else 0.0)
        + (0.025 if after_match else 0.0),
    )
    return _MatchEvidence(
        base.occurrence_id,
        target.occurrence_id,
        "source_modified",
        200,
        score,
        confidence,
        {**common, "source_similarity": round(similarity, 6)},
    )


def _source_ngrams(value: str, size: int = 3) -> tuple[str, ...]:
    text = _normalize_text(value)
    if len(text) < size:
        return ()
    return tuple(
        sorted(
            {
                text[index : index + size]
                for index in range(0, len(text) - size + 1)
                if text[index : index + size].strip()
            }
        )
    )


def _bounded_exact_group(
    base: UnitOccurrenceRecord,
    target_ids: Sequence[str],
    target_by_id: Mapping[str, UnitOccurrenceRecord],
) -> tuple[str, ...]:
    ordered = tuple(sorted(target_ids))
    if len(ordered) <= MAX_EXACT_GROUP_PAIR_CANDIDATES:
        return ordered
    distinctive = []
    for target_id in ordered:
        target = target_by_id[target_id]
        before_match, after_match = _context_matches(base, target)
        if (
            (_speaker_key(base) and _speaker_key(base) == _speaker_key(target))
            or before_match
            or after_match
        ):
            distinctive.append(target_id)
    selected = distinctive[:MAX_FUZZY_PAIR_CANDIDATES]
    chosen = set(selected)
    for target_id in ordered:
        if len(selected) >= MAX_EXACT_GROUP_PAIR_CANDIDATES:
            break
        if target_id in chosen:
            continue
        chosen.add(target_id)
        selected.append(target_id)
    return tuple(selected)


def _unique_best(
    candidates: Sequence[_MatchEvidence],
    *,
    rank: int,
) -> _MatchEvidence | None:
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda item: (-item.score, item.base_id, item.target_id))
    strongest = ordered[0]
    duplicate_source = bool(
        int(strongest.evidence.get("base_source_occurrences") or 0) > 1
        or int(strongest.evidence.get("target_source_occurrences") or 0) > 1
    )
    discriminating_context = any(
        bool(strongest.evidence.get(key))
        for key in (
            "speaker_equal",
            "context_before_equal",
            "context_after_equal",
        )
    )
    if rank in {300, 200} and duplicate_source and not discriminating_context:
        return None
    duplicate_content = bool(
        int(strongest.evidence.get("base_content_occurrences") or 0) > 1
        or int(strongest.evidence.get("target_content_occurrences") or 0) > 1
    )
    if rank == 450 and duplicate_content:
        return None
    if len(ordered) == 1:
        return strongest
    margin = SOURCE_MODIFIED_MIN_MARGIN if rank == 200 else 0.0
    if ordered[0].score - ordered[1].score <= margin:
        return None
    return strongest


@dataclass(frozen=True)
class ReconciliationItem:
    item_id: str
    disposition: str
    match_kind: str
    base_occurrence_id: str
    target_occurrence_id: str
    candidate_target_occurrence_ids: tuple[str, ...]
    confidence: float
    evidence: Mapping[str, Any]
    item_digest: str
    reconciliation_item_schema_version: int = RECONCILIATION_ITEM_SCHEMA_VERSION

    def stable_payload(self) -> dict[str, Any]:
        return {
            "reconciliation_item_schema_version": self.reconciliation_item_schema_version,
            "item_id": self.item_id,
            "disposition": self.disposition,
            "match_kind": self.match_kind,
            "base_occurrence_id": self.base_occurrence_id,
            "target_occurrence_id": self.target_occurrence_id,
            "candidate_target_occurrence_ids": list(
                self.candidate_target_occurrence_ids
            ),
            "confidence": round(
                _float_value(self.confidence, field_name="confidence"),
                6,
            ),
            "evidence": dict(self.evidence),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.stable_payload(), "item_digest": self.item_digest}

    @classmethod
    def create(
        cls,
        *,
        disposition: str,
        match_kind: str,
        base_occurrence_id: str = "",
        target_occurrence_id: str = "",
        candidate_target_occurrence_ids: Sequence[str] = (),
        confidence: float = 0.0,
        evidence: Mapping[str, Any] | None = None,
    ) -> ReconciliationItem:
        normalized_disposition = _required_text(
            disposition,
            field_name="disposition",
        )
        if normalized_disposition not in {
            "matched",
            "ambiguous",
            "ambiguous_target",
            "deleted",
            "added",
        }:
            raise VersioningArtifactError(
                f"Unsupported reconciliation disposition: {normalized_disposition}"
            )
        normalized_base_id = str(base_occurrence_id or "")
        normalized_target_id = str(target_occurrence_id or "")
        normalized_candidate_ids = tuple(sorted(set(candidate_target_occurrence_ids)))
        normalized_match_kind = str(match_kind or "")
        if normalized_disposition == "matched" and not (
            normalized_base_id and normalized_target_id and normalized_match_kind
        ):
            raise VersioningArtifactError(
                "Matched reconciliation items require base, target, and match kind."
            )
        if normalized_disposition == "ambiguous" and not (
            normalized_base_id and normalized_candidate_ids
        ):
            raise VersioningArtifactError(
                "Ambiguous reconciliation items require base and candidate targets."
            )
        if normalized_disposition == "ambiguous_target" and not normalized_target_id:
            raise VersioningArtifactError(
                "Ambiguous target items require a target ID."
            )
        if normalized_disposition == "deleted" and not normalized_base_id:
            raise VersioningArtifactError("Deleted reconciliation items require a base ID.")
        if normalized_disposition == "added" and not normalized_target_id:
            raise VersioningArtifactError("Added reconciliation items require a target ID.")
        if normalized_disposition == "matched" and normalized_candidate_ids:
            raise VersioningArtifactError(
                "Matched reconciliation items cannot carry candidate targets."
            )
        if normalized_disposition == "ambiguous" and normalized_target_id:
            raise VersioningArtifactError(
                "Ambiguous reconciliation items cannot carry a confirmed target."
            )
        if normalized_disposition == "ambiguous_target" and (
            normalized_base_id or normalized_candidate_ids
        ):
            raise VersioningArtifactError(
                "Ambiguous target items cannot carry base or candidate IDs."
            )
        if normalized_disposition == "deleted" and (
            normalized_target_id or normalized_candidate_ids
        ):
            raise VersioningArtifactError(
                "Deleted reconciliation items cannot carry target IDs."
            )
        if normalized_disposition == "added" and (
            normalized_base_id or normalized_candidate_ids
        ):
            raise VersioningArtifactError(
                "Added reconciliation items cannot carry base or candidate IDs."
            )
        normalized_confidence = round(
            _float_value(confidence, field_name="confidence"),
            6,
        )
        if not 0.0 <= normalized_confidence <= 1.0:
            raise VersioningArtifactError("Reconciliation confidence must be between 0 and 1.")
        identity = {
            "disposition": normalized_disposition,
            "base_occurrence_id": normalized_base_id,
            "target_occurrence_id": normalized_target_id,
            "candidate_target_occurrence_ids": list(normalized_candidate_ids),
        }
        item_id = "recon-item1:" + digest_json(identity)
        provisional = cls(
            item_id=item_id,
            disposition=normalized_disposition,
            match_kind=normalized_match_kind,
            base_occurrence_id=normalized_base_id,
            target_occurrence_id=normalized_target_id,
            candidate_target_occurrence_ids=normalized_candidate_ids,
            confidence=normalized_confidence,
            evidence=dict(evidence or {}),
            item_digest="",
        )
        return replace(
            provisional,
            item_digest=digest_json(provisional.stable_payload()),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReconciliationItem:
        payload = _mapping(value, field_name="reconciliation_item")
        schema = _int_value(
            payload.get("reconciliation_item_schema_version"),
            field_name="reconciliation_item_schema_version",
            minimum=1,
        )
        if schema != RECONCILIATION_ITEM_SCHEMA_VERSION:
            raise VersioningArtifactError("Unsupported reconciliation item schema version.")
        item = cls.create(
            disposition=str(payload.get("disposition") or ""),
            match_kind=payload.get("match_kind") or "",
            base_occurrence_id=payload.get("base_occurrence_id") or "",
            target_occurrence_id=payload.get("target_occurrence_id") or "",
            candidate_target_occurrence_ids=[
                str(item)
                for item in _sequence(
                    payload.get("candidate_target_occurrence_ids") or [],
                    field_name="candidate_target_occurrence_ids",
                )
            ],
            confidence=_float_value(
                payload.get("confidence"),
                field_name="confidence",
            ),
            evidence=_mapping(payload.get("evidence") or {}, field_name="evidence"),
        )
        if str(payload.get("item_id") or "") != item.item_id:
            raise VersioningArtifactError("Reconciliation item identity does not match.")
        if str(payload.get("item_digest") or "") != item.item_digest:
            raise VersioningArtifactError("Reconciliation item digest does not match.")
        return item


@dataclass(frozen=True)
class ReconciliationReport:
    base_snapshot_digest: str
    target_snapshot_digest: str
    base_version_id: str
    target_version_id: str
    base_coverage_dependency_digest: str
    target_coverage_dependency_digest: str
    status: str
    summary: Mapping[str, int]
    coverage_changes: Mapping[str, Any]
    items: tuple[ReconciliationItem, ...]
    generated_at: str
    reconciliation_digest: str
    reconciliation_schema_version: int = RECONCILIATION_SCHEMA_VERSION
    reconciliation_digest_schema_version: int = RECONCILIATION_DIGEST_SCHEMA_VERSION

    def stable_payload(self) -> dict[str, Any]:
        return {
            "reconciliation_schema_version": self.reconciliation_schema_version,
            "reconciliation_digest_schema_version": (
                self.reconciliation_digest_schema_version
            ),
            "inputs": {
                "base_snapshot_digest": self.base_snapshot_digest,
                "target_snapshot_digest": self.target_snapshot_digest,
                "base_version_id": self.base_version_id,
                "target_version_id": self.target_version_id,
                "base_coverage_dependency_digest": (
                    self.base_coverage_dependency_digest
                ),
                "target_coverage_dependency_digest": (
                    self.target_coverage_dependency_digest
                ),
            },
            "status": self.status,
            "summary": dict(sorted(self.summary.items())),
            "coverage_changes": dict(self.coverage_changes),
            "item_digests": [item.item_digest for item in self.items],
        }

    def to_manifest(
        self,
        *,
        items_file: str = DEFAULT_RECONCILIATION_ITEMS_FILENAME,
    ) -> dict[str, Any]:
        return {
            "kind": RECONCILIATION_KIND,
            "reconciliation_schema_version": self.reconciliation_schema_version,
            **self.stable_payload(),
            "generated_at": self.generated_at,
            "item_count": len(self.items),
            "paths": {"items": _normalize_path(items_file)},
            "reconciliation_digest": self.reconciliation_digest,
        }


def _coverage_changes(
    base: ProjectSnapshot,
    target: ProjectSnapshot,
) -> dict[str, Any]:
    keys = sorted(
        set(base.coverage.classification_counts)
        | set(target.coverage.classification_counts)
    )
    delta = {
        key: int(target.coverage.classification_counts.get(key, 0))
        - int(base.coverage.classification_counts.get(key, 0))
        for key in keys
    }
    unresolved_keys = ("unknown", "parse_error", "unsupported")
    return {
        "coverage_digest_changed": (
            base.coverage.coverage_digest != target.coverage.coverage_digest
        ),
        "review_digest_changed": (
            base.coverage.review_digest != target.coverage.review_digest
        ),
        "candidate_count_delta": (
            target.coverage.candidate_count - base.coverage.candidate_count
        ),
        "classification_delta": delta,
        "unresolved_structure_delta": sum(
            int(target.coverage.classification_counts.get(key, 0))
            - int(base.coverage.classification_counts.get(key, 0))
            for key in unresolved_keys
        ),
        "base": {
            "coverage_status": base.coverage.coverage_status,
            "review_status": base.coverage.review_status,
            "review_policy_satisfied": base.coverage.review_policy_satisfied,
        },
        "target": {
            "coverage_status": target.coverage.coverage_status,
            "review_status": target.coverage.review_status,
            "review_policy_satisfied": target.coverage.review_policy_satisfied,
        },
    }


def reconcile_project_snapshots(
    base: ProjectSnapshot,
    target: ProjectSnapshot,
    *,
    generated_at: str | None = None,
) -> ReconciliationReport:
    """Generate a deterministic, one-to-one, read-only reconciliation report."""

    validate_project_snapshot(base)
    validate_project_snapshot(target)
    if base.engine != target.engine:
        raise VersioningArtifactError("Cannot reconcile snapshots from different engines.")

    base_by_id = {item.occurrence_id: item for item in base.occurrences}
    target_by_id = {item.occurrence_id: item for item in target.occurrences}
    base_source_counts = Counter(
        _normalize_text(item.source_text) for item in base.occurrences
    )
    target_source_counts = Counter(
        _normalize_text(item.source_text) for item in target.occurrences
    )
    base_content_counts = Counter(item.content_fingerprint for item in base.occurrences)
    target_content_counts = Counter(item.content_fingerprint for item in target.occurrences)
    target_lineage_index: dict[str, list[str]] = defaultdict(list)
    target_locator_index: dict[str, list[str]] = defaultdict(list)
    target_content_index: dict[str, list[str]] = defaultdict(list)
    target_source_index: dict[str, list[str]] = defaultdict(list)
    target_ngram_index: dict[str, list[str]] = defaultdict(list)
    for target_item in target.occurrences:
        if target_item.lineage_id:
            target_lineage_index[target_item.lineage_id].append(
                target_item.occurrence_id
            )
        target_locator_index[stable_json_dumps(target_item.locator)].append(
            target_item.occurrence_id
        )
        target_content_index[target_item.content_fingerprint].append(
            target_item.occurrence_id
        )
        target_source_index[_normalize_text(target_item.source_text)].append(
            target_item.occurrence_id
        )
        for ngram in _source_ngrams(target_item.source_text):
            target_ngram_index[ngram].append(target_item.occurrence_id)

    evidence_by_pair: dict[tuple[str, str], _MatchEvidence] = {}
    related_group_keys_by_base: dict[str, list[tuple[str, str]]] = defaultdict(list)
    related_target_groups: dict[tuple[str, str], Sequence[str]] = {}
    for base_item in base.occurrences:
        candidate_target_ids: set[str] = set()
        if base_item.lineage_id:
            candidate_target_ids.update(
                target_lineage_index.get(base_item.lineage_id, ())
            )
        candidate_target_ids.update(
            target_locator_index.get(stable_json_dumps(base_item.locator), ())
        )
        content_group = target_content_index.get(base_item.content_fingerprint, ())
        content_group_key = ("content", base_item.content_fingerprint)
        related_group_keys_by_base[base_item.occurrence_id].append(content_group_key)
        related_target_groups[content_group_key] = content_group
        candidate_target_ids.update(
            _bounded_exact_group(base_item, content_group, target_by_id)
        )
        base_source = _normalize_text(base_item.source_text)
        source_group = target_source_index.get(base_source, ())
        source_group_key = ("source", base_source)
        related_group_keys_by_base[base_item.occurrence_id].append(source_group_key)
        related_target_groups[source_group_key] = source_group
        candidate_target_ids.update(
            _bounded_exact_group(base_item, source_group, target_by_id)
        )

        if not candidate_target_ids:
            fuzzy_counts: Counter[str] = Counter()
            informative_ngrams = sorted(
                (
                    (len(target_ngram_index.get(ngram, ())), ngram)
                    for ngram in _source_ngrams(base_item.source_text)
                    if 0
                    < len(target_ngram_index.get(ngram, ()))
                    <= MAX_NGRAM_POSTING
                ),
                key=lambda item: (item[0], item[1]),
            )[:MAX_NGRAM_KEYS_PER_SOURCE]
            for _posting_size, ngram in informative_ngrams:
                fuzzy_counts.update(target_ngram_index[ngram])
            base_length = len(base_source)
            fuzzy_ids = sorted(
                fuzzy_counts,
                key=lambda target_id: (-fuzzy_counts[target_id], target_id),
            )
            fuzzy_added = 0
            for target_id in fuzzy_ids:
                target_length = len(
                    _normalize_text(target_by_id[target_id].source_text)
                )
                if abs(base_length - target_length) > max(
                    8,
                    int(max(base_length, target_length) * 0.35),
                ):
                    continue
                candidate_target_ids.add(target_id)
                fuzzy_added += 1
                if fuzzy_added >= MAX_FUZZY_PAIR_CANDIDATES:
                    break

        for target_id in sorted(candidate_target_ids):
            target_item = target_by_id[target_id]
            evidence = _pair_evidence(base_item, target_item)
            if evidence is not None:
                target_source = _normalize_text(target_item.source_text)
                evidence = _MatchEvidence(
                    base_id=evidence.base_id,
                    target_id=evidence.target_id,
                    match_kind=evidence.match_kind,
                    rank=evidence.rank,
                    score=evidence.score,
                    confidence=evidence.confidence,
                    evidence={
                        **dict(evidence.evidence),
                        "base_source_occurrences": base_source_counts[base_source],
                        "target_source_occurrences": target_source_counts[target_source],
                        "base_content_occurrences": base_content_counts[
                            base_item.content_fingerprint
                        ],
                        "target_content_occurrences": target_content_counts[
                            target_item.content_fingerprint
                        ],
                    },
                )
                evidence_by_pair[(base_item.occurrence_id, target_item.occurrence_id)] = evidence

    evidence_by_rank: dict[int, list[_MatchEvidence]] = defaultdict(list)
    evidence_by_base: dict[str, list[_MatchEvidence]] = defaultdict(list)
    for evidence in evidence_by_pair.values():
        evidence_by_rank[evidence.rank].append(evidence)
        evidence_by_base[evidence.base_id].append(evidence)

    remaining_base = set(base_by_id)
    remaining_target = set(target_by_id)
    matched: list[_MatchEvidence] = []
    for rank in (600, 500, 450, 300, 200):
        pool = evidence_by_rank.get(rank, [])
        while True:
            candidates = [
                item
                for item in pool
                if item.base_id in remaining_base
                and item.target_id in remaining_target
            ]
            pool = candidates
            by_base: dict[str, list[_MatchEvidence]] = defaultdict(list)
            by_target: dict[str, list[_MatchEvidence]] = defaultdict(list)
            for item in candidates:
                by_base[item.base_id].append(item)
                by_target[item.target_id].append(item)
            base_choice = {
                key: _unique_best(values, rank=rank) for key, values in by_base.items()
            }
            target_choice = {
                key: _unique_best(values, rank=rank) for key, values in by_target.items()
            }
            mutual: list[_MatchEvidence] = []
            for choice in base_choice.values():
                if choice is None:
                    continue
                reverse = target_choice.get(choice.target_id)
                if reverse is not None and reverse.base_id == choice.base_id:
                    mutual.append(choice)
            if not mutual:
                break
            for choice in sorted(mutual, key=lambda item: (item.base_id, item.target_id)):
                if choice.base_id not in remaining_base or choice.target_id not in remaining_target:
                    continue
                matched.append(choice)
                remaining_base.remove(choice.base_id)
                remaining_target.remove(choice.target_id)

    items: list[ReconciliationItem] = []
    for match in matched:
        base_item = base_by_id[match.base_id]
        target_item = target_by_id[match.target_id]
        match_kind = match.match_kind
        source = _normalize_text(base_item.source_text)
        if (
            match_kind == "moved_exact"
            and (base_source_counts[source] > 1 or target_source_counts[source] > 1)
            and any(
                bool(match.evidence.get(key))
                for key in (
                    "speaker_equal",
                    "context_before_equal",
                    "context_after_equal",
                )
            )
        ):
            match_kind = "context_high_confidence"
        items.append(
            ReconciliationItem.create(
                disposition="matched",
                match_kind=match_kind,
                base_occurrence_id=match.base_id,
                target_occurrence_id=match.target_id,
                confidence=match.confidence,
                evidence=match.evidence,
            )
        )

    ambiguous_targets: set[str] = set()
    expanded_ambiguous_groups: set[str] = set()
    ambiguity_group_ids_by_target: dict[str, set[str]] = defaultdict(set)
    remaining_related_targets: dict[tuple[str, str], tuple[str, ...]] = {}
    remaining_related_target_sets: dict[tuple[str, str], frozenset[str]] = {}
    for base_id in sorted(remaining_base):
        candidates = [
            item
            for item in evidence_by_base.get(base_id, ())
            if item.target_id in remaining_target
        ]
        if not candidates:
            items.append(
                ReconciliationItem.create(
                    disposition="deleted",
                    match_kind="",
                    base_occurrence_id=base_id,
                )
            )
            continue
        ordered = sorted(
            candidates,
            key=lambda item: (-item.rank, -item.score, item.target_id),
        )
        top_rank = ordered[0].rank
        top_all = [item for item in ordered if item.rank == top_rank]
        top = top_all[:8]
        candidate_ids = [item.target_id for item in top]
        ambiguity_group_ids: set[str] = set()
        ambiguity_groups: list[dict[str, Any]] = []
        top_candidate_ids = tuple(sorted({item.target_id for item in top_all}))
        covered_top_candidate_ids: set[str] = set()

        related_groups = related_group_keys_by_base.get(base_id, ())
        for group_key in related_groups:
            if group_key not in remaining_related_targets:
                remaining_related_targets[group_key] = tuple(
                    sorted(
                        target_id
                        for target_id in related_target_groups.get(group_key, ())
                        if target_id in remaining_target
                    )
                )
                remaining_related_target_sets[group_key] = frozenset(
                    remaining_related_targets[group_key]
                )
            group_targets = remaining_related_targets[group_key]
            if not group_targets:
                continue
            covered_top_candidate_ids.update(
                target_id
                for target_id in top_candidate_ids
                if target_id in remaining_related_target_sets[group_key]
            )
            group_id = "ambiguity-group1:" + digest_json(
                {"kind": group_key[0], "value": group_key[1]}
            )
            ambiguity_group_ids.add(group_id)
            ambiguity_groups.append(
                {
                    "group_id": group_id,
                    "kind": group_key[0],
                    "target_count": len(group_targets),
                    "sample_target_occurrence_ids": list(group_targets[:8]),
                    "sample_truncated": len(group_targets) > 8,
                }
            )
            if len(candidate_ids) < 8:
                candidate_ids.extend(
                    target_id
                    for target_id in group_targets[:8]
                    if target_id not in candidate_ids
                )
                candidate_ids = candidate_ids[:8]
            if group_id not in expanded_ambiguous_groups:
                ambiguous_targets.update(group_targets)
                for target_id in group_targets:
                    ambiguity_group_ids_by_target[target_id].add(group_id)
                expanded_ambiguous_groups.add(group_id)

        unlinked_top_candidate_ids = tuple(
            target_id
            for target_id in top_candidate_ids
            if target_id not in covered_top_candidate_ids
        )
        if unlinked_top_candidate_ids:
            top_group_id = "ambiguity-group1:" + digest_json(
                {
                    "kind": "ranked_candidates",
                    "base_occurrence_id": base_id,
                    "rank": top_rank,
                    "target_occurrence_ids": list(unlinked_top_candidate_ids),
                }
            )
            ambiguity_group_ids.add(top_group_id)
            ambiguity_groups.append(
                {
                    "group_id": top_group_id,
                    "kind": "ranked_candidates",
                    "target_count": len(unlinked_top_candidate_ids),
                    "sample_target_occurrence_ids": list(
                        unlinked_top_candidate_ids[:8]
                    ),
                    "sample_truncated": len(unlinked_top_candidate_ids) > 8,
                }
            )
            ambiguous_targets.update(unlinked_top_candidate_ids)
            for target_id in unlinked_top_candidate_ids:
                ambiguity_group_ids_by_target[target_id].add(top_group_id)
            expanded_ambiguous_groups.add(top_group_id)

        candidate_ids = list(dict.fromkeys(candidate_ids))[:8]
        candidate_ids_truncated = any(
            bool(group["sample_truncated"]) for group in ambiguity_groups
        ) or any(
            target_id not in candidate_ids
            for group in ambiguity_groups
            for target_id in group["sample_target_occurrence_ids"]
        )
        items.append(
            ReconciliationItem.create(
                disposition="ambiguous",
                match_kind="ambiguous",
                base_occurrence_id=base_id,
                candidate_target_occurrence_ids=candidate_ids,
                confidence=top[0].confidence,
                evidence={
                    "candidates": [
                        {
                            "target_occurrence_id": item.target_id,
                            "match_kind": item.match_kind,
                            "score": round(item.score, 6),
                            "confidence": round(item.confidence, 6),
                            "evidence": dict(item.evidence),
                        }
                        for item in top
                    ],
                    "ambiguity_group_ids": sorted(ambiguity_group_ids),
                    "ambiguity_groups": ambiguity_groups,
                    "candidate_target_occurrence_ids_truncated": (
                        candidate_ids_truncated
                    ),
                },
            )
        )

    for target_id in sorted(ambiguous_targets):
        items.append(
            ReconciliationItem.create(
                disposition="ambiguous_target",
                match_kind="ambiguous_target",
                target_occurrence_id=target_id,
                evidence={
                    "ambiguity_group_ids": sorted(
                        ambiguity_group_ids_by_target[target_id]
                    )
                },
            )
        )

    for target_id in sorted(remaining_target - ambiguous_targets):
        items.append(
            ReconciliationItem.create(
                disposition="added",
                match_kind="",
                target_occurrence_id=target_id,
            )
        )

    disposition_order = {
        "matched": 0,
        "ambiguous": 1,
        "ambiguous_target": 2,
        "deleted": 3,
        "added": 4,
    }
    items.sort(
        key=lambda item: (
            disposition_order.get(item.disposition, 99),
            item.base_occurrence_id,
            item.target_occurrence_id,
            item.item_id,
        )
    )
    summary_counter: Counter[str] = Counter(item.disposition for item in items)
    summary_counter.update(
        item.match_kind for item in items if item.disposition == "matched"
    )
    summary = {
        "base_occurrence_count": len(base.occurrences),
        "target_occurrence_count": len(target.occurrences),
        "matched": summary_counter["matched"],
        "ambiguous": summary_counter["ambiguous"],
        "deleted": summary_counter["deleted"],
        "added": summary_counter["added"],
        "ambiguous_target_count": summary_counter["ambiguous_target"],
        **{
            kind: summary_counter[kind]
            for kind in (
                "confirmed_lineage",
                "locator_exact",
                "content_exact",
                "moved_exact",
                "context_high_confidence",
                "source_modified",
            )
        },
    }
    coverage_changes = _coverage_changes(base, target)
    needs_attention = bool(summary["ambiguous"])
    for snapshot in (base, target):
        if snapshot.coverage.coverage_status not in {"ready", "attention"}:
            needs_attention = True
        if not snapshot.coverage.review_policy_satisfied:
            needs_attention = True
    provisional = ReconciliationReport(
        base_snapshot_digest=base.snapshot_digest,
        target_snapshot_digest=target.snapshot_digest,
        base_version_id=base.game_version.version_id,
        target_version_id=target.game_version.version_id,
        base_coverage_dependency_digest=base.coverage.dependency_digest,
        target_coverage_dependency_digest=target.coverage.dependency_digest,
        status="attention" if needs_attention else "ready",
        summary=summary,
        coverage_changes=coverage_changes,
        items=tuple(items),
        generated_at=generated_at or _utc_now(),
        reconciliation_digest="",
    )
    report = replace(
        provisional,
        reconciliation_digest=digest_json(provisional.stable_payload()),
    )
    validate_reconciliation_report(report)
    return report


def validate_reconciliation_report(report: ReconciliationReport) -> None:
    """Validate report schemas, item identities, and the canonical digest."""

    if report.reconciliation_schema_version != RECONCILIATION_SCHEMA_VERSION:
        raise VersioningArtifactError("Unsupported reconciliation schema version.")
    if (
        report.reconciliation_digest_schema_version
        != RECONCILIATION_DIGEST_SCHEMA_VERSION
    ):
        raise VersioningArtifactError("Unsupported reconciliation digest schema version.")
    if report.status not in {"ready", "attention", "stale"}:
        raise VersioningArtifactError(
            f"Unsupported reconciliation status: {report.status}"
        )
    item_ids: set[str] = set()
    for item in report.items:
        if item.item_id in item_ids:
            raise VersioningArtifactError(f"Duplicate reconciliation item: {item.item_id}")
        item_ids.add(item.item_id)
        if digest_json(item.stable_payload()) != item.item_digest:
            raise VersioningArtifactError(
                f"Reconciliation item digest mismatch: {item.item_id}"
            )
    item_counts = Counter(item.disposition for item in report.items)
    match_counts = Counter(
        item.match_kind for item in report.items if item.disposition == "matched"
    )
    for key in ("matched", "ambiguous", "deleted", "added"):
        if int(report.summary.get(key, -1)) != item_counts[key]:
            raise VersioningArtifactError(
                f"Reconciliation summary does not match items: {key}"
            )
    if int(report.summary.get("ambiguous_target_count", -1)) != item_counts[
        "ambiguous_target"
    ]:
        raise VersioningArtifactError(
            "Reconciliation ambiguous target summary does not match items."
        )
    for key in (
        "confirmed_lineage",
        "locator_exact",
        "content_exact",
        "moved_exact",
        "context_high_confidence",
        "source_modified",
    ):
        if int(report.summary.get(key, -1)) != match_counts[key]:
            raise VersioningArtifactError(
                f"Reconciliation match summary does not match items: {key}"
            )
    expected_base_count = (
        item_counts["matched"]
        + item_counts["ambiguous"]
        + item_counts["deleted"]
    )
    if int(report.summary.get("base_occurrence_count", -1)) != expected_base_count:
        raise VersioningArtifactError(
            "Reconciliation base occurrence count does not match items."
        )
    expected_target_count = (
        item_counts["matched"]
        + item_counts["added"]
        + item_counts["ambiguous_target"]
    )
    if int(report.summary.get("target_occurrence_count", -1)) != expected_target_count:
        raise VersioningArtifactError(
            "Reconciliation target occurrence count does not match items."
        )
    target_ids = [
        item.target_occurrence_id
        for item in report.items
        if item.disposition in {"matched", "added", "ambiguous_target"}
    ]
    if len(target_ids) != len(set(target_ids)):
        raise VersioningArtifactError(
            "Reconciliation target occurrences must have one disposition each."
        )
    ambiguous_target_ids = {
        item.target_occurrence_id
        for item in report.items
        if item.disposition == "ambiguous_target"
    }
    referenced_candidate_ids = {
        target_id
        for item in report.items
        if item.disposition == "ambiguous"
        for target_id in item.candidate_target_occurrence_ids
    }
    if not referenced_candidate_ids <= ambiguous_target_ids:
        raise VersioningArtifactError(
            "Ambiguous candidates must have explicit ambiguous target items."
        )
    ambiguous_group_ids: set[str] = set()
    for item in report.items:
        if item.disposition != "ambiguous":
            continue
        group_ids = item.evidence.get("ambiguity_group_ids")
        if not isinstance(group_ids, list) or not all(
            isinstance(group_id, str) and group_id
            for group_id in group_ids
        ):
            raise VersioningArtifactError(
                "Ambiguous items require valid ambiguity group IDs."
            )
        ambiguous_group_ids.update(group_ids)
    for item in report.items:
        if item.disposition != "ambiguous_target":
            continue
        group_ids = item.evidence.get("ambiguity_group_ids")
        if not isinstance(group_ids, list) or not {
            group_id
            for group_id in group_ids
            if isinstance(group_id, str)
        } & ambiguous_group_ids:
            raise VersioningArtifactError(
                "Ambiguous target items must link to an ambiguous base group."
            )
    if digest_json(report.stable_payload()) != report.reconciliation_digest:
        raise VersioningArtifactError("Reconciliation digest does not match its payload.")


@dataclass(frozen=True)
class ReconciliationFreshness:
    effective_status: str
    stale_reasons: tuple[str, ...]


def validate_reconciliation_freshness(
    report: ReconciliationReport,
    base: ProjectSnapshot,
    target: ProjectSnapshot,
) -> ReconciliationFreshness:
    """Mark a saved report stale when either snapshot or coverage dependency changes."""

    validate_reconciliation_report(report)
    validate_project_snapshot(base)
    validate_project_snapshot(target)
    stale: list[str] = []
    comparisons = (
        ("base_snapshot_digest", report.base_snapshot_digest, base.snapshot_digest),
        ("target_snapshot_digest", report.target_snapshot_digest, target.snapshot_digest),
        (
            "base_coverage_dependency_digest",
            report.base_coverage_dependency_digest,
            base.coverage.dependency_digest,
        ),
        (
            "target_coverage_dependency_digest",
            report.target_coverage_dependency_digest,
            target.coverage.dependency_digest,
        ),
    )
    for name, recorded, current in comparisons:
        if recorded != current:
            stale.append(name)
    return ReconciliationFreshness(
        effective_status="stale" if stale else report.status,
        stale_reasons=tuple(stale),
    )


@dataclass(frozen=True)
class ReconciliationPackagePaths:
    package_dir: str
    report_path: str
    items_path: str


def export_reconciliation_report(
    report: ReconciliationReport,
    output_dir: str | os.PathLike[str],
) -> ReconciliationPackagePaths:
    """Atomically export the read-only reconciliation manifest and JSONL rows."""

    validate_reconciliation_report(report)
    package_dir = Path(output_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    items_path = package_dir / DEFAULT_RECONCILIATION_ITEMS_FILENAME
    report_path = package_dir / DEFAULT_RECONCILIATION_FILENAME
    atomic_write_lines(
        items_path,
        (stable_json_dumps(item.to_dict()) + "\n" for item in report.items),
        encoding="utf-8",
    )
    atomic_write_json(
        report_path,
        report.to_manifest(items_file=DEFAULT_RECONCILIATION_ITEMS_FILENAME),
        ensure_ascii=False,
        indent=2,
    )
    return ReconciliationPackagePaths(
        package_dir=str(package_dir.resolve()),
        report_path=str(report_path.resolve()),
        items_path=str(items_path.resolve()),
    )


def load_reconciliation_report(
    path: str | os.PathLike[str],
) -> ReconciliationReport:
    """Load and validate a reconciliation package directory or report path."""

    supplied = Path(path)
    report_path = supplied / DEFAULT_RECONCILIATION_FILENAME if supplied.is_dir() else supplied
    manifest = _read_json_object(report_path, artifact_name="reconciliation report")
    if str(manifest.get("kind") or "") != RECONCILIATION_KIND:
        raise VersioningArtifactError("Not a project snapshot reconciliation artifact.")
    schema = _int_value(
        manifest.get("reconciliation_schema_version"),
        field_name="reconciliation_schema_version",
        minimum=1,
    )
    if schema != RECONCILIATION_SCHEMA_VERSION:
        raise VersioningArtifactError("Unsupported reconciliation schema version.")
    digest_schema = _int_value(
        manifest.get("reconciliation_digest_schema_version"),
        field_name="reconciliation_digest_schema_version",
        minimum=1,
    )
    if digest_schema != RECONCILIATION_DIGEST_SCHEMA_VERSION:
        raise VersioningArtifactError("Unsupported reconciliation digest schema version.")
    paths = _mapping(manifest.get("paths"), field_name="paths")
    items_path = _artifact_path(
        report_path.parent,
        paths.get("items"),
        field_name="paths.items",
    )
    items = tuple(
        ReconciliationItem.from_dict(row)
        for row in _read_jsonl_objects(
            items_path,
            artifact_name="reconciliation items",
        )
    )
    if _int_value(manifest.get("item_count"), field_name="item_count") != len(items):
        raise VersioningArtifactError("Reconciliation item count does not match JSONL.")
    inputs = _mapping(manifest.get("inputs"), field_name="inputs")
    report = ReconciliationReport(
        base_snapshot_digest=_required_text(
            inputs.get("base_snapshot_digest"),
            field_name="inputs.base_snapshot_digest",
        ),
        target_snapshot_digest=_required_text(
            inputs.get("target_snapshot_digest"),
            field_name="inputs.target_snapshot_digest",
        ),
        base_version_id=_required_text(
            inputs.get("base_version_id"),
            field_name="inputs.base_version_id",
        ),
        target_version_id=_required_text(
            inputs.get("target_version_id"),
            field_name="inputs.target_version_id",
        ),
        base_coverage_dependency_digest=_required_text(
            inputs.get("base_coverage_dependency_digest"),
            field_name="inputs.base_coverage_dependency_digest",
        ),
        target_coverage_dependency_digest=_required_text(
            inputs.get("target_coverage_dependency_digest"),
            field_name="inputs.target_coverage_dependency_digest",
        ),
        status=_required_text(manifest.get("status"), field_name="status"),
        summary={
            str(key): _int_value(value, field_name=f"summary.{key}")
            for key, value in _mapping(
                manifest.get("summary"),
                field_name="summary",
            ).items()
        },
        coverage_changes=_mapping(
            manifest.get("coverage_changes"),
            field_name="coverage_changes",
        ),
        items=items,
        generated_at=str(manifest.get("generated_at") or ""),
        reconciliation_digest=_required_text(
            manifest.get("reconciliation_digest"),
            field_name="reconciliation_digest",
        ),
        reconciliation_schema_version=schema,
        reconciliation_digest_schema_version=digest_schema,
    )
    validate_reconciliation_report(report)
    return report
