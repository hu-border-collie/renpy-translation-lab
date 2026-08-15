"""Versioned translation records and human-confirmed reuse candidates (P4).

P4 builds on the read-only P3 reconciliation: base-version translations are
persisted as provenance-carrying records, reconciliation matches become
reviewable reuse candidates, and only accepted, fresh candidates can feed the
existing batch ``check -> apply`` path.  This module never writes game files,
never confirms lineage automatically, and never treats a source-modified match
as a completed translation.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from atomic_io import atomic_write_json, atomic_write_lines, atomic_write_text

from .coverage import digest_json, stable_json_dumps
from .versioning import (
    ProjectSnapshot,
    ReconciliationReport,
    UnitOccurrenceRecord,
    VersioningArtifactError,
    _artifact_path,
    _bool_value,
    _float_value,
    _int_value,
    _mapping,
    _normalize_path,
    _normalize_text,
    _read_json_object,
    _read_jsonl_objects,
    _required_text,
    _sequence,
    _utc_now,
    validate_reconciliation_freshness,
)


TRANSLATION_RECORD_SCHEMA_VERSION = 1
TRANSLATION_RECORD_SET_SCHEMA_VERSION = 1
REUSE_CANDIDATE_SCHEMA_VERSION = 1
REUSE_CANDIDATE_SET_SCHEMA_VERSION = 1
REUSE_DECISION_SCHEMA_VERSION = 1

TRANSLATION_RECORDS_KIND = "translation_records"
REUSE_CANDIDATES_KIND = "translation_reuse_candidates"

DEFAULT_RECORDS_FILENAME = "translation_records.jsonl"
DEFAULT_RECORDS_MANIFEST_FILENAME = "translation_records.json"
DEFAULT_CANDIDATES_FILENAME = "reuse_candidates.jsonl"
DEFAULT_REUSE_REPORT_FILENAME = "reuse_report.json"
DEFAULT_REUSE_REVIEW_FILENAME = "reuse_review.md"
DEFAULT_DECISIONS_TEMPLATE_FILENAME = "reuse_decisions_template.jsonl"

RECORD_ORIGINS = frozenset(
    {"model_initial", "human_confirmed", "revision_applied", "imported"}
)
RECORD_STATUSES = frozenset({"active", "superseded", "rejected"})
REVIEWER_TYPES = frozenset({"human", "agent"})
DECISION_ACTIONS = frozenset(
    {"accept", "reject", "override_translation", "split_lineage", "merge_lineage"}
)
REUSE_CLASSES = frozenset(
    {
        "exact_reuse",
        "moved_reuse",
        "context_match",
        "source_modified_reference",
        "ambiguous",
    }
)
REUSE_STATUSES = frozenset({"pending", "accepted", "rejected"})
DIRECT_REUSE_CLASSES = frozenset({"exact_reuse", "moved_reuse", "context_match"})
REFERENCE_ONLY_CLASSES = frozenset({"source_modified_reference"})
MATCH_KIND_REUSE_CLASSES = {
    "confirmed_lineage": "exact_reuse",
    "locator_exact": "exact_reuse",
    "content_exact": "exact_reuse",
    "moved_exact": "moved_reuse",
    "context_high_confidence": "context_match",
    "source_modified": "source_modified_reference",
}


def _ensure_dir(output_dir: str | os.PathLike[str]) -> Path:
    package_dir = Path(output_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    return package_dir


def _coerce_path(path: str | os.PathLike[str]) -> Path:
    return Path(path)


@dataclass(frozen=True)
class TranslationInput:
    """CLI-supplied translation evidence for one snapshot occurrence."""

    unit_id: str
    translation_text: str
    source_text: str = ""
    origin: str = "model_initial"
    chunk_key: str = ""
    row_key: str = ""
    source_manifest: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TranslationRecord:
    """Provenance-carrying translation for one base-version occurrence."""

    version_id: str
    snapshot_digest: str
    occurrence_id: str
    unit_id: str
    source_text: str
    translation_text: str
    target_language: str
    origin: str
    provenance: Mapping[str, Any]
    status: str
    revision_history: tuple[Mapping[str, Any], ...]
    record_id: str
    record_digest: str
    translation_record_schema_version: int = TRANSLATION_RECORD_SCHEMA_VERSION

    def stable_payload(self) -> dict[str, Any]:
        return {
            "translation_record_schema_version": (
                self.translation_record_schema_version
            ),
            "version_id": self.version_id,
            "snapshot_digest": self.snapshot_digest,
            "occurrence_id": self.occurrence_id,
            "unit_id": self.unit_id,
            "source_text": self.source_text,
            "translation_text": self.translation_text,
            "target_language": self.target_language,
            "origin": self.origin,
            "provenance": dict(self.provenance),
            "status": self.status,
            "revision_history": [dict(item) for item in self.revision_history],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.stable_payload(),
            "record_id": self.record_id,
            "record_digest": self.record_digest,
        }

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (
            _normalize_path(self.provenance.get("file_rel_path") or ""),
            int(self.provenance.get("line_number") or 0),
            self.occurrence_id,
        )

    @classmethod
    def create(
        cls,
        *,
        version_id: str,
        snapshot_digest: str,
        occurrence_id: str,
        unit_id: str,
        source_text: str,
        translation_text: str,
        target_language: str,
        origin: str,
        provenance: Mapping[str, Any] | None = None,
        status: str = "active",
        revision_history: Sequence[Mapping[str, Any]] = (),
        record_id: str = "",
    ) -> TranslationRecord:
        normalized_version = _required_text(version_id, field_name="record.version_id")
        normalized_snapshot = _required_text(
            snapshot_digest,
            field_name="record.snapshot_digest",
        )
        normalized_occurrence = _required_text(
            occurrence_id,
            field_name="record.occurrence_id",
        )
        normalized_unit = _required_text(unit_id, field_name="record.unit_id")
        normalized_translation = _required_text(
            translation_text,
            field_name="record.translation_text",
        )
        normalized_origin = _required_text(origin, field_name="record.origin")
        if normalized_origin not in RECORD_ORIGINS:
            raise VersioningArtifactError(
                f"Unsupported translation record origin: {normalized_origin}"
            )
        normalized_status = _required_text(status, field_name="record.status")
        if normalized_status not in RECORD_STATUSES:
            raise VersioningArtifactError(
                f"Unsupported translation record status: {normalized_status}"
            )
        normalized_history = tuple(
            _mapping(item, field_name="record.revision_history item")
            for item in revision_history
        )
        identity = {
            "version_id": normalized_version,
            "occurrence_id": normalized_occurrence,
            "unit_id": normalized_unit,
        }
        normalized_record_id = record_id or ("transrec1:" + digest_json(identity))
        if not normalized_record_id.startswith("transrec1:"):
            raise VersioningArtifactError("Invalid translation record id prefix.")
        provisional = cls(
            version_id=normalized_version,
            snapshot_digest=normalized_snapshot,
            occurrence_id=normalized_occurrence,
            unit_id=normalized_unit,
            source_text=str(source_text or ""),
            translation_text=normalized_translation,
            target_language=str(target_language or ""),
            origin=normalized_origin,
            provenance=_mapping(provenance or {}, field_name="record.provenance"),
            status=normalized_status,
            revision_history=normalized_history,
            record_id=normalized_record_id,
            record_digest="",
        )
        return replace(
            provisional,
            record_digest=digest_json(provisional.stable_payload()),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TranslationRecord:
        payload = _mapping(value, field_name="translation_record")
        schema = _int_value(
            payload.get("translation_record_schema_version"),
            field_name="translation_record_schema_version",
            minimum=1,
        )
        if schema != TRANSLATION_RECORD_SCHEMA_VERSION:
            raise VersioningArtifactError(
                "Unsupported translation record schema version."
            )
        record = cls.create(
            version_id=payload.get("version_id") or "",
            snapshot_digest=payload.get("snapshot_digest") or "",
            occurrence_id=payload.get("occurrence_id") or "",
            unit_id=payload.get("unit_id") or "",
            source_text=payload.get("source_text") or "",
            translation_text=payload.get("translation_text") or "",
            target_language=payload.get("target_language") or "",
            origin=payload.get("origin") or "",
            provenance=_mapping(
                payload.get("provenance") or {},
                field_name="record.provenance",
            ),
            status=payload.get("status") or "active",
            revision_history=_sequence(
                payload.get("revision_history") or [],
                field_name="revision_history",
            ),
            record_id=str(payload.get("record_id") or ""),
        )
        if str(payload.get("record_digest") or "") != record.record_digest:
            raise VersioningArtifactError(
                f"Translation record digest does not match: {record.record_id}"
            )
        return record


@dataclass(frozen=True)
class TranslationRecordSet:
    """Deterministic, portable set of translations for one game version."""

    version_id: str
    snapshot_digest: str
    target_language: str
    records: tuple[TranslationRecord, ...]
    generated_at: str
    record_set_digest: str
    translation_record_set_schema_version: int = TRANSLATION_RECORD_SET_SCHEMA_VERSION

    def stable_payload(self) -> dict[str, Any]:
        return {
            "translation_record_set_schema_version": (
                self.translation_record_set_schema_version
            ),
            "version_id": self.version_id,
            "snapshot_digest": self.snapshot_digest,
            "target_language": self.target_language,
            "record_count": len(self.records),
            "record_digests": [record.record_digest for record in self.records],
        }

    def to_manifest(
        self,
        *,
        records_file: str = DEFAULT_RECORDS_FILENAME,
    ) -> dict[str, Any]:
        return {
            "kind": TRANSLATION_RECORDS_KIND,
            "translation_record_set_schema_version": (
                self.translation_record_set_schema_version
            ),
            **self.stable_payload(),
            "origin_counts": dict(Counter(record.origin for record in self.records)),
            "generated_at": self.generated_at,
            "paths": {"records": _normalize_path(records_file)},
            "record_set_digest": self.record_set_digest,
        }


def build_translation_records(
    snapshot: ProjectSnapshot,
    inputs: Sequence[TranslationInput],
    *,
    generated_at: str | None = None,
) -> TranslationRecordSet:
    """Freeze validated translations as records linked to snapshot occurrences."""

    from .versioning import validate_project_snapshot

    validate_project_snapshot(snapshot)
    occurrence_by_unit: dict[str, UnitOccurrenceRecord] = {}
    for occurrence in snapshot.occurrences:
        if occurrence.unit_id in occurrence_by_unit:
            raise VersioningArtifactError(
                "Duplicate unit id in snapshot: " + occurrence.unit_id
            )
        occurrence_by_unit[occurrence.unit_id] = occurrence

    seen_units: set[str] = set()
    records: list[TranslationRecord] = []
    for item in inputs:
        unit_id = _required_text(item.unit_id, field_name="input.unit_id")
        if unit_id in seen_units:
            raise VersioningArtifactError(
                f"Duplicate translation input for unit: {unit_id}"
            )
        seen_units.add(unit_id)
        occurrence = occurrence_by_unit.get(unit_id)
        if occurrence is None:
            raise VersioningArtifactError(
                f"Translation input unit is not in the snapshot: {unit_id}"
            )
        if str(item.source_text or "").strip() and _normalize_text(
            item.source_text
        ) != _normalize_text(occurrence.source_text):
            raise VersioningArtifactError(
                "Translation input source does not match the snapshot "
                f"occurrence: {unit_id}"
            )
        origin = _required_text(item.origin, field_name="input.origin")
        if origin not in RECORD_ORIGINS:
            raise VersioningArtifactError(f"Unsupported input origin: {origin}")
        record = TranslationRecord.create(
            version_id=snapshot.game_version.version_id,
            snapshot_digest=snapshot.snapshot_digest,
            occurrence_id=occurrence.occurrence_id,
            unit_id=unit_id,
            source_text=occurrence.source_text,
            translation_text=item.translation_text,
            target_language=snapshot.target_language,
            origin=origin,
            provenance={
                "origin": origin,
                "chunk_key": str(item.chunk_key or ""),
                "row_key": str(item.row_key or ""),
                "source_manifest": str(item.source_manifest or ""),
                "extra": dict(item.extra or {}),
                "file_rel_path": occurrence.file_rel_path,
                "line_number": occurrence.line_number,
            },
        )
        records.append(record)
    records.sort(key=lambda record: record.sort_key)
    provisional = TranslationRecordSet(
        version_id=snapshot.game_version.version_id,
        snapshot_digest=snapshot.snapshot_digest,
        target_language=snapshot.target_language,
        records=tuple(records),
        generated_at=generated_at or _utc_now(),
        record_set_digest="",
    )
    return replace(
        provisional,
        record_set_digest=digest_json(provisional.stable_payload()),
    )


@dataclass(frozen=True)
class TranslationRecordsPackagePaths:
    package_dir: str
    manifest_path: str
    records_path: str


def export_translation_records(
    record_set: TranslationRecordSet,
    output_dir: str | os.PathLike[str],
) -> TranslationRecordsPackagePaths:
    """Atomically export the record manifest and JSONL rows."""

    package_dir = _ensure_dir(output_dir)
    records_path = package_dir / DEFAULT_RECORDS_FILENAME
    manifest_path = package_dir / DEFAULT_RECORDS_MANIFEST_FILENAME
    atomic_write_lines(
        records_path,
        (stable_json_dumps(record.to_dict()) + "\n" for record in record_set.records),
        encoding="utf-8",
    )
    atomic_write_json(
        manifest_path,
        record_set.to_manifest(records_file=DEFAULT_RECORDS_FILENAME),
        ensure_ascii=False,
        indent=2,
    )
    return TranslationRecordsPackagePaths(
        package_dir=str(package_dir.resolve()),
        manifest_path=str(manifest_path.resolve()),
        records_path=str(records_path.resolve()),
    )


def load_translation_records(path: str | os.PathLike[str]) -> TranslationRecordSet:
    """Load and validate a translation-records package directory or manifest."""

    supplied = _coerce_path(path)
    manifest_path = (
        supplied / DEFAULT_RECORDS_MANIFEST_FILENAME if supplied.is_dir() else supplied
    )
    manifest = _read_json_object(
        manifest_path,
        artifact_name="translation records manifest",
    )
    if str(manifest.get("kind") or "") != TRANSLATION_RECORDS_KIND:
        raise VersioningArtifactError("Not a translation records artifact.")
    schema = _int_value(
        manifest.get("translation_record_set_schema_version"),
        field_name="translation_record_set_schema_version",
        minimum=1,
    )
    if schema != TRANSLATION_RECORD_SET_SCHEMA_VERSION:
        raise VersioningArtifactError(
            "Unsupported translation record set schema version."
        )
    paths = _mapping(manifest.get("paths"), field_name="paths")
    records_path = _artifact_path(
        manifest_path.parent,
        paths.get("records"),
        field_name="paths.records",
    )
    records = tuple(
        TranslationRecord.from_dict(row)
        for row in _read_jsonl_objects(
            records_path,
            artifact_name="translation records",
        )
    )
    manifest_digests = tuple(
        _required_text(value, field_name=f"record_digests[{index}]")
        for index, value in enumerate(
            _sequence(manifest.get("record_digests"), field_name="record_digests")
        )
    )
    if manifest_digests != tuple(record.record_digest for record in records):
        raise VersioningArtifactError("Translation record digests do not match JSONL.")
    if _int_value(
        manifest.get("record_count"),
        field_name="record_count",
    ) != len(records):
        raise VersioningArtifactError(
            "Translation record count does not match JSONL."
        )
    provisional = TranslationRecordSet(
        version_id=_required_text(manifest.get("version_id"), field_name="version_id"),
        snapshot_digest=_required_text(
            manifest.get("snapshot_digest"),
            field_name="snapshot_digest",
        ),
        target_language=str(manifest.get("target_language") or ""),
        records=records,
        generated_at=str(manifest.get("generated_at") or ""),
        record_set_digest="",
        translation_record_set_schema_version=schema,
    )
    record_set = replace(
        provisional,
        record_set_digest=digest_json(provisional.stable_payload()),
    )
    if str(manifest.get("record_set_digest") or "") != record_set.record_set_digest:
        raise VersioningArtifactError(
            "Translation record set digest does not match its payload."
        )
    return record_set


@dataclass(frozen=True)
class ReuseCandidate:
    """One reviewable translation-reuse proposal derived from reconciliation."""

    reuse_class: str
    status: str
    reconciliation_item_id: str
    reconciliation_digest: str
    base_version_id: str
    target_version_id: str
    base_snapshot_digest: str
    target_snapshot_digest: str
    base_records_digest: str
    base_occurrence_id: str
    target_occurrence_id: str
    candidate_target_occurrence_ids: tuple[str, ...]
    base_record_id: str
    base_record_digest: str
    reference_translation: str
    effective_translation: str
    reference_origin: str
    has_translation_record: bool
    confidence: float
    evidence: Mapping[str, Any]
    decision: Mapping[str, Any]
    audit: tuple[Mapping[str, Any], ...]
    candidate_id: str
    candidate_digest: str
    reuse_candidate_schema_version: int = REUSE_CANDIDATE_SCHEMA_VERSION

    @property
    def reference_only(self) -> bool:
        return self.reuse_class in REFERENCE_ONLY_CLASSES

    def stable_payload(self) -> dict[str, Any]:
        return {
            "reuse_candidate_schema_version": self.reuse_candidate_schema_version,
            "reuse_class": self.reuse_class,
            "status": self.status,
            "reconciliation_item_id": self.reconciliation_item_id,
            "reconciliation_digest": self.reconciliation_digest,
            "base_version_id": self.base_version_id,
            "target_version_id": self.target_version_id,
            "base_snapshot_digest": self.base_snapshot_digest,
            "target_snapshot_digest": self.target_snapshot_digest,
            "base_records_digest": self.base_records_digest,
            "base_occurrence_id": self.base_occurrence_id,
            "target_occurrence_id": self.target_occurrence_id,
            "candidate_target_occurrence_ids": list(
                self.candidate_target_occurrence_ids
            ),
            "base_record_id": self.base_record_id,
            "base_record_digest": self.base_record_digest,
            "reference_translation": self.reference_translation,
            "effective_translation": self.effective_translation,
            "reference_origin": self.reference_origin,
            "has_translation_record": self.has_translation_record,
            "reference_only": self.reference_only,
            "confidence": round(
                _float_value(self.confidence, field_name="confidence"),
                6,
            ),
            "evidence": dict(self.evidence),
            "decision": dict(self.decision),
            "audit": [dict(item) for item in self.audit],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.stable_payload(),
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        reuse_class: str,
        status: str = "pending",
        reconciliation_item_id: str,
        reconciliation_digest: str,
        base_version_id: str,
        target_version_id: str,
        base_snapshot_digest: str,
        target_snapshot_digest: str,
        base_records_digest: str,
        base_occurrence_id: str,
        target_occurrence_id: str = "",
        candidate_target_occurrence_ids: Sequence[str] = (),
        base_record_id: str = "",
        base_record_digest: str = "",
        reference_translation: str = "",
        effective_translation: str = "",
        reference_origin: str = "",
        has_translation_record: bool = False,
        confidence: float = 0.0,
        evidence: Mapping[str, Any] | None = None,
        decision: Mapping[str, Any] | None = None,
        audit: Sequence[Mapping[str, Any]] = (),
        candidate_id: str = "",
    ) -> ReuseCandidate:
        normalized_class = _required_text(reuse_class, field_name="reuse_class")
        if normalized_class not in REUSE_CLASSES:
            raise VersioningArtifactError(
                f"Unsupported reuse class: {normalized_class}"
            )
        normalized_status = _required_text(status, field_name="status")
        if normalized_status not in REUSE_STATUSES:
            raise VersioningArtifactError(
                f"Unsupported reuse candidate status: {normalized_status}"
            )
        normalized_base = _required_text(
            base_occurrence_id,
            field_name="base_occurrence_id",
        )
        normalized_candidates = tuple(sorted(set(candidate_target_occurrence_ids)))
        normalized_has_record = _bool_value(
            has_translation_record,
            field_name="has_translation_record",
        )
        if normalized_class == "ambiguous":
            if not normalized_candidates:
                raise VersioningArtifactError(
                    "Ambiguous reuse candidates require candidate targets."
                )
            if target_occurrence_id:
                raise VersioningArtifactError(
                    "Ambiguous reuse candidates cannot carry a confirmed target."
                )
        else:
            target = _required_text(
                target_occurrence_id,
                field_name="target_occurrence_id",
            )
            if normalized_candidates:
                raise VersioningArtifactError(
                    "Non-ambiguous reuse candidates cannot carry candidate targets."
                )
            if normalized_class == "exact_reuse" and target == normalized_base:
                raise VersioningArtifactError(
                    "Reuse candidate target must differ from its base occurrence."
                )
        if normalized_has_record:
            _required_text(base_record_id, field_name="base_record_id")
            _required_text(base_record_digest, field_name="base_record_digest")
            _required_text(
                reference_translation,
                field_name="reference_translation",
            )
        else:
            if base_record_id or base_record_digest or reference_translation:
                raise VersioningArtifactError(
                    "Reuse candidates without a record cannot reference one."
                )
        normalized_effective = str(effective_translation or reference_translation)
        normalized_confidence = round(
            _float_value(confidence, field_name="confidence"),
            6,
        )
        if not 0.0 <= normalized_confidence <= 1.0:
            raise VersioningArtifactError("Reuse confidence must be between 0 and 1.")
        normalized_audit = tuple(
            _mapping(item, field_name="audit item") for item in audit
        )
        if normalized_effective != reference_translation and not any(
            item.get("action") == "override_translation" for item in normalized_audit
        ):
            raise VersioningArtifactError(
                "An overridden translation requires an override audit entry."
            )
        identity = {
            "reconciliation_item_id": _required_text(
                reconciliation_item_id,
                field_name="reconciliation_item_id",
            ),
            "reuse_class": normalized_class,
            "base_occurrence_id": normalized_base,
            "target_occurrence_id": str(target_occurrence_id or ""),
            "candidate_target_occurrence_ids": list(normalized_candidates),
        }
        normalized_candidate_id = candidate_id or (
            "reusecand1:" + digest_json(identity)
        )
        if not normalized_candidate_id.startswith("reusecand1:"):
            raise VersioningArtifactError("Invalid reuse candidate id prefix.")
        provisional = cls(
            reuse_class=normalized_class,
            status=normalized_status,
            reconciliation_item_id=identity["reconciliation_item_id"],
            reconciliation_digest=_required_text(
                reconciliation_digest,
                field_name="reconciliation_digest",
            ),
            base_version_id=_required_text(
                base_version_id,
                field_name="base_version_id",
            ),
            target_version_id=_required_text(
                target_version_id,
                field_name="target_version_id",
            ),
            base_snapshot_digest=_required_text(
                base_snapshot_digest,
                field_name="base_snapshot_digest",
            ),
            target_snapshot_digest=_required_text(
                target_snapshot_digest,
                field_name="target_snapshot_digest",
            ),
            base_records_digest=_required_text(
                base_records_digest,
                field_name="base_records_digest",
            ),
            base_occurrence_id=normalized_base,
            target_occurrence_id=str(target_occurrence_id or ""),
            candidate_target_occurrence_ids=normalized_candidates,
            base_record_id=str(base_record_id or ""),
            base_record_digest=str(base_record_digest or ""),
            reference_translation=str(reference_translation or ""),
            effective_translation=normalized_effective,
            reference_origin=str(reference_origin or ""),
            has_translation_record=normalized_has_record,
            confidence=normalized_confidence,
            evidence=_mapping(evidence or {}, field_name="evidence"),
            decision=_mapping(decision or {}, field_name="decision"),
            audit=normalized_audit,
            candidate_id=normalized_candidate_id,
            candidate_digest="",
        )
        return replace(
            provisional,
            candidate_digest=digest_json(provisional.stable_payload()),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReuseCandidate:
        payload = _mapping(value, field_name="reuse_candidate")
        schema = _int_value(
            payload.get("reuse_candidate_schema_version"),
            field_name="reuse_candidate_schema_version",
            minimum=1,
        )
        if schema != REUSE_CANDIDATE_SCHEMA_VERSION:
            raise VersioningArtifactError("Unsupported reuse candidate schema version.")
        candidate = cls.create(
            reuse_class=payload.get("reuse_class") or "",
            status=payload.get("status") or "pending",
            reconciliation_item_id=payload.get("reconciliation_item_id") or "",
            reconciliation_digest=payload.get("reconciliation_digest") or "",
            base_version_id=payload.get("base_version_id") or "",
            target_version_id=payload.get("target_version_id") or "",
            base_snapshot_digest=payload.get("base_snapshot_digest") or "",
            target_snapshot_digest=payload.get("target_snapshot_digest") or "",
            base_records_digest=payload.get("base_records_digest") or "",
            base_occurrence_id=payload.get("base_occurrence_id") or "",
            target_occurrence_id=payload.get("target_occurrence_id") or "",
            candidate_target_occurrence_ids=_sequence(
                payload.get("candidate_target_occurrence_ids") or [],
                field_name="candidate_target_occurrence_ids",
            ),
            base_record_id=payload.get("base_record_id") or "",
            base_record_digest=payload.get("base_record_digest") or "",
            reference_translation=payload.get("reference_translation") or "",
            effective_translation=payload.get("effective_translation") or "",
            reference_origin=payload.get("reference_origin") or "",
            has_translation_record=_bool_value(
                payload.get("has_translation_record"),
                field_name="has_translation_record",
            ),
            confidence=_float_value(
                payload.get("confidence"),
                field_name="confidence",
            ),
            evidence=_mapping(payload.get("evidence") or {}, field_name="evidence"),
            decision=_mapping(payload.get("decision") or {}, field_name="decision"),
            audit=_sequence(payload.get("audit") or [], field_name="audit"),
            candidate_id=str(payload.get("candidate_id") or ""),
        )
        if str(payload.get("candidate_id") or "") != candidate.candidate_id:
            raise VersioningArtifactError("Reuse candidate identity does not match.")
        if str(payload.get("candidate_digest") or "") != candidate.candidate_digest:
            raise VersioningArtifactError("Reuse candidate digest does not match.")
        return candidate


def _reuse_summary(
    candidates: Sequence[ReuseCandidate],
    reconciliation_summary: Mapping[str, int],
) -> dict[str, int]:
    summary: Counter[str] = Counter()
    for candidate in candidates:
        summary[f"class_{candidate.reuse_class}"] += 1
        summary[f"status_{candidate.status}"] += 1
        if candidate.has_translation_record:
            summary["with_translation_record"] += 1
        else:
            summary["without_translation_record"] += 1
        if candidate.status == "accepted" and candidate.reference_only:
            summary["accepted_reference_only"] += 1
        if candidate.status == "accepted" and not candidate.reference_only:
            summary["accepted_direct_reuse"] += 1
    for key in (
        "added",
        "deleted",
        "ambiguous_target",
        "orphaned_records",
    ):
        value = reconciliation_summary.get(
            f"reconciliation_{key}",
            reconciliation_summary.get(key, 0),
        )
        summary[f"reconciliation_{key}"] = int(value)
    return dict(sorted(summary.items()))


@dataclass(frozen=True)
class ReuseCandidateSet:
    """Deterministic package of reuse candidates plus lineage decisions."""

    base_version_id: str
    target_version_id: str
    base_snapshot_digest: str
    target_snapshot_digest: str
    reconciliation_digest: str
    base_records_digest: str
    status: str
    stale_reasons: tuple[str, ...]
    summary: Mapping[str, int]
    candidates: tuple[ReuseCandidate, ...]
    lineage_decisions: tuple[Mapping[str, Any], ...]
    generated_at: str
    candidate_set_digest: str
    reuse_candidate_set_schema_version: int = REUSE_CANDIDATE_SET_SCHEMA_VERSION

    def stable_payload(self) -> dict[str, Any]:
        return {
            "reuse_candidate_set_schema_version": (
                self.reuse_candidate_set_schema_version
            ),
            "inputs": {
                "base_version_id": self.base_version_id,
                "target_version_id": self.target_version_id,
                "base_snapshot_digest": self.base_snapshot_digest,
                "target_snapshot_digest": self.target_snapshot_digest,
                "reconciliation_digest": self.reconciliation_digest,
                "base_records_digest": self.base_records_digest,
            },
            "status": self.status,
            "stale_reasons": list(self.stale_reasons),
            "summary": dict(sorted(self.summary.items())),
            "lineage_decisions": [dict(item) for item in self.lineage_decisions],
            "candidate_digests": [
                candidate.candidate_digest for candidate in self.candidates
            ],
        }

    def to_manifest(
        self,
        *,
        candidates_file: str = DEFAULT_CANDIDATES_FILENAME,
    ) -> dict[str, Any]:
        return {
            "kind": REUSE_CANDIDATES_KIND,
            "reuse_candidate_set_schema_version": (
                self.reuse_candidate_set_schema_version
            ),
            **self.stable_payload(),
            "generated_at": self.generated_at,
            "candidate_count": len(self.candidates),
            "paths": {
                "candidates": _normalize_path(candidates_file),
            },
            "candidate_set_digest": self.candidate_set_digest,
        }


def _validate_reuse_target_uniqueness(
    candidates: Sequence[ReuseCandidate],
) -> None:
    seen_targets: dict[str, str] = {}
    for candidate in candidates:
        if candidate.status != "accepted" or not candidate.target_occurrence_id:
            continue
        target = candidate.target_occurrence_id
        previous = seen_targets.get(target)
        if previous is not None and previous != candidate.candidate_id:
            raise VersioningArtifactError(
                "Two accepted reuse candidates share one target occurrence: "
                f"{target}"
            )
        seen_targets[target] = candidate.candidate_id


def _assemble_candidate_set(
    *,
    inputs: Mapping[str, Any],
    candidates: Sequence[ReuseCandidate],
    lineage_decisions: Sequence[Mapping[str, Any]],
    reconciliation_summary: Mapping[str, int],
    generated_at: str,
) -> ReuseCandidateSet:
    _validate_reuse_target_uniqueness(candidates)
    provisional = ReuseCandidateSet(
        base_version_id=inputs["base_version_id"],
        target_version_id=inputs["target_version_id"],
        base_snapshot_digest=inputs["base_snapshot_digest"],
        target_snapshot_digest=inputs["target_snapshot_digest"],
        reconciliation_digest=inputs["reconciliation_digest"],
        base_records_digest=inputs["base_records_digest"],
        status=inputs.get("status", "fresh"),
        stale_reasons=tuple(inputs.get("stale_reasons") or ()),
        summary=_reuse_summary(candidates, reconciliation_summary),
        candidates=tuple(candidates),
        lineage_decisions=tuple(lineage_decisions),
        generated_at=generated_at,
        candidate_set_digest="",
    )
    assembled = replace(
        provisional,
        candidate_set_digest=digest_json(provisional.stable_payload()),
    )
    if inputs.get("candidate_set_digest") and (
        inputs["candidate_set_digest"] != assembled.candidate_set_digest
    ):
        raise VersioningArtifactError(
            "Reuse candidate set digest does not match its payload."
        )
    return assembled


def build_reuse_candidates(
    reconciliation: ReconciliationReport,
    base_snapshot: ProjectSnapshot,
    target_snapshot: ProjectSnapshot,
    base_records: TranslationRecordSet,
    *,
    generated_at: str | None = None,
) -> ReuseCandidateSet:
    """Derive reviewable reuse candidates from a fresh reconciliation report."""

    from .versioning import (
        validate_project_snapshot,
        validate_reconciliation_report,
    )

    validate_reconciliation_report(reconciliation)
    validate_project_snapshot(base_snapshot)
    validate_project_snapshot(target_snapshot)
    freshness = validate_reconciliation_freshness(
        reconciliation,
        base_snapshot,
        target_snapshot,
    )
    if freshness.stale_reasons:
        raise VersioningArtifactError(
            "Cannot build reuse candidates from a stale reconciliation: "
            + ", ".join(freshness.stale_reasons)
        )
    if base_records.version_id != base_snapshot.game_version.version_id:
        raise VersioningArtifactError(
            "Translation records do not belong to the base snapshot version."
        )
    if base_records.snapshot_digest != base_snapshot.snapshot_digest:
        raise VersioningArtifactError(
            "Translation records do not match the base snapshot digest."
        )

    records_by_occurrence = {
        record.occurrence_id: record for record in base_records.records
    }
    base_by_id = {
        occurrence.occurrence_id: occurrence
        for occurrence in base_snapshot.occurrences
    }
    target_by_id = {
        occurrence.occurrence_id: occurrence
        for occurrence in target_snapshot.occurrences
    }
    candidates: list[ReuseCandidate] = []
    for item in reconciliation.items:
        if item.disposition == "matched":
            source_equal = bool(item.evidence.get("source_equal"))
            if source_equal:
                reuse_class = MATCH_KIND_REUSE_CLASSES.get(item.match_kind)
                if reuse_class is None:
                    raise VersioningArtifactError(
                        "Unsupported matched reconciliation kind for reuse: "
                        + item.match_kind
                    )
            else:
                # A locator or lineage may survive while the text changed;
                # such matches keep the old translation as reference only.
                reuse_class = "source_modified_reference"
            base_occurrence = base_by_id.get(item.base_occurrence_id)
            target_occurrence = target_by_id.get(item.target_occurrence_id)
            if base_occurrence is None or target_occurrence is None:
                raise VersioningArtifactError(
                    "Reconciliation match references an occurrence missing from "
                    "its snapshot: " + item.item_id
                )
            record = records_by_occurrence.get(item.base_occurrence_id)
            candidates.append(
                ReuseCandidate.create(
                    reuse_class=reuse_class,
                    reconciliation_item_id=item.item_id,
                    reconciliation_digest=reconciliation.reconciliation_digest,
                    base_version_id=base_snapshot.game_version.version_id,
                    target_version_id=target_snapshot.game_version.version_id,
                    base_snapshot_digest=base_snapshot.snapshot_digest,
                    target_snapshot_digest=target_snapshot.snapshot_digest,
                    base_records_digest=base_records.record_set_digest,
                    base_occurrence_id=item.base_occurrence_id,
                    target_occurrence_id=item.target_occurrence_id,
                    base_record_id=record.record_id if record else "",
                    base_record_digest=record.record_digest if record else "",
                    reference_translation=(
                        record.translation_text if record else ""
                    ),
                    reference_origin=record.origin if record else "",
                    has_translation_record=record is not None,
                    confidence=item.confidence,
                    evidence={
                        "match_kind": item.match_kind,
                        "base_source": base_occurrence.source_text,
                        "target_source": target_occurrence.source_text,
                        "reconciliation_evidence": dict(item.evidence),
                    },
                )
            )
        elif item.disposition == "ambiguous":
            base_occurrence = base_by_id.get(item.base_occurrence_id)
            if base_occurrence is None:
                raise VersioningArtifactError(
                    "Ambiguous reconciliation references a base occurrence "
                    "missing from the base snapshot: " + item.item_id
                )
            record = records_by_occurrence.get(item.base_occurrence_id)
            candidates.append(
                ReuseCandidate.create(
                    reuse_class="ambiguous",
                    reconciliation_item_id=item.item_id,
                    reconciliation_digest=reconciliation.reconciliation_digest,
                    base_version_id=base_snapshot.game_version.version_id,
                    target_version_id=target_snapshot.game_version.version_id,
                    base_snapshot_digest=base_snapshot.snapshot_digest,
                    target_snapshot_digest=target_snapshot.snapshot_digest,
                    base_records_digest=base_records.record_set_digest,
                    base_occurrence_id=item.base_occurrence_id,
                    candidate_target_occurrence_ids=(
                        item.candidate_target_occurrence_ids
                    ),
                    base_record_id=record.record_id if record else "",
                    base_record_digest=record.record_digest if record else "",
                    reference_translation=(
                        record.translation_text if record else ""
                    ),
                    reference_origin=record.origin if record else "",
                    has_translation_record=record is not None,
                    confidence=item.confidence,
                    evidence={
                        "match_kind": item.match_kind,
                        "base_source": base_occurrence.source_text,
                        "target_sources": [
                            target_by_id[target_id].source_text
                            for target_id in item.candidate_target_occurrence_ids
                            if target_id in target_by_id
                        ],
                        "reconciliation_evidence": dict(item.evidence),
                    },
                )
            )
    orphaned_records = sum(
        1
        for item in reconciliation.items
        if item.disposition == "deleted"
        and item.base_occurrence_id in records_by_occurrence
    )
    inputs = {
        "base_version_id": base_snapshot.game_version.version_id,
        "target_version_id": target_snapshot.game_version.version_id,
        "base_snapshot_digest": base_snapshot.snapshot_digest,
        "target_snapshot_digest": target_snapshot.snapshot_digest,
        "reconciliation_digest": reconciliation.reconciliation_digest,
        "base_records_digest": base_records.record_set_digest,
        "status": "fresh",
        "stale_reasons": (),
    }
    candidate_set = _assemble_candidate_set(
        inputs=inputs,
        candidates=candidates,
        lineage_decisions=(),
        reconciliation_summary={
            **dict(reconciliation.summary),
            "orphaned_records": orphaned_records,
        },
        generated_at=generated_at or _utc_now(),
    )
    return candidate_set


@dataclass(frozen=True)
class ReuseDecision:
    """One human/agent decision submitted for a reuse candidate."""

    candidate_id: str
    action: str
    reviewer_type: str
    reviewer_name: str
    note: str = ""
    target_occurrence_id: str = ""
    translation_text: str = ""
    related_occurrence_ids: tuple[str, ...] = ()
    lineage_id: str = ""
    decided_at: str = ""
    reuse_decision_schema_version: int = REUSE_DECISION_SCHEMA_VERSION

    def stable_payload(self) -> dict[str, Any]:
        return {
            "reuse_decision_schema_version": self.reuse_decision_schema_version,
            "candidate_id": self.candidate_id,
            "action": self.action,
            "reviewer": {
                "type": self.reviewer_type,
                "name": self.reviewer_name,
            },
            "note": self.note,
            "target_occurrence_id": self.target_occurrence_id,
            "translation_text": self.translation_text,
            "related_occurrence_ids": list(self.related_occurrence_ids),
            "lineage_id": self.lineage_id,
            "decided_at": self.decided_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.stable_payload()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReuseDecision:
        payload = _mapping(value, field_name="reuse_decision")
        schema = _int_value(
            payload.get("reuse_decision_schema_version"),
            field_name="reuse_decision_schema_version",
            minimum=1,
        )
        if schema != REUSE_DECISION_SCHEMA_VERSION:
            raise VersioningArtifactError("Unsupported reuse decision schema version.")
        reviewer = _mapping(payload.get("reviewer") or {}, field_name="reviewer")
        decision = cls(
            candidate_id=_required_text(
                payload.get("candidate_id"),
                field_name="candidate_id",
            ),
            action=_required_text(payload.get("action"), field_name="action"),
            reviewer_type=_required_text(
                reviewer.get("type"),
                field_name="reviewer.type",
            ),
            reviewer_name=_required_text(
                reviewer.get("name"),
                field_name="reviewer.name",
            ),
            note=str(payload.get("note") or ""),
            target_occurrence_id=str(payload.get("target_occurrence_id") or ""),
            translation_text=str(payload.get("translation_text") or ""),
            related_occurrence_ids=tuple(
                str(item)
                for item in _sequence(
                    payload.get("related_occurrence_ids") or [],
                    field_name="related_occurrence_ids",
                )
            ),
            lineage_id=str(payload.get("lineage_id") or ""),
            decided_at=str(payload.get("decided_at") or ""),
            reuse_decision_schema_version=schema,
        )
        if decision.action not in DECISION_ACTIONS:
            raise VersioningArtifactError(
                f"Unsupported reuse decision action: {decision.action}"
            )
        if decision.reviewer_type not in REVIEWER_TYPES:
            raise VersioningArtifactError(
                f"Unsupported reviewer type: {decision.reviewer_type}"
            )
        return decision


def load_reuse_decisions(
    path: str | os.PathLike[str],
) -> tuple[ReuseDecision, ...]:
    """Load and validate a reuse-decisions JSONL file."""

    return tuple(
        ReuseDecision.from_dict(row)
        for row in _read_jsonl_objects(
            _coerce_path(path),
            artifact_name="reuse decisions",
        )
    )


@dataclass(frozen=True)
class ReuseFreshness:
    effective_status: str
    stale_reasons: tuple[str, ...]


def validate_reuse_freshness(
    candidate_set: ReuseCandidateSet,
    reconciliation: ReconciliationReport,
    base_snapshot: ProjectSnapshot,
    target_snapshot: ProjectSnapshot,
    base_records: TranslationRecordSet,
) -> ReuseFreshness:
    """Mark saved reuse candidates stale when any input digest drifts."""

    stale: list[str] = []
    comparisons = (
        (
            "reconciliation_digest",
            candidate_set.reconciliation_digest,
            reconciliation.reconciliation_digest,
        ),
        (
            "base_snapshot_digest",
            candidate_set.base_snapshot_digest,
            base_snapshot.snapshot_digest,
        ),
        (
            "target_snapshot_digest",
            candidate_set.target_snapshot_digest,
            target_snapshot.snapshot_digest,
        ),
        (
            "base_records_digest",
            candidate_set.base_records_digest,
            base_records.record_set_digest,
        ),
    )
    for name, recorded, current in comparisons:
        if recorded != current:
            stale.append(name)
    records_by_id = {
        record.record_id: record for record in base_records.records
    }
    for candidate in candidate_set.candidates:
        if not candidate.has_translation_record:
            continue
        record = records_by_id.get(candidate.base_record_id)
        if record is None or record.record_digest != candidate.base_record_digest:
            stale.append(
                f"base_record_digest:{candidate.base_record_id}"
            )
    return ReuseFreshness(
        effective_status="stale" if stale else "fresh",
        stale_reasons=tuple(sorted(set(stale))),
    )


def _decision_audit_entry(
    decision: ReuseDecision,
    *,
    previous_status: str,
) -> dict[str, Any]:
    return {
        "at": decision.decided_at or _utc_now(),
        "reviewer_type": decision.reviewer_type,
        "reviewer_name": decision.reviewer_name,
        "action": decision.action,
        "note": decision.note,
        "previous_status": previous_status,
    }


def apply_reuse_decisions(
    candidate_set: ReuseCandidateSet,
    decisions: Sequence[ReuseDecision],
    *,
    reconciliation: ReconciliationReport,
    base_snapshot: ProjectSnapshot,
    target_snapshot: ProjectSnapshot,
    base_records: TranslationRecordSet,
    generated_at: str | None = None,
) -> ReuseCandidateSet:
    """Apply reviewer decisions with provenance and return a new package."""

    freshness = validate_reuse_freshness(
        candidate_set,
        reconciliation,
        base_snapshot,
        target_snapshot,
        base_records,
    )
    if freshness.stale_reasons:
        raise VersioningArtifactError(
            "Cannot apply decisions to stale reuse candidates: "
            + ", ".join(freshness.stale_reasons)
        )
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in candidate_set.candidates
    }
    updated: dict[str, ReuseCandidate] = dict(candidates_by_id)
    lineage_decisions: list[dict[str, Any]] = [
        dict(item) for item in candidate_set.lineage_decisions
    ]
    for decision in decisions:
        candidate = updated.get(decision.candidate_id)
        if candidate is None:
            raise VersioningArtifactError(
                "Reuse decision references an unknown candidate: "
                + decision.candidate_id
            )
        audit_entry = _decision_audit_entry(
            decision,
            previous_status=candidate.status,
        )
        if decision.action == "accept":
            if candidate.status != "pending":
                raise VersioningArtifactError(
                    "Only pending reuse candidates can be accepted: "
                    + candidate.candidate_id
                )
            if not candidate.has_translation_record:
                raise VersioningArtifactError(
                    "Cannot accept a reuse candidate without a translation "
                    "record: " + candidate.candidate_id
                )
            resolved_target = candidate.target_occurrence_id
            if candidate.reuse_class == "ambiguous":
                if decision.target_occurrence_id not in (
                    candidate.candidate_target_occurrence_ids
                ):
                    raise VersioningArtifactError(
                        "Ambiguous acceptance must resolve one candidate target: "
                        + candidate.candidate_id
                    )
                resolved_target = decision.target_occurrence_id
            elif decision.target_occurrence_id and (
                decision.target_occurrence_id != candidate.target_occurrence_id
            ):
                raise VersioningArtifactError(
                    "Acceptance target does not match the candidate target: "
                    + candidate.candidate_id
                )
            decision_record = {
                "reviewer_type": decision.reviewer_type,
                "reviewer_name": decision.reviewer_name,
                "action": "accept",
                "note": decision.note,
                "decided_at": audit_entry["at"],
                "resolved_target_occurrence_id": resolved_target,
            }
            updated[decision.candidate_id] = replace(
                candidate,
                status="accepted",
                decision=decision_record,
                audit=tuple([*candidate.audit, audit_entry]),
                candidate_digest="",
            )
        elif decision.action == "reject":
            if candidate.status != "pending":
                raise VersioningArtifactError(
                    "Only pending reuse candidates can be rejected: "
                    + candidate.candidate_id
                )
            decision_record = {
                "reviewer_type": decision.reviewer_type,
                "reviewer_name": decision.reviewer_name,
                "action": "reject",
                "note": decision.note,
                "decided_at": audit_entry["at"],
            }
            updated[decision.candidate_id] = replace(
                candidate,
                status="rejected",
                decision=decision_record,
                audit=tuple([*candidate.audit, audit_entry]),
                candidate_digest="",
            )
        elif decision.action == "override_translation":
            if candidate.status not in ("pending", "accepted"):
                raise VersioningArtifactError(
                    "Overrides require a pending or accepted candidate: "
                    + candidate.candidate_id
                )
            if not decision.translation_text.strip():
                raise VersioningArtifactError(
                    "Translation overrides require a non-empty translation: "
                    + candidate.candidate_id
                )
            if not candidate.has_translation_record:
                raise VersioningArtifactError(
                    "Cannot override a candidate without a translation record: "
                    + candidate.candidate_id
                )
            updated[decision.candidate_id] = replace(
                candidate,
                effective_translation=decision.translation_text,
                audit=tuple([*candidate.audit, audit_entry]),
                candidate_digest="",
            )
        elif decision.action in ("split_lineage", "merge_lineage"):
            if not decision.lineage_id.strip():
                raise VersioningArtifactError(
                    "Lineage decisions require a lineage id: "
                    + candidate.candidate_id
                )
            related = tuple(sorted(set(decision.related_occurrence_ids)))
            if decision.action == "merge_lineage" and not related:
                raise VersioningArtifactError(
                    "Lineage merges require related occurrences: "
                    + candidate.candidate_id
                )
            if decision.action == "split_lineage" and related:
                raise VersioningArtifactError(
                    "Lineage splits cannot carry related occurrences: "
                    + candidate.candidate_id
                )
            lineage_entry = {
                "decision_id": "lineage1:"
                + digest_json(
                    {
                        "action": decision.action,
                        "base_occurrence_id": candidate.base_occurrence_id,
                        "lineage_id": decision.lineage_id,
                        "related_occurrence_ids": list(related),
                        "candidate_id": candidate.candidate_id,
                    }
                ),
                "action": decision.action,
                "candidate_id": candidate.candidate_id,
                "base_occurrence_id": candidate.base_occurrence_id,
                "lineage_id": decision.lineage_id,
                "related_occurrence_ids": list(related),
                "reviewer_type": decision.reviewer_type,
                "reviewer_name": decision.reviewer_name,
                "note": decision.note,
                "decided_at": audit_entry["at"],
            }
            lineage_decisions.append(lineage_entry)
            updated[decision.candidate_id] = replace(
                candidate,
                audit=tuple([*candidate.audit, audit_entry]),
                candidate_digest="",
            )
        else:
            raise VersioningArtifactError(
                f"Unsupported reuse decision action: {decision.action}"
            )
    finalized: list[ReuseCandidate] = []
    for candidate in candidate_set.candidates:
        updated_candidate = updated[candidate.candidate_id]
        if updated_candidate.candidate_digest:
            finalized.append(updated_candidate)
            continue
        finalized.append(
            replace(
                updated_candidate,
                candidate_digest=digest_json(
                    updated_candidate.stable_payload()
                ),
            )
        )
    inputs = {
        "base_version_id": candidate_set.base_version_id,
        "target_version_id": candidate_set.target_version_id,
        "base_snapshot_digest": candidate_set.base_snapshot_digest,
        "target_snapshot_digest": candidate_set.target_snapshot_digest,
        "reconciliation_digest": candidate_set.reconciliation_digest,
        "base_records_digest": candidate_set.base_records_digest,
        "status": "fresh",
        "stale_reasons": (),
    }
    return _assemble_candidate_set(
        inputs=inputs,
        candidates=finalized,
        lineage_decisions=lineage_decisions,
        reconciliation_summary=candidate_set.summary,
        generated_at=generated_at or _utc_now(),
    )


@dataclass(frozen=True)
class ReusePrefillEntry:
    """A gated, accepted reuse translation keyed for one target unit."""

    candidate_id: str
    candidate_digest: str
    target_occurrence_id: str
    target_unit_id: str
    source_text: str
    translation_text: str
    reference_origin: str
    reuse_class: str
    provenance: Mapping[str, Any]

    def stable_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "target_occurrence_id": self.target_occurrence_id,
            "target_unit_id": self.target_unit_id,
            "source_text": self.source_text,
            "translation_text": self.translation_text,
            "reference_origin": self.reference_origin,
            "reuse_class": self.reuse_class,
            "provenance": dict(self.provenance),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.stable_payload()


def collect_reuse_prefill(
    candidate_set: ReuseCandidateSet,
    *,
    reconciliation: ReconciliationReport,
    base_snapshot: ProjectSnapshot,
    target_snapshot: ProjectSnapshot,
    base_records: TranslationRecordSet,
) -> tuple[ReusePrefillEntry, ...]:
    """Return prefill entries for accepted, fresh, non-reference candidates.

    This is the write-back gate for reuse: every returned entry carries a
    human/agent decision, matches the live input digests, and reuses a
    translation whose normalized source still equals the target occurrence.
    Reference-only matches and unconfirmed candidates never pass this gate.
    """

    freshness = validate_reuse_freshness(
        candidate_set,
        reconciliation,
        base_snapshot,
        target_snapshot,
        base_records,
    )
    if freshness.stale_reasons:
        raise VersioningArtifactError(
            "Cannot export reuse prefill from stale candidates: "
            + ", ".join(freshness.stale_reasons)
        )
    records_by_occurrence = {
        record.occurrence_id: record for record in base_records.records
    }
    target_by_id = {
        occurrence.occurrence_id: occurrence
        for occurrence in target_snapshot.occurrences
    }
    entries: list[ReusePrefillEntry] = []
    seen_targets: set[str] = set()
    for candidate in candidate_set.candidates:
        if candidate.status != "accepted":
            continue
        if candidate.reference_only:
            continue
        target_id = candidate.target_occurrence_id or str(
            candidate.decision.get("resolved_target_occurrence_id") or ""
        )
        if not target_id:
            raise VersioningArtifactError(
                "Accepted reuse candidate has no resolved target: "
                + candidate.candidate_id
            )
        if target_id in seen_targets:
            raise VersioningArtifactError(
                "Duplicate reuse prefill target: " + target_id
            )
        seen_targets.add(target_id)
        target_occurrence = target_by_id.get(target_id)
        if target_occurrence is None:
            raise VersioningArtifactError(
                "Reuse target occurrence is not in the target snapshot: "
                + target_id
            )
        if not candidate.has_translation_record:
            raise VersioningArtifactError(
                "Accepted reuse candidate has no translation record: "
                + candidate.candidate_id
            )
        base_record = records_by_occurrence.get(candidate.base_occurrence_id)
        if base_record is None or (
            base_record.record_digest != candidate.base_record_digest
        ):
            raise VersioningArtifactError(
                "Reuse candidate record no longer matches live records: "
                + candidate.candidate_id
            )
        if _normalize_text(base_record.source_text) != _normalize_text(
            target_occurrence.source_text
        ):
            raise VersioningArtifactError(
                "Reuse sources no longer match between versions: "
                + candidate.candidate_id
            )
        if not candidate.effective_translation.strip():
            raise VersioningArtifactError(
                "Accepted reuse candidate has an empty translation: "
                + candidate.candidate_id
            )
        entries.append(
            ReusePrefillEntry(
                candidate_id=candidate.candidate_id,
                candidate_digest=candidate.candidate_digest,
                target_occurrence_id=target_id,
                target_unit_id=target_occurrence.unit_id,
                source_text=target_occurrence.source_text,
                translation_text=candidate.effective_translation,
                reference_origin=candidate.reference_origin,
                reuse_class=candidate.reuse_class,
                provenance={
                    "reconciliation_digest": candidate.reconciliation_digest,
                    "base_record_id": candidate.base_record_id,
                    "base_record_digest": candidate.base_record_digest,
                    "base_version_id": candidate.base_version_id,
                    "target_version_id": candidate.target_version_id,
                    "reviewer_type": candidate.decision.get("reviewer_type", ""),
                    "reviewer_name": candidate.decision.get("reviewer_name", ""),
                    "decided_at": candidate.decision.get("decided_at", ""),
                },
            )
        )
    return tuple(entries)


@dataclass(frozen=True)
class ReusePackagePaths:
    package_dir: str
    report_path: str
    candidates_path: str
    review_path: str
    decisions_template_path: str


def _review_excerpt(value: str, limit: int = 80) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def build_reuse_review_markdown(
    candidate_set: ReuseCandidateSet,
    *,
    target_snapshot: ProjectSnapshot | None = None,
) -> str:
    """Render the human/agent review sheet for a reuse candidate package."""

    target_by_id = (
        {
            occurrence.occurrence_id: occurrence
            for occurrence in target_snapshot.occurrences
        }
        if target_snapshot is not None
        else {}
    )
    lines: list[str] = [
        "# Translation Reuse Review",
        "",
        f"- Base version: `{candidate_set.base_version_id}`",
        f"- Target version: `{candidate_set.target_version_id}`",
        f"- Reconciliation digest: `{candidate_set.reconciliation_digest}`",
        f"- Base records digest: `{candidate_set.base_records_digest}`",
        f"- Status: `{candidate_set.status}`",
        "",
        "| Summary | Count |",
        "| --- | --- |",
    ]
    for key in sorted(candidate_set.summary):
        lines.append(f"| {key} | {candidate_set.summary[key]} |")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Candidate | Class | Status | Base source | Reference | Target | Decision |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for candidate in candidate_set.candidates:
        targets = [
            candidate.target_occurrence_id,
            *candidate.candidate_target_occurrence_ids,
        ]
        target_labels = []
        for target_id in targets:
            if not target_id:
                continue
            occurrence = target_by_id.get(target_id)
            if occurrence is None:
                target_labels.append(target_id)
            else:
                target_labels.append(
                    f"{occurrence.file_rel_path}:{occurrence.line_number}"
                )
        decision = candidate.decision.get("action") or ""
        reviewer = candidate.decision.get("reviewer_name") or ""
        decision_label = f"{decision}{' by ' + reviewer if reviewer else ''}"
        lines.append(
            "| "
            + " | ".join(
                (
                    candidate.candidate_id[:19],
                    candidate.reuse_class,
                    candidate.status,
                    _review_excerpt(candidate.evidence.get("base_source", "")),
                    _review_excerpt(candidate.reference_translation),
                    ", ".join(target_labels) or "-",
                    decision_label or "-",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision file format",
            "",
            "One JSON object per line; fill `reviewer.type` with `human` or",
            "`agent` and always identify the reviewer:",
            "",
            "- `accept`: confirm the reuse (ambiguous candidates must also set",
            "  `target_occurrence_id` to one of the candidate targets).",
            "- `reject`: refuse the match; the target keeps needing translation.",
            "- `override_translation`: replace the reused translation text.",
            "- `split_lineage` / `merge_lineage`: record a lineage decision with",
            "  `lineage_id` (merge also lists `related_occurrence_ids`).",
            "",
            "Source-modified matches are reference-only: accepting them keeps the",
            "old translation visible for reviewers but never marks the target as",
            "translated.",
            "",
        ]
    )
    return "\n".join(lines)


def build_decisions_template(
    candidate_set: ReuseCandidateSet,
) -> list[dict[str, Any]]:
    """Return fill-in decision rows for every pending candidate."""

    return [
        {
            "reuse_decision_schema_version": REUSE_DECISION_SCHEMA_VERSION,
            "candidate_id": candidate.candidate_id,
            "action": "accept",
            "reviewer": {"type": "", "name": ""},
            "note": "",
            "target_occurrence_id": "",
            "translation_text": "",
            "related_occurrence_ids": [],
            "lineage_id": "",
            "decided_at": "",
        }
        for candidate in candidate_set.candidates
        if candidate.status == "pending"
    ]


def export_reuse_candidates(
    candidate_set: ReuseCandidateSet,
    output_dir: str | os.PathLike[str],
    *,
    target_snapshot: ProjectSnapshot | None = None,
    input_paths: Mapping[str, str] | None = None,
) -> ReusePackagePaths:
    """Atomically export candidates, report, review sheet, and template."""

    package_dir = _ensure_dir(output_dir)
    candidates_path = package_dir / DEFAULT_CANDIDATES_FILENAME
    report_path = package_dir / DEFAULT_REUSE_REPORT_FILENAME
    review_path = package_dir / DEFAULT_REUSE_REVIEW_FILENAME
    template_path = package_dir / DEFAULT_DECISIONS_TEMPLATE_FILENAME
    atomic_write_lines(
        candidates_path,
        (
            stable_json_dumps(candidate.to_dict()) + "\n"
            for candidate in candidate_set.candidates
        ),
        encoding="utf-8",
    )
    atomic_write_json(
        report_path,
        {
            **candidate_set.to_manifest(
                candidates_file=DEFAULT_CANDIDATES_FILENAME
            ),
            "input_paths": {
                key: str(value)
                for key, value in sorted((input_paths or {}).items())
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    atomic_write_text(
        review_path,
        build_reuse_review_markdown(candidate_set, target_snapshot=target_snapshot),
        encoding="utf-8",
    )
    atomic_write_text(
        template_path,
        "\n".join(
            stable_json_dumps(row) for row in build_decisions_template(candidate_set)
        )
        + ("\n" if candidate_set.candidates else ""),
        encoding="utf-8",
    )
    return ReusePackagePaths(
        package_dir=str(package_dir.resolve()),
        report_path=str(report_path.resolve()),
        candidates_path=str(candidates_path.resolve()),
        review_path=str(review_path.resolve()),
        decisions_template_path=str(template_path.resolve()),
    )


def load_reuse_candidates(path: str | os.PathLike[str]) -> ReuseCandidateSet:
    """Load and validate a reuse-candidates package directory or report."""

    supplied = _coerce_path(path)
    report_path = (
        supplied / DEFAULT_REUSE_REPORT_FILENAME if supplied.is_dir() else supplied
    )
    manifest = _read_json_object(
        report_path,
        artifact_name="reuse candidates report",
    )
    if str(manifest.get("kind") or "") != REUSE_CANDIDATES_KIND:
        raise VersioningArtifactError("Not a translation reuse candidates artifact.")
    schema = _int_value(
        manifest.get("reuse_candidate_set_schema_version"),
        field_name="reuse_candidate_set_schema_version",
        minimum=1,
    )
    if schema != REUSE_CANDIDATE_SET_SCHEMA_VERSION:
        raise VersioningArtifactError(
            "Unsupported reuse candidate set schema version."
        )
    paths = _mapping(manifest.get("paths"), field_name="paths")
    candidates_path = _artifact_path(
        report_path.parent,
        paths.get("candidates"),
        field_name="paths.candidates",
    )
    candidates = tuple(
        ReuseCandidate.from_dict(row)
        for row in _read_jsonl_objects(
            candidates_path,
            artifact_name="reuse candidates",
        )
    )
    manifest_digests = tuple(
        _required_text(value, field_name=f"candidate_digests[{index}]")
        for index, value in enumerate(
            _sequence(
                manifest.get("candidate_digests"),
                field_name="candidate_digests",
            )
        )
    )
    if manifest_digests != tuple(
        candidate.candidate_digest for candidate in candidates
    ):
        raise VersioningArtifactError(
            "Reuse candidate digests do not match JSONL."
        )
    if _int_value(
        manifest.get("candidate_count"),
        field_name="candidate_count",
    ) != len(candidates):
        raise VersioningArtifactError(
            "Reuse candidate count does not match JSONL."
        )
    inputs = _mapping(manifest.get("inputs"), field_name="inputs")
    provisional = ReuseCandidateSet(
        base_version_id=_required_text(
            inputs.get("base_version_id"),
            field_name="inputs.base_version_id",
        ),
        target_version_id=_required_text(
            inputs.get("target_version_id"),
            field_name="inputs.target_version_id",
        ),
        base_snapshot_digest=_required_text(
            inputs.get("base_snapshot_digest"),
            field_name="inputs.base_snapshot_digest",
        ),
        target_snapshot_digest=_required_text(
            inputs.get("target_snapshot_digest"),
            field_name="inputs.target_snapshot_digest",
        ),
        reconciliation_digest=_required_text(
            inputs.get("reconciliation_digest"),
            field_name="inputs.reconciliation_digest",
        ),
        base_records_digest=_required_text(
            inputs.get("base_records_digest"),
            field_name="inputs.base_records_digest",
        ),
        status=_required_text(manifest.get("status"), field_name="status"),
        stale_reasons=tuple(
            str(item)
            for item in _sequence(
                manifest.get("stale_reasons") or [],
                field_name="stale_reasons",
            )
        ),
        summary={
            str(key): _int_value(value, field_name=f"summary.{key}")
            for key, value in _mapping(
                manifest.get("summary"),
                field_name="summary",
            ).items()
        },
        candidates=candidates,
        lineage_decisions=tuple(
            _mapping(item, field_name="lineage_decisions item")
            for item in _sequence(
                manifest.get("lineage_decisions") or [],
                field_name="lineage_decisions",
            )
        ),
        generated_at=str(manifest.get("generated_at") or ""),
        candidate_set_digest="",
        reuse_candidate_set_schema_version=schema,
    )
    candidate_set = replace(
        provisional,
        candidate_set_digest=digest_json(provisional.stable_payload()),
    )
    if str(manifest.get("candidate_set_digest") or "") != (
        candidate_set.candidate_set_digest
    ):
        raise VersioningArtifactError(
            "Reuse candidate set digest does not match its payload."
        )
    return candidate_set
