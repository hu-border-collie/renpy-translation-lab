"""Versioned, engine-neutral contracts for localization adapters.

P1 introduced read-only discovery, inventory, audit, and extraction.  P2 adds
relocation, engine validation, and declarative writeback plans while keeping
all project/manifest checks and actual file writes in common workflow code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import translation_core


ENGINE_ADAPTER_PROTOCOL_VERSION = 1
OCCURRENCE_SCHEMA_VERSION = 1
CONTENT_FINGERPRINT_SCHEMA_VERSION = 1
VALIDATION_SCHEMA_VERSION = 1
WRITEBACK_PLAN_SCHEMA_VERSION = 1
CANDIDATE_SCHEMA_VERSION = 1
COVERAGE_SCHEMA_VERSION = 1
COVERAGE_REVIEW_SCHEMA_VERSION = 1
COVERAGE_DIGEST_SCHEMA_VERSION = 1


class LocalizationMode(str, Enum):
    SOURCE_EXTRACTION = "source_extraction"
    NATIVE_CATALOG = "native_catalog"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class EngineCapabilities:
    engine: str
    adapter_version: str
    supported_localization_modes: tuple[LocalizationMode, ...]
    selected_localization_mode: LocalizationMode
    source_inventory: bool = True
    native_catalog: bool = False
    relocation: bool = False
    declarative_writeback: tuple[str, ...] = ()
    native_catalog_required_for_writeback: bool = False
    engine_adapter_protocol_version: int = ENGINE_ADAPTER_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_adapter_protocol_version": self.engine_adapter_protocol_version,
            "engine": self.engine,
            "adapter_version": self.adapter_version,
            "supported_localization_modes": [
                mode.value for mode in self.supported_localization_modes
            ],
            "selected_localization_mode": self.selected_localization_mode.value,
            "source_inventory": self.source_inventory,
            "native_catalog": self.native_catalog,
            "relocation": self.relocation,
            "declarative_writeback": list(self.declarative_writeback),
            "native_catalog_required_for_writeback": (self.native_catalog_required_for_writeback),
        }


@dataclass(frozen=True)
class ProjectDiscoveryRequest:
    project_root: str
    localization_root: str
    target_language: str = ""
    include_files: tuple[str, ...] = ()
    include_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceDocument:
    file_rel_path: str
    file_path: str
    size: int
    sha256: str
    content: bytes = field(repr=False, compare=False)

    def text(self) -> str:
        return self.content.decode("utf-8-sig")

    def lines(self) -> list[str]:
        return self.text().splitlines(keepends=True)

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "file_rel_path": self.file_rel_path,
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ProjectDiscovery:
    engine: str
    adapter_version: str
    project_root: str
    localization_root: str
    target_language: str
    project_snapshot_fingerprint: str
    source_fingerprint: str
    source_documents: tuple[SourceDocument, ...]
    localization_mode: LocalizationMode
    catalog_provenance: Mapping[str, Any] = field(default_factory=dict)
    coverage_digest: str = ""
    coverage_review_digest: str = ""

    def document_by_path(self) -> dict[str, SourceDocument]:
        return {document.file_rel_path: document for document in self.source_documents}


@dataclass(frozen=True)
class InventoryPolicy:
    review_policy: str = "agent_or_human"


@dataclass(frozen=True)
class OpaqueLocator:
    engine: str
    locator_schema_version: int
    locator: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "locator_schema_version": self.locator_schema_version,
            "locator": dict(self.locator),
        }


@dataclass
class Candidate:
    candidate_id: str
    engine: str
    adapter_version: str
    source_fingerprint: str
    locator: OpaqueLocator
    raw_excerpt: str
    structure_kind: str
    classification: str
    reason_codes: tuple[str, ...]
    translation_scope: str
    analysis_scope: str
    catalog_link: Mapping[str, Any] | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    candidate_schema_version: int = CANDIDATE_SCHEMA_VERSION
    unit: translation_core.TranslationUnit | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    legacy_item: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_schema_version": self.candidate_schema_version,
            "candidate_id": self.candidate_id,
            "engine": self.engine,
            "adapter_version": self.adapter_version,
            "source_fingerprint": self.source_fingerprint,
            "locator": self.locator.to_dict(),
            "raw_excerpt": self.raw_excerpt,
            "structure_kind": self.structure_kind,
            "classification": self.classification,
            "reason_codes": list(self.reason_codes),
            "translation_scope": self.translation_scope,
            "analysis_scope": self.analysis_scope,
            "catalog_link": (dict(self.catalog_link) if self.catalog_link is not None else None),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CandidateInventory:
    engine: str
    adapter_version: str
    source_fingerprint: str
    project_snapshot_fingerprint: str
    candidates: tuple[Candidate, ...]
    files_scanned: tuple[Mapping[str, Any], ...]

    def by_id(self) -> dict[str, Candidate]:
        return {candidate.candidate_id: candidate for candidate in self.candidates}


@dataclass(frozen=True)
class Occurrence:
    occurrence_id: str
    engine: str
    project_snapshot_fingerprint: str
    content_fingerprint: str
    candidate_id: str
    locator: OpaqueLocator
    unit: translation_core.TranslationUnit
    occurrence_schema_version: int = OCCURRENCE_SCHEMA_VERSION
    content_fingerprint_schema_version: int = CONTENT_FINGERPRINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        unit = self.unit
        translation_unit = {
            "core_schema_version": translation_core.CORE_SCHEMA_VERSION,
            "id": unit.id,
            "mode": unit.mode,
            "text": unit.text,
            "source": unit.source,
            "current_translation": unit.current_translation,
            "file_rel_path": unit.file_rel_path,
            "line": unit.line,
            "line_number": unit.display_line_number,
            "start": unit.start,
            "end": unit.end,
            "prefix": unit.prefix,
            "quote": unit.quote,
            "speaker_id": unit.speaker_id,
            "speaker_name": unit.speaker_name,
        }
        return {
            "occurrence_schema_version": self.occurrence_schema_version,
            "occurrence_id": self.occurrence_id,
            "engine": self.engine,
            "project_snapshot_fingerprint": self.project_snapshot_fingerprint,
            "content_fingerprint_schema_version": (self.content_fingerprint_schema_version),
            "content_fingerprint": self.content_fingerprint,
            "candidate_id": self.candidate_id,
            "locator": self.locator.to_dict(),
            "translation_unit": translation_unit,
        }


@dataclass(frozen=True)
class CoverageReportDraft:
    source_fingerprint: str
    reason_codes: tuple[str, ...] = ()
    catalog_provenance: Mapping[str, Any] = field(default_factory=dict)
    catalog_freshness: str = "unknown"
    source_changed_during_scan: bool = False


@dataclass(frozen=True)
class RelocationResult:
    occurrences: tuple[Occurrence, ...]
    unresolved_occurrence_ids: tuple[str, ...] = ()
    diagnostics: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrences": [occurrence.to_dict() for occurrence in self.occurrences],
            "unresolved_occurrence_ids": list(self.unresolved_occurrence_ids),
            "diagnostics": [dict(item) for item in self.diagnostics],
        }


@dataclass(frozen=True)
class ValidationResult:
    occurrence_id: str
    engine: str
    status: str
    reason_codes: tuple[str, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    source_constraints_digest: str
    translation_digest: str
    normalized_translation: str | None = None
    validation_schema_version: int = VALIDATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_schema_version": self.validation_schema_version,
            "occurrence_id": self.occurrence_id,
            "engine": self.engine,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "diagnostics": [dict(item) for item in self.diagnostics],
            "source_constraints_digest": self.source_constraints_digest,
            "translation_digest": self.translation_digest,
            "normalized_translation": self.normalized_translation,
        }


@dataclass(frozen=True)
class ValidatedTranslation:
    occurrence: Occurrence
    translated_text: str
    validation: ValidationResult


@dataclass(frozen=True)
class WritebackOperation:
    """Declarative replacement emitted by an engine adapter.

    ``expected_fragment_sha256`` is the common consumer's live raw-span guard.
    ``expected_text_digest`` binds the adapter-decoded source text into the
    operation and plan digests; the engine-neutral consumer deliberately does
    not decode engine-specific literals a second time.
    """
    operation_id: str
    kind: str
    occurrence_id: str
    target_root: str
    target_rel_path: str
    expected_file_sha256: str
    line: int
    start_col: int
    end_col: int
    expected_fragment_sha256: str
    expected_text_digest: str
    replacement_fragment: str
    validation_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "kind": self.kind,
            "occurrence_id": self.occurrence_id,
            "target_root": self.target_root,
            "target_rel_path": self.target_rel_path,
            "expected_file_sha256": self.expected_file_sha256,
            "line": self.line,
            "start_col": self.start_col,
            "end_col": self.end_col,
            "expected_fragment_sha256": self.expected_fragment_sha256,
            "expected_text_digest": self.expected_text_digest,
            "replacement_fragment": self.replacement_fragment,
            "validation_digest": self.validation_digest,
        }


@dataclass(frozen=True)
class WritebackPlan:
    engine: str
    adapter_version: str
    project_identity_digest: str
    source_snapshot_fingerprint: str
    coverage_digest: str
    coverage_review_digest: str
    operations: tuple[WritebackOperation, ...]
    plan_digest: str
    writeback_plan_schema_version: int = WRITEBACK_PLAN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "writeback_plan_schema_version": self.writeback_plan_schema_version,
            "engine": self.engine,
            "adapter_version": self.adapter_version,
            "project_identity_digest": self.project_identity_digest,
            "source_snapshot_fingerprint": self.source_snapshot_fingerprint,
            "coverage_digest": self.coverage_digest,
            "coverage_review_digest": self.coverage_review_digest,
            "operations": [operation.to_dict() for operation in self.operations],
            "plan_digest": self.plan_digest,
        }


@runtime_checkable
class EngineAdapter(Protocol):
    protocol_version: int
    engine: str
    adapter_version: str

    def capabilities(self) -> EngineCapabilities: ...

    def discover_project(
        self,
        request: ProjectDiscoveryRequest,
    ) -> ProjectDiscovery: ...

    def inventory_candidates(
        self,
        project: ProjectDiscovery,
        policy: InventoryPolicy,
    ) -> CandidateInventory: ...

    def audit_extraction(
        self,
        project: ProjectDiscovery,
        inventory: CandidateInventory,
    ) -> CoverageReportDraft: ...

    def extract_occurrences(
        self,
        project: ProjectDiscovery,
        inventory: CandidateInventory,
        approved_candidate_ids: Sequence[str],
    ) -> Sequence[Occurrence]: ...

    def relocate_occurrences(
        self,
        project: ProjectDiscovery,
        occurrences: Sequence[Occurrence],
        live_sources: Sequence[SourceDocument],
    ) -> RelocationResult:
        """Relocate occurrences against live sources (P2).

        Unsupported adapters must fail closed rather than returning an empty
        success result.
        """
        ...

    def validate_translation(
        self,
        occurrence: Occurrence,
        translated_text: str,
    ) -> ValidationResult:
        """Validate translated text for engine format rules (P2).

        Unsupported adapters must fail closed rather than returning a pass.
        """
        ...

    def build_writeback_plan(
        self,
        project: ProjectDiscovery,
        validated: Sequence[ValidatedTranslation],
        live_sources: Sequence[SourceDocument],
    ) -> WritebackPlan:
        """Build a declarative writeback plan (P2).

        Unsupported adapters must fail closed. Adapters never receive
        arbitrary file-write authority.
        """
        ...
