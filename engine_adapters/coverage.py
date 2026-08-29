"""Coverage artifact validation, digesting, export, and review import.

Adapters provide candidates and engine evidence.  This module independently
checks inventory invariants and creates the read-only report/review package;
an adapter cannot use its own extraction result as a coverage confirmation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from atomic_io import atomic_write_json, atomic_write_text

from .contracts import (
    CANDIDATE_SCHEMA_VERSION,
    COVERAGE_DIGEST_SCHEMA_VERSION,
    COVERAGE_REVIEW_SCHEMA_VERSION,
    COVERAGE_SCHEMA_VERSION,
    ENGINE_ADAPTER_PROTOCOL_VERSION,
    CandidateInventory,
    CoverageReportDraft,
    ProjectDiscovery,
)


CLASSIFICATIONS = frozenset(
    {
        "translatable",
        "already_translated",
        "explicitly_excluded",
        "unsupported",
        "parse_error",
        "unknown",
    }
)
SCOPES = frozenset({"include", "exclude", "unknown"})
REVIEW_POLICIES = frozenset({"agent_or_human", "human_required"})
REVIEWER_TYPES = frozenset({"agent", "human"})
REVIEW_STATUSES = frozenset(
    {
        "pending",
        "agent_reviewed",
        "human_reviewed",
        "changes_requested",
        "stale",
    }
)

CANDIDATE_REASON_CODES = frozenset(
    {
        "renpy.dialogue_string",
        "renpy.narration_string",
        "renpy.translate_comment_pair",
        "renpy.old_new_pair",
        "renpy.catalog.translation_present",
        "renpy.character_display_definition",
        "renpy.keyword_argument",
        "renpy.voice_asset",
        "renpy.asset_path",
        "renpy.non_player_visible_literal",
        "project.explicit_exclusion",
        "renpy.dynamic_string_expression",
        "renpy.custom_statement_unsupported",
        "renpy.visibility_unknown",
        "renpy.tokenize_error",
        "renpy.ast_parse_error",
        "renpy.source_marker_unpaired",
        "renpy.catalog.missing_entry",
        "renpy.catalog.duplicate_entry",
        "renpy.catalog.provenance_unknown",
        "renpy.catalog.stale",
        "project.extraction_override",
        # TyranoScript V600+ P5 (#399 / #265 P5)
        "tyrano.comment",
        "tyrano.engine_control_structure",
        "tyrano.character_definition",
        "tyrano.chara_ptext",
        "tyrano.text_node",
        "tyrano.registered_tag_parameter",
        "tyrano.tag_parameter_not_registered",
        "tyrano.unregistered_macro_invocation",
        "tyrano.dynamic_parameter_expression",
        "tyrano.iscript_boundary_tag",
        "tyrano.iscript_content",
        "tyrano.lang_set_control_tag",
        "tyrano.unterminated_quoted_parameter",
        "tyrano.official_parser_compensated",
        "tyrano.unquoted_parameter_sequence",
        "tyrano.unclosed_inline_tag",
    }
)
REVIEW_FINDING_CODES = frozenset(
    {
        "review.missed_candidate",
        "review.false_positive",
        "review.wrong_classification",
        "review.duplicate_candidate",
        "review.invalid_exclusion",
    }
)
REPORT_REASON_CODES = frozenset(
    {
        "coverage.inventory.duplicate_candidate",
        "coverage.inventory.invalid_candidate",
        "coverage.inventory.source_mismatch",
        "coverage.source_changed_during_scan",
        "renpy.catalog.provenance_unknown",
        "renpy.catalog.stale",
        "renpy.catalog.missing_entry",
        "renpy.catalog.duplicate_entry",
        "tyrano.catalog.provenance_unknown",
        "tyrano.catalog.stale",
        "tyrano.catalog.missing_file",
        "tyrano.catalog.missing_scenario",
        "tyrano.catalog.missing_row",
        "tyrano.catalog.empty_translation",
        "tyrano.catalog.invalid_json",
    }
)

REVIEW_PACKAGE_TEMPLATE_VERSION = 1
DEFAULT_SAMPLING_PLAN = {
    "strategy": "all_attention_and_deterministic_supported_sample",
    "supported_sample_per_kind": 5,
}


def stable_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def source_fingerprint_payload(project: ProjectDiscovery) -> list[dict[str, Any]]:
    return sorted(
        (document.manifest_entry() for document in project.source_documents),
        key=lambda item: item["file_rel_path"],
    )


def source_fingerprint(project: ProjectDiscovery) -> str:
    return digest_json(source_fingerprint_payload(project))


def inventory_digest(inventory: CandidateInventory) -> str:
    payload = [
        candidate.to_dict()
        for candidate in sorted(
            inventory.candidates,
            key=lambda item: item.candidate_id,
        )
    ]
    return digest_json(payload)


def classification_rules_digest() -> str:
    return digest_json(
        {
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "classifications": sorted(CLASSIFICATIONS),
            "scopes": sorted(SCOPES),
            "candidate_reason_codes": sorted(CANDIDATE_REASON_CODES),
        }
    )


@dataclass(frozen=True)
class CoverageReport:
    engine: str
    adapter_version: str
    adapter_behavior_digest: str
    localization_mode: str
    source_fingerprint: str
    project_snapshot_fingerprint: str
    inventory_digest: str
    classification_rules_digest: str
    extraction_overrides_digest: str
    catalog_provenance: Mapping[str, Any]
    catalog_freshness: str
    audit_reason_codes: tuple[str, ...]
    source_changed_during_scan: bool
    files_scanned: tuple[Mapping[str, Any], ...]
    candidate_count: int
    classification_counts: Mapping[str, int]
    translation_scope_counts: Mapping[str, int]
    analysis_scope_counts: Mapping[str, int]
    reason_counts: Mapping[str, int]
    coverage_status: str
    coverage_digest: str
    generated_at: str
    invariant_errors: tuple[str, ...] = ()
    engine_adapter_protocol_version: int = ENGINE_ADAPTER_PROTOCOL_VERSION
    candidate_schema_version: int = CANDIDATE_SCHEMA_VERSION
    coverage_schema_version: int = COVERAGE_SCHEMA_VERSION
    coverage_digest_schema_version: int = COVERAGE_DIGEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_schema_version": self.coverage_schema_version,
            "coverage_digest_schema_version": self.coverage_digest_schema_version,
            "candidate_schema_version": self.candidate_schema_version,
            "engine_adapter_protocol_version": (self.engine_adapter_protocol_version),
            "engine": self.engine,
            "adapter_version": self.adapter_version,
            "adapter_behavior_digest": self.adapter_behavior_digest,
            "localization_mode": self.localization_mode,
            "source_fingerprint": self.source_fingerprint,
            "project_snapshot_fingerprint": self.project_snapshot_fingerprint,
            "inventory_digest": self.inventory_digest,
            "classification_rules_digest": self.classification_rules_digest,
            "extraction_overrides_digest": self.extraction_overrides_digest,
            "catalog_provenance": dict(self.catalog_provenance),
            "catalog_freshness": self.catalog_freshness,
            "audit_reason_codes": list(self.audit_reason_codes),
            "source_changed_during_scan": self.source_changed_during_scan,
            "files_scanned": [dict(item) for item in self.files_scanned],
            "candidate_count": self.candidate_count,
            "classification_counts": dict(self.classification_counts),
            "translation_scope_counts": dict(self.translation_scope_counts),
            "analysis_scope_counts": dict(self.analysis_scope_counts),
            "reason_counts": dict(self.reason_counts),
            "coverage_status": self.coverage_status,
            "coverage_digest": self.coverage_digest,
            "generated_at": self.generated_at,
            "invariant_errors": list(self.invariant_errors),
        }


def _inventory_invariant_errors(
    project: ProjectDiscovery,
    inventory: CandidateInventory,
) -> tuple[list[str], Counter[str]]:
    errors: list[str] = []
    report_reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_positions: set[str] = set()

    if project.source_fingerprint != source_fingerprint(project):
        errors.append("Project source fingerprint does not match source documents.")
        report_reasons["coverage.inventory.source_mismatch"] += 1
    if inventory.source_fingerprint != project.source_fingerprint:
        errors.append("Inventory source fingerprint does not match project discovery.")
        report_reasons["coverage.inventory.source_mismatch"] += 1
    if inventory.project_snapshot_fingerprint != project.project_snapshot_fingerprint:
        errors.append("Inventory project snapshot does not match project discovery.")
        report_reasons["coverage.inventory.source_mismatch"] += 1
    if inventory.engine != project.engine:
        errors.append("Inventory engine does not match project discovery.")
        report_reasons["coverage.inventory.invalid_candidate"] += 1
    if inventory.adapter_version != project.adapter_version:
        errors.append("Inventory adapter version does not match project discovery.")
        report_reasons["coverage.inventory.invalid_candidate"] += 1

    expected_files = {
        document.file_rel_path: document.manifest_entry() for document in project.source_documents
    }
    reported_files: dict[str, Mapping[str, Any]] = {}
    for file_entry in inventory.files_scanned:
        rel_path = str(file_entry.get("file_rel_path") or "")
        if not rel_path or rel_path in reported_files:
            errors.append(f"Invalid or duplicate files_scanned path: {rel_path!r}")
            report_reasons["coverage.inventory.invalid_candidate"] += 1
            continue
        reported_files[rel_path] = file_entry
    if set(reported_files) != set(expected_files):
        errors.append("Inventory files_scanned does not match project discovery.")
        report_reasons["coverage.inventory.source_mismatch"] += 1
    for rel_path in set(reported_files) & set(expected_files):
        reported = reported_files[rel_path]
        expected = expected_files[rel_path]
        if (
            str(reported.get("sha256") or "") != expected["sha256"]
            or int(reported.get("size") or 0) != expected["size"]
        ):
            errors.append(f"Inventory file snapshot mismatch: {rel_path}")
            report_reasons["coverage.inventory.source_mismatch"] += 1

    for candidate in inventory.candidates:
        if candidate.candidate_id in seen_ids:
            errors.append(f"Duplicate candidate_id: {candidate.candidate_id}")
            report_reasons["coverage.inventory.duplicate_candidate"] += 1
        seen_ids.add(candidate.candidate_id)

        position_key = stable_json_dumps(candidate.locator.to_dict())
        if position_key in seen_positions:
            errors.append(f"Duplicate candidate position: {candidate.candidate_id}")
            report_reasons["coverage.inventory.duplicate_candidate"] += 1
        seen_positions.add(position_key)

        invalid = False
        if candidate.candidate_schema_version != CANDIDATE_SCHEMA_VERSION:
            invalid = True
        if not candidate.candidate_id:
            invalid = True
        if candidate.engine != inventory.engine:
            invalid = True
        if candidate.adapter_version != inventory.adapter_version:
            invalid = True
        if candidate.source_fingerprint != inventory.source_fingerprint:
            invalid = True
        if candidate.classification not in CLASSIFICATIONS:
            invalid = True
        if candidate.translation_scope not in SCOPES:
            invalid = True
        if candidate.analysis_scope not in SCOPES:
            invalid = True
        if not candidate.reason_codes:
            invalid = True
        if any(code not in CANDIDATE_REASON_CODES for code in candidate.reason_codes):
            invalid = True
        if candidate.locator.engine != inventory.engine:
            invalid = True
        if int(candidate.locator.locator_schema_version or 0) <= 0:
            invalid = True
        expected_translation_scope = {
            "translatable": "include",
            "already_translated": "include",
            "explicitly_excluded": "exclude",
            "unsupported": "unknown",
            "parse_error": "unknown",
            "unknown": "unknown",
        }.get(candidate.classification)
        if candidate.translation_scope != expected_translation_scope:
            invalid = True
        if invalid:
            errors.append(f"Invalid candidate contract: {candidate.candidate_id}")
            report_reasons["coverage.inventory.invalid_candidate"] += 1

    return errors, report_reasons


def build_coverage_report(
    project: ProjectDiscovery,
    inventory: CandidateInventory,
    draft: CoverageReportDraft,
    *,
    adapter_behavior_digest: str,
    extraction_overrides_digest: str = "",
    generated_at: str | None = None,
) -> CoverageReport:
    """Validate an inventory and produce the common-layer coverage report."""
    invariant_errors, report_reasons = _inventory_invariant_errors(project, inventory)
    if draft.source_fingerprint != project.source_fingerprint:
        invariant_errors.append("Audit source fingerprint does not match project discovery.")
        report_reasons["coverage.inventory.source_mismatch"] += 1
    classification_counts = Counter(candidate.classification for candidate in inventory.candidates)
    translation_scope_counts = Counter(
        candidate.translation_scope for candidate in inventory.candidates
    )
    analysis_scope_counts = Counter(candidate.analysis_scope for candidate in inventory.candidates)
    reason_counts = Counter(
        code for candidate in inventory.candidates for code in candidate.reason_codes
    )
    audit_reason_codes = tuple(sorted(set(draft.reason_codes)))
    for code in audit_reason_codes:
        if code not in CANDIDATE_REASON_CODES and code not in REPORT_REASON_CODES:
            invariant_errors.append(f"Unknown report reason code: {code}")
            report_reasons["coverage.inventory.invalid_candidate"] += 1
        reason_counts[code] += 1
    reason_counts.update(report_reasons)

    if draft.source_changed_during_scan:
        reason_counts["coverage.source_changed_during_scan"] += 1

    if (
        invariant_errors
        or draft.source_changed_during_scan
        or classification_counts["unknown"]
        or classification_counts["parse_error"]
        or draft.catalog_freshness == "missing"
        or any(
            code in reason_counts
            for code in {
                "tyrano.catalog.missing_file",
                "tyrano.catalog.missing_scenario",
                "tyrano.catalog.missing_row",
                "tyrano.catalog.empty_translation",
                "tyrano.catalog.invalid_json",
            }
        )
    ):
        status = "block"
    elif (
        classification_counts["unsupported"]
        or draft.catalog_freshness in {"unknown", "stale", "missing"}
        or any(
            code in reason_counts
            for code in {
                "renpy.catalog.provenance_unknown",
                "renpy.catalog.stale",
                "renpy.catalog.missing_entry",
                "renpy.catalog.duplicate_entry",
            }
        )
    ):
        status = "attention"
    else:
        status = "ready"

    inv_digest = inventory_digest(inventory)
    rules_digest = classification_rules_digest()
    overrides_digest = extraction_overrides_digest or digest_json([])
    digest_payload = {
        "coverage_digest_schema_version": COVERAGE_DIGEST_SCHEMA_VERSION,
        "engine": project.engine,
        "protocol_version": ENGINE_ADAPTER_PROTOCOL_VERSION,
        "adapter_version": project.adapter_version,
        "adapter_behavior_digest": adapter_behavior_digest,
        "localization_mode": project.localization_mode.value,
        "source_fingerprint": project.source_fingerprint,
        "catalog_digest_and_provenance": {
            "freshness": draft.catalog_freshness,
            "provenance": dict(draft.catalog_provenance),
        },
        "audit": {
            "reason_codes": list(audit_reason_codes),
            "source_changed_during_scan": bool(draft.source_changed_during_scan),
        },
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "coverage_schema_version": COVERAGE_SCHEMA_VERSION,
        "inventory_digest": inv_digest,
        "classification_rules_digest": rules_digest,
        "extraction_overrides_digest": overrides_digest,
    }
    coverage_digest = digest_json(digest_payload)
    created = generated_at or datetime.now(timezone.utc).isoformat()
    return CoverageReport(
        engine=project.engine,
        adapter_version=project.adapter_version,
        adapter_behavior_digest=adapter_behavior_digest,
        localization_mode=project.localization_mode.value,
        source_fingerprint=project.source_fingerprint,
        project_snapshot_fingerprint=project.project_snapshot_fingerprint,
        inventory_digest=inv_digest,
        classification_rules_digest=rules_digest,
        extraction_overrides_digest=overrides_digest,
        catalog_provenance=dict(draft.catalog_provenance),
        catalog_freshness=draft.catalog_freshness,
        audit_reason_codes=audit_reason_codes,
        source_changed_during_scan=bool(draft.source_changed_during_scan),
        files_scanned=inventory.files_scanned,
        candidate_count=len(inventory.candidates),
        classification_counts={
            key: classification_counts.get(key, 0) for key in sorted(CLASSIFICATIONS)
        },
        translation_scope_counts={
            key: translation_scope_counts.get(key, 0) for key in sorted(SCOPES)
        },
        analysis_scope_counts={key: analysis_scope_counts.get(key, 0) for key in sorted(SCOPES)},
        reason_counts=dict(sorted(reason_counts.items())),
        coverage_status=status,
        coverage_digest=coverage_digest,
        generated_at=created,
        invariant_errors=tuple(invariant_errors),
    )


@dataclass(frozen=True)
class CoverageFreshness:
    effective_status: str
    stale_reasons: tuple[str, ...]


def validate_coverage_report_freshness(
    record: Mapping[str, Any] | CoverageReport,
    project: ProjectDiscovery,
    *,
    adapter_behavior_digest: str,
) -> CoverageFreshness:
    """Compare a saved report with the current adapter/rules/source inputs."""
    payload = record.to_dict() if isinstance(record, CoverageReport) else dict(record)
    if int(payload.get("coverage_schema_version") or 0) != COVERAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported coverage report schema version.")
    if int(payload.get("coverage_digest_schema_version") or 0) != COVERAGE_DIGEST_SCHEMA_VERSION:
        raise ValueError("Unsupported coverage digest schema version.")
    if int(payload.get("candidate_schema_version") or 0) != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("Unsupported candidate schema version.")
    if int(payload.get("engine_adapter_protocol_version") or 0) != ENGINE_ADAPTER_PROTOCOL_VERSION:
        raise ValueError("Unsupported engine adapter protocol version.")

    stale_reasons: list[str] = []
    comparisons = (
        ("engine", project.engine),
        ("adapter_version", project.adapter_version),
        ("adapter_behavior_digest", adapter_behavior_digest),
        ("localization_mode", project.localization_mode.value),
        ("source_fingerprint", project.source_fingerprint),
        (
            "project_snapshot_fingerprint",
            project.project_snapshot_fingerprint,
        ),
        ("classification_rules_digest", classification_rules_digest()),
    )
    for field_name, expected in comparisons:
        if str(payload.get(field_name) or "") != str(expected or ""):
            stale_reasons.append(field_name)

    expected_coverage_digest = digest_json(
        {
            "coverage_digest_schema_version": COVERAGE_DIGEST_SCHEMA_VERSION,
            "engine": payload.get("engine"),
            "protocol_version": payload.get("engine_adapter_protocol_version"),
            "adapter_version": payload.get("adapter_version"),
            "adapter_behavior_digest": payload.get("adapter_behavior_digest"),
            "localization_mode": payload.get("localization_mode"),
            "source_fingerprint": payload.get("source_fingerprint"),
            "catalog_digest_and_provenance": {
                "freshness": payload.get("catalog_freshness"),
                "provenance": payload.get("catalog_provenance") or {},
            },
            "audit": {
                "reason_codes": payload.get("audit_reason_codes") or [],
                "source_changed_during_scan": bool(payload.get("source_changed_during_scan")),
            },
            "candidate_schema_version": payload.get("candidate_schema_version"),
            "coverage_schema_version": payload.get("coverage_schema_version"),
            "inventory_digest": payload.get("inventory_digest"),
            "classification_rules_digest": payload.get("classification_rules_digest"),
            "extraction_overrides_digest": payload.get("extraction_overrides_digest"),
        }
    )
    if str(payload.get("coverage_digest") or "") != expected_coverage_digest:
        stale_reasons.append("coverage_digest")

    recorded_status = str(payload.get("coverage_status") or "")
    if recorded_status not in {"ready", "attention", "block", "stale"}:
        raise ValueError(f"Unsupported coverage status: {recorded_status}")
    return CoverageFreshness(
        effective_status="stale" if stale_reasons else recorded_status,
        stale_reasons=tuple(stale_reasons),
    )


def review_input_digest(
    report: CoverageReport,
    *,
    review_policy: str,
    sampling_plan: Mapping[str, Any] | None = None,
) -> str:
    if review_policy not in REVIEW_POLICIES:
        raise ValueError(f"Unsupported review policy: {review_policy}")
    plan = dict(sampling_plan or DEFAULT_SAMPLING_PLAN)
    return digest_json(
        {
            "coverage_review_schema_version": COVERAGE_REVIEW_SCHEMA_VERSION,
            "coverage_digest": report.coverage_digest,
            "inventory_digest": report.inventory_digest,
            "review_policy": review_policy,
            "review_package_template_version": REVIEW_PACKAGE_TEMPLATE_VERSION,
            "sampling_plan": plan,
        }
    )


def build_review_template(
    report: CoverageReport,
    *,
    review_policy: str = "agent_or_human",
    sampling_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict(sampling_plan or DEFAULT_SAMPLING_PLAN)
    return {
        "coverage_review_schema_version": COVERAGE_REVIEW_SCHEMA_VERSION,
        "source_fingerprint": report.source_fingerprint,
        "coverage_digest": report.coverage_digest,
        "review_input_digest": review_input_digest(
            report,
            review_policy=review_policy,
            sampling_plan=plan,
        ),
        "review_policy": review_policy,
        "sampling_plan": plan,
        "reviewer": {
            "type": "",
            "id": "",
            "tool": "",
            "model": "",
            "session": "",
        },
        "status": "pending",
        "findings": [],
        "confirmed_at": "",
    }


def _markdown_cell(value: Any) -> str:
    return (
        str(value or "")
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", "")
        .replace("\n", " ")
    )


def render_review_markdown(
    project: ProjectDiscovery,
    inventory: CandidateInventory,
    report: CoverageReport,
) -> str:
    lines = [
        "# Coverage review package\n\n",
        "> This package is generated evidence, not a coverage confirmation. ",
        "Review raw scripts against the complete inventory before recording a result.\n\n",
        "## Summary\n\n",
        f"- Engine: `{report.engine}`\n",
        f"- Adapter: `{report.adapter_version}`\n",
        f"- Source fingerprint: `{report.source_fingerprint}`\n",
        f"- Coverage digest: `{report.coverage_digest}`\n",
        f"- Automatic status: `{report.coverage_status}`\n",
        f"- Candidates: {report.candidate_count}\n\n",
        "## Source files\n\n",
        "| File | Bytes | SHA-256 |\n",
        "| --- | ---: | --- |\n",
    ]
    for item in report.files_scanned:
        lines.append(
            f"| {_markdown_cell(item.get('file_rel_path'))} "
            f"| {int(item.get('size') or 0)} "
            f"| `{_markdown_cell(item.get('sha256'))}` |\n"
        )

    lines.extend(
        [
            "\n## Review procedure\n\n",
            "1. Open every source file listed above and compare it with this inventory.\n",
            "2. Check every excluded, unsupported, parse-error, and unknown candidate.\n",
            "3. Sample every supported structure kind, including translated entries.\n",
            "4. Record missed, duplicate, false-positive, wrongly classified, or ",
            "invalidly excluded candidates in `coverage_review_template.json`.\n",
            "5. Fix the adapter or add an auditable project override, then regenerate ",
            "the package; do not patch text directly into the review record.\n\n",
            "## Candidate inventory\n\n",
            "| ID | Locator | Kind | Classification | Reasons | Text / context | Evidence |\n",
            "| --- | --- | --- | --- | --- | --- | --- |\n",
        ]
    )
    for candidate in sorted(
        inventory.candidates,
        key=lambda item: (
            str(item.locator.locator.get("file_rel_path") or ""),
            int(item.locator.locator.get("line_hint") or 0),
            int(item.locator.locator.get("start_col_hint") or 0),
            item.candidate_id,
        ),
    ):
        lines.append(
            f"| `{_markdown_cell(candidate.candidate_id)}` "
            f"| `{_markdown_cell(stable_json_dumps(candidate.locator.to_dict()))}` "
            f"| {_markdown_cell(candidate.structure_kind)} "
            f"| `{_markdown_cell(candidate.classification)}` "
            f"| {_markdown_cell(', '.join(candidate.reason_codes))} "
            f"| {_markdown_cell(candidate.raw_excerpt)} "
            f"| `{_markdown_cell(stable_json_dumps(candidate.evidence))}` |\n"
        )
    lines.append("\n")
    return "".join(lines)


@dataclass(frozen=True)
class CoveragePackagePaths:
    package_dir: str
    candidates_path: str
    report_path: str
    review_markdown_path: str
    review_template_path: str


def export_coverage_package(
    output_dir: str | os.PathLike[str],
    project: ProjectDiscovery,
    inventory: CandidateInventory,
    report: CoverageReport,
    *,
    review_policy: str = "agent_or_human",
) -> CoveragePackagePaths:
    """Write read-only evidence artifacts outside the adapter."""
    if report.inventory_digest != inventory_digest(inventory):
        raise ValueError("Coverage report does not match candidate inventory.")
    freshness = validate_coverage_report_freshness(
        report,
        project,
        adapter_behavior_digest=report.adapter_behavior_digest,
    )
    if freshness.effective_status == "stale":
        raise ValueError("Coverage report is stale: " + ", ".join(freshness.stale_reasons))
    package_dir = Path(output_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = package_dir / "coverage_candidates.jsonl"
    report_path = package_dir / "coverage_report.json"
    review_markdown_path = package_dir / "coverage_review.md"
    review_template_path = package_dir / "coverage_review_template.json"

    candidate_text = "".join(
        stable_json_dumps(candidate.to_dict()) + "\n"
        for candidate in sorted(
            inventory.candidates,
            key=lambda item: item.candidate_id,
        )
    )
    atomic_write_text(candidates_path, candidate_text, encoding="utf-8")
    atomic_write_json(
        report_path,
        report.to_dict(),
        ensure_ascii=False,
        indent=2,
    )
    atomic_write_text(
        review_markdown_path,
        render_review_markdown(project, inventory, report),
        encoding="utf-8",
    )
    atomic_write_json(
        review_template_path,
        build_review_template(report, review_policy=review_policy),
        ensure_ascii=False,
        indent=2,
    )
    return CoveragePackagePaths(
        package_dir=str(package_dir),
        candidates_path=str(candidates_path),
        report_path=str(report_path),
        review_markdown_path=str(review_markdown_path),
        review_template_path=str(review_template_path),
    )


def load_review_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read coverage review record: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Coverage review record must be a JSON object.")
    return payload


@dataclass(frozen=True)
class ReviewValidation:
    effective_status: str
    review_policy: str
    policy_satisfied: bool
    unresolved_findings: int
    stale_reasons: tuple[str, ...]
    coverage_review_digest: str


def validate_review_record(
    record: Mapping[str, Any],
    report: CoverageReport,
    inventory: CandidateInventory,
) -> ReviewValidation:
    """Validate a structured human/agent review without mutating its provenance."""
    if report.inventory_digest != inventory_digest(inventory):
        raise ValueError("Coverage report does not match candidate inventory.")
    if int(record.get("coverage_review_schema_version") or 0) != COVERAGE_REVIEW_SCHEMA_VERSION:
        raise ValueError("Unsupported coverage review schema version.")
    review_policy = str(record.get("review_policy") or "")
    if review_policy not in REVIEW_POLICIES:
        raise ValueError(f"Unsupported review policy: {review_policy}")
    recorded_status = str(record.get("status") or "")
    if recorded_status not in REVIEW_STATUSES:
        raise ValueError(f"Unsupported review status: {recorded_status}")

    reviewer = record.get("reviewer")
    if not isinstance(reviewer, Mapping):
        raise ValueError("Review record reviewer must be an object.")
    reviewer_type = str(reviewer.get("type") or "")
    if reviewer_type and reviewer_type not in REVIEWER_TYPES:
        raise ValueError(f"Unsupported reviewer.type: {reviewer_type}")
    if recorded_status != "pending" and reviewer_type not in REVIEWER_TYPES:
        raise ValueError("Completed review must identify an agent or human reviewer.")
    if recorded_status != "pending" and not str(reviewer.get("id") or "").strip():
        raise ValueError("Completed review must identify reviewer.id.")
    if recorded_status == "human_reviewed" and reviewer_type != "human":
        raise ValueError("human_reviewed status requires reviewer.type=human.")
    if recorded_status == "agent_reviewed" and reviewer_type != "agent":
        raise ValueError("agent_reviewed status requires reviewer.type=agent.")
    if reviewer_type == "agent" and recorded_status != "pending":
        if not (
            str(reviewer.get("tool") or "").strip() or str(reviewer.get("model") or "").strip()
        ):
            raise ValueError("Agent review provenance requires reviewer.tool or reviewer.model.")
    if recorded_status != "pending" and not str(record.get("confirmed_at") or "").strip():
        raise ValueError("Completed review must record confirmed_at.")

    candidates_by_id = inventory.by_id()
    findings = record.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Review record findings must be a list.")
    unresolved = 0
    for finding in findings:
        if not isinstance(finding, Mapping):
            raise ValueError("Each coverage review finding must be an object.")
        code = str(finding.get("code") or "")
        if code not in REVIEW_FINDING_CODES:
            raise ValueError(f"Unsupported review finding code: {code}")
        candidate_id = str(finding.get("candidate_id") or "")
        if code != "review.missed_candidate":
            if not candidate_id:
                raise ValueError(f"{code} finding requires candidate_id.")
            if candidate_id not in candidates_by_id:
                raise ValueError(f"Review finding references unknown candidate: {candidate_id}")
        resolved = finding.get("resolved")
        if not isinstance(resolved, bool):
            raise ValueError("Review finding resolved must be a boolean.")
        if not resolved:
            unresolved += 1

    stale_reasons: list[str] = []
    if str(record.get("source_fingerprint") or "") != report.source_fingerprint:
        stale_reasons.append("source_fingerprint")
    if str(record.get("coverage_digest") or "") != report.coverage_digest:
        stale_reasons.append("coverage_digest")
    expected_review_input = review_input_digest(
        report,
        review_policy=review_policy,
        sampling_plan=record.get("sampling_plan") or DEFAULT_SAMPLING_PLAN,
    )
    if str(record.get("review_input_digest") or "") != expected_review_input:
        stale_reasons.append("review_input_digest")

    effective_status = "stale" if stale_reasons else recorded_status
    policy_satisfied = (
        effective_status == "human_reviewed"
        if review_policy == "human_required"
        else effective_status in {"agent_reviewed", "human_reviewed"}
    )
    normalized_record = dict(record)
    normalized_record.pop("display_message", None)
    # Display messages are for humans only and must not affect provenance digests.
    normalized_record["findings"] = [
        {
            key: value
            for key, value in dict(finding).items()
            if key != "display_message"
        }
        for finding in findings
        if isinstance(finding, Mapping)
    ]
    coverage_review_digest = digest_json(normalized_record)
    return ReviewValidation(
        effective_status=effective_status,
        review_policy=review_policy,
        policy_satisfied=policy_satisfied and unresolved == 0,
        unresolved_findings=unresolved,
        stale_reasons=tuple(stale_reasons),
        coverage_review_digest=coverage_review_digest,
    )
