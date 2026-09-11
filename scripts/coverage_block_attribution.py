"""Offline, read-only attribution of Ren'Py coverage-block candidates (issue #426).

This helper is deliberately CLI-only.  It does not change adapter
classification, coverage status, writeback, GUI, doctor, or Tyrano behavior.
It runs the existing Ren'Py discovery -> inventory -> audit -> coverage
pipeline once and groups non-ready candidates by classification, structure
kind, reason code, and observed syntax.  Every group keeps all candidate
locators, and an independent quote-run scan reports source spans that did not
produce a candidate.

Usage::

    python scripts/coverage_block_attribution.py path/to/game/tl/schinese \\
        --project-root path/to/game \\
        --target-language schinese \\
        --json-output logs/attribution/coverage_attribution.json \\
        --markdown-output logs/attribution/coverage_attribution.md \\
        --generated-at 2026-09-11T00:00:00+00:00

Human/manual judgment can be recorded without changing the scanner:

    python scripts/coverage_block_attribution.py path/to/tl \\
        --decisions logs/attribution/decisions.json

A decisions file is JSON::

    {
      "schema_version": 1,
      "reviewer": {"type": "human", "id": "local-reviewer"},
      "decisions": [
        {
          "group_key": "parse_error|tokenize_region|renpy.tokenize_error|",
          "category": "parser_defect",
          "rationale": "Confirmed multi-line string gap in the fixture."
        }
      ]
    }

The script never writes to the scanned project.  Output is desensitized by
default: only relative locators, structure kinds, and fingerprints are
included; raw excerpts require ``--include-excerpts`` and must not be
published from a private project.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ATTRIBUTION_SCHEMA_VERSION = 1
AGGREGATION_DIGEST_SCHEMA_VERSION = 1
DECISIONS_SCHEMA_VERSION = 1

BLOCKING_CLASSIFICATIONS = ("unknown", "unsupported", "parse_error")
CLASSIFICATION_ORDER = {
    "parse_error": 0,
    "unknown": 1,
    "unsupported": 2,
    "explicitly_excluded": 3,
    "already_translated": 4,
    "translatable": 5,
}

AUTO_CATEGORIES = frozenset(
    {
        "parser_defect_candidate",
        "unsupported_structure",
        "legitimate_exclusion",
        "false_positive_suspect",
        "unknown",
        "translated",
        "pending_translation",
    }
)
MANUAL_CATEGORIES = frozenset(
    {
        "parser_defect",
        "unsupported_structure",
        "legitimate_exclusion",
        "false_positive",
        "unknown",
    }
)
REVIEWER_TYPES = frozenset({"human", "agent"})

PARSER_GAP_REASONS = frozenset({"renpy.tokenize_error", "renpy.ast_parse_error"})
UNSUPPORTED_REASONS = frozenset(
    {
        "renpy.dynamic_string_expression",
        "renpy.custom_statement_unsupported",
    }
)
BLOCKING_REASON_CODES = frozenset(
    {
        "coverage.inventory.duplicate_candidate",
        "coverage.inventory.invalid_candidate",
        "coverage.inventory.source_mismatch",
        "coverage.source_changed_during_scan",
        "renpy.tokenize_error",
        "renpy.ast_parse_error",
        "renpy.source_marker_unpaired",
        "tyrano.catalog.missing_file",
        "tyrano.catalog.missing_scenario",
        "tyrano.catalog.missing_row",
        "tyrano.catalog.empty_translation",
        "tyrano.catalog.invalid_json",
        "tyrano.catalog.stale",
        "tyrano.catalog.language_code_invalid",
        "tyrano.catalog.language_code_collision",
        "tyrano.catalog.tag_registry_mismatch",
        "tyrano.lang_set.language_code_invalid",
        "tyrano.lang_set.catalog_missing",
        "tyrano.lang_set.target_mismatch",
    }
)

_DEVELOPER_ASSIGNMENT_RE = re.compile(r"^\s*(?:\$|define|default)\b")
_ASSERTION_RE = re.compile(r"^\s*(?:\$\s*)?assert\b")
_QUOTED_SPAN_RE = re.compile(r"""^\s*(?:[A-Za-z_]\w*\s+)?["']""")
_SINGLE_QUOTE_SPAN_RE = re.compile(r"""^\s*(?:[A-Za-z_]\w*\s+)?'""")


class AttributionError(RuntimeError):
    """Raised when attribution inputs or invariants cannot be honored."""


@dataclass(frozen=True)
class ManualDecisions:
    """Validated manual reviewer input keyed by deterministic group key."""

    reviewer: Mapping[str, Any]
    by_group: Mapping[str, Mapping[str, Any]]
    source_label: str


@dataclass(frozen=True)
class _CandidateRow:
    group_key: str
    candidate_id: str
    classification: str
    structure_kind: str
    reason_codes: tuple[str, ...]
    review_flags: tuple[str, ...]
    auto_category: str
    locator: Mapping[str, Any]


@dataclass(frozen=True)
class _RawSpan:
    file_rel_path: str
    line_index: int
    start_col: int
    end_col: int
    quote: str
    continues_from_previous: bool
    continues_to_next: bool


def group_key(
    classification: str,
    structure_kind: str,
    reason_codes: Sequence[str],
    review_flags: Sequence[str] = (),
) -> str:
    """Return the deterministic group key used by JSON and decision files."""

    return "{}|{}|{}|{}".format(
        classification,
        structure_kind,
        ",".join(reason_codes),
        ",".join(review_flags),
    )


def _bounded_excerpt(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _aggregation_digest(payload: Mapping[str, Any]) -> str:
    stable_inputs = {
        key: value
        for key, value in payload["inputs"].items()
        if key not in {"project_root_label", "localization_root_label"}
    }
    digest_input = {
        "aggregation_digest_schema_version": AGGREGATION_DIGEST_SCHEMA_VERSION,
        "inputs": stable_inputs,
        "coverage": {
            "coverage_status": payload["coverage"]["coverage_status"],
            "classification_counts": payload["coverage"]["classification_counts"],
            "reason_counts": payload["coverage"]["reason_counts"],
        },
        "candidate_count": payload["candidate_count"],
        "candidate_reason_counts": payload["candidate_reason_counts"],
        "structure_kind_counts": payload["structure_kind_counts"],
        "groups": [
            {
                "group_key": group["group_key"],
                "count": group["count"],
                "effective_category": group["effective_category"],
                "human_judgment": group["human_judgment"],
            }
            for group in payload["groups"]
        ],
    }
    canonical = json.dumps(
        digest_input,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _review_flags(candidate: Any) -> tuple[str, ...]:
    """Return heuristic review hints; these never change classification."""

    flags: list[str] = []
    reasons = set(candidate.reason_codes or ())
    excerpt = str(candidate.raw_excerpt or "").strip()
    if candidate.classification == "parse_error" and "renpy.source_marker_unpaired" in reasons:
        if excerpt.startswith("#"):
            flags.append("comment_span_may_be_non_player_visible")
        elif excerpt.startswith("old "):
            flags.append("orphan_old_row_needs_catalog_evidence")
    if candidate.classification == "unknown":
        if _QUOTED_SPAN_RE.match(excerpt):
            flags.append("unmarked_string_needs_source_evidence")
        if _SINGLE_QUOTE_SPAN_RE.match(excerpt):
            flags.append("single_quote_literal_needs_official_parser_check")
    if candidate.classification == "unsupported":
        if "renpy.dynamic_string_expression" in reasons:
            flags.append("dynamic_text_may_be_player_visible")
        if "renpy.custom_statement_unsupported" in reasons:
            flags.append("nonstandard_statement_needs_structure_evidence")
    if candidate.classification == "translatable":
        if _DEVELOPER_ASSIGNMENT_RE.match(excerpt) and "=" in excerpt:
            flags.append("developer_constant_may_be_non_player_visible")
        if _ASSERTION_RE.match(excerpt):
            flags.append("assertion_message_may_be_non_player_visible")
    return tuple(sorted(set(flags)))


def _auto_category(candidate: Any, review_flags: Sequence[str]) -> str:
    reasons = set(candidate.reason_codes or ())
    if candidate.classification == "translatable":
        return "false_positive_suspect" if review_flags else "pending_translation"
    if candidate.classification == "already_translated":
        return "translated"
    if candidate.classification == "explicitly_excluded":
        return "legitimate_exclusion"
    if candidate.classification == "parse_error":
        if reasons & PARSER_GAP_REASONS:
            return "parser_defect_candidate"
        return "unknown"
    if candidate.classification == "unsupported":
        if reasons & UNSUPPORTED_REASONS:
            return "unsupported_structure"
        return "unknown"
    return "unknown"


def _locator_payload(candidate: Any, *, include_excerpts: bool) -> dict[str, Any]:
    locator = candidate.locator.locator
    payload = {
        "candidate_id": candidate.candidate_id,
        "file_rel_path": str(locator.get("file_rel_path") or ""),
        "translate_block": str(locator.get("translate_block") or ""),
        "block_occurrence": int(locator.get("block_occurrence") or 0),
        "ordinal": int(locator.get("ordinal") or 0),
        "line_hint": int(locator.get("line_hint") or 0),
        "start_col_hint": int(locator.get("start_col_hint") or 0),
        "end_col_hint": int(locator.get("end_col_hint") or 0),
        "source_marker_kind": str(locator.get("source_marker_kind") or ""),
        "candidate_ordinal": int(locator.get("candidate_ordinal") or 0),
    }
    if locator.get("multiline"):
        payload["multiline"] = True
    if locator.get("end_line_hint") is not None:
        payload["end_line_hint"] = int(locator.get("end_line_hint") or 0)
    if include_excerpts:
        payload["raw_excerpt"] = _bounded_excerpt(candidate.raw_excerpt)
    return payload


def _candidate_rows(
    candidates: Iterable[Any],
    *,
    include_excerpts: bool,
) -> list[_CandidateRow]:
    rows: list[_CandidateRow] = []
    for candidate in candidates:
        reasons = tuple(str(code) for code in candidate.reason_codes)
        review_flags = _review_flags(candidate)
        rows.append(
            _CandidateRow(
                group_key=group_key(
                    str(candidate.classification),
                    str(candidate.structure_kind),
                    reasons,
                    review_flags,
                ),
                candidate_id=str(candidate.candidate_id),
                classification=str(candidate.classification),
                structure_kind=str(candidate.structure_kind),
                reason_codes=reasons,
                review_flags=review_flags,
                auto_category=_auto_category(candidate, review_flags),
                locator=_locator_payload(candidate, include_excerpts=include_excerpts),
            )
        )
    return rows


def _group_sort_key(group: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        CLASSIFICATION_ORDER.get(str(group["classification"]), 99),
        -int(group["count"]),
        str(group["structure_kind"]),
        ",".join(group["reason_codes"]),
        ",".join(group["review_flags"]),
    )


def _build_groups(
    rows: Sequence[_CandidateRow],
    decisions: ManualDecisions | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[_CandidateRow]] = {}
    for row in rows:
        grouped.setdefault(row.group_key, []).append(row)

    groups: list[dict[str, Any]] = []
    for key, members in grouped.items():
        members_sorted = sorted(
            members,
            key=lambda item: (
                item.locator["file_rel_path"],
                item.locator["line_hint"],
                item.locator["start_col_hint"],
                item.candidate_id,
            ),
        )
        first = members_sorted[0]
        automatic_evidence = {
            "category": first.auto_category,
            "evidence_class": "heuristic" if first.review_flags else "automatic",
            "basis": [
                f"classification={first.classification}",
                f"structure_kind={first.structure_kind}",
                "reason_codes=" + ",".join(first.reason_codes),
            ],
        }
        judgment: dict[str, Any] | None = None
        if decisions is not None and key in decisions.by_group:
            decision = decisions.by_group[key]
            judgment = {
                "category": str(decision["category"]),
                "rationale": str(decision["rationale"]),
                "reviewer": dict(decisions.reviewer),
            }
        effective_category = str(judgment["category"]) if judgment else first.auto_category
        group = {
            "group_key": key,
            "classification": first.classification,
            "structure_kind": first.structure_kind,
            "reason_codes": list(first.reason_codes),
            "review_flags": list(first.review_flags),
            "count": len(members_sorted),
            "automatic_evidence": automatic_evidence,
            "human_judgment": judgment,
            "effective_category": effective_category,
            "requires_human_review": effective_category
            in {"unknown", "parser_defect_candidate", "false_positive_suspect"},
            "follow_up_needed": effective_category
            in {
                "unknown",
                "parser_defect_candidate",
                "unsupported_structure",
                "false_positive_suspect",
            },
            "locators": [dict(item.locator) for item in members_sorted],
        }
        groups.append(group)
    groups.sort(key=_group_sort_key)
    return groups


def _opening_quote(line: str, index: int) -> str | None:
    for quote in ('"""', "'''"):
        if line.startswith(quote, index):
            return quote
    char = line[index]
    if char not in {'"', "'"}:
        return None
    if char == "'" and index > 0 and index + 1 < len(line):
        previous = line[index - 1]
        following = line[index + 1]
        if previous.isalnum() and following.isalnum():
            # Common English contraction inside dialogue, not a delimiter.
            return None
    return char


def _find_closing_quote(line: str, start: int, quote: str) -> int | None:
    if len(quote) == 3:
        position = line.find(quote, start)
        return position if position >= 0 else None
    index = start
    while index < len(line):
        char = line[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return index
        index += 1
    return None


def _scan_document_string_spans(
    file_rel_path: str,
    lines: Sequence[str],
) -> list[_RawSpan]:
    """Independently scan quote runs without reusing the adapter tokenizer."""

    spans: list[_RawSpan] = []
    pending: str | None = None
    for line_index, raw_line in enumerate(lines):
        line = raw_line.rstrip("\r\n")
        index = 0
        if pending is not None:
            if not line:
                continue
            closing = _find_closing_quote(line, 0, pending)
            if closing is None:
                spans.append(
                    _RawSpan(
                        file_rel_path=file_rel_path,
                        line_index=line_index,
                        start_col=0,
                        end_col=len(line),
                        quote=pending,
                        continues_from_previous=True,
                        continues_to_next=True,
                    )
                )
                continue
            end_col = closing + len(pending)
            spans.append(
                _RawSpan(
                    file_rel_path=file_rel_path,
                    line_index=line_index,
                    start_col=0,
                    end_col=end_col,
                    quote=pending,
                    continues_from_previous=True,
                    continues_to_next=False,
                )
            )
            pending = None
            index = end_col

        while index < len(line):
            if line[index] == "#":
                break
            quote = _opening_quote(line, index)
            if quote is None:
                index += 1
                continue
            closing = _find_closing_quote(line, index + len(quote), quote)
            if closing is None:
                spans.append(
                    _RawSpan(
                        file_rel_path=file_rel_path,
                        line_index=line_index,
                        start_col=index,
                        end_col=len(line),
                        quote=quote,
                        continues_from_previous=False,
                        continues_to_next=True,
                    )
                )
                pending = quote
                break
            end_col = closing + len(quote)
            spans.append(
                _RawSpan(
                    file_rel_path=file_rel_path,
                    line_index=line_index,
                    start_col=index,
                    end_col=end_col,
                    quote=quote,
                    continues_from_previous=False,
                    continues_to_next=False,
                )
            )
            index = end_col
    return spans


def _span_payload(
    span: _RawSpan,
    document: Any,
    *,
    include_excerpts: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "file_rel_path": span.file_rel_path,
        "line_hint": span.line_index + 1,
        "start_col_hint": span.start_col,
        "end_col_hint": span.end_col,
        "quote": span.quote,
        "continues_from_previous": span.continues_from_previous,
        "continues_to_next": span.continues_to_next,
    }
    if include_excerpts:
        lines = document.lines()
        if 0 <= span.line_index < len(lines):
            raw_line = lines[span.line_index].rstrip("\r\n")
            payload["excerpt"] = _bounded_excerpt(raw_line[span.start_col : span.end_col])
    return payload


def _scan_completeness(snapshot: Any, *, include_excerpts: bool) -> dict[str, Any]:
    documents = {document.file_rel_path: document for document in snapshot.project.source_documents}
    candidate_spans: dict[str, dict[int, list[tuple[int, int, str]]]] = {}
    for candidate in snapshot.inventory.candidates:
        locator = candidate.locator.locator
        file_rel_path = str(locator.get("file_rel_path") or "")
        start_line_index = max(int(locator.get("line_hint") or 0) - 1, 0)
        end_line_index = max(
            int(locator.get("end_line_hint") or start_line_index + 1) - 1,
            start_line_index,
        )
        document = documents.get(file_rel_path)
        for line_index in range(start_line_index, end_line_index + 1):
            if line_index == start_line_index:
                start_col = int(locator.get("start_col_hint") or 0)
            else:
                start_col = 0
            if document is not None and line_index < len(document.lines()):
                line_length = len(document.lines()[line_index].rstrip("\r\n"))
            else:
                line_length = 0
            if line_index == end_line_index:
                end_col = int(locator.get("end_col_hint") or 0)
            else:
                end_col = line_length
            candidate_spans.setdefault(file_rel_path, {}).setdefault(line_index, []).append(
                (
                    start_col,
                    max(end_col, start_col + 1),
                    str(candidate.classification),
                )
            )

    files: list[dict[str, Any]] = []
    total_raw = 0
    total_matched = 0
    total_parse_error_regions = 0
    total_uncovered = 0
    for file_rel_path in sorted(documents):
        document = documents[file_rel_path]
        lines = document.lines()
        spans = _scan_document_string_spans(file_rel_path, lines)
        matched = 0
        parse_error_regions: list[dict[str, Any]] = []
        uncovered: list[dict[str, Any]] = []
        for span in spans:
            overlaps = [
                classification
                for start, end, classification in candidate_spans.get(file_rel_path, {}).get(
                    span.line_index, []
                )
                if start < span.end_col and end > span.start_col
            ]
            if not overlaps:
                uncovered.append(_span_payload(span, document, include_excerpts=include_excerpts))
            elif all(classification == "parse_error" for classification in overlaps):
                parse_error_regions.append(
                    _span_payload(span, document, include_excerpts=include_excerpts)
                )
            else:
                matched += 1
        total_raw += len(spans)
        total_matched += matched
        total_parse_error_regions += len(parse_error_regions)
        total_uncovered += len(uncovered)
        files.append(
            {
                "file_rel_path": file_rel_path,
                "raw_spans": len(spans),
                "matched_spans": matched,
                "parse_error_region_spans": parse_error_regions,
                "uncovered_spans": uncovered,
            }
        )

    if total_uncovered:
        status = "uncovered_candidates"
    elif total_parse_error_regions:
        status = "parse_error_regions"
    else:
        status = "complete"
    return {
        "method": "independent_quote_run_scan_v1",
        "status": status,
        "raw_span_count": total_raw,
        "matched_span_count": total_matched,
        "parse_error_region_span_count": total_parse_error_regions,
        "uncovered_span_count": total_uncovered,
        "files": files,
        "notes": [
            "The independent scanner is a quote-run state machine, not the adapter tokenizer; "
            "unusual syntax can make it over- or under-count spans.",
            "A span covered only by parse_error candidates is reported separately because no "
            "extractable candidate exists for that string region.",
        ],
    }


def _coverage_payload(report: Any) -> dict[str, Any]:
    return {
        "coverage_status": report.coverage_status,
        "classification_counts": dict(report.classification_counts),
        "translation_scope_counts": dict(report.translation_scope_counts),
        "analysis_scope_counts": dict(report.analysis_scope_counts),
        "reason_counts": dict(report.reason_counts),
        "audit_reason_codes": list(report.audit_reason_codes),
        "invariant_errors": list(report.invariant_errors),
        "source_changed_during_scan": bool(report.source_changed_during_scan),
    }


def build_attribution_report(
    snapshot: Any,
    request: Any,
    *,
    decisions: ManualDecisions | None = None,
    generated_at: str | None = None,
    include_excerpts: bool = False,
) -> dict[str, Any]:
    """Build a deterministic attribution payload from one read-only snapshot."""

    report = snapshot.report
    rows = _candidate_rows(
        snapshot.inventory.candidates,
        include_excerpts=include_excerpts,
    )

    classification_counts = Counter(row.classification for row in rows)
    for classification, expected in report.classification_counts.items():
        if int(classification_counts.get(classification, 0)) != int(expected):
            raise AttributionError(
                "Attribution classification count does not match coverage report for "
                f"{classification!r}: {classification_counts.get(classification, 0)} != {expected}"
            )
    if len(rows) != int(report.candidate_count):
        raise AttributionError(
            f"Attribution rows ({len(rows)}) do not match coverage candidate_count "
            f"({report.candidate_count})."
        )

    candidate_reason_counts = Counter(code for row in rows for code in row.reason_codes)
    for code, count in candidate_reason_counts.items():
        reported = int(report.reason_counts.get(code, 0))
        if reported < count:
            raise AttributionError(
                f"Coverage reason count for {code!r} ({reported}) is smaller than the "
                f"attributed candidate count ({count})."
            )

    structure_kind_counts = Counter(row.structure_kind for row in rows)
    groups = _build_groups(rows, decisions)
    if decisions is not None:
        matched_keys = {str(item["group_key"]) for item in groups}
        missing = sorted(set(decisions.by_group) - matched_keys)
        if missing:
            raise AttributionError(
                "Manual decisions reference unknown group key(s): " + ", ".join(missing)
            )

    blocking_classification_counts = {
        classification: int(report.classification_counts.get(classification, 0))
        for classification in BLOCKING_CLASSIFICATIONS
        if int(report.classification_counts.get(classification, 0))
    }
    blocking_reason_counts = {
        code: int(count)
        for code, count in sorted(report.reason_counts.items())
        if code in BLOCKING_REASON_CODES and int(count)
    }

    manual_block = None
    if decisions is not None:
        manual_block = {
            "source_label": decisions.source_label,
            "reviewer": dict(decisions.reviewer),
            "group_count": len(decisions.by_group),
        }

    unresolved_groups = [
        group["group_key"]
        for group in groups
        if group["requires_human_review"] and group["human_judgment"] is None
    ]
    has_manual_judgment = bool(decisions is not None and decisions.by_group)
    evidence_gaps: list[str] = []
    scan = _scan_completeness(snapshot, include_excerpts=include_excerpts)
    if scan["uncovered_span_count"]:
        evidence_gaps.append(
            f"{scan['uncovered_span_count']} independently scanned string span(s) have no "
            "inventory candidate; possible missed candidates need manual confirmation."
        )
    if scan["parse_error_region_span_count"]:
        evidence_gaps.append(
            f"{scan['parse_error_region_span_count']} scanned string span(s) are covered only "
            "by parse_error candidates, so their text is not extractable yet."
        )
    if not has_manual_judgment:
        evidence_gaps.append(
            "No manual decisions were supplied; every category is automatic or heuristic "
            "evidence and has not been confirmed by a reviewer."
        )
    elif unresolved_groups:
        evidence_gaps.append(
            f"{len(unresolved_groups)} group(s) still have no manual judgment."
        )
    evidence_gaps.append(
        "Adapter inventory/coverage digests are included and, after #463, parse-error evidence is "
        "canonicalized so they are reproducible across processes for the same input/rules/version."
    )
    evidence_gaps.append(
        "This run does not assert real-project frequency.  Only run it on an authorized "
        "read-only copy, and publish fingerprints plus synthetic fixtures unless the project "
        "owner explicitly approves content release."
    )

    root_label = Path(snapshot.project.localization_root).name or "(localization root)"
    project_label = Path(snapshot.project.project_root).name if snapshot.project.project_root else ""
    payload: dict[str, Any] = {
        "attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "aggregation_digest_schema_version": AGGREGATION_DIGEST_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attribution_status": "manual_review_applied" if has_manual_judgment else "automatic_only",
        "inputs": {
            "engine": snapshot.project.engine,
            "adapter_version": snapshot.project.adapter_version,
            "target_language": snapshot.project.target_language,
            "project_root_label": project_label,
            "localization_root_label": root_label,
            "include_files": list(request.include_files),
            "include_prefixes": list(request.include_prefixes),
            "source_fingerprint": snapshot.project.source_fingerprint,
            "project_snapshot_fingerprint": snapshot.project.project_snapshot_fingerprint,
            "inventory_digest": report.inventory_digest,
            "coverage_digest": report.coverage_digest,
            "classification_rules_digest": report.classification_rules_digest,
        },
        "coverage": _coverage_payload(report),
        "blocking": {
            "coverage_status": report.coverage_status,
            "classification_counts": blocking_classification_counts,
            "reason_counts": blocking_reason_counts,
        },
        "candidate_count": len(rows),
        "candidate_reason_counts": dict(sorted(candidate_reason_counts.items())),
        "structure_kind_counts": dict(sorted(structure_kind_counts.items())),
        "groups": groups,
        "scan_completeness": scan,
        "unresolved_requires_human_review": sorted(unresolved_groups),
        "manual_decisions": manual_block,
        "evidence_gaps": evidence_gaps,
    }
    payload["aggregation_digest"] = _aggregation_digest(payload)
    return payload


def _md_cell(value: Any) -> str:
    text = str(value or "")
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", "")
        .replace("\n", "<br>")
    )


def render_markdown(
    payload: Mapping[str, Any],
    *,
    max_locators: int = 10,
    include_excerpts: bool = False,
) -> str:
    """Render the attribution payload for human review."""

    inputs = payload["inputs"]
    coverage = payload["coverage"]
    lines: list[str] = [
        "# Ren'Py coverage block attribution",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Attribution status: `{payload['attribution_status']}`",
        f"- Coverage status: `{coverage['coverage_status']}`",
        f"- Engine/adapter: `{inputs['engine']}` / `{inputs['adapter_version']}`",
        f"- Localization root label: `{_md_cell(inputs['localization_root_label'])}`",
        f"- Source fingerprint: `{inputs['source_fingerprint']}`",
        f"- Project snapshot fingerprint: `{inputs['project_snapshot_fingerprint']}`",
        f"- Classification rules digest: `{inputs['classification_rules_digest']}`",
        f"- Aggregation digest: `{payload['aggregation_digest']}`",
        f"- Candidates: {payload['candidate_count']}",
        "",
        "## Classification counts",
        "",
        "| classification | candidates |",
        "|---|---:|",
    ]
    for classification in sorted(coverage["classification_counts"]):
        lines.append(
            f"| `{_md_cell(classification)}` | {coverage['classification_counts'][classification]} |"
        )

    blocking = payload["blocking"]
    lines.extend(["", "## Blocking reasons", ""])
    if blocking["classification_counts"] or blocking["reason_counts"] or coverage["invariant_errors"]:
        lines.extend(["| kind | code | count |", "|---|---|---:|"])
        for classification, count in sorted(blocking["classification_counts"].items()):
            lines.append(f"| classification | `{_md_cell(classification)}` | {count} |")
        for code, count in sorted(blocking["reason_counts"].items()):
            lines.append(f"| reason | `{_md_cell(code)}` | {count} |")
        if coverage["invariant_errors"]:
            lines.append(
                "| invariant | `coverage.inventory.invalid_candidate` | "
                f"{len(coverage['invariant_errors'])} |"
            )
    else:
        lines.append("_No blocking classification or reason counts._")

    lines.extend(
        [
            "",
            "## Candidate groups",
            "",
            "| classification | structure_kind | reason_codes | count | automatic evidence | "
            "manual judgment | effective category | review flags |",
            "|---|---|---|---:|---|---|---|---|",
        ]
    )
    for group in payload["groups"]:
        judgment = group["human_judgment"]
        judgment_text = (
            f"{judgment['category']} ({judgment['reviewer'].get('type', '?')})"
            if judgment
            else "-"
        )
        lines.append(
            "| {classification} | {structure_kind} | {reasons} | {count} | {auto} | {human} | "
            "{effective} | {flags} |".format(
                classification=f"`{_md_cell(group['classification'])}`",
                structure_kind=f"`{_md_cell(group['structure_kind'])}`",
                reasons=", ".join(f"`{_md_cell(code)}`" for code in group["reason_codes"]),
                count=group["count"],
                auto=f"`{_md_cell(group['automatic_evidence']['category'])}` "
                f"({_md_cell(group['automatic_evidence']['evidence_class'])})",
                human=_md_cell(judgment_text),
                effective=f"`{_md_cell(group['effective_category'])}`",
                flags=", ".join(f"`{_md_cell(flag)}`" for flag in group["review_flags"]) or "-",
            )
        )

    for group in payload["groups"]:
        lines.extend(
            [
                "",
                f"## {group['classification']} | {group['structure_kind']} "
                f"({group['count']} candidate(s))",
                "",
                f"- Group key: `{group['group_key']}`",
                f"- Automatic evidence: `{group['automatic_evidence']['category']}` "
                f"({group['automatic_evidence']['evidence_class']})",
                "- Basis: " + "; ".join(_md_cell(item) for item in group["automatic_evidence"]["basis"]),
                f"- Effective category: `{group['effective_category']}`",
                f"- Requires human review: {group['requires_human_review']}",
                f"- Follow-up needed: {group['follow_up_needed']}",
            ]
        )
        if group["human_judgment"]:
            lines.extend(
                [
                    f"- Manual category: `{group['human_judgment']['category']}`",
                    f"- Manual rationale: {_md_cell(group['human_judgment']['rationale'])}",
                    "- Manual reviewer: "
                    f"`{_md_cell(group['human_judgment']['reviewer'].get('type', '?'))}` "
                    f"`{_md_cell(group['human_judgment']['reviewer'].get('id', ''))}`",
                ]
            )
        if group["review_flags"]:
            lines.append(
                "- Review flags: "
                + ", ".join(f"`{_md_cell(flag)}`" for flag in group["review_flags"])
            )
        lines.extend(["", "| locator |", "|---|"])
        for locator in group["locators"][: max(max_locators, 0)]:
            lines.append(
                "| `{path}:{line}:{start}-{end}` (`{candidate_id}`) |".format(
                    path=_md_cell(locator["file_rel_path"]),
                    line=locator["line_hint"],
                    start=locator["start_col_hint"],
                    end=locator["end_col_hint"],
                    candidate_id=_md_cell(locator["candidate_id"]),
                )
            )
        omitted = len(group["locators"]) - max(max_locators, 0)
        if omitted > 0:
            lines.append(f"| _{omitted} additional locator(s) omitted from Markdown._ |")
        if include_excerpts:
            excerpt_rows = [
                locator
                for locator in group["locators"][: max(max_locators, 0)]
                if locator.get("raw_excerpt")
            ]
            if excerpt_rows:
                lines.extend(
                    [
                        "",
                        "| locator | raw excerpt |",
                        "|---|---|",
                    ]
                )
                for locator in excerpt_rows:
                    lines.append(
                        "| `{path}:{line}` | {excerpt} |".format(
                            path=_md_cell(locator["file_rel_path"]),
                            line=locator["line_hint"],
                            excerpt=_md_cell(locator["raw_excerpt"]),
                        )
                    )

    scan = payload["scan_completeness"]
    lines.extend(
        [
            "",
            "## Independent scan completeness",
            "",
            f"- Method: `{scan['method']}`",
            f"- Status: `{scan['status']}`",
            f"- Raw quote spans: {scan['raw_span_count']}",
            f"- Matched by a non-parse-error candidate: {scan['matched_span_count']}",
            f"- Covered only by parse_error candidates: {scan['parse_error_region_span_count']}",
            f"- Without any candidate: {scan['uncovered_span_count']}",
            "",
            "| file | raw spans | matched | parse-error regions | uncovered |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for file_entry in scan["files"]:
        lines.append(
            "| {path} | {raw} | {matched} | {parse_error} | {uncovered} |".format(
                path=_md_cell(file_entry["file_rel_path"]),
                raw=file_entry["raw_spans"],
                matched=file_entry["matched_spans"],
                parse_error=len(file_entry["parse_error_region_spans"]),
                uncovered=len(file_entry["uncovered_spans"]),
            )
        )
    if scan["parse_error_region_span_count"]:
        lines.extend(
            [
                "",
                "### Parse-error string regions",
                "",
                "| file | line | columns | quote | continues to next line |",
                "|---|---:|---|---|---|",
            ]
        )
        for file_entry in scan["files"]:
            for span in file_entry["parse_error_region_spans"]:
                lines.append(
                    "| {path} | {line} | {start}-{end} | `{quote}` | {continues} |".format(
                        path=_md_cell(file_entry["file_rel_path"]),
                        line=span["line_hint"],
                        start=span["start_col_hint"],
                        end=span["end_col_hint"],
                        quote=_md_cell(span["quote"]),
                        continues=span["continues_to_next"],
                    )
                )
    if scan["uncovered_span_count"]:
        lines.extend(["", "### Uncovered spans", "", "| file | line | columns | quote |", "|---|---:|---|---|"])
        for file_entry in scan["files"]:
            for span in file_entry["uncovered_spans"]:
                lines.append(
                    "| {path} | {line} | {start}-{end} | `{quote}` |".format(
                        path=_md_cell(file_entry["file_rel_path"]),
                        line=span["line_hint"],
                        start=span["start_col_hint"],
                        end=span["end_col_hint"],
                        quote=_md_cell(span["quote"]),
                    )
                )

    lines.extend(["", "## Evidence gaps and unknowns", ""])
    for gap in payload["evidence_gaps"]:
        lines.append(f"- {_md_cell(gap)}")
    lines.append("")
    return "\n".join(lines)


def load_decisions(path: str | os.PathLike[str]) -> ManualDecisions:
    """Load and validate manual reviewer decisions."""

    raw_path = os.fspath(path)
    if not raw_path or not os.path.isfile(raw_path):
        raise AttributionError(f"decisions file not found: {raw_path or '<empty path>'}")
    try:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AttributionError(f"could not load decisions file: {raw_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AttributionError("decisions file must contain a JSON object")
    if int(payload.get("schema_version") or 0) != DECISIONS_SCHEMA_VERSION:
        raise AttributionError(
            f"unsupported decisions schema_version: {payload.get('schema_version')!r}"
        )
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, Mapping):
        raise AttributionError("decisions.reviewer must be an object")
    reviewer_type = str(reviewer.get("type") or "")
    reviewer_id = str(reviewer.get("id") or "").strip()
    if reviewer_type not in REVIEWER_TYPES:
        raise AttributionError(f"decisions.reviewer.type must be one of {sorted(REVIEWER_TYPES)}")
    if not reviewer_id:
        raise AttributionError("decisions.reviewer.id must be non-empty")

    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise AttributionError("decisions.decisions must be an array")
    by_group: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(raw_decisions):
        if not isinstance(item, Mapping):
            raise AttributionError(f"decisions[{index}] must be an object")
        key = str(item.get("group_key") or "").strip()
        category = str(item.get("category") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        if not key:
            raise AttributionError(f"decisions[{index}].group_key must be non-empty")
        if category not in MANUAL_CATEGORIES:
            raise AttributionError(
                f"decisions[{index}].category must be one of {sorted(MANUAL_CATEGORIES)}"
            )
        if not rationale:
            raise AttributionError(f"decisions[{index}].rationale must be non-empty")
        if key in by_group:
            raise AttributionError(f"duplicate manual decision for group key: {key}")
        by_group[key] = {
            "category": category,
            "rationale": rationale,
        }
    return ManualDecisions(
        reviewer={"type": reviewer_type, "id": reviewer_id},
        by_group=by_group,
        source_label=Path(raw_path).name,
    )


def run_read_only_scan(request: Any) -> Any:
    """Run the existing Ren'Py discovery/inventory/coverage pipeline once."""

    from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot

    snapshot = build_translation_snapshot(
        RenPyAdapter(),
        request,
        include_occurrences=False,
        include_task_payloads=False,
    )
    if not snapshot.project.source_documents:
        raise AttributionError(
            "No .rpy source documents were found under the localization root; "
            "check --localization-root and include filters."
        )
    return snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "localization_root",
        help="Read-only Ren'Py localization root to inventory (for example game/tl/schinese)",
    )
    parser.add_argument(
        "--project-root",
        default="",
        help="Project root used only for discovery metadata; defaults to the parent of the "
        "localization root",
    )
    parser.add_argument(
        "--target-language",
        default="",
        help="Target language metadata stored in the discovery request",
    )
    parser.add_argument(
        "--include-file",
        action="append",
        default=[],
        help="Restrict discovery to this localization-relative file; may be repeated",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Restrict discovery to this localization-relative prefix; may be repeated",
    )
    parser.add_argument(
        "--decisions",
        default="",
        help="Optional validated manual decisions JSON; unmatched group keys fail closed",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Write the attribution payload as JSON to this path",
    )
    parser.add_argument(
        "--markdown-output",
        default="",
        help="Write the Markdown report to this path instead of stdout",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Pin the generated timestamp for byte-identical reruns",
    )
    parser.add_argument(
        "--max-locators",
        type=int,
        default=10,
        help="Maximum locator rows per group in Markdown (JSON keeps every locator)",
    )
    parser.add_argument(
        "--include-excerpts",
        action="store_true",
        help="Include bounded raw excerpts.  Do not publish private-project output with excerpts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_locators < 0:
        parser.error("--max-locators must be >= 0")
    try:
        from engine_adapters.contracts import ProjectDiscoveryRequest

        decisions = load_decisions(args.decisions) if args.decisions else None
        request = ProjectDiscoveryRequest(
            project_root=str(args.project_root or ""),
            localization_root=str(args.localization_root),
            target_language=str(args.target_language or ""),
            include_files=tuple(str(value) for value in args.include_file),
            include_prefixes=tuple(str(value) for value in args.include_prefix),
        )
        snapshot = run_read_only_scan(request)
        payload = build_attribution_report(
            snapshot,
            request,
            decisions=decisions,
            generated_at=args.generated_at,
            include_excerpts=bool(args.include_excerpts),
        )
        markdown = render_markdown(
            payload,
            max_locators=args.max_locators,
            include_excerpts=bool(args.include_excerpts),
        )
        if args.json_output:
            output_path = Path(args.json_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        if args.markdown_output:
            output_path = Path(args.markdown_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8", newline="\n")
            print(f"Coverage attribution report written to: {output_path}")
        else:
            sys.stdout.write(markdown)
        return 0
    except (AttributionError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
