"""Rebuildable review index and human decisions for ordinary translation entries.

This is the core of #427: a derived, read-only index over the revision corpus
plus existing quality findings, with a separate decision log for human review.
The index is never a second translation store: deleting it and rebuilding from
the same corpus / findings / decisions must produce the same entries.  No game
file, manifest, glossary, quality acknowledgement, or writeback authorization
is touched here.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from atomic_io import atomic_write_json, atomic_write_jsonl, atomic_write_text

import revision_corpus
import translation_quality

REVIEW_INDEX_SCHEMA_VERSION = 1
REVIEW_ENTRY_SCHEMA_VERSION = 1
REVIEW_DECISION_SCHEMA_VERSION = 1

REVIEW_INDEX_JSONL_NAME = "review_index.jsonl"
REVIEW_INDEX_MANIFEST_NAME = "review_index_manifest.json"
REVIEW_INDEX_MARKDOWN_NAME = "review_index.md"
REVIEW_DECISIONS_JSONL_NAME = "review_decisions.jsonl"
REVIEW_DECISIONS_TEMPLATE_NAME = "review_decisions_template.jsonl"

LIFECYCLE_OPEN = "open"
LIFECYCLE_IGNORED = "ignored"
LIFECYCLE_RESOLVED = "resolved"
LIFECYCLE_NEEDS_RECHECK = "needs_recheck"
LIFECYCLE_IMPORTABLE = frozenset(
    {LIFECYCLE_OPEN, LIFECYCLE_IGNORED, LIFECYCLE_RESOLVED}
)
LIFECYCLE_ALL = frozenset(
    {
        LIFECYCLE_OPEN,
        LIFECYCLE_IGNORED,
        LIFECYCLE_RESOLVED,
        LIFECYCLE_NEEDS_RECHECK,
    }
)

REVIEWER_TYPES = frozenset({"human", "agent"})
BINDING_KEYS = (
    "entry_id",
    "snapshot_digest",
    "source_digest",
    "target_digest",
    "context_digest",
    "evidence_digest",
)
MAX_NOTE_LENGTH = 4000

DIAGNOSTIC_FINDING_UNMATCHED = "REVIEW_QUALITY_FINDING_UNMATCHED"
DIAGNOSTIC_DECISION_ORPHANED = "REVIEW_DECISION_ORPHANED"
DIAGNOSTIC_DECISION_PROJECT_MISMATCH = "REVIEW_DECISION_PROJECT_MISMATCH"
DIAGNOSTIC_DECISION_AMBIGUOUS_FINDING = "REVIEW_QUALITY_FINDING_AMBIGUOUS"


class ReviewIndexError(ValueError):
    """Stable local failure for review index / decisions input."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_payload(value: Any) -> str:
    return stable_text_sha256(_stable_json(value))


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_rel_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _file_digest(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewIndexError(
            "REVIEW_INDEX_INPUT_INVALID",
            f"无法读取 JSON：{path}",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(value, dict):
        raise ReviewIndexError(
            "REVIEW_INDEX_INPUT_INVALID",
            f"JSON 顶层不是对象：{path}",
            details={"path": str(path)},
        )
    return value


def load_jsonl(path: str | os.PathLike[str], *, label: str = "JSONL") -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise ReviewIndexError(
            "REVIEW_INDEX_INPUT_MISSING",
            f"{label} 文件不存在：{source}",
            details={"path": str(source)},
        )
    rows: list[dict[str, Any]] = []
    try:
        lines = source.read_text(encoding="utf-8-sig").split("\n")
    except (OSError, UnicodeError) as exc:
        raise ReviewIndexError(
            "REVIEW_INDEX_INPUT_INVALID",
            f"无法读取 {label}：{source}",
            details={"path": str(source), "error": str(exc)},
        ) from exc
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ReviewIndexError(
                "REVIEW_INDEX_INPUT_INVALID",
                f"{label} 第 {line_number} 行不是有效 JSON。",
                details={"path": str(source), "line": line_number},
            ) from exc
        if not isinstance(value, dict):
            raise ReviewIndexError(
                "REVIEW_INDEX_INPUT_INVALID",
                f"{label} 第 {line_number} 行不是对象。",
                details={"path": str(source), "line": line_number},
            )
        rows.append(value)
    return rows


def load_corpus_bundle(corpus_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Resolve a corpus directory / manifest / JSONL into rows + manifest."""

    supplied = Path(corpus_path)
    manifest_path: Path | None = None
    jsonl_path: Path | None = None
    if supplied.is_dir():
        manifest_path = supplied / revision_corpus.CORPUS_MANIFEST_NAME
        jsonl_path = supplied / revision_corpus.CORPUS_JSONL_NAME
    elif supplied.suffix.lower() == ".jsonl":
        jsonl_path = supplied
        candidate = supplied.parent / revision_corpus.CORPUS_MANIFEST_NAME
        manifest_path = candidate if candidate.is_file() else None
    elif supplied.is_file():
        manifest_path = supplied
    else:
        raise ReviewIndexError(
            "REVIEW_INDEX_INPUT_MISSING",
            f"corpus 输入不存在：{supplied}",
            details={"path": str(supplied)},
        )

    manifest: dict[str, Any] = {}
    if manifest_path is not None and manifest_path.is_file():
        manifest = _read_json_object(manifest_path)
    if jsonl_path is None:
        relative = (
            (manifest.get("paths") or {}).get("jsonl")
            if isinstance(manifest.get("paths"), Mapping)
            else ""
        )
        if relative:
            candidate = Path(str(relative))
            jsonl_path = (
                candidate
                if candidate.is_absolute()
                else (manifest_path.parent / candidate if manifest_path else candidate)
            )
        elif manifest_path is not None:
            candidate = manifest_path.parent / revision_corpus.CORPUS_JSONL_NAME
            jsonl_path = candidate if candidate.is_file() else None
    manifest_path = manifest_path.resolve() if manifest_path else None
    jsonl_path = jsonl_path.resolve() if jsonl_path else None
    if jsonl_path is None or not jsonl_path.is_file():
        raise ReviewIndexError(
            "REVIEW_INDEX_INPUT_MISSING",
            "corpus 缺少 revision_corpus.jsonl。",
            details={"path": str(supplied)},
        )
    try:
        rows = revision_corpus.load_corpus_items(str(jsonl_path))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ReviewIndexError(
            "REVIEW_INDEX_INPUT_INVALID",
            f"corpus JSONL 无法解析：{jsonl_path}",
            details={"path": str(jsonl_path), "error": str(exc)},
        ) from exc
    if not isinstance(manifest, dict) or not manifest:
        raise ReviewIndexError(
            "REVIEW_INDEX_INPUT_MISSING",
            f"corpus JSONL 缺少同目录 revision_corpus_manifest.json：{jsonl_path}",
            details={"jsonl": str(jsonl_path)},
        )
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path) if manifest_path else "",
        "manifest_digest": _file_digest(manifest_path) if manifest_path else "",
        "jsonl_path": str(jsonl_path),
        "jsonl_digest": _file_digest(jsonl_path),
        "rows": rows,
    }


def project_identity(corpus_manifest: Mapping[str, Any]) -> dict[str, Any]:
    project = corpus_manifest.get("project")
    project = project if isinstance(project, Mapping) else {}
    slug = str(project.get("slug") or "").strip() or "unknown"
    tl_subdir = str(project.get("tl_subdir") or "").strip()
    # Identity is deliberately path-portable: moving or copying a project must
    # keep human decisions recoverable.  Cross-project isolation relies on the
    # slug + target subdir plus occurrence/binding digests, not machine paths.
    identity_digest = stable_text_sha256(f"project\0{slug}\0{tl_subdir}")
    return {
        "slug": slug,
        "tl_subdir": tl_subdir,
        "identity_digest": identity_digest,
    }


def _finding_match_keys(finding: Mapping[str, Any]) -> tuple[str, int]:
    return _normalize_rel_path(finding.get("file")), _coerce_int(finding.get("line"))


def _finding_summary(finding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "finding_id": str(finding.get("finding_id") or ""),
        "schema_version": _coerce_int(finding.get("schema_version"), 1),
        "reason_code": str(finding.get("reason_code") or ""),
        "rule_id": str(finding.get("rule_id") or ""),
        "severity": str(finding.get("severity") or ""),
        "disposition": str(finding.get("disposition") or ""),
        "item_id": str(finding.get("item_id") or ""),
        "file": _normalize_rel_path(finding.get("file")),
        "line": _coerce_int(finding.get("line")),
        "evidence": str(finding.get("evidence") or ""),
        "suggestion": str(finding.get("suggestion") or ""),
        "rule_version": str(finding.get("rule_version") or ""),
    }


def load_quality_findings(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not path or not str(path).strip():
        return {"findings": [], "path": "", "digest": "", "count": 0}
    source = Path(path).resolve()
    rows = load_jsonl(source, label="quality findings")
    findings: list[dict[str, Any]] = []
    for row in rows:
        try:
            findings.append(translation_quality.normalize_finding(row))
        except ValueError as exc:
            raise ReviewIndexError(
                "REVIEW_INDEX_INPUT_INVALID",
                f"quality finding 无效：{exc}",
                details={"path": str(source)},
            ) from exc
    return {
        "findings": findings,
        "path": str(source),
        "digest": _file_digest(source),
        "count": len(findings),
    }


def load_translation_records(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not path or not str(path).strip():
        return {"records": [], "path": "", "digest": "", "count": 0}
    source = Path(path).resolve()
    rows = load_jsonl(source, label="translation records")
    records: list[dict[str, Any]] = []
    for row in rows:
        occurrence_id = str(row.get("occurrence_id") or "").strip()
        if not occurrence_id:
            continue
        records.append(
            {
                "occurrence_id": occurrence_id,
                "record_id": str(row.get("record_id") or ""),
                "record_digest": str(row.get("record_digest") or ""),
                "origin": str(row.get("origin") or ""),
                "status": str(row.get("status") or ""),
                "translation_text": str(row.get("translation_text") or ""),
                "target_language": str(row.get("target_language") or ""),
                "revision_history_count": len(
                    row.get("revision_history") or []
                ),
            }
        )
    return {
        "records": records,
        "path": str(source),
        "digest": _file_digest(source),
        "count": len(records),
    }


def _context_digest(context: Any) -> str:
    return _digest_payload(context if isinstance(context, Mapping) else {})


def _entry_binding(
    *,
    project: Mapping[str, Any],
    occurrence_id: str,
    file_rel_path: str,
    line: int,
    source: str,
    current_translation: str,
    context: Mapping[str, Any],
    snapshot_digest: str,
    findings: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    source_digest = stable_text_sha256(source)
    target_digest = stable_text_sha256(current_translation)
    context_digest = _context_digest(context)
    evidence_payload = {
        "source_digest": source_digest,
        "target_digest": target_digest,
        "context_digest": context_digest,
        "finding_ids": sorted(
            str(item.get("finding_id") or "") for item in findings
        ),
    }
    evidence_digest = _digest_payload(evidence_payload)
    entry_id = _digest_payload(
        {
            "project_id": str(project.get("identity_digest") or ""),
            "occurrence_id": occurrence_id,
            "file_rel_path": file_rel_path,
            "line": line,
            "source_digest": source_digest,
            "target_digest": target_digest,
            "context_digest": context_digest,
            "evidence_digest": evidence_digest,
        }
    )[:24]
    return {
        "entry_id": entry_id,
        "snapshot_digest": str(snapshot_digest or ""),
        "source_digest": source_digest,
        "target_digest": target_digest,
        "context_digest": context_digest,
        "evidence_digest": evidence_digest,
    }


def build_index_entries(
    corpus_rows: Sequence[Mapping[str, Any]],
    corpus_manifest: Mapping[str, Any],
    *,
    findings: Sequence[Mapping[str, Any]] = (),
    records: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build derived entries; findings/records only attach by stable identity."""

    project = project_identity(corpus_manifest)
    if (
        not project.get("slug")
        or project.get("slug") == "unknown"
        or not project.get("tl_subdir")
    ):
        raise ReviewIndexError(
            "REVIEW_INDEX_INPUT_INVALID",
            "corpus manifest 缺少 project.slug / project.tl_subdir，"
            "无法建立可隔离的项目身份。",
            details={"project": dict(project)},
        )
    findings_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    findings_by_item: dict[str, list[dict[str, Any]]] = {}
    for finding in findings:
        summary = _finding_summary(finding)
        findings_by_key.setdefault(_finding_match_keys(summary), []).append(summary)
        item_id = str(summary.get("item_id") or "")
        if item_id:
            findings_by_item.setdefault(item_id, []).append(summary)
    records_by_occurrence: dict[str, dict[str, Any]] = {}
    for record in records:
        occurrence_id = str(record.get("occurrence_id") or "")
        if occurrence_id:
            records_by_occurrence.setdefault(occurrence_id, dict(record))

    diagnostics: list[dict[str, Any]] = []
    matched_finding_ids: set[str] = set()
    entries: list[dict[str, Any]] = []
    for row in corpus_rows:
        if not isinstance(row, Mapping):
            continue
        occurrence_id = str(
            row.get("occurrence_id") or row.get("identity_v2") or ""
        ).strip()
        if not occurrence_id:
            diagnostics.append(
                {
                    "code": "REVIEW_CORPUS_ROW_MISSING_IDENTITY",
                    "file_rel_path": _normalize_rel_path(row.get("file_rel_path")),
                }
            )
            continue
        file_rel_path = _normalize_rel_path(row.get("file_rel_path"))
        locator = row.get("locator") if isinstance(row.get("locator"), Mapping) else {}
        line = _coerce_int(locator.get("line"))
        source = str(row.get("source") or "")
        current_translation = str(row.get("current_translation") or "")
        context = row.get("context") if isinstance(row.get("context"), Mapping) else {}
        matched = list(findings_by_key.get((file_rel_path, line), ()))
        if not matched:
            matched = list(findings_by_item.get(occurrence_id, ()))
        if not matched:
            matched = list(findings_by_item.get(str(row.get("identity_v2") or ""), ()))
        unique_matched: dict[str, dict[str, Any]] = {}
        for finding in matched:
            finding_id = str(finding.get("finding_id") or "")
            if finding_id:
                unique_matched.setdefault(finding_id, finding)
        matched = list(unique_matched.values())
        matched_finding_ids.update(unique_matched)
        if len(matched) > 1 and len(
            {str(item.get("item_id") or "") for item in matched}
        ) > 1:
            diagnostics.append(
                {
                    "code": DIAGNOSTIC_DECISION_AMBIGUOUS_FINDING,
                    "occurrence_id": occurrence_id,
                    "file_rel_path": file_rel_path,
                    "line": line,
                    "finding_ids": sorted(unique_matched),
                }
            )
        matched.sort(key=lambda item: str(item.get("finding_id") or ""))
        snapshot_digest = str(
            row.get("snapshot_digest")
            or revision_corpus.item_snapshot_digest(source, current_translation)
        )
        binding = _entry_binding(
            project=project,
            occurrence_id=occurrence_id,
            file_rel_path=file_rel_path,
            line=line,
            source=source,
            current_translation=current_translation,
            context=context,
            snapshot_digest=snapshot_digest,
            findings=matched,
        )
        record = records_by_occurrence.get(occurrence_id)
        entries.append(
            {
                "schema_version": REVIEW_ENTRY_SCHEMA_VERSION,
                "entry_id": binding["entry_id"],
                "project": dict(project),
                "occurrence_id": occurrence_id,
                "identity_v2": str(row.get("identity_v2") or occurrence_id),
                "file_rel_path": file_rel_path,
                "locator": {
                    "line": line,
                    "line_number": _coerce_int(locator.get("line_number")),
                    "start": _coerce_int(locator.get("start")),
                    "end": _coerce_int(locator.get("end")),
                    "ordinal": _coerce_int(locator.get("ordinal")),
                },
                "display_line": _coerce_int(
                    row.get("display_line"), _coerce_int(locator.get("line_number"))
                ),
                "speaker_id": str(row.get("speaker_id") or ""),
                "source": source,
                "current_translation": current_translation,
                "context": dict(context),
                "snapshot_digest": snapshot_digest,
                "binding": binding,
                "quality_findings": matched,
                "quality_finding_ids": sorted(
                    str(item.get("finding_id") or "") for item in matched
                ),
                "issue_count": len(matched),
                "has_issues": bool(matched),
                "translation_record": record,
                "review": {
                    "lifecycle": LIFECYCLE_OPEN,
                    "decision_id": "",
                    "reviewer": None,
                    "note": "",
                    "decided_at": "",
                    "needs_recheck": False,
                    "changed_bindings": [],
                    "history": [],
                },
            }
        )
    for finding in findings:
        finding_id = str(finding.get("finding_id") or "")
        if finding_id and finding_id not in matched_finding_ids:
            diagnostics.append(
                {
                    "code": DIAGNOSTIC_FINDING_UNMATCHED,
                    "finding_id": finding_id,
                    "reason_code": str(finding.get("reason_code") or ""),
                    "file": _normalize_rel_path(finding.get("file")),
                    "line": _coerce_int(finding.get("line")),
                    "item_id": str(finding.get("item_id") or ""),
                }
            )
    entries.sort(
        key=lambda entry: (
            str(entry.get("file_rel_path") or ""),
            _coerce_int((entry.get("locator") or {}).get("line")),
            str(entry.get("occurrence_id") or ""),
        )
    )
    return entries, diagnostics


def normalize_decision(
    raw: Mapping[str, Any],
    *,
    now: str = "",
) -> dict[str, Any]:
    """Validate one importable decision row; binding values are kept verbatim."""

    if not isinstance(raw, Mapping):
        raise ReviewIndexError("REVIEW_DECISION_INVALID", "decision 必须是对象。")
    raw_version = raw.get("schema_version")
    if raw_version not in (None, ""):
        try:
            version = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise ReviewIndexError(
                "REVIEW_DECISION_INVALID",
                "decision.schema_version 必须是整数。",
            ) from exc
        if version != REVIEW_DECISION_SCHEMA_VERSION:
            raise ReviewIndexError(
                "REVIEW_DECISION_INVALID",
                f"不支持的 decision schema_version：{version}",
                details={"schema_version": version},
            )
    occurrence_id = str(raw.get("occurrence_id") or "").strip()
    if not occurrence_id:
        raise ReviewIndexError(
            "REVIEW_DECISION_INVALID",
            "decision.occurrence_id 不能为空。",
        )
    lifecycle = str(raw.get("lifecycle") or "").strip().lower()
    if lifecycle not in LIFECYCLE_IMPORTABLE:
        raise ReviewIndexError(
            "REVIEW_DECISION_INVALID",
            f"decision.lifecycle 必须是 {sorted(LIFECYCLE_IMPORTABLE)} 之一。",
            details={"occurrence_id": occurrence_id, "lifecycle": lifecycle},
        )
    reviewer = raw.get("reviewer")
    if not isinstance(reviewer, Mapping):
        raise ReviewIndexError(
            "REVIEW_DECISION_INVALID",
            "decision.reviewer 必须是对象。",
            details={"occurrence_id": occurrence_id},
        )
    reviewer_type = str(reviewer.get("type") or "").strip().lower()
    reviewer_name = str(reviewer.get("name") or "").strip()
    if reviewer_type not in REVIEWER_TYPES:
        raise ReviewIndexError(
            "REVIEW_DECISION_INVALID",
            f"decision.reviewer.type 必须是 {sorted(REVIEWER_TYPES)} 之一。",
            details={"occurrence_id": occurrence_id},
        )
    if not reviewer_name:
        raise ReviewIndexError(
            "REVIEW_DECISION_INVALID",
            "decision.reviewer.name 不能为空。",
            details={"occurrence_id": occurrence_id},
        )
    if reviewer_name.upper() == "TODO":
        raise ReviewIndexError(
            "REVIEW_DECISION_INVALID",
            "decision.reviewer.name 仍是模板占位值 TODO，请填写实际 reviewer。",
            details={"occurrence_id": occurrence_id},
        )
    binding = raw.get("binding")
    if not isinstance(binding, Mapping):
        raise ReviewIndexError(
            "REVIEW_DECISION_INVALID",
            "decision.binding 必须是对象。",
            details={"occurrence_id": occurrence_id},
        )
    normalized_binding: dict[str, str] = {}
    for key in BINDING_KEYS:
        value = str(binding.get(key) or "").strip()
        if not value:
            raise ReviewIndexError(
                "REVIEW_DECISION_INVALID",
                f"decision.binding.{key} 不能为空。",
                details={"occurrence_id": occurrence_id},
            )
        normalized_binding[key] = value
    note = str(raw.get("note") or "")
    if len(note) > MAX_NOTE_LENGTH:
        raise ReviewIndexError(
            "REVIEW_DECISION_INVALID",
            f"decision.note 超过 {MAX_NOTE_LENGTH} 字符上限。",
            details={"occurrence_id": occurrence_id},
        )
    decided_at = str(raw.get("decided_at") or "").strip() or now or _utc_now()
    project_identity_digest = str(
        raw.get("project_identity_digest") or ""
    ).strip()
    if not project_identity_digest:
        raise ReviewIndexError(
            "REVIEW_DECISION_INVALID",
            "decision.project_identity_digest 不能为空。",
            details={"occurrence_id": occurrence_id},
        )
    identity_payload = {
        "occurrence_id": occurrence_id,
        "project_identity_digest": project_identity_digest,
        "lifecycle": lifecycle,
        "reviewer": {
            "type": reviewer_type,
            "name": reviewer_name,
            "run_id": str(reviewer.get("run_id") or ""),
        },
        "binding": normalized_binding,
        "note": note,
    }
    computed_id = _digest_payload(identity_payload)[:24]
    provided_id = str(raw.get("decision_id") or "").strip()
    if provided_id and provided_id != computed_id:
        raise ReviewIndexError(
            "REVIEW_DECISION_INVALID",
            "decision_id 与决定内容不一致；修改已有决定请去掉 decision_id 或追加新动作。",
            details={"occurrence_id": occurrence_id},
        )
    decision_id = provided_id or computed_id
    record = {
        "schema_version": REVIEW_DECISION_SCHEMA_VERSION,
        "decision_id": decision_id,
        "occurrence_id": occurrence_id,
        "project_identity_digest": project_identity_digest,
        "lifecycle": lifecycle,
        "reviewer": identity_payload["reviewer"],
        "binding": normalized_binding,
        "note": note,
        "decided_at": decided_at,
        "supersedes": str(raw.get("supersedes") or ""),
    }
    return record


def load_decisions(path: str | os.PathLike[str] | None) -> list[dict[str, Any]]:
    if not path or not str(path).strip():
        return []
    source = Path(path)
    if not source.is_file():
        return []
    rows = load_jsonl(source, label="review decisions")
    decisions: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        try:
            decisions.append(normalize_decision(row))
        except ReviewIndexError as exc:
            raise ReviewIndexError(
                exc.code,
                f"review decisions 第 {index} 行无效：{exc}",
                details={"path": str(source), "line": index, **exc.details},
            ) from exc
    return decisions


def merge_decisions(
    existing: Sequence[Mapping[str, Any]],
    incoming: Sequence[Mapping[str, Any]],
    *,
    known_occurrence_ids: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Append new decision actions while preserving existing file order/history."""

    known = set(known_occurrence_ids)
    merged: list[dict[str, Any]] = [dict(item) for item in existing]
    # Stable chronological append: explicit timestamps first, original order
    # for template rows without a timestamp.
    ordered_incoming = [
        normalize_decision(row)
        for _index, row in sorted(
            enumerate(incoming),
            key=lambda pair: (
                str(pair[1].get("decided_at") or ""),
                pair[0],
            ),
        )
    ]
    existing_ids_by_occurrence: dict[str, list[str]] = {}
    latest_existing_by_occurrence: dict[str, dict[str, Any]] = {}
    for item in merged:
        occurrence_id = str(item.get("occurrence_id") or "")
        if occurrence_id:
            existing_ids_by_occurrence.setdefault(occurrence_id, []).append(
                str(item.get("decision_id") or "")
            )
            latest_existing_by_occurrence[occurrence_id] = dict(item)
    incoming_ids_by_occurrence: dict[str, list[str]] = {}
    for decision in ordered_incoming:
        occurrence_id = str(decision.get("occurrence_id") or "")
        if occurrence_id:
            incoming_ids_by_occurrence.setdefault(occurrence_id, []).append(
                str(decision.get("decision_id") or "")
            )
    # A repeated multi-action import (for example ignored -> resolved) must be
    # idempotent: skip the longest incoming prefix that already matches the
    # tail of the existing per-occurrence log, then append only the new suffix.
    overlap_by_occurrence: dict[str, int] = {}
    for occurrence_id, incoming_ids in incoming_ids_by_occurrence.items():
        existing_ids = existing_ids_by_occurrence.get(occurrence_id, [])
        overlap = 0
        for size in range(min(len(incoming_ids), len(existing_ids)), 0, -1):
            if incoming_ids[:size] == existing_ids[-size:]:
                overlap = size
                break
        overlap_by_occurrence[occurrence_id] = overlap
    duplicate_count = 0
    orphaned_count = 0
    stale_count = 0
    incoming_index_by_occurrence: dict[str, int] = {}
    current_ids_by_occurrence = {
        occurrence_id: list(existing_ids)
        for occurrence_id, existing_ids in existing_ids_by_occurrence.items()
    }
    for decision in ordered_incoming:
        occurrence_id = str(decision.get("occurrence_id") or "")
        decision_id = str(decision.get("decision_id") or "")
        decision_index = incoming_index_by_occurrence.get(occurrence_id, 0)
        incoming_index_by_occurrence[occurrence_id] = decision_index + 1
        if decision_index < overlap_by_occurrence.get(occurrence_id, 0):
            duplicate_count += 1
            continue
        current_ids = current_ids_by_occurrence.setdefault(occurrence_id, [])
        if current_ids and decision_id == current_ids[-1]:
            duplicate_count += 1
            continue
        latest_existing = latest_existing_by_occurrence.get(occurrence_id)
        incoming_decided_at = str(decision.get("decided_at") or "")
        if latest_existing is not None:
            latest_decided_at = str(latest_existing.get("decided_at") or "")
            if (
                incoming_decided_at
                and latest_decided_at
                and incoming_decided_at < latest_decided_at
            ):
                # Re-importing an older exported file must not silently revert
                # a newer decision; it is a stale replay, not a new action.
                stale_count += 1
                continue
        if known and occurrence_id not in known:
            orphaned_count += 1
        merged.append(decision)
        current_ids.append(decision_id)
        latest_existing_by_occurrence[occurrence_id] = decision
    return merged, {
        "existing_count": len(existing),
        "imported_count": len(incoming),
        "duplicate_count": duplicate_count,
        "stale_count": stale_count,
        "orphaned_count": orphaned_count,
        "total_count": len(merged),
    }


def _decision_summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": str(decision.get("decision_id") or ""),
        "project_identity_digest": str(
            decision.get("project_identity_digest") or ""
        ),
        "lifecycle": str(decision.get("lifecycle") or ""),
        "reviewer": dict(decision.get("reviewer") or {}),
        "note": str(decision.get("note") or ""),
        "decided_at": str(decision.get("decided_at") or ""),
        "binding": dict(decision.get("binding") or {}),
    }


def binding_diff(decision: Mapping[str, Any], entry: Mapping[str, Any]) -> list[str]:
    binding = decision.get("binding")
    binding = binding if isinstance(binding, Mapping) else {}
    current = entry.get("binding")
    current = current if isinstance(current, Mapping) else {}
    return [
        key
        for key in BINDING_KEYS
        if str(binding.get(key) or "") != str(current.get(key) or "")
    ]


def apply_decisions(
    entries: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply latest decision per occurrence; binding changes derive needs_recheck."""

    by_occurrence: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        occurrence_id = str(decision.get("occurrence_id") or "")
        if occurrence_id:
            by_occurrence.setdefault(occurrence_id, []).append(dict(decision))
    diagnostics: list[dict[str, Any]] = []
    known_ids = {str(entry.get("occurrence_id") or "") for entry in entries}
    for occurrence_id, history in by_occurrence.items():
        if occurrence_id not in known_ids:
            diagnostics.append(
                {
                    "code": DIAGNOSTIC_DECISION_ORPHANED,
                    "occurrence_id": occurrence_id,
                    "decision_ids": [
                        str(item.get("decision_id") or "") for item in history
                    ],
                }
            )
    applied: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        occurrence_id = str(item.get("occurrence_id") or "")
        entry_project = str(
            (item.get("project") or {}).get("identity_digest") or ""
        )
        history: list[dict[str, Any]] = []
        for row in by_occurrence.get(occurrence_id, []):
            row_project = str(row.get("project_identity_digest") or "")
            if row_project != entry_project:
                diagnostics.append(
                    {
                        "code": DIAGNOSTIC_DECISION_PROJECT_MISMATCH,
                        "occurrence_id": occurrence_id,
                        "decision_id": str(row.get("decision_id") or ""),
                        "decision_project_identity_digest": row_project,
                        "entry_project_identity_digest": entry_project,
                    }
                )
                continue
            history.append(row)
        review = dict(item.get("review") or {})
        review["history"] = [_decision_summary(row) for row in history]
        if not history:
            review.update(
                {
                    "lifecycle": LIFECYCLE_OPEN,
                    "decision_id": "",
                    "reviewer": None,
                    "note": "",
                    "decided_at": "",
                    "needs_recheck": False,
                    "changed_bindings": [],
                    "previous_lifecycle": "",
                }
            )
        else:
            latest = history[-1]
            changed = binding_diff(latest, item)
            if changed:
                review.update(
                    {
                        "lifecycle": LIFECYCLE_NEEDS_RECHECK,
                        "decision_id": str(latest.get("decision_id") or ""),
                        "reviewer": dict(latest.get("reviewer") or {}),
                        "note": str(latest.get("note") or ""),
                        "decided_at": str(latest.get("decided_at") or ""),
                        "needs_recheck": True,
                        "changed_bindings": changed,
                        "previous_lifecycle": str(
                            latest.get("lifecycle") or LIFECYCLE_OPEN
                        ),
                    }
                )
            else:
                review.update(
                    {
                        "lifecycle": str(
                            latest.get("lifecycle") or LIFECYCLE_OPEN
                        ),
                        "decision_id": str(latest.get("decision_id") or ""),
                        "reviewer": dict(latest.get("reviewer") or {}),
                        "note": str(latest.get("note") or ""),
                        "decided_at": str(latest.get("decided_at") or ""),
                        "needs_recheck": False,
                        "changed_bindings": [],
                        "previous_lifecycle": "",
                    }
                )
        item["review"] = review
        applied.append(item)
    return applied, diagnostics


def decision_template(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    template: list[dict[str, Any]] = []
    for entry in entries:
        template.append(
            {
                "schema_version": REVIEW_DECISION_SCHEMA_VERSION,
                "occurrence_id": str(entry.get("occurrence_id") or ""),
                "project_identity_digest": str(
                    (entry.get("project") or {}).get("identity_digest") or ""
                ),
                "lifecycle": LIFECYCLE_OPEN,
                "reviewer": {"type": "human", "name": "TODO"},
                "binding": dict(entry.get("binding") or {}),
                "note": "",
                "decided_at": "",
            }
        )
    return template


def render_review_index_markdown(
    entries: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> str:
    scope = manifest.get("scope") if isinstance(manifest.get("scope"), Mapping) else {}
    lines = [
        "# 逐条审校索引",
        "",
        f"- 条目：{_coerce_int(scope.get('entry_count'))}",
        f"- 带 finding 条目：{_coerce_int(scope.get('entry_with_findings_count'))}",
        f"- needs_recheck：{_coerce_int(scope.get('needs_recheck_count'))}",
        "",
    ]
    current_file = ""
    for entry in entries:
        file_rel_path = str(entry.get("file_rel_path") or "")
        if file_rel_path != current_file:
            current_file = file_rel_path
            lines.extend([f"## {current_file}", ""])
        review = entry.get("review") if isinstance(entry.get("review"), Mapping) else {}
        lifecycle = str(review.get("lifecycle") or LIFECYCLE_OPEN)
        finding_ids = entry.get("quality_finding_ids") or []
        finding_label = f" findings={len(finding_ids)}" if finding_ids else ""
        lines.append(
            f"- L{_coerce_int((entry.get('locator') or {}).get('line_number'))} "
            f"[{lifecycle}]{finding_label} {entry.get('source') or ''}"
        )
        lines.append(f"  → {entry.get('current_translation') or ''}")
    return "\n".join(lines) + "\n"


def build_review_index(
    corpus_path: str | os.PathLike[str],
    *,
    quality_findings_path: str | os.PathLike[str] | None = None,
    translation_records_path: str | os.PathLike[str] | None = None,
    decisions_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build (or rebuild) one index package; decisions file is never overwritten."""

    target_dir = Path(output_dir)
    if decisions_path is None:
        existing_manifest_path = target_dir / REVIEW_INDEX_MANIFEST_NAME
        if existing_manifest_path.is_file():
            try:
                existing_manifest = _read_json_object(existing_manifest_path)
            except ReviewIndexError:
                existing_manifest = {}
            existing_inputs = existing_manifest.get("inputs")
            existing_inputs = (
                existing_inputs if isinstance(existing_inputs, Mapping) else {}
            )
            decisions_meta = existing_inputs.get("decisions")
            recorded_path = (
                str(decisions_meta.get("path") or "").strip()
                if isinstance(decisions_meta, Mapping)
                else ""
            )
            if recorded_path:
                decisions_path = recorded_path
            if quality_findings_path is None:
                findings_meta = existing_inputs.get("quality_findings")
                recorded_findings = (
                    str(findings_meta.get("path") or "").strip()
                    if isinstance(findings_meta, Mapping)
                    else ""
                )
                if recorded_findings:
                    quality_findings_path = recorded_findings
            if translation_records_path is None:
                records_meta = existing_inputs.get("translation_records")
                recorded_records = (
                    str(records_meta.get("path") or "").strip()
                    if isinstance(records_meta, Mapping)
                    else ""
                )
                if recorded_records:
                    translation_records_path = recorded_records
    corpus = load_corpus_bundle(corpus_path)
    findings_bundle = load_quality_findings(quality_findings_path)
    records_bundle = load_translation_records(translation_records_path)
    entries, diagnostics = build_index_entries(
        corpus["rows"],
        corpus["manifest"],
        findings=findings_bundle["findings"],
        records=records_bundle["records"],
    )
    decision_source = Path(decisions_path) if decisions_path else None
    decisions: list[dict[str, Any]] = []
    if decision_source is not None:
        if not decision_source.is_file():
            raise ReviewIndexError(
                "REVIEW_INDEX_INPUT_MISSING",
                f"decisions 文件不存在：{decision_source}",
                details={"path": str(decision_source)},
            )
        decisions = load_decisions(decision_source)
    entries, decision_diagnostics = apply_decisions(entries, decisions)
    diagnostics.extend(decision_diagnostics)

    target_dir.mkdir(parents=True, exist_ok=True)
    index_path = target_dir / REVIEW_INDEX_JSONL_NAME
    manifest_path = target_dir / REVIEW_INDEX_MANIFEST_NAME
    markdown_path = target_dir / REVIEW_INDEX_MARKDOWN_NAME
    decisions_file = target_dir / REVIEW_DECISIONS_JSONL_NAME
    template_path = target_dir / REVIEW_DECISIONS_TEMPLATE_NAME
    if decision_source is None and decisions_file.is_file():
        decision_source = decisions_file.resolve()
        decisions = load_decisions(decision_source)
        entries, extra = apply_decisions(entries, decisions)
        diagnostics.extend(extra)
    template_written = False
    if not decisions and not decisions_file.is_file():
        atomic_write_jsonl(
            template_path,
            decision_template(entries),
            ensure_ascii=False,
        )
        template_written = True
    atomic_write_jsonl(index_path, entries, ensure_ascii=False)
    lifecycle_counts = {key: 0 for key in sorted(LIFECYCLE_ALL)}
    for entry in entries:
        review = entry.get("review") if isinstance(entry.get("review"), Mapping) else {}
        lifecycle = str(review.get("lifecycle") or LIFECYCLE_OPEN)
        lifecycle_counts[lifecycle] = lifecycle_counts.get(lifecycle, 0) + 1
    entry_with_findings = sum(1 for entry in entries if entry.get("quality_finding_ids"))
    manifest = {
        "schema_version": REVIEW_INDEX_SCHEMA_VERSION,
        "kind": "review_index",
        "created_at": _utc_now(),
        "project": project_identity(corpus["manifest"]),
        "inputs": {
            "corpus_manifest": {
                "path": corpus["manifest_path"],
                "digest": corpus["manifest_digest"],
                "schema_version": _coerce_int(
                    (corpus["manifest"] or {}).get("schema_version")
                ),
                "snapshot_digest": str(
                    ((corpus["manifest"] or {}).get("source") or {}).get(
                        "snapshot_digest"
                    )
                    or ""
                ),
            },
            "corpus_jsonl": {
                "path": corpus["jsonl_path"],
                "digest": corpus["jsonl_digest"],
            },
            "quality_findings": {
                "path": findings_bundle["path"],
                "digest": findings_bundle["digest"],
                "count": findings_bundle["count"],
            },
            "translation_records": {
                "path": records_bundle["path"],
                "digest": records_bundle["digest"],
                "count": records_bundle["count"],
            },
            "decisions": {
                "path": (
                    os.path.abspath(str(decision_source))
                    if decision_source
                    else ""
                ),
                "digest": _file_digest(decision_source) if decision_source else "",
                "count": len(decisions),
            },
        },
        "scope": {
            "entry_count": len(entries),
            "finding_count": findings_bundle["count"],
            "matched_finding_count": sum(
                len(entry.get("quality_finding_ids") or []) for entry in entries
            ),
            "unmatched_finding_count": sum(
                1
                for item in diagnostics
                if item.get("code") == DIAGNOSTIC_FINDING_UNMATCHED
            ),
            "entry_with_findings_count": entry_with_findings,
            "needs_recheck_count": lifecycle_counts.get(
                LIFECYCLE_NEEDS_RECHECK, 0
            ),
            "project_mismatch_count": sum(
                1
                for item in diagnostics
                if item.get("code") == DIAGNOSTIC_DECISION_PROJECT_MISMATCH
            ),
            "orphaned_decision_count": sum(
                1
                for item in diagnostics
                if item.get("code") == DIAGNOSTIC_DECISION_ORPHANED
            ),
            "lifecycle_counts": lifecycle_counts,
        },
        "diagnostics": diagnostics,
        "paths": {
            "output_dir": os.path.abspath(target_dir),
            "jsonl": os.path.abspath(index_path),
            "manifest": os.path.abspath(manifest_path),
            "markdown": os.path.abspath(markdown_path),
            "decisions": os.path.abspath(decision_source) if decision_source else "",
            "template": (
                os.path.abspath(template_path) if template_written else ""
            ),
        },
    }
    atomic_write_text(
        markdown_path,
        render_review_index_markdown(entries, manifest),
    )
    atomic_write_json(manifest_path, manifest, ensure_ascii=False, indent=2)
    manifest["_output_dir"] = os.path.abspath(target_dir)
    manifest["_manifest_path"] = os.path.abspath(manifest_path)
    return manifest


def load_review_index(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    supplied = Path(path)
    if supplied.is_dir():
        supplied = supplied / REVIEW_INDEX_MANIFEST_NAME
    if not supplied.is_file():
        raise ReviewIndexError(
            "REVIEW_INDEX_MISSING",
            f"review index 不存在：{supplied}",
            details={"path": str(supplied)},
        )
    manifest = _read_json_object(supplied)
    jsonl_path = (
        (manifest.get("paths") or {}).get("jsonl")
        if isinstance(manifest.get("paths"), Mapping)
        else ""
    )
    if jsonl_path:
        candidate = Path(str(jsonl_path))
        if not candidate.is_absolute():
            candidate = supplied.parent / candidate
    else:
        candidate = supplied.parent / REVIEW_INDEX_JSONL_NAME
    if not candidate.is_file():
        raise ReviewIndexError(
            "REVIEW_INDEX_MISSING",
            f"review index JSONL 不存在：{candidate}",
            details={"path": str(candidate)},
        )
    entries = load_jsonl(candidate, label="review index")
    manifest["_manifest_path"] = str(supplied)
    manifest["_jsonl_path"] = str(candidate)
    return manifest, entries


def review_index_status(
    manifest: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lifecycle_counts = {key: 0 for key in sorted(LIFECYCLE_ALL)}
    for entry in entries:
        review = entry.get("review") if isinstance(entry.get("review"), Mapping) else {}
        lifecycle = str(review.get("lifecycle") or LIFECYCLE_OPEN)
        lifecycle_counts[lifecycle] = lifecycle_counts.get(lifecycle, 0) + 1
    scope = manifest.get("scope") if isinstance(manifest.get("scope"), Mapping) else {}
    diagnostics = list(manifest.get("diagnostics") or [])
    return {
        "entry_count": len(entries),
        "finding_count": _coerce_int(scope.get("finding_count")),
        "entry_with_findings_count": sum(
            1 for entry in entries if entry.get("quality_finding_ids")
        ),
        "needs_recheck_count": lifecycle_counts.get(LIFECYCLE_NEEDS_RECHECK, 0),
        "project_mismatch_count": sum(
            1
            for item in diagnostics
            if item.get("code") == DIAGNOSTIC_DECISION_PROJECT_MISMATCH
        ),
        "orphaned_decision_count": sum(
            1
            for item in diagnostics
            if item.get("code") == DIAGNOSTIC_DECISION_ORPHANED
        ),
        "lifecycle_counts": lifecycle_counts,
        "diagnostics": diagnostics,
        "index_manifest": str(manifest.get("_manifest_path") or ""),
        "index_jsonl": str(manifest.get("_jsonl_path") or ""),
    }


def import_decisions_into_index(
    index_path: str | os.PathLike[str],
    decisions_input: str | os.PathLike[str],
) -> dict[str, Any]:
    """Merge validated decisions into the package decision log and refresh index."""

    manifest, entries = load_review_index(index_path)
    decisions_path = str(
        ((manifest.get("inputs") or {}).get("decisions") or {}).get("path") or ""
    ).strip()
    if decisions_path:
        decisions_file = Path(decisions_path)
        if not decisions_file.is_file():
            raise ReviewIndexError(
                "REVIEW_INDEX_INPUT_MISSING",
                f"manifest 引用的 decisions 文件不存在：{decisions_file}",
                details={"path": str(decisions_file)},
            )
    else:
        decisions_file = (
            Path(manifest.get("_manifest_path", "")).resolve().parent
            / REVIEW_DECISIONS_JSONL_NAME
        )
        decisions_path = str(decisions_file)
    existing = load_decisions(decisions_path)
    incoming_rows = load_jsonl(decisions_input, label="review decisions import")
    incoming = [normalize_decision(row) for row in incoming_rows]
    current_project_id = (
        str((entries[0].get("project") or {}).get("identity_digest") or "")
        if entries
        else str((manifest.get("project") or {}).get("identity_digest") or "")
    )
    mismatched_incoming = [
        decision
        for decision in incoming
        if str(decision.get("project_identity_digest") or "")
        != current_project_id
    ]
    if mismatched_incoming:
        raise ReviewIndexError(
            DIAGNOSTIC_DECISION_PROJECT_MISMATCH,
            f"{len(mismatched_incoming)} 条决定属于其他项目，已拒绝导入。",
            details={
                "count": len(mismatched_incoming),
                "occurrence_ids": sorted(
                    str(decision.get("occurrence_id") or "")
                    for decision in mismatched_incoming
                ),
            },
        )
    known_ids = [str(entry.get("occurrence_id") or "") for entry in entries]
    merged, merge_summary = merge_decisions(
        existing,
        incoming,
        known_occurrence_ids=known_ids,
    )
    atomic_write_jsonl(decisions_file, merged, ensure_ascii=False)
    entries, decision_diagnostics = apply_decisions(entries, merged)
    jsonl_path = str(manifest.get("_jsonl_path") or "")
    atomic_write_jsonl(jsonl_path, entries, ensure_ascii=False)
    manifest_inputs = dict(manifest.get("inputs") or {})
    decisions_meta = dict(manifest_inputs.get("decisions") or {})
    decisions_meta.update(
        {
            "path": str(decisions_file),
            "digest": _file_digest(decisions_file),
            "count": len(merged),
        }
    )
    manifest_inputs["decisions"] = decisions_meta
    manifest["inputs"] = manifest_inputs
    manifest_paths = dict(manifest.get("paths") or {})
    manifest_paths["decisions"] = str(decisions_file)
    manifest["paths"] = manifest_paths
    stale_decision_codes = {
        DIAGNOSTIC_DECISION_ORPHANED,
        DIAGNOSTIC_DECISION_PROJECT_MISMATCH,
    }
    diagnostics = [
        item
        for item in (manifest.get("diagnostics") or [])
        if item.get("code") not in stale_decision_codes
    ]
    diagnostics.extend(decision_diagnostics)
    mismatched_count = sum(
        1
        for item in diagnostics
        if item.get("code") == DIAGNOSTIC_DECISION_PROJECT_MISMATCH
    )
    manifest["diagnostics"] = diagnostics
    status = review_index_status(manifest, entries)
    scope = dict(manifest.get("scope") or {})
    for key in (
        "entry_count",
        "finding_count",
        "entry_with_findings_count",
        "needs_recheck_count",
        "project_mismatch_count",
        "orphaned_decision_count",
        "lifecycle_counts",
    ):
        scope[key] = status[key]
    scope["unmatched_finding_count"] = sum(
        1
        for item in diagnostics
        if item.get("code") == DIAGNOSTIC_FINDING_UNMATCHED
    )
    manifest["scope"] = scope
    markdown_path = str(
        ((manifest.get("paths") or {}).get("markdown") or "")
    ).strip()
    if markdown_path:
        atomic_write_text(
            markdown_path,
            render_review_index_markdown(entries, manifest),
        )
    atomic_write_json(
        str(manifest.get("_manifest_path") or ""),
        {
            key: value
            for key, value in manifest.items()
            if not key.startswith("_")
        },
        ensure_ascii=False,
        indent=2,
    )
    merge_summary = {**merge_summary, "mismatched_count": mismatched_count}
    if mismatched_count:
        status = "blocked"
    elif manifest["scope"].get("needs_recheck_count"):
        status = "needs_recheck"
    else:
        status = "ready"
    return {
        "status": status,
        "merge": merge_summary,
        "scope": manifest["scope"],
        "decisions_path": str(decisions_file),
        "diagnostics": diagnostics,
    }


def export_decisions(
    index_path: str | os.PathLike[str],
    *,
    output_file: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Export current decisions, or a template when none have been imported."""

    manifest, entries = load_review_index(index_path)
    decisions_path = str(
        ((manifest.get("inputs") or {}).get("decisions") or {}).get("path") or ""
    ).strip()
    if decisions_path:
        decisions_file = Path(decisions_path)
        if not decisions_file.is_file():
            raise ReviewIndexError(
                "REVIEW_INDEX_INPUT_MISSING",
                f"manifest 引用的 decisions 文件不存在：{decisions_file}",
                details={"path": str(decisions_file)},
            )
        decisions = load_decisions(decisions_file)
        mode = "decisions"
    else:
        decisions = decision_template(entries)
        mode = "template"
    if output_file and str(output_file).strip():
        atomic_write_jsonl(
            str(output_file).strip(),
            decisions,
            ensure_ascii=False,
        )
    return {
        "status": mode,
        "mode": mode,
        "decision_count": len(decisions),
        "decisions_path": decisions_path,
        "output_file": str(output_file or ""),
        "index_manifest": str(manifest.get("_manifest_path") or ""),
    }
