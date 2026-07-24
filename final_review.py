"""Final-review campaign contract: readiness, snapshots, digests, package I/O.

Phase A (#255 PR A) owns report-only campaign structure without calling models
or writing ``.rpy`` files. Review execution (LLM / Batch) and revision hand-off
land in later PRs; they must reuse these digests and status rules rather than
re-implement readiness or ``fixed``/``applied`` claims in the GUI.

Design constraints from #255:
- Default is report-only (no autofix write path).
- Incomplete scope must refuse to start with actionable reasons.
- Context + translation snapshots are frozen into reproducible digests.
- Unit failure must never be recorded as done with zero findings.
- ``fixed`` / ``applied`` are never model-claimed; only revision apply may set them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from atomic_io import atomic_write_json, atomic_write_jsonl, atomic_write_text, file_sha256

SCHEMA_VERSION = 1
MANIFEST_MODE_FINAL_REVIEW = "final_review"
PROMPT_SCHEMA_VERSION = "final-review-v1"
DEFAULT_CHUNK_SIZE = 16

MANIFEST_FILENAME = "manifest.json"
SNAPSHOT_FILENAME = "snapshot.json"
REVIEW_UNITS_FILENAME = "review_units.jsonl"
FINDINGS_FILENAME = "findings.jsonl"
REPORT_MD_FILENAME = "report.md"
REQUESTS_JSONL_FILENAME = "requests.jsonl"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_STALE = "stale"

VALID_CAMPAIGN_STATUSES = frozenset(
    {
        STATUS_PENDING,
        STATUS_RUNNING,
        STATUS_DONE,
        STATUS_FAILED,
        STATUS_STALE,
    }
)

VALID_UNIT_STATUSES = frozenset(
    {
        STATUS_PENDING,
        STATUS_RUNNING,
        STATUS_DONE,
        STATUS_FAILED,
        STATUS_STALE,
    }
)

FINDING_TYPE_OMISSION = "omission"
FINDING_TYPE_MISTRANSLATION = "mistranslation"
FINDING_TYPE_ADDITION = "addition"
FINDING_TYPE_FORMAT = "format"
FINDING_TYPE_TERMINOLOGY = "terminology"
FINDING_TYPE_ADDRESS = "address"
FINDING_TYPE_STYLE_DRIFT = "style_drift"
FINDING_TYPE_NEEDS_CONFIRMATION = "needs_confirmation"

VALID_FINDING_TYPES = frozenset(
    {
        FINDING_TYPE_OMISSION,
        FINDING_TYPE_MISTRANSLATION,
        FINDING_TYPE_ADDITION,
        FINDING_TYPE_FORMAT,
        FINDING_TYPE_TERMINOLOGY,
        FINDING_TYPE_ADDRESS,
        FINDING_TYPE_STYLE_DRIFT,
        FINDING_TYPE_NEEDS_CONFIRMATION,
    }
)

SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"

VALID_SEVERITIES = frozenset(
    {
        SEVERITY_HIGH,
        SEVERITY_MEDIUM,
        SEVERITY_LOW,
        SEVERITY_INFO,
    }
)

REVISION_STATE_NONE = "none"
REVISION_STATE_CANDIDATE = "candidate"
REVISION_STATE_PREVIEWED = "previewed"
REVISION_STATE_APPLIED = "applied"

VALID_REVISION_STATES = frozenset(
    {
        REVISION_STATE_NONE,
        REVISION_STATE_CANDIDATE,
        REVISION_STATE_PREVIEWED,
        REVISION_STATE_APPLIED,
    }
)

# Model output must never promote findings into these applied-like states.
MODEL_FORBIDDEN_REVISION_STATES = frozenset(
    {
        REVISION_STATE_PREVIEWED,
        REVISION_STATE_APPLIED,
        "fixed",
        "applied",
        "done",
    }
)

SELECTION_STATE_OPEN = "open"
SELECTION_STATE_SELECTED = "selected"
SELECTION_STATE_DISMISSED = "dismissed"

VALID_SELECTION_STATES = frozenset(
    {
        SELECTION_STATE_OPEN,
        SELECTION_STATE_SELECTED,
        SELECTION_STATE_DISMISSED,
    }
)


class FinalReviewError(ValueError):
    """Base error for final-review contract violations."""


class FinalReviewSchemaError(FinalReviewError):
    """Raised when campaign / unit / finding shape is invalid."""


class FinalReviewReadinessError(FinalReviewError):
    """Raised when the review scope fails the completion contract."""

    def __init__(self, reasons: Sequence[str], *, details: Mapping[str, Any] | None = None):
        self.reasons = [str(r).strip() for r in reasons if str(r).strip()]
        self.details = dict(details or {})
        message = "; ".join(self.reasons) if self.reasons else "scope is not ready for final review"
        super().__init__(message)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _as_optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_str_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise FinalReviewSchemaError(f"{field_name} must be a list of strings")
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _normalize_rel_path(path: str) -> str:
    text = str(path or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


@dataclass
class ReadinessReport:
    ready: bool
    reasons: list[str] = field(default_factory=list)
    pending_task_count: int = 0
    pending_file_count: int = 0
    review_item_count: int = 0
    pending_files: list[dict[str, Any]] = field(default_factory=list)
    require_zero_pending: bool = True
    allow_pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": bool(self.ready),
            "reasons": list(self.reasons),
            "pending_task_count": int(self.pending_task_count),
            "pending_file_count": int(self.pending_file_count),
            "review_item_count": int(self.review_item_count),
            "pending_files": list(self.pending_files),
            "require_zero_pending": bool(self.require_zero_pending),
            "allow_pending": bool(self.allow_pending),
        }


def evaluate_readiness(
    *,
    pending_task_count: int = 0,
    pending_files: Sequence[Mapping[str, Any]] | None = None,
    review_item_count: int = 0,
    require_zero_pending: bool = True,
    allow_pending: bool = False,
) -> ReadinessReport:
    """Decide whether a final-review campaign may be built for the scope.

    Parameters
    ----------
    pending_task_count:
        Unfinished translation tasks inside the review scope.
    pending_files:
        Optional per-file rows ``{file_rel_path, pending_task_count, ...}`` used
        to produce actionable refusal reasons.
    review_item_count:
        Already-translated source/translation pairs available for review.
    require_zero_pending:
        When True (default), any pending task blocks start unless *allow_pending*.
    allow_pending:
        Explicit override (CLI ``--allow-pending``); still surfaces warnings in
        reasons when pending remains.
    """
    pending = max(0, int(pending_task_count or 0))
    items = max(0, int(review_item_count or 0))
    file_rows: list[dict[str, Any]] = []
    for row in pending_files or []:
        if not isinstance(row, Mapping):
            continue
        rel = _normalize_rel_path(str(row.get("file_rel_path") or row.get("path") or ""))
        count = int(row.get("pending_task_count") or row.get("task_count") or 0)
        if count <= 0 and not rel:
            continue
        file_rows.append(
            {
                "file_rel_path": rel,
                "pending_task_count": max(0, count),
            }
        )
    file_rows.sort(key=lambda r: (r["file_rel_path"], r["pending_task_count"]))
    pending_file_count = sum(1 for r in file_rows if r["pending_task_count"] > 0)
    if pending_file_count <= 0 and pending > 0:
        pending_file_count = 1

    reasons: list[str] = []
    ready = True

    if items <= 0:
        ready = False
        reasons.append(
            "审校范围内没有可审校的已译条目（需要同时存在原文与当前译文）。"
            "请先完成翻译，或调整 include 过滤范围。"
        )

    enforce_zero = bool(require_zero_pending) and not bool(allow_pending)
    if pending > 0 and enforce_zero:
        ready = False
        sample = ", ".join(
            f"{r['file_rel_path']}({r['pending_task_count']})"
            for r in file_rows[:8]
            if r["pending_task_count"] > 0
        )
        detail = f" 示例文件: {sample}。" if sample else ""
        reasons.append(
            f"审校范围内仍有 {pending} 条未完成翻译"
            f"（约 {pending_file_count} 个文件）。{detail}"
            "请先跑完 Batch check/apply，或用 include 排除未完成文件，"
            "或显式传 --allow-pending（不推荐）。"
        )
    elif pending > 0 and allow_pending:
        reasons.append(
            f"警告: 已用 --allow-pending 忽略 {pending} 条未完成翻译；"
            "审校结果可能不完整。"
        )

    return ReadinessReport(
        ready=ready,
        reasons=reasons,
        pending_task_count=pending,
        pending_file_count=pending_file_count,
        review_item_count=items,
        pending_files=file_rows,
        require_zero_pending=bool(require_zero_pending),
        allow_pending=bool(allow_pending),
    )


def require_readiness(report: ReadinessReport | Mapping[str, Any]) -> ReadinessReport:
    if isinstance(report, ReadinessReport):
        payload = report
    else:
        payload = ReadinessReport(
            ready=bool(report.get("ready")),
            reasons=list(report.get("reasons") or []),
            pending_task_count=int(report.get("pending_task_count") or 0),
            pending_file_count=int(report.get("pending_file_count") or 0),
            review_item_count=int(report.get("review_item_count") or 0),
            pending_files=list(report.get("pending_files") or []),
            require_zero_pending=bool(report.get("require_zero_pending", True)),
            allow_pending=bool(report.get("allow_pending", False)),
        )
    if not payload.ready:
        raise FinalReviewReadinessError(payload.reasons, details=payload.to_dict())
    return payload


# ---------------------------------------------------------------------------
# Snapshot digests
# ---------------------------------------------------------------------------


def digest_translation_items(items: Sequence[Mapping[str, Any]]) -> str:
    """Stable digest over source/current-translation pairs (identity keyed)."""
    rows: list[dict[str, str]] = []
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        identity = _as_optional_str(
            item.get("identity_v2") or item.get("id") or item.get("identity")
        )
        rows.append(
            {
                "id": identity,
                "file_rel_path": _normalize_rel_path(
                    str(item.get("file_rel_path") or item.get("path") or "")
                ),
                "source": str(item.get("source") or item.get("text") or ""),
                "current_translation": str(
                    item.get("current_translation") or item.get("translation") or ""
                ),
            }
        )
    rows.sort(key=lambda r: (r["file_rel_path"], r["id"], r["source"]))
    return stable_json_sha256(rows)


def digest_path_content(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Return path metadata + content hash when the file exists."""
    text = str(path or "").strip()
    if not text:
        return {"path": "", "exists": False, "sha256": "", "size": 0}
    abs_path = os.path.abspath(text)
    if not os.path.isfile(abs_path):
        return {"path": abs_path, "exists": False, "sha256": "", "size": 0}
    try:
        size = int(os.path.getsize(abs_path))
    except OSError:
        size = 0
    try:
        digest = file_sha256(abs_path)
    except OSError:
        digest = ""
    return {"path": abs_path, "exists": True, "sha256": digest, "size": size}


def digest_text_blob(text: str | None, *, label: str = "") -> dict[str, Any]:
    body = str(text or "")
    return {
        "label": label,
        "sha256": sha256_text(body) if body else "",
        "char_count": len(body),
        "present": bool(body.strip()),
    }


# Authoritative Source Index store files (rag_memory.SourceIndexStore).
# Locks / *.tmp.* / unrelated files must never contribute to the digest.
SOURCE_INDEX_AUTHORITATIVE_FILES = (
    "source_metadata.json",
    "source_segments.jsonl",
)


def digest_source_index_store(
    store_path: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Deterministic digest over Source Index store content.

    The store is a *directory* whose authoritative files are
    ``source_metadata.json`` and ``source_segments.jsonl``. Temporary write
    artifacts (``*.tmp.*``), lock files, and other siblings are ignored so a
    concurrent bootstrap cannot thrash digests.
    """
    text = str(store_path or "").strip()
    if not text:
        return {
            "path": "",
            "exists": False,
            "sha256": "",
            "size": 0,
            "files": {},
        }
    abs_path = os.path.abspath(text)
    # Allow passing a single file for tests; treat as that basename only.
    if os.path.isfile(abs_path):
        meta = digest_path_content(abs_path)
        return {
            "path": abs_path,
            "exists": bool(meta.get("exists")),
            "sha256": meta.get("sha256") or "",
            "size": int(meta.get("size") or 0),
            "files": {os.path.basename(abs_path): meta},
        }
    if not os.path.isdir(abs_path):
        return {
            "path": abs_path,
            "exists": False,
            "sha256": "",
            "size": 0,
            "files": {},
        }

    file_rows: list[dict[str, Any]] = []
    files_meta: dict[str, Any] = {}
    total_size = 0
    for name in SOURCE_INDEX_AUTHORITATIVE_FILES:
        file_path = os.path.join(abs_path, name)
        meta = digest_path_content(file_path)
        files_meta[name] = {
            "exists": bool(meta.get("exists")),
            "sha256": meta.get("sha256") or "",
            "size": int(meta.get("size") or 0),
        }
        total_size += int(meta.get("size") or 0)
        file_rows.append(
            {
                "name": name,
                "exists": bool(meta.get("exists")),
                "sha256": meta.get("sha256") or "",
                "size": int(meta.get("size") or 0),
            }
        )
    combined = stable_json_sha256(file_rows)
    any_exists = any(row["exists"] for row in file_rows)
    return {
        "path": abs_path,
        "exists": any_exists,
        "sha256": combined if any_exists else "",
        "size": total_size,
        "files": files_meta,
    }


def _project_analysis_digest_fields(
    *,
    enabled: bool,
    inject: bool,
    status: str,
    fingerprint: str,
    version: str | int | None,
    store_path: str,
    brief_text: str,
    lineage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build PA layer + digest-slice fields.

    When injectable, bind the *actual injected brief text* (already max_chars
    truncated by the loader) and artifact lineage — not merely schema_version
    or a structure fingerprint. Same fingerprint with a force-republished brief
    body must change the digest.
    """
    brief_meta = digest_text_blob(brief_text, label="project_analysis_brief")
    lineage_obj = dict(lineage or {}) if isinstance(lineage, Mapping) else {}
    # Keep only stable lineage keys that identify the artifact generation.
    lineage_slice = {
        "source_fingerprint": _as_optional_str(lineage_obj.get("source_fingerprint")),
        "upstream_dependency_digest": _as_optional_str(
            lineage_obj.get("upstream_dependency_digest")
        ),
        "prompt_schema_version": _as_optional_str(lineage_obj.get("prompt_schema_version")),
        "generated_at": _as_optional_str(lineage_obj.get("generated_at")),
        "provider": _as_optional_str(lineage_obj.get("provider")),
        "model": _as_optional_str(lineage_obj.get("model")),
    }
    lineage_digest = stable_json_sha256(lineage_slice) if any(lineage_slice.values()) else ""

    # Include when inject is on and we have either brief text or a fingerprint.
    included = bool(enabled and inject and (brief_meta["present"] or fingerprint))
    meta = {
        "enabled": bool(enabled),
        "inject": bool(inject),
        "status": _as_optional_str(status),
        "fingerprint": _as_optional_str(fingerprint),
        "version": version,
        "store_path": str(store_path or ""),
        "brief_text_sha256": brief_meta["sha256"] if included else "",
        "brief_char_count": brief_meta["char_count"] if included else 0,
        "lineage_digest": lineage_digest if included else "",
        "lineage": lineage_slice if included else {},
        "included_in_digest": included,
    }
    return meta


def build_context_snapshot(
    *,
    translation_items: Sequence[Mapping[str, Any]],
    glossary_path: str | os.PathLike[str] | None = None,
    glossary_enabled: bool = True,
    macro_setting_text: str = "",
    macro_setting_path: str | os.PathLike[str] | None = None,
    story_memory_enabled: bool = False,
    story_memory_graph_path: str | os.PathLike[str] | None = None,
    source_index_enabled: bool = False,
    source_index_store_path: str | os.PathLike[str] | None = None,
    project_analysis_enabled: bool = False,
    project_analysis_inject: bool = False,
    project_analysis_status: str = "",
    project_analysis_fingerprint: str = "",
    project_analysis_version: str | int | None = None,
    project_analysis_store_path: str | os.PathLike[str] | None = None,
    project_analysis_brief_text: str = "",
    project_analysis_lineage: Mapping[str, Any] | None = None,
    include_filters: Sequence[str] | None = None,
    base_dir: str = "",
    tl_dir: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze enabled context dependencies into a reproducible snapshot document.

    Digests are split on purpose:

    - ``context_digest`` — shared enabled context only (glossary/macro/story/
      source index/PA brief…). Used by each unit's ``input_digest`` so a
      translation edit in file A does not stale units in file B.
    - ``snapshot_digest`` — context + full-scope ``translations_digest`` for
      campaign-level audit / completeness. Not bound into unit digests.

    Only layers that are enabled (or always-on translation/glossary/macro)
    contribute content digests. Disabled optional layers record
    ``enabled: false`` so flipping a switch changes the digests.
    """
    items = list(translation_items or [])
    translation_digest = digest_translation_items(items)

    glossary_meta = (
        digest_path_content(glossary_path)
        if glossary_enabled
        else {"path": str(glossary_path or ""), "exists": False, "sha256": "", "size": 0}
    )
    macro_text_meta = digest_text_blob(macro_setting_text, label="macro_setting")
    macro_path_meta = digest_path_content(macro_setting_path) if macro_setting_path else {
        "path": "",
        "exists": False,
        "sha256": "",
        "size": 0,
    }

    if story_memory_enabled:
        story_meta = {
            "enabled": True,
            **digest_path_content(story_memory_graph_path),
        }
    else:
        story_meta = {"enabled": False, "path": "", "exists": False, "sha256": "", "size": 0}

    if source_index_enabled:
        source_meta = {
            "enabled": True,
            **digest_source_index_store(source_index_store_path),
        }
    else:
        source_meta = {
            "enabled": False,
            "path": "",
            "exists": False,
            "sha256": "",
            "size": 0,
            "files": {},
        }

    pa_meta = _project_analysis_digest_fields(
        enabled=bool(project_analysis_enabled),
        inject=bool(project_analysis_inject),
        status=project_analysis_status,
        fingerprint=project_analysis_fingerprint,
        version=project_analysis_version,
        store_path=str(project_analysis_store_path or ""),
        brief_text=project_analysis_brief_text,
        lineage=project_analysis_lineage,
    )

    scope = {
        "base_dir": str(base_dir or ""),
        "tl_dir": str(tl_dir or ""),
        "include_filters": sorted(
            {_normalize_rel_path(x) for x in (include_filters or []) if str(x).strip()}
        ),
        "item_count": len(items),
    }

    layers = {
        "translations": {
            "item_count": len(items),
            "digest": translation_digest,
        },
        "glossary": {
            "enabled": bool(glossary_enabled),
            **glossary_meta,
        },
        "macro_setting": {
            "text": macro_text_meta,
            "path": macro_path_meta,
        },
        "story_memory": story_meta,
        "source_index": source_meta,
        "project_analysis": pa_meta,
    }

    # Shared context only (no per-item translations). Absolute paths omitted.
    context_payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "include_filters": scope["include_filters"],
            "base_dir_name": os.path.basename(os.path.normpath(scope["base_dir"]))
            if scope["base_dir"]
            else "",
            "tl_dir_name": os.path.basename(os.path.normpath(scope["tl_dir"]))
            if scope["tl_dir"]
            else "",
        },
        "glossary_sha256": glossary_meta.get("sha256") or "",
        "macro_text_sha256": macro_text_meta.get("sha256") or "",
        "macro_path_sha256": macro_path_meta.get("sha256") or "",
        "story_memory": {
            "enabled": story_meta.get("enabled"),
            "sha256": story_meta.get("sha256") or "",
        },
        "source_index": {
            "enabled": source_meta.get("enabled"),
            "sha256": source_meta.get("sha256") or "",
        },
        "project_analysis": {
            "included_in_digest": pa_meta.get("included_in_digest"),
            "fingerprint": pa_meta.get("fingerprint") if pa_meta.get("included_in_digest") else "",
            "version": pa_meta.get("version") if pa_meta.get("included_in_digest") else None,
            "status": pa_meta.get("status") if pa_meta.get("included_in_digest") else "",
            "brief_text_sha256": pa_meta.get("brief_text_sha256")
            if pa_meta.get("included_in_digest")
            else "",
            "lineage_digest": pa_meta.get("lineage_digest")
            if pa_meta.get("included_in_digest")
            else "",
        },
        "extra": dict(extra or {}),
    }
    context_digest = stable_json_sha256(context_payload)

    # Full campaign audit digest: shared context + translation inventory.
    digest_payload = {
        **context_payload,
        "scope": {
            **context_payload["scope"],
            "item_count": scope["item_count"],
        },
        "translations_digest": translation_digest,
    }
    snapshot_digest = stable_json_sha256(digest_payload)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "context_digest": context_digest,
        "snapshot_digest": snapshot_digest,
        "scope": scope,
        "layers": layers,
        "context_payload": context_payload,
        "digest_payload": digest_payload,
    }


# ---------------------------------------------------------------------------
# Review units
# ---------------------------------------------------------------------------


def _item_identity(item: Mapping[str, Any]) -> str:
    return _as_optional_str(
        item.get("identity_v2") or item.get("id") or item.get("identity")
    )


def normalize_review_item(item: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise FinalReviewSchemaError("review item must be a mapping")
    identity = _item_identity(item)
    source = str(item.get("source") or item.get("text") or "")
    current = str(item.get("current_translation") or item.get("translation") or "")
    if not identity:
        # Deterministic fallback identity for fixtures without identity_v2.
        identity = stable_json_sha256(
            {
                "file": _normalize_rel_path(str(item.get("file_rel_path") or "")),
                "source": source,
                "line": item.get("line_number") or item.get("line"),
                "start": item.get("start"),
            }
        )[:16]
    return {
        "id": identity,
        "identity_v2": identity,
        "file_rel_path": _normalize_rel_path(str(item.get("file_rel_path") or "")),
        "source": source,
        "current_translation": current,
        "line_number": item.get("line_number") or item.get("line") or 0,
        "start": item.get("start"),
        "end": item.get("end"),
        "speaker_id": _as_optional_str(item.get("speaker_id") or item.get("speaker")),
    }


def compute_unit_input_digest(
    *,
    item_ids: Sequence[str],
    items_digest: str,
    context_digest: str,
    model: str = "",
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    chunk_index: int = 0,
    file_rel_path: str = "",
) -> str:
    """Digest for one review unit.

    Binds shared *context_digest* (glossary/macro/PA/…) and this unit's
    ``items_digest`` only — never the campaign-wide translations inventory —
    so edits outside the unit do not force a re-run.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "file_rel_path": _normalize_rel_path(file_rel_path),
        "chunk_index": int(chunk_index),
        "item_ids": list(item_ids),
        "items_digest": items_digest,
        "context_digest": _as_optional_str(context_digest),
        "model": _as_optional_str(model),
        "prompt_schema_version": _as_optional_str(prompt_schema_version)
        or PROMPT_SCHEMA_VERSION,
    }
    return stable_json_sha256(payload)


def build_review_units(
    items: Sequence[Mapping[str, Any]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_digest: str = "",
    snapshot_digest: str = "",
    model: str = "",
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    """Chunk normalized items into review units with stable input digests.

    ``context_digest`` is required for resume correctness (shared context).
    ``snapshot_digest`` is recorded for audit only and is not part of the unit
    input digest. For backward-compatible callers that only pass
    ``snapshot_digest``, it is used as a fallback context digest with a note
    that incremental granularity may be coarser.
    """
    shared_context = _as_optional_str(context_digest) or _as_optional_str(snapshot_digest)
    size = max(1, int(chunk_size or DEFAULT_CHUNK_SIZE))
    normalized = [normalize_review_item(item) for item in items or []]
    # Group by file to keep local context coherent (matches revision batching).
    by_file: dict[str, list[dict[str, Any]]] = {}
    for item in normalized:
        by_file.setdefault(item["file_rel_path"] or "_global", []).append(item)

    units: list[dict[str, Any]] = []
    for file_rel_path in sorted(by_file.keys()):
        file_items = by_file[file_rel_path]
        total = len(file_items)
        for start in range(0, total, size):
            chunk_items = file_items[start : start + size]
            chunk_index = start // size + 1
            item_ids = [it["id"] for it in chunk_items]
            items_digest = digest_translation_items(chunk_items)
            unit_id = (
                f"fr-{stable_json_sha256({'file': file_rel_path, 'idx': chunk_index, 'ids': item_ids})[:12]}"
            )
            input_digest = compute_unit_input_digest(
                item_ids=item_ids,
                items_digest=items_digest,
                context_digest=shared_context,
                model=model,
                prompt_schema_version=prompt_schema_version,
                chunk_index=chunk_index,
                file_rel_path=file_rel_path,
            )
            units.append(
                {
                    "unit_id": unit_id,
                    "status": STATUS_PENDING,
                    "file_rel_path": file_rel_path,
                    "chunk_index": chunk_index,
                    "item_ids": item_ids,
                    "item_count": len(chunk_items),
                    "items": chunk_items,
                    "items_digest": items_digest,
                    "input_digest": input_digest,
                    "context_digest": shared_context,
                    "snapshot_digest": _as_optional_str(snapshot_digest),
                    "model": _as_optional_str(model),
                    "prompt_schema_version": _as_optional_str(prompt_schema_version)
                    or PROMPT_SCHEMA_VERSION,
                    "error": "",
                    "finding_count": 0,
                    "completed_at": "",
                }
            )
    return units


def reevaluate_unit_status(
    unit: Mapping[str, Any],
    *,
    live_input_digest: str,
    force: bool = False,
) -> str:
    """Return effective unit status given a freshly computed live digest.

    - ``force`` always yields ``pending`` (caller will re-run).
    - Digest mismatch on a previously ``done`` unit yields ``stale``.
    - ``failed`` units stay ``failed`` unless force (still not auto-done).
    """
    current = _as_optional_str(unit.get("status")) or STATUS_PENDING
    if current not in VALID_UNIT_STATUSES:
        current = STATUS_PENDING
    stored = _as_optional_str(unit.get("input_digest"))
    live = _as_optional_str(live_input_digest)
    if force:
        return STATUS_PENDING
    if not live:
        return current
    if stored and live != stored:
        if current == STATUS_DONE:
            return STATUS_STALE
        if current == STATUS_FAILED:
            return STATUS_STALE
        if current == STATUS_PENDING:
            return STATUS_STALE
        return STATUS_STALE
    return current


def should_skip_unit(
    unit: Mapping[str, Any],
    *,
    live_input_digest: str = "",
    force: bool = False,
) -> bool:
    """True when a unit is done and digests still match (resume skip)."""
    if force:
        return False
    status = reevaluate_unit_status(
        unit, live_input_digest=live_input_digest or _as_optional_str(unit.get("input_digest")), force=False
    )
    if status != STATUS_DONE:
        return False
    live = _as_optional_str(live_input_digest) or _as_optional_str(unit.get("input_digest"))
    stored = _as_optional_str(unit.get("input_digest"))
    return bool(live and stored and live == stored)


def mark_unit_failed(unit: Mapping[str, Any], error: str) -> dict[str, Any]:
    """Return a unit record marked failed (never done / zero findings)."""
    out = dict(unit)
    out["status"] = STATUS_FAILED
    out["error"] = _as_optional_str(error) or "review_unit_failed"
    out["finding_count"] = int(out.get("finding_count") or 0)
    # Explicitly refuse to look like a clean completion.
    out["completed_at"] = ""
    return out


def mark_unit_done(
    unit: Mapping[str, Any],
    *,
    finding_count: int,
    completed_at: str | None = None,
) -> dict[str, Any]:
    out = dict(unit)
    out["status"] = STATUS_DONE
    out["error"] = ""
    out["finding_count"] = max(0, int(finding_count))
    out["completed_at"] = completed_at or utc_now_iso()
    return out


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


def normalize_finding(
    record: Mapping[str, Any],
    *,
    review_unit_id: str = "",
    review_unit_digest: str = "",
    provider: str = "",
    model: str = "",
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    allow_model_revision_state: bool = False,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise FinalReviewSchemaError("finding must be a mapping")

    finding_type = _as_optional_str(record.get("finding_type") or record.get("type")).lower()
    if finding_type not in VALID_FINDING_TYPES:
        # Uncertain / unknown types collapse to needs_confirmation rather than inventing certainty.
        finding_type = FINDING_TYPE_NEEDS_CONFIRMATION

    severity = _as_optional_str(record.get("severity")).lower() or SEVERITY_MEDIUM
    if severity not in VALID_SEVERITIES:
        severity = SEVERITY_MEDIUM

    identity = _as_optional_str(
        record.get("identity_v2") or record.get("id") or record.get("identity")
    )
    source = str(record.get("source") or "")
    current = str(record.get("current_translation") or record.get("translation") or "")
    unit_id = _as_optional_str(record.get("review_unit_id")) or _as_optional_str(review_unit_id)
    unit_digest = _as_optional_str(record.get("review_unit_digest")) or _as_optional_str(
        review_unit_digest
    )

    revision_state = _as_optional_str(record.get("revision_state")).lower() or REVISION_STATE_NONE
    if not allow_model_revision_state:
        # Strip model claims of fixed/applied.
        if revision_state in MODEL_FORBIDDEN_REVISION_STATES or revision_state not in VALID_REVISION_STATES:
            revision_state = REVISION_STATE_NONE

    selection_state = (
        _as_optional_str(record.get("selection_state")).lower() or SELECTION_STATE_OPEN
    )
    if selection_state not in VALID_SELECTION_STATES:
        selection_state = SELECTION_STATE_OPEN

    finding_id = _as_optional_str(record.get("finding_id"))
    if not finding_id:
        finding_id = stable_json_sha256(
            {
                "identity": identity,
                "type": finding_type,
                "source": source,
                "current": current,
                "reason": str(record.get("reason") or ""),
                "unit": unit_id,
            }
        )[:20]

    return {
        "finding_id": finding_id,
        "identity_v2": identity,
        "file_rel_path": _normalize_rel_path(str(record.get("file_rel_path") or "")),
        "source": source,
        "current_translation": current,
        "finding_type": finding_type,
        "severity": severity,
        "evidence": str(record.get("evidence") or ""),
        "reason": str(record.get("reason") or record.get("detail") or ""),
        "suggested_revision": str(
            record.get("suggested_revision") or record.get("suggestion") or ""
        ),
        "provider": _as_optional_str(record.get("provider")) or _as_optional_str(provider),
        "model": _as_optional_str(record.get("model")) or _as_optional_str(model),
        "prompt_schema_version": _as_optional_str(record.get("prompt_schema_version"))
        or _as_optional_str(prompt_schema_version)
        or PROMPT_SCHEMA_VERSION,
        "review_unit_id": unit_id,
        "review_unit_digest": unit_digest,
        "selection_state": selection_state,
        "revision_state": revision_state,
    }


# ---------------------------------------------------------------------------
# Campaign package
# ---------------------------------------------------------------------------


def default_final_review_config(
    *,
    enabled: bool = True,
    require_zero_pending: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    store_under_batch_jobs: bool = True,
    model: str = "",
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "require_zero_pending": bool(require_zero_pending),
        "chunk_size": max(1, int(chunk_size or DEFAULT_CHUNK_SIZE)),
        "prompt_schema_version": _as_optional_str(prompt_schema_version) or PROMPT_SCHEMA_VERSION,
        "store_under_batch_jobs": bool(store_under_batch_jobs),
        "model": _as_optional_str(model),
    }


def merge_final_review_config(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = default_final_review_config()
    if not isinstance(raw, Mapping):
        return base
    if "enabled" in raw:
        base["enabled"] = bool(raw.get("enabled"))
    if "require_zero_pending" in raw:
        base["require_zero_pending"] = bool(raw.get("require_zero_pending"))
    if "chunk_size" in raw:
        try:
            base["chunk_size"] = max(1, int(raw.get("chunk_size") or DEFAULT_CHUNK_SIZE))
        except (TypeError, ValueError):
            pass
    if "prompt_schema_version" in raw:
        text = _as_optional_str(raw.get("prompt_schema_version"))
        if text:
            base["prompt_schema_version"] = text
    if "store_under_batch_jobs" in raw:
        base["store_under_batch_jobs"] = bool(raw.get("store_under_batch_jobs"))
    if "model" in raw:
        base["model"] = _as_optional_str(raw.get("model"))
    return base


def build_campaign_manifest(
    *,
    package_dir: str,
    display_name: str,
    snapshot: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any] | ReadinessReport,
    base_dir: str = "",
    tl_dir: str = "",
    model: str = "",
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    batch_model: str = "",
    settings: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(readiness, ReadinessReport):
        readiness_dict = readiness.to_dict()
    else:
        readiness_dict = dict(readiness)

    unit_list = list(units or [])
    status_counts = summarize_unit_statuses(unit_list)
    campaign_status = derive_campaign_status(status_counts)

    manifest: dict[str, Any] = {
        "version": 1,
        "manifest_version": 1,
        "schema_version": SCHEMA_VERSION,
        "mode": MANIFEST_MODE_FINAL_REVIEW,
        "created_at": utc_now_iso(),
        "display_name": _as_optional_str(display_name),
        "status": campaign_status,
        "report_only": True,
        "autofix": False,
        "base_dir": str(base_dir or ""),
        "tl_dir": str(tl_dir or ""),
        "batch_model": _as_optional_str(batch_model) or _as_optional_str(model),
        "model": _as_optional_str(model) or _as_optional_str(batch_model),
        "prompt_schema_version": _as_optional_str(prompt_schema_version) or PROMPT_SCHEMA_VERSION,
        "package_dir": os.path.abspath(package_dir) if package_dir else "",
        "context_digest": _as_optional_str(snapshot.get("context_digest")),
        "snapshot_digest": _as_optional_str(snapshot.get("snapshot_digest")),
        "snapshot_path": SNAPSHOT_FILENAME,
        "review_units_path": REVIEW_UNITS_FILENAME,
        "findings_path": FINDINGS_FILENAME,
        "input_jsonl_path": REQUESTS_JSONL_FILENAME,
        "result_jsonl_path": "",
        "job_name": "",
        "job_state": "LOCAL_ONLY",
        "settings": {
            "chunk_size": max(1, int(chunk_size or DEFAULT_CHUNK_SIZE)),
            "require_zero_pending": bool(readiness_dict.get("require_zero_pending", True)),
            **dict(settings or {}),
        },
        "final_review_settings": {
            "chunk_size": max(1, int(chunk_size or DEFAULT_CHUNK_SIZE)),
            "prompt_schema_version": _as_optional_str(prompt_schema_version)
            or PROMPT_SCHEMA_VERSION,
            "report_only": True,
        },
        "readiness": readiness_dict,
        "summary": {
            "unit_count": len(unit_list),
            "item_count": sum(int(u.get("item_count") or 0) for u in unit_list),
            "finding_count": 0,
            "status_counts": status_counts,
        },
        "scope": dict(snapshot.get("scope") or {}),
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def summarize_unit_statuses(units: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(VALID_UNIT_STATUSES)}
    for unit in units or []:
        status = _as_optional_str(unit.get("status")) or STATUS_PENDING
        if status not in counts:
            status = STATUS_PENDING
        counts[status] = counts.get(status, 0) + 1
    return counts


def derive_campaign_status(status_counts: Mapping[str, int]) -> str:
    """Aggregate unit statuses into a campaign-level status.

    Failure is sticky: any failed unit prevents campaign ``done``.
    """
    counts = {k: int(status_counts.get(k) or 0) for k in VALID_UNIT_STATUSES}
    total = sum(counts.values())
    if total <= 0:
        return STATUS_PENDING
    if counts.get(STATUS_RUNNING, 0) > 0:
        return STATUS_RUNNING
    if counts.get(STATUS_FAILED, 0) > 0:
        return STATUS_FAILED
    if counts.get(STATUS_STALE, 0) > 0:
        return STATUS_STALE
    if counts.get(STATUS_PENDING, 0) > 0:
        if counts.get(STATUS_DONE, 0) > 0:
            return STATUS_RUNNING  # partially complete
        return STATUS_PENDING
    if counts.get(STATUS_DONE, 0) == total:
        return STATUS_DONE
    return STATUS_PENDING


def assert_failure_not_done(unit: Mapping[str, Any]) -> None:
    """Invariant: a unit with an error must not claim done."""
    status = _as_optional_str(unit.get("status"))
    error = _as_optional_str(unit.get("error"))
    if error and status == STATUS_DONE:
        raise FinalReviewSchemaError(
            "failed review unit must not be recorded as done "
            f"(unit_id={unit.get('unit_id')!r}, error={error!r})"
        )


def write_campaign_package(
    package_dir: str | os.PathLike[str],
    *,
    manifest: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]] | None = None,
    write_report: bool = True,
) -> dict[str, str]:
    """Atomically write campaign artifacts. Returns absolute paths."""
    root = os.path.abspath(os.fspath(package_dir))
    os.makedirs(root, exist_ok=True)

    for unit in units or []:
        assert_failure_not_done(unit)

    # Persist units without full item bodies? Keep items for resume without re-scan.
    unit_rows = []
    for unit in units or []:
        row = dict(unit)
        unit_rows.append(row)

    finding_rows = [normalize_finding(f) for f in (findings or [])]

    manifest_out = dict(manifest)
    manifest_out["package_dir"] = root
    manifest_out["_package_dir"] = root
    manifest_path = os.path.join(root, MANIFEST_FILENAME)
    manifest_out["_manifest_path"] = manifest_path
    status_counts = summarize_unit_statuses(unit_rows)
    manifest_out["summary"] = {
        **dict(manifest_out.get("summary") or {}),
        "unit_count": len(unit_rows),
        "item_count": sum(int(u.get("item_count") or 0) for u in unit_rows),
        "finding_count": len(finding_rows),
        "status_counts": status_counts,
    }
    manifest_out["status"] = derive_campaign_status(status_counts)

    snapshot_path = os.path.join(root, SNAPSHOT_FILENAME)
    units_path = os.path.join(root, REVIEW_UNITS_FILENAME)
    findings_path = os.path.join(root, FINDINGS_FILENAME)

    atomic_write_json(snapshot_path, dict(snapshot), ensure_ascii=False, indent=2)
    atomic_write_jsonl(units_path, unit_rows, ensure_ascii=False)
    atomic_write_jsonl(findings_path, finding_rows, ensure_ascii=False)
    atomic_write_json(manifest_path, manifest_out, ensure_ascii=False, indent=2)

    paths = {
        "package_dir": root,
        "manifest": manifest_path,
        "snapshot": snapshot_path,
        "review_units": units_path,
        "findings": findings_path,
    }

    if write_report:
        report_path = os.path.join(root, REPORT_MD_FILENAME)
        atomic_write_text(
            report_path,
            format_campaign_report_markdown(manifest_out, unit_rows, finding_rows),
        )
        paths["report"] = report_path

    return paths


def load_json_file(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_jsonl_file(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise FinalReviewSchemaError(
                    f"invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def resolve_campaign_paths(
    target: str | os.PathLike[str] | None,
) -> dict[str, str]:
    """Resolve a package dir or manifest path into artifact paths."""
    if not target:
        raise FinalReviewError("campaign target path is required")
    path = os.path.abspath(os.fspath(target))
    if os.path.isdir(path):
        package_dir = path
        manifest_path = os.path.join(package_dir, MANIFEST_FILENAME)
    elif os.path.isfile(path):
        manifest_path = path
        package_dir = os.path.dirname(manifest_path)
    else:
        raise FinalReviewError(f"campaign path not found: {path}")

    if not os.path.isfile(manifest_path):
        raise FinalReviewError(f"manifest not found: {manifest_path}")

    return {
        "package_dir": package_dir,
        "manifest": manifest_path,
        "snapshot": os.path.join(package_dir, SNAPSHOT_FILENAME),
        "review_units": os.path.join(package_dir, REVIEW_UNITS_FILENAME),
        "findings": os.path.join(package_dir, FINDINGS_FILENAME),
        "report": os.path.join(package_dir, REPORT_MD_FILENAME),
    }


def load_campaign_package(target: str | os.PathLike[str]) -> dict[str, Any]:
    paths = resolve_campaign_paths(target)
    manifest = load_json_file(paths["manifest"])
    if not isinstance(manifest, dict):
        raise FinalReviewSchemaError("manifest must be a JSON object")
    mode = _as_optional_str(manifest.get("mode"))
    if mode and mode != MANIFEST_MODE_FINAL_REVIEW:
        raise FinalReviewSchemaError(
            f"expected mode={MANIFEST_MODE_FINAL_REVIEW!r}, got {mode!r}"
        )
    snapshot = {}
    if os.path.isfile(paths["snapshot"]):
        loaded = load_json_file(paths["snapshot"])
        if isinstance(loaded, dict):
            snapshot = loaded
    units = load_jsonl_file(paths["review_units"])
    findings = load_jsonl_file(paths["findings"])
    for unit in units:
        assert_failure_not_done(unit)

    manifest = dict(manifest)
    manifest["_manifest_path"] = paths["manifest"]
    manifest["_package_dir"] = paths["package_dir"]
    return {
        "paths": paths,
        "manifest": manifest,
        "snapshot": snapshot,
        "units": units,
        "findings": findings,
    }


def collect_campaign_status(
    target: str | os.PathLike[str] | None = None,
    *,
    package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate human/machine status for a campaign package."""
    if package is None:
        if target is None:
            raise FinalReviewError("target or package is required")
        package = load_campaign_package(target)
    manifest = dict(package.get("manifest") or {})
    units = list(package.get("units") or [])
    findings = list(package.get("findings") or [])
    snapshot = dict(package.get("snapshot") or {})
    status_counts = summarize_unit_statuses(units)
    campaign_status = derive_campaign_status(status_counts)

    finding_type_counts: dict[str, int] = {}
    for finding in findings:
        ftype = _as_optional_str(finding.get("finding_type")) or "unknown"
        finding_type_counts[ftype] = finding_type_counts.get(ftype, 0) + 1

    return {
        "mode": MANIFEST_MODE_FINAL_REVIEW,
        "report_only": True,
        "autofix": False,
        "status": campaign_status,
        "display_name": manifest.get("display_name") or "",
        "package_dir": manifest.get("_package_dir")
        or manifest.get("package_dir")
        or "",
        "manifest_path": manifest.get("_manifest_path") or "",
        "context_digest": snapshot.get("context_digest")
        or manifest.get("context_digest")
        or "",
        "snapshot_digest": snapshot.get("snapshot_digest")
        or manifest.get("snapshot_digest")
        or "",
        "prompt_schema_version": manifest.get("prompt_schema_version")
        or PROMPT_SCHEMA_VERSION,
        "model": manifest.get("model") or manifest.get("batch_model") or "",
        "unit_count": len(units),
        "item_count": sum(int(u.get("item_count") or 0) for u in units),
        "finding_count": len(findings),
        "status_counts": status_counts,
        "finding_type_counts": dict(sorted(finding_type_counts.items())),
        "readiness": manifest.get("readiness") or {},
        "scope": snapshot.get("scope") or manifest.get("scope") or {},
        "created_at": manifest.get("created_at") or "",
        "job_state": manifest.get("job_state") or "",
    }


def format_status_text(status: Mapping[str, Any]) -> str:
    lines = [
        "Final Review Campaign",
        f"  status: {status.get('status')}",
        f"  report_only: {status.get('report_only')}  autofix: {status.get('autofix')}",
        f"  display_name: {status.get('display_name') or '(none)'}",
        f"  package: {status.get('package_dir') or '(none)'}",
        f"  context_digest: {str(status.get('context_digest') or '')[:16]}…",
        f"  snapshot_digest: {str(status.get('snapshot_digest') or '')[:16]}…",
        f"  units: {status.get('unit_count')}  items: {status.get('item_count')}  "
        f"findings: {status.get('finding_count')}",
        f"  unit_status: {status.get('status_counts')}",
    ]
    if status.get("finding_type_counts"):
        lines.append(f"  finding_types: {status.get('finding_type_counts')}")
    readiness = status.get("readiness") or {}
    if readiness:
        lines.append(
            f"  readiness.pending_task_count: {readiness.get('pending_task_count', 0)}"
        )
    return "\n".join(lines)


def format_campaign_report_markdown(
    manifest: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
) -> str:
    status_counts = summarize_unit_statuses(units)
    campaign_status = derive_campaign_status(status_counts)
    lines = [
        "# Final Review Campaign Report",
        "",
        f"- **status**: `{campaign_status}`",
        f"- **report_only**: `{bool(manifest.get('report_only', True))}`",
        f"- **autofix**: `{bool(manifest.get('autofix', False))}`",
        f"- **display_name**: {manifest.get('display_name') or ''}",
        f"- **snapshot_digest**: `{manifest.get('snapshot_digest') or ''}`",
        f"- **prompt_schema_version**: `{manifest.get('prompt_schema_version') or ''}`",
        f"- **units**: {len(list(units))} · **findings**: {len(list(findings))}",
        f"- **unit_status**: `{status_counts}`",
        "",
        "## Note",
        "",
        "This campaign is report-only. Selected findings must be converted to revision",
        "candidates and applied through `preview-revisions` → `apply-revisions`.",
        "There is no autofix path that writes `.rpy` files.",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        for finding in findings:
            lines.append(
                f"- `{finding.get('finding_id')}` · **{finding.get('finding_type')}** "
                f"({finding.get('severity')}) · `{finding.get('identity_v2')}`"
            )
            reason = str(finding.get("reason") or "").strip()
            if reason:
                lines.append(f"  - reason: {reason}")
            suggested = str(finding.get("suggested_revision") or "").strip()
            if suggested:
                lines.append(f"  - suggested: {suggested}")
        lines.append("")
    else:
        lines.extend(
            [
                "## Findings",
                "",
                "_No findings recorded yet (campaign may still be pending / not executed)._ ",
                "",
            ]
        )
    failed = [u for u in units if _as_optional_str(u.get("status")) == STATUS_FAILED]
    if failed:
        lines.extend(["## Failed units", ""])
        for unit in failed:
            lines.append(
                f"- `{unit.get('unit_id')}` · {unit.get('file_rel_path')} · "
                f"{unit.get('error') or 'failed'}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def export_findings(
    target: str | os.PathLike[str],
    *,
    output_jsonl: str = "",
    output_markdown: str = "",
) -> dict[str, Any]:
    """Export findings (+ refresh report.md). Returns paths and counts."""
    package = load_campaign_package(target)
    paths = package["paths"]
    findings = [normalize_finding(f) for f in package.get("findings") or []]
    units = package.get("units") or []
    manifest = package.get("manifest") or {}

    jsonl_path = output_jsonl.strip() if output_jsonl else paths["findings"]
    md_path = output_markdown.strip() if output_markdown else paths.get("report") or os.path.join(
        paths["package_dir"], REPORT_MD_FILENAME
    )

    atomic_write_jsonl(jsonl_path, findings, ensure_ascii=False)
    atomic_write_text(md_path, format_campaign_report_markdown(manifest, units, findings))

    return {
        "jsonl_path": os.path.abspath(jsonl_path),
        "markdown_path": os.path.abspath(md_path),
        "finding_count": len(findings),
        "status": collect_campaign_status(package=package),
    }


def reevaluate_campaign_units(
    units: Sequence[Mapping[str, Any]],
    *,
    live_items_by_unit: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    context_digest: str = "",
    snapshot_digest: str = "",
    model: str = "",
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Recompute live digests and refresh unit statuses (stale detection).

    Prefer ``context_digest`` (shared context only). ``snapshot_digest`` is
    accepted as a legacy fallback when context is omitted.
    """
    shared_context = _as_optional_str(context_digest) or _as_optional_str(snapshot_digest)
    out: list[dict[str, Any]] = []
    for unit in units or []:
        row = dict(unit)
        unit_id = _as_optional_str(row.get("unit_id"))
        # Prefer live shared context; fall back to the unit's frozen context.
        unit_context = shared_context or _as_optional_str(row.get("context_digest"))
        if live_items_by_unit and unit_id in live_items_by_unit:
            live_items = [normalize_review_item(i) for i in live_items_by_unit[unit_id]]
            items_digest = digest_translation_items(live_items)
            item_ids = [i["id"] for i in live_items]
            live_digest = compute_unit_input_digest(
                item_ids=item_ids,
                items_digest=items_digest,
                context_digest=unit_context,
                model=model or row.get("model") or "",
                prompt_schema_version=prompt_schema_version
                or row.get("prompt_schema_version")
                or PROMPT_SCHEMA_VERSION,
                chunk_index=int(row.get("chunk_index") or 0),
                file_rel_path=str(row.get("file_rel_path") or ""),
            )
        else:
            # Recompute from stored items when present.
            stored_items = row.get("items") or []
            if stored_items:
                items_digest = digest_translation_items(stored_items)
                item_ids = [
                    _as_optional_str(i.get("id") if isinstance(i, Mapping) else "")
                    for i in stored_items
                ]
                live_digest = compute_unit_input_digest(
                    item_ids=item_ids,
                    items_digest=items_digest,
                    context_digest=unit_context,
                    model=model or row.get("model") or "",
                    prompt_schema_version=prompt_schema_version
                    or row.get("prompt_schema_version")
                    or PROMPT_SCHEMA_VERSION,
                    chunk_index=int(row.get("chunk_index") or 0),
                    file_rel_path=str(row.get("file_rel_path") or ""),
                )
            else:
                live_digest = _as_optional_str(row.get("input_digest"))

        new_status = reevaluate_unit_status(
            row, live_input_digest=live_digest, force=force
        )
        row["live_input_digest"] = live_digest
        row["status"] = new_status
        if new_status == STATUS_STALE:
            row["error"] = row.get("error") or "input_digest_mismatch"
        out.append(row)
    return out


_PACKAGE_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def suggest_package_name(project_slug: str = "", *, timestamp: str | None = None) -> str:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = _PACKAGE_NAME_SAFE_RE.sub("_", (project_slug or "project").strip()) or "project"
    return f"{ts}_{slug}_final_review"
