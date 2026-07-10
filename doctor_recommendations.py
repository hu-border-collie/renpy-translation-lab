"""Stable doctor recommendation and workflow-state codes shared by CLI and GUI."""

from __future__ import annotations

import json
from typing import Any

SWITCH_TO_WORK = "switch_to_work"
BOOTSTRAP_WORK = "bootstrap_work"
GENERATE_TEMPLATE = "generate_template"
INSTALL_SDK_GENERATE_TEMPLATE = "install_sdk_generate_template"
ENABLE_PREPARE = "enable_prepare"
BOOTSTRAP_SOURCE_INDEX = "bootstrap_source_index"
BOOTSTRAP_SOURCE_INDEX_INCOMPLETE = "bootstrap_source_index_incomplete"
BOOTSTRAP_RAG = "bootstrap_rag"
BOOTSTRAP_RAG_OR_WARM_ON_BUILD = "bootstrap_rag_or_warm_on_build"
ENABLE_RAG_FOR_CONSISTENCY = "enable_rag_for_consistency"
SUBSTANTIALLY_COMPLETE = "substantially_complete"
ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT = "enable_source_index_for_new_project"
START_INCREMENTAL_BATCH = "start_incremental_batch"
START_PENDING_BATCH = "start_pending_batch"
NO_PENDING_LINES = "no_pending_lines"
UNKNOWN = "unknown"

ALL_CODES = frozenset(
    {
        SWITCH_TO_WORK,
        BOOTSTRAP_WORK,
        GENERATE_TEMPLATE,
        INSTALL_SDK_GENERATE_TEMPLATE,
        ENABLE_PREPARE,
        BOOTSTRAP_SOURCE_INDEX,
        BOOTSTRAP_SOURCE_INDEX_INCOMPLETE,
        BOOTSTRAP_RAG,
        BOOTSTRAP_RAG_OR_WARM_ON_BUILD,
        ENABLE_RAG_FOR_CONSISTENCY,
        SUBSTANTIALLY_COMPLETE,
        ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT,
        START_INCREMENTAL_BATCH,
        START_PENDING_BATCH,
        NO_PENDING_LINES,
    }
)

WORKFLOW_STATE_CODES = frozenset(
    {
        SUBSTANTIALLY_COMPLETE,
        START_INCREMENTAL_BATCH,
        START_PENDING_BATCH,
        NO_PENDING_LINES,
    }
)

# Optional quality tips: may coexist with workflow_state and must not block "ready".
OPTIONAL_RECOMMENDATION_CODES = frozenset(
    {
        BOOTSTRAP_RAG_OR_WARM_ON_BUILD,
        ENABLE_RAG_FOR_CONSISTENCY,
        ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT,
    }
)

_BOOTSTRAP_WORK_MESSAGE = (
    "work directory is missing or empty and original/game exists; "
    "run: python gemini_translate_batch.py bootstrap-work "
    "(copies original/game into work/game without generating TL)."
)
_GENERATE_TEMPLATE_MESSAGE = (
    "Missing translation files; run: python gemini_translate_batch.py generate-template "
    "(runs prepare only: unpack RPA if needed, generate tl template)."
)
_INSTALL_SDK_MESSAGE = (
    "Install Ren'Py SDK or set prepare.renpy_sdk_dir, then run: "
    "python gemini_translate_batch.py generate-template."
)
_ENABLE_PREPARE_MESSAGE = (
    "prepare is disabled; enable prepare.enabled in translator_config.json, then run build."
)
_BOOTSTRAP_SOURCE_INDEX_MESSAGE = (
    "Source index is enabled but not built; run bootstrap-source-index."
)
_BOOTSTRAP_SOURCE_INDEX_INCOMPLETE_MESSAGE = (
    "Source index bootstrap is incomplete; run bootstrap-source-index."
)
_BOOTSTRAP_RAG_MESSAGE = (
    "RAG store is enabled but empty; run bootstrap-rag before batch translation."
)
_BOOTSTRAP_RAG_OR_WARM_MESSAGE = (
    "RAG store is empty; run bootstrap-rag before batch translation, "
    "or start batch translation to warm the store automatically on build."
)
_ENABLE_RAG_MESSAGE = (
    "Existing translations detected with RAG disabled; enable RAG and run bootstrap-rag "
    "for better terminology consistency, then start batch translation."
)
_SUBSTANTIALLY_COMPLETE_MESSAGE = (
    "Project is substantially complete; remaining pending lines are minor. "
    "Batch translation and RAG bootstrap are optional."
)
_ENABLE_SOURCE_INDEX_MESSAGE = (
    "Source index is disabled; enable it and run bootstrap-source-index "
    "for better story context on a new translation project."
)
_START_INCREMENTAL_MESSAGE = (
    "Incremental translation is ready; start batch translation when API keys are configured."
)
_START_PENDING_MESSAGE = (
    "Pending translation lines are ready; start batch translation when API keys are configured."
)
_NO_PENDING_MESSAGE = (
    "No pending translation lines detected; review TL files or refresh templates "
    "before starting a new batch."
)

_LEGACY_EXACT_MESSAGES: dict[str, str] = {
    _BOOTSTRAP_WORK_MESSAGE: BOOTSTRAP_WORK,
    _GENERATE_TEMPLATE_MESSAGE: GENERATE_TEMPLATE,
    _INSTALL_SDK_MESSAGE: INSTALL_SDK_GENERATE_TEMPLATE,
    _ENABLE_PREPARE_MESSAGE: ENABLE_PREPARE,
    _BOOTSTRAP_SOURCE_INDEX_MESSAGE: BOOTSTRAP_SOURCE_INDEX,
    _BOOTSTRAP_SOURCE_INDEX_INCOMPLETE_MESSAGE: BOOTSTRAP_SOURCE_INDEX_INCOMPLETE,
    _BOOTSTRAP_RAG_MESSAGE: BOOTSTRAP_RAG,
    _BOOTSTRAP_RAG_OR_WARM_MESSAGE: BOOTSTRAP_RAG_OR_WARM_ON_BUILD,
    _ENABLE_RAG_MESSAGE: ENABLE_RAG_FOR_CONSISTENCY,
    _SUBSTANTIALLY_COMPLETE_MESSAGE: SUBSTANTIALLY_COMPLETE,
    _ENABLE_SOURCE_INDEX_MESSAGE: ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT,
    _START_INCREMENTAL_MESSAGE: START_INCREMENTAL_BATCH,
    _START_PENDING_MESSAGE: START_PENDING_BATCH,
    _NO_PENDING_MESSAGE: NO_PENDING_LINES,
}

_DETAIL_MESSAGES: dict[str, str] = {
    BOOTSTRAP_WORK: _BOOTSTRAP_WORK_MESSAGE,
    GENERATE_TEMPLATE: _GENERATE_TEMPLATE_MESSAGE,
    INSTALL_SDK_GENERATE_TEMPLATE: _INSTALL_SDK_MESSAGE,
    ENABLE_PREPARE: _ENABLE_PREPARE_MESSAGE,
    BOOTSTRAP_SOURCE_INDEX: _BOOTSTRAP_SOURCE_INDEX_MESSAGE,
    BOOTSTRAP_SOURCE_INDEX_INCOMPLETE: _BOOTSTRAP_SOURCE_INDEX_INCOMPLETE_MESSAGE,
    BOOTSTRAP_RAG: _BOOTSTRAP_RAG_MESSAGE,
    BOOTSTRAP_RAG_OR_WARM_ON_BUILD: _BOOTSTRAP_RAG_OR_WARM_MESSAGE,
    ENABLE_RAG_FOR_CONSISTENCY: _ENABLE_RAG_MESSAGE,
    SUBSTANTIALLY_COMPLETE: _SUBSTANTIALLY_COMPLETE_MESSAGE,
    ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT: _ENABLE_SOURCE_INDEX_MESSAGE,
    START_INCREMENTAL_BATCH: _START_INCREMENTAL_MESSAGE,
    START_PENDING_BATCH: _START_PENDING_MESSAGE,
    NO_PENDING_LINES: _NO_PENDING_MESSAGE,
}


def make_doctor_recommendation(code: str, **params: str) -> dict[str, Any]:
    if code not in ALL_CODES:
        raise ValueError(f"Unknown doctor recommendation code: {code}")
    return {"code": code, "params": dict(params)}


def doctor_recommendation_detail(rec: dict[str, Any]) -> str:
    code = str(rec.get("code") or "")
    params = rec.get("params") if isinstance(rec.get("params"), dict) else {}
    if code == SWITCH_TO_WORK:
        work_dir = str(params.get("work_dir") or "").strip()
        if work_dir:
            return f"game_root should use work directory; switch to {work_dir}"
        return "game_root should use work directory; switch to work directory"
    detail = rec.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return _DETAIL_MESSAGES.get(code, "")


def format_doctor_recommendation_cli_line(rec: dict[str, Any]) -> str:
    payload: dict[str, Any] = {"code": rec["code"]}
    params = rec.get("params") if isinstance(rec.get("params"), dict) else {}
    if params:
        payload["params"] = params
    detail = doctor_recommendation_detail(rec)
    if detail:
        payload["detail"] = detail
    return json.dumps(payload, ensure_ascii=False)


def parse_doctor_recommendation_cli_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict) and payload.get("code"):
            return normalize_doctor_recommendation(payload)
        return None
    return legacy_string_to_recommendation(text)


def legacy_string_to_recommendation(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if not normalized:
        raise ValueError("Doctor recommendation text is empty")

    switch_prefix = "game_root should use work directory; switch to "
    if normalized.startswith(switch_prefix):
        work_dir = normalized[len(switch_prefix) :].strip()
        return make_doctor_recommendation(SWITCH_TO_WORK, work_dir=work_dir)

    code = _LEGACY_EXACT_MESSAGES.get(normalized)
    if code:
        return make_doctor_recommendation(code)

    return {"code": UNKNOWN, "params": {}, "detail": normalized}


def normalize_doctor_recommendation(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        code = str(item.get("code") or "").strip()
        if not code:
            raise ValueError("Doctor recommendation dict is missing code")
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        rec = {"code": code, "params": dict(params)}
        detail = item.get("detail")
        if isinstance(detail, str) and detail.strip():
            rec["detail"] = detail.strip()
        return rec
    if isinstance(item, str):
        return legacy_string_to_recommendation(item)
    raise TypeError(f"Unsupported doctor recommendation value: {type(item)!r}")


def doctor_recommendation_codes(recommendations: list[Any]) -> list[str]:
    codes: list[str] = []
    for item in recommendations:
        try:
            rec = normalize_doctor_recommendation(item)
        except (TypeError, ValueError):
            continue
        code = str(rec.get("code") or "").strip()
        if code:
            codes.append(code)
    return codes


def recommendations_block_workflow_state(recommendations: list[Any]) -> bool:
    """True when any recommendation is required prep (not an optional tip)."""
    for code in doctor_recommendation_codes(recommendations):
        if code not in OPTIONAL_RECOMMENDATION_CODES:
            return True
    return False