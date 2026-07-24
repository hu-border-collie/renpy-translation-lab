"""Final-review campaign execution: prompts, Batch requests, result ingest (#255 PR B).

Report-only: never writes ``.rpy``. Parses model findings into the campaign package
and updates unit status. Resume skips completed units whose ``input_digest`` is
unchanged; ``--force`` re-queues them. Parse/row failures mark units ``failed``
and never invent a clean ``done`` with zero findings.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Mapping, Sequence

from atomic_io import atomic_write_json, atomic_write_jsonl, atomic_write_text
from final_review import (
    FINDINGS_FILENAME,
    MANIFEST_FILENAME,
    PROMPT_SCHEMA_VERSION,
    REPORT_MD_FILENAME,
    REQUESTS_JSONL_FILENAME,
    REVIEW_UNITS_FILENAME,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_STALE,
    FinalReviewError,
    FinalReviewSchemaError,
    assert_failure_not_done,
    collect_campaign_status,
    derive_campaign_status,
    export_findings,
    format_campaign_report_markdown,
    load_campaign_package,
    mark_unit_done,
    mark_unit_failed,
    normalize_finding,
    reevaluate_campaign_units,
    should_skip_unit,
    summarize_unit_statuses,
    utc_now_iso,
)

RESULT_JSONL_FILENAME = "results.jsonl"

# Finding types accepted from the model (unknown → needs_confirmation via normalize_finding).
FINDING_TYPE_ENUM = [
    "omission",
    "mistranslation",
    "addition",
    "format",
    "terminology",
    "address",
    "style_drift",
    "needs_confirmation",
]

SEVERITY_ENUM = ["high", "medium", "low", "info"]


def build_system_instruction() -> str:
    return (
        "你是 Ren'Py 视觉小说中文译文的最终审校助手。根据给定的原文/当前译文对照，"
        "报告明确的问题；不确定时标记为 needs_confirmation，不要伪装成确定错误。\n"
        "规则：\n"
        "1) 只依据提供的条目与简要上下文，不要编造未给出的剧情或术语；\n"
        "2) 问题类型仅使用：omission（漏译）、mistranslation（误译）、addition（增译）、"
        "format（格式/占位符/标签）、terminology（术语）、address（人物称谓/代词）、"
        "style_drift（明显文体漂移）、needs_confirmation（需人工确认）；\n"
        "3) 无问题则返回空 findings 列表；\n"
        "4) 不要声称 fixed / applied / done；不要直接改写游戏脚本；\n"
        "5) 严格输出 JSON，符合给定 schema。"
    )


def build_user_prompt(unit: Mapping[str, Any]) -> str:
    items = list(unit.get("items") or [])
    lines = [
        f"Review unit: {unit.get('unit_id') or ''}",
        f"File: {unit.get('file_rel_path') or ''}",
        f"Chunk index: {unit.get('chunk_index') or 0}",
        f"Item count: {len(items)}",
        "",
        "对照条目（source → current_translation）：",
    ]
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("id") or item.get("identity_v2") or f"item-{index}")
        source = str(item.get("source") or item.get("text") or "")
        current = str(item.get("current_translation") or item.get("translation") or "")
        speaker = str(item.get("speaker_id") or "").strip()
        speaker_part = f" speaker={speaker}" if speaker else ""
        lines.append(f"[{index}] id={item_id}{speaker_part}")
        lines.append(f"  source: {source}")
        lines.append(f"  current_translation: {current}")
    lines.extend(
        [
            "",
            "请输出 JSON：{\"findings\":[...]}。",
            "每条 finding 必须包含 item_id（对应该条目 id）、finding_type、severity、"
            "reason；可选 evidence、suggested_revision。",
            "若无问题：{\"findings\":[]}。",
        ]
    )
    return "\n".join(lines)


def build_response_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string"},
                        "finding_type": {"type": "string", "enum": FINDING_TYPE_ENUM},
                        "severity": {"type": "string", "enum": SEVERITY_ENUM},
                        "evidence": {"type": "string"},
                        "reason": {"type": "string"},
                        "suggested_revision": {"type": "string"},
                    },
                    "required": ["item_id", "finding_type", "severity", "reason"],
                },
            }
        },
        "required": ["findings"],
    }


def build_generation_config(
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
    thinking_level: str = "",
    model: str = "",
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
        "response_mime_type": "application/json",
        "response_json_schema": build_response_json_schema(),
    }
    if thinking_level and str(model or "").startswith("gemini-3"):
        config["thinking_config"] = {"thinking_level": str(thinking_level).upper()}
    return config


def build_batch_request(
    unit: Mapping[str, Any],
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
    thinking_level: str = "",
    model: str = "",
    safety_settings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Gemini Batch JSONL row: ``{key, request}`` keyed by unit_id."""
    unit_id = str(unit.get("unit_id") or "").strip()
    if not unit_id:
        raise FinalReviewSchemaError("unit_id is required to build a batch request")
    request: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": build_system_instruction()}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": build_user_prompt(unit)}],
            }
        ],
        "generation_config": build_generation_config(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
            model=model or str(unit.get("model") or ""),
        ),
    }
    if safety_settings:
        request["safety_settings"] = list(safety_settings)
    return {"key": unit_id, "request": request}


def write_requests_jsonl(
    path: str | os.PathLike[str],
    units: Sequence[Mapping[str, Any]],
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
    thinking_level: str = "",
    model: str = "",
    safety_settings: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    rows = [
        build_batch_request(
            unit,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
            model=model,
            safety_settings=safety_settings,
        )
        for unit in units
    ]
    atomic_write_jsonl(path, rows, ensure_ascii=False)
    return len(rows)


def plan_units_for_run(
    units: Sequence[Mapping[str, Any]],
    *,
    force: bool = False,
    live_context_digest: str = "",
) -> dict[str, Any]:
    """Split units into skip / run lists for resume.

    Done units with matching digest are skipped unless *force*.
    Failed / stale / pending always run (failed re-queues).
    """
    to_run: list[dict[str, Any]] = []
    to_skip: list[dict[str, Any]] = []
    refreshed: list[dict[str, Any]] = []

    for unit in units or []:
        row = dict(unit)
        live = live_context_digest or str(row.get("context_digest") or "")
        # Recompute digest from stored items + live context when possible.
        items = row.get("items") or []
        if items and live:
            from final_review import compute_unit_input_digest, digest_translation_items

            items_digest = digest_translation_items(items)
            item_ids = [
                str(i.get("id") or "") if isinstance(i, Mapping) else "" for i in items
            ]
            live_digest = compute_unit_input_digest(
                item_ids=item_ids,
                items_digest=items_digest,
                context_digest=live,
                model=str(row.get("model") or ""),
                prompt_schema_version=str(
                    row.get("prompt_schema_version") or PROMPT_SCHEMA_VERSION
                ),
                chunk_index=int(row.get("chunk_index") or 0),
                file_rel_path=str(row.get("file_rel_path") or ""),
            )
        else:
            live_digest = str(row.get("input_digest") or "")

        if force:
            row["status"] = STATUS_PENDING
            row["error"] = ""
            row["live_input_digest"] = live_digest
            to_run.append(row)
            refreshed.append(row)
            continue

        if should_skip_unit(row, live_input_digest=live_digest, force=False):
            row["live_input_digest"] = live_digest
            to_skip.append(row)
            refreshed.append(row)
            continue

        # Stale done/failed → pending for re-run.
        stored = str(row.get("input_digest") or "")
        status = str(row.get("status") or STATUS_PENDING)
        if stored and live_digest and stored != live_digest:
            row["status"] = STATUS_STALE
            row["error"] = row.get("error") or "input_digest_mismatch"
        if status in {STATUS_DONE, STATUS_FAILED, STATUS_STALE, STATUS_RUNNING}:
            # Queue for re-run (except pure skip done handled above).
            if status == STATUS_DONE and stored and live_digest and stored == live_digest:
                # should_skip already handled; defensive
                to_skip.append(row)
                refreshed.append(row)
                continue
            row["status"] = STATUS_PENDING
            if status != STATUS_FAILED:
                row["error"] = ""
        row["live_input_digest"] = live_digest
        to_run.append(row)
        refreshed.append(row)

    return {
        "to_run": to_run,
        "to_skip": to_skip,
        "units": refreshed,
        "run_count": len(to_run),
        "skip_count": len(to_skip),
    }


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def parse_json_payload(text: str) -> Any:
    body = str(text or "").strip()
    if not body:
        raise FinalReviewSchemaError("empty model response")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        match = _JSON_FENCE_RE.search(body)
        if match:
            return json.loads(match.group(1).strip())
        # Try first object slice.
        start = body.find("{")
        end = body.rfind("}")
        if start >= 0 and end > start:
            return json.loads(body[start : end + 1])
        raise


def extract_text_from_response_payload(response_payload: Any) -> str:
    """Best-effort text extraction from Gemini-style response payloads."""
    if response_payload is None:
        return ""
    if isinstance(response_payload, str):
        return response_payload.strip()
    if not isinstance(response_payload, Mapping):
        return str(response_payload).strip()

    # Common shapes: {candidates:[{content:{parts:[{text:…}]}}]}, {text:…}, {response_text:…}
    for key in ("response_text", "text", "output_text"):
        value = response_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    candidates = response_payload.get("candidates")
    if isinstance(candidates, list):
        texts: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            content = candidate.get("content") or {}
            if not isinstance(content, Mapping):
                continue
            parts = content.get("parts") or []
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif isinstance(part, str):
                    texts.append(part)
        joined = "\n".join(t for t in texts if t).strip()
        if joined:
            return joined
    return ""


def parse_unit_findings(
    response_text: str,
    unit: Mapping[str, Any],
    *,
    provider: str = "",
    model: str = "",
) -> tuple[list[dict[str, Any]], str]:
    """Parse model JSON into normalized findings.

    Returns ``(findings, error)``. On success *error* is empty (findings may be []).
    On failure *findings* is empty and *error* explains why — caller must mark failed.
    """
    unit_id = str(unit.get("unit_id") or "")
    unit_digest = str(unit.get("input_digest") or "")
    item_by_id: dict[str, Mapping[str, Any]] = {}
    for item in unit.get("items") or []:
        if isinstance(item, Mapping):
            iid = str(item.get("id") or item.get("identity_v2") or "").strip()
            if iid:
                item_by_id[iid] = item

    try:
        payload = parse_json_payload(response_text)
    except Exception as exc:  # noqa: BLE001 — surface as unit failure
        return [], f"failed_to_parse_model_json: {exc}"

    if not isinstance(payload, Mapping):
        return [], "failed_to_parse_model_json: root is not an object"

    raw_findings = payload.get("findings")
    if raw_findings is None:
        # Allow alternate key "issues" as soft alias, else fail (not silent zero).
        raw_findings = payload.get("issues")
    if raw_findings is None:
        return [], "failed_to_parse_model_json: missing findings array"
    if not isinstance(raw_findings, list):
        return [], "failed_to_parse_model_json: findings is not an array"

    findings: list[dict[str, Any]] = []
    for raw in raw_findings:
        if not isinstance(raw, Mapping):
            return [], "failed_to_parse_model_json: finding entry is not an object"
        item_id = str(raw.get("item_id") or raw.get("id") or raw.get("identity_v2") or "").strip()
        item = item_by_id.get(item_id) if item_id else None
        record = {
            "identity_v2": item_id
            or (str(item.get("identity_v2") or item.get("id") or "") if item else ""),
            "file_rel_path": (item or {}).get("file_rel_path") or unit.get("file_rel_path") or "",
            "source": (item or {}).get("source") or raw.get("source") or "",
            "current_translation": (item or {}).get("current_translation")
            or raw.get("current_translation")
            or "",
            "finding_type": raw.get("finding_type") or raw.get("type") or "",
            "severity": raw.get("severity") or "medium",
            "evidence": raw.get("evidence") or "",
            "reason": raw.get("reason") or raw.get("detail") or "",
            "suggested_revision": raw.get("suggested_revision") or raw.get("suggestion") or "",
            # Strip model-claimed applied/fixed in normalize_finding.
            "revision_state": raw.get("revision_state") or "none",
        }
        findings.append(
            normalize_finding(
                record,
                review_unit_id=unit_id,
                review_unit_digest=unit_digest,
                provider=provider,
                model=model or str(unit.get("model") or ""),
                prompt_schema_version=str(
                    unit.get("prompt_schema_version") or PROMPT_SCHEMA_VERSION
                ),
            )
        )
    return findings, ""


def apply_unit_result(
    unit: Mapping[str, Any],
    *,
    response_text: str = "",
    row_error: Any = None,
    provider: str = "",
    model: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Update one unit from a result row. Returns (unit, findings)."""
    if row_error:
        failed = mark_unit_failed(unit, f"row_error: {row_error}")
        assert_failure_not_done(failed)
        return failed, []

    text = str(response_text or "").strip()
    if not text:
        failed = mark_unit_failed(unit, "missing_response_text")
        assert_failure_not_done(failed)
        return failed, []

    findings, error = parse_unit_findings(
        text, unit, provider=provider, model=model
    )
    if error:
        failed = mark_unit_failed(unit, error)
        assert_failure_not_done(failed)
        return failed, []

    done = mark_unit_done(unit, finding_count=len(findings))
    return done, findings


def ingest_result_rows(
    units: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    provider: str = "",
    model: str = "",
    extract_text: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    """Merge Batch/sync result rows into units and aggregate findings.

    Units with no matching result row that were expected to run remain pending
    (or become failed if they were running). Already-done skipped units stay done.
    """
    extract = extract_text or extract_text_from_response_payload
    unit_map = {str(u.get("unit_id") or ""): dict(u) for u in units if u.get("unit_id")}
    findings_all: list[dict[str, Any]] = []
    processed: set[str] = set()
    summary = {
        "result_rows": 0,
        "processed_units": 0,
        "done_units": 0,
        "failed_units": 0,
        "finding_count": 0,
        "missing_units": 0,
        "unknown_keys": 0,
        "reason_counts": {},
    }

    def bump(reason: str) -> None:
        counts = summary["reason_counts"]
        counts[reason] = int(counts.get(reason) or 0) + 1

    for row in result_rows or []:
        if not isinstance(row, Mapping):
            bump("invalid_result_row")
            continue
        summary["result_rows"] += 1
        key = str(row.get("key") or row.get("unit_id") or "").strip()
        if not key or key not in unit_map:
            summary["unknown_keys"] += 1
            bump("unknown_chunk_key")
            continue
        processed.add(key)
        unit = unit_map[key]
        if row.get("error"):
            updated, findings = apply_unit_result(
                unit, row_error=row.get("error"), provider=provider, model=model
            )
        else:
            response_payload = row.get("response", row.get("response_payload", {}))
            # Allow pre-extracted text for tests.
            text = str(row.get("response_text") or "").strip()
            if not text:
                text = extract(response_payload)
            updated, findings = apply_unit_result(
                unit,
                response_text=text,
                provider=provider,
                model=model or str(row.get("model") or unit.get("model") or ""),
            )
        unit_map[key] = updated
        summary["processed_units"] += 1
        if updated.get("status") == STATUS_DONE:
            summary["done_units"] += 1
            findings_all.extend(findings)
            summary["finding_count"] += len(findings)
        elif updated.get("status") == STATUS_FAILED:
            summary["failed_units"] += 1
            bump(str(updated.get("error") or "failed")[:80])

    # Units that were running but missing from results → failed (not silent done).
    for unit_id, unit in list(unit_map.items()):
        if unit_id in processed:
            continue
        status = str(unit.get("status") or "")
        if status == STATUS_RUNNING:
            unit_map[unit_id] = mark_unit_failed(unit, "missing_result_row")
            summary["missing_units"] += 1
            summary["failed_units"] += 1
            bump("missing_result_row")

    ordered = []
    for unit in units:
        uid = str(unit.get("unit_id") or "")
        ordered.append(unit_map.get(uid, dict(unit)))

    for unit in ordered:
        assert_failure_not_done(unit)

    status_counts = summarize_unit_statuses(ordered)
    return {
        "units": ordered,
        "findings": findings_all,
        "summary": {
            **summary,
            "status_counts": status_counts,
            "campaign_status": derive_campaign_status(status_counts),
        },
    }


def load_result_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        raise FinalReviewError(f"result JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise FinalReviewSchemaError(
                    f"invalid result JSONL at {path}:{line_no}: {exc}"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def persist_campaign_state(
    package_dir: str,
    *,
    manifest: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None = None,
    units: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Rewrite units/findings/manifest/report after ingest or resume."""
    root = os.path.abspath(package_dir)
    for unit in units:
        assert_failure_not_done(unit)

    status_counts = summarize_unit_statuses(units)
    campaign_status = derive_campaign_status(status_counts)
    finding_rows = [normalize_finding(f) for f in findings]

    manifest_out = dict(manifest)
    manifest_out["status"] = campaign_status
    manifest_out["package_dir"] = root
    manifest_out["_package_dir"] = root
    manifest_path = os.path.join(root, MANIFEST_FILENAME)
    manifest_out["_manifest_path"] = manifest_path
    manifest_out["summary"] = {
        **dict(manifest_out.get("summary") or {}),
        "unit_count": len(list(units)),
        "item_count": sum(int(u.get("item_count") or 0) for u in units),
        "finding_count": len(finding_rows),
        "status_counts": status_counts,
    }
    manifest_out["last_ingest_at"] = utc_now_iso()

    units_path = os.path.join(root, REVIEW_UNITS_FILENAME)
    findings_path = os.path.join(root, FINDINGS_FILENAME)
    report_path = os.path.join(root, REPORT_MD_FILENAME)

    atomic_write_jsonl(units_path, list(units), ensure_ascii=False)
    atomic_write_jsonl(findings_path, finding_rows, ensure_ascii=False)
    atomic_write_text(
        report_path,
        format_campaign_report_markdown(manifest_out, units, finding_rows),
    )
    if snapshot is not None:
        from final_review import SNAPSHOT_FILENAME

        atomic_write_json(
            os.path.join(root, SNAPSHOT_FILENAME),
            dict(snapshot),
            ensure_ascii=False,
            indent=2,
        )
    atomic_write_json(manifest_path, manifest_out, ensure_ascii=False, indent=2)

    return {
        "package_dir": root,
        "manifest": manifest_path,
        "review_units": units_path,
        "findings": findings_path,
        "report": report_path,
    }


def merge_findings_preserve_selection(
    existing: Sequence[Mapping[str, Any]],
    new_findings: Sequence[Mapping[str, Any]],
    *,
    replaced_unit_ids: set[str],
) -> list[dict[str, Any]]:
    """Keep findings from units not re-run; replace findings for re-ingested units."""
    kept = [
        normalize_finding(f)
        for f in existing
        if str(f.get("review_unit_id") or "") not in replaced_unit_ids
    ]
    kept.extend(normalize_finding(f) for f in new_findings)
    return kept


def prepare_resume_requests(
    package_dir: str,
    *,
    force: bool = False,
    live_context_digest: str = "",
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
    thinking_level: str = "",
    model: str = "",
    safety_settings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Plan resume, rewrite requests.jsonl for units to run, update unit statuses."""
    package = load_campaign_package(package_dir)
    manifest = dict(package["manifest"])
    units = list(package["units"])
    findings = list(package["findings"])
    snapshot = dict(package.get("snapshot") or {})

    context = live_context_digest or str(
        snapshot.get("context_digest") or manifest.get("context_digest") or ""
    )
    plan = plan_units_for_run(units, force=force, live_context_digest=context)
    to_run = plan["to_run"]

    # Mark running for units about to be submitted.
    updated_units: list[dict[str, Any]] = []
    run_ids = {str(u.get("unit_id") or "") for u in to_run}
    for unit in plan["units"]:
        row = dict(unit)
        if str(row.get("unit_id") or "") in run_ids:
            row["status"] = STATUS_RUNNING if to_run else STATUS_PENDING
            if force:
                row["error"] = ""
        updated_units.append(row)

    requests_path = os.path.join(package_dir, REQUESTS_JSONL_FILENAME)
    request_count = write_requests_jsonl(
        requests_path,
        to_run,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_level=thinking_level,
        model=model or str(manifest.get("model") or ""),
        safety_settings=safety_settings,
    )

    manifest["input_jsonl_path"] = requests_path
    manifest["job_state"] = "LOCAL_ONLY" if request_count else "NO_PENDING_UNITS"
    manifest["job_name"] = ""
    manifest["result_jsonl_path"] = ""
    manifest["last_resume_at"] = utc_now_iso()
    manifest["last_resume"] = {
        "force": bool(force),
        "run_count": request_count,
        "skip_count": plan["skip_count"],
    }

    paths = persist_campaign_state(
        package_dir,
        manifest=manifest,
        snapshot=snapshot,
        units=updated_units,
        findings=findings,
    )
    return {
        "paths": paths,
        "run_count": request_count,
        "skip_count": plan["skip_count"],
        "to_run_unit_ids": [str(u.get("unit_id") or "") for u in to_run],
        "status": collect_campaign_status(
            package={
                "manifest": {**manifest, **{"_package_dir": package_dir}},
                "units": updated_units,
                "findings": findings,
                "snapshot": snapshot,
            }
        ),
    }


def ingest_results_into_package(
    package_dir: str,
    *,
    result_path: str = "",
    provider: str = "",
    model: str = "",
    extract_text: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    """Load result JSONL, update units/findings, persist package."""
    package = load_campaign_package(package_dir)
    manifest = dict(package["manifest"])
    units = list(package["units"])
    existing_findings = list(package["findings"])
    snapshot = dict(package.get("snapshot") or {})

    resolved_result = result_path.strip() if result_path else ""
    if not resolved_result:
        candidate = manifest.get("result_jsonl_path") or ""
        if candidate and not os.path.isabs(str(candidate)):
            candidate = os.path.join(package_dir, str(candidate))
        if candidate and os.path.isfile(str(candidate)):
            resolved_result = str(candidate)
        else:
            fallback = os.path.join(package_dir, RESULT_JSONL_FILENAME)
            if os.path.isfile(fallback):
                resolved_result = fallback
    if not resolved_result:
        raise FinalReviewError(
            "result JSONL not found; pass --result or run download first"
        )

    rows = load_result_jsonl(resolved_result)
    # Only replace findings for units present in this result set.
    result_keys = {
        str(r.get("key") or r.get("unit_id") or "")
        for r in rows
        if isinstance(r, Mapping)
    }
    # Also treat running units as candidates for replacement on missing-row fail.
    for unit in units:
        if str(unit.get("status") or "") == STATUS_RUNNING:
            result_keys.add(str(unit.get("unit_id") or ""))

    ingested = ingest_result_rows(
        units,
        rows,
        provider=provider,
        model=model or str(manifest.get("model") or ""),
        extract_text=extract_text,
    )
    merged_findings = merge_findings_preserve_selection(
        existing_findings,
        ingested["findings"],
        replaced_unit_ids={k for k in result_keys if k},
    )

    manifest["result_jsonl_path"] = resolved_result
    manifest["job_state"] = "RESULTS_INGESTED"
    paths = persist_campaign_state(
        package_dir,
        manifest=manifest,
        snapshot=snapshot,
        units=ingested["units"],
        findings=merged_findings,
    )
    return {
        "paths": paths,
        "summary": {
            **ingested["summary"],
            "finding_count": len(merged_findings),
        },
        "status": collect_campaign_status(
            package={
                "manifest": {**manifest, "_package_dir": package_dir},
                "units": ingested["units"],
                "findings": merged_findings,
                "snapshot": snapshot,
            }
        ),
    }


def run_units_with_generate(
    units: Sequence[Mapping[str, Any]],
    generate: Callable[[str, str], str],
    *,
    force: bool = False,
    live_context_digest: str = "",
    provider: str = "mock",
    model: str = "",
) -> dict[str, Any]:
    """Sync helper for tests: *generate(system, user) -> response_text*."""
    plan = plan_units_for_run(
        units, force=force, live_context_digest=live_context_digest
    )
    result_rows: list[dict[str, Any]] = []
    for unit in plan["to_run"]:
        system = build_system_instruction()
        user = build_user_prompt(unit)
        try:
            text = generate(system, user)
            result_rows.append(
                {
                    "key": unit.get("unit_id"),
                    "response_text": text,
                    "model": model or unit.get("model") or "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            result_rows.append({"key": unit.get("unit_id"), "error": str(exc)})

    # Merge skip units as-is with run results.
    return ingest_result_rows(
        plan["units"],
        result_rows,
        provider=provider,
        model=model,
        extract_text=lambda payload: str(payload or ""),
    )
