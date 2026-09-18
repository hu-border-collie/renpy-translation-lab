"""Sync execution for final-review campaign units (#431 S3).

The module keeps transport decisions outside ``final_review``: callers bind a
frozen ModelRoutingPlan / TaskRoute into ``generate`` and this module owns the
unit loop, at-least-once persistence and report-only ingest semantics.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import final_review as fr
import final_review_llm as fr_llm
from sync_model_backend import (
    SYNC_ERROR_CATEGORIES,
    sync_error_category,
    sync_error_summary,
)

EXECUTION_STRATEGY_SYNC = fr.EXECUTION_STRATEGY_SYNC
EXECUTION_STRATEGY_GEMINI_BATCH = fr.EXECUTION_STRATEGY_GEMINI_BATCH

# Categories that indicate the whole campaign cannot make progress; retrying
# every remaining unit would only multiply the same systemic failure.
FATAL_SYNC_CATEGORIES = frozenset({
    "authentication",
    "missing_dependency",
    "unsupported_capability",
})


class FinalReviewSyncError(fr.FinalReviewError):
    """Stable campaign-level refusal for sync execution."""


def build_sync_request_payload(
    unit: Mapping[str, Any],
    *,
    shared_context: Mapping[str, Any] | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
    thinking_level: str = "",
    model: str = "",
    structured_output_mode: str = "",
    safety_settings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one provider-neutral sync request payload for a review unit."""

    unit_id = str(unit.get("unit_id") or "").strip()
    if not unit_id:
        raise FinalReviewSyncError("unit_id is required to run a review unit")
    effective_model = str(model or unit.get("model") or "")
    config = fr_llm.build_generation_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_level=thinking_level,
        model=effective_model,
    )
    if structured_output_mode:
        config["structured_output_mode"] = str(structured_output_mode)
    payload: dict[str, Any] = {
        "contents": fr_llm.build_user_prompt(unit, shared_context=shared_context),
        "system_instruction": fr_llm.build_system_instruction(),
        "generation_config": config,
    }
    if safety_settings:
        payload["safety_settings"] = list(safety_settings)
    return payload


def _result_text(result: object) -> str:
    if isinstance(result, Mapping):
        text = result.get("response_text")
        if text:
            return str(text)
        payload = result.get("response")
        if payload is None:
            payload = result.get("response_payload")
        return str(fr_llm.extract_text_from_response_payload(payload) or "")
    return str(result or "")


def _result_field(result: object, key: str, fallback: str = "") -> str:
    if isinstance(result, Mapping):
        return str(result.get(key) or fallback or "")
    return str(fallback or "")


def _sync_row(unit_id: str, result: object, fallback_model: str) -> dict[str, Any]:
    text = _result_text(result)
    if not text.strip():
        return {
            "key": unit_id,
            "error": "sync_invalid_response: provider returned no response text",
            "model": _result_field(result, "model", fallback_model),
        }
    row: dict[str, Any] = {
        "key": unit_id,
        "response_text": text,
        "model": _result_field(result, "model", fallback_model),
    }
    if isinstance(result, Mapping):
        for key in (
            "provider",
            "response_payload",
            "usage_metadata",
            "finish_reason",
            "request_metadata",
        ):
            if result.get(key) not in (None, "", {}, ()):
                row[key] = result[key]
    return row


def _unit_error_category(unit: Mapping[str, Any]) -> str:
    code = str(unit.get("error") or "").split(":", 1)[0].strip()
    if code.startswith("sync_"):
        code = code[len("sync_"):]
    return code if code in SYNC_ERROR_CATEGORIES else "provider_error"


def _replace_unit(
    units: Sequence[Mapping[str, Any]],
    updated: Mapping[str, Any],
) -> list[dict[str, Any]]:
    unit_id = str(updated.get("unit_id") or "")
    replaced: list[dict[str, Any]] = []
    for unit in units:
        if str(unit.get("unit_id") or "") == unit_id:
            replaced.append(dict(updated))
        else:
            replaced.append(dict(unit))
    return replaced


def _call_progress(
    progress: Callable[[Mapping[str, Any]], None] | None,
    event: Mapping[str, Any],
) -> None:
    if progress is None:
        return
    try:
        progress(dict(event))
    except Exception:
        # Progress rendering must never break the campaign.
        pass


def _record_usage(
    usage_recorder: Callable[[Mapping[str, Any]], None] | None,
    *,
    unit_id: str,
    result: object,
) -> None:
    if usage_recorder is None:
        return
    try:
        usage_recorder({"unit_id": unit_id, "result": result})
    except Exception:
        # Ledger failures are best-effort by contract.
        pass


def run_sync_campaign(
    package_dir: str | os.PathLike[str],
    *,
    generate: Callable[[Mapping[str, Any]], object],
    shared_context: Mapping[str, Any] | None = None,
    live_context_digest: str = "",
    force: bool = False,
    limit: int = 0,
    dry_run: bool = False,
    fail_fast: bool = False,
    provider: str = "",
    model: str = "",
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
    thinking_level: str = "",
    structured_output_mode: str = "",
    safety_settings: Sequence[Mapping[str, Any]] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    usage_recorder: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute pending/stale/failed review units through *generate*.

    Each unit is persisted immediately after it finishes, so an interrupted
    process resumes from the campaign package instead of restarting the whole
    campaign. ``generate`` must perform exactly one logical provider call
    (its own retry contract may still apply) and return either a response
    mapping or response text.
    """

    package = fr.load_campaign_package(os.fspath(package_dir))
    package_root = str(package["paths"]["package_dir"])
    manifest = dict(package.get("manifest") or {})
    strategy = str(
        manifest.get("execution_strategy") or EXECUTION_STRATEGY_GEMINI_BATCH
    )
    if strategy != EXECUTION_STRATEGY_SYNC:
        raise FinalReviewSyncError(
            "final-review-run-sync only accepts sync campaigns; "
            f"this package uses execution_strategy={strategy!r}"
        )

    units = [dict(unit) for unit in (package.get("units") or [])]
    findings = [dict(finding) for finding in (package.get("findings") or [])]
    snapshot = dict(package.get("snapshot") or {})

    planned = fr_llm.plan_units_for_run(
        units,
        force=bool(force),
        live_context_digest=live_context_digest,
    )
    queued = list(planned["to_run"])
    effective_limit = int(limit or 0)
    if effective_limit < 0:
        raise FinalReviewSyncError("limit must be >= 0")
    to_run = queued[:effective_limit] if effective_limit else queued
    deferred_count = max(0, len(queued) - len(to_run))
    run_ids = [str(unit.get("unit_id") or "") for unit in to_run]
    skip_count = int(planned["skip_count"])

    if dry_run:
        status_counts = fr.summarize_unit_statuses(units)
        return {
            "status": "dry_run",
            "package_dir": package_root,
            "manifest_path": str(package["paths"]["manifest"]),
            "execution_strategy": strategy,
            "provider": str(provider or ""),
            "model": str(model or ""),
            "run_count": len(to_run),
            "skip_count": skip_count,
            "deferred_count": deferred_count,
            "done_delta": 0,
            "failed_delta": 0,
            "finding_count": len(findings),
            "to_run_unit_ids": run_ids,
            "campaign_status": {
                "status": fr.derive_campaign_status(status_counts),
                "status_counts": status_counts,
                "finding_count": len(findings),
            },
            "dry_run": True,
            "limit": effective_limit,
        }

    done_delta = 0
    failed_delta = 0
    aborted = False
    abort_category = ""
    abort_unit_id = ""
    effective_provider = str(provider or "")
    effective_model = str(model or "")
    for index, unit in enumerate(to_run, start=1):
        unit_id = str(unit.get("unit_id") or "")
        _call_progress(
            progress,
            {
                "event": "unit_start",
                "unit_id": unit_id,
                "index": index,
                "run_count": len(to_run),
            },
        )
        payload = build_sync_request_payload(
            unit,
            shared_context=shared_context,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
            model=effective_model or str(unit.get("model") or ""),
            structured_output_mode=structured_output_mode,
            safety_settings=safety_settings,
        )
        row: dict[str, Any]
        try:
            result = generate(payload)
        except Exception as exc:  # noqa: BLE001 - provider call failed for this unit
            row = {
                "key": unit_id,
                "error": f"sync_{sync_error_category(exc)}: {sync_error_summary(exc)}",
            }
        else:
            effective_provider = _result_field(
                result, "provider", effective_provider
            )
            effective_model = _result_field(result, "model", effective_model)
            row = _sync_row(unit_id, result, effective_model)
            _record_usage(usage_recorder, unit_id=unit_id, result=result)

        ingest = fr_llm.ingest_result_rows(
            [unit],
            [row],
            provider=effective_provider,
            model=effective_model or str(unit.get("model") or ""),
        )
        updated_unit = ingest["units"][0]
        units = _replace_unit(units, updated_unit)
        replaced = {unit_id} if unit_id else set()
        findings = fr_llm.merge_findings_preserve_selection(
            findings,
            ingest["findings"],
            replaced_unit_ids=replaced,
        )
        fr_llm.persist_campaign_state(
            package_root,
            manifest=manifest,
            snapshot=snapshot,
            units=units,
            findings=findings,
        )
        if str(updated_unit.get("status") or "") == fr.STATUS_DONE:
            done_delta += 1
            _call_progress(
                progress,
                {"event": "unit_done", "unit_id": unit_id, "index": index},
            )
        else:
            failed_delta += 1
            unit_category = _unit_error_category(updated_unit)
            _call_progress(
                progress,
                {
                    "event": "unit_failed",
                    "unit_id": unit_id,
                    "category": unit_category,
                    "index": index,
                    "run_count": len(to_run),
                },
            )
            if fail_fast or unit_category in FATAL_SYNC_CATEGORIES:
                aborted = True
                abort_category = unit_category
                abort_unit_id = unit_id
                break

    status = fr.collect_campaign_status(package_root)
    payload: dict[str, Any] = {
        "status": (
            "aborted"
            if aborted
            else (
                "failed"
                if failed_delta
                else ("completed" if done_delta else "no_work")
            )
        ),
        "package_dir": package_root,
        "manifest_path": str(package["paths"]["manifest"]),
        "execution_strategy": strategy,
        "provider": effective_provider,
        "model": effective_model,
        "run_count": len(to_run),
        "skip_count": skip_count,
        "deferred_count": deferred_count,
        "done_delta": done_delta,
        "failed_delta": failed_delta,
        "finding_count": int(status.get("finding_count") or 0),
        "to_run_unit_ids": run_ids,
        "campaign_status": status,
        "dry_run": False,
        "limit": effective_limit,
    }
    if aborted:
        payload["abort_category"] = abort_category
        payload["abort_unit_id"] = abort_unit_id
    return payload
