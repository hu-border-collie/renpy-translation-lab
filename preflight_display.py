# -*- coding: utf-8 -*-
"""Shared CLI/GUI display lines for translate-preflight summaries (#488 S2).

The JSON payload already carries ``cost`` / ``coverage`` / ``quality_summary``.
This module only renders those keys; it does not estimate cost, rescan
coverage, or reread quality reports. Missing or untrusted fields are shown as
unknown / not available / stale — never as zero cost or a quality pass.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

COST_REASON_LABELS = {
    "pricing_unavailable": "当前模型没有可用价格表",
    "partial_pricing": "价格表只有单侧费率，无法给出完整区间",
    "input_unreadable": "当前计划的请求文本无法读取",
    "no_requests": "当前计划没有请求",
}

PRICING_SOURCE_LABELS = {
    "translator_config": "translator_config（项目价格表）",
    "defaults": "defaults（内置默认）",
}

COST_SCOPE_LABELS = {
    "current_plan": "本次初译计划",
}

QUALITY_REASON_LABELS = {
    "quality_source_not_supported_for_strategy": (
        "当前执行方式没有可验证的质量报告来源"
    ),
    "latest_manifest_missing": "没有 latest manifest",
    "manifest_unreadable": "manifest 无法读取",
    "mode_mismatch": "manifest 模式不匹配",
    "project_mismatch": "不属于当前项目",
    "plan_mismatch": "不属于当前翻译计划",
    "report_missing": "质量报告文件不存在",
    "report_digest_mismatch": "报告内容与记录不一致",
    "report_unreadable": "质量报告无法读取或解析",
}

COVERAGE_STATUS_LABELS = {
    "ready": "ready（已识别）",
    "attention": "attention（有注意项）",
    "block": "block（存在阻断项）",
    "unknown": "unknown（未知）",
}

COVERAGE_COMPLETION_LABELS = {
    "confirmed": "confirmed（已确认）",
    "unconfirmed": "unconfirmed（未确认）",
    "unknown": "unknown（未知）",
}

_SEVERITY_ORDER = ("high", "medium", "low", "info")


def format_preflight_summary_fields(payload: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return the cost / coverage / quality lines consumed by CLI and GUI.

    The returned keys match ``TRANSLATION_PREFLIGHT_COPY.body`` placeholders.
    Callers must not invent a second set of known/unknown rules.
    """

    data = _mapping(payload)
    return {
        "cost": format_preflight_cost_line(data.get("cost")),
        "coverage": format_preflight_coverage_line(data.get("coverage")),
        "quality": format_preflight_quality_line(data.get("quality_summary")),
    }


def format_preflight_summary_lines(payload: Mapping[str, Any] | None = None) -> list[str]:
    """Return the three summary lines in display order."""

    fields = format_preflight_summary_fields(payload)
    return [fields["cost"], fields["coverage"], fields["quality"]]


def format_preflight_cost_line(cost: object) -> str:
    """Render ``payload['cost']``. Unknown never looks like zero or free."""

    data = _mapping(cost)
    status = _text(data.get("status"))
    reason = _reason_label(data.get("reason"), COST_REASON_LABELS)
    if not data or status != "known":
        if not data:
            reason = "成本摘要缺失"
        elif status and status != "unknown":
            reason = reason or status
        elif not reason:
            reason = "状态 unknown"
        return (
            f"成本：无法估算（{reason}）。"
            "缺少可靠价格或请求证据，不会给出金额。"
        )

    min_text = _money(data.get("estimated_cost_min"))
    max_text = _money(data.get("estimated_cost_max"))
    if min_text is None or max_text is None:
        return (
            "成本：无法估算（金额字段不可用）。"
            "缺少可靠价格或请求证据，不会给出金额。"
        )

    model = _text(data.get("model")) or "(unknown)"
    strategy = _text(data.get("strategy")) or "unknown"
    currency = _text(data.get("currency")) or "USD"
    source = _reason_label(data.get("pricing_source"), PRICING_SOURCE_LABELS) or "unknown"
    scope = _reason_label(data.get("scope"), COST_SCOPE_LABELS) or "unknown"
    excluded = _join_excluded(data.get("excluded"))
    line = (
        f"成本：{scope}，{model} / {strategy}，约 {min_text}–{max_text} {currency}"
        f"（不含 final review / repair / embedding；计价来源：{source}"
    )
    if excluded:
        line += f"；未计入：{excluded}"
    return line + "）"


def format_preflight_coverage_line(coverage: object) -> str:
    """Render ``payload['coverage']`` without claiming scan completion."""

    data = _mapping(coverage)
    status = _text(data.get("status")) or "unknown"
    completion = _text(data.get("completion")) or "unknown"
    if not data or status == "unknown" or completion == "unknown":
        status_label = COVERAGE_STATUS_LABELS.get(status, status or "unknown")
        completion_label = COVERAGE_COMPLETION_LABELS.get(
            completion, completion or "unknown"
        )
        return (
            "文本覆盖：扫描证据缺失或不可用"
            f"（状态 {status_label}，完成度 {completion_label}），"
            "不能报告为覆盖完成。"
        )

    status_label = COVERAGE_STATUS_LABELS.get(status, status)
    completion_label = COVERAGE_COMPLETION_LABELS.get(completion, completion)
    parts = [f"文本覆盖：状态 {status_label}，完成度 {completion_label}"]
    counts_text = _classification_text(data.get("classification_counts"))
    if counts_text:
        parts.append(f"分类 {counts_text}")
    unknown_count = _int_or_none(data.get("unknown_count"))
    parse_error_count = _int_or_none(data.get("parse_error_count"))
    count_bits = []
    if unknown_count is not None:
        count_bits.append(f"unknown {unknown_count}")
    if parse_error_count is not None:
        count_bits.append(f"parse_error {parse_error_count}")
    if count_bits:
        parts.append("，".join(count_bits))
    return "；".join(parts)


def format_preflight_quality_line(quality: object) -> str:
    """Render ``payload['quality_summary']``. Only ``available`` shows findings."""

    data = _mapping(quality)
    status = _text(data.get("status"))
    reason = _reason_label(data.get("reason"), QUALITY_REASON_LABELS)
    if not data:
        return (
            "质量摘要：摘要缺失，当前没有可验证的已有质量报告。"
            "不能当作当前质量结论。"
        )
    if status == "available":
        finding_count = _int_or_none(data.get("finding_count"))
        if finding_count is None:
            return (
                "质量摘要：报告状态 available，但 finding 计数不可用。"
                "不能当作当前质量结论。"
            )
        line = f"质量摘要：已有匹配报告，finding {finding_count}"
        severities = _severity_text(data.get("severity_counts"))
        if severities:
            line += f"（{severities}）"
        return line + "。预检结论不是 check/apply 授权。"
    if status == "stale":
        detail = reason or "与当前项目或翻译计划不匹配"
        return (
            f"质量摘要：已有报告已过期（{detail}），"
            "不能当作当前质量结论。"
        )
    if status == "not_available":
        detail = reason or "没有可用 manifest 或报告"
        return (
            f"质量摘要：当前没有可验证的已有质量报告（{detail}）。"
            "不能当作当前质量结论。"
        )
    detail = reason or (status if status else "状态 unknown")
    return (
        f"质量摘要：已有报告无法读取或解析（{detail}），"
        "不能当作当前质量结论。"
    )


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _text(value: object) -> str:
    return str(value or "").strip()


def _reason_label(reason: object, labels: Mapping[str, str]) -> str:
    key = _text(reason)
    if not key:
        return ""
    return labels.get(key, key)


def _money(value: object) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _join_excluded(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    parts = [_text(item) for item in value]
    return "、".join(part for part in parts if part)


def _classification_text(counts: object) -> str:
    data = _mapping(counts)
    parts = []
    for key, raw in data.items():
        number = _int_or_none(raw)
        if number is None or not _text(key):
            continue
        parts.append(f"{_text(key)}={number}")
    return ", ".join(parts)


def _severity_text(counts: object) -> str:
    data = _mapping(counts)
    parts = []
    seen = set()
    for key in _SEVERITY_ORDER:
        number = _int_or_none(data.get(key))
        if number:
            parts.append(f"{key}={number}")
            seen.add(key)
    for key, raw in data.items():
        label = _text(key)
        if label in seen:
            continue
        number = _int_or_none(raw)
        if number:
            parts.append(f"{label}={number}")
            seen.add(label)
    return ", ".join(parts)
