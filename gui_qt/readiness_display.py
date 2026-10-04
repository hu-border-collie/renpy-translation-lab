"""Read-only projections of existing doctor/preflight results for GUI pages.

These helpers never resolve a provider, scan scripts, or grant task/writeback
permission. The host owns freshness and the existing services own action gates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import doctor_recommendations as doctor_rec

from .doctor_report import DoctorSummary, doctor_report_to_parsed
from .user_copy import (
    READINESS_COPY,
    format_doctor_recommendation_fact,
    findings_require_attention,
    recommendation_requires_attention,
)


@dataclass(frozen=True)
class DoctorOverview:
    """Priority lines and existing settings destinations; no new gate state."""

    message: str
    counts: str
    attention: tuple[str, ...]
    details: tuple[str, ...]
    settings: tuple[str, ...]


def _count(value: object) -> str:
    return str(value) if type(value) is int and value >= 0 else READINESS_COPY["unknown"]


def count_line(report: Mapping[str, object] | None, *, preflight: bool = False) -> str:
    """Render only supplied nonnegative counters; missing is never zero."""
    data = report or {}
    counts = data.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}
    project = data.get("project")
    project = project if isinstance(project, Mapping) else {}
    return READINESS_COPY["counts"].format(
        pending=_count(
            counts.get("pending_items") if preflight else data.get("pending_task_count")
        ),
        pending_files=_count(
            counts.get("files_with_pending") if preflight else data.get("pending_file_count")
        ),
        files=_count(project.get("file_count") if preflight else counts.get("rpy_files")),
    )


def doctor_overview(
    summary: DoctorSummary,
    report: dict | None,
    *,
    completed: bool,
    api_key_count: int | None = None,
) -> DoctorOverview:
    """Prioritize existing required recommendations and retain every detail."""
    details = list(dict.fromkeys([summary.message, *summary.facts, *(summary.detail_facts or [])]))
    attention: list[str] = []
    settings: list[str] = []
    if completed and report is not None:
        parsed = doctor_report_to_parsed(report)
        for rec in parsed.get("recommendations", []):
            code = doctor_rec.normalize_doctor_recommendation(rec)["code"]
            if recommendation_requires_attention([code]):
                attention.append(format_doctor_recommendation_fact(rec))
            destination = {
                doctor_rec.ENABLE_PREPARE: "project",
                doctor_rec.INSTALL_SDK_GENERATE_TEMPLATE: "project",
                doctor_rec.CONFIGURE_PROJECT_ANALYSIS_MODEL: "profiles",
                doctor_rec.CONFIGURE_PROJECT_ANALYSIS_API: "api_keys",
                doctor_rec.ENABLE_RAG_FOR_CONSISTENCY: "context",
                doctor_rec.ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT: "context",
            }.get(code)
            if destination:
                settings.append(destination)
        if api_key_count == 0:
            attention.append("建议：在「设置 · 密钥」配置翻译凭据。")
            settings.append("api_keys")
        # Findings are warnings, not a second blocker classifier.
        if summary.findings:
            raw_warnings = report.get("warnings") or []
            attention.extend(
                finding
                for finding in summary.findings
                if finding not in raw_warnings and findings_require_attention([finding])
            )
            attention.append(f"注意事项：{len(summary.findings)} 项（完整内容见详情）。")
            if any(
                str(warning).startswith("Model routing preflight [") for warning in raw_warnings
            ):
                settings.append("profiles")
            if any(
                str(warning).startswith("Invalid tl_subdir / TL_DIR boundary:")
                for warning in raw_warnings
            ):
                settings.append("project")
        message = READINESS_COPY.get(summary.status, summary.message)
        if summary.mode == "can_generate_template":
            message = summary.message
        if summary.status == "blocked":
            attention.insert(0, summary.message)
        counts = count_line(report)
    else:
        message = (
            READINESS_COPY.get(summary.status, READINESS_COPY["idle"])
            if summary.status in {"idle", "running", "stale"}
            else READINESS_COPY["idle"]
        )
        if summary.status == "blocked":
            message = READINESS_COPY["failed"]
        counts = count_line(None)
    return DoctorOverview(
        message,
        counts,
        tuple(dict.fromkeys(attention)),
        tuple(details),
        tuple(dict.fromkeys(settings)),
    )


def preflight_line(status: str, payload: Mapping[str, object] | None = None) -> str:
    """Use the service's status; no inference from risk or quality counters."""
    line = READINESS_COPY.get("preflight_" + status, READINESS_COPY["preflight_failed"])
    if payload is not None and status in {"ready", "warning", "blocked"}:
        line += "\n" + count_line(payload, preflight=True)
        risks = [risk for risk in payload.get("risks", ()) if isinstance(risk, Mapping)]
        if risks:
            line += f"\n注意事项：{len(risks)} 项（启动确认与运行日志中可查看完整内容）。"
    return line
