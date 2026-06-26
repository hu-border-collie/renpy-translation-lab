"""Parse and summarize apply_failure_report.json for GUI diagnostics."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .check_failures_report import (
    CheckFailureItem,
    normalize_failure_entry,
    parse_check_failures_jsonl,
    reason_code_label,
    safe_preview,
)
from .diagnostics_context import join_directory_file, resolve_package_dir
from .user_copy import format_manifest_path_fact, safety_level_label

APPLY_REASON_CODE_LABELS = {
    "missing_check": "缺少检查结果",
    "stale_check_contract": "检查合约已过期",
    "stale_check_fingerprint": "检查结果已过期",
    "unsafe_check_status": "检查状态不安全",
    "unsafe_apply_recheck": "写回前复检失败",
    "unsafe_apply_revalidation": "源文件复检失败",
    "unclassified_failure": "未分类写回失败",
}

APPLY_REASON_SUGGESTIONS = {
    "missing_check": "请先重新检查当前任务，确认「可写回」后再尝试写回。",
    "stale_check_contract": "检查规则已更新；请重新检查并生成新的检查摘要。",
    "stale_check_fingerprint": "任务记录、结果包或源文件在检查后已有变化；请重新检查，不要重复写回。",
    "unsafe_check_status": "最近一次检查不是「可写回」；请先处理问题并重新检查。",
    "unsafe_apply_recheck": "写回前自动复检未通过；请查看失败条目，修复后重新检查。",
    "unsafe_apply_revalidation": "源文件与检查结果不一致；请修复源文件后重新生成任务并检查。",
    "unclassified_failure": "请查看错误摘要与诊断日志，确认下一步操作。",
}


@dataclass(frozen=True)
class ApplyFailureReportView:
    status: str
    heading: str
    message: str
    report_path: str
    failures_path: str
    reason_code: str
    facts: list[str]
    detail_lines: list[str]
    failure_item_count: int


def apply_reason_code_label(reason_code: str) -> str:
    """Return a Chinese label for an apply failure reason code."""
    text = str(reason_code or "").strip()
    return APPLY_REASON_CODE_LABELS.get(text, reason_code_label(text))


def apply_reason_suggestion(reason_code: str) -> str:
    """Return next-step guidance for an apply failure reason code."""
    text = str(reason_code or "").strip()
    return APPLY_REASON_SUGGESTIONS.get(
        text,
        APPLY_REASON_SUGGESTIONS["unclassified_failure"],
    )


def resolve_apply_failure_report_path(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
) -> str:
    """Resolve apply_failure_report.json from manifest metadata."""
    report_path = manifest.get("last_apply_failure_report_path")
    if isinstance(report_path, str) and report_path.strip():
        return report_path.strip()

    package_dir = resolve_package_dir(manifest_path, manifest)
    if package_dir:
        return join_directory_file(package_dir, "apply_failure_report.json")
    return ""


def resolve_apply_failures_jsonl_path(
    report_payload: dict[str, object],
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
) -> str:
    """Resolve failures.jsonl path referenced by an apply failure report."""
    failures_path = report_payload.get("failures_path")
    if isinstance(failures_path, str) and failures_path.strip():
        return failures_path.strip()

    package_dir = resolve_package_dir(manifest_path, manifest)
    if package_dir:
        return join_directory_file(package_dir, "failures.jsonl")
    return ""


def parse_apply_failure_report_payload(payload: dict[str, object]) -> dict[str, object]:
    """Normalize fields from apply_failure_report.json."""
    reason_code = payload.get("reason_code")
    reason_text = reason_code.strip() if isinstance(reason_code, str) else ""
    if not reason_text:
        reason_text = "unclassified_failure"

    error = str(payload.get("error") or "").strip()
    last_check_safety = payload.get("last_check_safety_level")
    safety_text = (
        last_check_safety.strip().lower()
        if isinstance(last_check_safety, str) and last_check_safety.strip()
        else ""
    )

    failure_count = payload.get("failure_count")
    count = failure_count if isinstance(failure_count, int) else 0

    summary = payload.get("summary")
    summary_dict = summary if isinstance(summary, dict) else {}

    fingerprint = payload.get("current_check_fingerprint")
    fingerprint_dict = fingerprint if isinstance(fingerprint, dict) else {}

    return {
        "reason_code": reason_text,
        "error": error,
        "last_check_safety_level": safety_text,
        "failure_count": count,
        "summary": summary_dict,
        "current_check_fingerprint": fingerprint_dict,
        "timestamp": str(payload.get("timestamp") or "").strip(),
        "last_check_at": str(payload.get("last_check_at") or "").strip(),
        "manifest_path": str(payload.get("manifest_path") or "").strip(),
        "failures_path": str(payload.get("failures_path") or "").strip(),
    }


def _default_path_exists(path: str) -> bool:
    try:
        return Path(path).exists()
    except OSError:
        return False


def _default_read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _format_fingerprint_facts(fingerprint: dict[str, object]) -> list[str]:
    facts: list[str] = []
    for key, label in (
        ("pending_files", "待写回文件数"),
        ("pending_lines", "待写回行数"),
        ("failure_items", "失败项"),
    ):
        value = fingerprint.get(key)
        if isinstance(value, int):
            facts.append(f"当前指纹 {label}：{value}")
    fingerprint_id = fingerprint.get("id")
    if isinstance(fingerprint_id, str) and fingerprint_id.strip():
        facts.append(f"当前检查指纹：{fingerprint_id.strip()}")
    return facts


def _format_summary_facts(summary: dict[str, object]) -> list[str]:
    facts: list[str] = []
    safety = summary.get("safety_level")
    if isinstance(safety, str) and safety.strip():
        facts.append(f"复检结果：{safety_level_label(safety.strip().lower())}")
    for key, label in (
        ("pending_files", "待写回文件数"),
        ("pending_lines", "待写回行数"),
        ("failure_items", "失败项"),
        ("source_mismatch_items", "源文本不匹配"),
    ):
        value = summary.get(key)
        if isinstance(value, int):
            facts.append(f"{label}：{value}")
    return facts


def _format_failure_item_lines(
    items: list[CheckFailureItem],
    *,
    omitted_count: int,
) -> list[str]:
    if not items:
        return []

    lines = ["", "【失败条目】"]
    for item in items:
        location_parts: list[str] = []
        if item.file_rel_path:
            location_parts.append(item.file_rel_path)
        if item.line is not None:
            location_parts.append(f"第 {item.line} 行")
        if item.item_id:
            location_parts.append(f"ID {item.item_id}")
        location = " / ".join(location_parts) if location_parts else "位置未知"
        lines.append(f"- [{reason_code_label(item.reason_code)}] {location}")
        if item.error:
            lines.append(f"  错误：{safe_preview(item.error, max_len=200)}")
    if omitted_count > 0:
        lines.append(f"… 另有 {omitted_count} 条未显示，请打开 failures.jsonl 查看。")
    return lines


def apply_failure_report_available(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
    path_exists: Callable[[str], bool] | None = None,
) -> bool:
    """Return whether a structured apply failure report can be opened."""
    if manifest.get("applied_at"):
        return False
    exists = path_exists or _default_path_exists
    report_path = resolve_apply_failure_report_path(manifest, manifest_path=manifest_path)
    if report_path and exists(report_path):
        return True
    stored = manifest.get("last_apply_failure_report_path")
    return isinstance(stored, str) and bool(stored.strip())


def build_apply_failure_report(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
    report_text: str | None = None,
    failures_text: str | None = None,
    path_exists: Callable[[str], bool] | None = None,
    read_file: Callable[[str], str] | None = None,
    max_items: int = 50,
) -> ApplyFailureReportView:
    """Build a structured GUI report from apply failure artifacts."""
    exists = path_exists or _default_path_exists
    reader = read_file or _default_read_file

    if not manifest_path:
        raw_manifest_path = manifest.get("_manifest_path")
        manifest_path = raw_manifest_path.strip() if isinstance(raw_manifest_path, str) else ""

    report_path = resolve_apply_failure_report_path(manifest, manifest_path=manifest_path)
    facts: list[str] = []
    if report_path:
        facts.append(f"写回失败报告：{report_path}")

    payload: dict[str, object] | None = None
    report_parse_error = ""
    failures_parse_error = ""

    if report_text is not None:
        try:
            loaded = json.loads(report_text)
            if not isinstance(loaded, dict):
                raise ValueError("报告根节点不是 JSON 对象。")
            payload = parse_apply_failure_report_payload(loaded)
        except (json.JSONDecodeError, ValueError) as exc:
            report_parse_error = str(exc)
    elif report_path and exists(report_path):
        try:
            loaded = json.loads(reader(report_path))
            if not isinstance(loaded, dict):
                raise ValueError("报告根节点不是 JSON 对象。")
            payload = parse_apply_failure_report_payload(loaded)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            report_parse_error = str(exc)

    reason_code = "unclassified_failure"
    error_message = ""
    failures_path = ""
    failure_items: list[object] = []

    if payload is not None:
        reason_code = str(payload.get("reason_code") or "unclassified_failure")
        error_message = str(payload.get("error") or "")
        failures_path = resolve_apply_failures_jsonl_path(
            payload,
            manifest,
            manifest_path=manifest_path,
        )
        if not failures_path:
            failures_path = str(payload.get("failures_path") or "")

        safety_text = str(payload.get("last_check_safety_level") or "")
        if safety_text:
            facts.append(f"最近检查：{safety_level_label(safety_text)}")

        last_check_at = str(payload.get("last_check_at") or "")
        if last_check_at:
            facts.append(f"最近检查时间：{last_check_at}")

        if manifest_path:
            facts.append(format_manifest_path_fact(manifest_path))

        summary = payload.get("summary")
        if isinstance(summary, dict):
            facts.extend(_format_summary_facts(summary))

        fingerprint = payload.get("current_check_fingerprint")
        if isinstance(fingerprint, dict):
            facts.extend(_format_fingerprint_facts(fingerprint))

        if failures_path:
            facts.append(f"失败明细：{failures_path}")

    if failures_text is not None:
        try:
            failure_items = parse_check_failures_jsonl(failures_text)
        except ValueError as exc:
            failures_parse_error = str(exc)
    elif failures_path and exists(failures_path):
        try:
            failure_items = parse_check_failures_jsonl(reader(failures_path))
        except (OSError, UnicodeError, ValueError) as exc:
            failures_parse_error = str(exc)

    normalized_items = [
        normalize_failure_entry(item) for item in failure_items if isinstance(item, dict)
    ]
    omitted_count = 0
    display_items = normalized_items
    if len(normalized_items) > max_items:
        display_items = normalized_items[:max_items]
        omitted_count = len(normalized_items) - max_items

    detail_lines = [
        f"【失败原因】{apply_reason_code_label(reason_code)} ({reason_code})",
        f"【建议处理】{apply_reason_suggestion(reason_code)}",
    ]
    if error_message:
        detail_lines.append(f"【错误摘要】{safe_preview(error_message, max_len=300)}")
    detail_lines.extend(
        _format_failure_item_lines(display_items, omitted_count=omitted_count)
    )

    if failures_parse_error:
        facts.append(f"失败明细解析错误：{failures_parse_error}")
        detail_lines.append(
            f"【失败明细】无法解析 failures.jsonl：{failures_parse_error}"
        )

    if report_parse_error:
        return ApplyFailureReportView(
            status="unreadable",
            heading="写回失败报告无法解析",
            message="找到了写回失败报告，但内容无法读取或解析。请打开原始报告或查看诊断日志。",
            report_path=report_path,
            failures_path=failures_path,
            reason_code=reason_code,
            facts=facts + [f"解析错误：{report_parse_error}"],
            detail_lines=detail_lines or [f"解析错误：{report_parse_error}"],
            failure_item_count=len(normalized_items),
        )

    if not report_path:
        return ApplyFailureReportView(
            status="missing_report",
            heading="未找到写回失败报告",
            message="任务记录中没有写回失败报告路径。请查看诊断日志。",
            report_path="",
            failures_path="",
            reason_code=reason_code,
            facts=facts,
            detail_lines=["未找到 apply_failure_report.json。"],
            failure_item_count=0,
        )

    if report_text is None and not exists(report_path):
        return ApplyFailureReportView(
            status="missing_report",
            heading="写回失败报告不可用",
            message="写回失败报告路径已记录，但文件当前不存在。请重新尝试写回或查看诊断日志。",
            report_path=report_path,
            failures_path=failures_path,
            reason_code=reason_code,
            facts=facts,
            detail_lines=[f"报告路径：{report_path}"],
            failure_item_count=len(normalized_items),
        )

    if payload is None:
        return ApplyFailureReportView(
            status="empty",
            heading="写回失败报告为空",
            message="写回失败报告存在，但没有可识别的内容。",
            report_path=report_path,
            failures_path=failures_path,
            reason_code=reason_code,
            facts=facts,
            detail_lines=["报告文件中没有可展示的内容。"],
            failure_item_count=0,
        )

    return ApplyFailureReportView(
        status="ok",
        heading="写回失败诊断",
        message="写回未完成。请按建议处理问题后重新检查，不要重复写回。",
        report_path=report_path,
        failures_path=failures_path,
        reason_code=reason_code,
        facts=facts,
        detail_lines=detail_lines,
        failure_item_count=len(normalized_items),
    )