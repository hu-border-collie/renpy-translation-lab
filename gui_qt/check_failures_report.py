"""Parse and summarize check_failures.jsonl for GUI issue lists."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .diagnostics_context import join_directory_file, resolve_package_dir
from .user_copy import safety_level_label

WARN_REASON_CODES = {
    "partial_result_items",
    "response_missing_item_id",
    "schema_or_item_mismatch",
    "validation_failed",
    "missing_chunk_rows",
}

BLOCK_REASON_CODES = {
    "invalid_result_jsonl_row",
    "unknown_chunk_key",
    "row_error",
    "missing_response_text",
    "failed_to_parse_model_json",
    "truncated_output",
    "duplicate_result_id",
    "source_line_missing",
    "source_text_mismatch",
    "missing_manifest_file",
    "target_file_missing",
    "target_file_path_escaped",
    "v2_relocation_missing",
}

REASON_CATEGORY_RETRY = "retry"
REASON_CATEGORY_REPAIR = "repair"
REASON_CATEGORY_MANUAL = "manual"
REASON_CATEGORY_REBUILD = "rebuild_check"

CATEGORY_LABELS = {
    REASON_CATEGORY_RETRY: "可尝试 retry",
    REASON_CATEGORY_REPAIR: "可能可 repair",
    REASON_CATEGORY_MANUAL: "需要人工处理",
    REASON_CATEGORY_REBUILD: "需要重新 build/check",
    "unknown": "待确认",
}

CATEGORY_ORDER = (
    REASON_CATEGORY_RETRY,
    REASON_CATEGORY_REPAIR,
    REASON_CATEGORY_MANUAL,
    REASON_CATEGORY_REBUILD,
    "unknown",
)

REASON_CODE_LABELS = {
    "partial_result_items": "部分结果条目",
    "response_missing_item_id": "响应缺少条目 ID",
    "schema_or_item_mismatch": "条目结构不匹配",
    "validation_failed": "译文校验失败",
    "missing_chunk_rows": "缺少 chunk 结果行",
    "invalid_result_jsonl_row": "结果 JSONL 行无效",
    "unknown_chunk_key": "未知 chunk 键",
    "row_error": "结果行错误",
    "missing_response_text": "响应缺少文本",
    "failed_to_parse_model_json": "模型 JSON 解析失败",
    "truncated_output": "输出被截断",
    "duplicate_result_id": "重复结果 ID",
    "source_line_missing": "源文件行缺失",
    "source_text_mismatch": "源文本不匹配",
    "missing_manifest_file": "清单文件条目缺失",
    "target_file_missing": "目标文件缺失",
    "target_file_path_escaped": "目标路径越界",
    "v2_relocation_missing": "重定位失败",
    "unclassified_failure": "未分类失败",
}

REASON_CATEGORY_BY_CODE = {
    "partial_result_items": REASON_CATEGORY_RETRY,
    "response_missing_item_id": REASON_CATEGORY_RETRY,
    "schema_or_item_mismatch": REASON_CATEGORY_RETRY,
    "validation_failed": REASON_CATEGORY_REPAIR,
    "missing_chunk_rows": REASON_CATEGORY_RETRY,
    "invalid_result_jsonl_row": REASON_CATEGORY_REBUILD,
    "unknown_chunk_key": REASON_CATEGORY_REBUILD,
    "row_error": REASON_CATEGORY_MANUAL,
    "missing_response_text": REASON_CATEGORY_RETRY,
    "failed_to_parse_model_json": REASON_CATEGORY_RETRY,
    "truncated_output": REASON_CATEGORY_RETRY,
    "duplicate_result_id": REASON_CATEGORY_REBUILD,
    "source_line_missing": REASON_CATEGORY_MANUAL,
    "source_text_mismatch": REASON_CATEGORY_MANUAL,
    "missing_manifest_file": REASON_CATEGORY_REBUILD,
    "target_file_missing": REASON_CATEGORY_MANUAL,
    "target_file_path_escaped": REASON_CATEGORY_MANUAL,
    "v2_relocation_missing": REASON_CATEGORY_MANUAL,
    "unclassified_failure": "unknown",
}

CATEGORY_SUGGESTIONS = {
    REASON_CATEGORY_RETRY: (
        "可生成 retry 包重新翻译失败 chunk，合并结果后重新 check；"
        "只有 safe 才能写回。"
    ),
    REASON_CATEGORY_REPAIR: (
        "部分条目可尝试 repair 流程；若 repair 不适用，请改走 retry 或人工修正。"
    ),
    REASON_CATEGORY_MANUAL: (
        "请检查源文件是否被修改、路径是否有效，或重定位是否失败；"
        "修复源文件后重新 build/check。"
    ),
    REASON_CATEGORY_REBUILD: (
        "结果包或清单可能已损坏或与当前任务不一致；"
        "建议重新 download、必要时重新 build，再 check。"
    ),
    "unknown": "请查看错误摘要与诊断日志，确认下一步操作。",
}


@dataclass(frozen=True)
class CheckFailureItem:
    reason_code: str
    status: str
    file_rel_path: str
    line: int | None
    item_id: str
    error: str
    text_preview: str


@dataclass(frozen=True)
class ReasonGroupSummary:
    reason_code: str
    status: str
    count: int
    category: str
    category_label: str
    reason_label: str
    suggestion: str


@dataclass(frozen=True)
class CheckIssuesReport:
    status: str
    heading: str
    message: str
    report_path: str
    safety_level: str
    category_counts: dict[str, int]
    reason_groups: list[ReasonGroupSummary]
    items: list[CheckFailureItem]
    omitted_item_count: int
    facts: list[str]
    detail_lines: list[str]


def reason_code_label(reason_code: str) -> str:
    """Return a Chinese label for a check failure reason code."""
    text = str(reason_code or "").strip()
    return REASON_CODE_LABELS.get(text, text or "未知原因")


def category_label(category: str) -> str:
    """Return a Chinese label for a remediation category."""
    return CATEGORY_LABELS.get(category, CATEGORY_LABELS["unknown"])


def classify_reason_category(reason_code: str) -> str:
    """Map a reason code to retry/repair/manual/rebuild_check guidance."""
    text = str(reason_code or "").strip()
    if not text:
        return "unknown"
    return REASON_CATEGORY_BY_CODE.get(text, "unknown")


def infer_reason_code_from_entry(entry: dict[str, object]) -> str:
    """Infer a stable reason code from a raw failure entry."""
    reason_code = entry.get("reason_code")
    if isinstance(reason_code, str) and reason_code.strip():
        return reason_code.strip()

    error = str(entry.get("error") or "").lower()
    if "invalid result jsonl row" in error:
        return "invalid_result_jsonl_row"
    if "unknown chunk key" in error:
        return "unknown_chunk_key"
    if "missing text in response payload" in error:
        return "missing_response_text"
    if "failed to parse model json" in error:
        return "failed_to_parse_model_json"
    if "response missing item id" in error:
        return "response_missing_item_id"
    if "validation failed" in error:
        return "validation_failed"
    if "no result row found" in error:
        return "missing_chunk_rows"
    if "source line missing" in error:
        return "source_line_missing"
    if "source text mismatch" in error:
        return "source_text_mismatch"
    if "manifest file entry missing" in error:
        return "missing_manifest_file"
    if "target file missing" in error:
        return "target_file_missing"
    if "v2 relocation missing" in error:
        return "v2_relocation_missing"
    if "escapes" in error:
        return "target_file_path_escaped"
    return "unclassified_failure"


def infer_status_for_reason(reason_code: str) -> str:
    """Infer warn/block status from a reason code."""
    if reason_code in BLOCK_REASON_CODES:
        return "block"
    if reason_code in WARN_REASON_CODES:
        return "warn"
    return "warn"


def safe_preview(text: str, *, max_len: int = 120) -> str:
    """Return a single-line, length-limited preview for GUI display."""
    normalized = str(text or "").replace("\r", "").replace("\n", "\\n").strip()
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 1] + "…"


def parse_check_failures_jsonl(text: str) -> list[dict[str, object]]:
    """Parse check_failures.jsonl text into failure entry dicts."""
    entries: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行无法解析为 JSON：{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"第 {line_number} 行不是 JSON 对象。")
        entries.append(payload)
    return entries


def normalize_failure_entry(entry: dict[str, object]) -> CheckFailureItem:
    """Normalize a raw failure entry into a GUI-friendly item."""
    reason_code = infer_reason_code_from_entry(entry)
    status = entry.get("status")
    status_text = status.strip().lower() if isinstance(status, str) and status.strip() else ""
    if not status_text:
        status_text = infer_status_for_reason(reason_code)

    file_rel_path = ""
    for key in ("file_rel_path", "file", "target_file"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            file_rel_path = value.strip()
            break

    item_id = ""
    for key in ("item_id", "id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            item_id = value.strip()
            break

    line_value = entry.get("line", entry.get("line_number"))
    line_number: int | None
    if isinstance(line_value, int):
        line_number = line_value
    elif isinstance(line_value, str) and line_value.strip().isdigit():
        line_number = int(line_value.strip())
    else:
        line_number = None

    error = str(entry.get("error") or "").strip()
    text_preview = safe_preview(str(entry.get("text") or ""))

    return CheckFailureItem(
        reason_code=reason_code,
        status=status_text,
        file_rel_path=file_rel_path,
        line=line_number,
        item_id=item_id,
        error=error,
        text_preview=text_preview,
    )


def group_failure_items(items: list[CheckFailureItem]) -> list[ReasonGroupSummary]:
    """Group failure items by reason code and status, sorted by count."""
    counter: Counter[tuple[str, str]] = Counter()
    status_by_reason: dict[str, str] = {}
    for item in items:
        counter[(item.reason_code, item.status)] += 1
        status_by_reason.setdefault(item.reason_code, item.status)

    groups: list[ReasonGroupSummary] = []
    for (reason_code, status), count in sorted(
        counter.items(),
        key=lambda pair: (-pair[1], pair[0][0], pair[0][1]),
    ):
        category = classify_reason_category(reason_code)
        groups.append(
            ReasonGroupSummary(
                reason_code=reason_code,
                status=status,
                count=count,
                category=category,
                category_label=category_label(category),
                reason_label=reason_code_label(reason_code),
                suggestion=CATEGORY_SUGGESTIONS.get(category, CATEGORY_SUGGESTIONS["unknown"]),
            )
        )
    return groups


def _default_path_exists(path: str) -> bool:
    try:
        return Path(path).exists()
    except OSError:
        return False


def _default_read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def resolve_check_report_path(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
) -> str:
    """Resolve the check_failures.jsonl path from manifest metadata."""
    report_path = manifest.get("last_check_report_path")
    if isinstance(report_path, str) and report_path.strip():
        return report_path.strip()

    package_dir = resolve_package_dir(manifest_path, manifest)
    if package_dir:
        return join_directory_file(package_dir, "check_failures.jsonl")
    return ""


def _summary_reason_groups(manifest: dict[str, object]) -> list[ReasonGroupSummary]:
    last_summary = manifest.get("last_check_summary")
    if not isinstance(last_summary, dict):
        return []

    safety_reasons = last_summary.get("safety_reasons")
    if not isinstance(safety_reasons, dict):
        return []

    groups: list[ReasonGroupSummary] = []
    for level in ("warn", "block"):
        reasons = safety_reasons.get(level)
        if not isinstance(reasons, dict):
            continue
        for reason_code, count in sorted(reasons.items()):
            if not isinstance(reason_code, str) or not reason_code.strip():
                continue
            if not isinstance(count, int) or count <= 0:
                continue
            category = classify_reason_category(reason_code)
            groups.append(
                ReasonGroupSummary(
                    reason_code=reason_code,
                    status=level,
                    count=count,
                    category=category,
                    category_label=category_label(category),
                    reason_label=reason_code_label(reason_code),
                    suggestion=CATEGORY_SUGGESTIONS.get(
                        category,
                        CATEGORY_SUGGESTIONS["unknown"],
                    ),
                )
            )
    return groups


def _category_counts(groups: list[ReasonGroupSummary]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in groups:
        counts[group.category] = counts.get(group.category, 0) + group.count
    return counts


def _format_item_line(item: CheckFailureItem) -> str:
    location_parts: list[str] = []
    if item.file_rel_path:
        location_parts.append(item.file_rel_path)
    if item.line is not None:
        location_parts.append(f"第 {item.line} 行")
    if item.item_id:
        location_parts.append(f"ID {item.item_id}")
    location = " / ".join(location_parts) if location_parts else "位置未知"

    lines = [
        (
            f"- [{safety_level_label(item.status)}] "
            f"{reason_code_label(item.reason_code)} ({item.reason_code})"
        ),
        f"  {location}",
    ]
    if item.error:
        lines.append(f"  错误：{safe_preview(item.error, max_len=200)}")
    if item.text_preview:
        lines.append(f"  原文摘要：{item.text_preview}")
    return "\n".join(lines)


def _build_detail_lines(
    groups: list[ReasonGroupSummary],
    items: list[CheckFailureItem],
    *,
    omitted_item_count: int,
) -> list[str]:
    lines: list[str] = []
    if groups:
        lines.append("【按原因汇总】")
        for group in groups:
            lines.append(
                f"- [{group.category_label}] {group.reason_label}"
                f"（{group.reason_code}，{group.count}） — {group.suggestion}"
            )

    if items:
        if lines:
            lines.append("")
        lines.append("【条目明细】")
        lines.extend(_format_item_line(item) for item in items)
        if omitted_item_count > 0:
            lines.append(f"… 另有 {omitted_item_count} 条未显示，请打开报告文件查看。")
    return lines


def build_check_issues_report(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
    report_text: str | None = None,
    path_exists: Callable[[str], bool] | None = None,
    read_file: Callable[[str], str] | None = None,
    max_items: int = 100,
) -> CheckIssuesReport:
    """Build a structured GUI report from manifest metadata and failure details."""
    exists = path_exists or _default_path_exists
    reader = read_file or _default_read_file

    if not manifest_path:
        raw_manifest_path = manifest.get("_manifest_path")
        manifest_path = raw_manifest_path.strip() if isinstance(raw_manifest_path, str) else ""

    last_summary = manifest.get("last_check_summary")
    safety_level = ""
    if isinstance(last_summary, dict):
        safety = last_summary.get("safety_level")
        if isinstance(safety, str) and safety.strip():
            safety_level = safety.strip().lower()

    report_path = resolve_check_report_path(manifest, manifest_path=manifest_path)
    facts: list[str] = []
    if safety_level:
        facts.append(f"检查结果：{safety_level_label(safety_level)}")
    if report_path:
        facts.append(f"检查报告：{report_path}")

    summary_groups = _summary_reason_groups(manifest)
    items: list[CheckFailureItem] = []
    parse_error = ""

    if report_text is not None:
        try:
            raw_entries = parse_check_failures_jsonl(report_text)
            items = [normalize_failure_entry(entry) for entry in raw_entries]
        except ValueError as exc:
            parse_error = str(exc)
    elif report_path and exists(report_path):
        try:
            raw_entries = parse_check_failures_jsonl(reader(report_path))
            items = [normalize_failure_entry(entry) for entry in raw_entries]
        except (OSError, UnicodeError, ValueError) as exc:
            parse_error = str(exc)

    reason_groups = group_failure_items(items) if items else summary_groups
    category_counts = _category_counts(reason_groups)

    omitted_item_count = 0
    display_items = items
    if len(items) > max_items:
        display_items = items[:max_items]
        omitted_item_count = len(items) - max_items

    detail_lines = _build_detail_lines(reason_groups, display_items, omitted_item_count=omitted_item_count)

    if parse_error:
        return CheckIssuesReport(
            status="unreadable",
            heading="检查报告无法解析",
            message=(
                "找到了检查报告，但内容无法读取或解析。"
                "请打开原始报告文件，或在诊断页查看完整日志。"
            ),
            report_path=report_path,
            safety_level=safety_level,
            category_counts=category_counts,
            reason_groups=reason_groups,
            items=display_items,
            omitted_item_count=omitted_item_count,
            facts=facts + [f"解析错误：{parse_error}"],
            detail_lines=detail_lines or [f"解析错误：{parse_error}"],
        )

    if not report_path and report_text is None:
        message = (
            "任务清单中没有记录检查报告路径，也无法从翻译包目录推断。"
            "请重新运行 check，或到诊断页查看任务上下文。"
        )
        if summary_groups:
            message = (
                "未找到检查报告文件路径；以下摘要来自任务清单中的最近检查结果。"
                "请重新运行 check 生成完整报告。"
            )
        return CheckIssuesReport(
            status="missing_report",
            heading="未找到检查报告",
            message=message,
            report_path="",
            safety_level=safety_level,
            category_counts=category_counts,
            reason_groups=reason_groups,
            items=display_items,
            omitted_item_count=omitted_item_count,
            facts=facts,
            detail_lines=detail_lines or ["未找到 check_failures.jsonl。"],
        )

    if report_text is None and not exists(report_path):
        fallback_message = (
            "检查报告路径已记录，但文件当前不存在。"
            "可能已被移动或清理；请重新 check 生成新报告。"
        )
        if summary_groups:
            fallback_message = (
                "详细报告文件暂不可用，以下摘要来自任务清单中的最近检查结果。"
            )
        return CheckIssuesReport(
            status="missing_report",
            heading="检查报告不可用",
            message=fallback_message,
            report_path=report_path,
            safety_level=safety_level,
            category_counts=category_counts,
            reason_groups=reason_groups,
            items=display_items,
            omitted_item_count=omitted_item_count,
            facts=facts,
            detail_lines=detail_lines or [f"报告路径：{report_path}"],
        )

    if not reason_groups and not items:
        return CheckIssuesReport(
            status="empty",
            heading="检查报告为空",
            message="检查报告存在，但没有记录失败条目。请查看诊断日志确认 check 输出。",
            report_path=report_path,
            safety_level=safety_level,
            category_counts={},
            reason_groups=[],
            items=[],
            omitted_item_count=0,
            facts=facts,
            detail_lines=["报告文件中没有失败条目。"],
        )

    failure_count = sum(group.count for group in reason_groups)
    return CheckIssuesReport(
        status="ok",
        heading="检查问题清单",
        message=(
            f"共发现 {failure_count} 个问题项。"
            "写回仍被禁用，请按建议处理后再重新 check 到 safe。"
        ),
        report_path=report_path,
        safety_level=safety_level,
        category_counts=category_counts,
        reason_groups=reason_groups,
        items=display_items,
        omitted_item_count=omitted_item_count,
        facts=facts,
        detail_lines=detail_lines,
    )


def format_category_overview(category_counts: dict[str, int]) -> list[str]:
    """Format remediation category counts for dialog overview text."""
    lines: list[str] = []
    for category in CATEGORY_ORDER:
        count = category_counts.get(category, 0)
        if count <= 0:
            continue
        lines.append(f"- {category_label(category)}：{count}")
    return lines