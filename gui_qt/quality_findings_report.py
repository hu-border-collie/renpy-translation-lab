"""Parse and summarize quality_findings.jsonl for GUI quality alarms."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import translation_quality

from .diagnostics_context import join_directory_file, resolve_package_dir
from .user_copy import QUALITY_DELIVERY_NOTICE

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
SEVERITY_LABELS = {"low": "低", "medium": "中", "high": "高"}

REASON_LABELS = {
    translation_quality.REASON_WAIT_TAG_INSIDE_CJK: "等待标签插入中文词内",
    translation_quality.REASON_UNCLOSED_DELIMITERS: "未闭合或破损的括号",
    translation_quality.REASON_ENGLISH_SUFFIX_ADJACENT: "中文与英文形态词尾粘连",
    translation_quality.REASON_SUSPICIOUS_ENGLISH_RESIDUE: "可疑英文残留",
    translation_quality.REASON_CJK_LATIN_SPACING: "CJK/拉丁字符间距",
    translation_quality.REASON_HALFWIDTH_PUNCTUATION: "半角标点或异常引号",
    translation_quality.REASON_ASCII_ELLIPSIS: "ASCII 省略号",
    translation_quality.REASON_GLOSSARY_TERM_NOT_APPLIED: "glossary 译法未满足",
    translation_quality.REASON_SPEAKER_LABEL_UNTRANSLATED: "说话人标签未翻译",
    translation_quality.REASON_INTERJECTION_UNTRANSLATED: "短感叹词/拟声词未翻译",
    translation_quality.REASON_KNOWN_GARBLED_PHRASE: "已知错乱词",
}


@dataclass(frozen=True)
class QualityFindingItem:
    reason_code: str
    disposition: str
    severity: str
    file_rel_path: str
    line: int | None
    item_id: str
    source: str
    translation: str
    evidence: str
    suggestion: str
    finding_id: str = ""


@dataclass(frozen=True)
class QualityFindingsReport:
    status: str
    heading: str
    message: str
    report_path: str
    warning_count: int
    blocker_count: int
    reason_counts: dict[str, int]
    items: list[QualityFindingItem]
    omitted_item_count: int
    facts: list[str]
    detail_lines: list[str]


def reason_label(reason_code: str) -> str:
    return REASON_LABELS.get(reason_code, reason_code or "未知规则")


def severity_label(severity: str) -> str:
    return SEVERITY_LABELS.get(severity, severity or "未知")


def _safe_preview(text: object, *, max_len: int = 160) -> str:
    normalized = str(text or "").replace("\r", "").replace("\n", "\\n").strip()
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 1] + "…"


def parse_quality_findings_jsonl(text: str) -> list[dict[str, object]]:
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


def normalize_quality_finding(entry: dict[str, object]) -> QualityFindingItem:
    line_value = entry.get("line")
    line_number: int | None
    if isinstance(line_value, int):
        line_number = line_value
    elif isinstance(line_value, str) and line_value.strip().isdigit():
        line_number = int(line_value.strip())
    else:
        line_number = None

    return QualityFindingItem(
        reason_code=str(entry.get("reason_code") or "").strip(),
        disposition=str(entry.get("disposition") or "warning").strip(),
        severity=str(entry.get("severity") or "medium").strip(),
        file_rel_path=str(entry.get("file") or "").strip(),
        line=line_number,
        item_id=str(entry.get("item_id") or "").strip(),
        source=str(entry.get("source") or ""),
        translation=str(entry.get("translation") or ""),
        evidence=str(entry.get("evidence") or ""),
        suggestion=str(entry.get("suggestion") or ""),
        finding_id=str(entry.get("finding_id") or ""),
    )


def resolve_quality_findings_path(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
) -> str:
    report_path = manifest.get("last_quality_findings_path")
    if isinstance(report_path, str) and report_path.strip():
        return report_path.strip()

    package_dir = resolve_package_dir(manifest_path, manifest)
    if package_dir:
        return join_directory_file(package_dir, "quality_findings.jsonl")
    return ""


def quality_gate_from_manifest(manifest: dict[str, object]) -> dict[str, object]:
    last_summary = manifest.get("last_check_summary")
    if isinstance(last_summary, dict):
        quality_gate = last_summary.get("quality_gate")
        if isinstance(quality_gate, dict):
            return dict(quality_gate)
    return {}


def quality_issues_report_ready(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
) -> bool:
    quality_gate = quality_gate_from_manifest(manifest)
    if int(quality_gate.get("warning_count") or 0) > 0:
        return True
    if int(quality_gate.get("blocker_count") or 0) > 0:
        return True
    report_path = resolve_quality_findings_path(manifest, manifest_path=manifest_path)
    return bool(report_path and Path(report_path).exists())


def filter_quality_items(
    items: Iterable[QualityFindingItem],
    *,
    reason_code: str = "",
    file_text: str = "",
    min_severity: str = "",
) -> list[QualityFindingItem]:
    selected_reason = str(reason_code or "").strip()
    selected_file = str(file_text or "").strip().casefold()
    minimum = SEVERITY_ORDER.get(str(min_severity or "").strip().lower(), 0)
    result: list[QualityFindingItem] = []
    for item in items:
        if selected_reason and item.reason_code != selected_reason:
            continue
        if selected_file and selected_file not in item.file_rel_path.casefold():
            continue
        if SEVERITY_ORDER.get(item.severity.lower(), 2) < minimum:
            continue
        result.append(item)
    return result


def _format_item_line(item: QualityFindingItem) -> str:
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
            f"- [{severity_label(item.severity)}/{item.disposition}] "
            f"{reason_label(item.reason_code)} ({item.reason_code})"
        ),
        f"  {location}",
    ]
    if item.source:
        lines.append(f"  原文：{_safe_preview(item.source)}")
    if item.translation:
        lines.append(f"  译文：{_safe_preview(item.translation)}")
    if item.evidence:
        lines.append(f"  证据：{_safe_preview(item.evidence)}")
    if item.suggestion:
        lines.append(f"  建议：{_safe_preview(item.suggestion)}")
    return "\n".join(lines)


def build_quality_findings_report(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
    report_text: str | None = None,
    path_exists: Callable[[str], bool] | None = None,
    read_file: Callable[[str], str] | None = None,
    max_items: int = 200,
) -> QualityFindingsReport:
    exists = path_exists or (lambda path: Path(path).exists())
    reader = read_file or (lambda path: Path(path).read_text(encoding="utf-8"))

    if not manifest_path:
        raw_manifest_path = manifest.get("_manifest_path")
        manifest_path = raw_manifest_path.strip() if isinstance(raw_manifest_path, str) else ""

    quality_gate = quality_gate_from_manifest(manifest)
    warning_count = int(quality_gate.get("warning_count") or 0)
    blocker_count = int(quality_gate.get("blocker_count") or 0)
    report_path = resolve_quality_findings_path(manifest, manifest_path=manifest_path)
    facts: list[str] = []
    if warning_count or blocker_count:
        facts.append(f"质量报警 {warning_count} 条，质量阻断 {blocker_count} 条")
    if report_path:
        facts.append(f"质量检查报告：{report_path}")

    items: list[QualityFindingItem] = []
    parse_error = ""

    if report_text is not None:
        try:
            raw_entries = parse_quality_findings_jsonl(report_text)
            items = [normalize_quality_finding(entry) for entry in raw_entries]
        except ValueError as exc:
            parse_error = str(exc)
    elif report_path and exists(report_path):
        try:
            raw_entries = parse_quality_findings_jsonl(reader(report_path))
            items = [normalize_quality_finding(entry) for entry in raw_entries]
        except (OSError, UnicodeError, ValueError) as exc:
            parse_error = str(exc)

    reason_counts = Counter(item.reason_code for item in items)
    omitted_item_count = 0
    display_items = items
    if len(items) > max_items:
        display_items = items[:max_items]
        omitted_item_count = len(items) - max_items

    detail_lines: list[str] = []
    if reason_counts:
        detail_lines.append("【按规则汇总】")
        for code, count in sorted(reason_counts.items(), key=lambda pair: (-pair[1], pair[0])):
            detail_lines.append(f"- {reason_label(code)}（{code}，{count}）")
    if display_items:
        if detail_lines:
            detail_lines.append("")
        detail_lines.append("【条目明细】")
        detail_lines.extend(_format_item_line(item) for item in display_items)
        if omitted_item_count:
            detail_lines.append(f"… 另有 {omitted_item_count} 条未显示，请使用筛选器缩小范围。")

    if parse_error:
        return QualityFindingsReport(
            status="unreadable",
            heading="质量检查报告无法解析",
            message="找到了质量检查报告，但内容无法读取或解析。请打开原始报告文件。",
            report_path=report_path,
            warning_count=warning_count,
            blocker_count=blocker_count,
            reason_counts=dict(reason_counts),
            items=display_items,
            omitted_item_count=omitted_item_count,
            facts=facts + [f"解析错误：{parse_error}"],
            detail_lines=detail_lines or [f"解析错误：{parse_error}"],
        )

    if not report_path and report_text is None:
        return QualityFindingsReport(
            status="missing_report",
            heading="未找到质量检查报告",
            message="任务记录中没有质量检查报告路径。请重新运行检查以生成报告。",
            report_path="",
            warning_count=warning_count,
            blocker_count=blocker_count,
            reason_counts=dict(reason_counts),
            items=display_items,
            omitted_item_count=omitted_item_count,
            facts=facts,
            detail_lines=detail_lines or ["未找到 quality_findings.jsonl。"],
        )

    if report_text is None and not exists(report_path):
        return QualityFindingsReport(
            status="missing_report",
            heading="质量检查报告不可用",
            message="质量检查报告路径已记录，但文件当前不存在。请重新运行检查。",
            report_path=report_path,
            warning_count=warning_count,
            blocker_count=blocker_count,
            reason_counts=dict(reason_counts),
            items=display_items,
            omitted_item_count=omitted_item_count,
            facts=facts,
            detail_lines=detail_lines or [f"报告路径：{report_path}"],
        )

    if not items:
        return QualityFindingsReport(
            status="empty",
            heading="质量检查报告为空",
            message="质量检查报告存在，但没有质量报警条目。",
            report_path=report_path,
            warning_count=warning_count,
            blocker_count=blocker_count,
            reason_counts={},
            items=[],
            omitted_item_count=0,
            facts=facts,
            detail_lines=["报告文件中没有质量报警。"],
        )

    message = (
        f"共发现 {len(items)} 条质量报警。质量报警默认不阻止写回；"
        f"写回后仍需按规则、文件与严重程度逐条处理。{QUALITY_DELIVERY_NOTICE}"
    )
    if blocker_count:
        message = (
            f"共发现 {len(items)} 条质量报警，其中 {blocker_count} 条已被项目配置提升为 blocker，"
            f"会阻止写回；请先处理 blocker 并重新检查。{QUALITY_DELIVERY_NOTICE}"
        )
    return QualityFindingsReport(
        status="ok",
        heading="译文质量报警",
        message=message,
        report_path=report_path,
        warning_count=warning_count,
        blocker_count=blocker_count,
        reason_counts=dict(reason_counts),
        items=display_items,
        omitted_item_count=omitted_item_count,
        facts=facts,
        detail_lines=detail_lines,
    )
