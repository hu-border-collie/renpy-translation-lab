"""User-facing summaries for GUI revision preview and apply runs."""
from __future__ import annotations

import re

from .check_report import WritebackSummary
from .summary_helpers import extend_facts_with_notices
from .translation_workflow import WorkflowUpdate
from .user_copy import format_manifest_path_fact


def _parse_int_field(output: str, prefix: str) -> int | None:
    match = re.search(rf"^\s*{re.escape(prefix)}\s*(-?\d+)\s*$", output, re.MULTILINE)
    return int(match.group(1)) if match else None


def _parse_line_value(output: str, prefix: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(prefix)}\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(output)
    return match.group(1).strip() if match else ""


def parse_revision_summary(output: str) -> dict[str, object]:
    parsed: dict[str, object] = {
        "findings": [],
    }
    for field, key in (
        ("Expected chunks:", "expected_chunks"),
        ("Result rows:", "result_rows"),
        ("Processed chunks:", "processed_chunks"),
        ("Expected items:", "expected_items"),
        ("Parsed items:", "parsed_items"),
        ("Candidate revision items:", "candidate_items"),
        ("Recoverable revision items:", "valid_items"),
        ("Unchanged items:", "unchanged_items"),
        ("Pending files:", "pending_files"),
        ("Pending lines:", "pending_lines"),
        ("Skipped items:", "skipped_items"),
        ("Source mismatches:", "source_mismatch_items"),
        ("Failure items:", "failure_items"),
        ("Applied files:", "applied_files"),
        ("Applied lines:", "applied_lines"),
        ("Failures logged:", "failures_logged"),
    ):
        value = _parse_int_field(output, field)
        if value is not None:
            parsed[key] = value

    preview_jsonl = _parse_line_value(output, "Preview JSONL:")
    if preview_jsonl:
        parsed["preview_jsonl"] = preview_jsonl
    preview_markdown = _parse_line_value(output, "Preview Markdown:")
    if preview_markdown:
        parsed["preview_markdown"] = preview_markdown

    current_section = ""
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line == "Failure categories:":
            current_section = "failure_categories"
            continue
        if current_section == "failure_categories" and line.startswith("- "):
            parsed.setdefault("findings", []).append(line[2:].strip())
            continue
        if line and not line.startswith("- "):
            current_section = ""

    return parsed


def _collect_revision_preview_facts(output: str, parsed: dict[str, object]) -> list[str]:
    facts: list[str] = []
    valid_items = parsed.get("valid_items")
    pending_files = parsed.get("pending_files")
    pending_lines = parsed.get("pending_lines")
    if isinstance(valid_items, int):
        facts.append(f"可写回订正项：{valid_items}")
    if isinstance(pending_files, int) and isinstance(pending_lines, int):
        facts.append(f"将影响 {pending_files} 个文件，约 {pending_lines} 处译文行")
    failure_items = parsed.get("failure_items")
    if isinstance(failure_items, int):
        facts.append(f"失败项：{failure_items}")
    preview_jsonl = parsed.get("preview_jsonl")
    if isinstance(preview_jsonl, str) and preview_jsonl.strip():
        facts.append(f"预览 JSONL：{preview_jsonl.strip()}")
    preview_markdown = parsed.get("preview_markdown")
    if isinstance(preview_markdown, str) and preview_markdown.strip():
        facts.append(f"预览 Markdown：{preview_markdown.strip()}")
    return facts


def summarize_revision_preview_output(output: str, exit_code: int) -> WorkflowUpdate:
    if exit_code != 0:
        return WorkflowUpdate(
            status="failed",
            heading="订正预览中断",
            message="preview-revisions 没有正常完成，请查看下方原始输出。",
            facts=_collect_revision_preview_facts(output, parse_revision_summary(output)),
        )

    parsed = parse_revision_summary(output)
    facts = _collect_revision_preview_facts(output, parsed)
    valid_items = parsed.get("valid_items")
    if not isinstance(valid_items, int):
        return WorkflowUpdate(
            status="failed",
            heading="订正预览结果异常",
            message="preview-revisions 已结束，但输出中没有可识别摘要；请查看原始输出。",
            facts=facts,
        )

    findings = [
        finding
        for finding in parsed.get("findings", [])
        if isinstance(finding, str) and finding.strip()
    ]
    if findings:
        facts = extend_facts_with_notices(facts, findings)

    if valid_items == 0:
        return WorkflowUpdate(
            status="done",
            heading="订正预览完成",
            message="预览已完成，但没有可写回的订正项；请查看预览报告了解详情。",
            facts=facts,
        )

    return WorkflowUpdate(
        status="done",
        heading="订正预览完成",
        message="预览已完成；可在「订正写回」页确认后写回。写回前请备份项目。",
        facts=facts,
    )


def summarize_sync_revision_output(output: str, exit_code: int) -> WorkflowUpdate:
    if exit_code != 0:
        return WorkflowUpdate(
            status="failed",
            heading="同步订正中断",
            message=_sync_failure_message(output),
            facts=_collect_sync_revision_facts(output),
        )

    if "No revision source lines found." in output:
        return WorkflowUpdate(
            status="done",
            heading="没有可订正的源行",
            message="当前项目没有可用于订正的已有译文行。",
            facts=_collect_sync_revision_facts(output),
        )

    parsed = parse_revision_summary(output)
    facts = _collect_sync_revision_facts(output, parsed)
    valid_items = parsed.get("valid_items")
    if not isinstance(valid_items, int):
        return WorkflowUpdate(
            status="failed",
            heading="同步订正结果异常",
            message="同步订正已结束，但未能识别结果摘要；请查看诊断日志。",
            facts=facts,
        )

    findings = [
        finding
        for finding in parsed.get("findings", [])
        if isinstance(finding, str) and finding.strip()
    ]
    if findings:
        facts = extend_facts_with_notices(facts, findings)

    if valid_items == 0:
        return WorkflowUpdate(
            status="done",
            heading="同步订正预览完成",
            message="同步订正预览已完成，但没有可写回的订正项；请查看预览报告了解详情。",
            facts=facts,
        )

    return WorkflowUpdate(
        status="done",
        heading="同步订正预览完成",
        message="同步订正预览已完成；可在「写回说明」页查看摘要，确认后写回订正。",
        facts=facts,
    )


def summarize_revision_apply_output(
    output: str,
    exit_code: int,
    *,
    manifest_path: str = "",
) -> WritebackSummary:
    if exit_code != 0:
        return WritebackSummary(
            status="failed",
            heading="订正写回失败",
            message="订正写回未完成，请查看诊断日志。",
            facts=[format_manifest_path_fact(manifest_path)] if manifest_path else [],
            findings=[],
            can_apply=False,
            manifest_path=manifest_path,
        )

    parsed = parse_revision_summary(output)
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))

    applied_files = parsed.get("applied_files")
    applied_lines = parsed.get("applied_lines")
    if isinstance(applied_files, int) and isinstance(applied_lines, int):
        facts.append(f"已写回 {applied_files} 个文件，{applied_lines} 处译文行")

    failures_logged = parsed.get("failures_logged")
    if isinstance(failures_logged, int) and failures_logged > 0:
        facts.append(f"失败日志条目：{failures_logged}")

    findings = [
        finding
        for finding in parsed.get("findings", [])
        if isinstance(finding, str) and finding.strip()
    ]

    return WritebackSummary(
        status="applied",
        heading="订正写回完成",
        message="订正已写回。建议在游戏中抽查关键剧情文本。",
        facts=extend_facts_with_notices(facts, findings),
        findings=findings,
        can_apply=False,
        manifest_path=manifest_path,
    )


def _sync_failure_message(output: str) -> str:
    if "TL dir does not exist" in output:
        return "翻译目录不存在；请先运行环境检查或准备工作目录。"
    if "No revision chunks available for the requested range." in output:
        return "当前范围没有可订正的内容，请调整范围后重试。"
    return "同步订正没有正常完成，请查看下方原始输出。"


def _collect_sync_revision_facts(
    output: str,
    parsed: dict[str, object] | None = None,
) -> list[str]:
    parsed = parse_revision_summary(output) if parsed is None else parsed
    facts = _collect_revision_preview_facts(output, parsed)
    run_match = re.search(r"^Sync revision run:\s*(.+?)\s*$", output, re.MULTILINE)
    if run_match:
        facts.insert(0, f"同步输出目录：{run_match.group(1).strip()}")
    return facts