"""User-facing writeback summaries for GUI revision preview/apply."""
from __future__ import annotations

from .check_report import WritebackSummary
from .revision_report import parse_revision_summary
from .summary_helpers import extend_facts_with_notices
from .user_copy import format_manifest_path_fact


def summarize_revision_writeback_from_preview_output(
    output: str,
    exit_code: int,
    *,
    manifest_path: str = "",
) -> WritebackSummary:
    if exit_code != 0:
        return WritebackSummary(
            status="failed",
            heading="订正预览失败",
            message="订正预览没有正常完成，请查看诊断日志。",
            facts=[format_manifest_path_fact(manifest_path)] if manifest_path else [],
            findings=[],
            can_apply=False,
            manifest_path=manifest_path,
        )

    parsed = parse_revision_summary(output)
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))

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

    findings = [
        finding
        for finding in parsed.get("findings", [])
        if isinstance(finding, str) and finding.strip()
    ]

    if not isinstance(valid_items, int):
        return WritebackSummary(
            status="unknown",
            heading="订正预览结果不明确",
            message="未能识别订正预览摘要，请查看诊断日志后重新预览。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=False,
            manifest_path=manifest_path,
        )

    if valid_items > 0:
        return WritebackSummary(
            status="safe",
            heading="可以写回订正",
            message="订正预览显示有可写回项。写回前请确认已备份项目，写回会修改游戏脚本。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=True,
            manifest_path=manifest_path,
        )

    return WritebackSummary(
        status="warn",
        heading="当前没有可写回订正",
        message="预览已完成，但没有可写回的订正项；请查看预览报告后再决定是否调整任务。",
        facts=extend_facts_with_notices(facts, findings),
        findings=findings,
        can_apply=False,
        manifest_path=manifest_path,
    )


def summarize_revision_writeback_from_manifest(
    manifest: dict[str, object],
) -> WritebackSummary | None:
    manifest_path = manifest.get("_manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        manifest_path = ""

    if manifest.get("revision_applied_at"):
        apply_summary = manifest.get("revision_apply_summary")
        facts: list[str] = []
        if manifest_path:
            facts.append(format_manifest_path_fact(manifest_path))
        if isinstance(apply_summary, dict):
            applied_files = apply_summary.get("applied_files")
            applied_lines = apply_summary.get("applied_lines")
            if isinstance(applied_files, int) and isinstance(applied_lines, int):
                facts.append(f"已写回 {applied_files} 个文件，{applied_lines} 处译文行")
        return WritebackSummary(
            status="applied",
            heading="订正已写回",
            message="该订正任务已经写回过。",
            facts=facts,
            findings=[],
            can_apply=False,
            manifest_path=manifest_path,
        )

    last_preview = manifest.get("last_revision_preview")
    if not isinstance(last_preview, dict):
        return None

    summary = last_preview.get("summary")
    if not isinstance(summary, dict):
        return None

    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))

    valid_items = summary.get("valid_items")
    pending_files = summary.get("pending_files")
    pending_lines = summary.get("pending_lines")
    if isinstance(valid_items, int):
        facts.append(f"可写回订正项：{valid_items}")
    if isinstance(pending_files, int) and isinstance(pending_lines, int):
        facts.append(f"将影响 {pending_files} 个文件，约 {pending_lines} 处译文行")

    failure_items = summary.get("failure_items")
    if isinstance(failure_items, int):
        facts.append(f"失败项：{failure_items}")

    jsonl_path = last_preview.get("jsonl_path")
    if isinstance(jsonl_path, str) and jsonl_path.strip():
        facts.append(f"预览 JSONL：{jsonl_path.strip()}")
    markdown_path = last_preview.get("markdown_path")
    if isinstance(markdown_path, str) and markdown_path.strip():
        facts.append(f"预览 Markdown：{markdown_path.strip()}")

    findings: list[str] = []
    reason_counts = summary.get("reason_counts")
    if isinstance(reason_counts, dict):
        for name in sorted(reason_counts):
            count = reason_counts[name]
            if isinstance(count, int):
                findings.append(f"{name}: {count}")

    if isinstance(valid_items, int) and valid_items > 0:
        return WritebackSummary(
            status="safe",
            heading="可以写回订正",
            message="最近一次订正预览显示有可写回项。写回前请确认已备份项目。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=True,
            manifest_path=manifest_path,
        )

    return WritebackSummary(
        status="warn",
        heading="当前没有可写回订正",
        message="最近一次订正预览没有可写回项；请查看预览报告后再决定是否调整任务。",
        facts=extend_facts_with_notices(facts, findings),
        findings=findings,
        can_apply=False,
        manifest_path=manifest_path,
    )