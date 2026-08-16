"""User-facing writeback summaries for GUI revision preview/apply."""
from __future__ import annotations

from .check_report import WritebackSummary
from .revision_report import parse_revision_summary
from .summary_helpers import extend_facts_with_notices
from .user_copy import format_manifest_path_fact, format_quality_gate_fact


def summarize_revision_writeback_from_preview_output(
    output: str,
    exit_code: int,
    *,
    manifest_path: str = "",
    already_applied: bool = False,
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

    if already_applied:
        return WritebackSummary(
            status="applied",
            heading="订正已写回",
            message="该订正任务已经写回过。",
            facts=facts,
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
        status="idle",
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
    """Derive the writeback summary from a revision manifest.

    Priority: an explicit ``revision_apply_state`` (blocked/no_op/partial)
    reflects the latest apply outcome; otherwise ``revision_applied_at`` means
    the task was written back; otherwise a valid ``last_revision_preview`` with
    recoverable items enables apply. Any terminal state disables apply, and a
    missing preview returns ``None`` so the caller can show the idle state.
    """
    manifest_path = manifest.get("_manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        manifest_path = ""

    proposal_import = manifest.get("proposal_import")
    if isinstance(proposal_import, dict) and (
        proposal_import.get("status") not in {"previewed", "no_op"}
        or not proposal_import.get("writeback_eligible")
    ):
        proposal_status = str(proposal_import.get("status") or "blocked")
        facts = [format_manifest_path_fact(manifest_path)] if manifest_path else []
        report_path = proposal_import.get("report_path")
        if isinstance(report_path, str) and report_path.strip():
            facts.append(f"导入报告：{report_path.strip()}")
        return WritebackSummary(
            status="failed",
            heading="润色提案禁止写回",
            message=f"提案导入状态为 {proposal_status}；请修正并重新导入后再写回。",
            facts=facts,
            findings=[],
            can_apply=False,
            manifest_path=manifest_path,
        )

    apply_state = manifest.get("revision_apply_state")
    if apply_state in ("blocked", "no_op", "partial"):
        facts: list[str] = []
        if manifest_path:
            facts.append(format_manifest_path_fact(manifest_path))
        apply_summary = manifest.get("revision_apply_summary")
        if isinstance(apply_summary, dict):
            applied_files = apply_summary.get("applied_files")
            applied_lines = apply_summary.get("applied_lines")
            if isinstance(applied_files, int) and isinstance(applied_lines, int):
                facts.append(f"已写回 {applied_files} 个文件，{applied_lines} 处译文行")
            unchanged_items = apply_summary.get("unchanged_items")
            if isinstance(unchanged_items, int) and unchanged_items > 0:
                facts.append(f"无需修改项：{unchanged_items}")
        if apply_state == "no_op":
            return WritebackSummary(
                status="idle",
                heading="当前没有可写回订正",
                message="最近一次订正预览有效，但没有需要写回的内容（no-op）。",
                facts=facts,
                findings=[],
                can_apply=False,
                manifest_path=manifest_path,
            )
        if apply_state == "partial":
            return WritebackSummary(
                status="applied",
                heading="订正部分写回",
                message="部分订正已写回，其余条目被跳过或失败；请查看失败日志后重新生成订正任务。",
                facts=facts,
                findings=[],
                can_apply=False,
                manifest_path=manifest_path,
            )
        reason = manifest.get("revision_apply_blocked_reason") or ""
        detail = manifest.get("revision_apply_message") or ""
        message = "订正写回被阻止，请查看诊断日志后重新预览。"
        if reason:
            message = f"订正写回被阻止（{reason}）；请查看诊断日志后重新预览。"
        return WritebackSummary(
            status="failed",
            heading="订正写回被阻止",
            message=message,
            facts=facts + ([detail] if detail else []),
            findings=[],
            can_apply=False,
            manifest_path=manifest_path,
        )

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

    quality_gate = last_preview.get("quality_gate")
    if not isinstance(quality_gate, dict):
        quality_gate = summary.get("quality_gate")
    if isinstance(quality_gate, dict):
        facts.append(format_quality_gate_fact(quality_gate))
        quality_findings_path = last_preview.get("quality_findings_path")
        if isinstance(quality_findings_path, str) and quality_findings_path.strip():
            facts.append(f"质量检查报告：{quality_findings_path.strip()}")

    findings: list[str] = []
    reason_counts = summary.get("reason_counts")
    if isinstance(reason_counts, dict):
        for name in sorted(reason_counts):
            count = reason_counts[name]
            if isinstance(count, int):
                findings.append(f"{name}: {count}")

    writeback_gate = last_preview.get("writeback_gate")
    if not isinstance(writeback_gate, dict):
        writeback_gate = summary.get("writeback_gate")
    if isinstance(writeback_gate, dict) and writeback_gate.get(
        "decision"
    ) != "allow":
        blocker_count = int(writeback_gate.get("blocker_count") or 0)
        return WritebackSummary(
            status="failed",
            heading="订正写回被质量门禁阻止",
            message=(
                f"最近一次订正预览有 {blocker_count} 个阻断项；"
                "请处理质量 blocker 或结构阻断后重新预览。"
            ),
            facts=facts,
            findings=findings,
            can_apply=False,
            manifest_path=manifest_path,
        )

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
        status="idle",
        heading="当前没有可写回订正",
        message="最近一次订正预览没有可写回项；请查看预览报告后再决定是否调整任务。",
        facts=extend_facts_with_notices(facts, findings),
        findings=findings,
        can_apply=False,
        manifest_path=manifest_path,
    )
