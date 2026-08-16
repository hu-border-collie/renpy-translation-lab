"""User-facing summaries for GUI revision preview and apply runs."""
from __future__ import annotations

import re

from .check_report import WritebackSummary
from .summary_helpers import extend_facts_with_notices
from .sync_usage_report import collect_sync_usage_facts
from .translation_workflow import WorkflowUpdate
from .user_copy import MODEL_CONTRACT_COPY, format_manifest_path_fact


def _parse_int_field(output: str, prefix: str) -> int | None:
    match = re.search(rf"^\s*{re.escape(prefix)}\s*(-?\d+)\s*$", output, re.MULTILINE)
    return int(match.group(1)) if match else None


def _parse_line_value(output: str, prefix: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(prefix)}\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(output)
    return match.group(1).strip() if match else ""


def parse_revision_summary(output: str) -> dict[str, object]:
    """Parse CLI revision summary text into stable keys.

    Recognized count fields map to ``expected_chunks``, ``result_rows``,
    ``processed_chunks``, ``expected_items``, ``parsed_items``,
    ``candidate_items``, ``valid_items``, ``unchanged_items``,
    ``pending_files``, ``pending_lines``, ``skipped_items``,
    ``source_mismatch_items``, ``failure_items``, ``applied_files``,
    ``applied_lines`` and ``failures_logged``. Preview/apply paths and the
    ``Revision apply state`` / ``Revision apply reason`` lines are returned as
    ``preview_jsonl`` / ``preview_markdown`` / ``apply_state`` /
    ``apply_reason``. Missing fields are simply absent from the returned dict;
    callers must not assume a count exists.
    """
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

    apply_state = _parse_line_value(output, "Revision apply state:")
    if apply_state:
        parsed["apply_state"] = apply_state
    apply_reason = _parse_line_value(output, "Revision apply reason:")
    if apply_reason:
        parsed["apply_reason"] = apply_reason
    quality_gate = _parse_line_value(output, "Quality gate:")
    if quality_gate:
        parsed["quality_gate"] = quality_gate
    quality_findings = _parse_line_value(output, "Quality findings:")
    if quality_findings:
        parsed["quality_findings"] = quality_findings
    writeback_gate = _parse_line_value(output, "Revision writeback gate:")
    if writeback_gate:
        parsed["revision_writeback_gate"] = writeback_gate

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
    quality_fact = _quality_gate_fact(str(parsed.get("quality_gate") or ""))
    if quality_fact:
        facts.append(quality_fact)
    quality_findings = parsed.get("quality_findings")
    if isinstance(quality_findings, str) and quality_findings.strip():
        facts.append(f"质量检查报告：{quality_findings.strip()}")
    return facts


def _quality_gate_fact(quality_gate_text: str) -> str:
    warnings_match = re.search(r"warnings=(\d+)", quality_gate_text)
    blockers_match = re.search(r"blockers=(\d+)", quality_gate_text)
    if warnings_match is None and blockers_match is None:
        return ""
    warnings = int(warnings_match.group(1)) if warnings_match else 0
    blockers = int(blockers_match.group(1)) if blockers_match else 0
    return f"质量报警 {warnings} 条，质量阻断 {blockers} 条"


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
        message="预览已完成；可在「订正」结果区点击「写回订正」确认后写回。写回前请备份项目。",
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
        unresolved = _contract_unresolved_count(output)
        contract_partial = (
            unresolved > 0 or _contract_partial_request_count(output) > 0
        )
        return WorkflowUpdate(
            status="warning" if contract_partial else "done",
            heading=(
                "同步订正预览部分完成"
                if contract_partial
                else "同步订正预览完成"
            ),
            message=(
                MODEL_CONTRACT_COPY["partial_revision"]
                if contract_partial
                else "同步订正预览已完成，但没有可写回的订正项；请查看预览报告了解详情。"
            ),
            facts=facts,
        )

    unresolved = _contract_unresolved_count(output)
    if unresolved > 0 or _contract_partial_request_count(output) > 0:
        return WorkflowUpdate(
            status="warning",
            heading="同步订正预览部分完成",
            message=(
                f"{MODEL_CONTRACT_COPY['partial_revision']}"
                "确认后只能写回已通过合同并进入预览的订正项。"
            ),
            facts=facts,
        )

    return WorkflowUpdate(
        status="done",
        heading="同步订正预览完成",
        message="同步订正预览已完成；可在「订正」结果区查看摘要，确认后点击「写回订正」。",
        facts=facts,
    )


def summarize_revision_apply_output(
    output: str,
    exit_code: int,
    *,
    manifest_path: str = "",
) -> WritebackSummary:
    """Summarize an apply-revisions run for the writeback page.

    ``blocked`` parsed from output takes precedence over ``exit_code`` so an
    all-items-blocked run (which exits 0 with ``Revision apply state: blocked``)
    and a preview-contract refusal (non-zero exit with the same state line) both
    render as blocked. ``no_op`` renders as idle, ``partial`` renders as applied
    with a partial heading, and a clean run renders as applied; every terminal
    state disables further apply.
    """
    parsed = parse_revision_summary(output)
    if parsed.get("apply_state") == "blocked":
        facts: list[str] = []
        if manifest_path:
            facts.append(format_manifest_path_fact(manifest_path))
        reason = _parse_line_value(output, "Revision apply reason:")
        if reason:
            facts.append(f"阻断原因：{reason}")
        return WritebackSummary(
            status="failed",
            heading="订正写回被阻止",
            message="存在阻断项，没有发生写回；请查看诊断日志后重新预览。",
            facts=facts,
            findings=[],
            can_apply=False,
            manifest_path=manifest_path,
        )
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

    apply_state = parsed.get("apply_state")
    if apply_state == "no_op":
        return WritebackSummary(
            status="idle",
            heading="没有需要写回的订正",
            message="订正写回已检查，但没有需要修改的内容（no-op）。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=False,
            manifest_path=manifest_path,
        )
    if apply_state == "blocked":
        return WritebackSummary(
            status="failed",
            heading="订正写回被阻止",
            message="存在阻断项，没有发生写回；请查看诊断日志后重新预览。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=False,
            manifest_path=manifest_path,
        )
    if apply_state == "partial":
        return WritebackSummary(
            status="applied",
            heading="订正部分写回",
            message="部分订正已写回，其余条目被跳过或失败；请查看失败日志后重新生成订正任务。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=False,
            manifest_path=manifest_path,
        )

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
    completeness = re.search(
        r"^Model contract completeness:\s*(\d+/\d+)\s*$", output, re.MULTILINE
    )
    if completeness:
        facts.append(f"{MODEL_CONTRACT_COPY['completeness']}：{completeness.group(1)}")
    retries = re.search(
        r"^Targeted retries:\s*(\d+) requests / (\d+) items\s*$",
        output,
        re.MULTILINE,
    )
    if retries:
        facts.append(
            f"{MODEL_CONTRACT_COPY['targeted_retries']}："
            f"{retries.group(1)} 次请求 / {retries.group(2)} 项"
        )
    unresolved = _contract_unresolved_count(output)
    if unresolved >= 0:
        facts.append(f"{MODEL_CONTRACT_COPY['unresolved_items']}：{unresolved} 个")
    partial_requests = _contract_partial_request_count(output)
    if partial_requests >= 0:
        facts.append(
            f"{MODEL_CONTRACT_COPY['partial_requests']}：{partial_requests} 个"
        )
    facts.extend(collect_sync_usage_facts(output))
    return facts


def _contract_unresolved_count(output: str) -> int:
    match = re.search(
        r"^Unresolved contract items:\s*(\d+)\s*$", output, re.MULTILINE
    )
    return int(match.group(1)) if match else -1


def _contract_partial_request_count(output: str) -> int:
    match = re.search(
        r"^Contract partial requests:\s*(\d+)\s*$", output, re.MULTILINE
    )
    return int(match.group(1)) if match else -1
