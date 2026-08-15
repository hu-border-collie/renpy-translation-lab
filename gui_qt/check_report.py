"""User-facing summaries for GUI check/apply commands."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

from .summary_helpers import extend_facts_with_notices
from .user_copy import (
    QUALITY_DELIVERY_NOTICE,
    format_manifest_path_fact,
    format_quality_gate_fact,
    quality_gate_label,
    safety_level_label,
)


@dataclass(frozen=True)
class WritebackSummary:
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]
    can_apply: bool
    manifest_path: str = ""


def _parse_int_field(output: str, prefix: str) -> int | None:
    match = re.search(rf"^\s*{re.escape(prefix)}\s*(-?\d+)\s*$", output, re.MULTILINE)
    return int(match.group(1)) if match else None


def _parse_line_value(output: str, prefix: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(prefix)}\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(output)
    return match.group(1).strip() if match else ""


def extract_safety_status(output: str) -> str:
    return _parse_line_value(output, "Safety status:")


def extract_check_status(output: str) -> str:
    return _parse_line_value(output, "Check status:") or extract_safety_status(output)


def extract_writeback_gate(output: str) -> str:
    return _parse_line_value(output, "Writeback gate:")


def extract_next_split_manifest(output: str) -> str:
    return _parse_line_value(output, "Next split manifest:")


def parse_check_output(output: str) -> dict[str, object]:
    parsed: dict[str, object] = {
        "safety_status": extract_safety_status(output),
        "findings": [],
    }
    for field, key in (
        ("Pending files:", "pending_files"),
        ("Pending lines:", "pending_lines"),
        ("Failure items:", "failure_items"),
        ("Recoverable valid items:", "valid_items"),
        ("Quality warnings:", "quality_warnings"),
        ("Quality blockers:", "quality_blockers"),
        ("Acknowledged warnings:", "acknowledged_warnings"),
    ):
        value = _parse_int_field(output, field)
        if value is not None:
            parsed[key] = value

    report_path = _parse_line_value(output, "Check failure report:")
    if report_path:
        parsed["check_failure_report"] = report_path
    quality_report_path = _parse_line_value(output, "Quality findings report:")
    if quality_report_path:
        parsed["quality_findings_report"] = quality_report_path
    parsed["check_status"] = extract_check_status(output)
    parsed["writeback_gate"] = extract_writeback_gate(output)

    current_section = ""
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line in {"Warn reasons:", "Block reasons:"}:
            current_section = line[:-1].lower().replace(" reasons", "")
            continue
        if current_section and line.startswith("- "):
            parsed.setdefault("findings", []).append(f"[{current_section}] {line[2:].strip()}")
            continue
        if line and not line.startswith("- "):
            current_section = ""

    return parsed


def _can_apply_from_check_fields(
    writeback_gate: object,
    check_status: str,
    safety_status: str,
) -> bool:
    if isinstance(writeback_gate, str) and writeback_gate.strip().lower() == "allow":
        return True
    if isinstance(writeback_gate, str) and writeback_gate.strip().lower() == "deny":
        return False
    return check_status in {"safe", "ready", "ready_with_warnings"} or safety_status == "safe"


def _can_apply_from_manifest_gate(
    writeback_gate: object,
    check_status: str,
    safety_status: str,
) -> bool:
    if isinstance(writeback_gate, dict):
        return writeback_gate.get("decision") == "allow" and bool(
            writeback_gate.get("can_apply")
        )
    return _can_apply_from_check_fields(writeback_gate, check_status, safety_status)


def _format_check_finding(finding: str) -> str:
    if finding.startswith("[warn] "):
        return f"[{safety_level_label('warn')}] {finding[7:]}"
    if finding.startswith("[block] "):
        return f"[{safety_level_label('block')}] {finding[8:]}"
    return finding


def summarize_check_envelope(
    envelope: Mapping[str, object],
    exit_code: int,
    *,
    manifest_path: str = "",
    already_applied: bool = False,
) -> WritebackSummary:
    """Build the GUI check summary from the shared CLI result envelope."""

    if not envelope.get("ok"):
        error = envelope.get("error")
        message = error.get("message") if isinstance(error, Mapping) else ""
        summary = summarize_check_output("", exit_code or 1, manifest_path=manifest_path)
        if isinstance(message, str) and message.strip():
            return replace(summary, message=message.strip())
        return summary

    result = envelope.get("result")
    check = result.get("check") if isinstance(result, Mapping) else None
    check_summary = dict(check) if isinstance(check, Mapping) else {}
    envelope_status = str(envelope.get("status") or "")
    check_summary.setdefault(
        "safety_level",
        "safe" if envelope_status in {"ready", "ready_with_warnings"} else envelope_status,
    )
    manifest: dict[str, object] = {
        "_manifest_path": manifest_path,
        "last_check_summary": check_summary,
    }
    artifacts = envelope.get("artifacts")
    if isinstance(artifacts, Mapping):
        report_path = artifacts.get("check_report")
        if isinstance(report_path, str) and report_path:
            manifest["last_check_report_path"] = report_path
    if already_applied:
        manifest["applied_at"] = "structured_result"

    summary = summarize_manifest_writeback(manifest)
    if summary is not None:
        return summary
    return summarize_check_output("", exit_code, manifest_path=manifest_path)


def summarize_check_output(
    output: str,
    exit_code: int,
    *,
    manifest_path: str = "",
    already_applied: bool = False,
) -> WritebackSummary:
    if exit_code != 0:
        return WritebackSummary(
            status="failed",
            heading="结果检查失败",
            message="结果检查没有正常完成，请查看诊断日志。",
            facts=[format_manifest_path_fact(manifest_path)] if manifest_path else [],
            findings=[],
            can_apply=False,
            manifest_path=manifest_path,
        )

    parsed = parse_check_output(output)
    safety = parsed.get("safety_status")
    safety_text = safety if isinstance(safety, str) else ""
    check_status = parsed.get("check_status")
    check_status_text = check_status if isinstance(check_status, str) else safety_text
    writeback_gate = parsed.get("writeback_gate")
    can_apply = _can_apply_from_check_fields(writeback_gate, check_status_text, safety_text)

    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))

    pending_files = parsed.get("pending_files")
    pending_lines = parsed.get("pending_lines")
    if isinstance(pending_files, int) and isinstance(pending_lines, int):
        facts.append(f"将影响 {pending_files} 个文件，约 {pending_lines} 处译文行")

    failure_items = parsed.get("failure_items")
    if isinstance(failure_items, int):
        facts.append(f"失败项：{failure_items}")

    if isinstance(parsed.get("check_failure_report"), str):
        facts.append(f"检查报告：{parsed['check_failure_report']}")
    if isinstance(parsed.get("quality_findings_report"), str):
        facts.append(f"质量检查报告：{parsed['quality_findings_report']}")

    quality_warnings = parsed.get("quality_warnings")
    quality_blockers = parsed.get("quality_blockers")
    if isinstance(quality_warnings, int) or isinstance(quality_blockers, int):
        warning_count = quality_warnings if isinstance(quality_warnings, int) else 0
        blocker_count = quality_blockers if isinstance(quality_blockers, int) else 0
        facts.append(f"质量检查：{quality_gate_label('needs_review')}（报警 {warning_count}，阻断 {blocker_count}）")

    findings = [
        _format_check_finding(finding)
        for finding in parsed.get("findings", [])
        if isinstance(finding, str) and finding.strip()
    ]

    if already_applied:
        return WritebackSummary(
            status="applied",
            heading="翻译已写回",
            message="该任务已经写回过，不会再次写回。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=False,
            manifest_path=manifest_path,
        )

    if can_apply:
        if quality_warnings or quality_blockers or check_status_text == "ready_with_warnings":
            return WritebackSummary(
                status="safe",
                heading="可以写回翻译（有质量报警）",
                message=(
                    "检查结果满足写回条件，但存在需要人工关注的质量问题。"
                    "可先写回，再按规则、文件或严重程度筛选报警并处理。"
                    + QUALITY_DELIVERY_NOTICE
                ),
                facts=extend_facts_with_notices(facts, findings),
                findings=findings,
                can_apply=True,
                manifest_path=manifest_path,
            )
        return WritebackSummary(
            status="safe",
            heading="可以写回翻译",
            message="检查结果为可写回。写回前请确认已备份项目，写回会修改游戏脚本。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=True,
            manifest_path=manifest_path,
        )

    if safety_text == "warn" or check_status_text in {"warn", "blocked", "block"}:
        return WritebackSummary(
            status="warn" if safety_text == "warn" else "block",
            heading="需要先处理问题" if safety_text == "warn" else "当前不能写回",
            message=(
                "检查结果为需处理，暂不能写回。"
                "可先查看问题清单，必要时生成「补译包」并预览；"
                "处理完重新检查后，显示「可写回」才能写入项目。"
            )
            if safety_text == "warn"
            else "检查结果为禁止写回。请修复源文件变化或重新生成任务后再检查。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=False,
            manifest_path=manifest_path,
        )

    return WritebackSummary(
        status="unknown",
        heading="检查结果不明确",
        message="未能识别检查结果，请查看诊断日志后重新检查。",
        facts=extend_facts_with_notices(facts, findings),
        findings=findings,
        can_apply=False,
        manifest_path=manifest_path,
    )


def summarize_apply_envelope(
    envelope: Mapping[str, object],
    exit_code: int,
    *,
    manifest_path: str = "",
) -> WritebackSummary:
    """Build the GUI apply summary from the shared CLI result envelope."""

    if not envelope.get("ok"):
        error = envelope.get("error")
        message = error.get("message") if isinstance(error, Mapping) else ""
        summary = summarize_apply_output("", exit_code or 1, manifest_path=manifest_path)
        if isinstance(message, str) and message.strip():
            return replace(summary, message=message.strip())
        return summary

    result = envelope.get("result")
    apply = result.get("apply") if isinstance(result, Mapping) else None
    apply_summary = dict(apply) if isinstance(apply, Mapping) else {}
    manifest: dict[str, object] = {
        "_manifest_path": manifest_path,
        "applied_at": "structured_result",
        "apply_summary": apply_summary,
    }
    next_manifest = apply_summary.get("next_split_manifest")
    if isinstance(next_manifest, str) and next_manifest:
        manifest["next_split_manifest_path"] = next_manifest

    summary = summarize_manifest_writeback(manifest)
    if summary is not None:
        return replace(summary, heading="翻译写回完成")
    return summarize_apply_output("", exit_code, manifest_path=manifest_path)


def summarize_apply_output(
    output: str,
    exit_code: int,
    *,
    manifest_path: str = "",
) -> WritebackSummary:
    if exit_code != 0:
        return WritebackSummary(
            status="failed",
            heading="写回失败",
            message="写回没有正常完成。可点击「查看写回失败报告」了解原因，或查看诊断日志。",
            facts=[format_manifest_path_fact(manifest_path)] if manifest_path else [],
            findings=[],
            can_apply=False,
            manifest_path=manifest_path,
        )

    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))

    applied_files = _parse_int_field(output, "Applied files:")
    applied_lines = _parse_int_field(output, "Applied lines:")
    if isinstance(applied_files, int) and isinstance(applied_lines, int):
        facts.append(f"已写回 {applied_files} 个文件，{applied_lines} 处译文行")

    failures_logged = _parse_int_field(output, "Failures logged:")
    if isinstance(failures_logged, int) and failures_logged > 0:
        facts.append(f"失败日志条目：{failures_logged}")

    next_split_manifest = extract_next_split_manifest(output)
    message = "写回已完成。建议在游戏中抽查关键剧情文本。"
    if next_split_manifest:
        facts.append(f"下一拆分包：{next_split_manifest}")
        message = "写回已完成，已切换到下一拆分包；可继续提交后续任务。"

    return WritebackSummary(
        status="applied",
        heading="翻译写回完成",
        message=message,
        facts=facts,
        findings=[],
        can_apply=False,
        manifest_path=manifest_path,
    )


def summarize_manifest_writeback(manifest: dict[str, object]) -> WritebackSummary | None:
    manifest_path = manifest.get("_manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        manifest_path = ""

    if manifest.get("applied_at"):
        apply_summary = manifest.get("apply_summary")
        facts: list[str] = []
        if manifest_path:
            facts.append(format_manifest_path_fact(manifest_path))
        last_summary = manifest.get("last_check_summary")
        if isinstance(last_summary, dict):
            quality_gate = last_summary.get("quality_gate")
            if isinstance(quality_gate, dict) and (
                int(quality_gate.get("warning_count") or 0) > 0
                or int(quality_gate.get("blocker_count") or 0) > 0
            ):
                facts.append(format_quality_gate_fact(quality_gate, prefix="写回后质量检查"))
        if isinstance(apply_summary, dict):
            applied_files = apply_summary.get("applied_files")
            applied_lines = apply_summary.get("applied_lines")
            if isinstance(applied_files, int) and isinstance(applied_lines, int):
                facts.append(f"已写回 {applied_files} 个文件，{applied_lines} 处译文行")
        next_split_manifest = manifest.get("next_split_manifest_path")
        message = "该任务已经写回过。"
        if isinstance(next_split_manifest, str) and next_split_manifest.strip():
            facts.append(f"下一拆分包：{next_split_manifest.strip()}")
            message = "该任务已经写回过，下一拆分包已准备继续。"
        return WritebackSummary(
            status="applied",
            heading="翻译已写回",
            message=message,
            facts=facts,
            findings=[],
            can_apply=False,
            manifest_path=manifest_path,
        )

    apply_failure_path = manifest.get("last_apply_failure_report_path")
    if isinstance(apply_failure_path, str) and apply_failure_path.strip():
        facts = [format_manifest_path_fact(manifest_path)] if manifest_path else []
        facts.append(f"写回失败报告：{apply_failure_path.strip()}")
        return WritebackSummary(
            status="failed",
            heading="写回失败",
            message="最近一次写回未成功。可点击「查看写回失败报告」了解原因，处理后再重新检查。",
            facts=facts,
            findings=[],
            can_apply=False,
            manifest_path=manifest_path,
        )

    last_summary = manifest.get("last_check_summary")
    if not isinstance(last_summary, dict):
        return None

    safety = last_summary.get("safety_level")
    safety_text = safety if isinstance(safety, str) else ""
    check_status = last_summary.get("check_status")
    check_status_text = check_status if isinstance(check_status, str) else safety_text
    writeback_gate = last_summary.get("writeback_gate")
    can_apply = _can_apply_from_manifest_gate(writeback_gate, check_status_text, safety_text)
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))

    pending_files = last_summary.get("pending_files")
    pending_lines = last_summary.get("pending_lines")
    if isinstance(pending_files, int) and isinstance(pending_lines, int):
        facts.append(f"将影响 {pending_files} 个文件，约 {pending_lines} 处译文行")

    failure_items = last_summary.get("failure_items")
    if isinstance(failure_items, int):
        facts.append(f"失败项：{failure_items}")

    report_path = manifest.get("last_check_report_path")
    if isinstance(report_path, str) and report_path.strip():
        facts.append(f"检查报告：{report_path}")

    quality_findings_path = manifest.get("last_quality_findings_path")
    if isinstance(quality_findings_path, str) and quality_findings_path.strip():
        facts.append(f"质量检查报告：{quality_findings_path}")

    quality_gate = last_summary.get("quality_gate")
    if isinstance(quality_gate, dict):
        facts.append(format_quality_gate_fact(quality_gate))

    findings: list[str] = []
    safety_reasons = last_summary.get("safety_reasons")
    if isinstance(safety_reasons, dict):
        for level in ("warn", "block"):
            reasons = safety_reasons.get(level)
            if isinstance(reasons, dict):
                for name, count in sorted(reasons.items()):
                    findings.append(f"[{safety_level_label(level)}] {name}: {count}")

    quality_reason_counts = last_summary.get("quality_reason_counts")
    if isinstance(quality_reason_counts, dict):
        for name, count in sorted(quality_reason_counts.items()):
            findings.append(f"[质量报警] {name}: {count}")

    if can_apply:
        has_warnings = (
            isinstance(quality_gate, dict)
            and (
                int(quality_gate.get("warning_count") or 0) > 0
                or int(quality_gate.get("blocker_count") or 0) > 0
            )
        ) or bool(quality_reason_counts)
        if has_warnings:
            return WritebackSummary(
                status="safe",
                heading="可以写回翻译（有质量报警）",
                message=(
                    "最近一次检查满足写回条件，但存在需要人工关注的质量问题。"
                    "可先写回，再按规则、文件或严重程度筛选报警并处理。"
                    + QUALITY_DELIVERY_NOTICE
                ),
                facts=extend_facts_with_notices(facts, findings),
                findings=findings,
                can_apply=True,
                manifest_path=manifest_path,
            )
        return WritebackSummary(
            status="safe",
            heading="可以写回翻译",
            message="最近一次检查结果为可写回。写回前请确认已备份项目。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=True,
            manifest_path=manifest_path,
        )
    if safety_text == "warn":
        return WritebackSummary(
            status="warn",
            heading="需要先处理问题",
            message=(
                "最近一次检查结果为需处理，暂不能写回。"
                "可先查看问题清单，必要时生成「补译包」并预览；"
                "处理完重新检查后，显示「可写回」才能写入项目。"
            ),
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=False,
            manifest_path=manifest_path,
        )
    if safety_text == "block" or check_status_text in {"blocked", "block"}:
        return WritebackSummary(
            status="block",
            heading="当前不能写回",
            message="最近一次检查结果为禁止写回。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
            can_apply=False,
            manifest_path=manifest_path,
        )
    return None


def idle_writeback_summary() -> WritebackSummary:
    return WritebackSummary(
        status="idle",
        heading="等待翻译完成",
        message="翻译完成并检查结果后，这里会显示是否可以写回。",
        facts=[],
        findings=[],
        can_apply=False,
    )


def idle_writeback_summary_for_work_mode(mode) -> WritebackSummary:
    from .work_modes import WorkMode, work_mode_spec

    spec = work_mode_spec(mode)
    if not spec.supports_translation_writeback:
        if spec.mode == WorkMode.SYNC_TRANSLATION:
            message = "同步翻译默认只生成差异预览；确认预览后才会通过同步任务页写回。"
        elif spec.mode == WorkMode.KEYWORD_EXTRACTION:
            message = "关键词模式只生成报告，不会修改游戏脚本。"
        elif spec.mode == WorkMode.REVISION:
            message = (
                "订正写回与普通翻译分开；请先在左侧「订正」生成预览，"
                "再在结果区点击「写回订正」确认。"
            )
        elif spec.mode == WorkMode.SYNC_REVISION:
            message = (
                "同步订正默认只出预览报告；请先在左侧「订正」生成预览，再在结果区点击「写回订正」。"
            )
        elif spec.mode == WorkMode.FINAL_REVIEW:
            message = (
                "最终审校必须先生成问题报告；只有人工选择的问题会进入订正预览，"
                "确认预览后才可写回所选订正。"
            )
        elif spec.mode == WorkMode.PROJECT_ANALYSIS:
            message = (
                "项目分析不会写回游戏脚本。请到左侧「上下文库」使用"
                "「审查内容」核对摘要，确认后再选择「启用到翻译」。"
            )
        elif spec.is_bootstrap:
            message = "预建库只更新本地上下文存储，不会启用普通「写回翻译」按钮。"
        else:
            message = "当前任务不使用普通翻译写回。"
        return WritebackSummary(
            status="idle",
            heading="此模式不写回翻译",
            message=message,
            facts=[],
            findings=[],
            can_apply=False,
        )
    return idle_writeback_summary()


def stale_writeback_summary() -> WritebackSummary:
    return WritebackSummary(
        status="stale",
        heading="写回状态已过期",
        message="项目或任务已切换；请针对当前任务重新检查后再决定是否写回。",
        facts=[],
        findings=[],
        can_apply=False,
    )


def recheck_writeback_ready(
    summary: WritebackSummary,
    *,
    supports_translation_writeback: bool,
) -> bool:
    if not supports_translation_writeback:
        return False
    if not summary.manifest_path:
        return False
    return summary.status not in {"idle", "running"}


def build_recheck_cli_args(manifest_path: str) -> list[str]:
    return ["check", manifest_path, "--output", "json", "--non-interactive"]


def running_writeback_summary(
    *,
    manifest_path: str = "",
    heading: str = "正在写回翻译",
    message: str = "正在写回；完成后这里会显示写回摘要。",
) -> WritebackSummary:
    facts = [format_manifest_path_fact(manifest_path)] if manifest_path else []
    return WritebackSummary(
        status="running",
        heading=heading,
        message=message,
        facts=facts,
        findings=[],
        can_apply=False,
        manifest_path=manifest_path,
    )
