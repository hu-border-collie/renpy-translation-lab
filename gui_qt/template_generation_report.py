"""User-facing summaries for GUI generate-template command."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .doctor_report import DoctorSummary
from .summary_helpers import append_unique_fact


TEMPLATE_GENERATION_HEADER = "Template generation summary:"


@dataclass(frozen=True)
class TemplateGenerationSummary:
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]
    tl_dir: str
    rpy_files: int


def _parse_summary_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    in_section = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == TEMPLATE_GENERATION_HEADER:
            in_section = True
            continue
        if in_section and line.startswith("- "):
            match = re.match(r"-\s*([a-z_]+):\s*(.*)$", line)
            if match:
                values[match.group(1)] = match.group(2).strip()
            continue
        if in_section and not line.startswith("- "):
            break
    return values


def running_template_generation_summary() -> TemplateGenerationSummary:
    return TemplateGenerationSummary(
        status="running",
        heading="正在生成翻译模板",
        message="正在准备 work/game 并调用 Ren'Py 生成翻译模板，请稍候。",
        facts=[],
        findings=[],
        tl_dir="",
        rpy_files=0,
    )


def summarize_template_generation_output(
    output: str,
    exit_code: int,
) -> TemplateGenerationSummary:
    values = _parse_summary_values(output)
    status = values.get("status", "")
    tl_dir = values.get("tl_dir", "")
    message = values.get("message", "")
    language = values.get("language", "")
    try:
        rpy_files = int(values.get("rpy_files", "0") or "0")
    except ValueError:
        rpy_files = 0

    facts: list[str] = []
    if language:
        append_unique_fact(facts, f"目标语言：{language}")
    if tl_dir:
        append_unique_fact(facts, f"翻译目录：{tl_dir}")
    if rpy_files > 0:
        append_unique_fact(facts, f"翻译文件：{rpy_files} 个")

    if exit_code != 0 or status == "failed":
        return TemplateGenerationSummary(
            status="blocked",
            heading="翻译模板生成失败",
            message=message or "翻译模板生成没有正常完成，请查看诊断日志。",
            facts=facts,
            findings=[],
            tl_dir=tl_dir,
            rpy_files=rpy_files,
        )

    ready_facts = list(facts)
    append_unique_fact(
        ready_facts,
        "建议：如需查看最新待译统计，可重新运行「环境检查」",
    )
    return TemplateGenerationSummary(
        status="ready",
        heading="翻译模板已生成",
        message="翻译模板已就绪，可以开始翻译流程。",
        facts=ready_facts,
        findings=[],
        tl_dir=tl_dir,
        rpy_files=rpy_files,
    )


def template_generation_to_doctor_summary(
    summary: TemplateGenerationSummary,
) -> DoctorSummary:
    mode = ""
    if summary.status == "ready" and summary.rpy_files > 0:
        mode = "existing_tl_only"
    return DoctorSummary(
        status=summary.status,
        heading=summary.heading,
        message=summary.message,
        facts=summary.facts,
        findings=summary.findings,
        mode=mode,
    )