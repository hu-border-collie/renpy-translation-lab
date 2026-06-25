"""User-facing summaries for GUI bootstrap-work command."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .doctor_report import DoctorSummary


WORK_BOOTSTRAP_HEADER = "Work bootstrap summary:"


@dataclass(frozen=True)
class WorkBootstrapSummary:
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]
    work_dir: str
    game_root_updated: bool


def _parse_summary_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    in_section = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == WORK_BOOTSTRAP_HEADER:
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


def running_work_bootstrap_summary() -> WorkBootstrapSummary:
    return WorkBootstrapSummary(
        status="running",
        heading="正在准备工作目录",
        message="正在从 original/game 复制到 work/game，请稍候。",
        facts=[],
        findings=[],
        work_dir="",
        game_root_updated=False,
    )


def summarize_work_bootstrap_output(output: str, exit_code: int) -> WorkBootstrapSummary:
    values = _parse_summary_values(output)
    status = values.get("status", "")
    work_dir = values.get("work_dir", "")
    files_copied = values.get("files_copied", "0")
    message = values.get("message", "")
    game_root_updated = values.get("game_root_updated", "").lower() == "true"

    facts: list[str] = []
    if work_dir:
        facts.append(f"work 目录：{work_dir}")
    if files_copied and files_copied != "0":
        facts.append(f"复制文件数：{files_copied}")
    if game_root_updated:
        facts.append("已自动将 game_root 更新为 work 目录")

    findings: list[str] = []
    if message and status == "skipped":
        findings.append(message)

    if exit_code != 0:
        return WorkBootstrapSummary(
            status="failed",
            heading="准备工作目录失败",
            message="命令未正常完成，请查看诊断日志。",
            facts=facts,
            findings=findings or ["请确认存在 original/game，且 work 目录不存在或为空。"],
            work_dir=work_dir,
            game_root_updated=False,
        )

    if status == "created":
        facts.append("建议：点击「环境检查」确认项目状态")
        return WorkBootstrapSummary(
            status="ready",
            heading="工作目录已准备完成",
            message="已从 original/game 复制到 work/game。",
            facts=facts,
            findings=findings,
            work_dir=work_dir,
            game_root_updated=game_root_updated,
        )

    if status == "skipped":
        return WorkBootstrapSummary(
            status="warning",
            heading="未准备工作目录",
            message="work 目录已存在且非空，未做任何修改。",
            facts=facts,
            findings=findings or [message or "如需重新初始化，请先手动清空或移除 work 目录。"],
            work_dir=work_dir,
            game_root_updated=False,
        )

    if status == "failed":
        return WorkBootstrapSummary(
            status="failed",
            heading="准备工作目录失败",
            message=message or "未找到 original/game，或复制过程中出错。",
            facts=facts,
            findings=findings,
            work_dir=work_dir,
            game_root_updated=False,
        )

    return WorkBootstrapSummary(
        status="warning",
        heading="准备工作目录已结束",
        message=message or "请查看诊断日志了解详情。",
        facts=facts,
        findings=findings,
        work_dir=work_dir,
        game_root_updated=game_root_updated,
    )


def work_bootstrap_to_doctor_summary(summary: WorkBootstrapSummary) -> DoctorSummary:
    return DoctorSummary(
        status=summary.status if summary.status != "ready" else "ready",
        heading=summary.heading,
        message=summary.message,
        facts=summary.facts,
        findings=summary.findings,
    )