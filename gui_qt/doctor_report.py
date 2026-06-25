"""User-facing summaries for the GUI doctor command."""
# ruff: noqa: RUF001
from __future__ import annotations

import re
from dataclasses import dataclass

from .user_copy import (
    doctor_mode_label,
    translate_doctor_recommendation,
    translate_doctor_warning,
)


@dataclass(frozen=True)
class DoctorSummary:
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]


MODE_MESSAGES = {
    "can_generate_template": (
        "Ren'Py 模板生成环境可用；如翻译模板尚不存在，可先准备工作目录，再运行开始翻译（build）。"
    ),
    "existing_tl_only": "已有翻译文件可处理；模板生成环境不可用，后续依赖现有翻译文件。",
    "blocked_missing_template": (
        "缺少可处理的翻译文件，也无法自动生成模板；如有 original/game，可先准备工作目录。"
    ),
}


def _parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _parse_counts(raw_counts: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in re.findall(r"([a-z_]+)=(-?\d+)", raw_counts):
        counts[key] = int(value)
    return counts


def format_tl_scan_facts(
    counts: dict[str, int],
    pending: dict[str, int] | None = None,
) -> list[str]:
    """Turn doctor TL scan counters into user-facing facts."""
    rpy_files = int(counts.get("rpy_files", 0))
    translate_blocks = int(counts.get("translate_blocks", 0))
    commented_lines = int(counts.get("commented_original_lines", 0))
    old_lines = int(counts.get("old_lines", 0))
    new_lines = int(counts.get("new_lines", 0))

    if rpy_files == 0 and translate_blocks == 0:
        return ["翻译文件：0 个"]

    facts: list[str] = [f"翻译文件：{rpy_files} 个"]

    if pending is not None:
        task_count = int(pending.get("task_count", 0))
        file_count = int(pending.get("file_count", 0))
        if task_count > 0:
            facts.append(
                f"待翻译条目：约 {task_count} 条（分布在 {file_count} 个文件中，与 build 统计一致）"
            )
        else:
            facts.append("待翻译条目：0 条（当前没有需要提交到 Batch 的英文待译行）")

    if commented_lines > 0:
        facts.append(f"剧情对话：{commented_lines} 条")
    elif translate_blocks > 0:
        facts.append(f"翻译块：{translate_blocks} 个")

    if old_lines > 0 or new_lines > 0:
        if old_lines == new_lines:
            facts.append(f"界面字符串：{old_lines} 条")
        else:
            facts.append(
                f"界面字符串：原文 {old_lines} 条，译文 {new_lines} 条（可能格式异常）"
            )

    return facts


def parse_doctor_output(output: str) -> dict[str, object]:
    parsed: dict[str, object] = {"warnings": [], "recommendations": []}
    in_warnings = False
    in_recommendations = False

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line == "Warnings:":
            in_warnings = True
            in_recommendations = False
            continue

        if line == "Recommendations:":
            in_recommendations = True
            in_warnings = False
            continue

        if in_warnings:
            if line.startswith("- "):
                parsed.setdefault("warnings", []).append(line[2:].strip())
                continue
            in_warnings = False

        if in_recommendations:
            if line.startswith("- "):
                parsed.setdefault("recommendations", []).append(line[2:].strip())
                continue
            in_recommendations = False

        if line.startswith("- Base dir:"):
            parsed["base_dir"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("- TL dir:"):
            match = re.match(r"- TL dir:\s*(.*?)\s*\(exists:\s*(True|False)\)", line)
            if match:
                parsed["tl_dir"] = match.group(1).strip()
                parsed["tl_exists"] = _parse_bool(match.group(2))
            continue

        if line.startswith("- Language:"):
            parsed["language"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("- Template generation:"):
            parsed["template_generation"] = line.split(":", 1)[1].strip()
            parsed["can_generate_template"] = line.startswith("- Template generation: available")
            continue

        if line.startswith("- Mode:"):
            parsed["mode"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("- TL scan:"):
            parsed["counts"] = _parse_counts(line.split(":", 1)[1])
            continue

        if line.startswith("- Pending translation:"):
            parsed["pending"] = _parse_counts(line.split(":", 1)[1])
            continue

    return parsed


def summarize_doctor_output(
    output: str,
    exit_code: int,
    api_key_count: int | None = None,
    api_key_source: str = "",
) -> DoctorSummary:
    parsed = parse_doctor_output(output)
    warnings = [
        translate_doctor_warning(warning)
        for warning in parsed.get("warnings", [])
        if isinstance(warning, str) and warning.strip()
    ]
    recommendations = [
        translate_doctor_recommendation(recommendation)
        for recommendation in parsed.get("recommendations", [])
        if isinstance(recommendation, str) and recommendation.strip()
    ]
    counts_value = parsed.get("counts")
    counts = counts_value if isinstance(counts_value, dict) else {}
    pending_value = parsed.get("pending")
    pending = pending_value if isinstance(pending_value, dict) else None
    mode = parsed.get("mode") if isinstance(parsed.get("mode"), str) else ""

    facts: list[str] = []
    if parsed.get("base_dir"):
        facts.append(f"项目目录：{parsed['base_dir']}")
    if parsed.get("tl_dir"):
        exists_text = "存在" if parsed.get("tl_exists") is True else "不存在"
        facts.append(f"翻译目录：{exists_text}")
    if parsed.get("language"):
        facts.append(f"目标语言：{parsed['language']}")
    if mode:
        facts.append(f"检查模式：{doctor_mode_label(mode)}")
    if counts:
        facts.extend(format_tl_scan_facts(counts, pending=pending))

    findings = list(warnings) + list(recommendations)
    if mode == "can_generate_template":
        if parsed.get("tl_exists") is False:
            findings.append(
                "翻译目录尚不存在；可先点击「准备工作目录」，再运行「开始翻译」生成模板，然后重新检查。"
            )
        elif counts and int(counts.get("rpy_files", 0)) == 0:
            findings.append(
                "翻译目录中没有翻译文件；可先准备工作目录，再通过「开始翻译」生成模板后重新检查。"
            )
    if api_key_count is not None:
        if api_key_count > 0:
            if api_key_source == "environment":
                facts.append(f"API 密钥：已通过环境变量配置 {api_key_count} 个")
            else:
                facts.append(f"API 密钥：已配置 {api_key_count} 个")
        else:
            findings.append("尚未配置 API 密钥；环境检查不调用模型，但翻译任务需要密钥。")

    if exit_code != 0:
        return DoctorSummary(
            status="blocked",
            heading="项目检查失败",
            message="环境检查没有正常完成，请查看诊断日志。",
            facts=facts,
            findings=findings,
        )

    if mode == "blocked_missing_template":
        status = "blocked"
        heading = "需要先准备翻译模板"
    elif findings:
        status = "warning"
        heading = "检查完成，但有需要处理的事项"
    else:
        status = "ready"
        heading = "项目检查通过"

    message = MODE_MESSAGES.get(mode, "项目检查已完成。")
    return DoctorSummary(
        status=status,
        heading=heading,
        message=message,
        facts=facts,
        findings=findings,
    )


def running_summary() -> DoctorSummary:
    return DoctorSummary(
        status="running",
        heading="正在检查项目",
        message="正在运行环境检查；完成后这里会显示摘要。",
        facts=[],
        findings=[],
    )


def idle_summary() -> DoctorSummary:
    return DoctorSummary(
        status="idle",
        heading="尚未运行项目检查",
        message="选择游戏 work 目录后点击「环境检查」。",
        facts=[],
        findings=[],
    )


def stale_summary() -> DoctorSummary:
    return DoctorSummary(
        status="stale",
        heading="项目已切换，请重新运行检查",
        message="当前摘要已清空；请针对新的 work 目录重新运行环境检查。",
        facts=[],
        findings=[],
    )