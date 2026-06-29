"""User-facing summaries for the GUI doctor command."""
# ruff: noqa: RUF001
from __future__ import annotations

import re
from dataclasses import dataclass

from .summary_helpers import append_unique_fact
from .user_copy import (
    doctor_mode_label,
    format_doctor_recommendation_fact,
    format_doctor_warning_fact,
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
    "can_generate_template": "Ren'Py 模板生成环境可用；翻译模板尚未生成。",
    "existing_tl_only": "已有翻译文件可处理；模板生成环境不可用，后续依赖现有翻译文件。",
    "blocked_missing_template": "缺少可处理的翻译文件，也无法自动生成模板。",
}

LAYOUT_STATUS_HEADINGS = {
    "failed": ("blocked", "项目检查失败"),
    "switch_to_work": ("warning", "建议使用 work 目录"),
    "ready": ("ready", "项目检查通过"),
    "attention": ("warning", "检查完成，但有需要处理的事项"),
}

LAYOUT_STATUS_MESSAGES = {
    "failed": "当前项目目录下没有 work 目录、没有可翻译内容，也未找到 original/game。",
    "failed_on_work": "work 目录为空，没有可翻译内容，且无法自动生成翻译模板。",
    "switch_to_work": "当前路径不是 work 目录，但项目内已有可继续处理的内容。",
    "ready": "work 目录已就绪，可以开始翻译流程。",
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


def _parse_context_fields(raw_fields: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    for part in raw_fields.split(", "):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        bool_value = _parse_bool(value)
        if bool_value is not None:
            fields[key] = bool_value
        elif re.fullmatch(r"-?\d+", value):
            fields[key] = int(value)
        else:
            fields[key] = value
    return fields


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
            facts.append("待翻译条目：0 条（当前没有需要批量翻译的待译行）")

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


def _int_context_value(context: dict[str, object], key: str) -> int:
    value = context.get(key)
    return value if isinstance(value, int) else 0


def format_context_status_facts(
    rag_context: dict[str, object] | None,
    source_index_context: dict[str, object] | None,
) -> list[str]:
    facts: list[str] = []

    if rag_context:
        if rag_context.get("enabled") is True:
            records = _int_context_value(rag_context, "history_records")
            bootstrap_on_build = rag_context.get("bootstrap_on_build") is True
            if rag_context.get("store_exists") is not True:
                detail = "尚未创建"
                if bootstrap_on_build:
                    detail += "（build 会自动补建）"
                else:
                    detail += "（建议先预建记忆库）"
            elif records == 0:
                detail = "记录数 0"
                if bootstrap_on_build:
                    detail += "（build 会自动补建）"
                else:
                    detail += "（建议先预建记忆库）"
            else:
                detail = f"记录数 {records}"
            facts.append(f"记忆库：已启用，{detail}")
            if rag_context.get("store_dir"):
                facts.append(f"记忆库路径：{rag_context['store_dir']}")
            if rag_context.get("error"):
                facts.append(f"记忆库读取异常：{rag_context['error']}")
        elif rag_context.get("enabled") is False:
            facts.append("记忆库：未启用")

    if source_index_context:
        if source_index_context.get("enabled") is True:
            segments = _int_context_value(source_index_context, "source_segments")
            schema_version = source_index_context.get("schema_version")
            if source_index_context.get("store_exists") is not True:
                detail = "尚未创建（建议先预建原文索引）"
            elif segments == 0:
                detail = "片段数 0（建议先预建原文索引）"
            else:
                detail = f"片段数 {segments}"
            if schema_version:
                detail += f"，schema v{schema_version}"
            facts.append(f"原文索引：已启用，{detail}")
            if source_index_context.get("store_dir"):
                facts.append(f"原文索引路径：{source_index_context['store_dir']}")
            if source_index_context.get("error"):
                facts.append(f"原文索引读取异常：{source_index_context['error']}")
        elif source_index_context.get("enabled") is False:
            facts.append("原文索引：未启用")

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

        if line.startswith("- Is work root:"):
            parsed["is_work_root"] = _parse_bool(line.split(":", 1)[1].strip())
            continue

        if line.startswith("- Work dir:"):
            match = re.match(
                r"- Work dir:\s*(.*?)\s*\(exists:\s*(True|False),\s*empty:\s*(True|False)\)",
                line,
            )
            if match:
                parsed["work_dir"] = match.group(1).strip()
                parsed["work_exists"] = _parse_bool(match.group(2))
                parsed["work_empty"] = _parse_bool(match.group(3))
            continue

        if line.startswith("- Original game dir:"):
            value = line.split(":", 1)[1].strip()
            if value == "(not found)":
                parsed["original_game_dir"] = ""
                parsed["original_game_exists"] = False
            else:
                parsed["original_game_dir"] = value
                parsed["original_game_exists"] = True
            continue

        if line.startswith("- Layout status:"):
            parsed["layout_status"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("- TL scan:"):
            parsed["counts"] = _parse_counts(line.split(":", 1)[1])
            continue

        if line.startswith("- Pending translation:"):
            parsed["pending"] = _parse_counts(line.split(":", 1)[1])
            continue

        if line.startswith("- RAG context:"):
            parsed["rag_context"] = _parse_context_fields(line.split(":", 1)[1])
            continue

        if line.startswith("- Source index context:"):
            parsed["source_index_context"] = _parse_context_fields(line.split(":", 1)[1])
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
    recommendation_facts = [
        format_doctor_recommendation_fact(recommendation)
        for recommendation in parsed.get("recommendations", [])
        if isinstance(recommendation, str) and recommendation.strip()
    ]
    counts_value = parsed.get("counts")
    counts = counts_value if isinstance(counts_value, dict) else {}
    pending_value = parsed.get("pending")
    pending = pending_value if isinstance(pending_value, dict) else None
    mode = parsed.get("mode") if isinstance(parsed.get("mode"), str) else ""
    layout_status = parsed.get("layout_status") if isinstance(parsed.get("layout_status"), str) else ""

    facts: list[str] = []
    base_dir = parsed.get("base_dir") if isinstance(parsed.get("base_dir"), str) else ""
    work_dir = parsed.get("work_dir") if isinstance(parsed.get("work_dir"), str) else ""
    is_work_root = parsed.get("is_work_root")

    if is_work_root is False and base_dir:
        append_unique_fact(facts, f"项目目录：{base_dir}")
    elif base_dir:
        append_unique_fact(facts, f"work 目录：{base_dir}")

    if is_work_root is False:
        if parsed.get("work_exists") is False:
            append_unique_fact(facts, "work 目录：不存在")
        elif parsed.get("work_empty") is True:
            if work_dir:
                append_unique_fact(facts, f"work 目录：{work_dir}（为空）")
            else:
                append_unique_fact(facts, "work 目录：存在（为空）")
        elif parsed.get("work_exists") is True:
            if work_dir:
                append_unique_fact(facts, f"work 目录：{work_dir}")
            else:
                append_unique_fact(facts, "work 目录：存在")

    if (
        is_work_root is True
        and work_dir
        and base_dir
        and work_dir != base_dir
    ):
        append_unique_fact(facts, f"work 路径：{work_dir}")

    if parsed.get("original_game_exists") is True:
        append_unique_fact(facts, "original/game：存在")
    elif parsed.get("original_game_exists") is False:
        append_unique_fact(facts, "original/game：不存在")
    if parsed.get("tl_dir"):
        exists_text = "存在" if parsed.get("tl_exists") is True else "不存在"
        append_unique_fact(facts, f"翻译目录：{exists_text}")
    if parsed.get("language"):
        append_unique_fact(facts, f"目标语言：{parsed['language']}")
    if mode:
        append_unique_fact(facts, f"检查模式：{doctor_mode_label(mode)}")
    if counts:
        for fact in format_tl_scan_facts(counts, pending=pending):
            append_unique_fact(facts, fact)
    for fact in format_context_status_facts(
        parsed.get("rag_context") if isinstance(parsed.get("rag_context"), dict) else None,
        parsed.get("source_index_context")
        if isinstance(parsed.get("source_index_context"), dict)
        else None,
    ):
        append_unique_fact(facts, fact)

    findings = list(warnings)
    if api_key_count is not None:
        if api_key_count > 0:
            if api_key_source == "environment":
                append_unique_fact(
                    facts,
                    f"API 密钥：已通过环境变量配置 {api_key_count} 个",
                )
            else:
                append_unique_fact(facts, f"API 密钥：已配置 {api_key_count} 个")
        else:
            append_unique_fact(facts, "建议：在配置页填写 API 密钥后再开始翻译")

    for fact in recommendation_facts:
        append_unique_fact(facts, fact)
    for warning in warnings:
        append_unique_fact(facts, format_doctor_warning_fact(warning))

    if exit_code != 0:
        return DoctorSummary(
            status="blocked",
            heading="项目检查失败",
            message="环境检查没有正常完成，请查看诊断日志。",
            facts=facts,
            findings=findings,
        )

    if layout_status in LAYOUT_STATUS_HEADINGS:
        status, heading = LAYOUT_STATUS_HEADINGS[layout_status]
        if layout_status == "failed" and parsed.get("is_work_root") is True:
            message = LAYOUT_STATUS_MESSAGES["failed_on_work"]
        else:
            message = LAYOUT_STATUS_MESSAGES.get(
                layout_status,
                MODE_MESSAGES.get(mode, "项目检查已完成。"),
            )
        if layout_status == "attention":
            message = MODE_MESSAGES.get(mode, message)
        if layout_status == "ready" and api_key_count == 0:
            status = "warning"
            heading = LAYOUT_STATUS_HEADINGS["attention"][1]
        elif layout_status == "ready" and findings:
            status = "warning"
            heading = LAYOUT_STATUS_HEADINGS["attention"][1]
    elif mode == "blocked_missing_template":
        status = "blocked"
        heading = "项目检查失败"
        message = MODE_MESSAGES.get(mode, "项目检查失败。")
    else:
        needs_attention = bool(findings) or bool(recommendation_facts) or (
            api_key_count is not None and api_key_count == 0
        )
        if needs_attention:
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
        message="选择 work 目录（或项目根目录）后点击「环境检查」。",
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