"""User-facing summaries for the GUI doctor command."""
# ruff: noqa: RUF001
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import doctor_recommendations as doctor_rec

from .summary_helpers import append_unique_fact
from .user_copy import (
    doctor_mode_label,
    findings_require_attention,
    format_doctor_recommendation_fact,
    format_doctor_warning_fact,
    primary_recommendation_message,
    recommendation_requires_attention,
    translate_doctor_warning,
    workflow_state_message,
)


@dataclass(frozen=True)
class DoctorSummary:
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]
    mode: str = ""
    # Secondary lines for the collapsible "更多详情" section (never discarded).
    detail_facts: list[str] | None = None


MODE_MESSAGES = {
    "can_generate_template": "Ren'Py 模板生成环境可用；翻译模板尚未生成。",
    "existing_tl_only": "翻译模板已就绪，可以开始翻译流程。",
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
    """Turn doctor TL scan counters into user-facing facts (full set; UI may split)."""
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
                f"待翻译条目：约 {task_count} 条"
                f"（{file_count} 个文件；可能含专名/名单等无汉字英文，不代表漏翻）"
            )
        else:
            facts.append("待翻译条目：0 条")

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
    """Full context-store facts (paths included; UI may fold secondary lines)."""
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
            expected = _int_context_value(source_index_context, "expected_segments")
            schema_version = source_index_context.get("schema_version")
            if source_index_context.get("store_exists") is not True:
                detail = "尚未创建（建议先预建原文索引）"
            elif segments == 0:
                detail = "片段数 0（建议先预建原文索引）"
            elif expected > 0 and segments < expected:
                percent = (segments * 100) // expected
                detail = f"片段数 {segments}/{expected}（约 {percent}%，预建未完成）"
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


def format_project_assets_facts(project_assets: dict[str, object] | None) -> list[str]:
    """Full project-asset facts (healthy and problem lines; UI may fold healthy ones)."""
    if not project_assets:
        return []

    facts: list[str] = []
    glossary_exists = project_assets.get("glossary_exists") is True
    macro_exists = project_assets.get("macro_exists") is True
    glossary_matches = project_assets.get("glossary_matches_project")
    macro_matches = project_assets.get("macro_matches_project")

    if glossary_matches is False:
        facts.append("术语表：路径与当前项目不匹配")
    elif glossary_exists:
        facts.append("术语表：已找到 glossary.json")
    else:
        facts.append("术语表：当前项目缺少 glossary.json（将使用默认保留词）")

    if macro_matches is False:
        facts.append("风格设定：路径与当前项目不匹配")
    elif macro_exists:
        facts.append("风格设定：已找到 macro_setting.md")
    else:
        facts.append("风格设定：当前项目缺少 macro_setting.md（批量翻译无项目口吻指引）")

    glossary_file = project_assets.get("glossary_file")
    if isinstance(glossary_file, str) and glossary_file.strip():
        facts.append(f"术语表路径：{glossary_file}")

    macro_file = project_assets.get("macro_setting_file")
    if isinstance(macro_file, str) and macro_file.strip():
        facts.append(f"风格设定路径：{macro_file}")

    return facts


def _is_detail_fact(fact: str, *, layout_status: str) -> bool:
    """Secondary lines go under the collapsible details section, not deleted."""
    text = fact.strip()
    if not text:
        return False
    if text.startswith(
        (
            "记忆库路径：",
            "原文索引路径：",
            "术语表路径：",
            "风格设定路径：",
            "TL 路径：",
            "检查模式：",
            "work 路径：",
        )
    ):
        return True
    if text in {
        "original/game：存在",
        "翻译目录：存在",
        "记忆库：未启用",
        "原文索引：未启用",
    }:
        return True
    if text.startswith("术语表：已找到") or text.startswith("风格设定：已找到"):
        return True
    if text.startswith("API 密钥：已配置") or text.startswith("API 密钥：已通过环境变量"):
        return True
    # Healthy work path on a ready work root is orientation noise.
    if (
        layout_status == "ready"
        and text.startswith("work 目录：")
        and "不存在" not in text
        and "为空" not in text
    ):
        return True
    return False


def partition_doctor_facts(
    all_facts: list[str],
    *,
    layout_status: str = "",
) -> tuple[list[str], list[str]]:
    primary: list[str] = []
    details: list[str] = []
    for fact in all_facts:
        if _is_detail_fact(fact, layout_status=layout_status):
            details.append(fact)
        else:
            primary.append(fact)
    return primary, details


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
                rec = doctor_rec.parse_doctor_recommendation_cli_line(line[2:].strip())
                if rec is not None:
                    parsed.setdefault("recommendations", []).append(rec)
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

        if line.startswith("- Workflow state:"):
            parsed["workflow_state"] = line.split(":", 1)[1].strip()
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

        if line.startswith("- Project assets:"):
            parsed["project_assets"] = _parse_context_fields(line.split(":", 1)[1])
            continue

    return parsed


def doctor_report_to_parsed(report: dict[str, Any]) -> dict[str, object]:
    """Normalize ``collect_doctor_report()`` output for summary rendering."""
    counts_value = report.get("counts")
    counts = counts_value if isinstance(counts_value, dict) else {}
    context_status = report.get("context_status")
    context_status = context_status if isinstance(context_status, dict) else {}
    original_game_dir = report.get("original_game_dir")
    original_game_dir = original_game_dir if isinstance(original_game_dir, str) else ""
    original_game_exists: bool | None
    if "original_game_dir" in report:
        original_game_exists = bool(original_game_dir)
    else:
        original_game_exists = None

    parsed: dict[str, object] = {
        "base_dir": str(report.get("base_dir") or ""),
        "tl_dir": str(report.get("tl_dir") or ""),
        "tl_subdir": str(report.get("tl_subdir") or ""),
        "tl_exists": report.get("tl_exists"),
        "language": str(report.get("language") or ""),
        "mode": str(report.get("mode") or ""),
        "layout_status": str(report.get("layout_status") or ""),
        "is_work_root": report.get("is_work_root"),
        "workflow_state": str(report.get("workflow_state") or ""),
        "work_dir": str(report.get("work_dir") or ""),
        "work_exists": report.get("work_exists"),
        "work_empty": report.get("work_empty"),
        "original_game_dir": original_game_dir,
        "original_game_exists": original_game_exists,
        "counts": counts,
        "warnings": [
            warning
            for warning in (report.get("warnings") or [])
            if isinstance(warning, str) and warning.strip()
        ],
        "recommendations": [
            doctor_rec.normalize_doctor_recommendation(recommendation)
            for recommendation in (report.get("recommendations") or [])
            if recommendation
        ],
    }

    if (
        report.get("tl_exists")
        and int(counts.get("rpy_files") or 0) > 0
        and (
            "pending_task_count" in report
            or "pending_file_count" in report
        )
    ):
        parsed["pending"] = {
            "task_count": int(report.get("pending_task_count") or 0),
            "file_count": int(report.get("pending_file_count") or 0),
        }

    rag_context = context_status.get("rag")
    if isinstance(rag_context, dict):
        parsed["rag_context"] = rag_context

    source_index_context = context_status.get("source_index")
    if isinstance(source_index_context, dict):
        parsed["source_index_context"] = source_index_context

    project_assets = report.get("project_assets")
    if isinstance(project_assets, dict):
        parsed["project_assets"] = project_assets

    return parsed


def _summarize_doctor_parsed(
    parsed: dict[str, object],
    exit_code: int,
    api_key_count: int | None = None,
    api_key_source: str = "",
) -> DoctorSummary:
    warnings = [
        translate_doctor_warning(warning)
        for warning in parsed.get("warnings", [])
        if isinstance(warning, str) and warning.strip()
    ]
    normalized_recommendations = [
        doctor_rec.normalize_doctor_recommendation(recommendation)
        for recommendation in parsed.get("recommendations", [])
        if recommendation
    ]
    recommendation_codes = doctor_rec.doctor_recommendation_codes(normalized_recommendations)
    recommendation_facts = [
        format_doctor_recommendation_fact(recommendation)
        for recommendation in normalized_recommendations
    ]
    counts_value = parsed.get("counts")
    counts = counts_value if isinstance(counts_value, dict) else {}
    pending_value = parsed.get("pending")
    pending = pending_value if isinstance(pending_value, dict) else None
    mode = parsed.get("mode") if isinstance(parsed.get("mode"), str) else ""
    layout_status = parsed.get("layout_status") if isinstance(parsed.get("layout_status"), str) else ""

    workflow_state = (
        parsed.get("workflow_state") if isinstance(parsed.get("workflow_state"), str) else ""
    )
    all_facts: list[str] = []
    base_dir = parsed.get("base_dir") if isinstance(parsed.get("base_dir"), str) else ""
    work_dir = parsed.get("work_dir") if isinstance(parsed.get("work_dir"), str) else ""
    is_work_root = parsed.get("is_work_root")

    if is_work_root is False and base_dir:
        append_unique_fact(all_facts, f"项目目录：{base_dir}")
    elif base_dir:
        append_unique_fact(all_facts, f"work 目录：{base_dir}")

    if is_work_root is False:
        if parsed.get("work_exists") is False:
            append_unique_fact(all_facts, "work 目录：不存在")
        elif parsed.get("work_empty") is True:
            if work_dir:
                append_unique_fact(all_facts, f"work 目录：{work_dir}（为空）")
            else:
                append_unique_fact(all_facts, "work 目录：存在（为空）")
        elif parsed.get("work_exists") is True:
            if work_dir:
                append_unique_fact(all_facts, f"work 目录：{work_dir}")
            else:
                append_unique_fact(all_facts, "work 目录：存在")

    if (
        is_work_root is True
        and work_dir
        and base_dir
        and work_dir != base_dir
    ):
        append_unique_fact(all_facts, f"work 路径：{work_dir}")

    if parsed.get("original_game_exists") is True:
        append_unique_fact(all_facts, "original/game：存在")
    elif parsed.get("original_game_exists") is False:
        append_unique_fact(all_facts, "original/game：不存在")
    if parsed.get("tl_dir"):
        exists_text = "存在" if parsed.get("tl_exists") is True else "不存在"
        append_unique_fact(all_facts, f"翻译目录：{exists_text}")
    if parsed.get("tl_subdir"):
        append_unique_fact(all_facts, f"TL 路径：{parsed['tl_subdir']}")
    if parsed.get("language"):
        append_unique_fact(all_facts, f"目标语言：{parsed['language']}")
    if mode:
        append_unique_fact(all_facts, f"检查模式：{doctor_mode_label(mode)}")
    if counts:
        for fact in format_tl_scan_facts(counts, pending=pending):
            append_unique_fact(all_facts, fact)
    for fact in format_context_status_facts(
        parsed.get("rag_context") if isinstance(parsed.get("rag_context"), dict) else None,
        parsed.get("source_index_context")
        if isinstance(parsed.get("source_index_context"), dict)
        else None,
    ):
        append_unique_fact(all_facts, fact)
    for fact in format_project_assets_facts(
        parsed.get("project_assets")
        if isinstance(parsed.get("project_assets"), dict)
        else None,
    ):
        append_unique_fact(all_facts, fact)

    findings = list(warnings)
    if api_key_count is not None:
        if api_key_count > 0:
            if api_key_source == "environment":
                append_unique_fact(
                    all_facts,
                    f"API 密钥：已通过环境变量配置 {api_key_count} 个",
                )
            else:
                append_unique_fact(all_facts, f"API 密钥：已配置 {api_key_count} 个")
        else:
            append_unique_fact(all_facts, "建议：在配置页填写 API 密钥后再开始翻译")

    recommendation_message = primary_recommendation_message(recommendation_codes)
    normal_state_message = workflow_state_message(workflow_state)

    for fact in recommendation_facts:
        append_unique_fact(all_facts, fact)
    for warning in warnings:
        append_unique_fact(all_facts, format_doctor_warning_fact(warning))

    facts, detail_facts = partition_doctor_facts(
        all_facts,
        layout_status=layout_status,
    )

    if exit_code != 0:
        return DoctorSummary(
            status="blocked",
            heading="项目检查失败",
            message="环境检查没有正常完成，请查看诊断日志。",
            facts=facts,
            findings=findings,
            detail_facts=detail_facts,
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
        elif layout_status == "ready" and findings_require_attention(findings):
            status = "warning"
            heading = LAYOUT_STATUS_HEADINGS["attention"][1]
    elif mode == "blocked_missing_template":
        status = "blocked"
        heading = "项目检查失败"
        message = MODE_MESSAGES.get(mode, "项目检查失败。")
    else:
        needs_attention = (
            findings_require_attention(findings)
            or recommendation_requires_attention(recommendation_codes)
            or (api_key_count is not None and api_key_count == 0)
        )
        if needs_attention:
            status = "warning"
            heading = "检查完成，但有需要处理的事项"
        else:
            status = "ready"
            heading = "项目检查通过"
        message = MODE_MESSAGES.get(mode, "项目检查已完成。")

    if recommendation_message:
        message = recommendation_message
        if recommendation_requires_attention(recommendation_codes) and status == "ready":
            status = "warning"
            heading = LAYOUT_STATUS_HEADINGS["attention"][1]

    elif normal_state_message:
        message = normal_state_message
    return DoctorSummary(
        status=status,
        heading=heading,
        message=message,
        facts=facts,
        findings=findings,
        mode=mode,
        detail_facts=detail_facts,
    )


def summarize_doctor_report(
    report: dict[str, Any],
    exit_code: int = 0,
    api_key_count: int | None = None,
    api_key_source: str = "",
) -> DoctorSummary:
    """Build a GUI summary directly from ``collect_doctor_report()`` output."""
    return _summarize_doctor_parsed(
        doctor_report_to_parsed(report),
        exit_code,
        api_key_count=api_key_count,
        api_key_source=api_key_source,
    )


def summarize_doctor_output(
    output: str,
    exit_code: int,
    api_key_count: int | None = None,
    api_key_source: str = "",
) -> DoctorSummary:
    return _summarize_doctor_parsed(
        parse_doctor_output(output),
        exit_code,
        api_key_count=api_key_count,
        api_key_source=api_key_source,
    )


def running_summary() -> DoctorSummary:
    return DoctorSummary(
        status="running",
        heading="正在检查项目",
        message="正在运行环境检查；完成后这里会显示摘要。",
        facts=[],
        findings=[],
        mode="",
    )


def idle_summary() -> DoctorSummary:
    return DoctorSummary(
        status="idle",
        heading="尚未运行项目检查",
        message="选择 work 目录（或项目根目录）后点击「环境检查」。",
        facts=[],
        findings=[],
        mode="",
    )


def stale_summary() -> DoctorSummary:
    return DoctorSummary(
        status="stale",
        heading="项目已切换，请重新运行检查",
        message="当前摘要已清空；请针对新的 work 目录重新运行环境检查。",
        facts=[],
        findings=[],
        mode="",
    )


def cancelled_summary() -> DoctorSummary:
    return DoctorSummary(
        status="idle",
        heading="环境检查已取消",
        message="上次检查未完成；需要时请重新运行环境检查。",
        facts=[],
        findings=[],
        mode="",
    )
