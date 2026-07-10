"""Shared Chinese labels and copy helpers for GUI summaries."""
from __future__ import annotations

from typing import Any

import doctor_recommendations as doctor_rec

SAFETY_LEVEL_LABELS = {
    "safe": "可写回",
    "warn": "需处理",
    "block": "禁止写回",
}

DOCTOR_MODE_LABELS = {
    "can_generate_template": "可生成翻译模板",
    "existing_tl_only": "已有翻译模板",
    "blocked_missing_template": "缺少模板且无法生成",
}

JOB_STATE_LABELS = {
    "JOB_STATE_SUCCEEDED": "已完成",
    "JOB_STATE_FAILED": "失败",
    "JOB_STATE_CANCELLED": "已取消",
    "JOB_STATE_EXPIRED": "已过期",
    "JOB_STATE_PENDING": "排队中",
    "JOB_STATE_RUNNING": "处理中",
}

MANIFEST_MODE_LABELS = {
    "translation": "普通翻译",
    "revision": "订正",
    "keyword_extraction": "关键词提取",
}

BOOTSTRAP_FIELD_LABELS = {
    "store_dir": "存储目录",
    "scan_scope": "扫描范围",
    "files_scanned": "扫描文件数",
    "scanned": "扫描条目",
    "embedded": "生成向量数",
    "upserted": "写入记录数",
    "reused_embeddings": "复用向量数",
    "stale_count": "过期记录数",
    "pruned": "清理记录数",
    "history_records_before": "更新前记录数",
    "history_records_after": "更新后记录数",
    "external_seed_records": "外部种子记录数",
}

DOCTOR_RECOMMENDATION_CODE_TRANSLATIONS: dict[str, str] = {
    doctor_rec.SWITCH_TO_WORK: "建议：将项目路径切换到",
    doctor_rec.BOOTSTRAP_WORK: "建议：点击「准备工作目录」",
    doctor_rec.GENERATE_TEMPLATE: "建议：点击「生成翻译模板」",
    doctor_rec.INSTALL_SDK_GENERATE_TEMPLATE: "建议：配置 Ren'Py SDK 后点击「开始翻译」",
    doctor_rec.ENABLE_PREPARE: "建议：在配置中启用 prepare 后点击「开始翻译」",
    doctor_rec.BOOTSTRAP_SOURCE_INDEX: "建议：先在「分析与准备」运行「预建原文索引」",
    doctor_rec.BOOTSTRAP_SOURCE_INDEX_INCOMPLETE: "建议：继续运行「预建原文索引」补全索引库",
    doctor_rec.BOOTSTRAP_RAG: "建议：先在「分析与准备」运行「预建记忆库」，再开始批量翻译",
    doctor_rec.BOOTSTRAP_RAG_OR_WARM_ON_BUILD: (
        "可选准备：记忆库为空；可先「预建记忆库」，也可直接「开始翻译」并自动暖库"
    ),
    doctor_rec.ENABLE_RAG_FOR_CONSISTENCY: (
        "可选优化：补译量较大且记忆库未启用；可在配置页启用并「预建记忆库」以提高一致性"
    ),
    doctor_rec.SUBSTANTIALLY_COMPLETE: (
        "建议：项目已基本译完；剩余待译行很少（可能含专名/标点），可忽略或按需补译，不必预建记忆库"
    ),
    doctor_rec.ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT: (
        "可选优化：全新初译项目可在配置页启用并「预建原文索引」，以获得更多剧情上下文"
    ),
    doctor_rec.START_INCREMENTAL_BATCH: (
        "建议：补译环境已就绪；切换到「翻译 · 批量翻译」，点击「开始翻译」打包并提交"
    ),
    doctor_rec.NO_PENDING_LINES: (
        "建议：当前没有待译条目；如需创建新批次，请先刷新翻译模板"
    ),
    doctor_rec.START_PENDING_BATCH: (
        "建议：切换到「翻译 · 批量翻译」，点击「开始翻译」打包并提交云端任务"
    ),
}

DOCTOR_RECOMMENDATION_UNKNOWN_FACT = "建议：收到未识别的诊断建议，请查看诊断日志了解详情。"
DOCTOR_RECOMMENDATION_UNKNOWN_SUMMARY = "收到未识别的诊断建议，请查看诊断日志。"

# Shared status copy for no-pending (legacy rec path and workflow_state path).
_NO_PENDING_STATUS_MESSAGE = "当前没有待译条目；如需创建新批次，请先刷新翻译模板。"

DOCTOR_RECOMMENDATION_PRIMARY_MESSAGES: dict[str, str] = {
    doctor_rec.SUBSTANTIALLY_COMPLETE: "项目已基本译完；剩余待译行很少，可忽略或按需补译。",
    doctor_rec.ENABLE_RAG_FOR_CONSISTENCY: "可选优化：补译量较大，可启用并预建记忆库以提高一致性。",
    doctor_rec.BOOTSTRAP_RAG: "记忆库尚未建立，建议先预建记忆库再开始翻译。",
    doctor_rec.BOOTSTRAP_RAG_OR_WARM_ON_BUILD: (
        "可选准备：记忆库尚未建立；可直接开始翻译自动暖库，也可先手动预建。"
    ),
    doctor_rec.BOOTSTRAP_SOURCE_INDEX: "原文索引尚未就绪，建议先完成预建再开始翻译。",
    doctor_rec.BOOTSTRAP_SOURCE_INDEX_INCOMPLETE: "原文索引尚未就绪，建议先完成预建再开始翻译。",
    doctor_rec.BOOTSTRAP_WORK: "请先准备工作目录，再开始翻译流程。",
    doctor_rec.ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT: "可选优化：全新项目可预建原文索引以获得更多剧情上下文。",
    doctor_rec.START_INCREMENTAL_BATCH: "补译环境已就绪，可以开始批量翻译。",
    doctor_rec.NO_PENDING_LINES: _NO_PENDING_STATUS_MESSAGE,
    doctor_rec.START_PENDING_BATCH: "翻译环境已就绪，可以开始批量翻译。",
    doctor_rec.UNKNOWN: DOCTOR_RECOMMENDATION_UNKNOWN_SUMMARY,
}

# Keep workflow-state copy in lockstep with primary recommendation messages.
_SHARED_WORKFLOW_STATE_CODES = (
    doctor_rec.SUBSTANTIALLY_COMPLETE,
    doctor_rec.START_INCREMENTAL_BATCH,
    doctor_rec.NO_PENDING_LINES,
    doctor_rec.START_PENDING_BATCH,
)
DOCTOR_WORKFLOW_STATE_MESSAGES: dict[str, str] = {
    code: DOCTOR_RECOMMENDATION_PRIMARY_MESSAGES[code]
    for code in _SHARED_WORKFLOW_STATE_CODES
}

# Legacy recommendation codes that mean "ready / no action required" (do not elevate status).
READY_DOCTOR_RECOMMENDATION_CODES = frozenset(
    {
        doctor_rec.START_INCREMENTAL_BATCH,
        doctor_rec.START_PENDING_BATCH,
        doctor_rec.SUBSTANTIALLY_COMPLETE,
        doctor_rec.NO_PENDING_LINES,
    }
)

OPTIONAL_DOCTOR_RECOMMENDATION_CODES = doctor_rec.OPTIONAL_RECOMMENDATION_CODES

DOCTOR_WARNING_TRANSLATIONS: tuple[tuple[str, str], ...] = (
    (
        "old/new line counts differ; string translation blocks may be malformed.",
        "界面字符串块的原文/译文行数不一致，格式可能异常。",
    ),
    (
        "Dialogue translation blocks do not include source comments; revision/RAG source pairing may be limited.",
        "部分对话块缺少原文注释，订正与记忆库配对可能受限。",
    ),
    (
        "No TL files and no Ren'Py SDK/game launcher found; template generation is required.",
        "没有翻译文件，也未找到 Ren'Py SDK；需要先生成翻译模板。",
    ),
    (
        "Ren'Py SDK/game launcher not found; existing TL files can still be processed.",
        "未找到 Ren'Py SDK，但仍可处理已有翻译文件。",
    ),
    (
        "No TL files and custom template command is unavailable; template generation is required.",
        "没有翻译文件，且自定义模板命令不可用；需要先生成翻译模板。",
    ),
    (
        "Custom template command is unavailable; existing TL files can still be processed.",
        "自定义模板命令不可用，但仍可处理已有翻译文件。",
    ),
)


def safety_level_label(level: str) -> str:
    text = str(level or "").strip().lower()
    return SAFETY_LEVEL_LABELS.get(text, level or "未知")


def doctor_mode_label(mode: str) -> str:
    text = str(mode or "").strip()
    return DOCTOR_MODE_LABELS.get(text, text or "未知")


def job_state_label(state: str) -> str:
    text = str(state or "").strip()
    return JOB_STATE_LABELS.get(text, text or "未知")


def manifest_mode_label(mode: str) -> str:
    text = str(mode or "").strip()
    return MANIFEST_MODE_LABELS.get(text, text or "未知")


def format_manifest_path_fact(path: str) -> str:
    return f"任务记录：{path}"


def format_package_dir_fact(path: str) -> str:
    return f"翻译包：{path}"


def format_job_fact(job_name: str) -> str:
    return f"云端任务：{job_name}"


def format_job_state_fact(state: str) -> str:
    return f"任务状态：{job_state_label(state)}"


def format_safety_fact(level: str, *, prefix: str = "检查结果") -> str:
    return f"{prefix}：{safety_level_label(level)}"


def format_notice_fact(text: str) -> str:
    """Render advisory lines in the same `标签：值` style as other facts."""
    normalized = text.strip()
    if normalized.startswith("注意："):
        return normalized
    return f"注意：{normalized}"


def format_doctor_warning_fact(warning: str) -> str:
    """Render doctor warnings in the same `标签：值` style as other facts."""
    return format_notice_fact(translate_doctor_warning(warning))


INFORMATIONAL_DOCTOR_FINDING_MARKERS: tuple[str, ...] = (
    "记忆库含有旧版键格式",
    "检测到旧版任务记录",
)


def findings_require_attention(findings: list[str]) -> bool:
    """Return True only when warnings should elevate the doctor status to warning."""
    for finding in findings:
        text = finding.strip()
        if not text:
            continue
        if any(marker in text for marker in INFORMATIONAL_DOCTOR_FINDING_MARKERS):
            continue
        return True
    return False


def recommendation_requires_attention(recommendation_codes: list[str]) -> bool:
    """Return True when the primary recommendation is a prep step, not ready-to-translate."""
    if not recommendation_codes:
        return False
    code = recommendation_codes[0]
    return (
        code not in READY_DOCTOR_RECOMMENDATION_CODES
        and code not in OPTIONAL_DOCTOR_RECOMMENDATION_CODES
    )


def workflow_state_message(workflow_state: str) -> str:
    return DOCTOR_WORKFLOW_STATE_MESSAGES.get(str(workflow_state or "").strip(), "")


def primary_recommendation_message(recommendation_codes: list[str]) -> str:
    """Map the first recommendation code to a short summary message."""
    if not recommendation_codes:
        return ""
    return DOCTOR_RECOMMENDATION_PRIMARY_MESSAGES.get(recommendation_codes[0], "")


def format_doctor_recommendation_fact(recommendation: Any) -> str:
    """Render doctor recommendations in the same `标签：值` style as other facts."""
    rec = doctor_rec.normalize_doctor_recommendation(recommendation)
    code = str(rec.get("code") or "")
    params = rec.get("params") if isinstance(rec.get("params"), dict) else {}
    if code == doctor_rec.UNKNOWN:
        return DOCTOR_RECOMMENDATION_UNKNOWN_FACT
    rendered = DOCTOR_RECOMMENDATION_CODE_TRANSLATIONS.get(code)
    if rendered is not None:
        if code == doctor_rec.SWITCH_TO_WORK:
            work_dir = str(params.get("work_dir") or "").strip()
            return f"{rendered}{work_dir}" if work_dir else rendered
        return rendered
    detail = doctor_rec.doctor_recommendation_detail(rec)
    if detail:
        return DOCTOR_RECOMMENDATION_UNKNOWN_FACT
    return DOCTOR_RECOMMENDATION_UNKNOWN_FACT


def translate_doctor_warning(warning: str) -> str:
    text = warning.strip()
    for source, translated in DOCTOR_WARNING_TRANSLATIONS:
        if text == source:
            return translated
    if text.startswith("Found ") and "legacy manifest" in text:
        return "检测到旧版任务记录，将使用兼容模式继续处理。"
    if text.startswith("Custom template command cannot be rendered:"):
        return "自定义模板命令无法解析，请检查配置。"
    if text.startswith("RAG store contains legacy ID format keys."):
        return "记忆库含有旧版键格式，下次写回时会自动迁移。"
    if text.startswith("glossary_file does not match current project;"):
        return "术语表路径仍指向其他项目，切换项目后应自动同步到当前 work 目录。"
    if text.startswith("glossary.json not found for current project"):
        return "当前项目缺少 glossary.json，批量翻译将使用默认保留词。"
    if text.startswith("macro_setting_file does not match current project;"):
        return "风格设定路径仍指向其他项目，切换项目后应自动同步到当前 work 目录。"
    if text.startswith("macro_setting.md not found for current project"):
        return "当前项目缺少 macro_setting.md，批量翻译将缺少项目口吻与风格指引。"
    if text.startswith("Translation conflict for "):
        return (
            "术语表与剧情记忆库对同一词条给出了不同译法，可能导致提示上下文互相冲突；"
            "请人工确认后统一 glossary.json 与 story_graph.json。"
        )
    return text


def format_bootstrap_fact(key: str, value: str) -> str:
    label = BOOTSTRAP_FIELD_LABELS.get(key, key)
    return f"{label}：{value}"