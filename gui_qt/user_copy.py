"""Shared Chinese labels and copy helpers for GUI summaries."""
from __future__ import annotations

SAFETY_LEVEL_LABELS = {
    "safe": "可写回",
    "warn": "需处理",
    "block": "禁止写回",
}

DOCTOR_MODE_LABELS = {
    "can_generate_template": "可生成翻译模板",
    "existing_tl_only": "仅处理已有翻译文件",
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
    return f"任务清单：{path}"


def format_package_dir_fact(path: str) -> str:
    return f"翻译包：{path}"


def format_job_fact(job_name: str) -> str:
    return f"云端任务：{job_name}"


def format_job_state_fact(state: str) -> str:
    return f"任务状态：{job_state_label(state)}"


def format_safety_fact(level: str, *, prefix: str = "检查结果") -> str:
    return f"{prefix}：{safety_level_label(level)}"


def translate_doctor_warning(warning: str) -> str:
    text = warning.strip()
    for source, translated in DOCTOR_WARNING_TRANSLATIONS:
        if text == source:
            return translated
    if text.startswith("Found ") and "legacy manifest" in text:
        return "检测到旧版任务清单，将使用兼容模式继续处理。"
    if text.startswith("Custom template command cannot be rendered:"):
        return "自定义模板命令无法解析，请检查配置。"
    if text.startswith("RAG store contains legacy ID format keys."):
        return "记忆库含有旧版键格式，下次写回时会自动迁移。"
    return text


def format_bootstrap_fact(key: str, value: str) -> str:
    label = BOOTSTRAP_FIELD_LABELS.get(key, key)
    return f"{label}：{value}"