"""Pure helpers for GUI-managed translator settings.

This module intentionally has no Qt dependency.  The GUI owns widgets and user
interaction; this module owns field metadata, coercion, validation, and
translator_config.json writes for managed advanced settings.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Literal


SettingKind = Literal["bool", "int", "float", "str", "text", "list", "json"]
SettingValue = Any


@dataclass(frozen=True)
class SettingField:
    key: str
    path: tuple[str, ...]
    label: str
    description: str
    kind: SettingKind
    default: SettingValue
    category: str
    recommended: SettingValue | None = None
    minimum: float | None = None
    maximum: float | None = None
    allow_empty: bool = False

    @property
    def recommended_value(self) -> SettingValue:
        return self.default if self.recommended is None else self.recommended


BASIC_RECOMMENDED_VALUES: dict[str, SettingValue] = {
    "theme": "system",
    "context_storage_location": "tool",
    "rag_enabled": True,
    "source_index_enabled": False,
    "bootstrap_on_build": True,
    "sync_model": "gemini-3.1-flash-lite",
    "batch_model": "gemini-3.1-flash-lite",
    "sync_embedding_model": "gemini-embedding-001",
    "batch_embedding_model": "gemini-embedding-001",
    "batch_thinking_level": "minimal",
}

# Primary toggles shown only on 设置 · 上下文 (P2b / #165). Must not also appear
# as independent advanced-page widgets — single write source via schema widgets.
CONTEXT_PRIMARY_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "sync_rag_enabled",
        "sync_story_memory_enabled",
        "batch_story_memory_enabled",
    }
)
CONTEXT_PRIMARY_SETTING_CATEGORY = "上下文主开关"


ADVANCED_SETTING_FIELDS: tuple[SettingField, ...] = (
    SettingField(
        "sync_chunk_size",
        ("sync", "chunk_size"),
        "同步 chunk 条数",
        "同步翻译单次请求最多包含的条目数。",
        "int",
        40,
        "翻译吞吐",
        minimum=1,
    ),
    SettingField(
        "sync_max_source_chars",
        ("sync", "max_source_chars"),
        "同步原文字符上限",
        "同步翻译单次请求允许的原文字符预算。",
        "int",
        12000,
        "翻译吞吐",
        minimum=1,
    ),
    SettingField(
        "sync_max_output_tokens",
        ("sync", "max_output_tokens"),
        "同步输出 token 上限",
        "同步请求传给模型的 max_output_tokens。",
        "int",
        24576,
        "翻译吞吐",
        minimum=1,
    ),
    SettingField(
        "batch_chunk_size",
        ("batch", "chunk_size"),
        "批量 chunk 条数",
        "Batch 构建时每个请求最多包含的条目数。",
        "int",
        60,
        "翻译吞吐",
        minimum=1,
    ),
    SettingField(
        "batch_max_source_chars",
        ("batch", "max_source_chars"),
        "批量原文字符上限",
        "Batch 构建时每个请求允许的原文字符预算。",
        "int",
        18000,
        "翻译吞吐",
        minimum=1,
    ),
    SettingField(
        "batch_context_before",
        ("batch", "context_before"),
        "前文条目数",
        "Batch 请求附带的前文条目数量。",
        "int",
        30,
        "翻译吞吐",
        minimum=1,
    ),
    SettingField(
        "batch_context_after",
        ("batch", "context_after"),
        "后文条目数",
        "Batch 请求附带的后文条目数量。",
        "int",
        10,
        "翻译吞吐",
        minimum=1,
    ),
    SettingField(
        "batch_max_output_tokens",
        ("batch", "max_output_tokens"),
        "批量输出 token 上限",
        "Batch 请求传给模型的 max_output_tokens。",
        "int",
        32768,
        "翻译吞吐",
        minimum=1,
    ),
    SettingField(
        "batch_temperature",
        ("batch", "temperature"),
        "批量 temperature",
        "Batch 请求的温度；越低越稳定。",
        "float",
        0.2,
        "翻译吞吐",
        minimum=0.0,
        maximum=2.0,
    ),
    SettingField(
        "batch_display_name_prefix",
        ("batch", "display_name_prefix"),
        "批量任务名称前缀",
        "提交 Batch 任务时使用的 display name 前缀。",
        "str",
        "renpy-translate",
        "翻译吞吐",
        allow_empty=True,
    ),
    SettingField(
        "batch_submit_max_cost",
        ("batch", "submit_max_cost"),
        "提交成本上限",
        "提交 Batch 任务前的估算成本上限（与 batch.pricing 货币一致）；留空或 0 表示不限制。",
        "float",
        0.0,
        "翻译吞吐",
        minimum=0.0,
        allow_empty=True,
    ),
    SettingField(
        "batch_non_chinese_extra_static_paths",
        ("batch", "non_chinese_validation", "extra_static_name_credit_rel_paths"),
        "非中文白名单追加路径",
        "追加到默认静态姓名/名单文件白名单的相对路径；每行一个，例如 credits.rpy。",
        "list",
        [],
        "翻译吞吐",
        allow_empty=True,
    ),
    SettingField(
        "context_storage_game_dir_name",
        ("context_storage", "game_dir_name"),
        "游戏目录上下文文件夹",
        "context storage 使用游戏目录模式时的文件夹名。",
        "str",
        "translation_context",
        "上下文路径",
    ),
    SettingField(
        "sync_rag_enabled",
        ("sync", "rag", "enabled"),
        "启用同步 RAG",
        "同步翻译时检索本地记忆库。",
        "bool",
        False,
        CONTEXT_PRIMARY_SETTING_CATEGORY,
    ),
    SettingField(
        "sync_rag_output_dimensionality",
        ("sync", "rag", "output_dimensionality"),
        "同步 RAG 向量维度",
        "同步 RAG 存储和查询使用的 embedding 维度。",
        "int",
        768,
        "同步 RAG",
        minimum=1,
    ),
    SettingField(
        "sync_rag_top_k_history",
        ("sync", "rag", "top_k_history"),
        "同步历史命中数",
        "同步翻译检索历史译文的最大命中数。",
        "int",
        4,
        "同步 RAG",
        minimum=1,
    ),
    SettingField(
        "sync_rag_top_k_terms",
        ("sync", "rag", "top_k_terms"),
        "同步术语命中数",
        "同步翻译检索术语/短语的最大命中数。",
        "int",
        8,
        "同步 RAG",
        minimum=1,
    ),
    SettingField(
        "sync_rag_min_similarity",
        ("sync", "rag", "min_similarity"),
        "同步相似度阈值",
        "同步 RAG 命中的最低相似度。",
        "float",
        0.72,
        "同步 RAG",
        minimum=0.0,
        maximum=1.0,
    ),
    SettingField(
        "sync_rag_segment_lines",
        ("sync", "rag", "segment_lines"),
        "同步索引分段行数",
        "写入同步记忆库时每段合并的行数。",
        "int",
        4,
        "同步 RAG",
        minimum=1,
    ),
    SettingField(
        "sync_rag_history_char_limit",
        ("sync", "rag", "history_char_limit"),
        "同步历史字符上限",
        "同步 RAG 单条历史命中注入 prompt 前的截断上限。",
        "int",
        220,
        "同步 RAG",
        minimum=1,
    ),
    SettingField(
        "sync_rag_store_dir",
        ("sync", "rag", "store_dir"),
        "同步 RAG 存储路径",
        "留空时使用当前 context storage 默认路径。",
        "str",
        "",
        "同步 RAG",
        allow_empty=True,
    ),
    SettingField(
        "sync_rag_update_on_success",
        ("sync", "rag", "update_on_success"),
        "同步成功后更新记忆库",
        "同步翻译成功后把结果写入本地 RAG 记忆库。",
        "bool",
        True,
        "同步 RAG",
    ),
    SettingField(
        "batch_rag_output_dimensionality",
        ("batch", "rag", "output_dimensionality"),
        "批量 RAG 向量维度",
        "Batch RAG 存储和查询使用的 embedding 维度。",
        "int",
        768,
        "批量 RAG",
        minimum=1,
    ),
    SettingField(
        "batch_rag_top_k_history",
        ("batch", "rag", "top_k_history"),
        "批量历史命中数",
        "Batch 请求检索历史译文的最大命中数。",
        "int",
        4,
        "批量 RAG",
        minimum=1,
    ),
    SettingField(
        "batch_rag_top_k_terms",
        ("batch", "rag", "top_k_terms"),
        "批量术语命中数",
        "Batch 请求检索术语/短语的最大命中数。",
        "int",
        8,
        "批量 RAG",
        minimum=1,
    ),
    SettingField(
        "batch_rag_min_similarity",
        ("batch", "rag", "min_similarity"),
        "批量相似度阈值",
        "Batch RAG 命中的最低相似度。",
        "float",
        0.72,
        "批量 RAG",
        minimum=0.0,
        maximum=1.0,
    ),
    SettingField(
        "batch_rag_segment_lines",
        ("batch", "rag", "segment_lines"),
        "批量索引分段行数",
        "预建批量记忆库时每段合并的行数。",
        "int",
        4,
        "批量 RAG",
        minimum=1,
    ),
    SettingField(
        "batch_rag_history_char_limit",
        ("batch", "rag", "history_char_limit"),
        "批量历史字符上限",
        "Batch RAG 单条历史命中注入 prompt 前的截断上限。",
        "int",
        220,
        "批量 RAG",
        minimum=1,
    ),
    SettingField(
        "batch_rag_store_dir",
        ("batch", "rag", "store_dir"),
        "批量 RAG 存储路径",
        "留空时使用当前 context storage 默认路径。",
        "str",
        "",
        "批量 RAG",
        allow_empty=True,
    ),
    SettingField(
        "batch_source_index_top_k",
        ("batch", "source_index", "top_k"),
        "原文索引命中数",
        "Batch 请求检索原文索引的最大命中数。",
        "int",
        4,
        "原文索引",
        minimum=1,
    ),
    SettingField(
        "batch_source_index_min_similarity",
        ("batch", "source_index", "min_similarity"),
        "原文索引相似度阈值",
        "原文索引命中的最低相似度。",
        "float",
        0.72,
        "原文索引",
        minimum=0.0,
        maximum=1.0,
    ),
    SettingField(
        "batch_source_index_char_limit",
        ("batch", "source_index", "char_limit"),
        "原文索引字符上限",
        "单条原文索引命中注入 prompt 前的截断上限。",
        "int",
        220,
        "原文索引",
        minimum=1,
    ),
    SettingField(
        "batch_source_index_store_dir",
        ("batch", "source_index", "store_dir"),
        "原文索引存储路径",
        "留空时使用当前 context storage 默认路径。",
        "str",
        "",
        "原文索引",
        allow_empty=True,
    ),
    SettingField(
        "sync_story_memory_enabled",
        ("sync", "story_memory", "enabled"),
        "启用同步剧情记忆",
        "同步翻译时注入本地 story_graph 上下文。",
        "bool",
        False,
        CONTEXT_PRIMARY_SETTING_CATEGORY,
    ),
    SettingField(
        "sync_story_memory_graph_file",
        ("sync", "story_memory", "graph_file"),
        "同步剧情图谱路径",
        "留空时使用当前 context storage 默认 story_graph 路径。",
        "str",
        "",
        "同步剧情记忆",
        allow_empty=True,
    ),
    SettingField(
        "sync_story_memory_max_context_chars",
        ("sync", "story_memory", "max_context_chars"),
        "同步剧情字符预算",
        "同步翻译注入剧情记忆的最大字符预算。",
        "int",
        800,
        "同步剧情记忆",
        minimum=1,
    ),
    SettingField(
        "sync_story_memory_top_k_relations",
        ("sync", "story_memory", "top_k_relations"),
        "同步关系命中数",
        "同步剧情记忆检索关系的最大命中数。",
        "int",
        4,
        "同步剧情记忆",
        minimum=1,
    ),
    SettingField(
        "sync_story_memory_top_k_terms",
        ("sync", "story_memory", "top_k_terms"),
        "同步剧情术语命中数",
        "同步剧情记忆检索术语的最大命中数。",
        "int",
        8,
        "同步剧情记忆",
        minimum=1,
    ),
    SettingField(
        "sync_story_memory_include_scene_summary",
        ("sync", "story_memory", "include_scene_summary"),
        "同步包含场景摘要",
        "同步剧情记忆命中时把场景摘要也注入 prompt。",
        "bool",
        True,
        "同步剧情记忆",
    ),
    SettingField(
        "batch_story_memory_enabled",
        ("batch", "story_memory", "enabled"),
        "启用批量剧情记忆",
        "Batch 构建时注入本地 story_graph 上下文。",
        "bool",
        False,
        CONTEXT_PRIMARY_SETTING_CATEGORY,
    ),
    SettingField(
        "batch_story_memory_graph_file",
        ("batch", "story_memory", "graph_file"),
        "批量剧情图谱路径",
        "留空时使用当前 context storage 默认 story_graph 路径。",
        "str",
        "",
        "批量剧情记忆",
        allow_empty=True,
    ),
    SettingField(
        "batch_story_memory_max_context_chars",
        ("batch", "story_memory", "max_context_chars"),
        "批量剧情字符预算",
        "Batch 请求注入剧情记忆的最大字符预算。",
        "int",
        1200,
        "批量剧情记忆",
        minimum=1,
    ),
    SettingField(
        "batch_story_memory_top_k_relations",
        ("batch", "story_memory", "top_k_relations"),
        "批量关系命中数",
        "Batch 剧情记忆检索关系的最大命中数。",
        "int",
        6,
        "批量剧情记忆",
        minimum=1,
    ),
    SettingField(
        "batch_story_memory_top_k_terms",
        ("batch", "story_memory", "top_k_terms"),
        "批量剧情术语命中数",
        "Batch 剧情记忆检索术语的最大命中数。",
        "int",
        12,
        "批量剧情记忆",
        minimum=1,
    ),
    SettingField(
        "batch_story_memory_include_scene_summary",
        ("batch", "story_memory", "include_scene_summary"),
        "批量包含场景摘要",
        "Batch 剧情记忆命中时把场景摘要也注入 prompt。",
        "bool",
        True,
        "批量剧情记忆",
    ),
)


FULL_COVERAGE_SETTING_FIELDS: tuple[SettingField, ...] = (
    SettingField(
        "game_root",
        ("game_root",),
        "游戏 work 目录",
        "当前项目的 work 目录；通常建议在工作台选择项目。",
        "str",
        "",
        "项目与资源",
    ),
    SettingField(
        "glossary_file",
        ("glossary_file",),
        "术语表路径",
        "glossary.json 路径；留空时使用默认术语表路径。",
        "str",
        "glossary.json",
        "项目与资源",
        allow_empty=True,
    ),
    SettingField(
        "tl_subdir",
        ("tl_subdir",),
        "翻译目录",
        "相对 work 目录的 Ren'Py 翻译目录；默认 schinese，可改为 japanese、korean 等。",
        "str",
        "game/tl/schinese",
        "项目与资源",
        allow_empty=True,
    ),
    SettingField(
        "include_files",
        ("include_files",),
        "仅包含文件",
        "可选过滤列表；每行一个相对路径。留空表示不过滤。",
        "list",
        [],
        "项目与资源",
        allow_empty=True,
    ),
    SettingField(
        "include_prefixes",
        ("include_prefixes",),
        "仅包含路径前缀",
        "可选过滤列表；每行一个相对路径前缀。留空表示不过滤。",
        "list",
        [],
        "项目与资源",
        allow_empty=True,
    ),
    SettingField(
        "prepare_enabled",
        ("prepare", "enabled"),
        "启用准备流程",
        "允许 build 前准备 work 目录、解包和生成翻译模板。",
        "bool",
        True,
        "准备流程",
    ),
    SettingField(
        "prepare_source_game_dir",
        ("prepare", "source_game_dir"),
        "源 game 目录",
        "准备流程复制/生成模板时使用的原始 game 目录。",
        "str",
        "../original/game",
        "准备流程",
        allow_empty=True,
    ),
    SettingField(
        "prepare_unpack_rpa",
        ("prepare", "unpack_rpa"),
        "解包 RPA",
        "准备流程中是否尝试解包 .rpa 资源。",
        "bool",
        True,
        "准备流程",
    ),
    SettingField(
        "prepare_generate_template",
        ("prepare", "generate_template"),
        "生成翻译模板",
        "准备流程中是否调用 Ren'Py 生成翻译模板。",
        "bool",
        True,
        "准备流程",
    ),
    SettingField(
        "prepare_refresh_existing_template",
        ("prepare", "refresh_existing_template"),
        "刷新已有模板",
        "已存在翻译模板时是否允许准备流程刷新。",
        "bool",
        True,
        "准备流程",
    ),
    SettingField(
        "prepare_language",
        ("prepare", "language"),
        "模板语言代码",
        "Ren'Py generate_translations 使用的语言目录名；需与 tl_subdir 末段一致。",
        "str",
        "schinese",
        "准备流程",
    ),
    SettingField(
        "prepare_renpy_sdk_dir",
        ("prepare", "renpy_sdk_dir"),
        "Ren'Py SDK 目录",
        "留空时自动发现或使用 RENPY_SDK_DIR 环境变量。",
        "str",
        "",
        "准备流程",
        allow_empty=True,
    ),
    SettingField(
        "prepare_python_exe",
        ("prepare", "python_exe"),
        "准备流程 Python",
        "自定义 Python 可执行文件路径；留空时使用默认。",
        "str",
        "",
        "准备流程",
        allow_empty=True,
    ),
    SettingField(
        "prepare_launcher_py",
        ("prepare", "launcher_py"),
        "Ren'Py launcher.py",
        "自定义 Ren'Py launcher.py 路径；留空时使用 SDK 默认路径。",
        "str",
        "",
        "准备流程",
        allow_empty=True,
    ),
    SettingField(
        "prepare_unpack_command",
        ("prepare", "unpack_command"),
        "自定义解包命令",
        "推荐 JSON 数组 argv，例如 [\"python\", \"unpack.py\", \"{archive}\"]。"
        "字符串形式会走 shell，需同时开启「允许 shell 字符串命令」。留空使用内置解包。",
        "json",
        "",
        "准备流程",
        allow_empty=True,
    ),
    SettingField(
        "prepare_template_command",
        ("prepare", "template_command"),
        "自定义模板命令",
        "推荐 JSON 数组 argv，例如 [\"python\", \"{launcher_py}\", \"{base_dir}\", \"translate\", \"{language}\"]。"
        "字符串形式会走 shell，需同时开启「允许 shell 字符串命令」。留空使用内置模板流程。",
        "json",
        "",
        "准备流程",
        allow_empty=True,
    ),
    SettingField(
        "prepare_allow_shell_commands",
        ("prepare", "allow_shell_commands"),
        "允许 shell 字符串命令（高风险）",
        "开启后，自定义准备命令才可用 shell 字符串执行。"
        "translator_config.json 视为可执行本地配置；仅在完全信任该配置时启用。默认关闭。",
        "bool",
        False,
        "准备流程",
    ),
    SettingField(
        "batch_retry_chunk_size",
        ("batch", "retry_chunk_size"),
        "补译 chunk 条数",
        "build-retry 生成补译请求时每个 chunk 的最大条目数。",
        "int",
        8,
        "翻译吞吐",
        minimum=1,
    ),
    SettingField(
        "batch_retry_max_source_chars",
        ("batch", "retry_max_source_chars"),
        "补译原文字符上限",
        "build-retry 生成补译请求时每个 chunk 的原文字符预算。",
        "int",
        4000,
        "翻译吞吐",
        minimum=1,
    ),
    SettingField(
        "batch_safety_settings",
        ("batch", "safety_settings"),
        "Batch safety settings",
        "可填 relaxed_adult 等预设，或 Google safety settings JSON 数组。留空表示默认。",
        "json",
        [],
        "翻译吞吐",
        allow_empty=True,
    ),
    SettingField(
        "batch_macro_setting_file",
        ("batch", "macro_setting_file"),
        "风格设定文件",
        "macro_setting.md 路径；通常位于当前 work 目录。",
        "str",
        "macro_setting.md",
        "术语与风格",
        allow_empty=True,
    ),
    SettingField(
        "batch_macro_setting",
        ("batch", "macro_setting"),
        "内联风格设定",
        "直接写入 prompt 的风格设定文本；留空时读取风格设定文件。",
        "text",
        "",
        "术语与风格",
        allow_empty=True,
    ),
    SettingField(
        "keyword_chunk_size",
        ("batch", "keyword_extraction", "chunk_size"),
        "关键词 chunk 条数",
        "批量关键词提取每个请求的最大条目数。",
        "int",
        40,
        "关键词提取",
        minimum=1,
    ),
    SettingField(
        "keyword_max_candidates_per_chunk",
        ("batch", "keyword_extraction", "max_candidates_per_chunk"),
        "每 chunk 候选上限",
        "关键词提取每个 chunk 允许返回的候选数量。",
        "int",
        12,
        "关键词提取",
        minimum=1,
    ),
    SettingField(
        "keyword_display_name_prefix",
        ("batch", "keyword_extraction", "display_name_prefix"),
        "关键词任务名称前缀",
        "提交关键词 Batch 任务时使用的 display name 前缀。",
        "str",
        "renpy-keywords",
        "关键词提取",
        allow_empty=True,
    ),
    SettingField(
        "revision_chunk_size",
        ("batch", "revision", "chunk_size"),
        "订正 chunk 条数",
        "批量订正每个请求的最大条目数。",
        "int",
        6,
        "订正",
        minimum=1,
    ),
    SettingField(
        "revision_display_name_prefix",
        ("batch", "revision", "display_name_prefix"),
        "订正任务名称前缀",
        "提交订正 Batch 任务时使用的 display name 前缀。",
        "str",
        "renpy-revise",
        "订正",
        allow_empty=True,
    ),
    SettingField(
        "sync_rag_query_task_type",
        ("sync", "rag", "query_task_type"),
        "同步查询 task type",
        "同步 RAG 查询 embedding 的 task type。",
        "str",
        "RETRIEVAL_QUERY",
        "同步 RAG",
    ),
    SettingField(
        "sync_rag_document_task_type",
        ("sync", "rag", "document_task_type"),
        "同步文档 task type",
        "同步 RAG 写入文档 embedding 的 task type。",
        "str",
        "RETRIEVAL_DOCUMENT",
        "同步 RAG",
    ),
    SettingField(
        "batch_rag_query_task_type",
        ("batch", "rag", "query_task_type"),
        "批量查询 task type",
        "Batch RAG 查询 embedding 的 task type。",
        "str",
        "RETRIEVAL_QUERY",
        "批量 RAG",
    ),
    SettingField(
        "batch_rag_document_task_type",
        ("batch", "rag", "document_task_type"),
        "批量文档 task type",
        "Batch RAG / 原文索引写入文档 embedding 的 task type。",
        "str",
        "RETRIEVAL_DOCUMENT",
        "批量 RAG",
    ),
)

ADVANCED_SETTING_FIELDS = FULL_COVERAGE_SETTING_FIELDS + ADVANCED_SETTING_FIELDS


ADVANCED_SETTING_FIELD_BY_KEY = {field.key: field for field in ADVANCED_SETTING_FIELDS}


def grouped_advanced_fields(
    *,
    include_context_primary: bool = True,
) -> list[tuple[str, list[SettingField]]]:
    """Group schema fields by category for settings UI.

    When *include_context_primary* is False, omit switches that live only on the
    上下文 page (P2b single-source rule).
    """
    groups: list[tuple[str, list[SettingField]]] = []
    by_category: dict[str, list[SettingField]] = {}
    for field in ADVANCED_SETTING_FIELDS:
        if (
            not include_context_primary
            and field.key in CONTEXT_PRIMARY_SETTING_KEYS
        ):
            continue
        if field.category not in by_category:
            by_category[field.category] = []
            groups.append((field.category, by_category[field.category]))
        by_category[field.category].append(field)
    return groups


def context_primary_setting_fields() -> tuple[SettingField, ...]:
    return tuple(
        field
        for field in ADVANCED_SETTING_FIELDS
        if field.key in CONTEXT_PRIMARY_SETTING_KEYS
    )


def read_advanced_settings(config: dict[str, Any]) -> dict[str, SettingValue]:
    return {field.key: read_setting(config, field) for field in ADVANCED_SETTING_FIELDS}


def recommended_advanced_settings() -> dict[str, SettingValue]:
    return {field.key: field.recommended_value for field in ADVANCED_SETTING_FIELDS}


def validate_advanced_settings(values: dict[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for field in ADVANCED_SETTING_FIELDS:
        error = validate_value(field, values.get(field.key))
        if error:
            errors[field.key] = error
    return errors


def apply_advanced_settings(
    config: dict[str, Any],
    values: dict[str, Any],
) -> dict[str, Any]:
    errors = validate_advanced_settings(values)
    if errors:
        first = next(iter(errors.values()))
        raise ValueError(first)

    for field in ADVANCED_SETTING_FIELDS:
        set_nested_value(config, field.path, normalize_for_write(field, values[field.key]))
    return config


def read_setting(config: dict[str, Any], field: SettingField) -> SettingValue:
    raw = get_nested_value(config, field.path)
    if raw is None:
        return field.default
    try:
        return normalize_for_write(field, raw)
    except ValueError:
        return field.default


def validate_value(field: SettingField, value: Any) -> str:
    try:
        normalized = normalize_for_write(field, value)
    except ValueError as exc:
        return str(exc)

    if field.kind in {"str", "text"}:
        if not field.allow_empty and not str(normalized).strip():
            return f"{field.label}不能为空。"
        return ""

    if field.kind in {"bool", "list", "json"}:
        return ""

    number = float(normalized)
    if field.minimum is not None and number < field.minimum:
        if field.kind == "int" and field.minimum == 1:
            return f"{field.label}必须是大于 0 的整数。"
        return f"{field.label}不能小于 {field.minimum:g}。"
    if field.maximum is not None and number > field.maximum:
        return f"{field.label}不能大于 {field.maximum:g}。"
    return ""


def normalize_for_write(field: SettingField, value: Any) -> SettingValue:
    if field.kind == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError(f"{field.label}必须是布尔值。")

    if field.kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"{field.label}必须是整数。")
        try:
            if isinstance(value, str) and not value.strip():
                raise ValueError
            return int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field.label}必须是整数。") from None

    if field.kind == "float":
        if isinstance(value, bool):
            raise ValueError(f"{field.label}必须是数字。")
        try:
            if isinstance(value, str) and not value.strip():
                raise ValueError
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field.label}必须是数字。") from None

    if field.kind in {"str", "text"}:
        return value.strip() if isinstance(value, str) else str(value).strip()

    if field.kind == "list":
        return normalize_list_value(field, value)

    if field.kind == "json":
        return normalize_json_value(field, value)

    raise ValueError(f"Unsupported setting type: {field.kind}")


def normalize_list_value(field: SettingField, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field.label}必须是 JSON 数组或逐行列表。") from exc
            value = parsed
        else:
            value = [part.strip() for line in text.splitlines() for part in line.split(",")]
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{field.label}必须是列表。")
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_json_value(field: SettingField, value: Any) -> Any:
    if value is None:
        return field.default if field.allow_empty else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return field.default if field.allow_empty else ""
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field.label}必须是有效 JSON。") from exc
        return text
    return value


def get_nested_value(config: dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = config
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_nested_value(config: dict[str, Any], path: tuple[str, ...], value: SettingValue) -> None:
    current: dict[str, Any] = config
    for part in path[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[path[-1]] = value

