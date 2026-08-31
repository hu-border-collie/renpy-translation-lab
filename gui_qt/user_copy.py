"""Shared Chinese labels and copy helpers for GUI summaries.

用户可见文案术语规则（#299）：
- 概念优先使用中文：术语表（glossary）、分块（chunk）、词条（unit）、写回（apply）。
- 文件名、配置键、CLI 标识符保留原文：``glossary.json``、``chunk_size``、``unit``。
- 通用技术词允许保留：token、API Key、Provider、model、LiteLLM、Ren'Py。
"""
from __future__ import annotations

from typing import Any

import doctor_recommendations as doctor_rec
from sync_model_backend import MAX_SYNC_TIMEOUT_SECONDS, MIN_SYNC_TIMEOUT_SECONDS

SAFETY_LEVEL_LABELS = {
    "safe": "可写回",
    "warn": "需处理",
    "block": "禁止写回",
}

CHECK_STATUS_LABELS = {
    "ready": "可写回（无质量报警）",
    "ready_with_warnings": "可写回，有质量报警",
    "blocked": "禁止写回",
}

QUALITY_GATE_LABELS = {
    "pass": "无质量报警",
    "needs_review": "需人工复核",
    "acknowledged": "质量报警已确认",
}

QUALITY_DELIVERY_NOTICE = "可写回 ≠ 可交付：写回门禁只证明结构安全，质量报警处理完前不建议交付。"
QUALITY_REPORT_EXPORT_LABEL = "导出 HTML 报告"
QUALITY_REPORT_EXPORT_TITLE = "导出质量体检报告"
QUALITY_REPORT_EXPORT_SUCCESS = "质量体检报告已导出。"

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

APP_SHUTDOWN_COPY = {
    "active_title": "任务仍在运行",
    "active_heading": "关闭前需要停止本机正在运行的任务。",
    "active_detail": (
        "停止只影响本机进程、下载和状态轮询；已经提交到云端的任务会继续运行，"
        "不会被远程取消。"
    ),
    "confirm": "停止任务并退出",
    "cancel": "取消",
    "stopping": "正在安全停止后台任务，完成后将自动退出…",
    "stalled": "部分后台任务仍在停止；窗口会保持打开，直到线程真实结束。",
}

# Workspace setup dialog: cooperative stop of the SDK download worker.
WORKSPACE_SETUP_STOP_COPY = {
    "cancelling": "正在取消 SDK 下载…任务真正结束后可重试或跳过。",
    "waiting_close": "正在等待 SDK 任务结束，窗口将在任务完成后自动关闭。",
    "still_stopping": "SDK 任务仍在结束，请稍候再试。",
    "cancelled": "SDK 安装已取消。",
}

MANIFEST_MODE_LABELS = {
    "translation": "普通翻译",
    "revision": "订正",
    "keyword_extraction": "关键词提取",
}

VERSION_ASSET_COPY = {
    "export_snapshot": "版本资产·导出项目快照",
    "reconcile_snapshots": "版本资产·比较两个快照",
    "build_translation_records": "版本资产·冻结译文记录",
    "build_reuse_candidates": "版本资产·生成复用候选",
    "import_reuse_decisions": "版本资产·导入复用决策",
    "export_reuse_results": "版本资产·导出复用结果",
}

DURABLE_SYNC_COPY = {
    "start": "耐久同步翻译·开始",
    "resume": "耐久同步翻译·继续",
    "status": "耐久同步翻译·状态",
    "cancel": "耐久同步翻译·取消",
    "derive": "耐久同步翻译·派生新运行",
    "check": "耐久同步翻译·检查并生成差异预览",
    "apply": "耐久同步翻译·写回已检查预览",
}

TRANSLATION_PLAN_COPY = {
    "current": "翻译计划：已绑定共享 TranslationPlan",
    "legacy": (
        "翻译计划：旧版兼容模式；仍执行原有 source/check/adapter 安全校验，"
        "但没有 plan/request 级 freshness 绑定。建议重新生成任务。"
    ),
    "fingerprint": "翻译计划指纹",
    "requests": "规范化请求数",
    "context_truncated": "上下文裁剪请求数",
    "context_dropped": "上下文舍弃记录数",
}

CONTEXT_LIBRARY_COPY = {
    "empty_title": "尚未启用上下文库",
    "empty_body": (
        "请先在设置 · 上下文启用记忆库、原文索引或项目剧情分析并保存，"
        "然后回到这里开始准备。"
    ),
    "project_gate_body": "选择项目并运行环境检查后，才能预建记忆库、原文索引或开始项目分析。",
}

TASK_PROJECT_GATE_COPY = {
    "title": "先完成环境检查",
    "action": "去环境检查",
    "status_section_title": "任务状态",
    "project_hint": "请先选择项目",
    "batch_body": "选择项目并运行环境检查后，才能开始翻译。",
    "sync_translation_body": (
        "选择项目并运行环境检查后，才能开始同步翻译。"
        "默认先生成差异预览，确认后才写回。"
    ),
    "keywords_body": (
        "选择项目并运行环境检查后，才能提取关键词。"
        "任务只生成候选报告，不修改游戏脚本；报告会附上历史首次译法与保留不译的人工作提示，审核后可合并到 glossary.json。"
    ),
    "revision_body": (
        "选择项目并运行环境检查后，才能生成订正预览；确认预览后才可写回。"
    ),
}

REVISION_PROPOSAL_COPY = {
    "action": "导入润色提案",
    "tooltip": "导入结构化 JSONL 提案并生成安全预览；此步骤不会修改 .rpy。",
    "dialog_title": "选择润色提案 JSONL",
    "corpus_dialog_title": "选择配套语料 manifest（没有则取消）",
    "running": "正在校验提案并准备候选会话。",
    "result_title": "润色提案候选",
    "select_action": "筛选并生成订正预览",
    "selection_hint": "只会把明确勾选的有效候选交给订正预览；无效、过期、冲突和无需修改项不能写回。",
    "selection_dialog_title": "筛选并选择润色候选",
    "selection_valid_only": "只看有效候选",
    "selection_select_all": "全选筛选中有效",
    "selection_clear": "清空选择",
    "selection_confirm": "生成订正预览",
    "selection_running": "正在确认润色候选并生成订正预览。",
    "selection_stale": "项目或提案文件已变化，候选会话已过期；请重新导入当前项目。",
}

REVISION_CORPUS_COPY = {
    "action": "导出润色语料",
    "tooltip": (
        "只读导出原文与当前译文的稳定 identity 语料；不会修改 .rpy。"
        "需要当前项目已通过环境检查且存在可导出的译文。"
    ),
    "running": "正在导出润色语料；扫描期间不会修改游戏脚本。",
    "gate_no_project": "请先选择项目。",
    "gate_doctor": "请先完成并通过当前项目的环境检查。",
    "gate_doctor_stale": "翻译文件最近发生过变化，请重新运行环境检查后再导出润色语料。",
    "gate_no_translations": "当前没有可导出的已有译文；请先准备或确认 TL 翻译文件。",
    "gate_running": "已有任务正在运行，请等待任务结束后再导出。",
    "gate_wrong_mode": "请在订正页的批量模式中导出润色语料。",
    "result_title": "润色语料导出结果",
    "open_output_dir": "打开输出目录",
    "copy_paths": "复制路径",
    "empty_result": "没有可导出的润色译文。",
    "success_message": "语料已生成；可交给人工或 Agent 通读和起草提案。",
    "source_changed_message": "语料已生成，但扫描期间源文件发生变化；请重新导出以获得一致快照。",
    "missing_artifact_message": "CLI 返回的导出 artifact 不完整；请查看运行日志并重试。",
    "invalid_result_message": "导出没有返回可识别的机器结果；请查看运行日志并重试。",
    "stale_message": "项目已切换，刚才的导出结果已丢弃；请在当前项目重新导出。",
}

USAGE_LEDGER_COPY = {
    "empty": "模型用量：当前项目暂无实际响应记录",
    "load_error": "模型用量账本读取失败，统计暂不可用",
    "total": "模型用量",
    "recent": "最近一次运行",
    "estimated_cost": "估算成本（非 provider 账单）",
    "actual_cost": "Provider 报告成本",
}

SYNC_TIMEOUT_COPY = {
    "label": "同步单请求超时",
    "description": (
        "普通翻译、项目分析、关键词、订正、补译与 A/B 对比中，每次模型请求的等待上限；"
        "不是整次任务的总时限。允许范围 "
        f"{MIN_SYNC_TIMEOUT_SECONDS}–{MAX_SYNC_TIMEOUT_SECONDS} 秒。"
    ),
}

SYNC_CONTEXT_BEFORE_COPY = {
    "label": "同步前文条目数",
    "description": (
        "同步请求附带的局部前文条目数量；上下文限定在同文件、同翻译块内，"
        "超过预算会被截断并记录诊断。0 表示关闭局部前文。"
    ),
}

SYNC_CONTEXT_AFTER_COPY = {
    "label": "同步后文条目数",
    "description": (
        "同步请求附带的局部后文条目数量；上下文限定在同文件、同翻译块内，"
        "超过预算会被截断并记录诊断。0 表示关闭局部后文。"
    ),
}

SYNC_MACRO_SETTING_FILE_COPY = {
    "label": "同步风格设定文件",
    "description": (
        "注入同步提示词的 macro_setting.md 路径。相对路径相对当前 work；"
        "切换项目时会同步到当前 work。留空则使用当前 work 下的 macro_setting.md。"
    ),
}

MODEL_CONTRACT_COPY = {
    "partial_translation": (
        "部分模型结果未通过完整性合同，其余安全预览已生成。"
    ),
    "partial_keyword": (
        "部分模型结果缺失或无效；已保留通过合同的关键词候选，"
        "请查看请求块完整率和不完整请求。"
    ),
    "partial_revision": (
        "部分模型结果缺失或无效；已保留通过合同的订正项，请查看完整率和未解决项。"
    ),
    "completeness": "结果完整率",
    "chunk_completeness": "请求块完整率",
    "targeted_retries": "定点重试",
    "unresolved_items": "未解决结果",
    "partial_requests": "不完整请求",
}

PROMPT_CONTEXT_COPY = {
    "local_context": "局部上下文",
    "macro_setting": "风格设定",
}

PROBE_COPY = {
    "failed": (
        "试跑样本请求没有正常完成，请查看诊断日志中的 API、格式或任务记录错误；"
        "若 request row 与 chunk 不再匹配，请重建翻译包。"
    ),
}

PROJECT_ANALYSIS_COPY = {
    "start": "开始分析",
    "generate": "生成项目摘要",
    "refresh": "更新项目摘要",
    "rebuild": "重新分析",
    "review": "审查内容",
    "publish": "启用到翻译",
    "unpublish": "停止用于翻译",
    "review_title": "项目剧情分析 · 审查与启用",
    "review_heading": "先核对摘要变化与来源，再决定是否用于翻译",
    "review_confirm": "确认已审查",
    "review_publish": "审查并启用到翻译",
    "publish_tip": (
        "将待启用的项目摘要保存为翻译使用版本；启用前会核对游戏脚本是否变化。"
    ),
    "unpublish_tip": "停止在翻译中使用当前项目摘要；待审查内容和游戏脚本都会保留。",
    "rebuild_tip": "重新读取剧情节点、跳转与路线，并生成新的待审查项目摘要。",
    "publish_confirm_title": "确认启用项目摘要",
    "publish_confirm_body": (
        "将把当前待审查摘要设为翻译使用版本。只有设置中的“用于翻译”已开启，"
        "且游戏脚本自分析后没有变化时才会实际使用；不会修改游戏脚本。"
    ),
    "unpublish_confirm_title": "确认停止使用项目摘要",
    "unpublish_confirm_body": (
        "将立即停止在翻译中使用当前项目摘要；待审查内容和游戏脚本都会保留。"
    ),
}

PROJECT_ANALYSIS_ARTIFACT_LABELS = {
    "chunk": "剧情概要",
    "chunk_summary": "剧情概要",
    "scene": "剧情场景",
    "label": "场景节点",
    "label_summary": "场景节点",
    "route": "剧情路线",
    "route_summary": "剧情路线",
    "project_brief": "项目摘要",
}

PROJECT_ANALYSIS_RECORD_STATUS_LABELS = {
    "missing": "未生成",
    "draft": "待审查",
    "review_required": "待确认",
    "published": "已启用",
    "stale": "已过期",
    "failed": "失败",
}


def project_analysis_artifact_label(kind: str) -> str:
    """Return a user-facing Project Analysis artifact name."""
    return PROJECT_ANALYSIS_ARTIFACT_LABELS.get(str(kind or ""), "分析条目")


def project_analysis_record_status_label(status: str) -> str:
    """Return a user-facing Project Analysis lifecycle label."""
    return PROJECT_ANALYSIS_RECORD_STATUS_LABELS.get(str(status or ""), "未知")


SETTINGS_WORKSPACE_IMMEDIATE_SAVE = (
    "项目列表操作即时保存，不受设置保存按钮影响。"
)
SETTINGS_WORKSPACE_UNSAVED_CHANGES = (
    "其他设置有未保存的更改；可保存、重新加载放弃，或切换项目时再处理。"
)

# Shared copy for the custom OpenAI-compatible LiteLLM provider management UI.
# Keep GUI wording in one place per the repo convention (AGENTS.md).
CUSTOM_LITELLM_PROVIDER_COPY = {
    "dialog_intro": (
        "自定义 OpenAI 兼容 Provider（如 OpenCode Go、各类中转站、本地 vLLM）。"
        "请求会改写为 openai/<模型> 并逐请求透传 API Base；"
        "模型显示名保持 <id>/<模型>。id 同时用作密钥存储用户名。"
    ),
    "id_tooltip": (
        "创建后不可修改；只能包含小写字母、数字、- 和 _，"
        "且不能与 LiteLLM 已知 provider 前缀冲突。"
    ),
    "env_tooltip": (
        "仅当系统凭据管理器中未保存该 Provider 的密钥时，"
        "后端才会读取此环境变量并显式传给请求。"
    ),
    "requires_key_tooltip": (
        "关闭后适用于无需鉴权的本地 vLLM / LocalAI 网关："
        "模型列表与请求都不会要求或携带密钥。"
    ),
    "table_empty": "尚未注册自定义 Provider。",
    "table_count": "已注册 {count} 个自定义 Provider。",
    "missing_key_title": "请先保存 API Key",
    "missing_key_body": (
        "{label} 的模型列表需要 API Key。\n\n"
        "请先在下方「Provider 凭据」中粘贴并保存密钥，再加载模型列表。"
    ),
    "missing_key_env_hint": "\n也可设置环境变量 {env} 作为回退。",
    "worker_missing_key": "请先保存 {label} API Key，再刷新官方模型列表",
    "missing_connection_key": (
        "自定义 Provider「{label}」还没有可用的密钥。\n\n"
        "请先在下方「Provider 凭据」中保存 API Key"
    ),
    "missing_connection_env_hint": "，或设置环境变量 {env}。",
    "missing_connection_env_suffix": "。",
    "delete_title": "删除自定义 Provider",
    "delete_confirm": (
        "确定删除自定义 Provider「{label}」（{id}）？\n\n"
        "删除只移除注册信息，不会删除系统凭据管理器中的密钥或用户目录缓存；"
        "如需清理密钥请到「管理密钥…」中删除。"
    ),
    "delete_current_note": (
        "\n\n该 Provider 当前正在使用：删除后当前模型选择会被清除，"
        "请重新选择 Provider 与模型。"
    ),
    "load_error_status": "已忽略无效的 custom_litellm_providers 配置：{error}",
    "load_error_title": "自定义 Provider 配置无效",
    "load_error_save_blocked": (
        "translator_config.json 中的 sync.custom_litellm_providers 存在无效条目，"
        "当前仅加载了部分内容。\n\n"
        "直接保存会用当前内存列表覆盖磁盘配置，可能丢失原本有效的条目；"
        "已阻止保存。请先手工修正配置文件，或删除无效条目后重试。\n\n"
        "错误详情：{error}"
    ),
    "keyless_status": (
        "该自定义 Provider 无需 API Key（requires_key=false），"
        "可直接加载模型列表与测试连接。"
    ),
}


LITELLM_CONNECTION_TEST_COPY = {
    "progress": "正在发起最小 JSON 连接测试请求…",
    "success": "连接成功。已通过最小 JSON 响应校验。",
    "errors": {
        "authentication": "身份验证失败，请检查供应商密钥。",
        "rate_limit": "供应商限流或配额不足，请稍后重试。",
        "service_unavailable": "供应商服务暂时不可用，请稍后重试。",
        "timeout": "连接测试已达到单请求超时上限。",
        "missing_dependency": "LiteLLM 尚未正确安装。",
        "invalid_response": (
            "模型未返回预期的 JSON 结果；请检查模型兼容性或 reasoning 输出预算。"
        ),
        "unsupported_capability": "当前 Provider 不支持连接测试所需的 JSON 能力。",
        "provider_error": "请求失败，请检查模型、API Base 和网络。",
    },
    "routing_errors": {
        "MODEL_PROFILE_INVALID": (
            "连接失败 [{code}]: 模型或 Provider 配置无效。{message}"
        ),
        "MODEL_ROUTE_CAPABILITY_MISSING": (
            "连接失败 [{code}]: 当前执行方式不支持该模型。{message}"
        ),
        "MODEL_PROFILE_CREDENTIAL_REF_MISSING": (
            "连接失败 [{code}]: 缺少可用凭据引用。{message}"
        ),
        "default": "连接失败 [{code}]: {message}",
    },
    "stale_result": "连接参数已修改，本次测试结果已忽略；请用当前设置重新测试。",
}


# Shared copy for the user-level LiteLLM catalog cache fallback (#360/#361).
# Keep GUI wording in one place per the repo convention (AGENTS.md).
LITELLM_CACHE_COPY = {
    "fallback_reason": (
        "默认 LiteLLM 用户目录缓存不可写，已回退到临时目录：{directory}"
    ),
    "save_status": "LiteLLM 选择已保存到临时目录；重启后可能不会保留。",
    "save_failed_status": "LiteLLM 选择已更新，但用户目录缓存写入失败。",
    "save_failed_log": "保存 LiteLLM 用户目录缓存失败：{error}",
}


GAMES_REGISTRY_SOURCE_URL_COPY = {
    "field_label": "发布地址",
    "placeholder": "https://…（可留空）",
    "tooltip": "游戏发布来源，仅接受 http:// 或 https://；不会自动访问网络。",
    "open_action": "打开发布页",
    "invalid_title": "发布地址无效",
    "open_failed_title": "无法打开发布页",
    "open_failed_body": (
        "系统浏览器未能打开该地址；请检查默认浏览器设置，"
        "或复制发布地址后手动打开。"
    ),
    "save_tooltip": "保存当前选中项目的名称、发布地址、游玩/翻译状态及备注。",
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
    doctor_rec.ENABLE_PREPARE: "建议：在「设置 · 高级」启用 prepare 后，再点「开始翻译」生成模板",
    doctor_rec.BOOTSTRAP_SOURCE_INDEX: "建议：先到左侧「上下文库」运行「预建原文索引」",
    doctor_rec.BOOTSTRAP_SOURCE_INDEX_INCOMPLETE: "建议：继续在「上下文库」运行「预建原文索引」补全索引",
    doctor_rec.BUILD_PROJECT_ANALYSIS: "建议：到「上下文库」开始项目分析并生成待审查摘要",
    doctor_rec.REFRESH_PROJECT_ANALYSIS: "建议：到「上下文库」重新分析；只会重建受影响的摘要",
    doctor_rec.CONFIGURE_PROJECT_ANALYSIS_MODEL: "建议：在「设置 · 上下文」配置项目分析模型",
    doctor_rec.CONFIGURE_PROJECT_ANALYSIS_API: "建议：配置 Gemini API Key 后再生成项目分析摘要",
    doctor_rec.BOOTSTRAP_RAG: "建议：先到左侧「上下文库」运行「预建记忆库」，再开始批量翻译",
    doctor_rec.REBUILD_RAG_STORE: "建议：到「上下文库」重新预建记忆库（当前向量后端不兼容）",
    doctor_rec.REBUILD_SOURCE_INDEX_STORE: "建议：到「上下文库」重新预建原文索引（当前向量后端不兼容）",
    doctor_rec.BOOTSTRAP_RAG_OR_WARM_ON_BUILD: (
        "可选准备：记忆库为空；可先到「上下文库」预建记忆库，也可直接「开始翻译」并自动暖库"
    ),
    doctor_rec.ENABLE_RAG_FOR_CONSISTENCY: (
        "可选优化：补译量较大且记忆库未启用；可在「设置 · 上下文」启用并保存，"
        "再到「上下文库」预建记忆库以提高一致性"
    ),
    doctor_rec.SUBSTANTIALLY_COMPLETE: (
        "建议：项目已基本译完；剩余待译行很少（可能含专名/标点），可忽略或按需补译，不必预建记忆库"
    ),
    doctor_rec.ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT: (
        "可选优化：全新初译项目可在「设置 · 上下文」启用原文索引并保存，"
        "再到「上下文库」预建，以获得更多剧情上下文"
    ),
    doctor_rec.START_INCREMENTAL_BATCH: (
        "建议：补译环境已就绪；在左侧「批量翻译」点击「开始翻译」打包并提交"
    ),
    doctor_rec.NO_PENDING_LINES: (
        "建议：当前没有待译条目；如需创建新批次，请先刷新翻译模板"
    ),
    doctor_rec.START_PENDING_BATCH: (
        "建议：在左侧「批量翻译」点击「开始翻译」打包并提交云端任务"
    ),
}

DOCTOR_RECOMMENDATION_UNKNOWN_FACT = "建议：收到未识别的诊断建议，请查看诊断日志了解详情。"
DOCTOR_RECOMMENDATION_UNKNOWN_SUMMARY = "收到未识别的诊断建议，请查看诊断日志。"

# Shared status copy for no-pending (legacy rec path and workflow_state path).
_NO_PENDING_STATUS_MESSAGE = "当前没有待译条目；如需创建新批次，请先刷新翻译模板。"

DOCTOR_RECOMMENDATION_PRIMARY_MESSAGES: dict[str, str] = {
    doctor_rec.SUBSTANTIALLY_COMPLETE: "项目已基本译完；剩余待译行很少，可忽略或按需补译。",
    doctor_rec.ENABLE_RAG_FOR_CONSISTENCY: (
        "可选优化：补译量较大，可在「设置 · 上下文」启用记忆库，再到「上下文库」预建。"
    ),
    doctor_rec.BOOTSTRAP_RAG: "记忆库尚未建立，请先到左侧「上下文库」预建记忆库再开始翻译。",
    doctor_rec.REBUILD_RAG_STORE: "记忆库向量后端不兼容，请到「上下文库」重新预建记忆库。",
    doctor_rec.REBUILD_SOURCE_INDEX_STORE: "原文索引向量后端不兼容，请到「上下文库」重新预建原文索引。",
    doctor_rec.BOOTSTRAP_RAG_OR_WARM_ON_BUILD: (
        "可选准备：记忆库尚未建立；可直接开始翻译自动暖库，也可先到「上下文库」手动预建。"
    ),
    doctor_rec.BOOTSTRAP_SOURCE_INDEX: "原文索引尚未就绪，请先到左侧「上下文库」预建原文索引。",
    doctor_rec.BOOTSTRAP_SOURCE_INDEX_INCOMPLETE: "原文索引尚未就绪，请先到左侧「上下文库」继续预建。",
    doctor_rec.BUILD_PROJECT_ANALYSIS: "项目分析已启用但尚未生成；可到「上下文库」开始分析。",
    doctor_rec.REFRESH_PROJECT_ANALYSIS: "项目分析已过期；请到「上下文库」增量更新。",
    doctor_rec.CONFIGURE_PROJECT_ANALYSIS_MODEL: "项目分析缺少生成模型；请先在设置中配置。",
    doctor_rec.CONFIGURE_PROJECT_ANALYSIS_API: "项目分析生成缺少 API Key；请先完成配置。",
    doctor_rec.BOOTSTRAP_WORK: "请先准备工作目录，再开始翻译流程。",
    doctor_rec.ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT: (
        "可选优化：全新项目可在「设置 · 上下文」启用原文索引，再到「上下文库」预建。"
    ),
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


def check_status_label(status: str) -> str:
    text = str(status or "").strip().lower()
    return CHECK_STATUS_LABELS.get(text, status or "未知")


def quality_gate_label(decision: str) -> str:
    text = str(decision or "").strip().lower()
    return QUALITY_GATE_LABELS.get(text, decision or "未知")


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


def format_quality_gate_fact(gate: Any, *, prefix: str = "质量检查") -> str:
    if not isinstance(gate, dict):
        return f"{prefix}：未知"
    warning_count = int(gate.get("warning_count") or 0)
    blocker_count = int(gate.get("blocker_count") or 0)
    acknowledged_count = int(gate.get("acknowledged_count") or 0)
    decision = quality_gate_label(str(gate.get("decision") or ""))
    if warning_count > 0:
        unacknowledged_count = max(0, warning_count - acknowledged_count)
        return (
            f"{prefix}：{decision}（报警 {warning_count}，"
            f"未确认 {unacknowledged_count}，阻断 {blocker_count}）"
        )
    return f"{prefix}：{decision}（报警 0，阻断 {blocker_count}）"


def _format_usage_cost_values(metric: Any) -> str:
    if not isinstance(metric, dict):
        return ""
    values = metric.get("values")
    if not isinstance(values, dict) or not values:
        return ""
    return "、".join(
        f"{float(value):.6f} {currency}"
        for currency, value in sorted(values.items())
    )


def format_usage_ledger_facts(report: Any) -> list[str]:
    """Render the shared usage report for the diagnostics facts list."""
    if not isinstance(report, dict):
        return []
    totals = report.get("totals")
    if not isinstance(totals, dict):
        return []
    records = int(totals.get("records") or 0)
    if records <= 0:
        return [USAGE_LEDGER_COPY["empty"]]

    calls = int(totals.get("calls") or 0)
    total_tokens = totals.get("total_tokens")
    unknown_tokens = int(totals.get("total_tokens_unknown_records") or 0)
    if total_tokens is None:
        token_text = "token 数未知"
    else:
        token_text = f"{int(total_tokens):,} token"
        if unknown_tokens:
            token_text += f"；另有 {unknown_tokens} 条记录未知"
    facts = [
        f"{USAGE_LEDGER_COPY['total']}：累计 {calls} 次调用；{token_text}",
    ]

    def token_metric(field: str) -> str:
        value = totals.get(field)
        unknown = int(totals.get(f"{field}_unknown_records") or 0)
        text = "unknown" if value is None else f"{int(value):,}"
        if unknown:
            text += f"（{unknown} 条未知）"
        return text

    facts.append(
        "输出 Token："
        f"completion {token_metric('completion_tokens')} / "
        f"reasoning {token_metric('reasoning_tokens')} / "
        f"正文 {token_metric('text_output_tokens')}"
    )
    reasoning_share = totals.get("reasoning_share")
    if reasoning_share is not None:
        facts.append(f"Reasoning 占已知输出：{float(reasoning_share):.1%}")
    diagnostics = totals.get("output_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    reasoning_warnings = int(
        diagnostics.get("reasoning_budget_pressure_records") or 0
    )
    truncated = int(diagnostics.get("truncated_records") or 0)
    if reasoning_warnings:
        facts.append(f"Reasoning 预算告警：{reasoning_warnings} 条响应")
    if truncated:
        facts.append(f"输出截断：{truncated} 条响应")

    recent = report.get("recent_run")
    if isinstance(recent, dict):
        recent_totals = recent.get("totals")
        recent_totals = recent_totals if isinstance(recent_totals, dict) else {}
        recent_tokens = recent_totals.get("total_tokens")
        recent_token_text = (
            "token 数未知" if recent_tokens is None else f"{int(recent_tokens):,} token"
        )
        dimensions = " / ".join(
            ", ".join(str(value) for value in recent.get(key) or [])
            for key in ("task_modes", "stages", "providers", "models")
        )
        facts.append(
            f"{USAGE_LEDGER_COPY['recent']}：{dimensions or '未知'}；{recent_token_text}"
        )

    estimated = _format_usage_cost_values(totals.get("estimated_cost"))
    if estimated:
        facts.append(f"{USAGE_LEDGER_COPY['estimated_cost']}：{estimated}")
    actual = _format_usage_cost_values(totals.get("actual_cost"))
    if actual:
        facts.append(f"{USAGE_LEDGER_COPY['actual_cost']}：{actual}")
    return facts


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
    if text.startswith("Model routing preflight ["):
        rest = text[len("Model routing preflight ["):]
        code, _sep, rest = rest.partition("]")
        rest = rest.strip()
        if rest.startswith("(") and "): " in rest:
            location, _sep, message = rest[1:].partition("): ")
            strategy, _slash, stage = location.partition("/")
            return (
                f"模型路由启动前检查失败 [{code.strip()}]"
                f"（{strategy}/{stage}）：{message}"
            )
        return "模型路由启动前检查失败；请检查 Provider、模型、执行方式和凭据引用。"
    if text.startswith("glossary_file does not match current project;"):
        return (
            "术语表路径仍指向其他位置，与当前 work 不一致。"
            "请用「切换项目」同步到当前项目，或在设置中改为当前 work 下的 glossary.json。"
        )
    if text.startswith("glossary.json not found for current project"):
        return "当前项目缺少 glossary.json，批量翻译将使用默认保留词。"
    if text.startswith("macro_setting_file does not match current project;"):
        return (
            "风格设定路径仍指向其他位置，与当前 work 不一致。"
            "请用「切换项目」同步到当前项目，或在设置中改为当前 work 下的 macro_setting.md。"
        )
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
