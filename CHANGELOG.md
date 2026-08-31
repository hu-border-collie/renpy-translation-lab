# 变更日志

本项目从 `v1.0.0` 开始记录面向使用者的发行变化。

## [Unreleased]

`main` 在 `v1.0.0` 之后的开发变化汇总如下；这些内容尚未对应新的稳定发行版，源码中的项目版本号仍会保持 `1.0.0`，直到下一次正式发版统一更新。

### 新增

- 普通 Sync 可按项目开关消费 Source Index（独立 `RELATED PROJECT CONTEXT` 分区）与 fresh published Project Analysis brief；draft/stale/missing 只报告原因、不注入。Embedding 生产路径统一使用 Gemini / OpenAI-compatible adapters，store 记录 identity，不兼容时拒绝混用并建议重建。
- 为核心 Batch 命令增加版本化 JSON 结果 envelope、严格语义退出码、非交互显式 target、能力发现与命令 schema。
- `sync-revisions` 与 `sync-keywords` 接入同一版本化 JSON envelope：预览/写回分别复用 `preview-revisions` / `apply-revisions` 形状，关键词导出复用 `export-keywords` 的四份报告；sync 包不更新 latest，`manifest_path` 指向本次运行。
- 增加 Project Analysis 的生成、审查、发布与按场景注入生命周期，以及 report-only 最终审校 campaign。
- 增加按项目归集 Batch、同步、订正、关键词与分析调用的实际模型用量账本。
- 增加 Ren'Py Engine Adapter 的稳定扫描/occurrence/coverage 产物和写回计划校验。
- 增加提交恢复、成本估算、翻译 A/B 对比、同步预览写回与订正预览写回等完整工作流。
- `check` 新增确定性机械质量检查与 `quality_findings.jsonl`：标签插字、未闭合括号、中英文粘连、可疑英文残留、CJK/拉丁间距、半角标点、glossary 未满足、说话人标签与短感叹词漏译、配置错乱词黑名单等。
- 新增 `quality-report` 和 GUI「导出 HTML 报告」：把当前机械质量报警、最终审校 finding 与 warning 确认状态汇总为可离线搜索、筛选、打印的单文件质量体检报告；导出只读，不修改译文、manifest 或写回门禁。
- sync 预览、revision 预览/写回与 final review 统一消费同一套 quality finding schema：共享 normalize / validate / load / filter / digest 工具，各路径生成同一 `quality_findings.jsonl` + `quality_gate` 摘要；final review 仅做 LLM 语义字段到公共模型的适配，不冒充机械规则。
- 同步初译增加局部前后文（`sync.context_before` / `sync.context_after`，默认 30/10，限定同文件与 translate block 边界）、`sync.macro_setting_file` 风格设定注入，以及不依赖 RAG 开关、不受 `top_k_terms` 截断的术语命中（`normalize_map` / 保留词 / 不可翻译词，全部实际命中进入提示词）；上下文构造事实与 macro 指纹写入 manifest 并纳入预览指纹。
- LiteLLM 用户目录缓存不可写（只读 home、沙箱写入受限等）时自动回退到系统临时目录，并在 GUI 日志/状态栏提示缓存不会跨重启保留。
- 新增执行策略无关的翻译计划纯核心 `translation_plan.py`（#346 P1）：D4 共享 chunking 与稳定 chunk/request ID、D1 block 边界局部上下文、D2 词法术语命中、上下文分层组装（required/local/project/retrieval/analysis 五个内容层，排序、去重与逐层预算/丢弃诊断作为组装级记账）、D3 canonical prompt（Batch system/user 形态合并 Sync 独有规则）、plan/prompt/request 三级内容派生指纹与敏感键脱敏；`tests/fixtures/translation_plan_minimal` 冻结 sync 与 gemini_batch 的等价语义合同（同 chunk 的 `prompt_fingerprint` 一致）。改为委托共享实现的既有函数：batch 的 `iter_translation_chunk_ranges` / `hash_key` / `task_text_char_count`，sync 的 `build_sync_local_context` / `retrieve_sync_glossary_hits`；其余生产路径（sync 内联 40/12000 切块、batch 原始切片上下文、batch 词法术语 RAG 门控）留待 #346 P2/P3 切换。
- 同步初译 prompt 中词法术语（`normalize_map` 命中）的渲染顺序改为按键排序的稳定顺序（#346 确定性要求）；配置中同批多个命中且插入序非字典序的项目，glossary 行顺序会与旧版不同，词法注入块内容不变（始终全量）。RAG 开启时 `LOCKED TERMS` 参考列表按 `top_k_terms` 取排序后命中前缀，命中数超过 top_k 时保留的术语集合可能与旧版不同（排序后的字典序首个优先保留）。chunking、hash key 与局部上下文的委托保持行为不变。

### 修复

- Batch `load_batch_settings()` 现在会把 `embedding_backend` / provider / endpoint / timeout / API key env 写回模块全局变量；配置 `openai_compatible` 后不再仍走 Gemini adapter。
- Embedding task type 在加载时按 provider-neutral 合同校验并持久化为 `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`；空 embedding 输入改为立即报错，避免 zip 错位静默丢记录。
- 明确选择非 Gemini embedding 后端但缺 model/provider 时启动失败；Gemini 解析失败仍可回退，但原因写入 doctor `embedding_load_error`。
- Batch `apply` / 订正写回 / `bootstrap_on_build` 不再对缺 identity 或不兼容的 RAG store 自动重建；只有显式 `bootstrap-rag` 才会清空旧向量。
- Sync/Batch `ContextPolicy` 把 Source Index 字符预算放到独立的 `source_index_char_limit`，不再抬高 `history_char_limit`。

### 变更

- 普通 Sync 初译改为消费共享 TranslationPlan（#346 P3）：默认 chunk 对齐为 60/18000，Sync/Gemini Batch 共用 canonical system/user prompt、响应 schema、ContextAssembler 与稳定 request/prompt/plan fingerprint；Gemini、LiteLLM 和自定义 OpenAI-compatible Provider 均显式传递 system instruction。缺项/无效响应只生成带 `--T` / `--L` / `--R` lineage 的确定性派生请求，不修改原 plan；sync preview manifest 绑定 plan、request IDs、逐文件 source digest 与既有 adapter writeback/source snapshot 门禁。旧显式 Sync 分块配置与无 plan 的旧 preview manifest 保持兼容。
- Gemini Batch 初译构建改为消费 TranslationPlan（#346 P2）：`build_chunks` 通过 `translation_plan.build_translation_plan` 生成 plan 与 requests，`build_batch_request` 只负责把 `TranslationRequest` 包成 Batch envelope；manifest v2 增加 `translation_plan` 块，chunk 与 `requests.jsonl` 写入 `request_id` / `prompt_fingerprint` / `request_fingerprint` 可审计字段。Batch 初译 prompt 切换为 canonical system/user 形态，词法 glossary（`normalize_map` / `preserve_terms` / `non_translatable_exact`）解除 `RAG_ENABLED` 门控且不再按 `RAG_TOP_K_TERMS` 截断，未开启 RAG 时也会注入命中的本地术语。

- 模型任务在 prepare、创建 package 或发送请求前按活动阶段校验 ModelProfile、ExecutionStrategy、能力与凭据引用；不支持的组合返回稳定机器错误。`doctor` 增加只读模型路由诊断，`model_routing.status` 为 `ok` / `attention`，不改变 doctor 退出码。`submit --model` 只覆盖当前 Batch 阶段并保留冻结快照。GUI LiteLLM 连接测试改用与生产请求相同的 profile resolver/backend factory，路由预检失败显示稳定错误码而不是 JSON 能力失败。
- 同步请求不再用 `sync.model` 覆盖调用方传入的阶段模型。显式 `TaskRoute` / 阶段配置优先，`sync.model` 只作为未单独配置阶段的 primary 回退；一次 run 开始时冻结 `ModelRoutingPlan`，中途改配置或重试不会换 profile。sync 与 translation / keyword / revision / final_review 四类 batch manifest 写入 `model_routing` 快照（仅凭据引用，不含凭据值）。没有 `model_routing` 的旧 manifest 在 probe / resume / execute 时继续使用当时记录的 `model` / `batch_model` / `provider`，不会改用当前运行时模型。
- GUI 异步任务完成时会比对项目路径、配置 digest 和 LiteLLM 连接参数身份；过期结果只做清理，不再覆盖当前界面。
- GUI 改为统一侧边导航与任务页自有状态，项目列表、上下文库、关键词、订正、同步翻译和批量翻译各自展示当前任务结果。
- 同步翻译默认只生成 diff 与 manifest 预览，必须显式确认后才写回；写回时重新校验项目、源快照和预览制品。
- 同步 manifest 增加 `prompt_context` 诊断（局部上下文设置、macro 身份、批次截断统计）；macro 文件的新增、删除或内容变化都会使旧预览无法写回。
- `doctor` 保持只读；初始化工作副本与生成 TL 模板分别使用 `bootstrap-work`、`generate-template`。
- Settings 增加项目列表、扩展、LiteLLM 供应商/凭据与模型目录等分区，并改进未保存设置和项目切换保护。
- CI 增加依赖锁、质量/类型/依赖审计、可选依赖安装、真实 Ren'Py SDK 和 provider 契约检查；PR-Agent 自动审查使用受限权限与固定 action commit。

### 安全与质量边界

- Batch `apply` 改为要求与当前 manifest/results 匹配的最近一次 `writeback_gate.decision=allow`，并在写回前重读源快照；`--force` 不绕过 stale check、源快照或结构阻断。
- `check` 结果拆分为 `writeback_gate`（结构写回安全）与 `quality_gate`（质量报警）；质量报警默认不阻止写回，但不会因 `apply` 成功自动清除，项目配置可把规则提升为 blocker，且配置变化会使旧 check stale。
- revision 预览/写回同样生成质量 findings；warning 不阻止订正写回，配置为 blocker 的机械规则进入 revision 自己的写回门禁。规则或质量策略版本变化会使 sync / revision 旧预览 stale。
- 订正与最终审校候选必须经过 `preview-revisions -> apply-revisions` 的独立 provenance、项目身份和源快照校验。
- `writeback_gate.decision=allow` 仅代表结构性可写回；`quality_gate` 负责机械质量报警，正式交付前仍需人工/LLM 语义审校。

## [1.0.0] - 2026-07-16

首个稳定源码发行版。

### 主要能力

- 提供 Gemini Batch 主工作流：`doctor -> build -> submit -> status -> download -> check -> apply`。
- 使用 manifest / identity v2，在写回前执行 `safe / warn / block` 分级与快照校验。
- 提供可选 PySide6 图形工作台，覆盖项目准备、批量翻译、同步翻译、关键词、订正、上下文库、设置与诊断日志。
- 提供本地 RAG、原文索引、可选 Story Memory、关键词提取与订正流程。
- 支持 Gemini 同步调用以及显式选择的 LiteLLM 同步后端。

### 验证范围

- 核心 Batch 链路已在十万词量级的私有 Ren'Py 项目副本上完整跑通。
- GUI 批量翻译主路径已在数千待译行的私有项目副本上完成烟测。
- LiteLLM + DeepSeek 同步路径已完成小规模真实供应商烟测。
- 自动化测试覆盖 Windows 与 Ubuntu，并单独验证不安装 GUI 依赖时的 CLI 路径。

### 发行边界

- 本版本以源码 ZIP 交付，需要 Python 3.11+，不是零配置安装包。
- 不包含游戏解包、重新打包或完整游戏 QA。
- 批量写回前必须先执行 `check`，仅在结果为 `safe` 时执行 `apply`。
- 同步翻译会直接修改项目副本，不经过 Batch 的 `check -> apply` 闸门。

[Unreleased]: https://github.com/hu-border-collie/renpy-translation-lab/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/hu-border-collie/renpy-translation-lab/releases/tag/v1.0.0
