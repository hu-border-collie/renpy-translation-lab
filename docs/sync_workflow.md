# 同步翻译工作流

文档地图：[docs/README.md](README.md)

同步 CLI 适合小范围即时翻译、补译和局部验证。它与 Gemini Batch 主路径使用不同的运行合同：默认命令会调用所选同步供应商，但**只生成可审查预览**；只有显式传入该次预览的 manifest 才会写回项目。

大型任务、远程排队、成本折扣和可恢复下载仍优先使用 [Batch 工作流](batch_workflows.md)。GUI 用户可直接阅读 [GUI 工作台 · 同步翻译](gui_workbench.md#同步翻译)。

## 前置条件

1. 使用 Python 3.11+ 安装主依赖；使用 GUI 或 LiteLLM 时再安装对应可选依赖。
2. 在 `translator_config.json` 明确设置当前 `game_root` 与 `tl_subdir`，并用只读环境检查确认项目：

   ```powershell
   python gemini_translate_batch.py doctor
   ```

3. Gemini 后端需要本地 `api_keys.json` 或 `GEMINI_API_KEY*` 环境变量；LiteLLM 后端需要在操作系统凭据管理器或供应商约定的环境变量中保存凭据。
4. 先备份项目，并用 `include_files` / `include_prefixes` 把第一次运行限制在少量文件。

同步设置来自 `translator_config.json` 的 `sync` 段，主要包括 `backend`、`model`、`chunk_size`、`max_source_chars`、`max_output_tokens`、`timeout_seconds`、`context_before`、`context_after` 和 `macro_setting_file`。完整字段和模型目录见 [安装与本地配置](setup.md#运行模式)。

`timeout_seconds` 默认 120 秒，可设为 5–600 秒。它是每一次模型请求的等待上限，不是整次任务的总时限；普通同步翻译、项目分析、同步关键词、同步订正、同步修补和翻译 A/B 对比均读取同一字段。Gemini backend 会把秒转换为 SDK 的毫秒级 `http_options.timeout`，LiteLLM backend 则按秒透传。手工配置超出范围时 runtime 会收敛到最近边界，避免异常值形成无界等待。

选择 LiteLLM 后端时，可通过 `sync.custom_litellm_providers` 注册 OpenAI 兼容但 LiteLLM 未内置的服务（OpenCode Go、中转站、本地 vLLM 等）：每项配置 `id` / `label` / `base_url` / `models_url` / `api_key_env`，请求会改写为 `openai/<模型>` 并逐请求透传 `api_base`，密钥优先使用系统凭据管理器。字段与示例见 [安装与本地配置 · 自定义 OpenAI 兼容 Provider](setup.md#自定义-openai-兼容-providerlitellm-同步)。

## Reasoning 与输出预算诊断

同步结果和实际模型用量账本会分别记录 Provider 可提供的 `completion_tokens`、`reasoning_tokens` 与正文输出 Token。正文计数缺失时显示 `unknown`，不会跨 Provider 假定 `completion - reasoning` 就是正文。若空正文/截断同时伴随 reasoning 计数和输出预算耗尽，结果会记录稳定原因码（例如 `reasoning_budget_exhausted`），GUI 摘要会显示 reasoning 预算告警和截断次数。结果、日志和 GUI 只显示计数与原因码，不回显 Provider 异常正文或凭据。

项目没有增加伪装成通用能力的 reasoning 配置。`thinking_level` / `thinking_config` 仍是 Gemini 专属语义：Gemini 原生后端可消费它；LiteLLM 后端不会把该字段发送给 OpenAI-compatible Provider，而是在安全的 `request_metadata.ignored_provider_options` 中记录该能力未应用。未来某个 Provider 若要支持专属 reasoning 参数，应通过显式 capability/provider options 接入并独立测试。

## 错误分类与有界恢复

生产同步入口优先读取 backend 提供的结构化错误分类；只有 Google SDK 未提供分类时才使用状态码、异常类型和窄化的兼容判断。恢复规则固定为：

| 分类 | 行为 |
|---|---|
| `authentication` | 立即失败；不重试、不拆包、不轮换模型 |
| `rate_limit` | 有界退避；Gemini 只使用 Gemini 自己的轮换策略，LiteLLM 可在同一 Provider 的保存密钥集合内尝试下一把 Key |
| `service_unavailable` / `timeout` | 在同一请求上有界重试；耗尽后失败，不用拆包制造更多请求 |
| `invalid_response` | 翻译请求可按现有合同做定点重试/拆包；不会当作网络故障无限重试 |
| `unsupported_capability` / `missing_dependency` | 立即失败，由用户修正 Provider、模型或依赖 |
| 未分类 Provider 错误（`provider_error`） | 不重试，但保留拆包兜底：多行 batch 失败时拆成两半继续，把可能由单行引起的问题隔离，健康行仍可产出；单行仍未解决时按既有合同记录失败 |

LiteLLM 多 Key 轮换只发生在当前 Provider 的凭据集合内，记录形如 `openai#2:key:3f9a2c1b4e` 的**不可逆** identity（密钥 SHA-256 摘要前缀，不含任何原始密钥字符）；不会调用或修改 Gemini keyring。401/403 不会尝试下一把 Key。所有退避与重试均受固定次数和单请求 `timeout_seconds` 约束。

后端错误在 UI 与结果文件中只显示安全摘要（如 `provider request failed [provider_error]`），不回显 Provider 异常正文或凭据。为保留排障线索，backend 会以 `raise ... from exc` 保留原始异常链，翻译流程的本地失败日志（`logs/`）会在未分类 Provider 错误时附上截断的原始消息；这些原始文本不会进入 GUI 或任何结果/manifest 文件。

GUI 的停止操作会使当前 CLI 任务以失败/取消状态收尾；即使子进程在终止竞态中返回 0，也不会接受其迟到成功状态或继续进入 preview/apply 流程。

## 模型结果合同与定点重试

同步翻译、同步订正和关键词提取共用同一条 provider-neutral 校验边界。新请求要求模型返回带名称的 JSON 对象：翻译为 `{"translations":[...]}`，订正为 `{"revisions":[...]}`，关键词为 `{"candidates":[...]}`。旧任务中的裸数组结果仍可读取，但只作为迁移兼容；新提示词和 schema 不再生成裸数组合同。

校验会稳定记录无效 JSON、缺失或重复 ID、未知 ID、缺失字段、字段类型错误、空译文及关键词证据引用未知源 ID 等原因。翻译与订正要求每个请求 ID 恰好出现一次；先返回的有效项会保留，只对缺失或无效 ID 发起定点重试，不会重译已经通过合同的项。关键词没有一对一输出数量，但每个候选必须引用至少一个属于当前请求的证据 ID；非空 `chunk_summary` 也必须通过 `summary_evidence_item_ids` 引用至少一个当前请求 ID。合同失败时会只重试当前关键词 chunk。

终端、GUI 与 manifest 都会显示首次/最终完整率、定点重试次数和未解决项。同步翻译中最终有效的条目数，只统计同时通过模型合同和本地 adapter 校验、可以进入安全 preview 的条目；关键词没有预期候选数量，因此明确显示完整请求块数与请求块完整率，而不把 chunk 数描述成候选条目数。重试后仍不完整时任务标为 `partial`，同时保留通过合同的结果：同步翻译只为这些结果生成安全 preview；同步订正和关键词报告也会明确显示部分完成。`partial` 不是质量认可，写回前仍须人工审查。

同步 `results.jsonl` 会同时保留审计原始数据和最终合同结果：`response` 始终是首轮 Provider 原始响应，供兼容读取与用量核算；`provider_response_attempts` 为首轮及定点重试逐次记录合同诊断，重试项还保留原始响应；`normalized_response` 是合并首轮与重试后的权威合同结果，校验、preview、导出和 apply 路径均应优先读取它。每行的 `response_semantics` 会显式记录这些字段的角色，避免把首轮缺项误判为最终仍不完整，同时保留已恢复违规的逐次审计信息。

供应商能力是显式策略：Gemini 原生使用 JSON schema；已知支持严格 schema 的 LiteLLM Provider 使用 strict JSON schema，只支持 JSON mode 的 Provider 使用 JSON object；未知或不可靠的端点降级为 prompt-only JSON，并继续经过同一校验与重试边界。使用真实凭据做有界冒烟测试时，可运行：

```powershell
python scripts/run_provider_contract_smoke.py --provider gemini
python scripts/run_provider_contract_smoke.py --provider deepseek
```

脚本每个 Provider 最多发送一次、限制输出 token 和超时；通过后只打印 completion/reasoning/正文 Token 的安全摘要，失败时只打印分类与安全错误文案。未配置对应凭据会跳过，不会修改项目文件。正式项目仍应先用小范围同步任务验证实际翻译/订正 envelope。

## 局部上下文、Macro 与术语命中

同步初译提示词包含三类基础上下文，全部只作参考，模型仍只能返回 TARGET 条目的 ID 与译文：

- **局部前后文**：`sync.context_before`（默认 30）与 `sync.context_after`（默认 10）控制每个请求附带的 `CONTEXT BEFORE/AFTER` 条目预算，与 Batch 默认对齐。窗口只取当前文件内的待译条目，并在可识别的 translate block 边界处提前截断——不会静默跨越场景；没有 block 信息时退化为纯预算截断。设为 0 可关闭对应方向。
- **项目风格设定**：`sync.macro_setting_file`（默认 `macro_setting.md`，相对当前 work）存在时，其文本会进入提示词的 `Setting` 段；文件不存在或未配置时保持向后兼容，提示词不含该段。
- **词法术语命中**：`normalize_map`、`preserve_terms` 与 `non_translatable_exact` 的本地命中不再依赖 `sync.rag.enabled`。即使 RAG 关闭，当前批次实际命中的固定译法与保留/不可翻译规则也会进入提示词；命中不受 `sync.rag.top_k_terms` 截断，全部注入（该配额在 RAG 开启时仍限制 `LOCKED TERMS` 检索列表）。RAG 开启时检索命中照常附加。

每次运行的上下文构造事实会写入 manifest 与预览制品：

- manifest 顶层 `prompt_context`：前后文设置、macro 文件与内容指纹、是否实际注入 macro、批次总数与截断批次数；
- 每个文件的 `prompt_context.batches`：该文件各请求实际的前后文条目数/字符数、预算截断与 block 边界截断标记。

`prompt_context` 与文件级上下文诊断都纳入 manifest 指纹。源文件变化或预览制品被修改会直接使旧 manifest 无法通过写前校验；`--apply` 会把当前 macro 文件指纹与 manifest 记录的指纹比对，两者不一致（macro 文件新增、删除或内容变化）即拦截写回——不要强行复用旧预览，应基于当前文件重新生成并审查。未记录 `prompt_context` 的旧 manifest 保持可用。macro 路径限定在当前项目（`game_root`）内，配置指向项目外时会被忽略。

## 预览后写回

### 1. 生成预览

```powershell
python gemini_translate.py
```

命令会扫描当前项目的待译项、调用同步模型，并在 `logs/sync_runs/<run>/` 生成：

- `manifest.json`：绑定当前项目、TL 目录、源文件快照、预览制品哈希和 adapter 写回计划；
- `preview.diff`：供人工逐项审查的差异；
- 预览候选文件与只读 coverage 证据。

manifest 同时记录本次运行的局部上下文、macro 与术语命中诊断（见[局部上下文、Macro 与术语命中](#局部上下文macro-与术语命中)），用于复现和解释每个请求的上下文构造。

默认命令不会修改 `.rpy`，终端会打印本次 manifest 和 diff 的绝对路径。若模型结果合同仍有未解决项，或部分文件未通过 adapter 写回计划校验，运行结果会标为 `partial`；无效项不会进入可写回预览，已经通过合同与 adapter 校验的项仍会保留。

只有明确需要运行配置中的 prepare 步骤时才使用：

```powershell
python gemini_translate.py --prepare
```

`translator_config.json` 的 prepare 自定义命令属于可执行本地配置；不要对来源不明的配置使用 `--prepare`。

### 2. 审查预览

写回前至少确认：

- `preview.diff` 中的原文、译文、占位符、Ren'Py 标签和说话人均正确；
- manifest 属于当前项目和本次运行，不是其他游戏或旧任务；
- 没有未理解的 `preview_failures`；
- 译文已经过必要的术语、语气和机械质量检查。

同步预览通过结构校验也不代表内容质量合格。与 Batch 的 `writeback_gate=allow` 一样，结构安全和翻译质量是两件事。

### 3. 显式写回

```powershell
python gemini_translate.py --apply logs/sync_runs/<run>/manifest.json
```

`--apply` 不会重新调用模型。写回前会重新核对当前项目、TL 目录、每个源文件快照、预览制品哈希和 adapter 计划；项目切换、源文件变化或预览制品被修改都会阻止写回。遇到阻断时不要强行复用旧 manifest，应基于当前文件重新生成并审查预览。

`--prepare` 与 `--apply` 不能同时使用。同步 CLI 当前输出面向人类阅读，不提供 Batch 核心命令的 JSON envelope；自动化需要稳定机器合同、远程状态轮询或断点恢复时应改用 Batch。

## Gemini 与 LiteLLM 数据边界

- **Gemini 同步**：本工具从本机通过 Google `google-genai` SDK 直接调用 Gemini API。本项目没有自建中转服务，也不会上传整个 `api_keys.json` 文件；但 API 调用必然会把认证信息发送到 Google，并把待译文本、提示词以及启用的 glossary、macro、RAG、Source Index、Story Memory 或 Project Analysis 上下文发送给 Gemini。
- **LiteLLM 同步**：本工具从本机调用 LiteLLM Python SDK，再按所选 Provider / API Base 访问供应商。本项目不提供自建 LiteLLM 代理；待译文本、提示词和必要上下文会发送到所选供应商。凭据保存在操作系统凭据管理器或环境变量中，不写入 `translator_config.json`。
- **本地产物**：manifest、diff、模型结果摘要、用量账本与日志保存在本机；它们可能包含私有游戏文本，不应提交到公开仓库或发给无权访问者。

免费与付费服务的数据使用、日志、训练和保留政策可能不同，也可能随供应商更新。处理敏感、保密或无权发送的游戏文本前，应先核对所用账号层级和供应商当前条款；Gemini 可参考 [Additional Terms](https://ai.google.dev/gemini-api/terms) 与 [Data Logging and Sharing](https://ai.google.dev/gemini-api/docs/logs-policy)，LiteLLM 则以实际 Provider 的政策为准。

## 与 Batch 的关键差异

| 维度 | 同步 CLI | Gemini Batch |
|---|---|---|
| 调用方式 | 进程内逐批即时请求 | 远程异步 job |
| 默认写回 | 只生成 preview，显式 `--apply` | `download -> check -> apply` |
| 安全合同 | sync manifest + 源快照 + 制品哈希 + adapter 计划 | manifest/results identity + 最近一次 `writeback_gate=allow` + 写回前复核 |
| 状态恢复 | 复用已生成 preview；阻断后重新生成 | `status` / `download` / submit recovery / retry package |
| 机器输出 | 人类可读文本 | 核心命令支持版本化 JSON envelope |
| 费用语义 | 供应商同步计费，无 Batch 折扣 | Gemini Batch 定价；提交前可 `estimate-cost` |

两条路径都不能代替完整游戏 QA。写回后仍应运行 Ren'Py lint、机械质量检查，并进行人工/LLM 语义审校。
