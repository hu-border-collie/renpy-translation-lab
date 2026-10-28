# #520 OpenRouter Batch 研究合同 v1

> 归档于 2026-10-27：保留原日期、基线、结果与未验边界。现行操作见[文档地图](../../README.md)，当前工作见[计划索引](../../plans/README.md)。

日期：2026-10-20。基线：`main@b964723f8f0e81ffdc7f6abaf3b5cd951b6fe482`。
这是研究草案，**未增加生产策略、配置或 CLI/GUI**。实测结论为
[暂缓接入](issue-520-openrouter-batch-spike-2026-10-20.md)。
用户后续授权换现代模型的条件与状态见 [DeepSeek v2 续验](issue-520-openrouter-batch-deepseek-2026-10-20.md)。

## 1. 可复用部分与新边界

| 现有模块 | 本研究复用 / 后续接入边界 |
| --- | --- |
| `model_profile.ModelProfile` / `ModelRoutingPlan` | 冻结模型、adapter、服务地址、凭据引用、路由及能力来源；`provider=openrouter` 与实际推理上游 `OpenAI` 分开记录 |
| `model_profile.ExecutionStrategy` | 当前只有 `sync` / `gemini_batch`；候选 `openrouter_batch` 尚未注册，不能拿 Sync 成功或结构化输出能力代替 Batch 能力 |
| `translation_plan` | 原样复用 source identity、chunk、system/user prompt、schema、生成参数、context、plan/request/prompt 指纹；传输不得重组 prompt |
| `openai_compatible_sync_backend` | 离线 harness 调用现有 `_build_payload` 构造同一 Chat Completions body；私有方法复用只用于本次研究，不承诺新公共 API |
| `translation_core.validate_model_response`、`structure_protection.validate_report` | 复用精确 item ID、字段/空译检查与绑定占位符恢复；不把字符串数组按位置猜成有 ID 的结果 |
| `sync_run_contracts` / `sync_run_store`、`batch_submit_recovery` | 沿用先持久化再发送、未知结果保留预留、不可盲目重试等原则；不伪造 Sync Attempt 或 Gemini upload/job 回执 |
| `sync_result_export`、既有 Batch check/apply | 后续才实现适配层：保留 `key=chunk_id`、plan/request 身份、原始回执与 `normalized_response` 的区别；研究产物不可直接导入现有 manifest |
| **必须新增** | OpenRouter 的 inline submit / workspace list / GET inline results / terminal delete、逐条错误解析、路由与费用审计及持久状态。它不是 Sync backend 上加一个异步开关，也不需要通用调度器 |

生产接入前必须适配现有 manifest/results reader 并回归 check/apply。
`writeback_gate.decision=allow` 仍须来自当前 manifest/results 的最近一次检查；
质量报警沿用项目规则，stale check、源快照、结构 blocker 和 `apply --force` 边界保持不变。
`candidates`、远端 `completed`、HTTP 202 都不授予写回权。

## 2. 请求身份与能力快照

研究根对象 `contract_version=1`，含 `source`、现有 `plan`、`mapping`、`envelope`
及 `frozen_sha256`。摘要采用现有 `canonical_json` 的 UTF-8 SHA-256；原有短指纹不改格式。
摘要提供完整性检查，不是签名，不能扩大制品信任边界。

能力记录至少包括：adapter/base URL、基础 model slug、目录 variant ID、上游 endpoint tag、
支持的 endpoint/参数、价格、目录查询时间和账号 smoke 状态。公开目录声明与账号实际可用性
是不同证据；本次已出现两者冲突。ModelProfile 的普通 `structured_output` 声明不足以启用 Batch。

凭据只引用现有 `keyring:openrouter` / `env`；值仅在必要调用子进程中解析，不进入计划、
回执摘要、命令行参数或仓库。不得读取配置中的 prepare 命令来运行研究 fixture。
账号的 BYOK 优先级/共享容量策略也是计费条件，不因 `provider.only` 自动固定为普通额度。

### 2.1 固定请求形状

```json
{
  "endpoint": "/v1/chat/completions",
  "model": "openai/gpt-4.1-mini",
  "provider": {"only": ["openai"], "allow_fallbacks": false},
  "completion_window": "24h",
  "requests": [{"custom_id": "i520-<40 hex>", "body": {"messages": [], "max_tokens": 1024}}]
}
```

上例只展示 envelope；真实 body 还含完整 system/user messages、temperature=0.2 和
现有 schema 的 `response_format={type:json_schema,json_schema:{name,schema,strict:true}}`。
不能发送这个空 messages 示例。

- 顶层字段按上例顺序序列化，`requests` 最后。语义摘要可排序，但不能让摘要序列化器
  替代官方要求的 wire 字段顺序。
- `model` 使用官方 Batch 示例的基础 slug；目录的 `:batch` 是能力发现依据，不能未经确认
  改写提交模型。body 省略 model 继承顶层值，省略 streaming。
- 每批只有一个模型、endpoint、上游约束。`only` 与 `allow_fallbacks` 是当前 Batch 文档允许的
  Provider 参数；不复制 Sync 的 `order`、`sort` 等参数，也不默许模型/Provider 轮换。
- `custom_id = i520- + sha256({plan:plan_fingerprint,request:完整冻结请求})[:40]`，批内唯一。
  mapping 保存完整 TranslationRequest、wire body SHA-256；遇到 ID 碰撞必须拒绝。
- 重新验证整个冻结摘要、现有 plan 指纹，并从原创 source 重建请求，避免只相信存储的摘要。
  真实发送还须等于预算授权绑定的冻结摘要。source snapshot、prompt、schema 或参数变更均是新实验。

## 3. 结果与错误映射

读取 GET 回执时首先校验已持久化的 batch ID、顶层 endpoint、model、实际 Provider、
已知状态及请求计数。单条结果再检查 model；本研究仅允许目录基础 slug 及明确的
`gpt-4.1-mini-2025-04-14` / `openai/gpt-4.1-mini-2025-04-14` snapshot，未知别名须核查。
现代 v2 仅允许 DeepSeek V4.1 Flash 基础 slug、目录 20260910 snapshot 和明确的上游
`deepseek-ai/DeepSeek-V4.1-Flash` / `DeepSeek-V4.1-Flash` ID，Provider 必须是 DeepInfra。
顶层 model 也可返回该冻结白名单内的解析后 snapshot，不能因此误报为未知模型切换。
原始回执保留在忽略目录，公开材料只记录脱敏摘要。

| 情况 | 本地规则 |
| --- | --- |
| 乱序结果 | 只用 `custom_id` 查 mapping，不按数组位置关联 chunk |
| 重复 custom_id / 未知 custom_id / 错 batch 或 route | 整份回执隔离，禁止选择“第一条”或“最后一条”覆盖 |
| 缺少预期 custom_id | 显式 `MISSING_CUSTOM_ID`，保留 unresolved 身份；成功兄弟请求可查看，整包不标完整 |
| response/error 同时存在或同时为空 | `RESPONSE_ERROR_EXCLUSIVITY`，拒绝回执 |
| per-item error / 非 200 response | `REMOTE_ITEM_ERROR` / `REMOTE_HTTP_ERROR`；错误与费用分开存储，不能生成成功回执 |
| `finish_reason=length`、拒答、非 stop | `INCOMPLETE_OR_REFUSED`，不 salvage 残缺 JSON |
| JSON、item ID、字段或占位符失效 | 现有 core/protection 验证失败，`INVALID_CONTENT_OR_ITEM_IDS`；不推测缺失 ID、不下游 strip |
| 正常结果 | 保存原始 result 摘要、generation ID、chunk/request/prompt 身份和恢复后的规范候选；仍为 `writeback_authorized=false` |
| 缺少 usage/cost | 保持 null/unknown，不能填零；原始回执摘要与候选摘要分开 |

`map_receipt` 只处理本地研究回执；输出不是生产 results.jsonl，不写游戏文件。
10-21 已从 DeepSeek v2 终态取得真实成功及乱序结果，2/2 请求、6/6 item 精确关联通过；
见[终态报告](issue-520-openrouter-batch-deepseek-2026-10-20.md)。其余异常、逐条失败及
部分完成分支仍只有原创离线证据，不能据此宣称真实故障恢复已全部验证。

## 4. 持久状态与安全恢复

独立研究记录包含：版本、冻结摘要、attempt 标识、发送时间、请求摘要、费用预留、已知
batch/generation ID、实际 route、最后查询时间、远端过期时间、回执摘要、下一步安全动作。
公开仓库不保存凭据、原始远端结果或私有账本。一次会话失败不能导致重新发送同一批次。

| 本地 / 远端状态 | 恢复规则 |
| --- | --- |
| prepared，尚未发送 | 验证预算及冻结摘要后，先落盘 dispatch marker 和预留，再发送一次 |
| dispatched，无确定回执 / 存储回执失败 / timeout / 5xx | `outcome_unknown`；保留全部预留，只读 list / GET / 支持核对。无 ID 不等于未提交，列表未匹配也不足以证明可重提 |
| 收到 202 + ID | 只表示 accepted；立即持久化 ID，然后 GET，不再 POST |
| 明确 400/401/402/403/404/422 拒绝 | 保存拒绝；停止该 attempt。认证、权限、TOS、能力问题不得通过静默换模型/Provider 绕过 |
| 429 / 容量失败 | 不假定 POST 幂等；先确认是否创建作业及账务，再决定是否需要单独授权的新 attempt |
| validating / in_progress / finalizing | GET；保存 `provider` 与 `expires_at`。用上游实际过期时间判断，不机械在 24 小时重提 |
| cancelling | 仅等待状态；本次未确认 OpenRouter 公开取消调用合同，不臆造 `/cancel` 接口 |
| completed | 校验 inline results 并关联；允许逐条失败，不把 batch completed 当作所有请求成功 |
| failed / expired / cancelled | 当前官方说明 results=null，即使 completed count>0 也不制造可恢复候选；费用可能不为零 |
| GET 暂时失败 | 只重试 GET，保存最后已知状态；不得触发提交重试 |
| GET 410 | 输入/结果保留已过；保留账务和缺口，不能重新生成相同请求来“恢复原回执” |
| 终态删除 | 先保存结果/摘要并做明确清理动作；DELETE 200 才表示适用清理完成。409 不是取消，5xx 仅按文档重试删除，不重提生成 |

针对已确定失败/缺失的修复若另行获准，使用新 attempt 与父子身份，只处理冻结 unresolved
白名单；不得覆盖旧成功项。未知提交的预留不能转给修复请求。两个独立的同内容提交不会
因 custom_id 相同而自动去重，官方未声明 submit 的服务端幂等键。

轮询拟采用 30/60/120 秒退避；每次查询独立有界，不做超过 60 秒的阻塞等待。
达到本次观察时限后持久化下次安全查询信息，不能靠重提缩短排队时间。研究未建设调度器。

## 5. 费用口径

普通额度记录 OpenRouter `usage.cost`；BYOK 另记 `usage.is_byok`、OpenRouter fee 和
`usage.cost_details.upstream_inference_cost` 及 prompt/completion 拆分。后者是上游估算，
不冒充上游账单，也不能与 OpenRouter fee 相加后再重复计入总成本。
usage token、目录估价、Provider 实报费用、预留和最终账单核对分别记录。

本单预算独立于 #508：明确授权 USD 0.10；固定两条原创请求及最多 1024 输出 token，
计划三臂合计 6 次生成，不自动重试。拒绝/未知的预留保持占用，未使用授权不代表可新增实验。

## 6. 离线复现与后续范围

```powershell
python -B scripts/openrouter_batch_spike.py freeze --output-dir tmp/issue-520/rebuilt
python -B scripts/openrouter_batch_spike.py freeze --experiment deepseek41-v2 --output-dir tmp/issue-520/deepseek-v2/rebuilt
python -B -m unittest tests.tooling.test_openrouter_batch_spike -q
python -B scripts/openrouter_batch_spike.py inspect --help
```

fixture 位于 [source.json](../../../tests/fixtures/openrouter_batch_spike/source.json)，
[harness](../../../scripts/openrouter_batch_spike.py) 不含 HTTP 或凭据访问；
[测试](../../../tests/tooling/test_openrouter_batch_spike.py) 的合成回执显式标记为离线，不构成 Provider smoke。
研究 harness 没有产品 GUI，这是范围内的有意例外，不是一个未同步的生产功能。

若未来有明确需求并补齐保留/质量与完整对照缺口，再独立拆最小生产范围：能力/配置与 runtime reader、现有
ModelRoutingPlan 的显式策略选择、共享请求/result adapter 和耐久作业服务；CLI
build/submit/status/results/check 接线、GUI 同等入口/状态/恢复提示、settings schema、
example config、`user_copy.py`、诊断命令参考、文档和针对性回归。默认策略、旧配置及
Gemini/LiteLLM 的保留/迁移另按既有合同处理。本次 PR 只发布研究合同、原创 fixture、
离线 harness 和脱敏报告，不创建生产实施任务。

## 官方依据（2026-10-20 查阅）

- [OpenRouter Batch](https://openrouter.ai/docs/batch-quickstart)：请求形状、route、状态、inline results、保留与删除。
- [实际 variant 目录](https://openrouter.ai/api/v1/models/openai/gpt-4.1-mini:batch/endpoints)：目标 endpoint、参数和价格声明。
- [OpenRouter BYOK](https://openrouter.ai/docs/guides/overview/auth/byok) 与[现行价格政策](https://openrouter.ai/pricing)：账号路由与 fee 口径。
- 现有生产行为参考 [Batch 工作流](../../batch_workflows.md)、[Provider smoke](../../provider_smoke_matrix.md)及 [#431 迁移矩阵](../research/issue-431-litellm-migration-matrix.md)。
