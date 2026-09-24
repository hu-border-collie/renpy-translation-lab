# Provider / 执行策略 / 最终审校 Smoke Matrix（#348、#431）

文档地图：[docs/README.md](README.md)

本页给出统一翻译入口、直连 OpenAI-compatible adapter 与最终审校 sync 路径的最小真实验证
路径、离线自动化覆盖、调用预算和安全边界。真实 Provider 调用会产生费用；除明确标注的
手动命令外，仓库内自动化测试全部使用假适配器，不联网、不计费。

实现合同：

- [#431 S1 直连 OpenAI-compatible Sync 生成合同](plans/issue-431-openai-compatible-contract.md)
- [#431 S2 目录发现、诊断与迁移矩阵合同](plans/issue-431-openai-compatible-s2-contract.md)
- [#431 S3 最终审校 Sync 执行合同](plans/issue-431-final-review-sync-contract.md)

## 前置条件

1. 在项目副本或备份目录上操作；先用 `include_files` / `include_prefixes` 限定范围。
2. `translator_config.json` 已包含合法的 `model_routing`，可用
   `python gemini_translate_batch.py profiles-validate --output json` 校验。
3. 凭据只放在安全存储或环境变量：Gemini 使用 `api_keys.json` / `GEMINI_API_KEY*`；
   LiteLLM 内置 Provider 使用系统凭据管理器或 Provider 约定的环境变量；
   直连 `openai_compatible` 使用 `credential_ref`（`none` / `env` / `keyring`）。
   `extra_headers` 只允许非敏感请求头，凭据不得写入配置、manifest 或诊断输出。
4. 直连 adapter 的 `base_url` / `models_url` 必须是干净 http(s) URL：不含 userinfo、query、
   fragment；模型目录不是白名单，目录失败或未收录时始终可以手填模型 ID。
5. 所有真实调用命令都应显式确认计费风险；`profiles-probe` 需要
   `--acknowledge-billable-request I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST`。
6. 最终审校 sync 必须是显式配置：`routes.final_review.strategy=sync`；未显式配置时保持
   legacy `gemini_batch` 入口行为，不会跟随 `defaults.execution_strategy` 静默切换。

## 矩阵

| # | 组合 | Profile 示例 | 单次能力探测 | 端到端最小运行 | 离线自动化覆盖 |
|---|------|--------------|--------------|----------------|----------------|
| 1 | Gemini Sync | `adapter=gemini`，策略 `sync` | `profiles-probe --profile gemini-main --acknowledge-billable-request <TOKEN>` | `sync-start --profile gemini-main --json` → `check <RUN>` → `apply <RUN>` | `test_model_routing_runtime`、`test_gui_durable_sync_page`、`test_sync_run_service`、`test_translation_plan_p4` |
| 2 | LiteLLM 内置 Provider | `adapter=litellm`，`provider=openai|anthropic|deepseek|...`，策略 `sync` | `profiles-probe` + `scripts/run_provider_contract_smoke.py --provider <name>` | `sync-start --profile litellm-main --json` → `check <RUN>` → `apply <RUN>` | `test_litellm_sync_backend`、`test_litellm_runtime_integration`、`test_model_routing_choices` |
| 3 | LiteLLM 自定义兼容 Provider（legacy） | `adapter=litellm` + `base_url` / `models_url` / `api_key_env` | `profiles-probe`；另见 `scripts/run_durable_sync_provider_smoke.py --provider-class custom --custom-base-url ...` | 同 LiteLLM（先跑小范围 `sync-start`） | `test_model_routing_choices`、`test_litellm_sync_backend` 自定义 Provider 用例 |
| 4 | **直连 `openai_compatible` 预设/自定义** | `adapter=openai_compatible`，`provider=openai|openrouter|deepseek|xai|ollama|<custom>`，策略 `sync` | `profiles-list-models --profile <ID> --output json` + `profiles-probe --profile <ID> --acknowledge-billable-request <TOKEN>` | `sync-start --profile <ID> --json` → `check <RUN>` → `apply <RUN>` | `test_openai_compatible_sync_backend`、`test_openai_compatible_config`、`test_openai_compatible_model_catalog`、`test_profile_cli`、`test_model_routing_runtime` |
| 5 | Gemini Batch | `adapter=gemini`，策略 `gemini_batch` | 能力探测只报告声明能力（不会提交 Batch 任务） | `build --profile gemini-batch --json` → `submit <manifest>` → `status` → `download` → `check` → `apply` | `test_gui_workflow_factory`、`test_gui_translation_workflow`、`test_batch_golden_corpus`、`test_durable_sync_workflow`（check/apply 安全层） |
| 6 | **最终审校 Sync** | `routes.final_review.strategy=sync`，profile 可为 Gemini / LiteLLM / 直连 adapter | 复用所属 profile 的 `profiles-probe`（1 次计费请求） | `final-review-build` → `final-review-run-sync <manifest> --limit 1` → `final-review-status <manifest>` → `final-review-export <manifest>` | `test_final_review_sync`、`test_final_review_llm`、`test_model_capability_probe`、`test_gui_final_review_workflow` |

## 最短命令

```powershell
# 1) Gemini Sync
python gemini_translate_batch.py profiles-probe --profile gemini-main `
  --acknowledge-billable-request I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST --output json
python gemini_translate_batch.py sync-start --profile gemini-main --json --strict-exit-codes

# 2) LiteLLM 内置 Provider
python scripts/run_provider_contract_smoke.py --provider openai
python gemini_translate_batch.py sync-start --profile litellm-openai --json --strict-exit-codes

# 3) LiteLLM 自定义兼容 Provider（legacy）
python scripts/run_durable_sync_provider_smoke.py --provider-class custom `
  --custom-base-url https://example.invalid/v1 --custom-model model-id `
  --acknowledge-billable-request I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST

# 4) 直连 openai_compatible：目录（只读）→ 能力探测（1 次计费）→ 小范围 sync
python gemini_translate_batch.py profiles-list-models --profile openai-main --output json
python gemini_translate_batch.py profiles-probe --profile openai-main `
  --acknowledge-billable-request I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST --output json
python gemini_translate_batch.py sync-start --profile openai-main --json --strict-exit-codes

# 5) Gemini Batch（小范围）
python gemini_translate_batch.py build --profile gemini-batch --json
python gemini_translate_batch.py submit <BUILD_MANIFEST> --json
python gemini_translate_batch.py status <BUILD_MANIFEST> --json
python gemini_translate_batch.py download <BUILD_MANIFEST> --json
python gemini_translate_batch.py check <BUILD_MANIFEST> --json
python gemini_translate_batch.py apply <BUILD_MANIFEST> --json

# 6) 最终审校 Sync：先 dry-run 看范围，再限量真实执行
python gemini_translate_batch.py final-review-build --output json
python gemini_translate_batch.py final-review-run-sync <REVIEW_MANIFEST> --dry-run --output json
python gemini_translate_batch.py final-review-run-sync <REVIEW_MANIFEST> --limit 1 --output json
python gemini_translate_batch.py final-review-status <REVIEW_MANIFEST> --output json
python gemini_translate_batch.py final-review-export <REVIEW_MANIFEST> --output json
```

CLI 与 GUI 使用同一选择语义：`--profile` 只覆盖本次运行；GUI 在「设置 → 模型与供应商」
选择后会把同一个 `--profile` 传给 `sync-start` / `build`。已有 run 始终使用冻结路由，
`sync-resume` / `sync-derive` 不接受 `--profile`。

## 调用预算与边界

| 命令 | Provider 调用预算 | 说明 |
|---|---|---|
| `profiles-list-models` | 1 次只读 models GET | 目录仅作填写建议；失败不阻止手填模型 ID。部分 Provider 仍可能计量该请求。 |
| `profiles-probe` | 恰好 1 次计费生成请求 | 报告 `auth` / `sync_generation` / `structured_output` / `reasoning` / `usage`；Batch/embedding 只报告声明能力。 |
| `sync-start` | 由 TranslationPlan 决定 | 先用 `include_files` / `include_prefixes` 限定到 1 个小文件；请求数 = chunk 数（受重试合同影响）。 |
| `final-review-build` | 0 | 只读扫描、冻结上下文、生成 review units；sync 策略不写 Batch requests。 |
| `final-review-run-sync --dry-run` | 0 | 只返回 `planned_unit_ids` / `to_run_unit_ids` / `deferred_count`。 |
| `final-review-run-sync --limit N` | 最多 N 个 unit × 1 次逻辑请求 | 同一 profile/凭据可使用既有重试合同；unit 失败默认继续，系统性失败中止；不会提交 Gemini Batch。 |
| `final-review-status` / `final-review-export` | 0 | 只读 campaign 状态 / findings。 |
| `build → submit → …`（Gemini Batch） | 0 + Batch 请求计费 | `profiles-probe` 不会为该策略提交 Batch 任务。 |

统一原则：不用 mocked 测试冒充真实 smoke；不在公开记录中写入密钥、请求正文、私有脚本或
完整本机路径；未取得凭据的组合保持「未执行」。

## 结果口径

- `profiles-probe` 分别报告：`auth`、`sync_generation`、`structured_output`、`reasoning`、
  `usage` 为 `pass / fail / skipped / not_reported`，`remote_batch`、`embedding` 为
  `declared / unsupported`（Batch 不会为探测提交任务）。
- 直连 adapter 错误使用稳定类别：`authentication` / `rate_limit` / `service_unavailable` /
  `timeout` / `invalid_response` / `unsupported_capability` / `missing_dependency` /
  `provider_error`；目录失败使用 `MODEL_CATALOG_*` / `CREDENTIAL_UNAVAILABLE`。
- `final-review-run-sync` 结果为 `completed / failed / no_work / dry_run`；`failed` 含 unit
  失败，严格模式退出 `4`。`FINAL_REVIEW_SYNC_ABORTED` 表示系统性失败中止，
  `FINAL_REVIEW_USE_RUN_SYNC` 表示 Batch 专用命令收到 sync campaign；
  `FINAL_REVIEW_PLAN_MISSING` 表示损坏 package 缺少冻结路由。
- 探测失败只输出稳定分类与状态，不回显 Provider 异常正文或凭据。
- 端到端结果必须经过 `check <RUN>` / `check <manifest>` 的 `writeback_gate=allow` 才能
  `apply`；`--force` 不能绕过 stale check、源快照或结构阻断。
- 最终审校始终 report-only：`final-review-create-revisions` 只生成标准 revision 预览，
  仍需普通 `preview-revisions → apply-revisions` 门禁。

## 可复制记录模板

真实 smoke 结果请按下面的模板记录到 issue #431（执行策略证据也可同时挂到 #344），
字段全部脱敏。建议每次只跑一个组合，避免把调用预算和错误归因混在一起。

```markdown
### YYYY-MM-DD 直连 openai_compatible smoke

- 环境：<OS / Python / 运行方式；不写完整本机路径>
- adapter：openai_compatible
- provider / 端点类别：<openai|openrouter|deepseek|xai|ollama|custom；host 可脱敏>
- profile id：<PROFILE_ID>
- model：<MODEL_ID>
- credential_ref kind：<none|env|keyring；不写名称以外的值>
- structured_output_mode：<strict_json_schema|json_object|prompt_only_json>
- 命令与调用预算：`profiles-list-models` × 1、`profiles-probe` × 1、`sync-start` 小范围 × <N>
- 实际结果：
  - 目录：<count / CREDENTIAL_UNAVAILABLE / MODEL_CATALOG_*>
  - 探测：auth=<...> sync_generation=<...> structured_output=<...> reasoning=<...> usage=<...>
  - 端到端：items=<...> requests=<...> check=<...> apply=<...>
- 错误 / 备注：<稳定机器码；不粘贴 Provider 原文、密钥、请求正文或私有文本>
```

```markdown
### YYYY-MM-DD final_review sync smoke

- 环境：<OS / Python / 运行方式>
- final_review 路由：profile=<PROFILE_ID> adapter=<gemini|litellm|openai_compatible> strategy=sync
- model / structured_output_mode：<MODEL_ID / MODE>
- 范围：chunk_size=<N> unit_count=<N> item_count=<N>
- 命令与调用预算：
  - `final-review-build`（0 次 Provider 调用）
  - `final-review-run-sync --dry-run`（0 次 Provider 调用；planned=<N> to_run=<N>）
  - `final-review-run-sync --limit <N>`（最多 N 次 Provider 调用）
  - `final-review-status` / `final-review-export`（0 次调用）
- 实际结果：status=<completed|failed|no_work> run_count=<N> skip_count=<N>
  done_delta=<N> failed_delta=<N> finding_count=<N>
- 失败 unit（如有）：unit_id / `sync_<category>`；不粘贴 Provider 原文
- 备注：<上下文变化、render 分组、是否进入 final-review-create-revisions；report-only>
```

记录时不要写：API key、`Authorization`、请求/响应正文、私有对白、完整本机路径、未授权
项目名称。只保留聚合计数、稳定 reason code、脱敏 host 和插件/模型类别。

## 真实执行记录（2026-09-10，历史）

环境：Linux CLI（WSL）；测试项目为真实 Ren'Py 项目的只读副本，范围限定到 1 个文件、
8 条待译对白、2 个 TranslationPlan chunk。所有真实调用均显式确认计费；下表不含凭据、
请求正文或完整本机路径。

| 组合 | Profile / Provider | 能力探测（各 1 次请求） | 端到端 | 写回结果 |
|---|---|---|---|---|
| LiteLLM 内置 | `openrouter` / `openrouter/deepseek/deepseek-v4-flash-0731` | `auth`、`sync_generation`、`structured_output`、`usage` 均 `pass` | `sync-start`：items 8/8、requests 2/2、tokens 2996 | `check=ready`；`apply` 1 文件 8 条；quality gate `pass`，warning 0 |
| LiteLLM 自定义兼容 | `custom-openrouter`，模型 `<id>/deepseek/deepseek-v4-flash-0731`，env 凭据引用 | 四项同步能力均 `pass` | `sync-start`：items 8/8、requests 2/2 | `check=ready`；`apply` 1 文件 8 条；quality gate `pass`，warning 0 |
| Gemini Sync / Batch | Gemini adapter profile | 未执行 | 未执行 | 本环境对 Gemini REST 返回 HTTP 400 `FAILED_PRECONDITION: User location is not supported for the API use.`；需在支持地区按上表手工执行 |

记录说明：

- 该轮自定义 Provider 使用一个 OpenAI-compatible 端点验证了注册、`<id>/<模型>` 改写、
  `api_base` 逐请求透传和 env 凭据读取；当时未取得 OpenCode Go 的独立密钥，因此本记录
  不代表 OpenCode Go 上游已经通过真实 smoke。
- 本轮 smoke 暴露并修复了两个统一入口问题：生成 Profile 不再强制绑定 embedding profile
  （RAG / Source Index 关闭时即可启动完整初译）；`kind=env` 凭据现在按 `name` 读取环境变量，
  preflight / probe 不再误报不可用。
- 离线自动化测试、假适配器测试和真实 Provider 结果是三类不同证据；未执行的组合保持
  「未执行」，不能用离线结果替代。

## 待执行清单（2026-09-19 起）

| 组合 | 需要的证据 | 状态 |
|---|---|---|
| 直连 `openai_compatible` 预设（OpenAI / OpenRouter / DeepSeek / xAI / Ollama 中可获得的） | 目录、probe、小范围 `sync-start` → `check`；至少 1 个 provider | 未执行 |
| 同一 adapter 两个不同 Base URL + 连接间凭据不串用 | 两个 profile、不同 host/credential_ref，交叉探测与 sync | 未执行（离线合同测试仅覆盖解析与序列化） |
| `final-review-run-sync`（至少直连 adapter 或 Gemini Sync） | build → dry-run → limit 1 真跑 → status/export，记录 unit 失败分类 | 未执行 |
| Gemini Sync | 支持地区/Gemini 凭据下的 probe + 小范围 `sync-start` → `check` | 未执行（此前地区受限） |
| Gemini Batch | 1 个小 manifest 的 build → submit → status → download → check | 未执行 |
| Durable sync 中断恢复的真实计费语义 | 中断一次、resume、确认不重复已完成请求 | 未执行 |

## 诊断摘要（可脱敏导出）

```powershell
python gemini_translate_batch.py profiles-show --output json --output-file routing-diagnostics.json
python gemini_translate_batch.py doctor --output json --output-file doctor-diagnostics.json
```

`profiles-show` 只含 Provider 元数据、ModelProfile、能力来源与阶段路由，不含凭据值；
`doctor` 含本机路径与项目结构信息，属于**本地**诊断，提交 issue 前请只保留所需片段。
`profiles-validate` 失败时在 `error.details.issues` 给出稳定 issue 列表，可直接粘贴。

## 记录约定

- 真实 smoke 结果按「日期 / 组合 / profile id / 命令 / 状态 / 请求数 / 备注」记录到
  issue #431（执行策略类同步挂到 #344）；不要在记录中粘贴密钥、请求正文、私有对白或完整
  本机路径。
- 未取得真实凭据的组合保持「未执行」状态，不能用离线测试结果冒充真实 smoke。
- 完成后优先更新本页「待执行清单」与 #431 / #344 的验收证据；必要真实 smoke 全部记录后，
  #431 才进入收口判断（是否移除 LiteLLM、是否扩展 Anthropic/Responses 另行决策）。

## 执行策略差异（成本 / 延迟 / 可靠性 / 能力）

| 维度 | 同步（sync） | 最终审校 sync（`final_review`） | Gemini Batch（gemini_batch） |
|---|---|---|---|
| 计费 | 按即时调用计费，无 Batch 折扣 | 与 sync 相同，按 unit 即时调用 | 可能享受 Batch 折扣，按提交的批量请求计费 |
| 延迟 | 请求级即时返回，受单请求 timeout 限制 | 每 unit 即时返回；长 campaign 受 limit/中断边界影响 | 云端排队与异步处理，完成时间取决于任务量与配额 |
| 可靠性 / 恢复 | 耐久运行持久化 request/attempt，可 status/resume/cancel/derive | 每个 unit 原子落盘；重复运行按 digest 续跑，不做 socket/跨进程恢复 | manifest + job state + download/check 恢复 |
| 能力前提 | profile 需声明 `sync_generation`；RAG/embedding 可即时参与 | profile 需声明 `sync_generation`；`routes.final_review.strategy=sync` 显式配置；report-only | profile 需为 Gemini adapter 且声明 `remote_batch` |
| 典型用途 | 小批量、补译、需要即时反馈或分阶段路由 | 无 Gemini Batch 时对普通译文逐条审校；找不到问题时也进入 findings 生命周期 | 大规模初译、可等待排队、需要 Batch 折扣 |

Sync / Batch 是执行策略，没有“主流/备选”之分；同一 ModelProfile 可以同时支持两者，切换不复制模型或凭据。
直连 `openai_compatible` adapter 只服务同步生成与最终审校 sync，不提供远程 Batch。
