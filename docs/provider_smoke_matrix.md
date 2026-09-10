# Provider / 执行策略 Smoke Matrix（#348 P3）

文档地图：[docs/README.md](README.md)

本页给出统一翻译入口四类组合的最小验证路径、离线自动化覆盖，以及需要真实凭据时
`最多调用次数` 与安全边界。真实 Provider 调用会产生费用；除明确标注的手动命令外，
仓库内自动化测试全部使用假适配器，不联网、不计费。

## 前置条件

1. 在项目副本或备份目录上操作；先用 `include_files` / `include_prefixes` 限定范围。
2. `translator_config.json` 已包含合法的 `model_routing`，可用
   `python gemini_translate_batch.py profiles-validate --output json` 校验。
3. 凭据只放在安全存储或环境变量：Gemini 使用 `api_keys.json` / `GEMINI_API_KEY*`；
   LiteLLM 内置 Provider 使用系统凭据管理器或 Provider 约定的环境变量；
   自定义 OpenAI-compatible Provider 使用其 `api_key_env` 或 keyring 槽位。
4. 所有真实调用命令都应显式确认计费风险；`profiles-probe` 需要
   `--acknowledge-billable-request I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST`。

## 矩阵

| # | 组合 | Profile 示例 | 单次能力探测（1 次请求） | 端到端最小运行 | 离线自动化覆盖 |
|---|------|--------------|--------------------------|----------------|----------------|
| 1 | Gemini Sync | `adapter=gemini`，策略 `sync` | `profiles-probe --profile gemini-main --acknowledge-billable-request <TOKEN>` | `sync-start --profile gemini-main --json` → `check <RUN>` → `apply <RUN>` | `test_model_routing_runtime`、`test_gui_durable_sync_page`、`test_sync_run_service`、`test_translation_plan_p4` |
| 2 | LiteLLM 内置 Provider | `adapter=litellm`，`provider=openai|anthropic|deepseek|...`，策略只能 `sync` | 同上 + `scripts/run_provider_contract_smoke.py --provider <name>` | `sync-start --profile litellm-main --json` → `check <RUN>` → `apply <RUN>` | `test_litellm_sync_backend`、`test_litellm_runtime_integration`、`test_model_routing_choices` |
| 3 | 自定义 OpenAI-compatible | `adapter=litellm` + `base_url` / `models_url` / `api_key_env` | 同上；另见 `scripts/run_durable_sync_provider_smoke.py --provider-class custom --custom-base-url ...` | 同 LiteLLM（先跑小范围 `sync-start`） | `test_model_routing_choices`、`test_litellm_sync_backend` 的自定义 Provider 用例 |
| 4 | Gemini Batch | `adapter=gemini`，策略 `gemini_batch` | 能力探测只报告声明能力（不会提交 Batch 任务） | `build --profile gemini-batch --json` → `submit <manifest>` → `status` → `download` → `check` → `apply` | `test_gui_workflow_factory`、`test_gui_translation_workflow`、`test_batch_golden_corpus`、`test_durable_sync_workflow`（check/apply 安全层） |

## 四类最短命令

```powershell
# 1) Gemini Sync
python gemini_translate_batch.py profiles-probe --profile gemini-main `
  --acknowledge-billable-request I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST --output json
python gemini_translate_batch.py sync-start --profile gemini-main --json --strict-exit-codes

# 2) LiteLLM 内置 Provider
python scripts/run_provider_contract_smoke.py --provider openai
python gemini_translate_batch.py sync-start --profile litellm-openai --json --strict-exit-codes

# 3) 自定义 OpenAI-compatible
python scripts/run_durable_sync_provider_smoke.py --provider-class custom `
  --custom-base-url https://example.invalid/v1 --custom-model model-id `
  --acknowledge-billable-request I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST

# 4) Gemini Batch（小范围）
python gemini_translate_batch.py build --profile gemini-batch --json
python gemini_translate_batch.py submit <BUILD_MANIFEST> --json
python gemini_translate_batch.py status <BUILD_MANIFEST> --json
python gemini_translate_batch.py download <BUILD_MANIFEST> --json
python gemini_translate_batch.py check <BUILD_MANIFEST> --json
python gemini_translate_batch.py apply <BUILD_MANIFEST> --json
```

CLI 与 GUI 使用同一选择语义：`--profile` 只覆盖本次运行；GUI 在「设置 → 模型与 Provider」
选择后会把同一个 `--profile` 传给 `sync-start` / `build`。已有 run 始终使用冻结路由，
`sync-resume` / `sync-derive` 不接受 `--profile`。

## 结果口径

- `profiles-probe` 分别报告：`auth`、`sync_generation`、`structured_output`、`reasoning`、
  `usage` 为 `pass / fail / skipped / not_reported`，`remote_batch`、`embedding` 为
  `declared / unsupported`（Batch 不会为探测提交任务）。
- 探测失败只输出稳定分类（如 `authentication` / `rate_limit` / `timeout`）与状态，
  不回显 Provider 异常正文或凭据。
- 端到端结果必须经过 `check <RUN>` / `check <manifest>` 的 `writeback_gate=allow` 才能
  `apply`；`--force` 不能绕过 stale check、源快照或结构阻断。

## 真实执行记录（2026-09-10）

环境：Linux CLI（WSL）；测试项目为真实 Ren'Py 项目的只读副本，范围限定到 1 个文件、
8 条待译对白、2 个 TranslationPlan chunk。所有真实调用均显式确认计费；下表不含凭据、
请求正文或完整本机路径。

| 组合 | Profile / Provider | 能力探测（各 1 次请求） | 端到端 | 写回结果 |
|---|---|---|---|---|
| LiteLLM 内置 | `openrouter` / `openrouter/deepseek/deepseek-v4-flash-0731` | `auth`、`sync_generation`、`structured_output`、`usage` 均 `pass` | `sync-start`：items 8/8、requests 2/2、tokens 2996 | `check=ready`；`apply` 1 文件 8 条；quality gate `pass`，warning 0 |
| 自定义 OpenAI-compatible | `custom-openrouter`，模型 `<id>/deepseek/deepseek-v4-flash-0731`，env 凭据引用 | 四项同步能力均 `pass` | `sync-start`：items 8/8、requests 2/2 | `check=ready`；`apply` 1 文件 8 条；quality gate `pass`，warning 0 |
| Gemini Sync / Batch | Gemini adapter profile | 未执行 | 未执行 | 本环境对 Gemini REST 返回 HTTP 400 `FAILED_PRECONDITION: User location is not supported for the API use.`；需在支持地区按上表手工执行 |

记录说明：

- 自定义 Provider 使用一个 OpenAI-compatible 端点验证了注册、`<id>/<模型>` 改写、
  `api_base` 逐请求透传和 env 凭据读取；当时未取得 OpenCode Go 的独立密钥，因此本记录
  不代表 OpenCode Go 上游已经通过真实 smoke。
- 本轮 smoke 暴露并修复了两个统一入口问题：生成 Profile 不再强制绑定 embedding profile
  （RAG / Source Index 关闭时即可启动完整初译）；`kind=env` 凭据现在按 `name` 读取环境变量，
  preflight / probe 不再误报不可用。
- 离线自动化测试、假适配器测试和真实 Provider 结果是三类不同证据；未执行的组合保持
  「未执行」，不能用离线结果替代。

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
  issue #348 或发布说明；不要在记录中粘贴密钥、请求正文或完整本机路径。
- 未取得真实凭据的组合保持「未执行」状态，不能用离线测试结果冒充真实 smoke。

## 执行策略差异（成本 / 延迟 / 可靠性 / 能力）

| 维度 | 同步（sync） | Gemini Batch（gemini_batch） |
|---|---|---|
| 计费 | 按即时调用计费，无 Batch 折扣 | 可能享受 Batch 折扣，按提交的批量请求计费 |
| 延迟 | 请求级即时返回，受单请求 timeout 限制 | 云端排队与异步处理，完成时间取决于任务量与配额 |
| 可靠性 / 恢复 | 耐久运行持久化 request/attempt，可 status/resume/cancel/derive | manifest + job state + download/check 恢复 |
| 能力前提 | profile 需声明 `sync_generation`；RAG/embedding 可即时参与 | profile 需为 Gemini adapter 且声明 `remote_batch` |
| 典型用途 | 小批量、补译、需要即时反馈或分阶段路由 | 大规模初译、可等待排队、需要 Batch 折扣 |

Sync / Batch 是执行策略，没有“主流/备选”之分；同一 ModelProfile 可以同时支持两者，切换不复制模型或凭据。
