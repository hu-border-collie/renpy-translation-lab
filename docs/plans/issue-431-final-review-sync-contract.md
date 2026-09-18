# #431 S3：最终审校 Sync 执行合同（final_review 解绑 gemini_batch）

> **状态**：S3 实现完成（PR 待合并）；S3.0 合同、S3.1 core/CLI 与 S3.2 GUI 分流均已落地。
> S1/S2 见
> [issue-431-openai-compatible-contract.md](issue-431-openai-compatible-contract.md) 与
> [issue-431-openai-compatible-s2-contract.md](issue-431-openai-compatible-s2-contract.md)。
> **关联**：#431、#344、#341、#348、#486。
> **基线**：`main@1b32bd0`（PR #505 合并点）。

## 1. 范围

S3 让「最终审校」按 ModelProfile 已声明的 stage route strategy 执行：

- `gemini_batch`：保持现有 Batch 流程（build → submit → status → download → ingest），行为不变。
- `sync`：新增 `final-review-run-sync`，用冻结的 ModelRoutingPlan / TaskRoute 与
  `run_sync_request` 逐 unit 调用同步后端，直接写入同一 campaign package 的 units/findings，
  仍然是 report-only，不写 `.rpy`。

同一份 campaign package、prompt、response schema、unit identity、findings/quality 映射与
人工选择/订正交接全部复用现有 `final_review` / `final_review_llm` 合同；Sync 只替换
「请求如何发出、结果如何返回」的传输/调度层。

非目标：

- 不删除 Gemini Batch 路径，不改已有 package 的语义。
- 不移除 `final-review-*` Batch 子命令；`final-review-resume` 仍是 Batch 的重新入队命令。
- 不引入 durable Sync run store / state DB；本切片不实现跨进程 socket 级取消。
- 不做 Anthropic Messages / OpenAI Responses、不做 Provider 自动 fallback。
- 不改变最终审校 prompt schema、finding schema 或 quality gate 阈值。

## 2. 路由与策略

- `final_review` 是普通可配置 stage；`routes.final_review.strategy` 允许 `sync` 与
  `gemini_batch`，profile 必须声明对应 strategy 能力（`require_valid_routing_plan` 检查）。
- v1 `model_routing` 中 final_review **只有显式配置 `routes.final_review.strategy`** 时才采用该
  策略；未显式配置时不会跟随 `defaults.execution_strategy` 静默切换，而是保留/要求 legacy
  `gemini_batch` 入口解析。旧配置迁移仍显式生成 `gemini_batch`，语义不变。
- 旧（无 `model_routing`）配置继续解析为 `gemini_batch`，不自动升级。
- final-review build 对 v1 `model_routing` 直接按配置解析，不经过 legacy entrypoint 的
  `final_review == gemini_batch` 策略检查，因此也不要求 `legacy_entrypoints` 指针；
  Batch 专用入口仍保留该检查。无 `model_routing` 的旧配置继续按 `gemini_batch` 解析。
- `run_sync_request` 明确支持 `openai_compatible` adapter：直接走
  `build_sync_backend`，不再为该分支创建 Gemini client / 要求 Gemini API key；
  与 LiteLLM / Gemini 分支并列，重试仍复用同一同步恢复合同。
  其他 entrypoint（translation/keyword/revision/project_analysis）的既有约束不变。
- 最终审校不支持跨 strategy / 跨 profile fallback；失败必须显示真实 profile、strategy
  与错误分类。

## 3. Build 合同

`final-review-build`：

1. 解析实时最终审校路由（不强制 `GEMINI_BATCH`），得到 `profile / strategy / model`。
2. readiness、context snapshot、unit 构建与现有实现一致。
3. manifest 新增/维护：
   - `execution_strategy`: `sync` | `gemini_batch`；
   - `model_routing`: 冻结计划快照（两种策略都有）；
   - `input_jsonl_path`：Batch 仍为 `requests.jsonl`；Sync 为空字符串；
   - `summary.request_count`：Batch 为 JSONL 行数；Sync 为 `0`；
   - `final_review_settings.execution_strategy`。
4. Batch 策略：继续写 Gemini Batch `requests.jsonl`，输出 `Next: submit → …`。
5. Sync 策略：不写 Batch JSONL，不伪造 `request_count`；输出
   `Next: final-review-run-sync <manifest>`。

`final-review-status` / `-export` / `-create-revisions` 对两种策略保持只读兼容；
status 增加 `execution_strategy` 字段与文本展示。

## 4. Sync 执行合同

新增命令：

```text
final-review-run-sync [TARGET] [--force] [--limit N] [--dry-run] [--fail-fast]
                      [--output json] [--non-interactive]
```

行为：

1. 只接受最终审校 package；从 manifest 读取冻结 `model_routing`，取
   `routes.final_review`；manifest 无快照的旧 package 视为 `gemini_batch`，本命令拒绝并
   提示改用 Batch 流程。
2. 实时重算 shared prompt context 与 digest；用 `plan_units_for_run(force=...)` 计算
   `to_run / to_skip`：
   - done 且 digest 未变：skip；
   - pending / failed / stale / running：重新入队；
   - `--force`：全部重新入队。
   `to_run` unit 携带本次 live `input_digest` / `context_digest`；成功 ingest 时写回同一
   摘要，因此共享上下文变化后重跑一次即可重新收敛到 `no_work`。
3. 对 `to_run` 中每个 unit：
   - 用 `final_review_llm.build_system_instruction` + `build_user_prompt` 生成 prompt；
   - 用冻结 route/profile 经 `run_sync_request` 执行恰好一次逻辑请求
     （`run_sync_request` 内部允许同一 profile/凭据的既有重试合同）；
   - 成功后经 `ingest_result_rows` 解析 findings 并 `persist_campaign_state` 原子落盘，
     再进入下一个 unit；
   - 失败则把 unit 标为 `failed`，错误前缀使用稳定 `sync_<category>` 分类；默认继续下一个
     unit；`authentication` / `missing_dependency` / `unsupported_capability` 视为系统性
     失败，记录该 unit 后中止。
4. `--limit N` 最多执行 N 个 unit；`--dry-run` 只返回 to_run/to_skip 计划；
   `--fail-fast` 首个 unit 失败即中止。实际结果中 `run_count` = 本次真正发起过的 unit 数，
   `deferred_count` = 计划队列中未发起的 unit 数（limit 剩余或 systemic abort 跳过）。
5. 返回 JSON result：`status`（`completed` / `failed` / `no_work` / `dry_run`）与
   `profile_id / provider / model / execution_strategy / package_dir /
   manifest_path / run_count / skip_count / done_delta / failed_delta / finding_count /
   planned_unit_ids / attempted_unit_ids / to_run_unit_ids / campaign_status / dry_run / limit`。
   存在失败 unit 时 `status=failed`，严格模式退出 `4`，自动化不应只看 `ok=true`。
6. 文本模式打印 summary 与 campaign status；进度事件可写 stderr，不污染 `--output json`。

## 5. 状态、持久化与幂等

- unit 状态仍使用 `pending / running / done / failed / stale`；finding 仍由
  `ingest_result_rows` 与 `merge_findings_preserve_selection` 合并，人工 selection 不丢。
- 每个 unit 完成或失败后立即原子持久化 manifest / units / findings / report；进程中断时
  已落盘 unit 不重跑。正在 HTTP 中的 unit 语义为 at-least-once：下次运行重新入队，
  可能重复一次请求与计费。
- 已 done 且 input/context digest 未变的 unit 永不重跑，除非 `--force`。
- Sync 不创建 durable run 目录；取消（Ctrl-C）在 unit 边界生效。socket 级 abort 与
  跨进程恢复留后续切片。
- Batch package 的 `submit/download/ingest` 状态字段保持原义；Sync 不写 `job_name` /
  `job_state` 的云端语义，只记录本地运行结果。

## 6. 生成参数与结构化输出

- prompt：复用 `build_system_instruction()` / `build_user_prompt()`。
- schema：复用 `build_response_json_schema()`。
- generation config：`temperature` / `max_output_tokens` 沿用现有
  `BATCH_TEMPERATURE` / `BATCH_MAX_OUTPUT_TOKENS`；`response_mime_type=application/json`、
  `response_json_schema=<final review schema>`；`structured_output_mode` 由 profile
  capabilities 决定，直接 adapter 不支持所选模式时显式失败，不静默假装 strict。
- Gemini sync profile：沿用 Gemini config 过滤与 thinking level 语义；Batch 专用
  `safety_settings` 在 Sync 路径按现有同步请求合同传参。
- provider / model / usage / finish_reason 与失败分类进入 unit 结果与 usage ledger；
  usage ledger 使用 `task_mode='final_review'`、`stage='final_review'`、operation 绑定
  campaign unit，尽力记录、不因账本失败中断审校。

## 7. CLI / GUI

- CLI：build 输出按策略给出下一步；同步执行使用 `final-review-run-sync`；Batch 的
  `final-review-resume` 仍只服务 Batch，遇到 sync manifest 时返回稳定机器码并提示正确命令。
- GUI 最终审校流程按 manifest `execution_strategy` 分流：
  - Batch：现有 build → submit → status → download → ingest；
  - Sync：build → final-review-run-sync → status/export；
  - resume 时读取 manifest：sync 且未完成执行 run-sync，Batch 沿用现有 resume/submit/status。
- `doctor` / diagnostics 命令参考同步补充新命令，不改变现有 Batch 推荐路径的展示优先级。

## 8. 测试范围

- route/plan：sync final_review 可通过解析与 capability 校验；Batch 专用入口仍拒绝
  非 gemini_batch；旧配置仍解析为 gemini_batch。
- build：两种策略 manifest 字段、request_count、输出下一步与 readiness 行为。
- sync runner：skip/force/limit/dry-run、逐 unit 持久化、findings 合并、unit 失败继续、
  systemic 失败中止、稳定 `sync_<category>` 错误分类、usage 回调。
- CLI：`--output json` envelope、错误机器码、文本模式、对旧 package 的拒绝。
- GUI：workflow 分步、sync resume、命令参数；无 PySide6 时按既有约定跳过。
- 回归：Batch 的 build/resume/ingest 行为与现有测试不变；final-review-create-revisions 不变。

## 9. 机器码与失败边界

- Sync 运行中 unit 失败使用 unit 错误前缀 `sync_authentication` /
  `sync_rate_limit` / `sync_service_unavailable` / `sync_timeout` /
  `sync_invalid_response` / `sync_unsupported_capability` /
  `sync_missing_dependency` / `sync_provider_error`。
- 命令级错误（package 不存在、旧 package 无 routing、manifest 不是 final_review、
  Batch 专用命令收到 sync package）使用 `cli_contract.MachineContractError` 稳定
  code；不把 provider 原始异常正文写入公开输出。
- report-only / autofix=false 不变；任何情况下不得把失败 unit 标成 done。
