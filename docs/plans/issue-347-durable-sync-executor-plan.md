# #347 设计 P0：耐久同步执行器、崩溃恢复与统一安全链路

状态：**设计 P0 已完成，生产实现尚未开始** · 基线 `main@ed07a99`（2026-08-23）·
必须等待 #346 P3/P4 的 Sync `TranslationPlan` 消费、统一 result 与 freshness 门禁落定后再接生产路径。

关联 issue：[#347](https://github.com/hu-border-collie/renpy-translation-lab/issues/347) ·
上游：[#346](https://github.com/hu-border-collie/renpy-translation-lab/issues/346) ·
并行边界：[#341](https://github.com/hu-border-collie/renpy-translation-lab/issues/341) ·
下游：[#348](https://github.com/hu-border-collie/renpy-translation-lab/issues/348)

## 0. 决策摘要

本设计冻结以下方向，后续 PR 不应另起一套临时执行合同：

1. **SQLite 是执行状态的唯一事实源**。每个 run 使用一个本地 SQLite 数据库，开启
   `WAL`、`foreign_keys=ON`、`synchronous=FULL`；一个 run 同时只允许一个调度写者，
   `status` 可并发只读。JSON/JSONL manifest、plan、result 和摘要均为可校验、可重建的物化产物。
2. **一次 attempt 的响应、规范化结果、usage outbox、状态转换和审计事件在同一数据库事务提交**。
   汇总从明细投影，不是完成事实。由此保证“模型结果已经耐久提交、但汇总/`results.jsonl`
   尚未更新”时，resume 不会再次调用模型。
3. **不伪造远端 exactly-once**。若进程死在 Provider 已经受理请求、但本地尚未提交响应的窗口，
   且 Provider 没有幂等键或结果查询能力，该 attempt 进入 `outcome_unknown`；普通 resume 不自动重调，
   以免重复调用或计费。用户只能先对账，或显式派生新 run 并接受可能重复计费。
4. **`TranslationPlan` 与 `TranslationRequest` 是只读输入**。执行器不重建 prompt、上下文、
   chunk 或 schema；动态定点补偿/拆分只能通过 #346 拥有的纯派生 request builder 产生，并记录 lineage。
   当前基线尚无该 helper，#346 必须在 #347 P2 前先落一个纯核心切片；scheduler 不得复制 ID/payload 规则。
5. **resume 只继续同一冻结合同**。plan、source snapshot、resolved profile、关键配置或持久化请求
   fingerprint 不兼容时拒绝原地 resume；原 run 保持可查看，后续操作是派生新 run，不覆盖历史。
6. **usage 以 attempt 为去重单位**。attempt 事务先写 usage outbox；再幂等投递到现有
   [`model_usage_ledger.py`](../../model_usage_ledger.py)。崩溃后重复投递不会重复累计。
7. **执行器没有项目写回权限**。它只生成统一 result 产物；durable Sync 必须经过
   `check -> bound preview/diff -> apply`。Gemini Batch translation 保持现行 `check -> apply`，
   两者共享 `writeback_gate`、source snapshot、adapter 和 `--force` 不可绕过谓词，而不是强制共享一条 CLI 序列。
8. **#347 冻结服务和机器合同，#348 决定最终产品信息架构**。本 issue 提供
   `start/resume/status/cancel/derive` 服务与兼容 CLI 命令；统一翻译页、配置迁移、最终命令别名和文案归 #348。

## 1. 范围、非目标与当前事实

### 1.1 本设计负责

- Run / Request / Attempt 的耐久状态机、合法转换和审计事件；
- 稳定 ID、调度 lease、有界并发、背压、分类重试、取消和迟到结果；
- 崩溃恢复、幂等 resume、动态补偿/拆分 lineage、attempt/time/cost 上限；
- attempt 级结果与 usage 的原子提交、usage outbox 和统一 result 导出；
- `start/resume/status/cancel/derive` 服务层及版本化 CLI JSON 合同；
- 与统一 result/check 门禁、durable Sync preview/apply 的接口和 fault-injection 验收矩阵。

### 1.2 本设计不负责

- 不实现远程 Batch API、跨机器 worker、自动跨 Provider/model fallback 或通用工作流引擎；
- 不在执行器中组装 prompt、检索上下文、解析项目配置或写 `.rpy`；
- 不决定统一 GUI 布局、配置迁移和新用户默认值；
- 不把当前同步路径临时改造成“半耐久”路径。生产接线必须等待 #346 P3/P4；
- 不承诺 Provider 不支持幂等或查询时的远端 exactly-once。

### 1.3 `main@ed07a99` 核对结果

| 区域 | 当前事实 | #347 缺口 |
|---|---|---|
| [`translation_plan.py`](../../translation_plan.py) | #346 P1 已有 schema v1、16 hex `plan_id`/`request_id`、稳定 chunk、source identity、profile/config 脱敏快照、prompt/request/plan fingerprint。`plan_id` 不含 `run_id` 和检索结果；检索内容进入 request 的 `prompt_fingerprint`，并经 request summary 进入 `plan_fingerprint` | 还没有执行状态、attempt、checkpoint、resume 或耐久 result |
| [`model_profile.py`](../../model_profile.py) | `ModelProfile`、`ModelCapabilities`、`TaskRoute`、`ModelRoutingPlan` 可序列化；routing plan 明确要求一次 run 内冻结 | 当前普通 Sync 仍会在错误恢复中轮换模型；没有 attempt 级实际目标与 freshness 记录 |
| [`translator_runtime.py`](../../translator_runtime.py) | 普通同步初译仍在内存中按 `40/12000` 切块、递归重试/二分；处理完文件后才构造 preview；进程退出即丢失尚未汇总的成功结果 | 无稳定 request checkpoint、scheduler lease、cancel 状态或 resume |
| [`sync_model_backend.py`](../../sync_model_backend.py) | 已有 provider-neutral 请求/结果边界、有限 timeout 和基础错误分类；`rate_limit`、`timeout` 等有恢复决策 | 分类粒度不足以覆盖 quota、policy、local persistence 等；恢复不是耐久状态机 |
| [`model_usage_ledger.py`](../../model_usage_ledger.py) | 项目级 JSON ledger 使用文件锁和原子替换，并按 response/payload/call fingerprint 去重；普通 Sync 先缓存在内存，`finally` 才批量 flush | 崩溃会丢 usage buffer；execution result 与 ledger 不能原子提交 |
| [`sync_translation_preview.py`](../../sync_translation_preview.py) | preview manifest 绑定项目、源/候选快照、diff、质量 finding、adapter writeback plan；apply 前全量验证，并用多文件事务 journal 写回 | preview 生成仍与内存执行耦合，尚未消费统一耐久 result；没有独立的最新 `check` 绑定 |
| [`gemini_translate_batch.py`](../../gemini_translate_batch.py) | Batch 已有 result fingerprint、最近一次 `writeback_gate=allow`、apply 时二次 source/adapter revalidation；同步结果行已有 `response`、`provider_response_attempts`、`normalized_response` 语义 | 本地同步结果仍整批在内存生成后一次性写 `results.jsonl`，中断不可恢复 |
| [`atomic_io.py`](../../atomic_io.py) | 单文件使用 fsync + replace；项目多文件写回有 prepared/committed journal 和回滚恢复 | 适合最终 artifact/writeback，不适合作为并发 request 状态数据库 |

## 2. 名词和不可破坏的不变量

- **Plan**：#346 生成的不可变语义合同；一份 plan 可被多个 run 执行。
- **Run**：一次有独立生命周期、预算、取消意图和审计历史的本地执行。
- **Root request**：plan 中原始 `TranslationRequest`。
- **Derived request**：因缺项补偿或解释得通的二分产生，必须指向 root/parent lineage。
- **Attempt**：一次可能到达 Provider、可能计费的物理调用。逻辑重试永远新增 attempt，不覆盖旧 attempt。
- **Accepted item result**：同时通过统一响应合同和本地 adapter/机械验证、可进入 check 的条目结果。
- **Authoritative receipt**：同一事务内落盘的 attempt 响应、规范化诊断、accepted items 和 usage outbox。

必须用数据库约束和测试守住以下不变量：

1. 首次模型调用前，run、完整 plan payload/引用、所有 root request 及其 fingerprint 已耐久提交。
2. `(run_id, request_id)`、`attempt_id`、`usage_event_id` 全局唯一；终态记录不可原地改写。
3. 每个状态变化与对应 append-only `events` 行在同一事务提交。
4. 每个 request 同时最多一个非终态 attempt；每个 run 同时最多一个有效 scheduler lease。
5. `succeeded` request 必须能追溯到一个已提交成功 receipt；汇总字段不能单独制造成功。
6. `superseded` request 的 children 必须与父状态在同一事务插入，不能出现只有父或只有子。
7. resume 只调度 `pending`、已到 `next_eligible_at` 的 `retryable_failed`，以及从未 dispatch 的 `prepared` attempt。
8. `succeeded`、`terminal_failed`、`cancelled`、`outcome_unknown` 和 `superseded` request 不会被普通 resume 重调。
9. cancel 一旦提交，scheduler 不再创建新 attempt；迟到响应只进入审计，不改变 request/run 的取消终态。
10. usage 先进入同事务 outbox，再投递外部 ledger；重复投递只能得到 duplicate，不能增加 calls/tokens/cost。
11. result、check、preview、apply 的 artifact hash 和 source identity 逐层绑定；任何下游重建都不能修改执行明细。
12. manifest、日志、事件和 JSON 输出不含凭据值、敏感 header 或完整 prompt；完整 prompt 只存在受保护的 plan/request artifact 和本地数据库中。
13. 任一 run 进入 `completed`、`completed_with_errors`、`failed` 或 `cancelled` 时，所有 active leaf 都必须是终态；不得遗留 `pending`、`in_flight` 或 `retryable_failed`，终态 run 的模型调度永远为零。

## 3. 稳定 ID、Plan 引用与 freshness

### 3.1 ID 规则

| ID | 规则 | 说明 |
|---|---|---|
| `plan_id` / `plan_fingerprint` | 直接复用 #346，不在 #347 重算 | `plan_id` 是构建身份；resume/check 以更完整的 `plan_fingerprint` 和逐 request fingerprint 为准 |
| `run_id` | 无 token 时生成 `sync-run-v1-<YYYYMMDDThhmmss.ffffffZ>-<uuid4hex>`，其中时间是无冒号 UTC（Python `%Y%m%dT%H%M%S.%fZ`）；有 token 时由 project identity + token digest 派生稳定 `sync-run-v1-token-<hash>` | Windows/POSIX 路径安全，写入后永不改变；相同非空 token 定位同一 run，输入 digest 不同则返回 conflict |
| root `request_id` | 直接复用 `TranslationRequest.request_id`，数据库主键作用域为 run | 同一 plan 跨 run 保持相同，便于安全复用判断 |
| 定点 request | `<parent_id>--M-<sha256(canonical(sorted missing ids))[:12]>` | 相同父 request 和缺项集合得到相同 ID；不得包含已 accepted ID |
| split request | 复用 #346 D7：`<parent_id>--L` / `--R`，递归追加 `L/R` | children 保持原顺序、并集等于父 expected IDs、交集为空 |
| `attempt_id` | `sha256(run_id, request_id, attempt_ordinal)[:24]`，同时保存 ordinal | 重启后不会因重新枚举产生新身份 |
| `usage_event_id` | `usage:<attempt_id>`；投递 ledger 时作为显式 `dedupe_key` 的输入 | 一个实际调用至多累计一次；Provider response id 仍保留作审计 |
| `event_seq` | 数据库自增整数 | 只表达单 run 的提交顺序，不用墙钟排序证明因果 |

16 hex 的既有 plan/request fingerprint 是 #346 v1 合同；#347 不擅自扩位。run/attempt 的随机或哈希部分使用更长身份，避免生命周期记录碰撞。

上述 `--M-` / `--L` / `--R` 只冻结 ID 结果，不授权 #347 自行实现 builder。#346 的纯 helper 必须同时生成完整 derived `TranslationRequest`、fingerprint 和 lineage-safe ID；#347 P2 的 fake scheduler 与后续生产 scheduler 都调用同一个 helper。

### 3.2 Run 冻结快照

`runs` 必须引用并校验：

- 完整 `TranslationPlan` canonical JSON、`schema_version`、`plan_id`、`plan_fingerprint`；
- 每个 root request 的 `request_id`、`prompt_fingerprint`、`request_fingerprint`、expected IDs 和 payload digest；
- `source_identity`：engine、adapter version、project identity、整体 source snapshot、逐文件 digest；
- resolved `ModelProfile`、`TaskRoute`、`ModelCapabilities` 和 `ModelRoutingPlan.config_origins` 的脱敏快照/digest；
- 影响语义或请求的 config fingerprint，以及独立的 executor policy snapshot；
- result/check schema version、错误分类版本和重试策略版本。

凭据只保存 `CredentialRef` 和不可逆 key identity；凭据值永不 fingerprint。相同 ref 下更新/轮换密钥属于兼容操作，但每个 attempt 必须记录实际使用的不可逆 credential identity。v1 在 run start 时从 resolved profile 选定并锁住实际 model；现有 `ModelProfile.models` 候选池不能在错误后被当作静默 fallback。未来若支持显式轮换策略，它必须进入冻结 policy/fingerprint，并逐 attempt 记录目标，仍不得越出同一 resolved profile 或切换 Provider。

另存 `resume_compatibility_fingerprint`，只覆盖会改变请求、执行目标或恢复边界的显式字段：plan/config fingerprint、resolved adapter/provider/model/base URL、非敏感请求 header/params、route/capabilities、generation config、credential ref、错误/重试/预算 policy 和已启用 context asset identity。`config_origins` 的整文件 fingerprint 用于提示“配置文件动过”，但不能仅因无关 GUI/文案字段变化就阻断 resume；必须重算上述兼容 fingerprint 后再决定。凭据值变化不进入该 fingerprint。

### 3.3 Freshness 判定

| 检查 | start | resume | status | derive |
|---|---|---|---|---|
| DB schema / integrity、plan/request artifact hash | 必须通过 | 必须通过 | 只读报告；损坏时标红 | 源 run 可读时才允许复用 |
| live source snapshot / adapter / project identity | dispatch 前再验 | 必须与冻结快照相等 | 报告 fresh/stale | 以当前 source 重建新 plan |
| `plan_fingerprint` 与逐 request payload | 必须自洽 | 不得从 live config 重建替换 | 报告 stored integrity | 新 run 保存新 fingerprint |
| resolved profile/route/capabilities/config origins | 冻结并校验能力 | 关键 digest 改变则拒绝原地 resume | 报告差异，不泄露值 | 重新 resolve，生成新 run |
| credential value | 只验证可解析/存在 | 可更新；仍须属于同一 ref/Provider | 只报告可用性 | 走新 profile 快照 |
| executor concurrency | 保存初值 | 允许显式降低；提高需受 profile/provider cap 限制并记事件 | 报告 effective 值 | 可重新选择 |
| attempt/time/cost 上限 | 冻结 | 只允许收紧；扩大必须派生新 run | 报告已用/剩余 | 可重新选择 |

resume 不重新组装 prompt。即使旧 request payload 仍可执行，只要 live source、profile 或关键配置已经变化，也按 #347 要求拒绝原地 resume，避免一半 run 使用旧环境、一半使用新环境。#341 的 retrieval/analysis identity 必须由 #346 纳入 request/plan fingerprint；context provider 变化因此走同一 freshness 规则。

### 3.4 派生新 run

`derive_run(source_run_id, current_plan, policy)` 创建新 `run_id`，保存 `derived_from_run_id`、原因和操作者选择，绝不修改源 run。默认只复用同时满足以下条件的 succeeded 结果：

- 新旧 `request_id`、`prompt_fingerprint`、expected IDs、source unit digest 和响应合同版本完全相等；
- 源结果不是 late/ignored、outcome unknown 或人工强制接受；
- 当前 adapter validation 仍通过。

`outcome_unknown` 永不自动复用；derive 对 unknown IDs 只允许以下三个互斥结果：

1. 默认 `retry_unknown=false` 且未显式排除 unknown：拒绝 derive，返回 `SYNC_RUN_OUTCOME_UNKNOWN`，原 run 和 expected set 不变；
2. 用户显式确认可能重复计费：在新 run 为这些 IDs 创建全新 request/attempt，并记录 `duplicate_billing_risk_acknowledged=true`；
3. 用户显式创建范围化 scope：新 plan/package 的 expected IDs 排除所选 unknown IDs，记录 `excluded_unknown_ids` 和来源 run，并作为新 scope 完整运行 check。

“无法证明等价就重新执行”只适用于候选为 succeeded、但未通过复用等价谓词的结果；不适用于 unknown、late 或 ignored 行。不得默认留下缺口，也不得静默重调 unknown。

## 4. 状态机与合法转换

状态枚举和转换必须集中定义；服务、scheduler、CLI 和 GUI 不得各自推断。

### 4.1 Run 状态机

| 状态 | 含义 | 合法下一状态 |
|---|---|---|
| `planned` | plan/root requests 已提交，尚无 dispatched attempt | `running`、`cancel_requested`、`failed` |
| `running` | 可调度或正等待退避/worker；进程退出不会自动改状态 | `cancel_requested`、`completed`、`completed_with_errors`、`failed` |
| `cancel_requested` | 取消意图已耐久；禁止新 dispatch，等待 in-flight 收口 | `cancelled` |
| `cancelled` | 取消终态；可保留此前 accepted 结果供查看/派生 | 无 |
| `completed` | 所有必需 ID 都有唯一 accepted winner，无 terminal/unknown/cancelled leaf，也无可调度状态 | 无 |
| `completed_with_errors` | scheduler 已静止且至少有可用结果，但仍有 terminal/unknown leaf；其余 active leaf 已被明确终结 | 无 |
| `failed` | run 级不变量/本地耐久性失败，或无任何可用结果且所有 active leaf 已终结 | 无 |

重复 `cancel`、对终态 `resume`、重复 finalization 都是幂等 no-op，不创造自环事件。进程崩溃只使 lease 过期，不直接改变 run 状态；恢复判断由下一次 resume 执行。

### 4.2 Request 状态机

| 状态 | 合法下一状态 | 说明 |
|---|---|---|
| `pending` | `in_flight`、`cancelled` | 尚无已 dispatch attempt |
| `in_flight` | `succeeded`、`retryable_failed`、`terminal_failed`、`superseded`、`cancelled`、`outcome_unknown` | 一个 active attempt；结果和转换同事务 |
| `retryable_failed` | `in_flight`、`superseded`、`terminal_failed`、`cancelled` | 到 `next_eligible_at` 后才能新增 attempt |
| `succeeded` | 无 | 完整 request 或派生 leaf 已通过合同；accepted items 不可覆盖 |
| `terminal_failed` | 无 | 分类/预算决定不再尝试 |
| `superseded` | 无 | 父 request 的有效结果保留，剩余工作由原子插入的 children 承担 |
| `cancelled` | 无 | 取消后迟到结果不改变此状态 |
| `outcome_unknown` | 无 | 可能已远端执行但本地无权威 receipt；普通 resume 不重调 |

部分响应有已通过项时，父 request 不能简单标 `succeeded`。事务应提交 accepted item winners，并将父标为 `superseded`，同时插入只覆盖缺失/无效 ID 的定点 child。Run 完整性按 unit winners 和 active leaf 计算，而不是按父 request 数量猜测。

### 4.3 Attempt 状态机

| 状态 | 合法下一状态 | 说明 |
|---|---|---|
| `prepared` | `dispatched`、`cancelled`、`terminal_failed` | attempt 行和预算 reservation 已提交，尚未声明开始外部调用；run-level guard 可在释放 reservation 后直接终结 |
| `dispatched` | `succeeded`、`retryable_failed`、`terminal_failed`、`cancel_requested`、`outcome_unknown`、`late_succeeded_ignored`、`late_failed_ignored` | 标记必须在网络调用前提交；崩溃恢复保守处理；late 转换只允许 T3/T4 guard 失败时发生 |
| `cancel_requested` | `cancelled`、`late_succeeded_ignored`、`late_failed_ignored`、`outcome_unknown` | 已尽力调用 backend cancel，但不能假设成功 |
| `succeeded` | 无 | receipt、usage outbox、contract diagnostics 已原子提交 |
| `retryable_failed` | 无 | request 决定是否产生下一 attempt；旧 attempt 不复用 |
| `terminal_failed` | 无 | 保存安全分类/原因码，不保存敏感异常全文 |
| `cancelled` | 无 | Provider/本地任务确认未产生可接受结果 |
| `outcome_unknown` | 无 | 无法证明是否执行/计费 |
| `late_succeeded_ignored` / `late_failed_ignored` | 无 | 只作审计和可能的 usage 记账，不参与结果 winner 或 run 成功 |

`prepared -> dispatched` 与真正发送网络字节不能组成跨系统原子事务。崩溃后只要看到 orphaned `dispatched`，就先尝试 Provider reconciliation；没有查询能力时必须进入 `outcome_unknown`，不能假设“没发出”而自动重试。

## 5. 持久化介质与事务设计

### 5.1 方案比较

| 方案 | 优点 | 主要问题 | 结论 |
|---|---|---|---|
| 原子重写单个 manifest JSON | 最简单，现有 helper 可复用 | 每次 O(n) 重写；并发抢锁；request/result/usage 多对象无法一个提交；汇总容易被误当事实源 | 不采用为事实源 |
| append-only JSONL journal + 独立 result shard | 人工易读，尾部损坏可截断 | 多文件之间没有事务；去重、索引、状态重放、split 原子性和并发读写都要自建数据库语义 | 只可作导出，不作 v1 主存储 |
| 每 request 一个 JSON 文件 + 汇总 | 局部提交简单 | 大项目小文件风暴；父子插入、预算、usage outbox、cancel 与 lease 仍需跨文件协调 | 不采用 |
| SQLite | Python stdlib；唯一约束、事务、索引、单写多读、WAL 恢复均成熟 | 仅适合本地受控文件系统；需管理 WAL/checkpoint 和 schema migration | **最终建议** |

### 5.2 Run 目录

```text
<log_dir>/sync_runs/<run_id>/
  state.sqlite3                 # 唯一事实源；运行中可能有 -wal / -shm
  translation_plan.json        # immutable、hash-bound 导出
  requests.jsonl               # root + derived request 的脱敏审计投影，不含 payload/prompt/source/response
  run_manifest.json            # DB 的可重建物化视图
  results.jsonl                # 终态统一 result，按 plan/root 顺序稳定输出
  results.jsonl.sha256
  events.jsonl                 # 可选终态审计导出；事实源仍是 events 表
```

目录必须在所选项目的受控 `log_dir` 内，路径经过 containment 校验；不得放入共享网络盘执行。运行中备份需包含 DB、`-wal`、`-shm`，或先由服务执行 checkpoint；终态执行 `wal_checkpoint(TRUNCATE)` 后再发布 artifacts。

`requests.jsonl` 不是 `requests.payload_json` 的逐字导出，只允许包含 request/root/parent ID、lineage kind/depth、状态、expected item 的计数/摘要、request/prompt fingerprint、attempt 计数与安全诊断码；不得包含完整 payload、prompt、source/context 文本、response、credential ref/header 或原始异常。`state.sqlite3` 和 `translation_plan.json` 属于受保护的本地 plan/request artifact，继承 run 目录的最小权限，不能作为普通诊断附件分发。

### 5.3 最小表与约束

- `schema_meta(version, created_by)`；
- `runs(run_id PK, client_token_digest NULL, status, revision, plan_id, plan_fingerprint,
  source/profile/config/policy digests, derived_from_run_id, budget fields, timestamps)`；
- `plans(run_id PK/FK, canonical_json, payload_sha256)`；
- `requests(run_id, request_id, root_request_id, parent_request_id, lineage_kind,
  lineage_depth, status, expected_ids_json, payload_json, prompt/request fingerprint,
  attempt_count, next_eligible_at, PK(run_id, request_id))`；
- `attempts(attempt_id PK, run_id/request_id FK, ordinal, status, provider/model/profile,
  credential_identity, claim_owner_token, claim_cancel_epoch, dispatch/finish times,
  error category, response/normalized payload,
  contract diagnostics, usage metadata, UNIQUE(run_id, request_id, ordinal))`；
- `item_results(run_id, item_id, winner_attempt_id, translation payload/digest,
  validation diagnostics, PK(run_id, item_id))`；
- `late_receipts(receipt_id PK, run_id, attempt_id, observed_owner_token/cancel_epoch,
  response/error/usage payload, ignored_reason, received_at)`；
- `usage_outbox(usage_event_id PK, attempt_id UNIQUE, record_json, delivered_at, delivery_error)`；
- `events(event_seq INTEGER PK AUTOINCREMENT, run_id, entity_type/id, old/new status,
  event_type, safe_details_json, committed_at)`；
- `leases(run_id PK, owner_token, pid, acquired_at, heartbeat_at, expires_at)`；
- `artifacts(run_id, kind, relative_path, sha256, schema_version, created_at,
  UNIQUE(run_id, kind))`。

所有 payload JSON 使用 canonical serialization；schema migration 只向前，未知较新版本拒绝写入但可给出升级建议。数据库连接启用 `busy_timeout`。只有持有当前 run lease 的 scheduler writer 能变更 request/attempt；不持 lease 的 `cancel` caller 只可用短 `BEGIN IMMEDIATE` 事务写 run 级取消意图/epoch/event，request/attempt closeout 仍由 lease owner 执行。

`client_token` 省略、空白或空字符串时规范化为 SQL `NULL`，语义始终是“创建新 run”。只有非空 token 的 digest 参与稳定 run path/ID；bootstrap 由 `<log_dir>/sync_runs/.start.lock` 串行化，相同 digest 因确定性目录只能打开同一个 run。相同 token 但 plan/policy 输入 digest 不同返回 `SYNC_RUN_CLIENT_TOKEN_CONFLICT`，不能误当旧 run。token 原文不落盘；若后续增加项目级 run registry，唯一性必须是 `WHERE client_token_digest IS NOT NULL` 的 partial index，绝不能把 `""` 当唯一键。

`runs` 行就是权威 run manifest，`plans` 行就是与之同事务提交的 `TranslationPlan` payload/引用；磁盘上的 `run_manifest.json` 和 `translation_plan.json` 只是 hash-bound 物化视图。因而“首次调用前原子写入 manifest 与 plan 引用”由 T0 的单个数据库事务满足，而不是依赖两个 JSON 文件碰巧同时 replace。

### 5.4 事务边界

| 边界 | 同一事务内必须完成 | 崩溃后结果 |
|---|---|---|
| T0 bootstrap | 插入 run、完整 plan、全部 root requests、`run_created` 事件 | 事务前无可执行 run；事务后首次调用所需合同齐全 |
| T1 claim | 校验 lease/cancel/budget，reserve token/cost，插入 `prepared` attempt，将 request 置 `in_flight` | prepared 可安全继续 dispatch 或在 cancel 时取消 |
| T2 dispatch intent | attempt `prepared -> dispatched`、dispatch timestamp/event | orphaned dispatched 必须 reconcile/unknown，不能盲重试 |
| T3 successful receipt | 先校验 lease owner token、claim 时的 cancel epoch 和 attempt 状态；匹配时保存原始/规范化响应、contract diagnostics、accepted winners、usage outbox，并同事务转换 attempt/request/event | 提交后即为权威成功；summary/export 可任意重建；guard 失败则只走 late receipt |
| T4 failed receipt | 使用与 T3 相同的 owner/epoch/status guard；匹配时保存分类、safe error、usage（若有）、下一退避时间或 terminal 决策、释放 reservation | resume 只按已提交决定继续；guard 失败不改 request/run 结果 |
| T5 derive/split | 父 request 置 `superseded`，插入所有 children 与 lineage events | 不会出现半棵 lineage |
| T6a cancel intent | 任意 caller 用短事务校验 run 非终态，首次取消把 run 置 `cancel_requested`、递增一次 epoch 并写一个事件；不改 request/attempt | 与 T1/T3/T4 由 SQLite writer lock 排出唯一提交顺序；意图先提交则后续 claim/receipt guard 必须看见新 epoch |
| T6b cancel closeout | 当前 lease owner 把 pending/retryable request 置 `cancelled`；`prepared` attempt/request 直接置 `cancelled`；只有 `dispatched` attempt 置 `cancel_requested` | 此后禁止创建 attempt；prepared 从未调用 Provider；无第二个 request/attempt writer |
| T7 outbox ack | usage ledger 已幂等写入后标 `delivered_at` | 任一侧崩溃都可重放，ledger 去重 |
| T8 finalize | 由明细验证终态，记录 run 终态和 result artifact generation | artifacts 可在事务外原子重建；run 终态不依赖文件存在 |

不可把网络调用放在长 SQLite 事务里。T2 提交后释放 DB 锁，再调用 Provider；T3/T4 使用短事务接收结果。两者必须在写 winner 前同事务重读 `leases.owner_token`、attempt claim 时保存的 `cancel_epoch`，并确认 attempt 仍为 `dispatched`（正常收口）或 `cancel_requested`（只能 late/cancel 收口）。owner/epoch/status 任一不匹配时，追加 `late_receipts` 与 usage outbox；若 attempt 尚未终态可转 `late_*_ignored`，若已是 `outcome_unknown` 等终态则保持原状态，只追加 ignored audit，绝不写 winner 或改变 request/run 成功。调度器在 T3/T4 本地提交失败时立即停止新 dispatch，并将本地耐久故障作为 run 级错误处理。

## 6. 关键崩溃窗口与恢复语义

| 崩溃点 | 耐久事实 | resume 行为 | 是否可能重复计费 |
|---|---|---|---|
| T0 前或中途 | 无已提交 run | 清理/忽略孤儿目录；start 用相同 `client_token` 可重取已提交 run | 否 |
| T0 后、首个 T1 前 | `planned` + 完整 plan/requests | freshness 通过后转 `running` | 否 |
| T1 后、T2 前 | attempt=`prepared`，尚未声明 dispatch | 同一 attempt 可继续；cancel 可直接取消 | 否 |
| T2 后、实际发送前 | attempt=`dispatched`，远端可能未收到 | 无查询/幂等能力时仍保守标 `outcome_unknown` | 自动路径不重复，但可能留下未执行项 |
| 远端受理后、响应前 | `dispatched` | Provider 可查询则 reconcile；否则 `outcome_unknown` | 自动路径不重复；显式派生重调有风险 |
| **响应已返回进程、T3 未提交** | 本地仍只有 `dispatched` | 同上；本地无法证明收到的内存结果 | 同上，这是无法消除的跨系统窗口 |
| **T3 已提交、run summary 未更新** | 成功 receipt、winner、usage outbox 已在 DB | 重建投影/汇总，绝不再次调用该 request | 否；满足 #347 关键验收语义 |
| T3 后、usage ledger 写入前 | outbox pending | 重放 outbox | 否，dedupe key 固定 |
| ledger 写入后、T7 ack 前 | ledger 已有记录，outbox 仍 pending | 重投得到 duplicate，再 ack | 否 |
| T5 中途 | SQLite 原子提交 | 要么父仍可恢复，要么父和全部 children 都存在 | 否 |
| `results.jsonl` replace 后、artifact ref 前 | DB 已有全部结果 | 重建相同字节、hash 并补 artifact ref | 否 |
| cancel 提交后、响应到达 | cancel epoch/status 更早提交 | 保存 `late_*_ignored`；usage 若可得仍记账；不写 winner | 不新增调用；既有调用可能计费 |
| 磁盘满导致 T3 失败 | 远端可能成功，本地无 receipt | 停止所有新 dispatch；重启后该 attempt 为 unknown | 自动路径不重复 |

如果 Provider 将来支持可靠 idempotency key，应使用 `attempt_id` 作为 key；若支持按 key/response id 查询，reconciliation 可以把 orphaned `dispatched` 收口为 succeeded/failed。能力必须由 `ModelCapabilities` 显式声明，不能按 Provider 名称猜测。

## 7. 幂等 resume、取消、迟到结果与 usage

### 7.1 Resume 算法

1. 只读打开 DB，验证 schema、`quick_check`、plan/request payload hash；
2. `BEGIN IMMEDIATE` 获取/续约单 writer lease；活跃 lease 存在时返回可重试的 `SYNC_RUN_BUSY`，不启动第二个 scheduler；
3. **阶段 A：crash closeout（终态或 stale run 也执行）**：
   - 对过期 owner 遗留的 `dispatched` / `cancel_requested` 只做 Provider reconciliation；可证明的既有结果作为历史 receipt 收口，无查询能力则转 `outcome_unknown`；这是恢复已发生调用，不是新调度；
   - `prepared` 绝不能在此阶段 dispatch；cancel requested 时直接取消，终态 run 若仍有 prepared 则报告不变量损坏；
   - 重放 usage outbox，重建 summary/result/manifest artifact 投影并 checkpoint；
4. closeout 后重新读取 run。终态 resume 返回当前 snapshot；没有修复时 `changed=false`，补投 outbox、补 artifact 或追加 recovery receipt/event 时 `changed=true`；终态只禁止模型调度，不禁止幂等审计修复；
5. 若 run 为 `cancel_requested`，由当前 lease owner 完成 T6b 收口并返回，不做 freshness 后的新调用；
6. **阶段 B：scheduling** 前重新验证 source/profile/config freshness；不兼容时释放 lease并返回 `SYNC_RUN_FRESHNESS_MISMATCH`。阶段 A 已提交的历史 closeout/物化修复保留并在 error details 报告，但不替换 plan、不 dispatch prepared、不产生新模型调用；
7. freshness 通过后，才允许把既有 `prepared` 继续到 T2，并只选择 eligible `pending` / `retryable_failed` request 调度；
8. 达到终态/静止态后，把因 run-level cap 不可再调度的 active leaf 原子终结，再计算 `completed`、`completed_with_errors` 或 `failed`；
9. 原子导出 result/manifest，checkpoint WAL，释放 lease。

同一 resume 被重复调用、进程在任何步骤被杀、或两个进程竞争时，Provider 调用次数都只能由通过 freshness 后已提交的 T2 解释。`status` 默认严格只读，不顺便抢 lease或修状态；需要 closeout 时返回 `next_action=resume`。

### 7.2 取消

- 第一次 `cancel(run_id)` 通过 T6a 提交取消意图、递增一次 `cancel_epoch` 并写一个事件；重复 cancel 返回同一 snapshot、`changed=false`，不再增加 epoch、revision 或事件；scheduler 每次 T1/T2 前检查 epoch；
- T6a 不要求抢占活跃 lease，也不直接改 request/attempt：若已有 scheduler owner，它在 heartbeat/调度边界观察新 epoch 并执行 T6b；若没有活跃 owner，cancel service 可获取 lease 后以 closeout scheduler 身份执行 T6b。未取得 lease 的 caller 可立即返回耐久的 `cancel_requested` snapshot，不伪报已完成取消；
- T6b 中 pending/retryable request 转 `cancelled`；`prepared` attempt 与对应 request 同事务直接转 `cancelled`，不调用 backend，也不会进入 unknown/late；
- `dispatched` attempt 才由 T6b 转 `cancel_requested` 并调用 backend best-effort cancel；对应 request 保持 `in_flight`，直到 attempt 在同一 receipt 事务落到 cancelled、outcome unknown 或 late/ignored 后再转 request `cancelled` / `outcome_unknown`；
- T6a 与 T1/T3/T4 都是短 `BEGIN IMMEDIATE` 事务：T1/T3 先提交时其 claim/receipt 是取消前事实，T6b 随后按已提交状态收口；T6a 先提交时，后续 T1 必须拒绝，旧 T3/T4 因 claim epoch 不匹配只能走 late/ignored。不存在两个事务同时改 request/attempt；
- GUI/CLI 的“关闭/停止本地 worker”与“取消 run”必须是两个动作：关闭只释放/等待 lease，run 仍可 resume；取消是不可逆业务终态；
- 若 worker 已死，cancel 可在 lease 过期后把 orphaned dispatched attempt 标 unknown，并将 run 收口为 `cancelled`；
- cancel 后任何响应都只能进入 `late_succeeded_ignored` / `late_failed_ignored`，不能恢复 request 或把 run 标成功。

### 7.3 Usage 去重和成本事实

- 每个已 dispatch attempt 是独立潜在计费单元；success、provider failure、late response 只要有 usage 都建立 outbox；
- `usage_event_id=usage:<attempt_id>`，调用 `build_usage_record(..., dedupe_key=...)`，不再依赖结果文件路径或内存 run label；
- DB 中的 attempt usage 是执行审计事实，项目 usage ledger 是查询/聚合副本；ledger 不可用不阻止保存模型结果，但会使 run/status 显示 `usage_delivery_pending`；
- `outcome_unknown` 单独计入 `billing_unknown_attempts`，没有 Provider 证据时不伪造 token/cost；
- status 同时报告 known actual、known estimated、unknown records 和 remaining budget，不能把 unknown 当 0。

## 8. 错误分类、退避、拆分和上限

### 8.1 v1 分类表

| 分类 | 默认决策 | 退避/派生规则 |
|---|---|---|
| `authentication` | terminal | 不轮换 Provider/model；credential ref 修复后派生新 run |
| `configuration` / `missing_dependency` | run-level terminal | 首次 dispatch 前尽量 preflight；不发送模型请求 |
| `unsupported_capability` / `unsupported_request` | terminal | 不重试、不拆分；提示更正 profile/request |
| `rate_limit` | retryable | 尊重 `Retry-After`，否则 exponential full jitter；只用冻结 profile 允许的同 Provider credential 集合 |
| `quota_exhausted` | terminal，除非 Provider 给出有限 reset time | 有明确 reset 时持久化 `next_eligible_at`；不得静默换模型/Provider |
| `timeout` / `transport` | retryable | 同一 request 重试；耗尽后 terminal；网络故障不拆分 |
| `provider_server` | retryable | 5xx/服务暂不可用退避；耗尽后 terminal |
| `invalid_structured_response` | derived | 有有效项则只对无效/缺失 ID 定点 child；完全无进展且可解释时二分 |
| `incomplete_ids` | derived | 只补缺失/无效 ID，不重翻 accepted IDs |
| `content_policy` | terminal | 默认不重试/不拆分；只有 backend 明确标记 `isolatable=true` 才可一次二分定位 |
| `local_validation` | policy-dependent | 可模型修复的占位符/结构问题走定点 child；确定性 adapter 拒绝为 terminal |
| `local_persistence` / `local_artifact_write` | run-level stop | 不发新调用；已有 dispatched attempt 按崩溃窗口处理；绝不以模型重试修复本地写失败 |
| `cancelled` | terminal | 不重试；迟到结果 ignored |
| `unknown_provider` | terminal | 不保留当前“未知错误即二分”兜底，避免扩大调用/计费 |

错误对象只保存稳定 category、reason code、HTTP/provider code、retry-after 和安全摘要。原始异常文本只允许进入权限受控的本地 debug log，并先脱敏；不得进入 manifest、result 或 GUI。

### 8.2 退避

`next_eligible_at = failure_time + min(cap, base * 2^(same_request_retry_index-1)) + full_jitter`，
或使用受全局 cap 限制的 Provider `Retry-After`。实际时间一经计算立即持久化；重启不重新抽 jitter，避免反复 resume 缩短等待。调度使用 UTC deadline 作持久化、monotonic clock 作单进程等待。

### 8.3 动态 lineage

- 定点 child 的 expected IDs 必须是父未 accepted ID 的严格子集；
- split 只能发生在 invalid structure、reasoning/output truncation 或显式 isolatable policy 等可解释分类；
- split children 保持父顺序，复用父冻结的 ModelProfile、source/context identity；每个 child 有新的 request/prompt fingerprint；
- 父的 accepted results 保留且不可被 child 覆盖；同一 item 只能有一个 winner，重复/迟到候选进入 conflict audit；
- lineage 终态按 root 汇总，防止每个 child 重新获得一整套预算造成指数爆炸。

### 8.4 必须同时生效的上限

- `max_attempts_per_request`：包含首轮；建议内部默认 3；
- `max_attempts_per_root`：覆盖所有 targeted/split descendants；
- `max_lineage_depth`：建议沿用当前递归深度上限 5，但不得单独作为唯一保护；
- `max_derived_requests_per_root` 和 `max_total_attempts_per_run`；
- 每 attempt 的 Provider timeout 必须有限并来自冻结 request/profile policy；
- `max_elapsed_seconds`：从 first dispatch 计，等待退避也计入；
- `max_estimated_cost` / `max_actual_cost`：T1 原子 reserve，T3/T4 reconcile，防止并发越界；
- `max_unknown_billing_attempts`：限制没有逐 ID 风险确认时可为 unknown lineage 自动创建的 replacement attempt；v1 默认 0，所以普通 resume 永不为 unknown leaf 建后继 attempt，但该计数不是停止无关 request 的 run-level 阈值。显式 derive acknowledgement 只为所选 unknown IDs 写入一次性 allowance，并仍受 root/run attempt 与 cost cap 限制；
- profile/provider concurrency 和全局 `max_in_flight`，默认保守值由 #347 内部常量给出，#348 再决定用户配置表面。

没有可信 pricing 时，设置硬 cost cap 必须 preflight 失败或使用明确标注的保守上界，不能把未知成本当 0。per-request/root cap 只把受影响 leaf 转 `terminal_failed`。run-level attempt/time/cost hard guard 触发后立刻停止新 T1，等待已 dispatched attempt 按正常/unknown/late 规则收口，再把其余 `pending` / `retryable_failed` active leaf，以及尚未 dispatch 的 `prepared` attempt/request，同事务释放 reservation 并转为 `terminal_failed`，使用稳定原因码（如 `run_budget_exhausted.cost`、`run_budget_exhausted.time`）。有部分结果则 run 为 `completed_with_errors`，无可用结果则 `failed`，`next_action=derive`；终态 snapshot 不得显示 stranded pending/retryable/in-flight。扩大预算只能派生新 run，不能让原终态 run 重新调度。

`outcome_unknown` 与上述全局 hard guard 不同：它只终结发生不确定性的 leaf，并禁止该 lineage 的普通 replacement；只要 freshness 和其他 cap 仍允许，resume 必须继续调度无关的 `pending` / `retryable_failed` / `prepared` 工作。所有独立工作收口后，有 accepted 结果的 run 进入 `completed_with_errors`，否则进入 `failed`，`next_action=derive`；用户再按 §3.4 对 unknown IDs 选择风险确认重调或范围化排除。这样既不重复计费，也满足“崩溃后继续未完成且可安全重试 chunk”，且终态没有 stranded leaf。

## 9. 调度、并发和背压

- 一个 run 单 writer scheduler；worker 只执行已由 T1/T2 认领的 attempt，不能自行派生 request 或改状态；
- 调度器按 plan 顺序 + `next_eligible_at` 稳定排序，最多预取 `max_in_flight`，DB 即持久队列，不另建无界内存队列；
- 同时应用 run、Provider/profile 和 credential 级 semaphore/rate limiter；capability/profile 给出的上限优先于用户较大值；
- T1 在同一事务检查 cancel、lease、run/request 状态、attempt/root/run/cost 上限并做 reservation；
- worker 返回后必须先通过 T3/T4 的 owner-token、claim-epoch、attempt-status guard，再释放并发位；guard 失败只追加 late receipt/usage，commit 失败立即打开全局 dispatch circuit breaker；
- cancel intent writer 只改 run/epoch/event；request/attempt 的 T6b 转换与 T1–T5/T8 一样只由当前 lease owner 执行。scheduler 的 heartbeat/下一调度边界必须检查 epoch 并优先 closeout，不能在活跃 lease 外另起一个 cancel writer；
- heartbeat 只证明 scheduler owner 活跃，不证明某个远端调用结果；lease 过期绝不能把 dispatched attempt 变回 pending。生产 worker 在 Provider I/O 期间使用独立 heartbeat 线程；lease TTL 必须大于 heartbeat interval + scheduler grace。若实现不能提供独立 heartbeat，则 TTL 至少为冻结的最大 attempt timeout + cancel/commit grace，避免仍在合法网络等待的 owner 被过早夺租。

## 10. 统一 result/check 门禁与策略特有 preview/apply 接口

```text
TranslationPlan + frozen requests
              |
              v
     durable sync executor ----> state.sqlite3 / attempt audit / usage outbox
              |
              v
   immutable unified results.jsonl + sha256
              |
              v
        check (fresh fingerprint + structural/quality gates)
              | writeback_gate=allow
              +-------------------------------+
              |                               |
              v                               v
 durable Sync bound preview/diff      Gemini Batch translation
              |                       existing direct apply preflight
              v                               |
 durable Sync explicit apply                  v
              +----------> shared source/adapter/--force predicates
```

### 10.1 执行器输出边界

终态导出按 root request/plan 顺序生成统一 result 行。基线合同就是当前 [`gemini_translate_batch.py`](../../gemini_translate_batch.py) 和现行 Sync 文档已经使用的 result-row 字段角色；#347 拥有“DB 明细 -> 该合同”的 durable exporter adapter，#346 P3/P4 只需把 plan/request ID 与 fingerprint 接入生产行，不必另交一份新 result codec 才能解锁 #347 P3。字段必须保留当前兼容语义：

- `run_id`、`plan_id`、root `request_id`、`chunk_id`、request/prompt fingerprint；
- `response`：首轮 Provider 原始响应的安全序列化；
- `provider_response_attempts[]`：attempt id、lineage request、kind、provider/model、finish/error、usage、contract diagnostics；
- `normalized_response`：首轮和派生 request 合并后的权威 envelope；
- `response_semantics`：明确 raw/attempt/final 字段角色；
- `contract_diagnostics`、accepted/unresolved IDs、late/unknown/cancel flags；
- 每行和整个 artifact 的 schema version/hash。

执行器不得复制 `collect_result_actions`、placeholder 校验或 adapter writeback 语义。#347 exporter 只适配/版本化既有 result-row contract；#346 的生产接线负责提供 plan/request fingerprint 字段，两者共同走现有 check 读取边界。

### 10.2 Check 与部分失败

- `completed` 才能候选完整成功；仍必须运行 check，不能依据 run 状态直接 apply；
- `completed_with_errors`、`cancelled` 和含 unknown 的 run 可导出/检查成功部分，但默认完整性 check 必须使 `writeback_gate=deny`；
- 若用户明确只交付成功部分，必须创建一个新的**范围化派生 package/run**，记录 excluded IDs、来源 run 和人工选择，再对新 scope 完整运行 check；不能用 `--force` 或 acknowledgement 修改原 check 的结构结果；
- check fingerprint 绑定 run manifest、result hash、target shape、plan/request fingerprint、source identity、质量规则/策略/glossary digest；results 或 manifest 变化后旧 check 自动 stale。

### 10.3 Durable Sync Preview 与两种 Apply 路径

- durable Sync preview 只接受最近一次与当前 artifacts 完全匹配、`writeback_gate.decision=allow` 的 check；
- durable Sync preview manifest 绑定 run/plan/result/check fingerprint、source snapshot、adapter version/writeback plan、diff/候选文件 hash、质量 finding；
- durable Sync apply 前再次验证所有绑定和 live source，先全量验证再第一次写；项目多文件写回继续复用 `atomic_write_many_lines` journal；
- durable Sync apply 以 writeback operation/preview fingerprint 为幂等键：目标已经等于候选内容时返回 `already_applied`，不重复写进度、RAG 或 usage；目标既非 source 也非 preview 时拒绝；
- Gemini Batch translation 保持现行 `check -> apply`，不因 #347 新增 mandatory preview 命令；其 apply 继续要求 fresh check、`writeback_gate=allow` 和 apply-time source/adapter revalidation；
- 两种路径的共享测试断言 gate/fingerprint/source/adapter/`--force` 谓词，不断言所有策略拥有同一 CLI 步骤；`--force` 不能跳过 stale check/source/structure/quality blocker。

当前 Sync 的 `create_sync_preview` 同时生成 preview 和质量 finding，而 Batch 使用独立 check。#346 P3/P4 先把生产 Sync 请求绑定 plan/request fingerprint；#347 P5 再把 durable result 接入共享 check 谓词和 Sync bound preview，不在 scheduler 内 import GUI、不复制 check/apply 语义，也不改变 Batch translation 的现行命令序列。

## 11. 服务层与 CLI JSON 合同

### 11.1 服务层

建议公共入口位于新的 `sync_run_service.py`，对 GUI/CLI 暴露纯 Python 数据对象：

```python
start(plan_build, *, policy, client_token=None) -> RunSnapshot
resume(run_id, *, policy_overrides=None) -> RunSnapshot
status(run_id) -> RunSnapshot
cancel(run_id, *, reason="user") -> RunSnapshot
derive(
    run_id,
    current_plan_build,
    *,
    reuse_policy,
    retry_unknown=False,
    ack_duplicate_billing_risk=False,
    exclude_unknown=False,
) -> RunSnapshot
```

`client_token=None`（以及入口规范化后的空字符串）总是新 run；非空 token 才启用 start 幂等，且同 token 输入不一致时拒绝。`retry_unknown=true` 必须同时有 `ack_duplicate_billing_risk=true`，并与 `exclude_unknown=true` 互斥；否则按 §3.4 拒绝或创建范围化新 run。`start`/`resume` 默认前台运行到终态、取消或本地 worker 被优雅停止；优雅停止不等同 cancel，run 保持 `running` 且 `next_action=resume`。服务对象不读取 GUI state，不调用 `sys.exit`，错误使用稳定 code/category 和 safe details。

### 11.2 CLI 命令

为符合仓库现有 flat subcommand 与 #296 机器输出合同，#347 首期建议增加：

```text
sync-start [--client-token TOKEN]
sync-resume RUN
sync-status RUN|--latest
sync-cancel RUN
sync-derive RUN [--retry-unknown --ack-duplicate-billing-risk | --exclude-unknown]
```

每个新 `sync-*` 命令支持现行 `--output text|json`、`--strict-exit-codes`、`--non-interactive`、`--fields`、`--compact` 和 `--output-file`。为满足 issue 的 `--json` 验收，只在这些新命令上增加 `--json`：它与 `--output json` 使用同一个 parser destination 和 schema-v1 envelope，因此 `--json --compact` 合法。不得修改 `final-review-status`、`project-analysis-status`、`usage-import`、`usage-report` 等既有命令的 legacy `--json` 语义；#348 可再提供统一 `translate start/resume/...` 别名并保留映射。

`RUN` 选择器不允许隐式 cwd/latest。`--non-interactive` 必须显式给出 `RUN` 或命令明确支持的 `--latest`。`sync-status --latest` 只枚举目录名匹配 `sync-run-v1-*`、含可读 `state.sqlite3`、且 DB 内 `runs.run_id` 与目录一致的 durable run；现有 `YYYYMMDDThhmmss.uuuuuuZ` Sync preview 目录没有 state DB，必须忽略。latest 以 DB `created_at` 排序；没有候选或无法唯一判定时返回 `SYNC_RUN_NOT_FOUND`。

机器模式 stdout 只输出一个 [`cli_contract.py`](../../cli_contract.py) schema-v1 envelope；进度写 stderr 或受控日志。顶层键固定为 `schema_version`、`command`、`ok`、`status`、`result`、`artifacts`、`warnings`、`error`；下面是 `result` 的稳定主体：

```json
{
  "run_id": "sync-run-v1-...",
  "run_status": "running",
  "revision": 17,
  "changed": true,
  "plan": {
    "plan_id": "...",
    "plan_fingerprint": "..."
  },
  "freshness": {
    "resume_allowed": true,
    "source": "fresh",
    "profile": "fresh",
    "config": "fresh",
    "reasons": []
  },
  "progress": {
    "requests": {
      "total": 12,
      "active_leaf_total": 10,
      "pending": 3,
      "in_flight": 1,
      "succeeded": 5,
      "retryable_failed": 1,
      "terminal_failed": 0,
      "superseded": 2,
      "outcome_unknown": 0,
      "cancelled": 0
    },
    "items": {"expected": 420, "accepted": 260, "unresolved": 160},
    "attempts": {"total": 7, "late_ignored": 0},
    "usage": {
      "known_calls": 6,
      "billing_unknown_attempts": 0,
      "total_tokens": 12345,
      "estimated_cost": 0.12,
      "actual_cost": null,
      "currency": "USD",
      "delivery_pending": 0
    }
  },
  "cancellation": {"requested": false},
  "next_action": "resume"
}
```

`requests.total` 统计 root + derived 的所有 request 行，必须严格等于八个状态 bucket 之和；`active_leaf_total = total - superseded`，用于区分仍代表交付范围的 leaf 与只保留 lineage/accepted 历史的父 request。终态 run 的 `pending`、`in_flight`、`retryable_failed` 必须全为 0。

Artifacts 使用 envelope 顶层 `artifacts`：`run_dir`、`state_db`、`plan`、`requests`、`manifest`、`results`、`result_sha256`。不返回 prompt、response 正文、credential ref 细节或原始异常。

稳定错误 code 至少包括：`SYNC_RUN_NOT_FOUND`、`SYNC_RUN_BUSY`、`SYNC_RUN_FRESHNESS_MISMATCH`、`SYNC_RUN_CLIENT_TOKEN_CONFLICT`、`SYNC_RUN_STORAGE_ERROR`、`SYNC_RUN_SCHEMA_UNSUPPORTED`、`SYNC_RUN_BUDGET_EXHAUSTED`、`SYNC_RUN_OUTCOME_UNKNOWN`。重复 cancel 和终态 resume 返回成功 snapshot，不应变成错误。

`ok` 表示命令合同是否成功执行，不等于 run 完整成功；成功 snapshot 的 envelope `status` 必须等于 `result.run_status`。`--strict-exit-codes` 对新命令冻结如下：

| 结果 | `ok` / `error.retryable` | strict exit |
|---|---|---|
| `planned` / `running` / `cancel_requested` / `completed` | `true` / 不适用 | `EXIT_OK=0` |
| `completed_with_errors` | `true` / 不适用 | `EXIT_NEEDS_ACTION=3` |
| `cancelled` / `failed` | `true` / 不适用 | `EXIT_BLOCKED=4` |
| `SYNC_RUN_NOT_FOUND`、`SYNC_RUN_FRESHNESS_MISMATCH`、`SYNC_RUN_CLIENT_TOKEN_CONFLICT`、`SYNC_RUN_SCHEMA_UNSUPPORTED`、invalid-state refusal | `false` / `false` | `EXIT_INVALID_STATE=5` |
| start/pre-dispatch `SYNC_RUN_BUDGET_EXHAUSTED`、`SYNC_RUN_STORAGE_ERROR` | `false` / `false` | `EXIT_BLOCKED=4` |
| `SYNC_RUN_BUSY` | `false` / `true` | `EXIT_RETRYABLE=6` |

`SYNC_RUN_OUTCOME_UNKNOWN` 只在 `sync-derive` 发现 unknown 且用户既未确认重调风险、也未显式排除时作为 `ok=false` / `EXIT_INVALID_STATE=5`；`sync-status` 或终态 resume 仅把 unknown 数量作为成功 snapshot 字段，并通过 `completed_with_errors -> EXIT_NEEDS_ACTION` 表达。执行期间命中预算后按 §8.4 返回成功的终态 snapshot；`SYNC_RUN_BUDGET_EXHAUSTED` 仅用于 start/pre-dispatch 无可执行预算等命令级拒绝。

## 12. Fault-injection 测试矩阵

测试使用可计数 fake Provider、阻塞 barrier、可注入 store failpoint 和真正的子进程强杀；只抛 Python 异常不足以验证 fsync/WAL 崩溃恢复。

| 编号 | 注入点/场景 | 期望断言 |
|---|---|---|
| F01 | T0 提交前强杀 | 不存在可 resume run；Provider 调用 0 |
| F02 | T0 后、首 dispatch 前强杀 | resume 执行全部 request；每个 root 一次 |
| F03 | T1 prepared 后强杀 | freshness 通过才复用该 attempt id 并调用一次；stale 时保持 prepared、调用 0 |
| F04 | T2 dispatch intent 后、fake send 前强杀 | request 进入 unknown 或由幂等 Provider reconcile；普通 resume 不为该 lineage 新增 attempt，但继续其他独立 pending/retryable/prepared leaf |
| F05 | Provider 已计数成功、T3 前强杀 | 无查询能力时 unknown；显式派生前显示 duplicate billing 风险 |
| F06 | T3 commit 后、summary 更新前强杀 | resume 不调用该 request；accepted/usage/summary 正确重建 |
| F07 | T3 后、ledger flush 前强杀 | outbox 重放一次；ledger inserted=1 |
| F08 | ledger insert 后、outbox ack 前强杀 | 重放得到 duplicate；calls/tokens/cost 不增加 |
| F09 | rate limit/timeout/5xx | 按持久化 backoff 重试；重启不重抽 jitter；达到 cap 停止 |
| F10 | auth/config/unsupported | 0 次重试、0 个 split child，错误分类稳定 |
| F11 | invalid JSON/截断且整批无进展 | 只在允许分类下创建 L/R；父和 children 原子出现 |
| F12 | 部分有效 + missing IDs | accepted ID 不重调；只创建稳定 `--M-...` child |
| F13 | split T5 任一语句前后强杀 | DB 中不存在半棵 lineage；root attempt cap 不被 children 放大 |
| F14 | prepared + dispatched 并存及 cancel 与 T1/T3 交错 | T6a 只写 intent；lease owner 的 T6b 将 prepared 直接 cancelled、Provider cancel 0，dispatched 才 cancel_requested；intent 先于 T1/T3 时 claim 被拒绝/receipt ignored，intent 后提交时按先前事实收口；cancel 后无新 dispatch |
| F15 | 重复 cancel/status/resume | 第二次 cancel `changed=false` 且 epoch/revision/event 不增长；终态 resume 可修 outbox/artifact 但 Provider 调用数不变 |
| F16 | 两个进程同时 resume | 一个取得 lease；另一个 `SYNC_RUN_BUSY` / retryable / exit 6；无重复 dispatch |
| F17 | worker I/O 时 lease 过期/owner 被替换 | dispatched 不回 pending；旧 worker T3/T4 guard 失败，只写 late receipt/usage，不写 winner |
| F18 | response receipt 时磁盘满/permission error | circuit breaker 停止新调用；相关 attempt 不自动重调 |
| F19 | result JSONL replace 后、artifact DB 更新前强杀 | 终态 resume `changed=true` 并重导出相同字节/hash；check 只消费完整 artifact |
| F20 | source snapshot、adapter、profile、config 分别变化 | closeout/outbox 可修；prepared 不 dispatch；scheduling 以 `SYNC_RUN_FRESHNESS_MISMATCH` 拒绝；derive 新 run 成功 |
| F21 | cost reservation 临界并发/attempt-time-cost run hard cap 触发 | reservation 总和不越 cap；未知 pricing preflight 失败；剩余 pending/retryable/prepared 原子 terminal_failed 并释放 reservation，终态无 stranded leaf |
| F22 | completed_with_errors 直接 check/apply | 原 scope gate deny；范围化派生后重新 check 才可能 allow |
| F23 | 同一 run 多次 export/check/preview/apply | usage 不重复、result hash 不变、已应用文件不重复写 |
| F24 | cancel 后 late response 带 usage | winner 不变、run 仍 cancelled；usage 只记一次并标 late |
| F25 | DB/manifest/plan/request/result 任一被篡改 | integrity/fingerprint 阻断，不调用模型、不写项目 |
| F26 | start token 省略/空/重复/冲突 | 省略或空值各建新 run；同非空 token + 同输入返回同 run；不同输入返回 client-token conflict |
| F27 | derive 遇到 unknown | 默认拒绝；ack 后重调并记风险；exclude 后生成新 scoped expected set，三者互斥 |
| F28 | CLI selector/progress/strict exit | `--latest` 忽略 legacy preview；request bucket 可加总；状态和 error code 严格命中 §11.2 exit 表 |
| F29 | Durable Sync 与 Gemini Batch 安全路径 | Sync 要求 check + bound preview + apply；Batch 保持 check + apply；两者共享 gate/source/adapter/force 谓词 |
| F30 | run path 与 requests 审计导出 | 无 token `run_id` 匹配 `sync-run-v1-YYYYMMDDThhmmss.ffffffZ-<uuid4hex>` 且 Windows/POSIX 可创建；`requests.jsonl` 不含 payload/prompt/source/response/credential/raw error canary |

Provider 验收分三层：

1. CI：fake Gemini、fake LiteLLM built-in、fake custom OpenAI-compatible adapter 跑完整矩阵；
2. 可选本地 smoke：三类真实 Provider 各跑最小 plan 和可控中断，不在 CI 使用凭据；
3. 与 #346 P4 golden fixture 集成：Sync 与 Gemini Batch 规范化请求一致；durable Sync 与 Batch 产物通过同一 result/check gate 测试，另分别验证 Sync bound preview/apply 与 Batch direct apply 序列。

## 13. 分阶段 PR 计划与预计文件

P0 即本文，只改计划/索引；#346 P2 正在另一工作树实施，本阶段不碰共享生产文件。

| 阶段 | 内容 | 依赖/出口 | 预计文件 |
|---|---|---|---|
| P0 | 本设计、状态机、存储/崩溃语义、测试矩阵 | 文档链接与 diff 通过 | `docs/plans/issue-347-durable-sync-executor-plan.md`、plans 索引 |
| P1 | 纯状态/SQLite store/schema migration/event/lease/投影 | 无网络、无生产入口；转换表和事务单测全过 | 新增 `sync_run_store.py`、`sync_run_contracts.py`、`tests/test_sync_run_store.py`、fixtures |
| P2 | fake-provider scheduler、预算 reservation、退避、取消、lineage、fault harness | 先消费 #346-owned pure derived-request helper；不接普通 Sync；F01–F21/F26–F27 通过 | 新增 `durable_sync_executor.py`、`sync_retry_policy.py`、`tests/test_durable_sync_executor.py`、subprocess fixture；不复制 `translation_plan.py` ID 逻辑 |
| P3 | attempt usage outbox 与 durable result/exporter adapter | 直接适配既有 Batch/Sync result-row 字段角色，不等待新的 #346 result codec；usage 重放、result golden 与 F30 脱敏断言通过 | 新增 `sync_result_export.py`，修改 `model_usage_ledger.py` 及对应测试 |
| P4 | 生产 `TranslationPlan`/backend 接线和服务层 | **必须基于 #346 P3/P4 合并点**；不再走 legacy prompt/chunk builder | 新增 `sync_run_service.py`；消费 #346 API，按最终接缝修改 `translator_runtime.py`、`sync_model_backend.py` 及测试，不回改 plan builder |
| P5 | CLI JSON、共享 check gate、durable Sync preview/apply 集成 | F22–F25/F28–F29、partial gate、重复 apply 验收通过；不得给 Batch translation 增加 mandatory preview | 修改 `gemini_translate_batch.py`、`cli_contract.py`（仅必要扩展）、`sync_translation_preview.py`、CLI/preview/gate 测试、现行 workflow 文档，并同步 GUI“诊断与运行日志”的命令参考/user copy 测试；不改页面布局 |
| P6 | 三 Provider 中断 smoke 与 #348 handoff | #347 后端验收；统一页面和配置仍留 #348 | provider smoke 脚本、脱敏诊断与 handoff 文档；不在本阶段实现统一 GUI |

每阶段先跑针对性测试；P3 起跑完整 CLI，P4/P5 跑 CLI + GUI + quality gates。不得把依赖升级、#341 context provider 或 #348 Settings 重构混入 #347 PR。

## 14. 依赖与文件所有权边界

### #346 P3/P4：共享语义合同与安全接缝

- **拥有**：`translation_plan.py`、root/derived request builder、Sync 消费 plan、system instruction backend 接缝、plan/source/profile fingerprint，以及 Sync/Batch 请求黄金等价；
- #347 把持久化请求当不可变 payload，不能在 scheduler 中重建 prompt/schema/chunk；
- #346 须在 #347 P2 前先提供纯 derived-request helper；P3/P4 合并前，#347 最多开发 store/fake scheduler/result adapter，不得接 `translator_runtime.py` 或临时复制 result/check 逻辑。

### #341：上下文与 Embedding

- **拥有**：retrieval/analysis provider、Source Index/Published PA 注入、Embedding backend/store identity 和相关配置/诊断；
- #341 通过 #346 provider seam 改变 request/plan fingerprint；#347 只做 freshness 比较并执行已持久化请求；
- executor 不访问 vector store，不在 resume 时重新检索，也不决定 stale PA 是否可注入。

### #347：耐久执行生命周期

- **拥有**：run DB/schema、state/event/lease、attempt scheduler、retry/cancel/late/unknown、预算、usage outbox、基于既有 result-row contract 的 durable exporter adapter、服务和机器状态合同；
- 对共享文件的修改必须在 #346 P3/P4 后小范围接线，不能接管 prompt、context 或项目写回语义。

### #348：产品化与迁移

- **拥有**：版本化用户配置 schema、旧 `sync.*`/`batch.*` 迁移、统一翻译页、Settings/MainWindow/user copy、最终 CLI 名称/弃用映射、用户默认值和发布文档；
- #348 消费 #347 的 `RunSnapshot` 与服务方法，不直接读 SQLite 表或在 GUI 复制状态机；
- #347 可提供保守内部 policy/CLI flags，但不抢先建立第二套长期 GUI 配置表面。

#347 P5 因新增 CLI 命令而必须按仓库约定同步现有 GUI“诊断与运行日志”的命令参考和相关 user-copy 测试；这只是兼容性维护，不包含 #348 所拥有的统一页面、Settings 或迁移设计。

共享热点 `gemini_translate_batch.py`、`translator_runtime.py`、`translation_plan.py`、`translation_core.py`、`sync_translation_preview.py` 必须按 **#346 P3/P4 -> #347 生产接线 -> #348 产品化** 顺序修改。#341 只经冻结 provider seam 并行，冲突时由 #346 合同优先。

## 15. #347 验收映射

- 部分 chunk 后强杀：F02/F03/F06，resume 只调用未完成且可安全重试项；
- 模型成功但汇总未落盘：以 T3 为耐久成功边界，F06 证明不重调；T3 前的不可判定窗口明确为 unknown；
- timeout/rate limit/invalid JSON/缺失 ID：F09/F11/F12；
- 多 in-flight 取消与迟到响应：F14/F24；
- 多次 resume/status/apply、usage 去重：F07/F08/F15/F23；
- source/profile/config 改变：F20/F25；
- Gemini、LiteLLM 内置、自定义 OpenAI-compatible：三层 Provider 验收；
- 与 Batch 共享安全谓词而不强制同一命令序列：P3/P5 和 F22/F23/F29，执行器自身永无写回权限。

## 16. 后续实现门禁

P0 纯文档至少运行 tracked 本地 Markdown 链接检查和：

```powershell
git diff --check
```

后续代码 PR 按范围运行：

```powershell
python -m unittest tests.test_sync_run_store tests.test_durable_sync_executor -q
python -m unittest tests.test_model_usage_ledger tests.test_sync_translation_preview -q
python -B tests/run_cli_tests.py -q
python -B tests/run_gui_tests.py -q
python scripts/run_quality_gates.py all
git diff --check
```

真实 Provider smoke 必须显式 opt-in、有限请求/timeout/cost，不打印 prompt、响应正文、原始异常或凭据；CI 只运行 fake Provider fault matrix。
