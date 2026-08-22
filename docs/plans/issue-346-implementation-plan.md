# #346 实施分步计划：Sync / Batch 共用 TranslationPlan、ContextAssembler 与请求合同

状态：草案（未开工）· 基线 `main@fa69d14`（2026-08-21）· 本文是实施计划，不是已合并的设计
关联 issue：<https://github.com/hu-border-collie/renpy-translation-lab/issues/346>

## 0. 结论

#346 尚无实施代码，建议按五阶段落地。P0–P4 不依赖 #341；P5 与 #341 联动收口。

| 阶段 | 内容 | 是否依赖 #341 |
|---|---|---|
| P0 | 决策冻结与失败测试骨架 | 否 |
| P1 | 纯核心：schema、确定性 chunking、ContextAssembler、统一 prompt、fingerprint | 否 |
| P2 | Gemini Batch 改为消费 TranslationPlan | 否 |
| P3 | Sync 初译改为消费同一 TranslationPlan | 否 |
| P4 | 诊断、黄金等价测试与 source snapshot 安全收口 | 否 |
| P5 | Source Index / Published Project Analysis / Embedding provider 接入 | 是 |

## 1. 基线事实（main@`fa69d14`）

当前存在两套并行的翻译 prompt builder，各自只被一条路径消费：

| 函数 | 消费方 | 形态 |
|---|---|---|
| `translation_core.build_translation_system_instruction`（`translation_core.py:625`）与 `build_translation_user_prompt`（`translation_core.py:645`） | 仅 Batch（`gemini_translate_batch.py:3086/3093`） | system/user 分离 |
| `translation_core.build_sync_translation_prompt`（`translation_core.py:680`） | 仅 Sync（`translator_runtime.py:4413`） | 单 blob；`call_gemini_sdk` 只传 `contents` |

关键漂移：

- 规则措辞、TARGET payload 序列化（compact JSON vs 默认空格）、reference blocks 配置均不同；
- chunking 不同：Batch `60 items / 18000 chars`（`gemini_translate_batch.py:3192`），Sync `40 items / 12000 chars`（`translator_runtime.py:5947` 附近内联循环）；
- 局部上下文不同：Batch 30/10 原始切片无 block 边界；Sync 已有 `build_sync_local_context`（`translator_runtime.py:3278`）的 block 边界与预算诊断；
- 词法术语门控不同：Batch 的 `glossary_hits` 被 `RAG_ENABLED` 门控（`gemini_translate_batch.py:3217` 起的 `build_chunks`），Sync 已解耦（`retrieve_sync_glossary_hits`，`translator_runtime.py:3239`）；
- Sync 无 Source Index 与 Published Project Analysis；Story Memory 预算等数值不同（Batch 1200 / Sync 800）。

已统一的地基（#339/#345 成果，应直接复用）：

- `translation_core.build_response_json_schema`（`translation_core.py:998`）与 `validate_model_response`（`translation_core.py:1410`）已被两条路径共用；
- `model_profile.py` 已提供 `ModelProfile` / `ModelCapabilities` / `ExecutionStrategy` / `TaskRoute` 及 manifest 序列化；
- Sync preview 已有 source snapshot 与 writeback plan 绑定（`sync_translation_preview.py`），Batch apply 已有 source snapshot 校验。

## 2. 总原则

1. **TranslationPlan 只描述“模型看到的语义合同”**；transport 只允许添加 Batch envelope、job id、并发/轮询 metadata 等外层差异。
2. **相同 source snapshot、配置与 ModelProfile 必须产生相同 plan**。计划 ID、chunk key、request ID 全部由内容哈希派生，不用时间戳或随机数参与指纹。
3. **本地 glossary / Macro 不得因 RAG/Embedding 关闭而消失**；是否命中 RAG 只影响检索层。
4. **不新增用户配置表面**（#348 负责产品化配置）。#346 只定义代码内 plan settings 与解析规则；现有 `sync.*` / `batch.*` 键冲突时在 preflight 显式失败，不静默二选一。
5. **旧产物可读**：旧 Batch manifest/requests.jsonl、旧 Sync preview 继续可 check/apply；新字段向后兼容添加。
6. **不绕过 `check -> apply` 与写回门禁**；plan/source snapshot 不匹配必须在模型调用前或写回前失败，`--force` 也不能跳过。
7. **任何 plan/diagnostic/测试夹具不得含 API key、凭据值或敏感 header**。

## 3. P0：决策冻结与 golden fixture 骨架

P0 只产出决策记录和 golden fixture，不改生产路径。目标是把基线评论中的三个漂移决策扩展为可执行决策表；fixture 用于 P4 暴露并锁定 Sync/Batch 漂移，不在 P0 提交会失败的断言。

### 3.1 决策表（P0 必须定稿）

| 编号 | 决策 | 建议默认 | 影响 |
|---|---|---|---|
| D1 | 局部上下文 | 采用 Sync 的 block 边界算法（`build_sync_local_context` 上移为共享 provider）；Batch 原始切片弃用 | Batch 与 Sync 上下文语义一致，Batch 的切片行为会变化 |
| D2 | 词法术语 | 始终注入 `normalize_map` / `preserve_terms` / `non_translatable_exact`；Batch 解除 `RAG_ENABLED` 门控 | Batch 行为变化，正是 #346 要求的方向 |
| D3 | prompt 形态 | 统一为 system/user 分离；Sync backend 扩展 system instruction | `SyncGenerationRequest` 需扩展，`GeminiSyncBackend` / `LiteLLMSyncBackend` 需传递 system instruction |
| D4 | chunking | 二选一：A=`60/18000`（保留 Batch 当前默认，Sync 请求变大）；B=`40/12000`（保留 Sync 当前默认，Batch 请求数增加）。建议先按 A 验证，preflight 用 `ModelCapabilities.context_budget_tokens` 拒绝超限组合 | 任选都会改变一条路径的请求分组；需在 PR 中记录成本/质量影响 |
| D5 | 上下文预算 | `history_char_limit=220`（已一致）、`include_source_text=true`、`story_char_limit` 统一（建议 1200，与 Batch 一致）；各层预算进入 plan 并输出裁剪诊断 | 与 D4 一样需要一次行为变更，golden fixture 冻结 |
| D6 | generation config | `temperature` 统一为 0.2；`max_output_tokens` / `thinking_config` 等先保留策略/Provider 差异，但必须写入 request metadata 与 request fingerprint | 黄金等价只比较 prompt/schema/context，不比较 transport/generation 差异 |
| D7 | Sync 重试拆分 | 初版 plan chunk 固定；Sync 因 invalid response 动态二分时，生成确定性派生 request（如 `<parent_id>--L` / `--R`），并在 diagnostics 记录 lineage | 不破坏稳定 ID；完整 checkpoint/重试仍是 #347 的范围 |

### 3.2 P0 交付物与出口条件

- 本文状态从“草案”改为“已定稿决策”（更新决策表）。
- 新增最小 golden fixture（小规模 `.rpy` + 固定配置 + 固定 ModelProfile），只提交 fixture，不提交断言生产代码相等的测试。
- 出口：D1–D7 在 issue 内确认；fixture 的 source snapshot、glossary、macro、story 文件齐备。

## 4. P1：纯核心（无执行器接线）

新增纯模块 `translation_plan.py`，不 import 任何可选 SDK，不访问网络与凭据。

### 4.1 Schema v1（草案字段）

```text
TranslationPlan
  schema_version=1
  plan_id / run_id
  source_identity:
    engine, adapter_version, project_identity_digest,
    source_snapshot_fingerprint, per-file digests
  config_fingerprint            # 非敏感配置快照
  model_profile_snapshot        # 复用 ModelProfile.to_manifest_dict()，无凭据值
  execution_strategy            # sync | gemini_batch
  chunk_policy                  # D4 决定值
  context_policy                # provider 顺序与各层预算（D5）
  chunks[]
  artifacts                     # 产物目录与状态引用（相对路径）
  plan_fingerprint

PlanChunk
  chunk_id / chunk_index
  file_rel_path / line_numbers
  unit_ids / source_char_count
  context_window_spec

TranslationRequest
  request_id / plan_id / chunk_id
  system_instruction / user_prompt        # 规范化后的语义内容
  response_schema / expected_ids           # 复用 #339 合同
  capability_requirements
  generation_config                        # 策略差异可存在（D6）
  transport_metadata                       # batch key / sync run id 等
  prompt_fingerprint                       # system+user+schema+expected_ids+context，跨策略可比较
  request_fingerprint                      # 含 generation/transport 的完整指纹

ContextLayer / ContextAssembly
  layer / rank / blocks / char_used / char_limit / truncated / diagnostics
```

### 4.2 P1 实现内容

1. **统一 chunking 与稳定 ID**
   - 从 `gemini_translate_batch.iter_translation_chunk_ranges` 抽出共享 `iter_translation_chunks`，按 D4 参数化；
   - `chunk_id = hash(file_rel_path)-NNNNN`（保留 Batch 现格式），`request_id = sha256(canonical(plan_id, chunk_id, expected_ids))[:16]`；
   - 同一输入、配置、策略参数下重复构建得到字节一致的 plan。
2. **ContextAssembler**
   - 定义 `ContextProvider` protocol 与固定层顺序：
     1. 必需层（TARGET 条目 + speaker/file/label/scene 结构信息）；
     2. 局部层（D1 的 block 边界窗口）；
     3. 项目层（Macro + 词法 glossary，D2 始终注入）；
     4. 检索层（history RAG / Source Index / Story Memory，按现有开关与预算）；
     5. 分析层（Published Project Analysis brief，先留 provider stub）；
     6. 预算与裁剪层（确定性排序、去重、逐层截断并记录保留/舍弃原因）。
   - 迁移 `build_sync_local_context` 到共享模块，并让 Batch 使用同一实现。
3. **统一 prompt 模板**
   - 以现有 Batch system/user 为基础，合并 Sync 独有的规则（如 No markdown / No Pinyin 等），去重后冻结为 canonical prompt；
   - user prompt 固定 compact JSON（`separators=(',', ':')`）、`include_source_text=true`（D5）；
   - 旧 `build_sync_translation_prompt` / Batch 包装函数保留为兼容 shim，但 P2/P3 新路径不得再调用旧实现。
4. **fingerprint 工具**
   - `canonical_json()` 稳定序列化（排序键、固定 separators、UTF-8）；
   - `prompt_fingerprint` 只覆盖模型语义合同；`plan_fingerprint` / `request_fingerprint` 覆盖审计全量；
   - 提供 `redact_sensitive()` 测试钩子，夹具和产物扫描不得出现 `api_key` / `Authorization` / key 值。

### 4.3 涉及文件与测试

- 新增 `translation_plan.py`、`tests/test_translation_plan.py`、`tests/fixtures/translation_plan_*`；
- 修改 `translation_core.py`（canonical prompt 与共享 chunking helper）与 `tests/test_translation_core.py`；
- 出口：纯函数单测通过；P2/P3 可同时基于 P1 接口评审，不触发生产路径变化。

## 5. P2：Gemini Batch 消费 TranslationPlan

1. `build_chunks`（`gemini_translate_batch.py:3217`）改为调用 P1 的 plan builder，不再自行决定 chunk/context/prompt；
2. `build_batch_request`（`gemini_translate_batch.py:3156`）只负责把 `TranslationRequest` 包装成 Gemini Batch envelope；
3. manifest v2 增加 `translation_plan` 块与逐 request 的 `request_id` / `prompt_fingerprint` / `request_fingerprint`；`requests.jsonl` 中保留可审计字段；
4. `submit` / `probe` / `split` / `repair` 路径消费 plan request，不重新生成 prompt；`batch_cost_estimate.py` 优先读 plan 字段，旧包回退旧估算；
5. Batch 词法 glossary 解除 `RAG_ENABLED` 门控（D2），并记录 `glossary_hits` diagnostics；
6. 旧 manifest/requests 读取路径保持兼容，新增读取测试。

出口：

- `python -m unittest tests.test_translation_plan tests.test_gemini_translate_batch_cli_contract -q` 通过；
- `python -B tests/run_cli_tests.py -q` 通过；
- 真实小项目 build → submit（dry-run）→ check 流程不变，manifest 新字段可解释。

## 6. P3：Sync 初译消费同一 TranslationPlan

1. 替换 `translator_runtime.py` 主循环里的内联 batching（`5947` 附近）为 P1 的 plan chunks；
2. `build_prompt` / `process_batch` / `call_gemini_sdk` 改为消费 `TranslationRequest`；
3. `SyncGenerationRequest` 增加 `system_instruction`（放 `config`，避免改坏现有 backend 合约）；`GeminiSyncBackend.generate` 传给 google-genai，`LiteLLMSyncBackend` 已支持 `config["system_instruction"]`，补充 golden 测试；
4. Sync 动态二分重试按 D7 生成派生 request 与 lineage 诊断；
5. `sync_translation_preview.create_sync_preview` 在 manifest 增加 `plan_fingerprint` / `request_ids` / 各文件 snapshot 绑定；`prepare_sync_preview_apply` 在写回前校验 source snapshot 与 plan 匹配；
6. `translation_ab_experiment.py` 的 `compare-variants` 若依赖旧 prompt builder，则改为走同一 plan request，避免 A/B 成为第三条 prompt 路径。

出口：

- `python -m unittest tests.test_translation_plan tests.test_sync_model_backend tests.test_sync_translation_preview -q` 通过；
- `python -B tests/run_cli_tests.py -q` 通过；
- Sync 小项目 preview → apply 的 source snapshot 与 plan fingerprint 门禁生效。

## 7. P4：诊断、黄金等价与安全收口

1. **golden tests**：同一 fixture 分别以 `sync` 和 `gemini_batch` 构建 plan，剥离 Batch envelope 后，比较每个 request 的 `prompt_fingerprint` 与规范化 `system/user/schema/expected_ids/context` 字节一致；生成 `plan_diff` 可读报告。
2. **diagnostics 落盘**：
   - Batch manifest、Sync preview manifest 都写 `translation_plan` 摘要与 request 指纹；
   - 裁剪、缺项、provider 缺失等诊断复用现有 report 结构；doctor 只读展示，不新增检查项。
3. **stale guard**：
   - executor 在模型调用前校验 `source_identity` 与当前 adapter snapshot 一致；
   - apply 沿用 Batch/Sync 现有 source snapshot 校验，并把 `plan_fingerprint` 纳入 preview/apply 绑定；`--force` 不得绕过。
4. **文档与 GUI 同步**：
   - `docs/sync_workflow.md`、`docs/batch_workflows.md`、`docs/context_systems.md` 更新 plan 字段与行为变化；
   - GUI 只在现有“诊断与运行日志”/manifest 摘要处展示新字段；无新增配置页。
5. **兼容性收口**：旧 Batch 包与旧 Sync preview 继续可读；新字段缺省时降级为 legacy 路径并明确诊断。

出口（本阶段代表 #346 P0–P4 完成）：

- 黄金等价测试通过；
- 关闭 RAG/Embedding 不丢失本地 glossary/Macro（有专测）；
- 裁剪顺序确定且 diagnostics 可解释（有专测）；
- source snapshot / adapter 不匹配在模型调用或写回前拒绝（有专测）；
- plan/diagnostic/夹具无凭据（有扫描测试）。

## 8. P5：#341 联动

#341 在 P1 冻结 provider 接口后即可并行开发，P4 完成后合并最终集成：

1. #341 实现 `source_index` provider（复用现有 Source Index Store，不复制索引格式）；
2. #341 实现 `published_project_analysis` provider（只注入 fresh/published brief，记录 identity）；
3. #341 实现 provider-neutral Embedding 边界（model/task type/dimension 与 store 一致，否则拒绝混用）；
4. #346 侧补充两条路径对这些 provider 的等价 golden case 与预算/降级诊断；
5. 用 #139 或真实项目 A/B 验证收益后再决定是否默认开启。

#346 的最终验收项中涉及 Source Index / PA 的部分在此阶段关闭。

## 9. 阶段与 issue checkbox 映射

| #346 工作范围 | 阶段 |
|---|---|
| 定义并版本化 plan/request/result schema | P1 |
| 抽取公共 chunking 和稳定 ID 规则 | P1 |
| 把 Batch 已有上下文构建能力迁到共享 ContextAssembler | P1–P2 |
| 按 #338 接入局部前后文、Macro 和相关术语 | P1 |
| 为 #341 预留可插拔 context provider | P1 / P5 |
| 统一 prompt 模板、schema/envelope、完整性校验及缺项诊断；复用 #339 | P1 / P4 |
| Gemini Batch 构建器改为消费 TranslationPlan | P2 |
| 同步初译入口改为消费同一 TranslationPlan | P3 |
| plan 和每个 request 输出规范化 fingerprint/diagnostics | P1 / P4 |
| golden tests：Sync 与 Batch 交给模型前规范化请求相同 | P4 |
| 与 #265 source snapshot/引擎适配标识对齐 | P2–P4 |

## 10. 门禁命令

按阶段增量跑针对性测试，P2 起跑完整门禁：

```bash
python -m unittest tests.test_translation_plan tests.test_translation_core -q
python -m unittest tests.test_gemini_translate_batch_cli_contract tests.test_sync_model_backend tests.test_sync_translation_preview -q
python -B tests/run_cli_tests.py -q
python -B tests/run_gui_tests.py -q      # P3/P4 若 GUI 诊断或 workflow 有变化
python scripts/run_quality_gates.py all
git diff --check
```

## 11. 风险与回滚

| 风险 | 缓解 |
|---|---|
| D4 改变一条路径的 chunk 分组，Batch 请求数或 Sync 单请求大小变化 | P0 记录决策；preflight 用能力预算拒绝超限；真实小项目 smoke 对比 `batch_cost_estimate.py` |
| 统一 prompt 措辞改变既有译文质量 | golden fixture 冻结前后差异；先用 `compare-variants` / #139 做小样本 A/B，不自动改变默认路径 |
| Sync 动态二分与稳定 ID 冲突 | D7 派生 request + lineage 诊断；完整 checkpoint 留给 #347 |
| 旧 manifest/preview 读取回归 | P2/P4 保留 legacy 读取路径并加 fixture 测试 |
| P1 provider 接口与 #341 实现漂移 | #341 分支基于 P1 合入点开发，P5 前做接口一致性测试 |
| 与 #362/#363 等并行质量门禁工作冲突 | 本计划只改翻译计划/请求路径，不混入 writeback gate 与 quality finding 重构 |

## 12. 非目标

- 不实现 Sync 长任务 checkpoint、取消、并发调度（#347）；
- 不实现 GUI 配置或统一产品化配置（#348）；
- 不要求所有 Provider 的 token 估算完全一致，无法精确时使用保守预算并记录来源；
- 不在 #346 内实现 Source Index / PA / Embedding 的真实检索逻辑（#341）；
- 不改变默认模型、默认执行策略，也不自动开启 RAG / Source Index / Story Memory / Project Analysis。
