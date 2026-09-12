# 开放 Issues 审计（2026-09-12）

> **状态：时间点快照，不是长期事实来源。**
> 本文记录 `main@d164c2b` 与 2026-09-12 GitHub 状态的交叉审计结果。
> Issue 的最新状态以 GitHub 正文、当前 checkout、现行文档和 CLI `--help` 为准。
> 上一份快照：[开放 Issues 审计（2026-08-08）](open_issues_audit_2026-08-08.md)。

文档地图：[历史文档归档](README.md) · [规划与设计草案](../plans/README.md) · [项目文档](../README.md)

## 范围与方法

本轮覆盖**当时全部 11 个开放 issues**；审计时仓库没有开放 PR。判断同时核对：

- 当前 `main@d164c2b`（PR #473 合并点）的代码、测试与 `--help`；
- issue 正文、评论、checkbox、关闭状态，以及子 issue / PR / commit 元数据；
- 现行文档与 `docs/plans/` 设计稿的对应关系；
- 对 #474 另用两个真实进程在合成目录复现了 stale 抢占交错（互斥失效与 lost update）。

证据按强度区分：**代码 / 测试** > **文档记录** > **issue 自述**。私有项目门禁实测、
真实 Provider smoke、生产 workspace 采纳状态和 CI smoke 有效性无法仅由公开仓库独立验证，
本文按“作者记录 / 外部证据”标注，不当作代码级事实。

难度按一名熟悉仓库的维护者估算，包含实现、测试、文档和审阅：

| 等级 | 参考投入 |
| --- | --- |
| XS | 不超过 1 天 |
| S | 1–2 天 |
| M | 3–7 天 |
| L | 1–3 周 |
| XL | 多 PR，通常超过 3 周 |

## 总结

- **上轮 21 项已收口 17 项**：#202、#292、#296、#297、#298、#305、#307、#308、#309、
  #313、#314、#315、#316、#318、#320、#321、#322 均已关闭；上轮名单只剩
  #15、#147、#265、#272 仍开放。
- **本轮新增 7 项**：#344、#424、#427、#431、#457、#470、#474。
- **Epic 记录再次失真**：#265 的 P5 已由 #399 / PR #409 于 2026-09-04 交付并关闭，
  但正文 checkbox 与顶部状态栏仍写“P5–P6 尚未完成”；#344 的子 issue #348 已关闭，
  8 条 Epic 验收全部未勾选且状态未更新；#272 仍写“Blocked by #265 P5/P6”。
- **最高优先级正确性缺陷**：#474 的 read-then-unlink 竞态经独立复现成立，可造成互斥失效
  与 lost update；受影响调用方除 issue 所列 `config_store` / `sync_run_store` 外，
  还有 `model_usage_ledger.py:722`。
- **最大价值主线**：#265 P6（= #424）尚未开工。底层 adapter / snapshot / reuse API 已具备，
  但 coverage digest 尚未接入 Project Analysis 发布、Final Review completion、doctor 和
  翻译入口，GUI 也没有 snapshot / diff / 复用候选页面。
- **可立即推进的行政项**：#457 可依自身第 5 条以“维持现状 + 行为保持证据”关闭并另开数据
  follow-up；#272 应改为仅被 #265 P6 阻塞；#470 的“无法取得样本”证据缺口分支已满足。
- **外部阻塞**：#470 缺授权真实大型项目只读样本；#344 / #457 的 Gemini Sync / Batch
  真实 smoke 缺支持地区与凭据；#272 缺可再分发 fixture。
- **未开工**：#424、#427、#431、#15 均无实现 PR；#431 的直连 OpenAI-compatible 后端
  尚不存在，全部生成调用仍经 LiteLLM。

## 全量状态矩阵

| Issue | 审计结论 | 难度 | 下一步 |
| --- | --- | --- | --- |
| [#474 锁 stale 回收竞态](https://github.com/hu-border-collie/renpy-translation-lab/issues/474) | 竞态真实，已独立复现互斥失效与 lost update；issue 漏列 `model_usage_ledger.py:722` | S（方案 A）/ M（方案 B） | 先补确定性交错回归测试；维护者在 A/B 间拍板；同步三处调用方语义、测试与文档 |
| [#470 真实项目 coverage 归因](https://github.com/hu-border-collie/renpy-translation-lab/issues/470) | 只读脚本、合成归因与脱敏均已就绪；唯一实质缺口是授权真实样本；fallback 证据缺口已记录 | S（样本就绪后） | 保持 open/help wanted，补一条证据缺口评论；可顺手让脚本采集 Ren'Py 版本 |
| [#457 默认 ModelProfile 决策](https://github.com/hu-border-collie/renpy-translation-lab/issues/457) | 代码默认仍为 `legacy-batch + gemini_batch` 且旧行为有测试证据；Gemini smoke 与 A/B 未做；GUI 空白创建默认 sync 存在隐性不一致 | XS–S（维持现状关闭）/ L（改默认） | 以“维持现状 + 证据”关闭并开数据 follow-up；不勾选未完成项；处理 GUI 创建默认 |
| [#431 直连协议后端](https://github.com/hu-border-collie/renpy-translation-lab/issues/431) | 无直连后端，生成全部经 LiteLLM；ModelProfile/TaskRoute 基础可复用但有 adapter 枚举、extra_headers、params、final_review 四个接线缺口；协调段已过时 | 首期 M；完整 L–XL | 重写协调段（Parent #344）；先做直连 OpenAI-compatible 纵向切片，不删 LiteLLM、不改默认 |
| [#427 普通译文审校索引](https://github.com/hu-border-collie/renpy-translation-lab/issues/427) | #320/#321/#322/#354/#362/#363 全部已关闭；review index、人工决定生命周期、`needs_recheck`、普通条目 GUI 均不存在 | L（建议 2–3 PR） | 先写 index / decision / 失效规则设计合同，再落核心 + CLI，最后 GUI |
| [#424 #265 P6 产品化](https://github.com/hu-border-collie/renpy-translation-lab/issues/424) | P6 未开工；GUI/doctor/下游门禁缺失；coverage CLI 仅有只读副产物；Tyrano adapter 只在库层，生产 CLI/GUI 无调用 | L（含 Tyrano 产品化则 XL） | 先接 coverage 门禁（PA 发布 / Final Review / doctor / 翻译入口），再做 CLI/GUI；钉死 Tyrano 是否产品化 |
| [#344 同步与多 Provider Epic](https://github.com/hu-border-collie/renpy-translation-lab/issues/344) | 不能关闭：8 条验收仅 1/5/6 基本完成；④有 keyword/revision 路由旁路，⑦状态不显示 frozen provider/model，⑧真实 smoke 仅 2/4 | Epic XL；剩余代码/文档 S–M | 修路由旁路与 final_review 取 route、暴露 frozen provider/model、清文档残留；地区 smoke 后再收口 |
| [#272 第三 Adapter 研究](https://github.com/hu-border-collie/renpy-translation-lab/issues/272) | 矩阵文档已更新为 Naninovel A+ / RPG Maker B+ 双轨；P5 已关闭，当前只被 P6 阻塞；正文、矩阵日期与“span 不能跨行”表述过时 | M（研究收口） | 更新为仅 P6 阻塞；刷新矩阵日期与跨行 writeback 表述；在允许范围内做 P0/P1 准备 |
| [#265 引擎适配 Epic](https://github.com/hu-border-collie/renpy-translation-lab/issues/265) | P0–P5 均有代码/测试/PR 证据；P6 由 #424 承接；正文 P5 checkbox 与状态栏过时 | L（剩余 P6） | 勾选 P5、更新状态栏、指向 #424；P6 完成前不关闭 Epic |
| [#147 Story Graph 试点](https://github.com/hu-border-collie/renpy-translation-lab/issues/147) | 代码与 A/B 工具齐备（`compare-variants` 可切 story_memory），默认关闭；缺真实试点；生产采纳无法由仓库验证 | S（试点本身） | 核对真实 workspace 后做最小图谱 + 1–2 文件 A/B；先不要开发维护工具 |
| [#15 通用文本翻译入口](https://github.com/hu-border-collie/renpy-translation-lab/issues/15) | 完全无通用文本入口；manifest mode 仅 4 种；adapter 与声明式写回接缝已成熟；正文背景早于 #265 | XL（完整首批）/ L（单格式先行） | 先出设计合同（`Refs #15`），按 4–5 个阶段拆分，全部完成后再 `Closes #15` |

## 重点结论

### A. 正确性缺陷

#### #474 `exclusive_file_lock` stale 回收竞态

- 竞态位于 `atomic_io.py:173` 的最终 `_read_lock_owner` 校验与 `:176` 的 `os.unlink` 之间；
  docstring（`:125-128`）与代码注释（`:161-164`）已自认该窗口无法消除。默认
  `stale_after=300.0`、`preempt_dead_owner=True`（`:108-116`）。
- 本次审计用两个真实进程复现：A 停在最终 owner 重读后，B 完成抢占并进入临界区，
  A 恢复后 unlink 掉 B 的新锁也进入临界区；进一步用读-改-写共享计数复现 lost update。
- 调用方三处：`config_store.py:91`（timeout 1s）、`sync_run_store.py:471`（timeout 30s）、
  `model_usage_ledger.py:722`（默认抢占路径，issue 未列出）。latest 服务已在
  `atomic_io.py:244-249`、`:267-272` 硬编码 `stale_after=-1` / `preempt_dead_owner=False`
  规避，并有 `tests/test_latest_manifest_lock.py` 保护。
- 测试缺口：generic `exclusive_file_lock` 没有“两个抢占者交错”用例；
  `tests/test_sync_run_store.py` 没有 `.start.lock` 的 stale 回收测试。
- 附带发现：`write_latest_manifest_locked` / `compare_and_swap_latest_manifest_locked`
  的公开 `stale_after` 参数（`atomic_io.py:234`、`:259`）在函数体内从未使用，
  `gemini_translate_batch.py:2313-2314/2326-2327` 仍传 300，属误导性悬空参数。
- 方案 A（默认不自动抢占）代码量 XS，但会把 `config_store` / `sync_run_store` 的遗弃锁
  变成永久失败，需要人工清理入口、文档和测试重写；方案 B（POSIX `flock` / Windows
  `LockFileEx`）约 M，且锁文件必须常驻（现有测试断言释放后锁文件消失，需改写），
  Windows CI 需要额外验证。

### B. 引擎、coverage 与产品化

#### #265 Epic：P0–P5 已交付，P6 由 #424 承接

- P0–P4：PR #286 / #290 / #291 / #331 / #358 已合并，对应
  `engine_adapters/contracts.py`、`renpy.py`、`writeback.py`、`versioning.py`、`reuse.py`，
  相关测试 6 / 31 / 21 / 21 / 16 项通过。
- P5：#399 于 2026-09-04 关闭，PR #400（调研与 fixture）、#402（只读 adapter）、
  #407（catalog 写回）、#409（收口）已在 main；`engine_adapters/tyrano.py`
  `ADAPTER_VERSION=0.3.0`，43 + 17 项 Tyrano 测试通过。
- 正文失真：P5 checkbox 仍为 `[ ]`，顶部状态栏仍写“P5–P6 尚未完成”；
  2026-09-04 评论只有悬空引用 `@.codex-issue-comment-265.md`。
- P1–P4 的真实私有 Ren'Py 项目门禁实测只有文档记录（`docs/engine_adapter.md:227-267`、
  `docs/project_notes.md:44-45`），仓库内没有可复现样本，按“作者声明”看待。

#### #424：P6 未开工，缺口集中在门禁与展示层

- 底层 API 已具备：`versioning.py`、`reuse.py`、`contracts.EngineCapabilities`；
  coverage review 的 provenance / policy / freshness 合同在 `coverage.py:500-566`、
  `:765-890` 已实现。
- 缺失 1：GUI 没有任何 snapshot / reconciliation / reuse / coverage 页面；
  只有 `gui_qt/diagnostics_context.py:300-370` 的命令模板。
- 缺失 2：coverage digest 未接入 `project_analysis.publish_project_brief`、
  `final_review.evaluate_readiness`、doctor 与翻译入口；`pending=0` 仍会直接报告
  “无待译”，无法区分“确实译完”和“解析器没识别出条目”。
- 缺失 3：doctor 不 import `engine_adapters`，`docs/doctor_recommendations.md` 无
  engine / coverage 内容。
- 缺失 4：`TyranoAdapter` 在生产代码中没有调用者，snapshot 导出硬编码 `RenPyAdapter`；
  CLI/GUI 没有引擎选择；Tyrano 真实运行时语言切换 / fallback 未验证。
- 范围歧义：“产品化接入”是否要求 Tyrano 进入 CLI/GUI 未写明；若要求，工作量由 L 升为 XL。
  建议在 #424 中钉死，避免 P6 验收被两种口径解释。

#### #470：只缺授权样本

- `scripts/coverage_block_attribution.py` 已就绪（只读扫描，默认脱敏，输出确定性有测试），
  合成 fixture 归因与 #460–#464、#471 的修复均已落在 main。
- 验收第 1 条的 fallback 分支（无法取得样本时记录证据缺口并保持 open/blocked）已在
  `docs/plans/renpy_coverage_block_attribution.md` 满足；验收 2–6 必须拿到样本才能完成。
- 次要缺口：脚本不自动采集游戏使用的 Ren'Py 版本，两次独立运行需人工执行。

#### #272：只被 P6 阻塞，文档与正文已脱节

- 矩阵文档已由 PR #374 建立、PR #423 更新为六引擎 12 维度、证据等级、
  Naninovel A+ 首选 + RPG Maker 叙事模式 B+ 双轨。
- 正文仍写 `Blocked by: #265 P5 / P6`，但 P5 已于 2026-09-04 关闭；唯一评论未反映
  P5 收口与 PR #423 结论。
- 矩阵自身两处过时：核对日期仍写 2026-08-18；“span 不能跨行”与 #471 已交付的
  `multiline_text_span_replace` 冲突（仅空源片段插入仍不支持）。
- 在 #265 关闭前不得创建第三 Adapter 实现 issue；P0/P1 的资料复核与 fixture 准备
  可以在允许范围内先做。

### C. 模型主路径与协议架构

#### #344：Epic 不能关闭

- 验收 ①（Sync / Batch 等价 prompt / schema / context）已在离线合同级完成：
  共用 `translation_plan.build_translation_plan`，`tests/test_translation_plan.py:1102`
  断言两侧 `prompt_fingerprint` 相同；但没有真实端到端对比。
- 验收 ②（中断只重跑缺失 chunk）只有假 Provider 子进程测试；真实计费中断 smoke 脚本
  `scripts/run_durable_sync_provider_smoke.py` 尚未执行。
- 验收 ③（三类 Provider 可作初译主模型）真实 smoke 2/3：LiteLLM 内置与自定义
  OpenAI-compatible 已通过；Gemini Sync / Batch 因 `User location is not supported`
  未执行（`docs/provider_smoke_matrix.md:77-79`）。CI `provider-contract-smoke` 最近一次
  运行日志为 `passed=0 skipped=6`（secrets 为空），其 success 不能当作真实 smoke 证据。
- 验收 ④（同一 resolver 覆盖五阶段、无旁路）存在真实旁路：`build-keywords` /
  `build-revisions` 打包入口没有 model 参数，`effective_model` 回退 `BATCH_MODEL`
  （`gemini_translate_batch.py:7009`、`:5400`；manifest 写入 `:7087`、`:5477`），
  `runtime_settings_view` 只投影 translation / project_analysis / final_review
  （`model_routing_reader.py:392-405`），keyword / revision route 不生效；final_review 走
  `FINAL_REVIEW_MODEL or BATCH_MODEL`（`:6656`），且 legacy runtime 硬限
  final_review = `gemini_batch`，非 Gemini Provider 不能承担最终审校。
- 验收 ⑦：无静默跨模型 fallback，但 `sync-status` 快照不含 frozen provider / model，
  失败只显示错误分类；durable 注释要求一次 attempt = 一次 Provider 调用，而 LiteLLM
  后端在限流时仍会遍历同一 Provider 的多把 Key，litellm 分支未传
  `allow_credential_rotation=False`。
- 文档残留：`docs/sync_workflow.md:5,7`、`docs/setup.md:164`、`README.md:38` 仍把
  Batch 描述为主路径、Sync 描述为小范围/补译，与 #344 / #348 的产品方向冲突。
- 补充：#341 已关闭，但正文 9 条验收未勾选，评论仍有 fresh published PA、
  非 Gemini embedding smoke、扩大 A/B 三项外部验收未归属；#344 收口前需要明确承接方。
- 正文失真：`[ ] #348` 过期；8 条 Epic 验收全部未勾选；最后评论是悬空引用
  `@.codex-issue-comment-344.md`；关系表仍列已关闭的 #202 与已合并的 #343。

#### #457：可“维持现状”关闭，但要诚实记录未完成项

- 5 条待办 0/5：Gemini 地区 smoke、候选默认 A/B、默认值决策、代码变更、维持现状结论
  均未完成。
- 代码默认仍为迁移默认 `legacy-batch + gemini_batch`，无 `model_routing` 的旧配置原样
  走 legacy；旧行为保持有迁移签名比较与回归测试证据，`CHANGELOG.md` 也明确“默认不变”。
- 隐性不一致：GUI「创建 Model Routing 配置」从空白创建时默认 `gemini-main + sync`
  （`gui_qt/settings/profiles_page.py:727-754`），与迁移默认 batch 不一致，是未决策
  状态下的隐性默认入口。
- issue 第 5 条本身提供了“若决定维持现状：记录结论与证据后关闭”的出口。推荐以
  “无数据不改默认，维持 legacy-batch + gemini_batch”关闭，另开一条在 Gemini 地区 /
  A/B 数据可用时再决策的 follow-up；不要勾选前两项冒充已完成。

#### #431：方向有效，但首期范围需要收窄

- 直连 OpenAI-compatible 后端不存在：无 `import openai` 生产代码，全部生成经
  `LiteLLMSyncBackend`（`litellm_sync_backend.py:218/249`、`:405/442`）。
  Embedding 还有两处重复 `litellm.embedding`；目录 / 安装管理分散在
  `litellm_provider_config.py`、GUI worker 与锁文件。
- 可复用基础：`ModelProfile` 已含 adapter / provider / base_url / extra_headers /
  capability_overrides / params；`TaskRoute` 覆盖五阶段；`SyncModelBackend` Protocol
  与 `build_sync_backend` 工厂是新增 adapter 的自然接缝。
- 四个接线缺口：adapter 枚举三处硬编码 `gemini/litellm`；`extra_headers` 字段存在但
  reader 不读取、GUI 无入口；generation profile `params` 被 legacy entrypoint 明确拒绝；
  final_review 不能由非 Gemini 执行。
- 协调关系已过时：#348（09-11 关闭）、#202（09-10 关闭）不再是协调对象；应改为
  Parent #344、复用已完成 schema，并声明不改 #457 默认值。
- 首期建议只做“直连 OpenAI-compatible Sync 生成 + 最小配置/UI 接线 + 真实 smoke”，
  估 M；完整 issue（Anthropic / Responses、最终审校解绑、迁移矩阵、LiteLLM 删除决策）
  为 L–XL。

### D. 工作流增强与低优先级

#### #427：依赖已齐，合同与 GUI 均未实现

- 已交付：#320 只读润色语料导出、#321 提案导入与安全预览、#322 订正页导出/导入/筛选、
  #354 复用候选与人工决定、#362 quality-ack、#363 统一 finding schema、#348 统一身份。
- 但 corpus 只覆盖 revision scanner 识别的条目，不含 glossary / 结构 finding /
  provenance / 人工状态；`revision_selection` 是“一次性确认”，状态仅
  valid / no_op / invalid / stale / conflict；GUI 只有按钮与只读候选表。
- 全仓库没有 `needs_recheck` 实现；review index、决定持久化与失效规则、CLI JSON、
  普通条目编辑草稿均缺失。issue 正文无虚假完成声明，主要是依赖状态需要更新。

#### #15：没有入口，但复用接缝比上次审计时成熟

- 当前无 novel / text 命令，manifest mode 仅 translation / keyword_extraction /
  revision / final_review；`build` / `check` / `apply` 与 `collect_pending_file_jobs`
  硬编码 `RenPyAdapter`，因此“直接复用现有流程”不成立，需要 adapter 注入。
- 可复用：engine-neutral `EngineAdapter` 协议、声明式 `text_span_replace` /
  `multiline_text_span_replace` / `json_catalog_set`、共享 TranslationPlan、glossary /
  RAG / Story Memory 注入接缝、`build-retry` 失败修复合同。
- 正文应改写为“基于 EngineAdapter + 声明式写回新增 Plain Text adapter / mode”，
  而不是“新增一套独立脚本”；与 #265 P6 需协调但非硬依赖。

#### #147：代码与 A/B 工具齐备，缺真实试点

- `story_memory.py`、batch / sync / repair 注入、manifest summary、seed 导出、
  doctor glossary 冲突检查均已存在；默认关闭，示例配置为 false。
- `compare-variants` 可直接切换 story_memory 开 / 关（`translation_ab_experiment.py:207-209`），
  #139 已不再是阻塞。
- 生产采纳状态（“所有生产项目未启用”）被 gitignore 挡住，无法由仓库验证；
  2026-08-07 评论已自行更正。试点本身约 S：人工 2–4 小时建最小图 + 1–2 个文件 A/B，
  验收 manifest `story_memory_summary` 非零后记录继续投入或搁置。

## 关键依赖

```text
#474 ────────────────> 独立安全修复（无前置依赖）

#265 P6 = #424 ─┬────> #272 P2（第三 Adapter 实现单）
                └────> #427 首版 engine 证据（非硬阻塞）

#424 内部顺序：coverage 门禁接线（PA 发布 / Final Review / doctor / 翻译入口）
              → CLI coverage / review 命令
              → GUI snapshot / diff / 复用候选页面
              → 配置 / 文档
              → 决定 Tyrano 是否产品化

#344 剩余 ───────────> #457（默认值决策）
#344 真实 smoke ←───── Gemini 支持地区与计费凭据
#431 ←──────────────── 复用 #348 / #202 已完成表面；首期与 #457 默认值解耦
#431 ←──────────────── 与 #344 的 final_review 路由改造有交集

#470 ←──────────────── 授权真实大型 Ren'Py 只读样本

#15 ──────────────────> 设计合同先行；与 #265 P6 协调但非硬依赖
#147 ─────────────────> 独立低优先级试点（compare-variants 已可用）
```

需要避免的新“双事实源”：

- #424 的 coverage digest / review status 与 doctor / Project Analysis / Final Review
  各自的 pending 判断；
- #427 的 review index / decision 与 #354 reuse decision、#362 quality-ack 的语义边界；
- #431 的直连 adapter 配置与现有 `custom_litellm_providers` 的迁移等价性。

## 推荐执行顺序

1. **#474**：先补确定性交错回归测试，再由维护者选方案 A / B；同步三处调用方语义、
   测试与人工清理文档。
2. **修正 Epic 记录**：#265 勾选 P5 并更新状态栏；#344 勾选 #348、标注 8 条验收真实状态；
   #272 改为仅被 #265 P6 阻塞。
3. **#424 coverage 门禁接线**：先接 Project Analysis 发布、Final Review completion、
   doctor 与翻译入口的 `pending=0` 判定；这是 #427、#272 等下游消费 coverage 证据的前置。
4. **#344 代码 / 文档收口**：修 keyword / revision 路由旁路与 final_review 取 route；
   在 `sync-status` 暴露 frozen provider / model；清理 Sync 仍被描述为小任务的残留文案。
5. **#457 维持现状关闭** + 数据收集 follow-up；**#431 重写协调段**后按首期 M 切片排期。
6. **#424 展示层**：CLI coverage / review 命令 → GUI snapshot / 版本 diff / 复用候选页面 →
   配置 / 文档；同时决定 Tyrano 是否进入产品 CLI / GUI。
7. **#427 设计合同 → 核心 + CLI → GUI**；**#15 设计合同 → 单格式阶段**。
8. **#147 试点并行**；**#272 在 #265 P6 完成后刷新矩阵并收口**；
   **#470 等授权样本**，若长期不可得则保持 open / help wanted 并记录证据缺口。

## Issue 记录维护清单

按上一份快照的维护规则，以下记录应更正（本轮只审计，不代改 GitHub）：

| Issue | 需要更正的内容 |
| --- | --- |
| #265 | P5 checkbox 改 `[x]`；顶部状态改为“P0–P5 已交付，P6 由 #424 承接”；处理悬空评论 `@.codex-issue-comment-265.md` |
| #344 | 勾选 #348；8 条验收按本审计标注真实状态；处理悬空评论 `@.codex-issue-comment-344.md`；关系表更新（#202 已关闭、#343 已合并、#341 外部验收归属） |
| #272 | `Blocked by` 改为仅 #265 P6；同步 PR #423 的 Naninovel A+ / RPG Maker B+ 双轨结论；刷新矩阵日期与跨行 writeback 表述 |
| #470 | 补一条“证据缺口已按 #426 格式记录、等待授权样本”的评论，并提及 #471 已实现 |
| #424 | 明确“产品化接入”是否包含 Tyrano CLI / GUI，避免验收口径歧义 |
| #457 | 记录默认值保持证据、Gemini smoke 与 A/B 未完成；说明 GUI 空白创建默认 sync 的差异 |
| #431 | 重写 #348 / #202 协调段；修正“params 已可承载”“最终审校可统一路由”等与现状不符的表述 |
| #474 | 补充第三个调用方 `model_usage_ledger.py:722`；说明 latest 的悬空 `stale_after` 参数 |
| #341 / #346 / #347 | 已关闭但正文验收 checkbox 全未勾选，建议各补一条收口评论说明实际交付，不改写历史结论 |

## Issue 维护规则

- 正文中的事实或 checkbox 错误时，应修改正文，并用一条日期评论记录依据；
- 仅有阶段性判断时只留评论，不把审计全文复制到每个 issue；
- PR 已覆盖的 issue 不再并行实施，等待 PR 合并关闭；
- 真实 Provider smoke、私有项目实测与生产 workspace 事实必须标注证据等级，
  不得用离线 / mocked 结果冒充；
- 后续复核应生成新的日期快照，不回写本文伪装成当时事实。
