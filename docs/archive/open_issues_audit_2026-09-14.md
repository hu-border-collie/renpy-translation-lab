# 开放 Issues 审计（2026-09-14）

> **状态：时间点快照，不是长期事实来源。**
> 本文记录 `main@b178923`（PR #489 合并点）与 2026-09-14 GitHub 状态的交叉审计结果。
> Issue 的最新状态以 GitHub 正文、当前 checkout、现行文档和 CLI `--help` 为准。
> 上一份快照：[开放 Issues 审计（2026-09-12）](open_issues_audit_2026-09-12.md)。

文档地图：[历史文档归档](README.md) · [规划与设计草案](../plans/README.md) · [项目文档](../README.md)

## 范围与方法

本轮覆盖**当前全部 9 个开放 issues**；审计时仓库没有开放 PR。判断同时核对：

- 当前 `main@b178923` 的代码、测试与 `--help`；
- issue 正文、评论、checkbox、关闭状态，以及子 issue / PR / commit 元数据；
- 现行文档与 `docs/plans/` 设计稿的对应关系；
- 最近两天（2026-09-13 ～ 09-14）合并的 PR #477–#489 对既有结论的影响。

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

- **上轮 11 个开放 issue 已收口 4 个**：
  - **#474** 由 PR #475 修复并关闭（kernel 级 `flock` / `msvcrt.locking` 替换 stale 抢占）；
  - **#424（#265 P6）** 由 PR #478–#484 交付 coverage 门禁、CLI、doctor、GUI 与离线 corpus，2026-09-13 关闭；
  - **#265** 在 P6 交付后于 2026-09-13 关闭，PR #485 把 P6 收口写回矩阵文档；
  - **#15** 于 2026-09-13 按 **not planned / wontfix** 关闭：无真实需求信号，且实现需要改造共享
    TranslationPlan、结构保护、manifest mode 与 build/check/apply adapter 注入，成本 L–XL；未来如有
    真实场景应另开聚焦单格式的新 issue。
- **本轮新增 2 个**：#487（字体字形只读验证 spike）、#488（预检补齐成本 / coverage / 质量摘要）。
  两单均来自 `docs/plans/github_localization_projects_research.md`，审计时无评论、无实现 PR，正文范围与非目标清晰。
- **当前 9 个开放 issue 中，无一是“未记录的真实 blocker”**：3 个外部阻塞（#344 的 Gemini smoke、
  #457 的 A/B 数据、#470 的授权样本）**按本轮口径暂不排期**；2 个新研究单可立即开工（#487、#488），
  #431 / #427 / #272 有明确的首期切片，剩余为 #147 低优先级与 #344 的代码 / 文档收口。
- **最大价值主线已从 P6 转移到“模型接入 + 审校产品化”**：#431 的直连 OpenAI-compatible 后端仍为 0 实现
  （与 #344 ⑧ 的多 Provider 验收同源）；#427 的普通译文 review index 仍无核心 / CLI / 持久化。
- **#344 仍不能关闭**：8 条 Epic 验收中 ①⑤⑥ 已由正文标注为合同级完成，②③④⑦⑧ 中只有 ② 有假 Provider
  子进程证据；③⑧ 真实 smoke 2/4；④ 的 `build-keywords` / `build-revisions` 仍用 `BATCH_MODEL` 生成请求，
  `final_review` 仍取 legacy 字段且硬限 `gemini_batch`；⑦ `sync-status` 仍不暴露 frozen provider / model。
- **行政欠账集中在 #272**：正文仍写 `Blocked by: #265 P6`，但 #265 已于 2026-09-13 关闭；矩阵文档虽有
  PR #485 的关闭记录，正文 P0/P1 checkbox 仍未勾选，§5 还有“待 P6 后复核”的残留表述。

## 全量状态矩阵

| Issue | 审计结论 | 难度 | 下一步 |
| --- | --- | --- | --- |
| [#488 预检补齐成本 / coverage / 质量摘要](https://github.com/hu-border-collie/renpy-translation-lab/issues/488) | 前提准确：现有 `translate-preflight` payload 只有 profile / 计数 / chunk 策略 / 上下文开关 / 风险，无成本、coverage 分类、质量摘要；成本估算、`coverage-status`、quality report 三套底层能力均已存在 | M | 先冻结最小增量合同与字段来源，再扩 payload，最后 CLI/GUI 同步与新鲜 / stale / unknown 测试 |
| [#487 RenPy 字体字形只读 spike](https://github.com/hu-border-collie/renpy-translation-lab/issues/487) | 前提准确：仓库只有工具 GUI 字体下载 / 设置，没有游戏字体解析或字形覆盖检查；Ren'Py adapter 不解析字体配置，`tests/fixtures` 无字体 fixture，`fonttools` 仅是 relation-analyzer 锁文件的传递依赖 | S–M | 先定“最小自研 TTF cmap 解析 or 可选 fontTools”依赖决策与原创 fixture，再做只读脚本 + 报告字段 + 生产接入结论 |
| [#470 真实项目 coverage 归因](https://github.com/hu-border-collie/renpy-translation-lab/issues/470) | 只读脚本、脱敏与确定性测试均已就绪；唯一实质缺口仍是授权真实大型样本，证据缺口已按 #426 格式记录；次要缺口是脚本不自动采集游戏 Ren'Py 版本 | S（样本就绪后） | **本轮暂不排期**（等授权样本）；保持 open / help wanted；可顺手补 Ren'Py 版本采集；不修改生产分类语义 |
| [#457 默认 ModelProfile 决策](https://github.com/hu-border-collie/renpy-translation-lab/issues/457) | 代码默认仍为迁移默认 `legacy-batch + gemini_batch`；Gemini 地区 smoke 与 A/B 仍未执行；GUI「空白创建 Model Routing 配置」仍默认 `gemini-main + sync` | XS–S（维持现状关闭）/ L（改默认） | 本轮以“无数据不改默认 + 现有证据”关闭并记录结论，不勾选未完成项；数据 follow-up 暂缓；决策前先修正 / 标注 GUI 创建默认 |
| [#431 直连协议后端](https://github.com/hu-border-collie/renpy-translation-lab/issues/431) | 无生产 `import openai`，全部生成仍经 LiteLLM；adapter 枚举仍三处硬编码；`ModelProfile.extra_headers` 字段存在但 reader 不读；generation `params` 仍被 legacy 入口拒绝；final_review 仍不能由非 Gemini 承担 | 首期 M；完整 L–XL | 收窄并冻结首期“直连 OpenAI-compatible Sync 生成 + 最小配置 / UI”；实现与离线合同测试先行，真实 smoke 延后；不删 LiteLLM、不改默认 |
| [#427 普通译文审校索引](https://github.com/hu-border-collie/renpy-translation-lab/issues/427) | 依赖 #320/#321/#322/#354/#362/#363 全部已关闭；但全仓库仍无 review index、`needs_recheck`、人工决定持久化 / 失效规则与普通条目编辑草稿；GUI 只有 revision 语料 / 提案 / 订正页 | L（建议 2–3 PR） | 先写 index / decision / 失效规则设计合同，再落可重建核心 + CLI JSON，最后接订正页 GUI |
| [#344 同步与多 Provider Epic](https://github.com/hu-border-collie/renpy-translation-lab/issues/344) | ①⑤⑥ 已标注完成；② 仅假 Provider 子进程证据；③⑧ 真实 smoke 2/4；④ keyword / revision 请求与 final_review 模型仍旁路 route；⑦ `sync-status` 不暴露 frozen provider / model（durable 路径已修“一次 attempt 一次调用”）；文档仍把 Batch 写为主路径 | Epic XL；剩余代码 / 文档 S–M + 外部 smoke | 本轮只做离线收口：修 keyword / revision 与 final_review 取 route、`sync-status` 暴露 frozen provider / model、清理文档残留；Epic 收口等 Gemini 地区 smoke（暂缓） |
| [#272 第三 Adapter 研究](https://github.com/hu-border-collie/renpy-translation-lab/issues/272) | **已解锁**：#265 于 2026-09-13 关闭，矩阵文档已记录 P6 收口并允许创建第三 Adapter 实现单；但正文 `Blocked by`、P0/P1 checkbox 与矩阵 §5 残留“待 P6 后复核”均未更新 | M（研究收口） | 先更新正文与残留表述；做 P0 官方资料复核 + fixture 许可 / 可再分发性调研；P2 只选一个候选并另开实现单 |
| [#147 Story Graph 试点](https://github.com/hu-border-collie/renpy-translation-lab/issues/147) | 代码与 A/B 工具齐备（`compare-variants` 可切 story_memory），默认关闭；仍缺真实试点，生产采纳无法由仓库验证；2026-08-07 的 re-triage 结论未变 | S（试点本身） | **本轮暂缓**（需真实项目试点）；启动前重新核对真实 workspace，再做最小图谱 + 1–2 文件 A/B；先不要开发维护工具 |

## 重点结论

### A. 新单评估

#### #488 预检扩展：基础齐全，价值高，需防“零成本 / 质量通过”伪成功

- **现状证据**：`_run_translate_preflight()`（`gemini_translate_batch.py:22523`）产出的 payload 字段为
  `status / strategy / plan_contract / profile / project / counts / chunk_policy / source_snapshot /
  context_sources / credential_available / risks / environment`（`:22726-22772`）；其中与 coverage 相关的只有
  “零待译”分支的 `COVERAGE_UNCONFIRMED / COVERAGE_EVIDENCE_MISSING` 风险（`:22692-22724`），没有分类计数、
  适用范围或 fresh / stale / unknown 摘要。GUI 启动确认复用同一 payload（`gui_qt/app.py:11279`、`:11348`）。
- **可复用底层均已存在**：`batch_cost_estimate.py` 有 `load_pricing_config`（`:48`）、`resolve_model_pricing`
  （`:95`）、`estimate_manifest_cost`（`:236`）；`coverage-status` / `coverage-review-import` CLI 已交付
  （`gemini_translate_batch.py:23374-23388`），门禁由 `engine_adapters/coverage.py:523` 与
  `evaluate_project_coverage_gate`（`gemini_translate_batch.py:3497`）提供；质量报告有
  `quality_report_export.py:321` 与 manifest 的 `last_quality_findings_path` / `sha256`（`:13308`）。
- **风险与合同**：本单的核心难点不是取数，而是**身份匹配与新鲜度**——报告必须绑定当前项目 / 任务 / 源结果 /
  profile，不能只看文件时间或 latest 指针；无可靠价格时返回 `unknown` 而不是 0；无质量报告时 `not_available`
  而不是“通过”。正文已明确这些边界，审计认为范围合理，可直接按 5 步实施顺序推进。
- **与已交付门禁的关系**：本单只做展示与摘要，不新增放行语义；零待译 coverage 合同（#424 P6）保持不变，
  预检结论也不是 check / apply 授权。文档化“最小增量字段表”应作为第一步合并物，避免范围膨胀为 dashboard。

#### #487 字体字形 spike：前提成立，但要先解决依赖与许可证两个前置

- **现状证据**：仓库内的字体能力只服务工具 GUI——`gui_qt/font_helpers.py:1`（`QFont` 加载与系统字体回退）、
  `gui_qt/font_worker.py`、`scripts/download_gui_fonts.py` 与对应测试；issue 正文已明确这类“工具 GUI 字体是否安装”
  不能作为游戏字体有效性的证据。Ren'Py adapter 不解析字体配置（`engine_adapters/renpy.py` 无 `font` 命中），
  `tests/fixtures/renpy_smoke` 没有任何字体 / `gui.text_font` 配置；`fonttools` 只作为 matplotlib 的传递依赖
  出现在 `requirements-lock/py311-relation-analyzer.txt:77`，不在任何权威 requirements 输入中。
- **研究来源成立**：`docs/plans/github_localization_projects_research.md` §6.1.8（`:657`）与 §8（`:739`）已把
  “目标语言字体有效性与渲染回退检查”列为 doctor / preflight 候选；`RenForge` 为 GPL 参考，结论是学“字形探测”
  思路而不是复制实现（`:714`）。
- **范围判断**：首版只解析静态字体路径、可确定的样式引用与 TTF/OTF cmap，FontGroup / fallback / 动态样式一律
  `unknown`，且只读、不执行游戏 Python——这与仓库现有 `EngineAdapter` 只读快照边界一致。真正的成本在
  fixture 许可证与解析依赖：若坚持零新依赖，需要自研最小 cmap 解析；若引入可选 `fonttools`，需按
  `AGENTS.md` / `CONTRIBUTING.md` 的权威 requirements 输入与生成器规则处理，不能手改锁文件。

### B. Epic 与既有主线

#### #344：不能关闭，剩余项复核

- **② 中断只重跑缺失 chunk**：durable 路径已有假 Provider 子进程恢复测试；真实计费中断 smoke
  （`scripts/run_durable_sync_provider_smoke.py`）仍未执行，属外部证据缺口。
- **③⑧ 真实 smoke 2/4**：`docs/provider_smoke_matrix.md:79` 仍记录 Gemini Sync / Batch “未执行”
  （HTTP 400 `User location is not supported`）；LiteLLM 内置与自定义 OpenAI-compatible 已通过。
- **④ route 旁路仍存在**：`create_revision_package`（`:5825`）与 `create_keyword_package`（`:7451`）虽然
  现在会 `freeze_runtime_routing_plan(... required_stages=...)` 并把计划 `attach_model_routing` 进 manifest
  （`:5927`、`:7531`），但请求 JSONL 仍用 `build_revision_request(chunk)` / `build_keyword_request(chunk, ...)`
  的默认模型，即 `BATCH_MODEL`（`:5792`、`:7416`），manifest 也写 `batch_model: BATCH_MODEL`（`:5869`、`:7494`）；
  若用户在 `model_routing` 中覆盖 keyword / revision 路由，实际生成不会使用。`final_review` 同理：
  任务冻结 `STAGE_FINAL_REVIEW` 计划（`:7012`），但模型仍取 `FINAL_REVIEW_MODEL or BATCH_MODEL`（`:7060`），
  且 legacy 入口要求 final_review 策略必须是 `gemini_batch`（`model_routing_reader.py:428-433`），非 Gemini
  Provider 不能承担最终审校。
- **⑦ 观测性**：durable 服务已把每次 attempt 收敛为一次 Provider 调用（`sync_run_service.py:786-802`，
  `retry_attempts=1`、`allow_credential_rotation=False`）；但 `sync-status` 的 store snapshot
  （`sync_run_store.py:2869`、`sync_run_service.py:651`）仍只有 run / plan fingerprint / 进度 / artifacts，
  没有 frozen provider / model；非 durable 的 `run_sync_request` 调用方仍默认 `allow_credential_rotation=True`
  （同一 Provider / 模型内换 Key，不是跨模型 fallback）。
- **文档残留**：`docs/sync_workflow.md:7`、`docs/setup.md:164`、`README.md:38` 仍把 Sync 描述为小范围 / 补译、
  Batch 描述为优先主路径，与 Epic 方向冲突。
- **外部验收归属**：#341 三项外部验收（fresh published PA、非 Gemini embedding smoke、扩大 A/B）仍未指定承接方，
  需在收口评论中明确。
- **2026-09-14 增量**：#486 由 PR #489 以严格 final-review 结果 schema 与精确重复 finding 关闭，不改变
  final_review 的模型路由缺口。

#### #457：可“维持现状”关闭，GUI 隐性默认仍待处理

- 代码默认证据：`model_routing_migration.py:174` 的 `defaults` 为 `legacy-batch + gemini_batch`；
  `CHANGELOG.md:70` 明确“新用户 / 新项目的默认主模型与执行策略保持不变”；旧行为迁移有签名比较
  （`:183-188`）与回归测试。
- 未完成项 2/5 仍成立：`docs/provider_smoke_matrix.md:79` Gemini 未执行；A/B 数据在矩阵文档中仍无记录。
- GUI 隐性不一致仍在：`gui_qt/settings/profiles_page.py:727-753` 的空白创建默认 `gemini-main + sync`，
  与迁移默认 `gemini_batch` 不一致。决策前至少应加提示或复用迁移默认。
- 推荐路径不变：以“维持现状 + 证据”关闭，未完成项不勾选，另开数据可用时的 follow-up。

#### #431：方向有效，仍是 0 实现的首期 M 切片

- 无直连后端：生产代码无 `import openai`（仅 `logs/.venv-smoke` 里的第三方依赖），生成全部经
  `litellm_sync_backend.py`。
- 四个接线缺口复核结果不变：adapter 枚举在 `model_profile.py:54-56`、`model_routing_config.py:25-27`、
  `model_profiles_editor.py:33` 三处硬编码 `gemini / litellm`；`ModelProfile.extra_headers` 字段存在
  （`model_profile.py:371`，且已进 manifest 序列化），但 `model_routing_reader.py` 不读取、GUI 无入口；
  generation `params` 仍被 legacy 入口拒绝（`model_routing_reader.py:335-337`）；final_review 仍硬限
  `gemini_batch`（见 #344）。
- 与 #344 的交集：本单的 final_review 统一路由无法在 #344 ④ 修复前完成，首期验收应继续排除最终审校；
  与 #457 解耦，不修改默认值。

#### #427：依赖已齐，核心 / CLI / GUI 均未实现

- 已有：`revision_corpus.py`、`revision_proposals.py`、`revision_selection.py`（状态只有
  `valid / no_op / invalid / stale / conflict`，`revision_selection.py:25-29`）、GUI 的
  `revision_corpus_workflow.py` / `revision_selection_dialog.py` / `revision_workflow.py` 等。
- 缺失：全仓库无 `needs_recheck` / review index / 人工决定持久化实现（仅研究文档与上轮审计提及）；
  corpus 只覆盖 revision scanner 识别的条目，不含 glossary / 结构 finding / provenance / 人工状态。
- 建议不变：先冻结 index / decision schema 与失效规则，再落可重建核心 + CLI JSON，最后接订正页；
  避免与 #354 reuse decision、#362 quality-ack 的语义边界重叠。

### C. 研究与低优先级

#### #272：已被 #265 关闭解锁，先清行政欠账再做 P0/P1

- 矩阵文档头部已更新为 2026-09-13 核对：P5 于 2026-09-04 交付、P6（#424）已交付并通过 CI、#265 已于
  2026-09-13 关闭，“后续推荐候选可创建独立实现 Issue”（`docs/plans/visual_novel_localization_matrix.md:4-6`）。
- 但 issue 正文仍写 `Blocked by: #265 P6（由 #424 承接）`，P0 第一条 checkbox 仍写“等待 P6 #424”，
  与事实不符；矩阵 §5 还有“待 #265 P6 后复核”的段落（`:356`）。
- 推荐结论未变：`Naninovel A+ 首选 + RPG Maker MV/MZ 叙事模式 B+ 双轨`；P2 只选一个候选、写清受支持版本 /
  制品 / 排除项 / fixture / 验收，并另开实现单。#265 关闭后这个动作已经允许。

#### #470：只缺授权样本，保持 blocked / help wanted

- `scripts/coverage_block_attribution.py` 已就绪（只读、默认脱敏、确定性复跑有测试）；
  `docs/plans/renpy_coverage_block_attribution.md:8` 与 §6（`:251-253`）已记录“没有授权副本，结论不得外推”。
- 2026-09-13 的证据缺口评论满足验收第 1 条的 fallback 分支；验收 2–6 必须拿到样本才能完成。
- 次要缺口复核成立：脚本记录 engine / adapter version，但不自动采集游戏自身的 Ren'Py 版本，需人工记录。

#### #147：结论不变，试点优先于工具化

- `translator_config.example.json` 的 batch / sync `story_memory.enabled` 仍默认 `false`（`:81-82`、`:141-142`）；
  `translation_ab_experiment.py:207-209` 的 `compare-variants` 仍可直接切换 story_memory；
  `story_memory.py` 自 2026-08-07 以来没有实质功能变更。
- 生产采纳状态被 gitignore 挡住，无法由公开仓库验证；2026-08-07 评论已自行更正。启动前应重新核对真实
  workspace，再做“≥5 角色 / ≥10 关系 + 1–2 个 split / file A/B”，不要先开发维护工具。

## 上轮修复的快速验证

| 上轮 issue | 修复证据 |
| --- | --- |
| #474 | `atomic_io.py` 改为进程级 kernel 锁（POSIX `fcntl.flock` / Windows `msvcrt.locking`，`:96-121`），不再有 read-then-unlink 抢占窗口；`tests/test_atomic_io.py` 覆盖 |
| #424 / #265 P6 | coverage 门禁接入零待译完成声明（`gemini_translate_batch.py:22692-22724`）、PA 发布（`project_analysis.py:1545-1567`）、Final Review readiness（`:7034-7049`）；doctor `engine_status`（`:19080+`）；GUI coverage / review 与 engine snapshot 浏览；CLI `coverage-status` / `coverage-review-import`；矩阵文档 PR #485 记录 #265 关闭 |
| #15 | 2026-09-13 按 not planned / wontfix 关闭，并留下了“重新开聚焦单格式 issue”的明确指引；不属于已完成交付 |

## 关键依赖

```text
#344 剩余代码收口 ──┬──> #431 首期（final_review 统一路由）
                    └──> #431 完整验收（未安装 LiteLLM 时全阶段可用）

#344 真实 smoke ←──── Gemini 支持地区与凭据 ──> #457 默认值决策
#457 维持现状关闭 ──── 不依赖数据，可先执行；数据可用后再开 follow-up

#272 P0/P1（官方资料复核 + fixture 许可）──> P2 选一个候选 ──> 第三 Adapter 实现单（#265 已关闭，已解锁）

#488 ── 复用 coverage-status / PA / quality report / pricing；不新增放行语义
#487 ── 字体解析依赖与 fixture 决策 ──> 只读 spike ──> doctor / preflight 接入另行立项

#427 ── 独立核心 + CLI ──> 订正页 GUI；与 #354 / #362 语义边界需在 schema 中钉死
#470 ── 授权真实样本（外部阻塞）
#147 ── 独立低优先级试点（compare-variants 已可用）
```

需要避免的新“双事实源”：

- #488 的 coverage / 成本 / 质量摘要与 doctor、Project Analysis、Final Review、`coverage-status` 各自的判定；
- #427 的 review index / decision 与 #354 reuse decision、#362 quality-ack 的语义边界；
- #431 的直连协议配置与现有 `custom_litellm_providers` 的迁移等价性；
- #487 的字体字形结论与 RenPy 运行时渲染 / 排版验收的边界。

## 排期口径：暂不考虑真实项目 / 真实 Provider 依赖

本轮明确不排期需要在**授权真实项目副本**、**真实计费 Provider**、**Gemini 支持地区**或**生产 workspace**
上执行的事项。按此口径拆分如下。

**暂不排期（等外部条件，本轮只保留现状记录）**

- **#470 全单**：必须有授权的真实大型 Ren'Py 只读副本；样本到位前不做任何替代性结论。
- **#147 试点**：`≥5 角色 / ≥10 关系` 的最小图谱与 A/B 必须落在真实项目上；代码与 A/B 工具已就绪，试点延后。
- **#344 的 Epic 收口**：② 真实计费中断 smoke、③⑧ Gemini Sync / Batch 真实 smoke，以及 #341 的
  fresh published PA / 非 Gemini embedding smoke / 扩大 A/B 三项外部验收全部延后；不影响 ④⑦ 与文档的离线收口。
- **#457 的“带数据决策”分支**：Gemini 支持地区 smoke 与真实项目 A/B 延后。只保留不依赖新数据的
  “维持现状关闭 + 数据 follow-up”。
- **#431 的真实连接 smoke 验收**：真实 Provider 兼容性验证延后到发布前；实现与离线合同测试先行。

**本轮可推进（离线 / 本地 fixture 即可完成）**

- **#488**：正文明确离线无 Key 可运行、不调用 Provider；coverage / 成本 / 质量三套底层 API 已存在。
- **#487**：只读 spike + 原创 fixture，明确不启动游戏、不下载字体；依赖与许可证决策可先做。
- **#427**：index / decision 核心、CLI JSON 与 GUI 均可用合成语料和现有测试沙箱覆盖。
- **#272**：官方资料复核、矩阵刷新与 fixture 可再分发性调研均为离线研究，且 #265 关闭后已解锁。
- **#431 实现主体**：直连 OpenAI-compatible adapter、配置 / UI 接线、错误映射与离线合同测试。
- **#344 剩余代码 / 文档**：keyword / revision 与 final_review 取 route、`sync-status` 暴露
  frozen provider / model、清理文档残留；均可用离线 fixture / 假 Provider 验证。
- **#457 维护者决策**：记录“维持现状 + 现有证据”本身不需要新数据。

## 推荐执行顺序（剔除真实项目 / 真实 Provider 依赖后）

1. **#457 维持现状关闭决策（XS–S）**：记录“无数据不改默认 + 现有迁移 / 回归证据”，另开数据 follow-up；
   不勾选未完成项，不需要真实项目。
2. **#272 行政收口 + P0/P1（M）**：先改正文 `Blocked by`、P0 状态与矩阵 §5 残留；再做官方资料复核与
   fixture 许可 / 可再分发性调研；P2 选型与实现单可随后立。
3. **#344 代码 / 文档收口（S–M，不含 smoke）**：修 keyword / revision 与 final_review 取 route、
   `sync-status` 暴露 frozen provider / model、清理 Sync / Batch 文档残留；Epic 保持 open。
4. **#431 首期实现（M，smoke 延后）**：直连 OpenAI-compatible Sync 生成 + 最小配置 / UI 接线 + 离线合同测试；
   与 #344 ④ 协调 final_review 改造排期。
5. **#488 预检扩展（M）**：最小字段合同 → 成本 / coverage / 质量摘要 → CLI/GUI 同步。
6. **#487 字体 spike（S–M）**：依赖与 fixture 许可证决策 → 只读脚本 / 报告 → 生产接入结论。
7. **#427 审校索引（L，2–3 PR）**：设计合同 → 核心 + CLI → 订正页 GUI。
8. **暂缓**：#470（等授权样本）、#147（等真实项目试点）、#344 的真实 smoke 与 Epic 收口、
   #457 的数据 follow-up、#431 的真实连接 smoke。

## Issue 记录维护清单

按上两份快照的维护规则，以下记录应更正（本轮只审计，不代改 GitHub）：

| Issue | 需要更正的内容 |
| --- | --- |
| #272 | `Blocked by` 改为“无（#265 已于 2026-09-13 关闭）”；P0 第一条改为“基于 P5/P6 结果复核官方资料”；同步矩阵 §5 / §6.1 的 P6 残留表述 |
| #344 | ②–⑧ 继续保留未完成标注；说明 #341 三项外部验收与真实 smoke 暂缓的原因；处理悬空评论 `@.codex-issue-comment-344.md` |
| #457 | 记录维护者决策：维持 `legacy-batch + gemini_batch` 或进入改默认流程；说明 GUI 空白创建默认 `sync` 的差异 |
| #431 | 冻结首期范围与验收（排除最终审校），标注与 #344 ④ 的依赖；不改默认值 |
| #487 / #488 | 保持正文范围与非目标；#488 实施时补 P6 已交付门禁的复用说明，#487 明确依赖决策与许可证来源 |
| #470 | 保持 open / blocked / help wanted；可补“脚本暂不采集 Ren'Py 版本”为已知限制 |
| #147 | 启动试点前重新核对真实 workspace；生产采纳事实不沿用 2026-07 记录 |

## Issue 维护规则

- 正文中的事实或 checkbox 错误时，应修改正文，并用一条日期评论记录依据；
- 仅有阶段性判断时只留评论，不把审计全文复制到每个 issue；
- PR 已覆盖的 issue 不再并行实施，等待 PR 合并关闭；
- 真实 Provider smoke、私有项目实测与生产 workspace 事实必须标注证据等级，
  不得用离线 / mocked 结果冒充；
- 后续复核应生成新的日期快照，不回写本文伪装成当时事实。
