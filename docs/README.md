# 文档地图

根目录 `README.md` 负责快速上手和项目定位；本目录收纳配置、Batch、上下文、GUI 与项目边界等专题说明。

**发行状态：最新稳定版为 v1.0.0，`main` 为后续开发线。** 稳定范围、未发布变化、验证记录、交付方式与不覆盖事项见 [变更日志](../CHANGELOG.md) 和 [项目说明](project_notes.md)。

**读文档时**：用户手册以「现行」一组为准；[archive/](archive/README.md) 中的历史文稿只作设计/审计背景，不要当当前界面说明。

## 推荐阅读（现行）

| 你想… | 读 |
|---|---|
| 第一次通过 GUI 完成翻译 | [GUI 快速开始](quickstart_gui.md) → [GUI 工作台](gui_workbench.md) |
| 让 Agent / 脚本通过 CLI 翻译 | [Agent / CLI 快速开始](quickstart_agent.md) → [Batch 工作流与安全检查](batch_workflows.md) |
| 由宿主 Agent 提供初译，工具接收并安全写回 | [外部初译工作包（实验 CLI）](external_translation_work.md) |
| 手动使用完整 CLI 工作流 | 根目录 `README.md` → [安装与本地配置](setup.md) → [Batch 工作流与安全检查](batch_workflows.md) 或 [同步翻译工作流](sync_workflow.md) |
| 多游戏工作区总表 | [工作区项目总表](games_registry.md) |
| 理解环境检查建议 | [环境检查智能建议机制](doctor_recommendations.md) · [状态矩阵](doctor_states_matrix.md) |
| 理解引擎扫描边界、coverage 与安全写回 | [Engine Adapter、覆盖审计与安全写回](engine_adapter.md) |
| 启用 RAG / 原文索引 / 剧情记忆 | [上下文系统](context_systems.md) · [setup.md](setup.md) 中的项目级开关 |
| 角色关系 / 语义分析 | [关系与语义分析](relation_analysis.md) · [`relation_analyzer/README.md`](../relation_analyzer/README.md) |
| 项目边界与安全 | [项目说明](project_notes.md) |
| 了解项目如何从 OpenCode 原型演化而来 | [项目沿革](project_history.md) |
| 查看版本变化 | [未发布与发行变化](../CHANGELOG.md) · [v1.0.0 发行说明](releases/v1.0.0.md) |
| 参与开发 / AI 协作 | 根目录 [AGENTS.md](../AGENTS.md) → [CONTRIBUTING.md](../CONTRIBUTING.md)（含 **CLI / GUI 同步**） |
| 理解 PR 门禁与定时集成 | [CI 与定时集成检查](ci.md)（含 lint / type / audit） |
| 在 Codex 沙箱内跑 GUI 测试卡住 | [GUI 测试沙箱环境说明](gui_tests_sandbox_notes.md) |
| 配置离线迁移与回滚（P1） | [模型配置离线迁移（P1）](model_config_migration.md) |
| 选择模型 / 执行方式 / 最终审校并做真实 smoke | [Provider / 执行策略 / 最终审校 Smoke Matrix](provider_smoke_matrix.md) |
| 理解依赖所有权与哈希锁 | [依赖输入与哈希锁](dependencies.md) |

## 文档分组

### 现行：快速开始

- [GUI 快速开始](quickstart_gui.md)：源码安装、选择项目、配置密钥、环境检查和第一次安全写回。
- [Agent / CLI 快速开始](quickstart_agent.md)：面向操作工具的 Agent，说明环境准备、显式 manifest 流程和 `check -> apply` 门禁。
- [Agent 开发约定](../AGENTS.md)：面向修改本仓库的 Agent；完整开发规则仍以 [CONTRIBUTING.md](../CONTRIBUTING.md) 为准。

### 现行：配置与 GUI

- [模型配置迁移（#348 P1/P2）](model_config_migration.md)：配置副本的预览、暂存、备份与回滚；P2 已接入兼容入口的生产读取。
- [Provider / 执行策略 / 最终审校 Smoke Matrix（#348、#431）](provider_smoke_matrix.md)：Gemini Sync、LiteLLM 内置/自定义、直连 `openai_compatible`、Gemini Batch 与 `final_review` sync 的最短验证路径、调用预算、能力探测、记录模板与脱敏诊断导出。
- [安装与本地配置](setup.md)：`translator_config.json`、**按项目**的 `project_context_settings.json`、work 目录、SDK / TL 模板。
- [GUI 工作台](gui_workbench.md)：当前 PySide6 界面（统一侧边导航、项目与环境、任务页、设置、诊断与运行日志、写回安全边界）。
- [工作区项目总表](games_registry.md)：`games_registry.json` / `GAMES.md`、CLI 与 GUI 刷新边界。

### 现行：Batch、上下文与检查

- [Batch 工作流与安全检查](batch_workflows.md)：`build → apply`、订正、最终审校 campaign、关键词、identity v2、A/B、golden corpus。
- [同步翻译工作流](sync_workflow.md)：同步 CLI 的 preview → 人工审查 → 显式 apply、配置、供应商与安全边界。
- [外部初译工作包（实验 CLI）](external_translation_work.md)：无项目模型调用的材料导出、部分成果接收、候选订正、持久冲突与绑定预览写回；真实章节对照尚待验证。
- [Engine Adapter、覆盖审计与安全写回](engine_adapter.md)：Ren'Py P1–P4 扫描、coverage、版本快照与复用，以及 TyranoScript P5 原生 catalog 写回边界。
- [实际模型用量账本](model_usage_ledger.md)：按当前项目、任务、阶段、provider 与模型归集 Batch / 同步实际调用，并说明离线补录、成本和未知值语义。
- [上下文系统](context_systems.md)：RAG、原文索引、Story Memory、store 路径与 benchmark。
- [环境检查智能建议机制](doctor_recommendations.md)：建议等级、必需/可选并列、workflow_state。
- [环境检查状态矩阵](doctor_states_matrix.md)：layout / pending / 上下文派生字段与决策漏斗（开发对照）。

### 现行：分析与项目状态

- [关系与语义分析](relation_analysis.md)
- [CI 与定时集成检查](ci.md)
- [GUI 测试沙箱环境说明](gui_tests_sandbox_notes.md)：GUI 测试写入 `%LOCALAPPDATA%` 缓存，Codex 沙箱内会被拦截导致卡住。
- [依赖输入与哈希锁](dependencies.md)
- [项目说明](project_notes.md)
- [项目沿革](project_history.md)
- [story_graph.example.json](story_graph.example.json) · [story_graph.schema.json](story_graph.schema.json)

### 规划、合同与研究（非用户手册）

完整的计划目录、当前状态、合并证据与剩余 issue 见[规划与设计索引](plans/README.md)（核对至 2026-09-27 main）。有日期的研究仍保留原基线；索引中的当前状态说明不会改写其历史结论。

- 本轮工程已分别由 PR #522（#521）、#515 / #517 / #523（#427 S1/S2）、#524（#512）和 #527（#513）合并。#427 仍等待真实语料、操作与游戏验收；#512 / #513 的 issue 验收清单尚未回填，OPEN 不表示实现缺失。
- 统一真实项目 / Provider 验收仍跟踪 #518、#508、#344、#431、#470；工程合并、CI 或合成 fixture 均不能替代这些证据。操作与记录入口分别见[外部初译工作包](external_translation_work.md)、[Provider Smoke Matrix](provider_smoke_matrix.md)及计划索引。
- #509 Translator++ 往返、#520 OpenRouter Batch、#272 第三引擎与 #147 Story Graph 不属于本轮；#509 / #520 先做实验，再决定是否创建生产接入任务。
- [维护复杂度治理路线](plans/maintainability_roadmap.md)：#528 已由 PR #529 / #530 完成并关闭；#531 已登记但尚未实施，属于独立维护队列，不是 #510 或本轮功能的前置条件。
- [#202 Settings 页面合同](plans/issue-202-settings-page-contract.md)：Phase A–D 已完成，issue 已关闭；当前界面见[GUI 工作台](gui_workbench.md)。
- [#348 Model Routing 合同](plans/issue-348-model-routing-config-contract.md)：P0–P3 已完成，issue 已关闭；当前迁移见[模型配置迁移](model_config_migration.md)，真实 smoke 归 #344 / #431。
- [#422 apply/export 事务合同](plans/issue-422-p2-apply-export-contract.md)：PR #473 已交付；安全和恢复合同仍有效，用户用法见[Agent 快速开始](quickstart_agent.md)与[Batch 工作流](batch_workflows.md)。
- [Engine Adapter 合同](plans/engine_adapter_contract.md)：#265 P0–P6 已完成并关闭；当前用法和边界见[Engine Adapter 手册](engine_adapter.md)。#272 的生产立项仍暂缓。
- [#364 质量校准手册与基线](plans/issue-364-calibration-runbook.md)：真实项目校准已完成并关闭；保留历史结论和复现步骤。当前白名单字段语义见[Batch 工作流](batch_workflows.md)。
- [#426 coverage 归因报告](plans/renpy_coverage_block_attribution.md)：合成归因与 #460–#464、#471 修复已完成；真实大型样本仍归 #470。
- [#487 字体覆盖只读 spike](plans/renpy_font_coverage_spike.md)：PR #511 已交付并关闭；字体生产接入仍待评估，没有纳入本轮。
- [#427 普通译文审校合同](plans/issue-427-review-index-contract.md)：S1/S2 工程已合并；真实验收仍归 #427，当前 GUI 说明见[GUI 工作台](gui_workbench.md)。
- [#431 OpenAI-compatible 与 final_review Sync 合同](plans/issue-431-openai-compatible-contract.md)：S1–S3 及 smoke 准备已合并；解绑 final_review 已完成，真实 Provider 验收仍归 #431 / #344。
- [Translator++ 接入评估](plans/translatorpp_integration_assessment.md)：尚无 bridge 或真实往返证据；按 #509 先实验、后决定是否实施。

### 历史参考（已归档）

下列文档在 [archive/](archive/README.md)，**保留为过程与审计记录**；界面与交付状态以 [GUI 工作台](gui_workbench.md) 和代码为准。

- [GUI 信息架构重组计划](archive/gui_ia_redesign.md)：Epic #157 的 P0–P3 历史设计 SSOT，并记录后续 #176 页面化交付衔接。文中「分析与准备」「诊断页」等多为改造前用语。
- [翻译全生命周期审计](archive/translation_workflow_audit.md)：流水线与门禁的代码审计快照，可能落后于最新实现细节。
- [Design QA 验收记录](archive/design-qa.md)：2026-07 统一侧边导航与页面归属的视觉/自动化验收快照；不是用户手册。
- [文译参考对照与 Batch 主路径增强计划](archive/wenyi_reference_and_batch_roadmap.md)：已落地路线的设计与取舍记录；现行用法以 Batch、上下文和用量账本文档为准。
- [#346 实施分步计划](archive/issue-346-implementation-plan.md)：Sync / Batch 共用 TranslationPlan、ContextAssembler 与请求合同全阶段交付记录（PR #403）。
- [#347 设计 P0：耐久同步执行器](archive/issue-347-durable-sync-executor-plan.md)：Run/Request/Attempt 状态机与耐久同步执行器核心设计（CLI 已交付，交接见 [plans/issue-347-to-348-handoff.md](plans/issue-347-to-348-handoff.md)）。
- [#341：Provider-neutral Embedding 纯核心](archive/issue-341-provider-neutral-embedding-core.md) · [Embedding adapters 与 store identity](archive/issue-341-embedding-adapters-store-identity.md)：跨 Provider 向量合同、store 校验与生产接线。
- [TyranoScript V600+ P5 parser 调研与 fixture 基线](archive/tyranoscript_v600_parser_research.md)：P5 验证 adapter 调研与基线（P5 已合入）。
- [开放 Issues 审计（2026-08-08）](archive/open_issues_audit_2026-08-08.md) · [开放 Issues 审计（2026-09-12）](archive/open_issues_audit_2026-09-12.md) · [Final Review 失败分类探针](archive/final_review_result_failure_spike.md)：历史 issues 快照与审校结果失败分类 spike。

## 配置分层（速查）

| 范围 | 文件 | 典型内容 |
|---|---|---|
| 工具全局 | `translator_config.json`、`api_keys.json` | `game_root`、模型、chunk、SDK、RAG/索引的**默认** |
| **当前项目** | `<work>/project_context_settings.json` | 是否启用批量 RAG、原文索引、build 时暖库 |
| 项目资产 | `<work>/glossary.json`、`macro_setting.md` 等 | 术语、口吻 |
| 项目运行状态 | `<work>/translation_usage/usage_ledger.json` | 实际模型调用的本地用量账本（不要提交） |
| 工作区总表 | 工作区根 `games_registry.json` | 多游戏进度（非 lab 仓库内默认） |

## 维护约定

- 改 GUI 入口名称或设置分区时，**同步改** `gui_workbench.md` 与 `gui_qt/user_copy.py` 中的用户文案。
- 改 doctor 规则时，同步 `doctor_recommendations.md` 与测试。
- 历史文稿放在 `archive/`，只加状态横幅或修断链，不整篇改写成新手册，避免与「现行」重复维护。
