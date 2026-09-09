# 规划与设计草案

本目录放**尚未成为现行用户手册**的计划、路线图与参考对照。

| 状态 | 放哪 |
|------|------|
| 进行中 / 待拆 issue | 本目录 `docs/plans/` |
| 已交付、仅作回顾 | 迁到 [`docs/archive/`](../archive/README.md) |
| 已稳定的用法说明 | 写进 `docs/` 现行手册（如 `batch_workflows.md`） |

读本目录时：以文首状态横幅为准；实现与界面以代码和现行手册为准。

## 当前设计

- [#202 Phase A：Settings 页面契约与现状基线](issue-202-settings-page-contract.md)：
  记录 10 个 Settings 页面、字段/即时持久化所有权、局部 worker 与 as-is 缺口，并冻结 Phase B 的
  `load/collect/validate/reset` + 错误聚焦/宿主事件边界；Phase B 的 `gui_qt/settings/` contract、
  registry、coordinator 与 legacy adapter 已合并（PR #434）；Phase C 已将 LiteLLM 页迁到独立
  `SettingsPage`。Phase D 已将 10 个 Settings 页迁出为独立 `SettingsPage`；保存收口
  仍属 Phase D。
- [#348 P0/P1：Model Routing 配置、迁移合同与离线迁移](issue-348-model-routing-config-contract.md)：
  冻结版本化 `model_routing` schema 与迁移事务；P1 离线 reader、显式暂存迁移/回滚已实现（详见现行手册 [model_config_migration.md](../model_config_migration.md)），P2 生产解析与 P3 设置页待推进。
- [#347 → #348 耐久 Sync 产品化交接](issue-347-to-348-handoff.md)：
  冻结服务/snapshot/CLI/制品接缝、Provider 中断 smoke 与 #348 的 GUI、配置迁移和
  最终命令别名验收边界。
- [Engine Adapter P0：Ren'Py 当前调用链与合同设计](engine_adapter_contract.md)：
  #265 / #285 的调用链审计、adapter/coverage schema 与阶段接入边界。P1–P4 门禁实测已完成；当前实现说明见 [Engine Adapter、覆盖审计与安全写回](../engine_adapter.md)。
- [视觉小说引擎本地化能力矩阵与后续 Adapter 路线](visual_novel_localization_matrix.md)：
  #272 针对 Naninovel、Godot+Dialogic 2、Visual Novel Maker、Monogatari、
  KiriKiri/KAG、RPG Maker MV/MZ 六大引擎的 12 维本地化能力评估矩阵与第三 Adapter 路线决策。
- [GitHub 视觉小说本地化工具源码研究](github_localization_projects_research.md)：
  对 RenLocalizer、Dialogue Visual Editor、RenPyTranslator、translate-renpy、
  2R-Tools、GameStringer、rpgmaker-translator 等项目的源码级对照，以及与当前 manifest、EngineAdapter、
  quality gate 和 `check -> apply` 合同的迁移边界分析。
- [多引擎本地化适配与候选引擎开源生态研究](multi_engine_localization_ecosystem_research.md)：
  针对 #265（引擎适配边界与版本化翻译资产）与 #272（第三引擎选型）对标 VNTextPatch、Translator++、
  Weblate/translate-toolkit 架构以及 Naninovel、Godot+Dialogic 2、RPG Maker 原生生态的深度研究。
- [#364 真实项目质量规则校准执行手册](issue-364-calibration-runbook.md)：
  A1 离线语料与 A3 校准报告工具的使用方法，以及已完成的 B 线执行步骤。
- [真实项目机械质量校准基线](quality_calibration_baseline.md)：
  基于三项目真实语料与人工标注的机械质量规则校准数据，支撑 #364 规则默认值调整。

## 已交付并归档的设计与调研

已完成并由主干合入的阶段性设计与探针已归档至 [`docs/archive/`](../archive/README.md)：

- [#346 实施分步计划：Sync / Batch 共用 TranslationPlan、ContextAssembler 与请求合同](../archive/issue-346-implementation-plan.md)：P0–P5 全阶段已收口并合入主干（PR #403）。
- [#347 设计 P0：耐久同步执行器、崩溃恢复与统一安全链路](../archive/issue-347-durable-sync-executor-plan.md)：核心实现与 CLI `sync-*` 命令已交付。
- [#341 P0：Provider-neutral Embedding 纯核心](../archive/issue-341-provider-neutral-embedding-core.md)：跨 Provider 向量合同与指纹规范（PR #401 已交付）。
- [#341：Embedding adapters 与 store identity](../archive/issue-341-embedding-adapters-store-identity.md)：Adapter 与 store identity 持久化校验（PR #388 / PR #401 已交付）。
- [TyranoScript V600+ P5 parser 调研与 fixture 基线](../archive/tyranoscript_v600_parser_research.md)：P5 验证 adapter 调研与基线（P5 已合入）。
- [开放 Issues 审计（2026-08-08）](../archive/open_issues_audit_2026-08-08.md)：21 个开放 issues 快照与收口顺序审计。
- [Final Review 结果失败分类与 targeted resume spike（#309）](../archive/final_review_result_failure_spike.md)：模型审校结果失败分类 probe 与 targeted resume 设计。
