# 规划与设计草案

本目录放**尚未成为现行用户手册**的计划、路线图与参考对照。

| 状态 | 放哪 |
|------|------|
| 进行中 / 待拆 issue | 本目录 `docs/plans/` |
| 已交付、仅作回顾 | 迁到 [`docs/archive/`](../archive/README.md) |
| 已稳定的用法说明 | 写进 `docs/` 现行手册（如 `batch_workflows.md`） |

读本目录时：以文首状态横幅为准；实现与界面以代码和现行手册为准。

## 当前设计

- [开放 Issues 审计（2026-08-08）](open_issues_audit_2026-08-08.md)：
  基于 `main@78ab050` 的 21 个开放 issues 时间点快照、依赖关系、难度与收口顺序；
  最新状态仍以 GitHub、当前代码和现行文档为准。
- [Engine Adapter P0：Ren'Py 当前调用链与合同设计](engine_adapter_contract.md)：
  #265 / #285 的调用链审计、adapter/coverage schema 与阶段接入边界。P1 的只读
  adapter / coverage 与 P2 的 relocation、validation、writeback plan 消费均已落地；
  本文继续作为 reconciliation、coverage 下游门禁和后续引擎阶段的合同基线。
  当前实现说明见 [Ren'Py Engine Adapter 与覆盖审计](../engine_adapter.md)。
- [#346 实施分步计划：Sync / Batch 共用 TranslationPlan、ContextAssembler 与请求合同](issue-346-implementation-plan.md)：
  基于 `main@fa69d14` 的 P0–P5 分阶段实施计划；D1–D7 已冻结，P1 纯核心已合并，
  P2–P5 继续按本文边界推进。
- [#347 设计 P0：耐久同步执行器、崩溃恢复与统一安全链路](issue-347-durable-sync-executor-plan.md)：
  基于 `main@ed07a99` 冻结 Run/Request/Attempt 状态机、SQLite 事务边界、
  `outcome_unknown` 恢复语义、usage 去重、共享 result/check 门禁、durable Sync preview/apply 与 Batch direct apply 接缝、
  CLI JSON 合同、fault-injection 矩阵及与 #346/#341/#348 的所有权边界。
- [#347 → #348 耐久 Sync 产品化交接](issue-347-to-348-handoff.md)：
  冻结服务/snapshot/CLI/制品接缝、Provider 中断 smoke 与 #348 的 GUI、配置迁移和
  最终命令别名验收边界。
- [#364 真实项目质量规则校准执行手册](issue-364-calibration-runbook.md)：
  A1 离线语料与 A3 校准报告工具的使用方法，以及已完成的 B 线执行步骤；
  三项目聚合统计、人工标注和 A2 前后对比见
  [真实项目机械质量校准基线](quality_calibration_baseline.md)。
- [视觉小说引擎本地化能力矩阵与后续 Adapter 路线](visual_novel_localization_matrix.md)：
  #272 针对 Naninovel、Godot+Dialogic 2、Visual Novel Maker、Monogatari、
  KiriKiri/KAG、RPG Maker MV/MZ 六大引擎的 12 维本地化能力评估矩阵与第三 Adapter 路线决策。
