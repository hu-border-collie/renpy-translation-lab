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
- [视觉小说引擎本地化能力矩阵与后续 Adapter 路线](visual_novel_localization_matrix.md)：
  #272 针对 Naninovel、Godot+Dialogic 2、Visual Novel Maker、Monogatari、
  KiriKiri/KAG、RPG Maker MV/MZ 六大引擎的 12 维本地化能力评估矩阵与第三 Adapter 路线决策。
