# 历史文档归档

本目录存放**过程稿、设计 SSOT 与审计/验收快照**，**不是**现行用户手册。

日常使用与开发请从 [文档地图](../README.md) 进入「现行」文档。根目录 [README.md](../../README.md) 负责快速上手。

| 文稿 | 性质 |
|---|---|
| [GUI 信息架构重组计划](gui_ia_redesign.md) | Epic #157 设计与交付计划 SSOT（改造前用语较多） |
| [翻译全生命周期审计](translation_workflow_audit.md) | 流水线与门禁代码审计快照，可能落后于实现 |
| [Design QA 验收记录](design-qa.md) | 2026-07 侧边导航与页面归属的视觉/自动化验收快照 |
| [文译参考对照与 Batch 主路径增强计划](wenyi_reference_and_batch_roadmap.md) | Project Analysis、最终审校与用量账本等已落地路线的设计取舍记录 |
| [润色工具原型设计摘录](prototypes/renpy_polishing_toolkit/README.md) | 脱敏后的设计结论，不保存脚本副本 |
| [#346 实施分步计划](issue-346-implementation-plan.md) | Sync / Batch 共用 TranslationPlan、ContextAssembler 与请求合同全阶段实施与收口记录（PR #403 已合入） |
| [#347 设计 P0：耐久同步执行器](issue-347-durable-sync-executor-plan.md) | Run/Request/Attempt 状态机与耐久同步执行器核心设计（CLI 已交付，交接见 [issue-347-to-348-handoff.md](../plans/issue-347-to-348-handoff.md)） |
| [#341 P0：Embedding 纯核心](issue-341-provider-neutral-embedding-core.md) | Provider-neutral Embedding 纯核心合同与指纹规范（PR #401 已交付） |
| [#341：Embedding adapters 与 store identity](issue-341-embedding-adapters-store-identity.md) | Provider adapter、store identity 与兼容检查落地记录（PR #388 / PR #401 已合入） |
| [TyranoScript V600+ P5 parser 调研与 fixture 基线](tyranoscript_v600_parser_research.md) | 官方 runtime parser/catalog 调研与离线 fixture 规范（P5 已合入主干） |
| [开放 Issues 审计（2026-08-08）](open_issues_audit_2026-08-08.md) | 基于 `main@78ab050` 的 21 个开放 issues 快照与收口顺序审计 |
| [开放 Issues 审计（2026-09-12）](open_issues_audit_2026-09-12.md) | 基于 `main@d164c2b` 的 11 个开放 issues 快照、难度与收口顺序审计；含 #474 竞态复现 |
| [Final Review 失败分类探针（#309）](final_review_result_failure_spike.md) | 模型审校结果失败分类 fixture probe 与 targeted resume 设计 |

维护约定：历史文稿只加状态横幅或修正断链，不整篇改写成新手册，避免与现行文档双重维护。
