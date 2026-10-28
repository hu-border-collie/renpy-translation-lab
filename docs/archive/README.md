# 历史资料

这里保存已收口实验、路线研究、审计和旧设计。日常操作从[文档地图](../README.md)开始，
当前任务与有效合同见[计划索引](../plans/README.md)。
各报告保留原日期、代码基线、失败和未验边界；文中的旧 OPEN、待合并或模型排名不表示当前状态。

## 已收口实验

| 记录 | 查什么 |
| --- | --- |
| [#344 四类 Provider 矩阵](experiments/issue-344-provider-matrix-2026-10-16.md) | 固定版本 smoke、首次失败与恢复后的结果 |
| [#344 上下文与恢复](experiments/issue-344-context-acceptance-2026-10-17.md) | 真实中断、费用、fresh PA、embedding 和限定上下文 A/B |
| [#431／#508 续验](experiments/issue-431-508-acceptance-2026-10-15.md) | 双连接／凭据隔离与章节复审 |
| [#508 串行／并行章节](experiments/issue-508-parallel-acceptance-2026-10-17.md) | 552 项、混合恢复、Agent 审阅和收缩结论 |
| [#509 Translator++ 往返](experiments/issue-509-translatorpp-roundtrip-2026-10-21.md) · [原创材料](../plans/issue-509-materials/README.md) | 单版本导出、context 漂移丢译与非目标副作用；生产 bridge 暂缓 |
| [#520 首轮](experiments/issue-520-openrouter-batch-spike-2026-10-20.md) · [研究合同](experiments/issue-520-openrouter-batch-contract-v1.md) · [DeepSeek 续验](experiments/issue-520-openrouter-batch-deepseek-2026-10-20.md) | 保留各轮不同条件与负结果；最终暂缓生产接入 |
| [10-13 Provider 补证](experiments/provider-acceptance-reassessment-2026-10-13.md) | CLI／GUI 五阶段、真实恢复及后续修复线索 |
| [模型、上下文与分块重评](experiments/translation_model_context_reassessment_2026-10-11.md) | 七轮实验、语义与结构差异、失败及费用边界 |

## 路线研究

| 文稿 | 原范围与后续 |
| --- | --- |
| [Agent 工具包提案](research/agent_translation_toolkit.md) | 原始方向；已采纳子集的操作见[外部工作包](../external_translation_work.md) |
| [GitHub 本地化工具源码研究](research/github_localization_projects_research.md) | 9-06 的公开源码快照；建议已有部分交付，不能整篇当作待办 |
| [多引擎生态研究](research/multi_engine_localization_ecosystem_research.md) | 原生态比较及有限续验，其他引擎未全量刷新 |
| [视觉小说引擎能力矩阵](research/visual_novel_localization_matrix.md) | 历史候选边界与有限复核，旧排名不作当前支持承诺 |
| [Translator++ 接入评估](research/translatorpp_integration_assessment.md) | 9-16 研究，后续由 #509 实验收缩为独立工具使用 |
| [#272 有限研究与调查衔接](research/issue-272-research-closeout-2026-10-22.md) | 10-22 的路线判断和线上回填草案，后续 MV 实施归 #567 |
| [#431 LiteLLM 迁移矩阵](research/issue-431-litellm-migration-matrix.md) | 当时调用点与迁移边界；不是当前所有调用点的持续清单 |

## 历次盘点与审计

| 记录 | 当时范围 |
| --- | --- |
| [计划盘点与候选重评（10-27）](audits/plan_inventory_2026-10-27.md) | 39 份文档、旧建议取舍与 80 项针对性回归；当前状态仍由[计划索引](../plans/README.md)维护 |
| [七类方向重评（09-19）](audits/plan_reassessment_2026-09-19.md) | 当时立项 #512／#513，其后已交付；其他建议已有后续取舍 |
| [开放 issues 审计（09-14）](open_issues_audit_2026-09-14.md) | 9 个开放 issue 快照 |
| [开放 issues 审计（09-12）](open_issues_audit_2026-09-12.md) | 11 个开放 issue 快照及竞态复现 |
| [开放 issues 审计（08-08）](open_issues_audit_2026-08-08.md) | 21 个开放 issue 快照 |

## 旧设计与调查

| 文稿 | 保留用途 |
| --- | --- |
| [GUI 信息架构重组](gui_ia_redesign.md) | #157 的设计与交付过程 |
| [翻译全生命周期审计](translation_workflow_audit.md) | 流水线与门禁的旧代码快照 |
| [Design QA](design-qa.md) | 2026-07 的视觉与自动化验收 |
| [文译参考与 Batch 路线](wenyi_reference_and_batch_roadmap.md) | PA、终审、用量账本等早期设计取舍 |
| [润色工具原型](prototypes/renpy_polishing_toolkit/README.md) | 脱敏设计结论 |
| [#346 实施计划](issue-346-implementation-plan.md) | 共用 TranslationPlan、ContextAssembler 与请求合同的实施过程 |
| [#347 耐久执行器设计](issue-347-durable-sync-executor-plan.md) | Run／Request／Attempt 原始设计；现行接缝见[交接合同](../plans/issue-347-to-348-handoff.md) |
| [#341 Embedding 纯核心](issue-341-provider-neutral-embedding-core.md) | 纯核心与指纹规范的形成过程 |
| [#341 adapters 与 store identity](issue-341-embedding-adapters-store-identity.md) | Provider adapter 与兼容检查落地记录 |
| [TyranoScript parser 调查](tyranoscript_v600_parser_research.md) | 固定版本 runtime／catalog 与离线 fixture 基线 |
| [Final Review 失败分类探针](final_review_result_failure_spike.md) | #309 失败分类与 targeted resume 的设计记录 |
| [旧项目说明](project_notes.md) | 归档前的支持边界与早期实验摘要；当前范围并入[首页](../../README.md#支持范围) |
| [GUI 沙箱故障调查](gui_tests_sandbox_notes.md) | 2026-08 的权限故障与修复过程；当前处理并入[CI](../ci.md#沙箱与临时目录) |

归档仅调整文稿位置和引用，不改写历史成功／失败、费用、审阅者身份或未验范围。
技术合同和冻结材料若仍被实现、测试或复现使用，保留在原路径；
普通功能与状态更新优先修改已有手册和计划条目，避免继续新增重复汇报文档。
