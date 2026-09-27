# 规划、合同与研究索引

本目录收纳尚未进入现行用户手册的计划、仍需稳定查阅的技术合同、研究快照和操作记录。状态总览核对至 2026-09-27 的 main@3cbd597；研究文档自己的日期和代码基线仍是其证据范围，不因本索引更新而自动刷新。

以当前代码和现行手册判断用户行为。GitHub issue 的 OPEN 状态不代表没有工程交付；PR 合并或自动化通过也不等于真实 Provider、项目或游戏验收。只归档纯历史过程稿；仍被现行手册、测试、未完实验或后续决策引用的合同和证据留在此目录。

## 本轮开发与验证映射（2026-09-27）

| 项目 | 已交付证据与当前状态 | 剩余跟踪 |
| --- | --- | --- |
| #521 双字符串台词身份修复 | PR [#522](https://github.com/hu-border-collie/renpy-translation-lab/pull/522) 已合并；修复工程交付不等于真实项目样本验收。 | #521 仍 OPEN；真实项目回归按 issue 验收补齐。 |
| #427 普通译文逐条审校 | S1 由 PR [#515](https://github.com/hu-border-collie/renpy-translation-lab/pull/515) / [#517](https://github.com/hu-border-collie/renpy-translation-lab/pull/517)，S2 GUI 与 revision proposal 衔接由 PR [#523](https://github.com/hu-border-collie/renpy-translation-lab/pull/523) 合并；合同见[计划](issue-427-review-index-contract.md)，操作见[GUI 工作台](../gui_workbench.md)。 | #427 仍 OPEN；真实大语料、实际操作与游戏验收按 issue 统一实测。issue 正文阶段说明和验收勾选尚未回填。 |
| #512 复用审校 GUI / #513 质量白名单拆分 | PR [#524](https://github.com/hu-border-collie/renpy-translation-lab/pull/524) / [#527](https://github.com/hu-border-collie/renpy-translation-lab/pull/527) 已合并。GUI 合同见[引擎适配手册](../engine_adapter.md)；新字段与旧字段语义见[Batch 工作流](../batch_workflows.md)。 | 两个 issue 仍 OPEN，原验收清单未回填；状态应由 issue 维护者按 PR 证据复核，不能据 OPEN 判断功能未开发。 |
| #518 with 从句 / #508 外部初译工作包 | #518 的工程修复 PR [#519](https://github.com/hu-border-collie/renpy-translation-lab/pull/519) 已合并；#508 工程工作包 PR [#516](https://github.com/hu-border-collie/renpy-translation-lab/pull/516) 已合并，现行入口见[外部初译工作包](../external_translation_work.md)。 | #518 需对原 issue 记录的真实样本回归；#508 需获准真实章节、独立译文审阅和可用预算。两项仍 OPEN，不以合成 fixture 代替实测。 |
| #344 多 Provider 主路径 / #431 OpenAI-compatible | #431 的 S1–S3 与 smoke 准备 PR [#504](https://github.com/hu-border-collie/renpy-translation-lab/pull/504)–[#507](https://github.com/hu-border-collie/renpy-translation-lab/pull/507) 已合并；当前记录模板见[Provider Smoke Matrix](../provider_smoke_matrix.md)。 | #344、#431 仍 OPEN，待统一真实 Provider / 恢复验收及 #341 承接项；S3 的 final_review Sync 解绑已交付，不再是待开发项。 |
| #470 coverage 真实样本 | 合成归因及 #460–#464、#471 修复已完成，见[归因报告](renpy_coverage_block_attribution.md)与只读脚本。 | #470 仍 OPEN；等待授权真实大型项目副本，未获得样本时记录证据缺口。 |
| 本轮之外：#509 / #520 | 两者都是决定是否继续开发的实验任务，没有生产接入交付。 | 暂不进入本轮；先完成 Translator++ 工程往返或 OpenRouter Batch fixture / 真实可用性与费用实验，再由结论决定是否另立生产任务。 |
| 本轮之外：#272 / #147 | #272 的第三 Adapter 立项已暂缓；#147 是低优先级 Story Graph 试点。 | 不进入本轮，不把研究建议写成实施承诺；分别保留在原 issue 的触发条件与优先级下。 |
| 维护减负路线 | #528 已按 completed 关闭，PR [#529](https://github.com/hu-border-collie/renpy-translation-lab/pull/529) / [#530](https://github.com/hu-border-collie/renpy-translation-lab/pull/530) 合并；[路线图](maintainability_roadmap.md)由文档 PR [#532](https://github.com/hu-border-collie/renpy-translation-lab/pull/532) 建立。 | #531 仍 OPEN、尚未实施；与 #510 文档收尾及上述功能队列互不构成前置条件。候选生命周期、其余 CLI 依赖和全局状态治理未承诺。 |

## 计划、合同与研究目录

| 文档 | 当前用途与状态 | 当前入口或剩余跟踪 |
| --- | --- | --- |
| [维护复杂度治理路线](maintainability_roadmap.md) | #528 第一轮完成；#531 第二轮已登记，尚未实施。 | 实现状态按 #531 更新；其余内容仅为候选。 |
| [七类规划方向重新评估（2026-09-19）](plan_reassessment_2026-09-19.md) | 评估报告由 PR [#514](https://github.com/hu-border-collie/renpy-translation-lab/pull/514) 合并；它是日期快照，不代替本索引的全量状态映射。#512/#513 后由 PR #524/#527 合并，#487 spike 由 PR #511 合并。 | 真实章节、Provider、coverage 实验仍分别跟踪 #508、#344/#431、#470；生产字体接入待评估。 |
| [Agent 翻译工具包提案](agent_translation_toolkit.md) | #508 的可复用工作包与实验 CLI 由 PR #516 交付；整体产品收益未由 fixture 证明。 | #508 真实章节对照仍待做；现行操作见[外部初译工作包](../external_translation_work.md)。 |
| [Engine Adapter P0 合同](engine_adapter_contract.md) | #265 P0–P6 已交付并关闭；本文保留技术设计与安全合同。 | 现行实现见[Engine Adapter 手册](../engine_adapter.md)；第三引擎 #272 暂缓，真实 coverage 样本见 #470。 |
| [GitHub 本地化项目源码研究](github_localization_projects_research.md) | 2026-09-06 的公开源码研究快照，不是兼容性或生产实现证明。 | 当前选型以 #272、#509 及[Translator++ 评估](translatorpp_integration_assessment.md)为准。 |
| [#202 Settings 页面合同](issue-202-settings-page-contract.md) | Phase A–D 已完成，#202 于 2026-09-10 关闭；保留页面所有权和 coordinator 合同。 | 当前界面见[GUI 工作台](../gui_workbench.md)；宿主持有两文件事务是合同边界，不是未完成阶段。 |
| [#347 → #348 产品化交接](issue-347-to-348-handoff.md) | #347 已交付并关闭，#348 P0–P3 已完成并关闭；本文留作耐久执行器接缝记录。 | 真实 Provider / 恢复验收由 #344 与 #431 跟踪。 |
| [#348 Model Routing 合同](issue-348-model-routing-config-contract.md) | P0–P3 已交付，#348 于 2026-09-11 关闭；默认值决策由 #457 单独完成并维持现状。 | 当前迁移见[模型配置迁移](../model_config_migration.md)；真实 smoke 归 #344 / #431。 |
| [#364 校准执行手册](issue-364-calibration-runbook.md) | 真实项目 B 线与 A1–A3 已完成，#364 于 2026-08-23 关闭；保留复现步骤。 | 这是历史校准的操作参考。新规则字段语义见[Batch 工作流](../batch_workflows.md)，不能把旧样本结论重写为新配置实验。 |
| [#422 apply/export 事务合同](issue-422-p2-apply-export-contract.md) | P1 / P2 由 PR #473 交付，#422 已关闭；安全回执、恢复和平台限制仍是有效合同。 | 用户操作见[Agent 快速开始](../quickstart_agent.md)和[Batch 工作流](../batch_workflows.md)。 |
| [#427 普通译文审校合同](issue-427-review-index-contract.md) | S1 / S2 工程由 PR #515、#517、#523 合并；issue 仍等待真实验收。 | GUI 操作见[GUI 工作台](../gui_workbench.md)；真实语料、运行和游戏验收归 #427。 |
| [#431 S1 OpenAI-compatible 合同](issue-431-openai-compatible-contract.md) | S1 由 PR #504 合并；文档保留该阶段原始基线。 | S2/S3 与 smoke 准备见下列合同；真实 Provider 证据归 #431 / #344。 |
| [#431 LiteLLM 迁移矩阵](issue-431-litellm-migration-matrix.md) | S2 研究矩阵随 PR #505 合并；仅记录 LiteLLM 调用点，不代表删除依赖或更改默认路径。 | 当前 Provider 验收步骤见[Smoke Matrix](../provider_smoke_matrix.md)，待 #431 / #344。 |
| [#431 S2 目录发现与诊断合同](issue-431-openai-compatible-s2-contract.md) | S2 由 PR #505 合并；保留以 S1 为基础的阶段合同。 | 真实 Provider 与取消行为验收仍归 #431 / #344。 |
| [#431 S3 final_review Sync 合同](issue-431-final-review-sync-contract.md) | S3 由 PR #506 合并；解绑 gemini_batch 已交付，Batch 默认路径不变。 | 仅真实 Provider smoke 仍待 #431 / #344；不是开发待办。 |
| [#488 翻译预检合同](issue-488-preflight-summary-contract.md) | S1、S1.5、S2、S3 均已交付，#488 已关闭；合同保留 freshness / unknown 语义。 | 当前预检用法见[Batch 工作流](../batch_workflows.md)及[Provider Smoke Matrix](../provider_smoke_matrix.md)。 |
| [多引擎生态研究](multi_engine_localization_ecosystem_research.md) | 2026-09-06 研究及 2026-09-16 专项补充；不代表候选引擎已支持。 | #265 已关闭，#272 暂缓；Translator++ 先做 #509 往返实验。 |
| [质量规则校准基线](quality_calibration_baseline.md) | #364 的历史真实项目校准结果；形成于 #513 字段拆分前，原数据与结论保持原样。 | 当前语言允许与排版豁免字段见[Batch 工作流](../batch_workflows.md)。 |
| [#426 coverage block 归因报告](renpy_coverage_block_attribution.md) | 合成归因与 #460–#464、#471 修复已完成；#426 已关闭。 | 真实大型项目频率归因仍由 #470 跟踪。 |
| [#487 字体覆盖只读 spike](renpy_font_coverage_spike.md) | PR #511 已交付，#487 已关闭；不等于 doctor / GUI 生产接入。 | 字体生产接入仍待评估，未作为本轮承诺。 |
| [Translator++ 接入评估](translatorpp_integration_assessment.md) | 2026-09-16 的外部格式与合同缺口快照；尚无 bridge 或真实往返证据。 | #509 先决定是否继续，完成前不安排生产接入。 |
| [视觉小说引擎能力矩阵](visual_novel_localization_matrix.md) | 2026-09-13 官方版本/语法核对，9 月 14 日记录 #272 暂缓决定。 | #272 保持原触发条件；不把候选引擎扩成承诺。 |

本次未移动文件：保留项都仍提供当前合同、实验前置证据、决策理由或操作参考；没有仅为凑数量而归档。已归档的历史材料见[归档索引](../archive/README.md)。

## 已交付并归档的设计与调研

已完成且仅作过程回顾的文稿位于 [`docs/archive/`](../archive/README.md)。归档不代表其中引用的运行时安全合同失效；用户用法与当前实现仍以现行手册为准。

- [#346 实施分步计划](../archive/issue-346-implementation-plan.md)：Sync / Batch 共用 TranslationPlan、ContextAssembler 与请求合同（PR #403）。
- [#347 耐久同步执行器设计](../archive/issue-347-durable-sync-executor-plan.md)：Run / Request / Attempt 状态机核心设计。
- [#341 Provider-neutral Embedding 纯核心](../archive/issue-341-provider-neutral-embedding-core.md) · [Embedding adapters 与 store identity](../archive/issue-341-embedding-adapters-store-identity.md)。
- [TyranoScript V600+ parser 研究](../archive/tyranoscript_v600_parser_research.md)：P5 调研与 fixture 基线。
- [开放 Issues 审计（2026-08-08）](../archive/open_issues_audit_2026-08-08.md) · [2026-09-12](../archive/open_issues_audit_2026-09-12.md) · [2026-09-14](../archive/open_issues_audit_2026-09-14.md)：均为指定基线的历史快照。
- [Final Review 失败分类探针](../archive/final_review_result_failure_spike.md)：模型审校失败分类与 targeted resume 设计。

## 现行文档

稳定的用户操作、安全边界与当前界面以[文档地图](../README.md)列出的现行手册为准；新增能力先更新现行手册，再将已完成且仅具历史价值的计划归档。
