# 规划、合同与研究索引

本目录收纳尚未进入现行用户手册的计划、仍需稳定查阅的技术合同、研究快照和操作记录。状态总览核对至 2026-09-27 的 main@3cbd597；#531 维护第二轮状态于 2026-09-29 按 main@e792a03 和 issue 关闭记录单独更新。研究文档自己的日期和代码基线仍是其证据范围，不因本索引更新而自动刷新。

以当前代码和现行手册判断用户行为。GitHub issue 的 OPEN 状态不代表没有工程交付；PR 合并或自动化通过也不等于真实 Provider、项目或游戏验收。只归档纯历史过程稿；仍被现行手册、测试、未完实验或后续决策引用的合同和证据留在此目录。

## 历史开发与验证映射（2026-09-27）

以下 OPEN / 待执行描述只代表当日状态。2026-10-13 的相关实测补证与剩余验收见
[Provider 验收补证](provider-acceptance-reassessment-2026-10-13.md)：Gemini Batch / Sync、
DS 直连五阶段及中断恢复已有真实结果；#508 已做真实 A/B1 和章节交付。
截至 2026-10-16，开放 issue 为 #147、#272、#344、#508、#509、#520；#431 已按首期范围关闭。
其余旧表项不再据此标为开放。

2026-10-15 的[续验与修复](issue-431-508-acceptance-2026-10-15.md)已补双地址/凭据隔离、
无 Gemini Key 的直连五阶段、552 项独立 Agent 复审与单项订正后的原生引擎验证。
PR #556 / #557 已合并，#431 已完成收口；#344 / #508 的其余验收继续保持开放。
10-16 的[#344 固定版本矩阵](issue-344-provider-matrix-2026-10-16.md)已在连接恢复后补齐
Gemini Sync / Batch，四类执行链路均通过检查与隔离写回；首次失败保留，Epic 的其他验收
继续跟踪。Sync 用量状态显示差异已于 10-17 修复并离线回放验证，随后提交 PR 评审。
10-17 的[#344 章节续验](issue-344-context-acceptance-2026-10-17.md)已补真实中断与请求级费用、
fresh PA、非 Gemini embedding 和 128 对原生上下文实验；语义收益未获支持，默认不变。
修复已通过完整 CLI / GUI，代码与证据进入 PR 评审；线上 Epic 保持开放。

10-17～20 的[#508 串行 / 并行续验](issue-508-parallel-acceptance-2026-10-17.md)完成同条件
新 B1/B2 各 552 项候选、独立匿名审阅、明确订正与隔离写回；新 B2 完成原生整章执行。
旧 A/B1 评分保持不变。C 经明确授权恢复，原参数 87 项与压缩版 390 项候选分别审阅，
先交付 488 项，再经新的明确授权对旧未知 64 项各重试一次，单独审阅订正后完整交付
混合路线 552 项并完成原生重放。旧未知费用仍保留预留；本轮按原清单提交收缩结论，
桌面扩展检查不作为新增前置条件，#508 当前保持开放待收尾。

| 项目 | 已交付证据与当前状态 | 剩余跟踪 |
| --- | --- | --- |
| #521 双字符串台词身份修复 | PR [#522](https://github.com/hu-border-collie/renpy-translation-lab/pull/522) 已合并；修复工程交付不等于真实项目样本验收。 | #521 仍 OPEN；真实项目回归按 issue 验收补齐。 |
| #427 普通译文逐条审校 | S1 由 PR [#515](https://github.com/hu-border-collie/renpy-translation-lab/pull/515) / [#517](https://github.com/hu-border-collie/renpy-translation-lab/pull/517)，S2 GUI 与 revision proposal 衔接由 PR [#523](https://github.com/hu-border-collie/renpy-translation-lab/pull/523) 合并；合同见[计划](issue-427-review-index-contract.md)，操作见[GUI 工作台](../gui_workbench.md)。 | #427 仍 OPEN；真实大语料、实际操作与游戏验收按 issue 统一实测。issue 正文阶段说明和验收勾选尚未回填。 |
| #512 复用审校 GUI / #513 质量白名单拆分 | PR [#524](https://github.com/hu-border-collie/renpy-translation-lab/pull/524) / [#527](https://github.com/hu-border-collie/renpy-translation-lab/pull/527) 已合并。GUI 合同见[引擎适配手册](../engine_adapter.md)；新字段与旧字段语义见[Batch 工作流](../batch_workflows.md)。 | 两个 issue 仍 OPEN，原验收清单未回填；状态应由 issue 维护者按 PR 证据复核，不能据 OPEN 判断功能未开发。 |
| #518 with 从句 / #508 外部初译工作包 | #518 的工程修复 PR [#519](https://github.com/hu-border-collie/renpy-translation-lab/pull/519) 已合并；#508 工程工作包 PR [#516](https://github.com/hu-border-collie/renpy-translation-lab/pull/516) 已合并，现行入口见[外部初译工作包](../external_translation_work.md)。 | #518 需对原 issue 记录的真实样本回归；#508 已有 B1/B2 章节、C 混合恢复交付及运行证据，按原清单与收缩结论收尾。两项当前仍 OPEN，不以合成 fixture 代替实测。 |
| #344 多 Provider 主路径 / #431 OpenAI-compatible | #431 的 S1–S3 与 smoke 准备 PR [#504](https://github.com/hu-border-collie/renpy-translation-lab/pull/504)–[#507](https://github.com/hu-border-collie/renpy-translation-lab/pull/507) 已合并；当前记录模板见[Provider Smoke Matrix](../provider_smoke_matrix.md)。 | #344、#431 仍 OPEN，待统一真实 Provider / 恢复验收及 #341 承接项；S3 的 final_review Sync 解绑已交付，不再是待开发项。 |
| #470 coverage 真实样本 | 合成归因及 #460–#464、#471 修复已完成，见[归因报告](renpy_coverage_block_attribution.md)与只读脚本。 | #470 仍 OPEN；等待授权真实大型项目副本，未获得样本时记录证据缺口。 |
| 本轮之外：#509 / #520 | 两者都是决定是否继续开发的实验任务，没有生产接入交付。 | 暂不进入本轮；先完成 Translator++ 工程往返或 OpenRouter Batch fixture / 真实可用性与费用实验，再由结论决定是否另立生产任务。 |
| 本轮之外：#272 / #147 | #272 的第三 Adapter 立项已暂缓；#147 是低优先级 Story Graph 试点。 | 不进入本轮，不把研究建议写成实施承诺；分别保留在原 issue 的触发条件与优先级下。 |
| 维护减负路线 | #528、#531 均按 completed 关闭；第一轮 PR [#529](https://github.com/hu-border-collie/renpy-translation-lab/pull/529) / [#530](https://github.com/hu-border-collie/renpy-translation-lab/pull/530)，第二轮 PR [#534](https://github.com/hu-border-collie/renpy-translation-lab/pull/534) 均已合并；范围与证据见[路线图](maintainability_roadmap.md)。 | 真实 Provider / 恢复验收仍归 #344 / #431；候选生命周期、其余 CLI 依赖和全局状态治理未承诺，不成为既有任务的关闭前置条件。 |

## 计划、合同与研究目录

2026-10-23 的 [#565 Komorebi 离线调查](issue-565-komorebi-survey-2026-10-23.md)
已冻结实际 MV 包、记录 core 头部 1.6.2 / 常量 1.6.1 冲突，完成 205 份 JSON 的范围分类，
建立三个候选场景的 101 项主要 occurrence / 7 个镜像目标与 20 项宿主 Agent 静态核对。
原生启动授权超时后，用户选择先完成离线调查；可达性、字体显示、存读档和人类验收未完成。
公共数组路径的 P1 核心实现见下，Adapter / 耐久 Sync 产品接入另拆；线上回填草稿尚未发送，
#272 / #565 均不能据本报告自动关闭。

用户随后授权实施的 P1 [公共源 JSON 核心合同](../engine_adapter.md#source-json-writeback-p1)
已在本地增加 schema 2、独立源 / 目标快照、类型化数组路径与字符串 token 渲染，
用[原创对白 / 选项 / 402 夹具](../../tests/fixtures/source_json_writeback/README.md)验证。
这是共享核心增量，P2 Adapter / P3 CLI、GUI、doctor 和配置接线及原生游戏验收仍待实施。

2026-10-22 的 [#272 研究归档与 #565 调查衔接](issue-272-research-closeout-2026-10-22.md)
保留候选合同判断、#509 正反证据与未完成验收。新 [#565 Komorebi 调查](https://github.com/hu-border-collie/renpy-translation-lab/issues/565)
已登记真实需求，先确认版本、插件/文本范围及最小试点，再回填 #272；生产实现待调查后另拆。
#272 / #565 保持 open，#147 维持低优先级；本轮文档合并不关闭 issue，不勾选未完成验收。
下方旧表/日期维持各自历史范围，不沿用此前按 `not_planned` 关闭 #272 的草案。

2026-10-21～22 的 [#509 Translator++ 单版本实测](issue-509-translatorpp-roundtrip-2026-10-21.md)
固定 MV 编辑器/core 1.5.1、TPP 8.8.9 / 实际 RMMVJS 2.19（新建入口3.3）。限定译文导出精确19处，
25参数独立对账、部分原生/HTML5运行及staging恢复重试已有证据；无修改导出存在插件注释副作用，
前插命令后丢失一处context译文。独立null保存/重开/导出补测已通过，限定spike证据已齐；
决定保留独立工具使用、生产bridge暂缓；PR #562 已合并，#509 已关闭。

2026-10-20 的 [#520 OpenRouter Batch 有界验证](issue-520-openrouter-batch-spike-2026-10-20.md)
已完成首轮研究并决定暂缓生产接入；后续 [DeepSeek v2 续验](issue-520-openrouter-batch-deepseek-2026-10-20.md)
已在 10-21 读回真实 Batch 2/2 请求、6/6 item 成功结果，Sync 同样成功；最终暂缓生产接入，
费用/保留和完整同口径比较保留缺口。[合同 v1](issue-520-openrouter-batch-contract-v1.md)
与原创离线 harness 保留。首轮 GPT 路线实际被拒，Gemini 对照完成但内容合同失败；
这些负结果不改变下表各历史快照的原始验收状态。

| 文档 | 当前用途与状态 | 当前入口或剩余跟踪 |
| --- | --- | --- |
| [#520 OpenRouter Batch 研究](issue-520-openrouter-batch-spike-2026-10-20.md) · [合同 v1](issue-520-openrouter-batch-contract-v1.md) | 首轮 2026-10-20 的目录/实际提交冲突、Sync TOS 拒绝、Gemini 对照和费用未知；后续 [DeepSeek 终态](issue-520-openrouter-batch-deepseek-2026-10-20.md)已真实完成 Batch/Sync、6/6 item 关联和费用比较。 | 最终暂缓生产接入；保留上游保留/清理和完整同口径比较缺口，研究 issue 按 not_planned 收口，不注册新执行策略。 |
| [#508 串行 / 并行章节续验（2026-10-17～20）](issue-508-parallel-acceptance-2026-10-17.md) | 新 B1/B2 各 552 项及独立匿名审阅、订正交付；B2 原生执行 552/552。C 原参数、压缩版与另获授权的 64 项重试分别审阅，混合路线完整交付和原生重放 552 项；锁定环境修复两处恢复故障。 | 旧未知费用保留预留；变体限制架构因果和效率结论。Computer Use 与扩展桌面检查不是本轮收尾前置，#508 当前保持开放待收尾。 |
| [#344 真实章节上下文与恢复验收（2026-10-17）](issue-344-context-acceptance-2026-10-17.md) | 两个真实中断边界与请求级费用、fresh published PA、Qwen3 embedding 预建 / 查询、128 对原生上下文及隔离写回均已记录；语义收益未获支持。 | Token 摘要、bootstrap 凭据与向量编码修复提交 PR 评审；合并及记录回填前 #344 保持开放，不改变默认开关。 |
| [#344 固定版本 Provider 矩阵（2026-10-16）](issue-344-provider-matrix-2026-10-16.md) | 固定 main 的四类真实 CLI smoke、相同 prompt 指纹、检查与隔离写回全部完成；后续实验与修复见 10-17 报告。 | 保留首次失败及固定发布基线，区别于含本地修复的章节续验。 |
| [#431 / #508 续验（2026-10-15）](issue-431-508-acceptance-2026-10-15.md) | 双连接/凭据隔离、无 Gemini Key 五阶段、独立 552 项复审与订正后引擎验证。 | 修复随 #557 合并，#431 已关闭；#508 仍缺 B2/C 和剩余视觉/交互，#344 外部验收保持未完成。 |
| [Provider 验收补证（2026-10-13）](provider-acceptance-reassessment-2026-10-13.md) | Gemini Sync、DS CLI/GUI 五阶段、真实中断恢复与三处接线修复的历史快照。 | 该批修复随 #556 合并；双连接和 #508 独立复审已续验，纯直连凭据的新缺陷与剩余边界见 10-15 记录。 |
| [翻译模型、上下文与分块重新评估（2026-10-11）](translation_model_context_reassessment_2026-10-11.md) | 分块失败归因、上下文与 Gemini/DeepSeek 控制实验的日期快照；含 2026-10-12 JSON 字段/思考成本对照，私有制品不入库。 | 语义审校、覆盖率和总成本分别评价；不据结构通过率自动修改生产默认值。 |
| [维护复杂度治理路线](maintainability_roadmap.md) | #528 第一轮与 #531 第二轮工程均已合并并关闭。 | 真实 Provider / 项目验收仍归 #344 / #431；其余维护方向仅为候选。 |
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
| [多引擎生态研究](multi_engine_localization_ecosystem_research.md) | 历史生态研究；10-22 有限修正身份、工具链与候选排名表述，未全量重查。 | #509 已有单版本有限实测；独立工具保留，bridge 暂缓。 |
| [质量规则校准基线](quality_calibration_baseline.md) | #364 的历史真实项目校准结果；形成于 #513 字段拆分前，原数据与结论保持原样。 | 当前语言允许与排版豁免字段见[Batch 工作流](../batch_workflows.md)。 |
| [#426 coverage block 归因报告](renpy_coverage_block_attribution.md) | 合成归因与 #460–#464、#471 修复已完成；#426 已关闭。 | 真实大型项目频率归因仍由 #470 跟踪。 |
| [#487 字体覆盖只读 spike](renpy_font_coverage_spike.md) | PR #511 已交付，#487 已关闭；不等于 doctor / GUI 生产接入。 | 字体生产接入仍待评估，未作为本轮承诺。 |
| [Translator++ 接入评估](translatorpp_integration_assessment.md) | 2026-09-16 研究快照，补充 #509 单版本真实导出及迁移缺口；尚无生产 bridge。 | 本轮决定保留独立使用、bridge暂缓；不提升引擎支持范围。 |
| [视觉小说引擎能力矩阵](visual_novel_localization_matrix.md) | 历史六引擎矩阵；10-22 有限复核 Naninovel、MV/MZ 与 #509，不预定第三引擎排名。 | #565 真实游戏调查回填 #272，生产实现另拆；不从 #509 继承 Komorebi 的版本或支持结论。 |

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
