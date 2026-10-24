# 规划、合同与研究索引

本目录保存技术合同、方向研究和有日期的实验记录。用户操作从[文档地图](../README.md)
进入现行手册；本页集中维护任务状态和证据入口。

**状态核对：2026-10-24，合并基线 `main@4aa751c`；A2 开发由 #570 跟踪。**
已读回相关 GitHub issue 并核对主线合并历史；本分支 A2 增量尚未合并。报告自己的日期、代码基线和原始结果
保持原样；旧报告里的 OPEN、待验或本地草稿描述属于当时状态。

## 当前开放工作

当前开放 issue 为 #147、#272、#565、#567、#570。工程合并、离线 fixture、原生运行和译文质量
分别验收，不能相互代替。

| 跟踪 | 当前交付与下一步 |
| --- | --- |
| [#567 RPG Maker MV 与 Komorebi 交付](https://github.com/hu-border-collie/renpy-translation-lab/issues/567) | 统领 MV 首期正式支持与完整游戏汉化两个里程碑。A1 公共 JSON 核心已由 [PR #568](https://github.com/hu-border-collie/renpy-translation-lab/pull/568) 合并；正文已回填，A2 由 #570 承接。A3 耐久 Sync / CLI / GUI / doctor / 配置及真实游戏验收仍待完成，不能宣称 MV 已支持。 |
| [#570 A2 MV 叙事 Adapter 与结构保护](https://github.com/hu-border-collie/renpy-translation-lab/issues/570) | [共享核心](../engine_adapter.md#rpgmaker-mv-a2)与[原创离线材料](../../tests/fixtures/rpgmaker_mv_narrative/README.md)覆盖显式材料窗口、物理 occurrence、独立 coverage、消息 / 选项 / 镜像和 schema 2 计划。合并与原生验收分别核对；A3 产品接入、A4 真实小样本不在本子任务。 |
| [#565 Komorebi 调查](https://github.com/hu-border-collie/renpy-translation-lab/issues/565) | [离线报告](issue-565-komorebi-survey-2026-10-23.md)与[公开材料说明](issue-565-materials/README.md)已入主线：205 份 JSON、101 个试点 occurrence 和 7 个镜像目标；core 版本声明冲突仍保留。原生可达性、字体显示、存读档及 issue 验收回填尚未完成。 |
| [#272 多引擎研究与选型](https://github.com/hu-border-collie/renpy-translation-lab/issues/272) | [有限研究与调查衔接](issue-272-research-closeout-2026-10-22.md)保留研究范围；#565 提供真实 MV 需求，#567 承接生产实现。其他引擎未全量重查，不按旧评级自动排期或承诺支持。 |
| [#147 Story Graph 试点](https://github.com/hu-border-collie/renpy-translation-lab/issues/147) | 保持低优先级；仅在实际关系、称呼或场景问题超出现有 glossary / source index 能力时评估，不作为 #567 前置条件。 |

公共 JSON 的现行边界见[源 JSON 核心 API](../engine_adapter.md#source-json-writeback-p1)与
[原创夹具](../../tests/fixtures/source_json_writeback/README.md)：schema 2、独立源 / 目标
快照、类型化数组路径和字符串 token 渲染已交付；它没有新增可对游戏运行的产品入口。

## 已关闭任务与结果边界

以下按 GitHub 当前状态列出，替代旧索引中相互交错的历史待办。关闭仅对应原 issue 的
限定验收；未测范围、unknown 费用和未采纳方向仍按原记录保留。

| 任务 | 已核对结果与证据 |
| --- | --- |
| [#344 多 Provider 主路径](https://github.com/hu-border-collie/renpy-translation-lab/issues/344)、[#431 直连协议](https://github.com/hu-border-collie/renpy-translation-lab/issues/431) | 均按 completed 关闭。四类固定版本矩阵、五阶段直连、真实中断 / 请求费用和上下文实验分别留档；修复 PR #556 / #557 / #558 已合并。现有数据不支持默认开启高级上下文。操作见 [Provider Smoke Matrix](../provider_smoke_matrix.md)。 |
| [#508 外部初译工作包](https://github.com/hu-border-collie/renpy-translation-lab/issues/508) | 按 completed 关闭，核心 PR #516 与恢复修复 PR #559 已合并。B1/B2、C 混合恢复、独立匿名 Agent 审阅及 552 项交付见[章节续验](issue-508-parallel-acceptance-2026-10-17.md)。结论收缩到范围核验、持久候选、恢复与安全交付；不据此宣称质量或效率优势。 |
| [#509 Translator++ 往返](https://github.com/hu-border-collie/renpy-translation-lab/issues/509) | PR #562 已合并，按 completed 关闭；单版本限定实测完成，保留独立工具使用，生产 bridge 暂缓。 |
| [#520 OpenRouter Batch](https://github.com/hu-border-collie/renpy-translation-lab/issues/520) | 按 not_planned 关闭。DeepSeek Sync / Batch 真实完成，首轮 GPT 被拒和 Gemini 内容合同失败均保留；最终暂缓生产接入，不新增执行策略。 |
| [#427 普通审校](https://github.com/hu-border-collie/renpy-translation-lab/issues/427)、[#512 复用审校 GUI](https://github.com/hu-border-collie/renpy-translation-lab/issues/512)、[#513 质量字段拆分](https://github.com/hu-border-collie/renpy-translation-lab/issues/513) | 均按 completed 关闭；工程 PR #515 / #517 / #523 / #524 / #527 已合并。当前操作与语义分别见[GUI 工作台](../gui_workbench.md)、[Engine Adapter](../engine_adapter.md)和[Batch 工作流](../batch_workflows.md)。 |
| [#518 with 从句](https://github.com/hu-border-collie/renpy-translation-lab/issues/518)、[#521 双字符串身份](https://github.com/hu-border-collie/renpy-translation-lab/issues/521)、[#470 真实 coverage](https://github.com/hu-border-collie/renpy-translation-lab/issues/470) | 均按 completed 关闭。#470 保留 12,686 个候选的只读归因和独立漏扫核对；4 个结构 / tokenizer 区域的精确引擎语义仍为 unknown，coverage block 未放宽。 |
| [#528](https://github.com/hu-border-collie/renpy-translation-lab/issues/528)、[#531 维护减负](https://github.com/hu-border-collie/renpy-translation-lab/issues/531) | 两轮均按 completed 关闭，PR #529 / #530 / #534 已合并；其余方向仅为候选，见[维护路线](maintainability_roadmap.md)。 |
| [#538–#541 GUI 改善](https://github.com/hu-border-collie/renpy-translation-lab/issues/538) | 四项均按 completed 关闭；当前导航、关键词候选、设置与诊断交互见[GUI 工作台](../gui_workbench.md)，后续目录迁移由 PR #563 / #564 合并。 |

## 技术合同与复现手册

这些文档仍供实现和测试查阅。已交付阶段的原设计不能代替现行用户说明，也不意味着
文中的旧 target、阶段顺序或当时待办仍未完成。

| 文档 | 保留用途与现行入口 |
| --- | --- |
| [Engine Adapter P0 合同](engine_adapter_contract.md) | #265 P0–P6 已交付并关闭；Ren'Py 产品路径与 Tyrano 核心 / 离线范围见[现行手册](../engine_adapter.md)。MV 后续归 #567。 |
| [#202 Settings 页面合同](issue-202-settings-page-contract.md) | Phase A–D 已完成；保留页面、coordinator 与宿主写盘的所有权合同。当前分层见[架构](../architecture.md)。 |
| [#347 → #348 交接](issue-347-to-348-handoff.md) | 两项已完成；保留耐久执行器接缝，操作见[同步工作流](../sync_workflow.md)。 |
| [#348 Model Routing 合同](issue-348-model-routing-config-contract.md) | P0–P3 已交付；#457 默认值决策维持现状，迁移见[现行说明](../model_config_migration.md)。 |
| [#422 apply / export 事务合同](issue-422-p2-apply-export-contract.md) | PR #473 已交付；回执、事务、恢复与平台边界继续有效，操作见[Batch 工作流](../batch_workflows.md)。 |
| [#427 普通译文审校合同](issue-427-review-index-contract.md) | 保留 index、决定日志、提案与 needs_recheck 合同；当前 GUI 见[工作台](../gui_workbench.md)。 |
| [#431 S1 直连生成](issue-431-openai-compatible-contract.md) | PR #504 已交付；能力、凭据与结构化输出合同。 |
| [#431 S2 目录发现与诊断](issue-431-openai-compatible-s2-contract.md) | PR #505 已交付；目录失败、机器码、脱敏与取消边界。 |
| [#431 S3 final_review Sync](issue-431-final-review-sync-contract.md) | PR #506 已交付；Sync 解绑已实现，未显式配置时保留 legacy Batch 默认。 |
| [#488 翻译预检合同](issue-488-preflight-summary-contract.md) | S1–S3 已交付并关闭；保留 freshness / unknown 语义，操作见[Batch 工作流](../batch_workflows.md)。 |

## 实验与验收记录

按时间和实验范围查证据，后续结果通过相应续验记录衔接。原始失败、不同参数的变体、
费用未知和未执行的桌面 / 人类验收均不补写为通过。

| 文档 | 证据范围 |
| --- | --- |
| [#509 Translator++ 单版本往返（10-21～22）](issue-509-translatorpp-roundtrip-2026-10-21.md) · [原创材料](issue-509-materials/README.md) | MV 编辑器 / core 1.5.1、TPP 8.8.9、实际 RMMVJS 2.19；19 处限定译文、25 参数对账、漂移丢译与非目标副作用。不能沿用为 Komorebi 版本或支持证明。 |
| [#520 OpenRouter Batch 首轮（10-20）](issue-520-openrouter-batch-spike-2026-10-20.md) · [合同 v1](issue-520-openrouter-batch-contract-v1.md) · [DeepSeek 续验](issue-520-openrouter-batch-deepseek-2026-10-20.md) | 首轮负结果与后续 Batch 2/2 请求、6/6 item 成功分开；保留费用 / 清理和完整同口径比较缺口。 |
| [#508 串行 / 并行章节续验（10-17～20）](issue-508-parallel-acceptance-2026-10-17.md) | 新 B1/B2、C 各来源批次与混合交付分别审阅；552 项原生重放、恢复修复、未知费用和结论限制。 |
| [#344 章节上下文与恢复（10-17）](issue-344-context-acceptance-2026-10-17.md) | 真实中断、请求级费用、fresh published PA、Qwen3 embedding 和两组各 128 条原生上下文实验。 |
| [#344 固定版本 Provider 矩阵（10-16）](issue-344-provider-matrix-2026-10-16.md) | 固定 main 的四类真实 CLI smoke、prompt 指纹、check 与隔离 apply，保留首次失败。 |
| [#431 / #508 续验（10-15）](issue-431-508-acceptance-2026-10-15.md) | 双连接 / 凭据隔离、无 Gemini Key 五阶段、独立章节复审与订正后引擎验证；后续 B2/C 见 10-17～20。 |
| [Provider 验收补证（10-13）](provider-acceptance-reassessment-2026-10-13.md) | Gemini Sync、DS CLI / GUI 五阶段、真实恢复和接线修复的历史快照。 |
| [翻译模型、上下文与分块重评（10-11）](translation_model_context_reassessment_2026-10-11.md) | 分块失败归因、模型与上下文对照、JSON 字段及思考成本；结构通过率不等于语义收益。 |

## 方向研究、校准与只读工具

| 文档 | 阅读边界 |
| --- | --- |
| [#272 有限研究与 #565 衔接（10-22）](issue-272-research-closeout-2026-10-22.md) | 当时的研究决定与线上回填草案；后续实施归 #567，未完成项不因报告合并自动完成。 |
| [Agent 翻译工具包提案](agent_translation_toolkit.md) | 历史方向与实验设计；已实现子集见[外部工作包](../external_translation_work.md)，#508 已按收缩结论关闭。 |
| [七类规划方向重评（09-19）](plan_reassessment_2026-09-19.md) | PR #514 的日期快照；后续 #512 / #513 与 #487 已交付，本页维护当前映射。 |
| [维护复杂度路线](maintainability_roadmap.md) | #528 / #531 两轮已完成；未承诺的后续方向不作为旧任务关闭前置。 |
| [GitHub 本地化项目源码研究](github_localization_projects_research.md) | 2026-09-06 公开源码快照，不是生产兼容性证明。 |
| [多引擎生态研究](multi_engine_localization_ecosystem_research.md) | 历史研究，10-22 有限修正；其他引擎未全量刷新。 |
| [视觉小说引擎能力矩阵](visual_novel_localization_matrix.md) | 10-22 有限复核 Naninovel、MV/MZ 与 #509；旧排名不自动成为当前支持承诺。 |
| [Translator++ 接入评估](translatorpp_integration_assessment.md) | 09-16 研究与 #509 续验衔接；独立使用保留，生产 bridge 暂缓。 |
| [#431 LiteLLM 迁移矩阵](issue-431-litellm-migration-matrix.md) | 调用点与依赖边界记录；未据此移除 LiteLLM 或更改默认路径。 |
| [#364 校准执行手册](issue-364-calibration-runbook.md) · [质量校准基线](quality_calibration_baseline.md) | 已完成的历史真实项目校准，形成于 #513 字段拆分前；新字段语义见[Batch 工作流](../batch_workflows.md)。 |
| [#426 coverage block 归因](renpy_coverage_block_attribution.md) | 合成归因与修复记录，#470 后续真实样本已完成；具体引擎语义未知仍保留。 |
| [#487 字体覆盖只读 spike](renpy_font_coverage_spike.md) | PR #511 已交付并关闭；静态 cmap 不等于游戏字体验收，doctor / GUI 生产接入未承诺。 |

## 归档与维护

[归档索引](../archive/README.md)保存已只具过程回顾价值的设计和审计稿。
仍被现行手册、测试、实验或路线决策使用的合同和材料保留在本目录，路径不因 issue
关闭而自动变化。新增文稿应登记到对应分组；更新任务状态时读回 GitHub 并注明核对日期，
避免继续向索引堆叠互相矛盾的“当前状态”补记。
