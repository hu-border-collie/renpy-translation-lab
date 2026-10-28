# 计划文档盘点（2026-10-27）

> 归档于 2026-10-27：保留原日期、基线、结果与未验边界。现行操作见[文档地图](../../README.md)，当前工作见[计划索引](../../plans/README.md)。

核对基线：本地 `main` 与远端 `refs/heads/main` 均为
[`30fe96d19a1d9bf15f9c2a359ddd2c62eb4efa07`](https://github.com/hu-border-collie/renpy-translation-lab/commit/30fe96d19a1d9bf15f9c2a359ddd2c62eb4efa07)。
当前状态入口仍为[计划索引](../../plans/README.md)，用户操作以[文档地图](../../README.md)所列现行手册为准。

本次盘点覆盖既有 **39 份**计划、合同、实验报告与材料说明，不计本报告和目录索引：
37 份已由 Git 跟踪，B0 报告及其材料说明两份是既有本地未跟踪文件。对应 B0 元数据也尚未入库。
这些本地材料参与范围判断，但不计作主线或远端已经交付。
`docs/archive/` 只核对归档职责与现行衔接，不把历史清单重新变成待办。

## 1. 当前结论

GitHub 当前有 **4 个开放 issue、0 个开放 PR**。开放项是 #147、#272、#565、#567。
已立项工程大多交付，剩余计划仍包含实际游戏交付、未立项候选和主动暂缓方向。
开放 issue 数量不能代表剩余工作量：#567 同时包含 MV 支持验收和 Komorebi 全游戏汉化。

| 类别 | 本次判定 | 处理方式 |
| --- | --- | --- |
| 已交付合同与限定实验 | Settings、耐久 Sync、模型路由、Provider 首期、审校、复用、质量字段拆分、预检等已经交付；旧文档中的 OPEN 或 target 保留的是当时状态 | 不重复开发，不重开已完成 issue；按原验收边界使用 |
| 已有开放跟踪的交付 | #567 的原生补证、完整章节与全游戏交付仍未完成；#565 / #272 有调查验收及选型回填；#147 保留低优先级试点 | 沿用原 issue，按实际材料拆下一项工作 |
| 旧建议重评 | 首次提取的 11 类建议中，8 类撤出独立开发待办，3 类仅条件保留；部分目标已覆盖，部分方向未被采纳或已过时 | 第 3 节逐项给出取舍；本轮没有足够依据新增可立即开工的通用实现单 |
| 明确暂缓 | Translator++ 生产 bridge、OpenRouter Batch、终审 Agent Loop，以及缺乏需求的通用调度、新协议和存储重构 | 第 4 节保留重启条件，不因文档中出现过就恢复开发 |

本次核对的是仓库、线上状态和已有证据归属，没有新增模型调用、游戏写入或原生运行，
也没有重新评分历史译文、刷新所有外部工具版本或执行全部产品回归；候选重评另复跑了 80 项针对性回归。
工程、离线 fixture、真实 Provider、Agent / 人类语义审阅、原生显示和游玩分别报告。
本次未穷举游戏目录中的私有制品和暂定资产；仓库缺少完整交付验收记录，不表示相关准备从未开始。

## 2. 已有开放跟踪：交付还差什么

### 2.1 #567：MV 支持与 Komorebi 交付

| 阶段 | 已有交付 | 仍未完成或尚未取得的证据 |
| --- | --- | --- |
| A0 / #565 调查 | 脱敏报告、205 份 JSON 调查、独立 101 个主要位置与 7 个镜像；PR #568 已合并 | 原生场景可达、路线 / 存档检查点、显示与调查验收回填；准确上游 core 版本仍为 unknown，使用实际材料摘要窗口 |
| A1 公共 JSON 合同 | PR #568：schema 2、独立源 / 目标、类型化数组路径、字符串 token 渲染与事务组合 | 本阶段工程已交付；公共 API 不等于所有引擎或格式已有产品入口 |
| A2 MV Adapter | PR #571；#570 已按 completed 关闭 | 未知 core / 插件和 MZ 不继承该支持窗口 |
| A3 耐久 Sync 产品接线 | PR #573；#572 已按 completed 关闭，CLI / GUI、配置、doctor、双快照检查与写回已接通 | 本阶段工程已交付；不把产品接线当作正式支持的原生验收 |
| A4 真实小样本 | PR #574：101＋7 离线试点、兼容修复、Agent 语义审阅、共享 check / apply 和独立字节核对 | S1–S3 原生到达、五个分支、字体 / 布局、回顾、保存后重启读取；原报告将这些验收后置，不能补写为通过 |
| B0 范围与必要能力 | 既有本地报告冻结 C0 候选 2,788 个主要位置＋55 镜像；PR #575 已开放显式系统术语 / 地图显示名，PR #576 已接入关键词、原文索引与已审阅历史预处理 | 发行路线与完整首章分母尚未验收；字体、实际使用 UI、图片文字、数据库、运行时姓名及动态 / 插件文本逐项核对，所需能力另按实际需求补齐 |
| B1 完整章节 | 已有小样本与 C0 静态范围基础 | 尚无可据本次仓库记录认定通过的完整 C0 交付；需要冻结范围、术语 / 风格、执行路线及预算，完成翻译、独立审阅、订正、安全写回和原生验收 |
| B2–B3 全量与交付 | 由同一 epic 跟踪 | 按章节 / 路线扩展、遗漏和冲突对账、全量审阅、显示 / 分支 / 存读档矩阵及可恢复中文副本均未完成 |

依据：[#567](https://github.com/hu-border-collie/renpy-translation-lab/issues/567)、
[#565](https://github.com/hu-border-collie/renpy-translation-lab/issues/565)、
[MV 现行手册](../../rpgmaker_mv_sync.md)、[A4 离线记录](../../plans/issue-567-a4-offline-pilot-2026-10-25.md)、
[B0 本地报告](../../plans/issue-567-b0-scope-audit-2026-10-25.md)，以及已合并的
[PR #575](https://github.com/hu-border-collie/renpy-translation-lab/pull/575)和
[PR #576](https://github.com/hu-border-collie/renpy-translation-lab/pull/576)。

两项近期增量都不能被继续列为完全未实现：#575 已实现显示字段的限定子集，
#576 已实现预处理产品接线。它们没有完成真实游戏预处理或原生显示验收；
数据库、320 姓名修改、MapInfos 回退、插件 JS、字体和图片资源仍不在这些 PR 的写入范围。
MV Batch、外部 Agent 工作包、Story Memory 和 Project Analysis 注入也仍未开放。

B0 的静态连通范围、图片文字下界、显示字段与叙事 occurrence 是不同分母。
2,788＋55 不能自动变成完整发行章节，也不能用旧 A4 结果或旧检查直接授权新范围写回。
当前调查和计划中没有完成的实际交付，不能靠补勾工程清单结案。

### 2.2 #272 与 #147

| 开放项 | 当前基础 | 下一步与停止条件 |
| --- | --- | --- |
| [#272 多引擎研究](https://github.com/hu-border-collie/renpy-translation-lab/issues/272) | 有限研究、#509 负面 / 正面证据、#565 的真实 MV 需求；#567 已承接具体实施 | 回填 MV 选型、准确支持窗口和实现归属；其余候选未全量重查。先明确研究 issue 的收口范围，不能把其余引擎未验项全部勾选，也不按旧 Naninovel 排名批量实现 Adapter |
| [#147 Story Graph](https://github.com/hu-border-collie/renpy-translation-lab/issues/147) | MVP、seed、注入和诊断基础已在，默认关闭；试点验收尚未完成 | 出现 glossary / source index 难解决的关系、称呼或口吻问题后，用人工确认的最小图谱做同条件开关对照，记录非零命中、质量与维护成本。收益不足继续暂缓，先不开发全量维护工具 |

#147 正文关于所有私人项目启用情况的描述是历史快照；本次未审计全部私人项目，
不把该描述重述为当前事实。#272 的旧排序、官方能力描述和未勾选项也不能单独证明当前实现状态。

## 3. 原 11 类建议的实质重评

本节替换首次把 11 类旧建议统称为“未立项候选”的判断。
没有开放实现单，只能证明没有这张单，不能证明功能未做、方向仍适用或值得另开一单。
本轮对照最新代码、已合并 PR、研究收口决定和现行用法重新取舍：
**8 类撤出笼统的独立开发待办，3 类只条件保留，没有立即新增通用实现单的充分依据。**

### 3.1 逐项判定

| 原建议 | 最新证据与需要纠正的判断 | 本次决定 |
| --- | --- | --- |
| 1. 字体检查接入产品 | #487 / #511 的 Ren'Py 只读核心和脚本已做，doctor / 预检 / GUI 消费确实未做；但当前 #567 是 MV，不能直接套用 Ren'Py 静态引用解析。本地 C0 已有暂定字体方案和字形路由检查，尚未安装、未做原生验收 | **条件保留，撤回优先推荐。** 先完成实际游戏显示交付；只有可复用检查确实减少人工工作时，再定义引擎对应的窄接入范围 |
| 2. 撤销成功 apply | 失败事务恢复、安全预览、独立目标副本和导出路径已在；持久成功撤销合同仍不存在。旧研究把底层恢复直接映射成“一键回滚”过于乐观 | **继续暂缓。** 先取得现有副本 / 恢复流程不能满足的真实撤销场景，再决定持久备份、外部编辑冲突和后续状态合同 |
| 3. 可靠场景边界 | #410 的局部窗口已经按文件、共享 block 与显式 scene_boundary / route_id 处理，已有 unknown / 文件顺序退回及回归；不能再概括成缺少场景边界能力。通用剧情推断和自动生产语义边界没有交付，也未被采纳为当前前置 | **移出独立待办。** 已有边界处理继续用；如真实任务证实缺少某个可审查输入或发生跨场景污染，再针对该例立项，不重做通用上下文框架 |
| 4. 自动两阶段编排 | 关键词→候选审核→glossary 合并→正文消费已有核心 / GUI；#576 也已补 MV 关键词及受控检索预处理。旧“所有项目数据库优先、自动生成术语、再译正文”并非现行必要流程；宿主 Agent 已负责顺序与策略 | **撤下旧固定流水线提案。** 复用现有能力组合；自动编排并未实现，也没有因此形成一笔应补交的工程欠项 |
| 5. RAG 读写性能 | JsonRagStore 的逐条相似度 / 全量排序与历史文件重写仍在，Source Index 是另一实现；但历史译文 RAG 默认关闭，#344 的限定上下文实验没有支持默认开启，旧合成基准没有证明当前真实任务瓶颈。#576 的预处理、身份与 stale 合同已经做过 | **仅保留性能触发条件。** 先量实际使用路径的耗时占比；不重复建设 RAG 接入，也不先换库或启动泛化优化 |
| 6. 主窗口任务生命周期 | 已有 ShutdownCoordinator、CliRunner 和 workflow / 就绪摘要的项目、任务及代次保护；#529、#541 与后续 MV 回调接线都已处理具体生命周期问题。“尚未统一所有 workflow”不能作为无限期未完成标准 | **撤下全面治理任务。** 保留现有设施；若出现具体取消、关闭、切换或回调缺陷，再按一个真实 workflow 定界，不宣称所有旧复杂度已经消失 |
| 7. CLI 反向依赖与共享状态 | #528 / #531 已完成共享请求下移和创建时绑定依赖，#563 / #564 已整理目录。external_translation_work 的薄入口依赖、手工导入脚本等残留并不自动证明跨任务污染；runtime 装配入口也有保留职责 | **撤下笼统清理任务。** 两轮目标已交付；只有独立调用的具体阻碍、维护成本或可复现污染才支持下一切片，不把一次 import 命中当缺陷 |
| 8. 工作包 GUI | #508 已完成实验 CLI、范围核验、持久候选、恢复和安全交付，并明确收缩为高级 / 脚本工具；操作手册明确不增加工作包编辑器或第二套审校 GUI | **从当前候选队列移除。** 这是未采纳的扩展方向，不能因没有 GUI 就把已完成的 #508 重新解释为缺少产品化。新的人工使用需求出现时再评估 |
| 9. 其他执行 / 引擎产品接入 | MV 当前主线已经是耐久 Sync，A1–A3 及显示 / 预处理增量均完成。MV Batch、MV 外部工作包和 Tyrano 产品端到端入口是明确未开放的额外路线，不等于当前主线仍未接入 | **撤下合并式扩展任务。** #567 沿用已选择路线；若实际改选宿主 Agent，再单独定义 MV 工作包。无具体游戏需求时不启动其他路线 |
| 10. 模型 / 思考档 / 分块 / 定向重译 | 七轮对照、Provider 四类矩阵、两个连接、真实 PA / embedding / 上下文 A/B，以及 #508 的 552 项路线实验均已做过；#556 已落实请求合同和默认值。当前普通翻译 canonical、Sync / Batch 示例均为 60 条 / 18,000 字符，“生产默认 40 条仍待对照”是旧基线 | **撤下泛化实验待办，改为按问题选型。** 不能因为没有全局最优模型就说实验未做；新模型、特化协议、不同思考档或定向重译只在实际决策需要时另做有预算的窄实验，旧结果也不外推成质量冠军 |
| 11. 通用过滤 / 术语占位 / 自动断行 | 扫描分类、coverage 审阅、共享 protect / restore、保留词与 glossary 约束、语言 / 排版字段拆分均已做。结构 token 占位不等于翻译术语强制替换；更激进过滤、词形恢复和自动布局仍未被采纳 | **拆开后撤下总括任务。** 已完成部分归入交付；其余只有真实误报、漏扫或显示样例才能触发，保持原结构和语言 / 排版语义边界 |

### 3.2 关键证据与本轮复验

- **场景处理已有实现。** [共享窗口](../../../quality_review/translation_plan.py)明确区分
  `file_order`、`shared_block_label` 和 `explicit_scene`，不把每句唯一 translate ID 当作场景。
  [对应回归](../../../tests/quality_review/test_translation_plan.py)覆盖共享 block、显式场景、
  已译邻居、跨块及中间已译条目。通用剧情推断或新的输入产品接口仍须另有需求。
- **术语流程已有实现。** [共享合并核心](../../../quality_review/keyword_glossary_merge.py)、
  [MV 现行流程](../../rpgmaker_mv_sync.md)及
  [PR #576](https://github.com/hu-border-collie/renpy-translation-lab/pull/576)可共同组织两阶段使用；
  审核与版本绑定应保留，固定数据库优先和自动采纳不能借这套实现被默认授权。
- **生命周期与请求隔离已有实现。** [退出协调](../../../gui_qt/runtime/lifecycle.py)、
  [CliRunner](../../../gui_qt/runtime/cli_runner.py)和
  [请求隔离回归](../../../tests/execution/test_durable_request_isolation.py)是实际调用设施。
  本轮没有复现需要“全面治理”才能解决的新缺陷，也没有据此证明整个进程可并发处理任意项目。
- **普通翻译 40 条的基线已过时。** [canonical 常量](../../../quality_review/translation_core.py)为
  60 / 18,000；[当前示例](../../../translator_config.example.json)的 Sync / Batch 也为 60。
  keyword 等其他阶段自己的 40 条设置不能混用为普通翻译默认。
  [PR #556](https://github.com/hu-border-collie/renpy-translation-lab/pull/556)已将实验取舍接入共同路径；
  历史明确保存的项目配置继续优先，不能反过来宣称私人项目全部已迁移为该默认。
- **工作包 GUI 的排除是已有决定。**
  [#508 收口](https://github.com/hu-border-collie/renpy-translation-lab/issues/508)与
  [现行手册](../../external_translation_work.md)均明确高级 CLI 边界；已有审校 GUI 可继续用于写回后订正。
- **字体准备已有局部工作。** 本次只读核对本地 C0 `selection` / `validation` 元数据：
  已有暂定字体和字形路由准备，`font_overlay_installed=false`、`native_validation=not_performed`。
  不把准备说成尚未开工，也不把它升级为主线字体检查接入或原生显示通过；素材和本机路径不入本报告。

本轮复跑共 **80 项**：局部上下文、默认分块、关键词审核合并和结构保护 65 项，
退出协调 7 项及耐久请求依赖隔离 8 项，全部通过。
GUI 首次启动因默认临时目录写权限失败；仅改用工作区独立 TEMP / TMP、空 keyring 和 offscreen 后，
后两组 15 项通过，没有修改产品代码、开发者配置或测试断言。
这些测试证明对应既有合同和限定行为仍在，不能证明语义收益、完整 GUI 人工验收或通用性能瓶颈。

### 3.3 收缩后的候选边界

目前只把 **字体检查的引擎对应接入、成功写回撤销、真实 RAG 热点优化**保留为条件候选。
三项均未满足本轮立即立项的充分依据：分别缺少可复用消费场景 / 窄支持合同、
现有副本与恢复方式不能满足的实际撤销需求、以及当前任务性能剖析。
保留条件不代表其他扩展已经实现或永久取消；它们需要新的具体需求，不能继续以旧宏观标题挂成默认欠项。

取舍沿用[七类重评](plan_reassessment_2026-09-19.md)、[维护路线](../../plans/maintainability_roadmap.md)、
[Agent 提案](../research/agent_translation_toolkit.md)与[模型实验](../experiments/translation_model_context_reassessment_2026-10-11.md)的证据边界，
并以当前代码和后续收口覆盖其旧状态。字体、撤销和性能的当前实现分别见
[只读字体核心](../../../foundation/font_coverage.py)、[事务清理](../../../foundation/atomic_io.py)、
[JsonRagStore](../../../context_engine/rag_memory.py)。

## 4. 已有明确暂缓决定的方向

| 方向 | 已核对状态 | 重启条件 / 保留边界 |
| --- | --- | --- |
| Translator++ 生产 bridge、.trans writer / MCP 自动化 | #509 已 completed；PR #562 已合并，决定保留独立工具使用，暂缓生产 bridge | 有独立使用不能满足的真实缺口，并能处理 staging、context 漂移丢译与非目标副作用；不能把限定 19 处导出扩成通用支持 |
| OpenRouter Batch 生产策略 | #520 已按 not_planned 关闭；PR #560 已合并，DeepSeek 实测成功但最终仍暂缓 | 新需求足以抵偿延迟与维护成本，并补齐数据保留 / 清理、费用和完整同口径比较。现有成功样本不等于已新增生产策略 |
| 取证式 Final Review Agent Loop | #305 按 not_planned 关闭；静态终审及 Sync 已交付 | 先证明动态取证能改善误报 / 漏报或确认成本；模型自报完成回执不能证明覆盖完整 |
| 全自动 Story Graph、图数据库及更强 scene 检索 | #147 仍只要求最小确认图谱试点；这些扩展没有成为试点关闭前置 | 先验证试点质量收益与维护成本，不预先全量建图或换库 |
| 移除 LiteLLM、新增 Responses / Anthropic 等协议 | #431 首期已 completed，仍保留 LiteLLM。`translator_runtime` 的 live embedding 路径仍调用 `litellm.embedding` | 指定模型 / 协议需求无法由现有入口满足时再拆 transport；移除依赖还需兼容、迁移和回滚，不因直连生成可用就判定 LiteLLM 冗余 |
| 跨宿主 Agent runtime、动态领取 / 租约 / 失联回收 | #508 以收缩结论完成；当前并行为互斥 ID 分工、独立成果、协调者串行接收 | 出现多个协调者动态领取的实际需求后再设计，不为一次章节分工引入通用平台 |
| 通用多媒体 Adapter、多引擎同时扩展、RAG 换库 | 保留为研究方向，未承诺完整接入 | #567 实际所需图片文字须继续交付，但单游戏资产处理不自动要求建设图像 / 音频通用平台；其他引擎与存储迁移均需独立需求和合同 |

来源：[#509](https://github.com/hu-border-collie/renpy-translation-lab/issues/509)、
[#520](https://github.com/hu-border-collie/renpy-translation-lab/issues/520)、
[#305](https://github.com/hu-border-collie/renpy-translation-lab/issues/305)、
[#508](https://github.com/hu-border-collie/renpy-translation-lab/issues/508)、
[LiteLLM 矩阵](../research/issue-431-litellm-migration-matrix.md)与[当前工作包合同](../../external_translation_work.md)。
这些方向中的关闭、暂缓与非目标各有原范围，不能一律解释为功能已经实现，
也不能一律解释为仍欠原 issue 的交付。

## 5. 需要校正的状态表达

| 文档 / 线上记录 | 残留表达 | 本轮确认的当前状态与处理 |
| --- | --- | --- |
| 计划索引 | 仍以 10-25 / `1d44863` 为基线，B0 缺口容易被理解为全部仍未实现 | 更新到 `30fe96d`，分别记录 #575 / #576 的工程完成和未验范围，保留 B0 本地状态 |
| #567 正文 | 当前任务仍指向 A3 / #572，子任务仍未勾选 | #572 已 completed，#573 已合并；本次只在仓库对账，未修改线上 issue |
| #565 / #272 正文及研究衔接 | 仍有未检查目标游戏、等待公共数组路径 / Adapter / 产品接线的旧基线 | 调查与 A1–A3 已交付，正式调查验收和线上选型回填仍未完成；不把整个研究清单补勾 |
| #427 合同 | 9 月横幅仍写 OPEN、等待实测 | 9-30 按 completed 关闭；范围是工程与 Agent 可完成的隔离 / 原生 GUI 操作，不含真实游戏、译文语义或 Provider 验收 |
| #431 S1 / S2 / S3 合同 | 9 月横幅仍写真实 Provider 待验 | #431 已于 10-16 完成限定首期验收，后续 #344 于 10-17 关闭；不代表所有协议或 Provider 都已支持 |
| #344 / #508 及 10-13、10-15 实验报告 | 原日期正文仍写 OPEN、待合并或后续实验未做 | #344 已于 10-17 completed，#508 已于 10-20 completed；续验与修复已合并，原失败、费用未知和质量 / 效率限制仍保留 |
| 9-19 七类重评 | #512 / #513 当时未交付或仍 OPEN | 两项均已交付并于 9-30 completed；字体生产接入与成功撤销未随之完成 |
| coverage 归因报告与旧 Adapter 合同 | 真实大项目频率归因仍由 #470 等待 | #470 已于 9-30 completed；12,686 候选及独立漏扫核对已有记录，4 个结构 / tokenizer 区域的精确引擎语义仍为 unknown，未放宽 coverage block |

有日期的实验与研究保留原始基线、失败和当时决定；本次只更新索引并增加本报告，
不批量改写旧报告，也不把 closed 当作原文每项设想都已完成。

## 6. 全量文件对账

下列 39 项是本次盘点对象。它们是文档数量，不是剩余任务数量。

### 6.1 技术合同与交接：10 份

| 文档 | 当前归属与剩余边界 |
| --- | --- |
| [Engine Adapter P0](../../plans/engine_adapter_contract.md) | #265 已完成；Ren'Py 产品路径、Tyrano 核心 / 离线边界继续有效；MV 新增量归 #567 |
| [#202 Settings](../../plans/issue-202-settings-page-contract.md) | Phase A–D 已交付并关闭；宿主写盘 / Qt 边界保留，不是遗漏阶段 |
| [#347 → #348 交接](../../plans/issue-347-to-348-handoff.md) | 两项已完成；Provider 验收后续亦由 #344 / #431 完成限定收口 |
| [#348 Model Routing](../../plans/issue-348-model-routing-config-contract.md) | P0–P3 已完成；#457 默认策略决策为维持现状，非等待默认切换 |
| [#422 apply / export](../../plans/issue-422-p2-apply-export-contract.md) | 已交付；事务恢复与成功撤销不同，平台 TOCTOU / writer 混跑边界仍在 |
| [#427 普通审校](../../plans/issue-427-review-index-contract.md) | 工程及限定操作验收已完成；提案、草稿、决定不直接写游戏 |
| [#431 S1 直连](../../plans/issue-431-openai-compatible-contract.md) | 已交付；首期真实验收已完成，未覆盖协议和 LiteLLM 移除不在原关闭范围 |
| [#431 S2 目录 / 诊断](../../plans/issue-431-openai-compatible-s2-contract.md) | 已交付；固定连接实测不泛化所有 Provider |
| [#431 S3 Sync 终审](../../plans/issue-431-final-review-sync-contract.md) | 已交付；保留未显式配置时的 legacy Batch 默认 |
| [#488 预检](../../plans/issue-488-preflight-summary-contract.md) | S1–S3 已完成；unknown / freshness 合同继续有效，未包含字体生产接入 |

### 6.2 实验、调查与验收记录：13 份

| 文档 | 当前归属与剩余边界 |
| --- | --- |
| [#344 四类 Provider](../experiments/issue-344-provider-matrix-2026-10-16.md) | 固定版本四类真实 smoke；原失败与恢复分列，#344 后续已关闭 |
| [#344 上下文 / 恢复](../experiments/issue-344-context-acceptance-2026-10-17.md) | 恢复、费用、fresh PA、非 Gemini embedding 和限定 A/B 已完成；不支持默认开启高级上下文的结论不变 |
| [#431 / #508 续验](../experiments/issue-431-508-acceptance-2026-10-15.md) | 双连接、凭据隔离和章节复审记录；与后续收口相接，不是待重复执行清单 |
| [#508 串行 / 并行续验](../experiments/issue-508-parallel-acceptance-2026-10-17.md) | 552 项与混合恢复已交付，#508 已关闭；未证明通用质量 / 效率优势，未测桌面范围不补写为通过 |
| [#509 单版本往返](../experiments/issue-509-translatorpp-roundtrip-2026-10-21.md) | 实验完成、PR #562 合并、#509 关闭；非目标副作用及漂移缺译保留，bridge 暂缓 |
| [#520 首轮](../experiments/issue-520-openrouter-batch-spike-2026-10-20.md) | 保留 GPT / Gemini 首轮负结果；后续 DeepSeek 成功不覆盖历史失败 |
| [#520 合同 v1](../experiments/issue-520-openrouter-batch-contract-v1.md) | 研究合同，不是生产策略；#520 按 not_planned 收口 |
| [#520 DeepSeek 续验](../experiments/issue-520-openrouter-batch-deepseek-2026-10-20.md) | Sync / Batch 2 请求、6 item 实测；保留费用 / 清理及完整比较缺口，生产接入暂缓 |
| [#565 调查](../../plans/issue-565-komorebi-survey-2026-10-23.md) | 离线报告已合并，原生及调查验收尚未完成；后续工程归 #567 |
| [#567 A4](../../plans/issue-567-a4-offline-pilot-2026-10-25.md) | 离线 101＋7 与修复已由 #574 合并；原生部分待验，Agent 审阅不冒充人类 |
| [#567 B0](../../plans/issue-567-b0-scope-audit-2026-10-25.md) | 本地未跟踪报告；物理范围及 C0 候选，不是发行 / 完整章节验收；部分能力已由后续 #575 / #576 交付 |
| [10-13 Provider 补证](../experiments/provider-acceptance-reassessment-2026-10-13.md) | 历史补证，后续由 #556–#558 与 #344 / #431 收口；其旧待办不独立重开 |
| [模型 / 上下文 / 分块重评](../experiments/translation_model_context_reassessment_2026-10-11.md) | 七轮实验已做，合同与默认值由 #556 续接；旧生产 40 条基线过时，不重复建立泛化实验任务，新对照按具体选型问题决定 |

### 6.3 方向、研究、校准与只读工具：13 份

| 文档 | 当前归属与剩余边界 |
| --- | --- |
| [Agent 工具包提案](../research/agent_translation_toolkit.md) | #508 实现并验证了收缩子集；GUI、其他引擎、MCP 和动态调度未成为全量承诺 |
| [GitHub 工具源码研究](../research/github_localization_projects_research.md) | 9-06 研究快照；结构保护 / 审校 / 预检 / 复用等已实现，字体 / 撤销 / 编排等分别评估 |
| [#272 有限研究衔接](../research/issue-272-research-closeout-2026-10-22.md) | 研究与回填草案已合并；#565 / #567 后续不由该历史合同自动判定完成 |
| [LiteLLM 迁移矩阵](../research/issue-431-litellm-migration-matrix.md) | 保留调用点和迁移边界；live embedding 仍使用 LiteLLM，移除依赖与新协议未承诺 |
| [#364 校准手册](../../plans/issue-364-calibration-runbook.md) | 历史真实校准已完成；旧白名单口径保留，当前新字段按现行手册 |
| [维护路线](../../plans/maintainability_roadmap.md) | #528 / #531 已完成；#563 / #564 目录迁移也已合并，生命周期 / 剩余依赖 / 状态治理另按收益选择 |
| [多引擎生态](../research/multi_engine_localization_ecosystem_research.md) | 历史研究加有限续验；MV 已进入 #567，其他引擎未全量刷新，bridge 暂缓 |
| [9-19 七类重评](plan_reassessment_2026-09-19.md) | 当时两个新单 #512 / #513 均已完成；其余建议按本次分类，原数据不重写 |
| [质量校准基线](../../plans/quality_calibration_baseline.md) | #364 已完成的聚合基线，非持续未完计划 |
| [coverage block 归因](../../plans/renpy_coverage_block_attribution.md) | 合成 follow-up 与 #470 真实归因均已完成；具体引擎语义未知及门禁继续保留 |
| [字体覆盖 spike](../../plans/renpy_font_coverage_spike.md) | #487 / PR #511 已完成；doctor / 预检 / GUI 接入仍为候选，静态 cmap 不等于显示验收 |
| [Translator++ 接入评估](../research/translatorpp_integration_assessment.md) | 原评估已由 #509 实测续接；保留独立工具，暂缓生产 bridge，不是未交付的接入承诺 |
| [引擎能力矩阵](../research/visual_novel_localization_matrix.md) | 10-22 有限复核与 #565 调查说明；MV 实现以 #567 / 当前代码为准，旧候选排名不作排期依据 |

### 6.4 材料说明：3 份

| 文档 | 当前归属与剩余边界 |
| --- | --- |
| [#509 原创材料](../../plans/issue-509-materials/README.md) | 已跟踪，可复现限定数据覆盖层；不包含引擎 / 编辑器 / 商业素材，不继承为其他 MV 或 MZ 的验证 |
| [#565 脱敏材料](../../plans/issue-565-materials/README.md) | 已跟踪，包含调查元数据；不是可启动游戏或可重放的完整私人试点 |
| [#567 B0 材料](../../plans/issue-567-b0-materials/README.md) | 本地未跟踪，静态汇总与候选范围；不是配置、覆盖批准、翻译成果或写回计划 |

## 7. 建议的下一步与本轮交付

1. 保持 #567 为实际交付主线。按用户既定安排推进范围、字体、图片与剩余显示需求，
   原生后置安排不因本次盘点改变；到相应阶段分别验收 A0 / A4、完整章节和全游戏。
2. 将 #567 已合并阶段与 #565 / #272 选型证据回填线上。该操作属于建议，
   本次未修改 issue、未关闭任务、未创建新单或发布 PR。
3. 撤回将 11 类旧建议整体作为新增队列、以及默认优先做字体检查接入的判断。
   当前没有充分依据立即新增通用实现单；3 项条件候选各按真实需求再决定，
   其余 8 类的已覆盖目标、未采纳扩展和具体触发条件按第 3 节处理。
4. 继续以实际翻译 / 游玩收益选择下一项工作，不重复已经完成的实验和基础设施。
   不用旧路线图顺序、功能数量、某个高级设想尚未实现或沉没开发成本替代收益证据。

本轮新增本报告并更新计划索引，保留既有 B0 文件与索引中的本地补记。
全部 98 份 tracked Markdown 加本报告及既有 B0 两份 Markdown，共 **101 份**，
经 Markdown 解析检查 **711 条本地链接、57 个 Markdown 锚点**，无缺失；
39 份既有文档的盘点覆盖为 **39/39**，所引用本地源码路径均存在。
`git diff --check` 通过；既有 B0 报告、材料说明和元数据三文件的 SHA-256 与盘点前一致。
本次纯文档改动未重跑 CLI / GUI 全量门禁；第 3 节 80 项为候选重评的新执行，
其余历史测试与 PR 验收数字均保持原归属；上述文档计数已按候选重评后的文本重新核对。
