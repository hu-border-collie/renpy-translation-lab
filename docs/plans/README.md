# 当前工作与技术合同

这里只维护当前工作、仍有效的工程合同和复现工具。日常操作从[文档地图](../README.md)开始；
已收口的实验、旧路线研究和历次盘点统一查[历史资料](../archive/README.md)。

状态核对：**2026-10-27，代码基线 `main@30fe96d`；当前工作由下列 4 个开放 issue 跟踪。**
合并工程、离线检查、语义审阅和原生游戏验收分别判断。

## 当前工作

| 跟踪 | 已有基础与剩余工作 |
| --- | --- |
| [#567 MV 与 Komorebi 交付](https://github.com/hu-border-collie/renpy-translation-lab/issues/567) | A1–A3、A4 离线增量及限定显示字段／预处理已由 #568、#571、#573–#576 合并；当前路线是[MV 耐久 Sync](../rpgmaker_mv_sync.md)。原生补证、完整章节与全游戏交付仍未验收。 |
| [#565 Komorebi 调查](https://github.com/hu-border-collie/renpy-translation-lab/issues/565) | 离线调查与独立 101＋7 试点已交付；原生可达、显示、存读档和验收回填待完成。core 版本声明冲突继续保留，以实际材料摘要固定窗口。 |
| [#272 引擎研究与选型](https://github.com/hu-border-collie/renpy-translation-lab/issues/272) | 具体 MV 实施已归 #567；研究收口与线上选型回填仍待处理。其他引擎没有全量重查，不按旧评级自动立项。 |
| [#147 Story Graph 试点](https://github.com/hu-border-collie/renpy-translation-lab/issues/147) | 保持低优先级。出现现有 glossary／原文索引难解决的关系、称呼或口吻问题后，再做最小确认图谱对照；不是 #567 的前置。 |

当前交付的材料：

- [#565 调查](issue-565-komorebi-survey-2026-10-23.md)及[脱敏元数据](issue-565-materials/README.md)：已合并的离线范围与独立清单。
- [#567 A4 离线试点](issue-567-a4-offline-pilot-2026-10-25.md)：101＋7 真实结果、Agent 审阅和安全写回；原生部分后置。
- [#567 B0 范围盘点](issue-567-b0-scope-audit-2026-10-25.md)及[材料说明](issue-567-b0-materials/README.md)：只读范围与脱敏汇总；C0 为候选范围，发行与完整章节分母尚未验收。

## 已重评的旧建议

原 11 类宏观建议已重新取舍：8 类撤出独立待办，仅以下 3 类条件保留。
详细依据和当时的验证记录见[10-27 重评记录](../archive/audits/plan_inventory_2026-10-27.md#3-原-11-类建议的实质重评)。

| 条件候选 | 再立项需要的证据 |
| --- | --- |
| 字体检查的引擎对应接入 | 先证明可复用检查能减少实际人工工作，并冻结静态检查的支持／未知边界；Ren'Py spike 不直接等于 MV 接入。 |
| 撤销成功写回 | 出现独立副本和现有事务恢复无法满足的真实撤销需求，再定义持久备份、外部编辑冲突与后续状态。 |
| RAG 热点优化 | 当前真实任务的性能剖析表明检索或持久化成本显著；不以旧合成基准直接换库。 |

场景窗口、术语审核合并、结构保护、退出协调、请求隔离和多轮实验已有交付。
工作包 GUI、固定自动流水线与额外执行路线不作为默认欠项。
Translator++ 生产 bridge、OpenRouter Batch、终审 Agent Loop 保持暂缓；结论与失败记录见[归档实验](../archive/README.md#已收口实验)。

## 技术合同

这些路径供实现和测试持续引用，保留稳定。历史设计中的 target 或阶段顺序不代替现行操作手册。

| 合同 | 用途与现行入口 |
| --- | --- |
| [Engine Adapter P0](engine_adapter_contract.md) | occurrence、coverage、版本与写回基础合同；当前支持范围见[Engine Adapter](../engine_adapter.md)。 |
| [#202 Settings](issue-202-settings-page-contract.md) | 页面、coordinator 与宿主写盘所有权；当前分层见[架构](../architecture.md)。 |
| [#347 → #348 交接](issue-347-to-348-handoff.md) | 耐久执行器接缝与职责；操作见[Sync 工作流](../sync_workflow.md)。 |
| [#348 Model Routing](issue-348-model-routing-config-contract.md) | 配置、兼容和迁移事务；操作见[模型配置迁移](../model_config_migration.md)。 |
| [#422 apply／export](issue-422-p2-apply-export-contract.md) | 双写事务、回执与恢复；操作见[Batch 工作流](../batch_workflows.md)。 |
| [#427 普通审校](issue-427-review-index-contract.md) | index、决定、提案与 needs_recheck；操作见[GUI 工作台](../gui_workbench.md)。 |
| [#431 S1 直连](issue-431-openai-compatible-contract.md) | 连接、凭据和结构化生成合同。 |
| [#431 S2 目录／诊断](issue-431-openai-compatible-s2-contract.md) | 目录失败、诊断机器码、脱敏和取消。 |
| [#431 S3 Sync 终审](issue-431-final-review-sync-contract.md) | Sync 终审分流与 legacy Batch 默认。 |
| [#488 预检](issue-488-preflight-summary-contract.md) | freshness、unknown 和派发前摘要；操作见[Batch 工作流](../batch_workflows.md)。 |

公共 JSON 的 schema 2、独立源／目标和数组路径已交付；详见[核心 API](../engine_adapter.md#source-json-writeback-p1)及[原创夹具](../../tests/fixtures/source_json_writeback/README.md)。

## 复现与维护参考

| 文档 | 保留原因 |
| --- | --- |
| [字体覆盖只读工具](renpy_font_coverage_spike.md) | 已交付 API／脚本的支持范围、reason code 与 fixture；不代表游戏显示验收。 |
| [coverage block 归因](renpy_coverage_block_attribution.md) | 原创归因工具和已修问题的复现；#470 真实归因已完成，精确引擎语义 unknown 仍保留。 |
| [质量校准手册](issue-364-calibration-runbook.md)与[历史基线](quality_calibration_baseline.md) | 测试夹具和校准脚本仍引用；数据保持旧白名单口径，当前策略见[Batch 工作流](../batch_workflows.md)。 |
| [维护路线](maintainability_roadmap.md) | 保留已完成的两轮范围及停止条件；后续只按具体问题选择切片。 |
| [#509 原创材料](issue-509-materials/README.md) | 冻结实验输入和字节摘要保留原路径；结果已转入[往返实验记录](../archive/experiments/issue-509-translatorpp-roundtrip-2026-10-21.md)。 |

更新功能时修改现有手册；更新任务状态时修改本页。新合同或有独立数据的实验才单列文稿，
日常讨论、状态补记和阶段汇报优先并入既有条目；已收口快照转入[归档索引](../archive/README.md)。
