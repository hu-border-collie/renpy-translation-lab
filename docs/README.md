# 文档地图

按当前要做的事选入口，无需顺次阅读全部文档。
稳定 tag 使用该 tag 的手册；本目录随 `main` 更新，发行差异见[变更日志](../CHANGELOG.md)和[v1.0.0 说明](releases/v1.0.0.md)。

## 使用工具

| 任务 | 阅读入口 |
| --- | --- |
| 了解适用范围与交付方式 | [项目首页](../README.md#支持范围) |
| 第一次通过 GUI 翻译 | [GUI 快速开始](quickstart_gui.md) → [GUI 工作台](gui_workbench.md) |
| 让 Agent／脚本通过 CLI 翻译 | [Agent／CLI 快速开始](quickstart_agent.md) → [Batch](batch_workflows.md)或[耐久 Sync](sync_workflow.md) |
| 由宿主 Agent 提供初译 | [外部初译工作包](external_translation_work.md)：实验 CLI，当前仅 Ren'Py |
| 处理限定窗口的 RPG Maker MV | [MV 耐久 Sync 与预处理](rpgmaker_mv_sync.md) |
| 安装、配置模型或迁移旧配置 | [安装与配置](setup.md) · [模型配置迁移](model_config_migration.md) |
| 看检查结果、覆盖范围和写回阻断 | [doctor 建议](doctor_recommendations.md) · [Engine Adapter](engine_adapter.md) · [结构 token 保护](structure_protection.md) |
| 管理多个游戏项目 | [工作区项目总表](games_registry.md) |
| 配置上下文或查看调用成本 | [上下文系统](context_systems.md) · [模型用量账本](model_usage_ledger.md) |
| 使用关系／语义分析 | [分析器入口与边界](relation_analysis.md)；独立模块细节见[模块说明](../relation_analyzer/README.md) |

Story Memory 数据格式：[示例](story_graph.example.json) · [schema](story_graph.schema.json)。

## 修改代码与验证

| 任务 | 阅读入口 |
| --- | --- |
| 开始开发 | [AGENTS.md](../AGENTS.md) → [贡献约定](../CONTRIBUTING.md) |
| 理解分层、配置与所有权 | [架构概览](architecture.md) · [代码路径索引](code_paths.md) |
| 运行测试与排查 CI／沙箱 | [测试目录](../tests/README.md) · [CI](ci.md) · [沙箱与临时目录](ci.md#沙箱与临时目录) |
| 维护依赖和哈希锁 | [依赖说明](dependencies.md) |
| 修改 doctor 展示与 gating | [环境检查状态矩阵](doctor_states_matrix.md) |
| 验证真实 Provider | [Provider Smoke Matrix](provider_smoke_matrix.md)，调用预算与验收范围单列 |
| 查公共合同与复现工具 | [当前工作与技术合同](plans/README.md#技术合同) |

## 当前工作与历史证据

- [当前工作与技术合同](plans/README.md)：开放任务、条件候选、稳定合同和仍在使用的复现说明。
- [历史资料](archive/README.md)：已收口实验、旧路线研究、审计和设计过程稿；保留原日期、失败与未知项。
- [项目沿革](project_history.md)：项目演变的简要时间线。

当前操作由对应手册维护，任务状态只在计划索引维护。归档报告中的 OPEN、待验、旧路径或旧排名
按原基线理解；归档不代表其中所有设想都已实现，也不把旧建议重新加入当前队列。
