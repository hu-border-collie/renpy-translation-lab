# 文档地图

本页按使用任务和维护职责导航。根目录 [README](../README.md) 介绍项目定位与安装入口；
具体命令、配置和安全边界由下列专题手册维护。

**版本范围：最新稳定版为 v1.0.0，`main` 为后续开发线。** 本页的现行手册随 `main`
更新；使用稳定 tag 时应阅读该 tag 的文档。变化见[变更日志](../CHANGELOG.md)与
[v1.0.0 发行说明](releases/v1.0.0.md)。

## 按任务开始

| 你想做什么 | 阅读顺序 |
| --- | --- |
| 第一次通过 GUI 翻译 | [GUI 快速开始](quickstart_gui.md) → [GUI 工作台](gui_workbench.md) |
| 让 Agent / 脚本通过 CLI 翻译 | [Agent / CLI 快速开始](quickstart_agent.md) → [Batch 工作流](batch_workflows.md)或[耐久同步工作流](sync_workflow.md) |
| 由宿主 Agent 提供初译 | [外部初译工作包（实验 CLI）](external_translation_work.md) |
| 安装、配置模型或迁移旧配置 | [安装与配置](setup.md) → [模型配置迁移](model_config_migration.md) → [Provider Smoke Matrix](provider_smoke_matrix.md) |
| 查检查结果、覆盖范围和写回阻断 | [doctor 建议](doctor_recommendations.md) · [Engine Adapter](engine_adapter.md) · [结构 token 保护](structure_protection.md) |
| 管理多个游戏项目 | [工作区项目总表](games_registry.md) |
| 配置上下文或查看调用成本 | [上下文系统](context_systems.md) · [模型用量账本](model_usage_ledger.md) |
| 修改仓库代码或文档 | [AGENTS.md](../AGENTS.md) → [贡献约定](../CONTRIBUTING.md) → [架构概览](architecture.md) → [代码路径索引](code_paths.md) |
| 运行测试或排查 CI | [测试目录](../tests/README.md) · [依赖与哈希锁](dependencies.md) · [CI](ci.md) · [GUI 沙箱说明](gui_tests_sandbox_notes.md) |
| 查实施状态、合同或研究证据 | [规划、合同与研究索引](plans/README.md) |
| 追溯项目与设计历史 | [项目沿革](project_history.md) · [历史归档](archive/README.md) |

## 现行操作手册

### 安装、配置与 GUI

- [GUI 快速开始](quickstart_gui.md)、[Agent / CLI 快速开始](quickstart_agent.md)：分别面向 GUI 使用者和通过 CLI 操作工具的 Agent。
- [安装与本地配置](setup.md)：源码环境、`translator_config.json`、项目级 `project_context_settings.json`、work 目录、SDK 与 TL 模板。
- [模型配置迁移](model_config_migration.md)：离线 preview / migrate / rollback，以及迁移后生产读取和 GUI 编辑的边界。
- [GUI 工作台](gui_workbench.md)：任务页、模型与供应商、设置、检查摘要、订正和诊断与运行日志。
- [工作区项目总表](games_registry.md)：`games_registry.json` / `GAMES.md`，CLI 与 GUI 的维护边界。

### 翻译、检查与写回

- [Batch 工作流与安全检查](batch_workflows.md)：Batch 生命周期、重试与恢复、订正、最终审校、关键词和质量规则。
- [同步翻译工作流](sync_workflow.md)：耐久 `sync-start / resume / status`、检查、预览与安全写回；兼容同步入口另有边界。
- [外部初译工作包（实验 CLI）](external_translation_work.md)：导出材料、接收部分成果、候选订正、持久冲突、恢复和绑定预览写回；当前支持范围为 Ren'Py。
- [Engine Adapter、覆盖审计与安全写回](engine_adapter.md)：Ren'Py 产品路径、TyranoScript 原生 catalog 核心与离线验证，以及公共源 JSON 核心 API 的范围。
- [结构 token 保护](structure_protection.md)：保护、恢复与校验引擎结构，旧请求兼容和失败边界。
- [环境检查建议](doctor_recommendations.md)：必需 / 可选能力、建议等级与准备状态。
- [Provider / 执行策略 / 最终审校 Smoke Matrix](provider_smoke_matrix.md)：真实验证步骤、调用预算、诊断与记录模板。

### 上下文、分析与用量

- [上下文系统](context_systems.md)：glossary、原文索引、RAG、Project Analysis、Story Memory 与存储路径。
- [关系与语义分析](relation_analysis.md) · [分析器说明](../relation_analyzer/README.md)。
- [模型用量账本](model_usage_ledger.md)：实际调用、离线补录、成本与 unknown 语义。
- Story Memory 数据参考：[示例](story_graph.example.json) · [schema](story_graph.schema.json)。

## 开发与维护

| 文档 | 职责 |
| --- | --- |
| [Agent 开发约定](../AGENTS.md)、[贡献约定](../CONTRIBUTING.md) | 改动约束、CLI / GUI 同步、验证与文档维护规则 |
| [架构概览](architecture.md) | 当前分层、配置事实来源、Settings 所有权与安全边界 |
| [代码路径索引](code_paths.md) | 从用户动作找到模块、关键函数和对应测试 |
| [测试目录与运行方式](../tests/README.md) | 测试分组、完整模块名、runner 与共享夹具 |
| [依赖输入与哈希锁](dependencies.md) | 直接依赖所有权、可复现安装与锁生成 |
| [CI 与定时集成检查](ci.md) | lint / type / audit、Provider smoke 与发布检查 |
| [环境检查状态矩阵](doctor_states_matrix.md) | doctor 派生状态与 gating 的开发对照 |
| [GUI 测试沙箱说明](gui_tests_sandbox_notes.md) | 缓存写入、权限与 GUI 测试执行环境 |
| [项目说明](project_notes.md) | 支持范围、交付方式、安全与验证限制 |
| [项目沿革](project_history.md) | OpenCode 原型到当前工作台的演变 |

## 规划、合同与历史材料

[规划、合同与研究索引](plans/README.md)集中维护 issue 状态、合并证据和剩余工作，
并按技术合同、实验记录和方向研究分类。本页只维护阅读入口，避免重复更新任务状态。
有日期的报告按其原基线阅读；最新状态由计划索引的核对日期与 GitHub 记录共同确认。

[历史归档](archive/README.md)保存过程稿、旧设计和审计快照。仍供当前实现、测试或决策
使用的合同与材料保留在 `plans/`；关闭 issue 本身不是移动文件的理由。
