# 文档地图

根目录 `README.md` 负责快速上手和项目定位；本目录收纳配置、Batch、上下文、GUI 与项目边界等专题说明。

**读文档时**：用户手册以「现行」一组为准；标为历史的文稿只作设计/审计背景，不要当当前界面说明。

## 推荐阅读（现行）

| 你想… | 读 |
|---|---|
| 先跑通 CLI 翻译 | 根目录 `README.md` → [安装与本地配置](setup.md) → [Batch 工作流与安全检查](batch_workflows.md) |
| 使用图形界面 | [GUI 工作台](gui_workbench.md) → 按需回看 [setup.md](setup.md) |
| 多游戏工作区总表 | [工作区项目总表](games_registry.md) |
| 理解环境检查建议 | [环境检查智能建议机制](doctor_recommendations.md) · [状态矩阵](doctor_states_matrix.md) |
| 启用 RAG / 原文索引 / 剧情记忆 | [上下文系统](context_systems.md) · [setup.md](setup.md) 中的项目级开关 |
| 角色关系 / 语义分析 | [关系与语义分析](relation_analysis.md) · [`relation_analyzer/README.md`](../relation_analyzer/README.md) |
| 项目边界与安全 | [项目说明](project_notes.md) |
| 参与开发 / AI 协作 | 根目录 [CONTRIBUTING.md](../CONTRIBUTING.md)（含 **CLI / GUI 同步**） |

## 文档分组

### 现行：配置与 GUI

- [安装与本地配置](setup.md)：`translator_config.json`、**按项目**的 `project_context_settings.json`、work 目录、SDK / TL 模板。
- [GUI 工作台](gui_workbench.md)：当前 PySide6 界面（左导航五页、状态三栏、设置分区、诊断与工具、写回安全边界）。
- [工作区项目总表](games_registry.md)：`games_registry.json` / `GAMES.md`、CLI 与 GUI 刷新边界。

### 现行：Batch、上下文与检查

- [Batch 工作流与安全检查](batch_workflows.md)：`build → apply`、订正、关键词、identity v2、A/B、golden corpus。
- [上下文系统](context_systems.md)：RAG、原文索引、Story Memory、store 路径与 benchmark。
- [环境检查智能建议机制](doctor_recommendations.md)：建议等级、必需/可选并列、workflow_state。
- [环境检查状态矩阵](doctor_states_matrix.md)：layout / pending / 上下文派生字段与决策漏斗（开发对照）。

### 现行：分析与项目状态

- [关系与语义分析](relation_analysis.md)
- [项目说明](project_notes.md)
- [story_graph.example.json](story_graph.example.json) · [story_graph.schema.json](story_graph.schema.json)

### 历史参考（非现行用户手册）

下列文档**保留为过程与审计记录**，界面与交付状态以 [GUI 工作台](gui_workbench.md) 和代码为准。

- [GUI 信息架构重组计划](gui_ia_redesign.md)：Epic #157 的 P0–P3 设计 SSOT；**P0a–P3 已合并**。文中「分析与准备」「诊断页」等多为改造前用语。
- [翻译全生命周期审计](translation_workflow_audit.md)：流水线与门禁的代码审计快照，可能落后于最新实现细节。

## 配置分层（速查）

| 范围 | 文件 | 典型内容 |
|---|---|---|
| 工具全局 | `translator_config.json`、`api_keys.json` | `game_root`、模型、chunk、SDK、RAG/索引的**默认** |
| **当前项目** | `<work>/project_context_settings.json` | 是否启用批量 RAG、原文索引、build 时暖库 |
| 项目资产 | `<work>/glossary.json`、`macro_setting.md` 等 | 术语、口吻 |
| 工作区总表 | 工作区根 `games_registry.json` | 多游戏进度（非 lab 仓库内默认） |

## 维护约定

- 改 GUI 入口名称或设置分区时，**同步改** `gui_workbench.md` 与 `gui_qt/user_copy.py` 中的用户文案。
- 改 doctor 规则时，同步 `doctor_recommendations.md` 与测试。
- 历史文稿只加状态横幅、不整篇改写成新手册，避免与「现行」重复维护。
