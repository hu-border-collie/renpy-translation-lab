# 文档地图

根目录 `README.md` 负责快速上手和项目定位；本目录收纳配置、Batch、上下文、GUI 与项目边界等专题说明。

**当前状态：稳定版。** 稳定范围、验证记录、交付方式与不覆盖事项见 [项目说明](project_notes.md)。

**读文档时**：用户手册以「现行」一组为准；[archive/](archive/README.md) 中的历史文稿只作设计/审计背景，不要当当前界面说明。

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
| 查看版本变化 | [变更日志](../CHANGELOG.md) · [v1.0.0 发行说明](releases/v1.0.0.md) |
| 参与开发 / AI 协作 | 根目录 [CONTRIBUTING.md](../CONTRIBUTING.md)（含 **CLI / GUI 同步**） |
| 理解 PR 门禁与定时集成 | [CI 与定时集成检查](ci.md)（含 lint / type / audit） |
| 理解依赖所有权与哈希锁 | [依赖输入与哈希锁](dependencies.md) |

## 文档分组

### 现行：配置与 GUI

- [安装与本地配置](setup.md)：`translator_config.json`、**按项目**的 `project_context_settings.json`、work 目录、SDK / TL 模板。
- [GUI 工作台](gui_workbench.md)：当前 PySide6 界面（统一侧边导航、项目与环境、任务页、设置、诊断与运行日志、写回安全边界）。
- [工作区项目总表](games_registry.md)：`games_registry.json` / `GAMES.md`、CLI 与 GUI 刷新边界。

### 现行：Batch、上下文与检查

- [Batch 工作流与安全检查](batch_workflows.md)：`build → apply`、订正、关键词、identity v2、A/B、golden corpus。
- [上下文系统](context_systems.md)：RAG、原文索引、Story Memory、store 路径与 benchmark。
- [环境检查智能建议机制](doctor_recommendations.md)：建议等级、必需/可选并列、workflow_state。
- [环境检查状态矩阵](doctor_states_matrix.md)：layout / pending / 上下文派生字段与决策漏斗（开发对照）。

### 现行：分析与项目状态

- [关系与语义分析](relation_analysis.md)
- [CI 与定时集成检查](ci.md)
- [依赖输入与哈希锁](dependencies.md)
- [项目说明](project_notes.md)
- [story_graph.example.json](story_graph.example.json) · [story_graph.schema.json](story_graph.schema.json)

### 历史参考（已归档）

下列文档在 [archive/](archive/README.md)，**保留为过程与审计记录**；界面与交付状态以 [GUI 工作台](gui_workbench.md) 和代码为准。

- [GUI 信息架构重组计划](archive/gui_ia_redesign.md)：Epic #157 的 P0–P3 历史设计 SSOT，并记录后续 #176 页面化交付衔接。文中「分析与准备」「诊断页」等多为改造前用语。
- [翻译全生命周期审计](archive/translation_workflow_audit.md)：流水线与门禁的代码审计快照，可能落后于最新实现细节。
- [Design QA 验收记录](archive/design-qa.md)：2026-07 统一侧边导航与页面归属的视觉/自动化验收快照；不是用户手册。

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
- 历史文稿放在 `archive/`，只加状态横幅或修断链，不整篇改写成新手册，避免与「现行」重复维护。
