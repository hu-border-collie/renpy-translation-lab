# 文档地图

根目录 `README.md` 保持快速上手和项目定位；本目录收纳配置、Batch 流程、上下文系统、GUI、分析工具和项目边界等专题说明。

## 推荐阅读顺序

- 只想先跑通一次 CLI 翻译：先读根目录 `README.md`，再读 [安装与本地配置](setup.md) 和 [Batch 工作流与安全检查](batch_workflows.md)。
- 想用图形界面：先读 [GUI 工作台](gui_workbench.md)，再按需回看 [安装与本地配置](setup.md)。
- 想管理工作区内多个游戏项目总表：读 [工作区项目总表](games_registry.md)。
- 想启用 RAG、原文索引或剧情图谱：读 [上下文系统](context_systems.md)。
- 想做角色关系或语义分析：读 [关系与语义分析](relation_analysis.md)，内部模块见 [`relation_analyzer/README.md`](../relation_analyzer/README.md)。
- 想确认项目成熟度、边界和安全注意事项：读 [项目说明](project_notes.md)。
- 想参与改动或让 AI 按仓库规则开发：读根目录 [CONTRIBUTING.md](../CONTRIBUTING.md)（含 **CLI / GUI 同步** 约定）。

## 文档分组

### 配置与运行

- [安装与本地配置](setup.md)：本地私有配置、游戏 `work` 目录、Ren'Py SDK / TL 模板和运行模式。
- [GUI 工作台](gui_workbench.md)：可选 PySide6 图形界面的安装、任务模式（批量翻译 / 同步翻译 / 关键词 / 订正 / 翻译 A/B 对比）、诊断页和写回安全边界。
- [工作区项目总表](games_registry.md)：`games_registry.json` 与 `GAMES.md`、CLI / GUI 刷新、与写回和 environment check 的边界。

### Batch 与安全

- [Batch 工作流与安全检查](batch_workflows.md)：命令注意事项、`check/apply` 安全闸门、订正、关键词提取、manifest identity v2 和 golden corpus 测试。
- [上下文系统](context_systems.md)：RAG history store、Batch source-only index、Structured Story Memory 和 RAG store benchmark。

### 分析与项目状态

- [关系与语义分析](relation_analysis.md)：`extract_relations.py` 的 relation / semantic 模式和 Story Memory seed 导出。
- [项目说明](project_notes.md)：环境要求、当前边界、项目状态、安全说明和适用人群。
- [CONTRIBUTING.md](../CONTRIBUTING.md)：贡献与开发约定（CLI / GUI 同步、完成定义、测试要求）。

### 相邻参考

- [`relation_analyzer/README.md`](../relation_analyzer/README.md)：关系分析模块内部划分。
- [`macro_setting.example.md`](../macro_setting.example.md)：Batch macro setting 模板。
- [`story_graph.example.json`](story_graph.example.json) 与 [`story_graph.schema.json`](story_graph.schema.json)：Structured Story Memory 示例和 schema。
