# 贡献与开发约定

本文档是仓库的正式开发约定。无论是人工改代码、AI 辅助改代码，还是作者自己维护，都应遵守。

## CLI 与 GUI 必须同步

本仓库同时提供 **CLI**（`gemini_translate_batch.py` / `gemini_translate.py`）和 **可选 GUI**（`gui_qt/`）。  
CLI 是事实来源和高级用户主路径，但 **GUI 不是二等公民**：任何面向用户的功能改动，默认都要在两边可用，除非明确记录为「仅 CLI」或「仅 GUI」的例外。

### 什么叫「同步」

| 层级 | CLI | GUI |
|------|-----|-----|
| 核心逻辑 | 实现在可复用模块或 CLI 脚本中 | 调用同一套 Python API，或通过 `QProcess` 执行同一 CLI 子命令与参数 |
| 用户可见行为 | 命令、参数、输出、错误语义 | 工作台摘要、「诊断与工具」、设置项、失败提示与 CLI 一致 |
| 配置 | `translator_config.json` 等 | 设置页可读写同一配置键；workflow 构造 CLI 参数时不得漏传 |
| 测试 | `tests/test_*.py` | `tests/test_gui_*.py` 覆盖对应 GUI 包装层 |

### 推荐实现顺序

1. **先写可复用核心**  
   优先放在 CLI 侧模块（例如 `gemini_translate_batch.py`、`batch_cost_estimate.py`、`story_memory.py`），或抽到 CLI/GUI 共用的纯函数模块（例如 `gui_qt/batch_workflow_support.py` 只做参数与展示辅助，核心估算仍来自 `batch_cost_estimate.py`）。

2. **再接 CLI 入口**  
   子命令、参数、manifest 字段、doctor 报告等。

3. **最后接 GUI**  
   - 批量类流程：`translation_workflow.py` / `keyword_workflow.py` / `revision_workflow.py` / `split_batch_workflow.py`  
   - 环境检查：`doctor_worker.py` + `doctor_report.py` + `user_copy.py`  
   - 设置：`settings_schema.py`  
   - 诊断：`diagnostics_context.py`  
   - 应用壳：`app.py`

4. **两边一起测**  
   至少各有一条自动化测试；能跑 unittest 时不要只用手动点 GUI 代替。

### 完成定义（Definition of Done）

一个功能 PR / 改动在以下都满足前 **不算完成**：

- [ ] CLI 行为已实现，并有对应测试或 golden/smoke 覆盖（视改动类型而定）
- [ ] GUI 能触发同一能力，或展示同一诊断/配置结果
- [ ] 新增配置项在 `translator_config.example.json` 与 GUI 高级设置中同步出现（如适用）
- [ ] 用户可见中文文案在 `user_copy.py` 或相关 report 模块中有对应翻译（如适用）
- [ ] 未同步的一方在 PR 描述中 **明确写出原因** 和后续 issue（仅允许有意的例外）

### 常见落点对照

| 能力类型 | CLI | GUI |
|----------|-----|-----|
| 新子命令 | `gemini_translate_batch.py` `build_arg_parser()` / `main()` | `diagnostics_context.py` 命令参考；workflow 的 `args` |
| doctor 检查项 | `collect_doctor_report()` | `doctor_report_to_parsed()`、警告翻译、`summarize_doctor_report()` |
| 成本 / 限额 / 闸门 | 命令行 flag 或 manifest 字段 | 设置项 + workflow 传参 + 摘要 facts |
| 进度 / ETA | 标准输出（如有） | `*_report.py` / `workflow_progress.py` 等解析与展示 |
| Gemini 内置模型列表 | `gemini_model_catalog.py`（单一源） | 同上；用户扩展写 `translator_config.json` → `model_catalog` |

### 允许的例外（须在 PR 中说明）

- **仅 CLI**：调试子命令、一次性迁移工具、明确标注「高级/脚本-only」且 GUI 无等价入口需求。
- **仅 GUI**：纯界面主题、布局、字体、窗口行为，不改变翻译语义。
- **分阶段交付**：允许先合并核心 + CLI，但 **必须** 在同一 epic 内跟 GUI PR，或先开带 `gui` 标签的跟踪 issue；不得无限期搁置。

## 其他约定

### Docstring 约定

本仓库 **不追求全库高 docstring 覆盖率**（历史代码多数无 docstring）。CodeRabbit 的 docstring pre-merge check 在 `.coderabbit.yaml` 中配置为 **advisory**（`mode: warning`，`threshold: 0`），避免 PR 上出现「0.72% vs 80%」一类误导性失败项。

请在以下情况补充 docstring：

- 新增或显著改动的 **公共 API**（模块级函数、类、workflow 入口、可被 GUI/CLI 共用的 helper）
- 行为非显而易见的配置解析、写回安全、manifest 字段含义

不要求为：

- 一次性脚本、测试夹具、纯 UI 文案常量
- 仅为通过 bot 阈值而批量补注释

若将来要提高覆盖率，应 **分阶段调高 `threshold`**，而不是一次性要求 80%。

- **配置向后兼容**：改 `translator_config.json` 结构时更新 `translator_config.example.json` 与 `gui_qt/settings_schema.py`。
- **写回安全**：不得绕过 `check -> apply` 合约；GUI 写回按钮仍须服从 `safe` / `warn` / `block` 语义。
- **不要提交私有文件**：见 [docs/project_notes.md](docs/project_notes.md) 安全说明。
- **测试**：改动后运行相关 `python -m unittest tests.test_*`；GUI 改动优先跑 `tests.test_gui_*`。
- **依赖**：直接版本只在对应的 `requirements-*.txt` 权威输入中维护；依赖升级应单独成 PR，并按[依赖输入与哈希锁](docs/dependencies.md)重生成全部锁。不要手改 `requirements-lock/`。
- **质量门禁**：PR 会跑 `scripts/run_quality_gates.py all`（lint / 类型检查 / 依赖审计）。工具版本在 `requirements-dev.txt`，配置在 `pyproject.toml`，漏洞例外在 `quality/pip-audit-exceptions.json`。blocking 范围刻意收窄；扩大范围前先测量基线并单独 PR。说明见 [CI 与定时集成检查](docs/ci.md)。

## 相关文档

- [项目说明](docs/project_notes.md)
- [GUI 工作台](docs/gui_workbench.md)
- [Batch 工作流与安全检查](docs/batch_workflows.md)
