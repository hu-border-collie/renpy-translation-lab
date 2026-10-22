# 测试目录与运行方式

测试按生产模块的职责分组，文件仍保留 `test_*.py` 名称：

| 目录 | 范围 |
| --- | --- |
| `cli/`、`runtime/` | CLI 合约、运行时配置和集成行为 |
| `foundation/`、`context_engine/`、`providers/` | 基础安全、上下文和供应商合同 |
| `quality_review/`、`execution/` | 质量检查、订正、任务执行和恢复 |
| `engine_adapters/`、`relation_analyzer/` | 引擎适配与关系分析 |
| `gui/` | 按应用壳、设置、工作台、workflow、报告、弹窗、worker、控件和外观分组的 GUI 相关测试 |
| `tooling/` | 质量门禁、依赖锁、发布、公开快照和开发工具 |
| `support/` | runner、GUI 守卫、运行时隔离等共用辅助代码 |
| `fixtures/` | 原创、可公开的输入和 golden 文件，不参与测试发现 |

完整套件入口不变：

```powershell
python -B tests/run_cli_tests.py -q
python -B tests/run_gui_tests.py -q
```

两套 runner 递归查找模块，并按完整包名加载。`test_gui_*.py` 与
`test_settings_*_page.py` 由带模态弹窗守卫的 GUI runner 执行；其他测试归 CLI/core。
因此，`gui/settings/` 中纯协议、协调器、注册表和保存逻辑的测试仍由 CLI runner 执行。

定向运行使用新的完整模块名：

```powershell
python -B -m unittest tests.providers.test_model_profile
python -B -m unittest tests.gui.workflows.test_gui_translation_workflow
python -B -m unittest tests.gui.settings.test_settings_profiles_page
```

需要完全无窗口运行时设置 `QT_QPA_PLATFORM=offscreen`。GUI 辅助代码使用
`from tests.support import gui_test_support`，具体守卫与窗口清理约定见
[gui_test_support.py](support/gui_test_support.py)。

新增测试子目录需带 `__init__.py`，以便标准 unittest discovery 与仓库 runner 使用相同模块身份。
复用其他测试的 fixture 时导入其模块，再访问 helper；直接导入 `TestCase` 类会使 unittest
把它作为当前模块的测试再次发现。[runner 回归测试](tooling/test_test_runners.py)
会验证递归发现、GUI/CLI 分流、重复和遗漏。
