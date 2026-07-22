# Agent 开发约定

本文件面向**修改本仓库代码或文档**的 Agent。若目标是使用本工具完成一次翻译任务，请改读 [Agent / CLI 快速开始](docs/quickstart_agent.md)。完整开发规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 开始前

- 使用 Python 3.11 或更高版本。
- 先阅读任务对应的 issue、当前 checkout 和 `git status`；保留用户已有改动，不要混入无关重构。
- 以当前代码、`--help` 和现行文档为准；`docs/archive/` 只作历史参考。
- 不要提交 API Key、本地配置、私有游戏脚本、Batch 结果或 `logs/`。

## 实现约束

- CLI 是行为事实来源，但 GUI 不是二等公民。新增面向用户的能力默认要同步 CLI、GUI、配置、文案和测试；仅 CLI / 仅 GUI 的例外须明确说明。
- 优先把核心行为放进可复用模块，再接 CLI 入口和 GUI 包装层。不要在 GUI 中另写一套翻译语义。
- 不得绕过 Batch `check -> apply` 合约。只有与当前 manifest/results 匹配的最近一次检查为 `safe` 时才允许写回；`apply --force` 也不能绕过 stale check、源快照校验或 `block`。
- `translator_config.json` 属于可执行的本地配置；不要运行来源不明的 prepare 自定义命令。
- 直接依赖只在对应的 `requirements-*.txt` 权威输入中维护。不要手改 `requirements-lock/`；依赖升级应单独提交，并用既有生成器重建全部锁。

## 同步修改点

- 改 CLI 子命令、参数或输出：同步 argparse 帮助、相关 workflow、GUI「诊断与运行日志」命令参考和测试。
- 改 GUI 入口或用户用语：同步 `gui_qt/user_copy.py`、现行 GUI 文档和相关 GUI 测试。
- 改 doctor 规则：同步 doctor report/user copy、`docs/doctor_recommendations.md`、状态矩阵和测试。
- 改配置结构：同步 `translator_config.example.json`、runtime reader、`gui_qt/settings_schema.py`、GUI 设置页和文档。
- 新增或显著改动公共 API、非显而易见的配置解析或写回安全逻辑时补充 docstring；不要为追求覆盖率批量添加无信息注释。

## 验证

按改动范围先运行针对性测试，再决定是否扩到完整门禁：

```powershell
python -m unittest tests.test_<module>
python -B tests/run_cli_tests.py -q
python -B tests/run_gui_tests.py -q
python scripts/run_quality_gates.py all
git diff --check
```

纯文档改动至少检查所有 tracked 本地 Markdown 链接和 `git diff --check`。不要用手动 GUI 点击代替已有自动化测试。
