# GUI workbench

本文档记录可选 PySide6 GUI 工作台的当前能力、安装方式、安全边界和剩余验收项。GUI 对应 issue #42，当前仍使用 `Refs #42`，不应因为阶段性 MVP 自动关闭该 issue。

## Status

当前 GUI 是实验性桌面工作台，定位是现有 CLI 和 JSON 配置之上的外壳层：

- GUI 依赖放在 `requirements-gui.txt`，不进入主 `requirements.txt`。
- 核心 CLI 不 import GUI；未安装 PySide6 时 CLI 仍可运行。
- GUI 通过 `QProcess` 调用 `gemini_translate_batch.py`，不重写翻译核心。
- 配置继续使用 `api_keys.json` 和 `translator_config.json`。
- 写回继续以 CLI 的 `check -> apply` 安全合约为事实来源。

已经合并到 `main` 的 GUI PR：

- #69：可选 PySide6 GUI 外壳、项目选择、doctor、实时日志、基础 API Key 管理。
- #70：真实模型名称、embedding model 和 Batch thinking level 配置。
- #71：配置保存边界修复，保留未知字段和既有 fallback。
- #72：doctor 普通语言摘要。
- #73：一键基础 Batch 翻译流程和 latest manifest 恢复。
- #74：检查结果摘要和 safe-only apply 写回保护。
- #75：浅色 / 深色 / 跟随系统主题。

## Install and launch

先安装核心依赖，再按需安装 GUI 依赖：

```powershell
pip install -r requirements.txt
pip install -r requirements-gui.txt
python -m gui_qt
```

如果未安装 PySide6，`python -m gui_qt` 会打印安装提示并退出；这不会影响 CLI。

## Main workflow

GUI 的普通主流程是：

```text
选择项目 -> 配置 API / 模型 -> 检查项目 -> 开始翻译 -> 检查结果 -> 写回翻译
```

对应的底层 CLI 仍是：

```text
doctor -> build -> submit -> status -> download -> check -> apply
```

工作台页提供项目选择、项目检查、翻译任务和写回状态。配置页管理 API Key、模型参数和外观主题。诊断日志页显示原始 CLI 输出，供排查失败时使用。

## Configuration compatibility

GUI 不引入新的主配置系统。

- API Key 仍保存到 `api_keys.json` 的 `api_keys` 列表。
- 模型、embedding model、Batch thinking level、GUI 主题等写入 `translator_config.json`。
- 保存配置时应保留未知字段，避免破坏高级配置。
- 如果 API Key 来自 `GEMINI_API_KEY` 等环境变量，GUI 只提示只读状态，不强行写回文件。

## Translation workflow

点击“开始翻译任务”后，GUI 会编排基础 Batch 流程：

```text
build -> submit -> status -> download -> check
```

如果 Batch 任务仍在处理中，GUI 会停在等待状态，用户稍后可以点击“继续最新任务”。恢复逻辑会读取 latest manifest，并校验它是当前项目的基础翻译任务；如果 latest manifest 属于其他项目或其他模式，会拒绝继续。

如果 build 已生成 package 但还没有 job，恢复会从 submit 继续，而不是错误地直接跑 status。

## Check and apply safety

GUI 写回按钮只在最近一次 check 为 `safe` 时启用。

- `safe`：允许进入写回确认，并调用 `gemini_translate_batch.py apply <manifest>`。
- `warn`：禁用写回，引导用户查看问题、retry、repair 或重新检查。
- `block`：禁用写回，要求修复源文件漂移或重新生成任务。
- 非零退出、未知 safety、已 apply 的 manifest：都不会启用写回。

GUI 不提供普通用户入口来运行 `apply --force`。`apply --force` 只用于绕过“已经 apply 过”的重复写回保护，不应被设计成确认 `warn` 的入口。

写回前确认框会提醒用户先在副本或备份上验证。即使 GUI 显示 `safe`，也不要在唯一原项目上直接整批写回。

## Current limits

当前 GUI 仍不是完整发行产品：

- 还没有打包安装器。
- 还没有完整多项目管理。
- 还没有完整可视化 diff 编辑器。
- 还没有 repair / retry 的图形化编排入口。
- 高级诊断页目前主要是原始日志，还没有完整展示 manifest、package、job、reports、可复制 CLI 命令和原始 JSON。
- README / 本文档说明当前能力和边界，但 #42 的最终关闭仍需要确认剩余验收项是否拆出新 issue 或已完成。

## Acceptance checklist

当前已经覆盖：

- GUI 可选依赖隔离。
- 核心 CLI 不 import GUI。
- 选择 game/work 目录。
- 读取 / 保存 `api_keys.json`，支持添加、删除、隐藏显示 API Key。
- 读取 / 保存真实模型名称和 embedding model。
- 保存 `translator_config.json` 时保留未知字段。
- 运行 doctor 并显示普通语言摘要。
- 一键编排基础 Batch 翻译流程。
- 长任务不阻塞 UI。
- 从 latest manifest 恢复状态。
- 将 `safe / warn / block` 映射为写回状态。
- 只在 `safe` 时启用 apply。
- `warn / block` 或 stale/unknown 状态不会盲目写回。
- 诊断日志可查看原始输出。
- 主题可跟随系统或手动切换。

仍建议在关闭 #42 前处理或明确拆出：

- 高级诊断页展示 manifest、package、job、reports、失败报告和可复制 CLI 命令。
- GUI 使用文档中的真实项目 smoke 截图或操作记录。
- 未安装 GUI 依赖时 CLI 正常运行的显式 CI/smoke 记录。
- README / 文档是否足以作为第一版 GUI 验收说明。