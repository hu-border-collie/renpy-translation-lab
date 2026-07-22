# Agent / CLI 快速开始

文档地图：[docs/README.md](README.md)

本文面向**使用本工具完成翻译任务**的自动化 Agent。若任务是修改 Ren'Py Translation Lab 本身，请改读根目录 [AGENTS.md](../AGENTS.md) 和 [CONTRIBUTING.md](../CONTRIBUTING.md)。

CLI 是自动化操作的事实来源。必要时可以使用 GUI 做可视验证，但不要依赖点击界面完成可重复的无人值守流程。

## 操作原则

- 使用 Python 3.11 或更高版本，在仓库根目录运行命令。
- 先检查当前项目、配置和 Git 状态；只在游戏副本或已有备份的目录中工作。
- 先运行只读的 `doctor`，不要直接提交任务。
- 用 `python gemini_translate_batch.py --help` 和子命令 `--help` 读取当前参数，不要根据其他项目猜命令。
- 记录 `build` 输出的确切 manifest 路径，并在后续命令中显式传入；不要依赖“最新任务”推断。
- 只有 `check` 对当前 manifest/results 返回 `safe` 时才执行 `apply`。
- 不要用 `--force` 规避安全判断；它只处理有限的重复/恢复场景，不能绕过 stale check、源快照校验或 `block`。

## 1. 安装核心依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python gemini_translate_batch.py --version
python gemini_translate_batch.py --help
```

Linux / macOS 激活命令为 `source .venv/bin/activate`。CLI 不需要安装 `requirements-gui.txt`。

## 2. 准备配置与项目

默认本地文件为：

| 位置 | 文件 | 用途 |
|---|---|---|
| 工具仓库根目录 | `api_keys.json` | Gemini API Key；也可使用 `GEMINI_API_KEY`、`GEMINI_API_KEY_2`、`GEMINI_API_KEY_3` 环境变量 |
| 工具仓库根目录 | `translator_config.json` | 当前 `game_root`、模型、过滤器和运行参数 |
| 当前游戏 `work` 目录 | `glossary.json` | 项目术语，可选 |
| 当前游戏 `work` 目录 | `macro_setting.md` | 角色口吻和世界观约束，可选 |
| 当前游戏 `work` 目录 | `project_context_settings.json` | 当前项目的 Batch RAG / 原文索引开关，可选 |

首次配置时，可从仓库中的 `*.example.*` 文件复制；如果目标文件已存在，先读取并保留现有值，不要直接覆盖。`translator_config.json.game_root` 应指向游戏的 `work` 目录，例如：

```json
{
  "game_root": "C:/games/Game_Example/work"
}
```

典型项目结构：

```text
Game_Example/
├─ original/
├─ work/
│  └─ game/
│     └─ tl/
│        └─ <language>/
└─ build/
```

如果只有 `original/game` 且 `work` 不存在或为空，可运行：

```powershell
python gemini_translate_batch.py bootstrap-work
```

这只创建工作副本，不生成 TL 模板，也不调用 Gemini。生成模板所需的 Ren'Py SDK 和 prepare 配置见 [安装与本地配置](setup.md)。不要运行来源不明的 `prepare.unpack_command` 或 `prepare.template_command`；`translator_config.json` 是可执行的本地配置。

## 3. 只读预检

```powershell
python gemini_translate_batch.py doctor
```

`doctor` 不调用 Gemini，也不写回 `.rpy`。确认输出中的 `game_root`、`tl_subdir`、目标语言和待译数量符合预期。遇到阻塞项时先修复；可选建议不应被误判为强制失败。

## 4. 执行安全 Batch 流程

先构建本地任务包：

```powershell
python gemini_translate_batch.py build
```

从输出中取得 manifest，例如 `logs/batch_jobs/<package>/manifest.json`。后续始终使用同一个确切路径：

```powershell
python gemini_translate_batch.py submit logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py status logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py download logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py check logs/batch_jobs/<package>/manifest.json
```

执行约束：

- 提交前按任务要求设置并确认成本上限；参数以 `submit --help` 为准。
- `status` 显示云端任务成功后才运行 `download`；仍在运行时等待并再次查询，不要重复提交。
- `check` 是干跑校验，不修改 `.rpy`；它会输出 `safe / warn / block`，并把失败报告写入任务包目录。
- `warn` 或 `block` 时停止写回，阅读 `check_failures.jsonl` 及命令输出，再按 [Batch 工作流与安全检查](batch_workflows.md) 使用 retry、repair 或 revision 流程。

只有检查明确为 `safe` 时才执行：

```powershell
python gemini_translate_batch.py apply logs/batch_jobs/<package>/manifest.json
```

`apply` 写回前会再次验证 manifest、results 和当前源文本；任一方漂移都应视为需要重新检查，而不是绕过。

## 5. 完成与交付

- 检查 `apply` 摘要和目标 `.rpy` diff。
- 在 Ren'Py 中运行 lint 或项目既有 smoke test。
- 报告使用的 manifest、最终安全等级、写回结果和仍未处理的失败项。
- 不提交 `api_keys.json`、`translator_config.json`、私有游戏脚本、`logs/` 或 Batch 结果到公开仓库。

完整子命令和恢复流程见 [Batch 工作流与安全检查](batch_workflows.md)；配置、SDK 与目标语言见 [安装与本地配置](setup.md)。
