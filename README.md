# Ren'Py Translation Lab

面向 Ren'Py 视觉小说的翻译工作台：Gemini Batch 作业流、上下文增强、轻量 RAG 记忆层，以及写回前安全校验。

**稳定版（v1.0.0）。** GUI 是普通用户的推荐入口；CLI 是 Agent、脚本、CI 与高级用户的自动化事实来源。正式交付为源码安装运行，暂无零配置安装包。版本变化见 [CHANGELOG.md](CHANGELOG.md)。

## 这是什么

- **GUI**（`python -m gui_qt`）：普通用户的推荐入口，覆盖项目准备、翻译、检查、问题处理和安全写回。
- **Batch CLI**（`gemini_translate_batch.py`）：Agent 与自动化主路径，覆盖 `build / submit / status / probe / download / check / apply / split / repair` 等。
- **同步 CLI**（`gemini_translate.py`）：小范围即时翻译、补译、局部修复与 smoke test。
- **上下文**：glossary / macro setting、RAG（`rag_memory.py`）、可选 Story Memory（`story_memory.py`）。
- **共用 runtime**（`translator_runtime.py`）：配置、SDK、校验、响应解析与文件处理。
- **可选分析**（`extract_relations.py` / `relation_analyzer/`）：关系与语义分析。
- **可独立使用的 CLI**：不安装 GUI 依赖时，Agent、脚本和高级用户仍可完成全部核心工作流。

稳定范围是翻译工作台本身；一键安装、游戏解包/重打包与面向普通用户的零配置发行不在当前支持范围内。边界与烟测记录见 [项目说明](docs/project_notes.md)。

首次使用请按角色选择 [GUI 快速开始](docs/quickstart_gui.md) 或 [Agent / CLI 快速开始](docs/quickstart_agent.md)。完整 GUI 说明与写回安全边界见 [GUI 工作台](docs/gui_workbench.md)；修改仓库前请阅读 [AGENTS.md](AGENTS.md) 与 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 核心能力

- 扫描 `game/tl/<language>/`（默认 `schinese`），抽取待译条目并跳过 `old`。
- 按配置预处理项目（脚本提取、`tl` 模板生成）。
- 构造带 glossary、macro setting、RAG 与可选 Story Memory 的请求。
- 完整 Batch 异步流程，以及 `check` / `apply` / `repair` 写回闸门（`apply` 默认只接受最近一次 `safe` check）。
- 订正、关键词候选提取，以及独立关系 / 语义分析工作流。

## 快速开始

| 目标 | 从这里开始 |
|---|---|
| 通过图形界面完成第一次翻译 | [GUI 快速开始](docs/quickstart_gui.md) |
| 让 Agent 或脚本通过 CLI 操作 | [Agent / CLI 快速开始](docs/quickstart_agent.md) |
| 查完整 Batch 命令与恢复流程 | [Batch 工作流与安全检查](docs/batch_workflows.md) |
| 修改本仓库代码或文档 | [AGENTS.md](AGENTS.md) → [CONTRIBUTING.md](CONTRIBUTING.md) |

### 安装基础环境

发行版可从 [GitHub Releases](https://github.com/hu-border-collie/renpy-translation-lab/releases) 下载 `source.zip` 并按随附 SHA-256 校验；也可克隆对应 tag。

```bash
git clone https://github.com/hu-border-collie/renpy-translation-lab.git
cd renpy-translation-lab
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GUI 还需安装可选依赖并从模块入口启动：

```powershell
python -m pip install -r requirements-gui.txt
python -m gui_qt
```

可复现安装与哈希锁见 [依赖输入与哈希锁](docs/dependencies.md)。无论使用 GUI 还是 CLI，都应先运行环境检查；Batch 只有最近一次 `check` 对当前结果给出 `safe` 时才允许 `apply`。

## 文档

完整导航见 [文档地图](docs/README.md)。

| 主题 | 文档 |
|---|---|
| GUI 快速开始 | [quickstart_gui.md](docs/quickstart_gui.md) |
| Agent / CLI 快速开始 | [quickstart_agent.md](docs/quickstart_agent.md) |
| 安装与配置 | [setup.md](docs/setup.md) |
| GUI 工作台 | [gui_workbench.md](docs/gui_workbench.md) |
| Batch 与写回安全 | [batch_workflows.md](docs/batch_workflows.md) |
| 实际模型用量账本 | [model_usage_ledger.md](docs/model_usage_ledger.md) |
| RAG / 索引 / Story Memory | [context_systems.md](docs/context_systems.md) |
| 边界、安全与烟测 | [project_notes.md](docs/project_notes.md) |
| 项目沿革 | [project_history.md](docs/project_history.md) |
| Agent 开发约定 | [AGENTS.md](AGENTS.md) |
| 贡献与 CLI/GUI 同步 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 历史设计/审计稿 | [docs/archive/](docs/archive/README.md) |

## 安全提示

执行任何会修改项目文件的操作前，请先备份，并优先在副本上测试。不要把 API key、本地配置、私有游戏脚本、batch 结果或日志提交到公开仓库。详见 [项目说明 · 安全](docs/project_notes.md#安全说明)。
