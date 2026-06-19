# Ren'Py Translation Lab

一个面向 Ren'Py 视觉小说的翻译工作台，聚焦 Gemini Batch 作业流、上下文增强、轻量 RAG 记忆层和写回前安全校验。

当前核心 Batch 流程已经在约 11 万英文词规模的真实 Ren'Py 项目上完整跑通：从预建上下文、生成 Batch 包、提交和下载结果，到 `check` 安全校验、retry 合并和 `apply` 写回。CLI 仍是事实来源和高级用户主路径；仓库另提供可选 PySide6 GUI 降低普通流程门槛，但还不是零配置安装包产品。

## 这是什么

这个仓库主要提供：

- `gemini_translate.py`：同步翻译入口，适合直接运行、补译、局部修复和 smoke test。
- `gemini_translate_batch.py`：当前更推荐的 Batch 入口，覆盖 `build / submit / status / probe / download / check / apply / split / repair`。
- `rag_memory.py`：本地 JSON history store、文本哈希和相似度检索。
- `story_memory.py`：可选结构化剧情记忆，从本地 `story_graph.json` 注入角色、关系、术语和场景上下文。
- `translator_runtime.py`：同步脚本和 Batch 脚本共用的配置、SDK、校验、响应解析和文件处理 runtime。
- `extract_relations.py`：独立的关系 / 语义分析入口。

如果你想找的是一键打包、面向普通用户的整套发行流程，这个仓库还不是那个方向。

**实验性可选 GUI**：

仓库包含一个使用 PySide6 的可选桌面 GUI（`gui_qt/`）。它作为现有 CLI 和 JSON 配置之上的外壳层，依赖单独安装，不进入主 `requirements.txt`；未安装 PySide6 时 CLI 不受影响：

```powershell
pip install -r requirements-gui.txt
python -m gui_qt
```

界面分为 **工作台**、**配置**、**诊断日志** 三个顶层 Tab。普通主流程：

```text
选择项目 -> 配置 API / 模型 -> 环境检查 -> 开始翻译 -> 查看写回状态 -> 写回翻译
```

当前 GUI 已覆盖：

- 选择 game/work 目录（写入 `translator_config.json`）
- 管理 API Key（兼容 `api_keys.json`；环境变量 Key 只读提示）
- 配置真实模型名称、embedding model、Batch thinking level
- 开关 Batch 上下文（RAG / 原文索引），并提供 **预建 RAG 库**、**预建原文索引** 图形入口
- 浅色 / 深色 / 跟随系统主题
- **工作台**内层 Tab：环境检查、翻译进度、写回（`safe / warn / block` 摘要；仅 `safe` 可写回）
- 一键编排基础 Batch 流程：`build -> submit -> status -> download -> check`；可从 latest manifest 继续长任务
- **诊断日志**：上半展示 manifest / package / job、已存在报告路径、可复制 CLI 命令与 manifest JSON 预览；下半保留原始终端输出

GUI 通过 `QProcess` 以参数列表调用 `gemini_translate_batch.py`，不重写翻译核心，也不向普通用户暴露 `apply --force`。详见 [GUI workbench](docs/gui_workbench.md) 和 issue #42。

## 核心能力

- 扫描 Ren'Py `game/tl/schinese` 下的 `.rpy` 文件，抽取待翻译条目并跳过 `old`。
- 自动预处理项目，包括提取脚本和生成 `tl/schinese` 模板。
- 构造带 glossary、macro setting、RAG 和可选 Story Memory 的 Gemini 请求。
- 生成 Batch 请求包并执行完整异步流程。
- 用 `check/apply/repair` 对写回做快照校验和安全分级；`apply` 默认只接受最近一次 `safe` check 对应的结果。
- 从已有译文、全文原文或结构化剧情图谱中补充上下文。
- 支持订正、关键词候选提取和独立关系 / 语义分析工作流。

## 快速开始

### 1. 安装

```bash
git clone https://github.com/hu-border-collie/renpy-translation-lab.git
cd renpy-translation-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 准备本地配置

复制示例配置并按本地项目修改：

- `api_keys.example.json` -> `api_keys.json`
- `translator_config.example.json` -> `translator_config.json`
- `glossary.example.json` -> `glossary.json`
- `macro_setting.example.md` -> `macro_setting.md`（可选）

至少需要提供 Gemini API key，并让 `translator_config.json`、`GAME_ROOT` 或 `SA_GAME_ROOT` 指向目标游戏的 `work` 目录。详细配置说明见 [Setup and local configuration](docs/setup.md)。

### 3. 准备项目目录

脚本默认处理某个游戏的 `work` 目录，典型结构如下：

```text
Game_Example/
├─ original/
├─ work/
│  └─ game/
│     └─ tl/
│        └─ schinese/
└─ build/
```

如果已有第三方汉化或非标准 TL 文件，先跑诊断：

```bash
python gemini_translate_batch.py doctor
```

`doctor` 不调用 Gemini，也不会写回 `.rpy`。

### 4. 运行

当前更推荐优先使用 Batch 模式；同步模式仍可用于小范围即时翻译或补修。

同步模式：

```bash
python gemini_translate.py
```

Batch 模式：

```bash
python gemini_translate_batch.py doctor
python gemini_translate_batch.py bootstrap-rag
python gemini_translate_batch.py bootstrap-source-index
python gemini_translate_batch.py build
python gemini_translate_batch.py submit
python gemini_translate_batch.py status
python gemini_translate_batch.py probe
python gemini_translate_batch.py download
python gemini_translate_batch.py check
python gemini_translate_batch.py apply
```

说明：

- `python gemini_translate.py --help` 会显示同步脚本的最小 CLI 帮助。
- `gemini_translate_batch.py` 需要显式子命令；不带子命令会打印帮助并退出。
- Batch 产物默认会写到本地 `logs/` 目录。
- `check` 是干跑校验，不会修改 `.rpy`；`apply` 默认只接受最近一次 `safe` check 对应的结果。
- `bootstrap-rag` 用已有译文暖 history store；`bootstrap-source-index` 用全文原文建立 source-only 索引。
- Structured Story Memory 默认关闭；需要通过 `batch.story_memory.enabled=true` 或 `sync.story_memory.enabled=true` 启用。

## 文档索引

- [Setup and local configuration](docs/setup.md)
  - 本地配置、项目目录、Ren'Py SDK / TL 模板和运行模式
- [Batch workflows and safety checks](docs/batch_workflows.md)
  - manifest / identity v2、`check/apply` 安全闸门、订正、关键词和 golden corpus 测试
- [Context systems](docs/context_systems.md)
  - RAG history store、Batch source-only index 和 Structured Story Memory
- [Relation and semantic analysis](docs/relation_analysis.md)
  - `extract_relations.py`、relation / semantic 模式
- [GUI workbench](docs/gui_workbench.md)
  - 可选 PySide6 GUI 的安装、三 Tab 布局、预建库、高级诊断页、安全写回边界和剩余验收项
- [Project notes](docs/project_notes.md)
  - 环境要求、当前边界、项目状态、安全说明和适用人群

## 当前状态

核心 Batch 翻译链路已经可以用于真实项目试跑，推荐口径是“高级用户稳定版”：适合熟悉 Ren'Py 目录结构、能维护本地配置、能阅读检查报告的使用者。

它仍不是已经打磨完成的零配置产品，也不应直接在唯一原项目上整批写回。执行任何会修改项目文件的操作前，请先备份，并优先在副本上测试。推荐使用顺序仍是：

```bash
python gemini_translate_batch.py doctor
python gemini_translate_batch.py build
python gemini_translate_batch.py submit logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py status logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py download logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py check logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py apply logs/batch_jobs/<package>/manifest.json
```

只有 `check` 输出 `safe` 时才继续 `apply`。如果出现 `warn` / `block`，先查看失败报告并使用 retry / repair / revision 等流程处理。
