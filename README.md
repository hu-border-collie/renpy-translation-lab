# Ren'Py Translation Lab

一个面向 Ren'Py 视觉小说的翻译工作台，聚焦 Gemini Batch 作业流、上下文增强、轻量 RAG 记忆层和写回前安全校验。

**当前状态：稳定版（v1.0.0）。** 核心 Batch 流程已经在约 11 万英文词规模的真实 Ren'Py 项目上完整跑通：从预建上下文、生成 Batch 包、提交和下载结果，到 `check` 安全校验、retry 合并和 `apply` 写回。CLI 仍是事实来源和高级用户主路径；仓库另提供稳定的可选图形工作台降低普通流程门槛。当前正式交付方式是从源码安装运行，暂不提供零配置安装包。版本变化见 [CHANGELOG.md](CHANGELOG.md)。

## 这是什么

这个仓库主要提供：

- `gemini_translate.py`：同步翻译入口，适合直接运行、补译、局部修复和 smoke test。
- `gemini_translate_batch.py`：当前更推荐的 Batch 入口，覆盖 `build / submit / status / probe / download / check / apply / split / repair`。
- `rag_memory.py`：本地 JSON history store、文本哈希和相似度检索。
- `story_memory.py`：可选结构化剧情记忆，从本地 `story_graph.json` 注入角色、关系、术语和场景上下文。
- `translator_runtime.py`：同步脚本和 Batch 脚本共用的配置、SDK、校验、响应解析和文件处理 runtime。
- `extract_relations.py`：独立的关系 / 语义分析入口。

稳定版覆盖 Ren'Py 翻译工作台本身；一键安装、游戏解包/重新打包和面向普通用户的零配置发行体验不在当前支持范围内。

**可选图形工作台**：

除 CLI 外，仓库提供可选桌面界面。当前界面使用统一左侧导航，按 **工作流 / 翻译资产 / 系统** 分组：项目与环境、批量翻译、同步翻译、关键词/术语、订正、上下文库和设置各有独立入口；项目路径、切换、环境检查与工作目录准备集中在「项目与环境」，任务页只展示本任务的操作、进度与结果。页眉「运行日志」会打开「诊断与运行日志」，集中展示任务上下文、命令参考、任务记录与原始输出。批量翻译主路径已有真实项目烟测，其余模式已实现并有自动化测试；图形依赖单独安装，不进入主 `requirements.txt`；**不装图形界面时，命令行可照常使用**。

```powershell
pip install -r requirements-gui.txt
python -m gui_qt
```

批量翻译写回仅在检查显示**可写回**（CLI 的 `safe`）时启用；需处理问题收在结果区 **问题处理** 折叠内。同步翻译、关键词、订正与上下文库各有独立导航页。安装步骤、各模式说明与安全边界见 [GUI 工作台](docs/gui_workbench.md)。完整文档地图见 [docs/README.md](docs/README.md)。Epic #157 的设计过程稿（历史）见 [GUI IA 重组](docs/gui_ia_redesign.md)。参与开发请遵守 [CONTRIBUTING.md](CONTRIBUTING.md)（含 CLI / GUI 同步约定）。

工作区根目录可用 **`games_registry.json`** 维护多项目总表并生成 `GAMES.md`；GUI 可在 **项目与环境** 或 **设置 → 工作区** 浏览与切换。说明见 [工作区项目总表](docs/games_registry.md)。

## 核心能力

- 扫描 Ren'Py `game/tl/<language>/` 下的 `.rpy` 文件（默认 `schinese`），抽取待翻译条目并跳过 `old`。
- 自动预处理项目，包括提取脚本和生成 `tl/<language>` 模板（由 `translator_config.json` 的 `tl_subdir` 与 `prepare.language` 决定）。
- 构造带 glossary、macro setting、RAG 和可选 Story Memory 的 Gemini 请求。
- 生成 Batch 请求包并执行完整异步流程。
- 用 `check/apply/repair` 对写回做快照校验和安全分级；`apply` 默认只接受最近一次 `safe` check 对应的结果。
- 从已有译文、全文原文或结构化剧情图谱中补充上下文。
- 支持订正、关键词候选提取和独立关系 / 语义分析工作流。

## 快速开始

### 1. 安装

正式发行版可从 [GitHub Releases](https://github.com/hu-border-collie/renpy-translation-lab/releases) 下载 `source.zip` 并按随附 SHA-256 文件校验；也可以克隆对应 tag。GUI 推荐字体改为从发布者官方来源按需下载，详见 [GUI 工作台文档](docs/gui_workbench.md#可选字体)。

```bash
git clone https://github.com/hu-border-collie/renpy-translation-lab.git
cd renpy-translation-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

上述命令使用人工维护的直接依赖入口。Python 3.11 的 Windows/Linux 可复现安装可改用提交的 `requirements-lock/py311-<platform>-cli.txt`；GUI 与 LiteLLM 也有独立 profile。详见[依赖输入与哈希锁](docs/dependencies.md)。

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 准备本地配置

先在仓库根目录复制工具级配置：

- `api_keys.example.json` -> `api_keys.json`
- `translator_config.example.json` -> `translator_config.json`

再把项目资产放到目标游戏的 `work` 目录：

- `glossary.example.json` -> `<work>/glossary.json`
- `macro_setting.example.md` -> `<work>/macro_setting.md`（可选）

至少需要提供 Gemini API key，并让 `translator_config.json` 中的 `game_root` 字段，或环境变量 `GAME_ROOT` / `SA_GAME_ROOT` 指向目标游戏的 `work` 目录。`glossary_file` 与 `batch.macro_setting_file` 应指向该项目的资产；GUI 切换项目时会同步这些路径。详细配置说明见 [安装与本地配置](docs/setup.md)。

### 3. 准备项目目录

脚本默认处理某个游戏的 `work` 目录，典型结构如下：

```text
Game_Example/
├─ original/
├─ work/
│  └─ game/
│     └─ tl/
│        └─ <language>/   # 默认 schinese；可配置为 japanese、korean 等
└─ build/
```

目标语言与 TL 路径由 `translator_config.json` 配置；`doctor` 与 `build` 会打印当前 `tl_subdir` 和 `prepare.language`。详见 [安装与本地配置](docs/setup.md)。

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
# 检查命令输出给出的 preview.diff 后，再显式写回：
python gemini_translate.py --apply logs/sync_runs/<run>/manifest.json
```

Batch 模式：

```bash
python gemini_translate_batch.py doctor
python gemini_translate_batch.py bootstrap-rag
python gemini_translate_batch.py bootstrap-source-index
python gemini_translate_batch.py build
python gemini_translate_batch.py probe
python gemini_translate_batch.py submit
python gemini_translate_batch.py status
python gemini_translate_batch.py download
python gemini_translate_batch.py check
python gemini_translate_batch.py apply
```

说明：

- `python gemini_translate.py --help` 会显示同步脚本的最小 CLI 帮助。
- 同步脚本默认只在 `logs/sync_runs/` 生成源文件快照、候选文件和统一 diff，不修改 `.rpy`；只有显式 `--apply MANIFEST` 才会在项目、源文件和制品复核通过后原子写回。
- 若预览前需要执行配置中的解包或模板生成步骤，显式使用 `python gemini_translate.py --prepare`；普通预览不会隐式运行这些会修改工作目录的准备步骤。
- `gemini_translate_batch.py` 需要显式子命令；不带子命令会打印帮助并退出。
- Batch 产物默认会写到本地 `logs/` 目录。
- `probe` 是可选的提交前小样本检查；确认 API、模型和请求格式正常后再 `submit`。
- `check` 是干跑校验，不会修改 `.rpy`；`apply` 默认只接受最近一次 `safe` check 对应的结果。
- `bootstrap-rag` 用已有译文暖 history store；`bootstrap-source-index` 用全文原文建立 source-only 索引。
- Structured Story Memory 默认关闭；需要通过 `batch.story_memory.enabled=true` 或 `sync.story_memory.enabled=true` 启用。

## 文档索引

完整导航见 [文档地图](docs/README.md)。常用入口：

- 配置与运行
  - [安装与本地配置](docs/setup.md)：全局配置、**项目级**上下文开关、SDK / TL 模板。
  - [GUI 工作台](docs/gui_workbench.md)：统一侧边导航、项目与环境、任务页、设置、诊断与运行日志、写回安全边界。
- Batch 与安全
  - [Batch 工作流与安全检查](docs/batch_workflows.md)：manifest / identity v2、`check/apply`、订正、关键词、golden corpus。
  - [上下文系统](docs/context_systems.md)：RAG、原文索引、Story Memory、store 路径。
- 分析与项目状态
  - [关系与语义分析](docs/relation_analysis.md)
  - [项目说明](docs/project_notes.md)
  - [CONTRIBUTING.md](CONTRIBUTING.md)

## 当前状态

当前项目状态为**稳定版**。核心 Batch 翻译链路、写回安全闸门、CLI 与可选 GUI 工作台可以按文档用于真实 Ren'Py 项目；适合熟悉 Ren'Py 目录结构、能维护本地配置、能阅读检查报告的使用者。

“稳定版”表示当前支持范围和已记录工作流具备稳定使用基线，不表示零配置安装器、所有可选实验能力或任意环境均已覆盖。也不应直接在唯一原项目上整批写回；执行任何会修改项目文件的操作前，请先备份，并优先在副本上测试。

2026-06-19 已在一部约 3,300 待译行的真实 Ren'Py 项目上完成一次 GUI 主路径烟测：预处理、原文索引预建、GUI 启动批量翻译、排队后继续、下载、检查、可写回状态下写回和游戏启动验证均跑通；首轮检查为「需处理」时 GUI 正确禁用写回，问题经 CLI / 人工修正后重新检查为「可写回」，最终写回 6 个文件、约 3,300 处译文行。该烟测不等于完整游戏 QA，仍未覆盖语言切换入口、全流程试玩、文本溢出和润色质量。

2026-07-14 已在隔离的最小 Ren'Py 项目副本上完成一次 **LiteLLM + DeepSeek 同步翻译**真实供应商烟测：系统凭据读取、模型选择、同步启动、JSON 返回兼容和 5/5 条译文写入均成功，`[player_name]` 与 Ren'Py color tag 保持不变。该烟测验证的是小规模同步替代路径，不代表完整项目质量、成本或吞吐量验证。

如果使用图形工作台，可按左导航选择流程：

- **批量翻译**（已有 2026-06-19 真实项目烟测）：`选择项目 -> 设置 -> 环境检查 -> 可选上下文库预建 -> 翻译进度(开始翻译) -> 写回(检查/写回)`
- **同步翻译**（已有 2026-07-14 LiteLLM + DeepSeek 小规模真实供应商烟测）：调用 `gemini_translate.py` 并可显式选择 Gemini 或 LiteLLM；默认生成 diff 预览，确认后再显式写回
- **关键词 / 术语**（批量|同步）：只生成报告，经「合并到 glossary」写入术语表（GUI 已实现，尚无独立真实项目烟测）
- **订正**（批量|同步）：预览后通过「写回订正」单独写回，与普通翻译写回分离（GUI 已实现，尚无独立真实项目烟测）
- **上下文库**：状态卡上预建记忆库 / 原文索引（须先在设置 · 上下文开启并保存）

批量翻译路径下，`doctor` 在准备阶段独立运行；「开始翻译」会协调 `build -> submit -> status -> download -> check`；只有检查为「可写回」（`safe`）后才通过结果区「写回翻译」调用 `apply`。若检查为「需处理」或「禁止写回」，不要强行写回；在 **问题处理** 中查看清单、补译/修补或重新检查，或回到 CLI 处理。

高级用户或需要手动指定 manifest 时，CLI 推荐顺序仍是：

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
