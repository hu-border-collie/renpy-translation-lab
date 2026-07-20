# Ren'Py Translation Lab

面向 Ren'Py 视觉小说的翻译工作台：Gemini Batch 作业流、上下文增强、轻量 RAG 记忆层，以及写回前安全校验。

**稳定版（v1.0.0）。** CLI 是事实来源和高级用户主路径；可选图形工作台降低常规流程门槛。正式交付为源码安装运行，暂无零配置安装包。版本变化见 [CHANGELOG.md](CHANGELOG.md)。

## 这是什么

- **Batch CLI**（`gemini_translate_batch.py`）：推荐主路径，覆盖 `build / submit / status / probe / download / check / apply / split / repair` 等。
- **同步 CLI**（`gemini_translate.py`）：小范围即时翻译、补译、局部修复与 smoke test。
- **上下文**：glossary / macro setting、RAG（`rag_memory.py`）、可选 Story Memory（`story_memory.py`）。
- **共用 runtime**（`translator_runtime.py`）：配置、SDK、校验、响应解析与文件处理。
- **可选分析**（`extract_relations.py` / `relation_analyzer/`）：关系与语义分析。
- **可选 GUI**（`python -m gui_qt`）：统一侧边导航（项目与环境、批量/同步翻译、关键词、订正、上下文库、设置）；不装 GUI 时 CLI 可独立使用。

稳定范围是翻译工作台本身；一键安装、游戏解包/重打包与面向普通用户的零配置发行不在当前支持范围内。边界与烟测记录见 [项目说明](docs/project_notes.md)。

```powershell
pip install -r requirements-gui.txt
python -m gui_qt
```

GUI 说明与写回安全边界见 [GUI 工作台](docs/gui_workbench.md)。开发约定（含 **CLI / GUI 同步**）见 [CONTRIBUTING.md](CONTRIBUTING.md)。多项目总表见 [工作区项目总表](docs/games_registry.md)。

## 核心能力

- 扫描 `game/tl/<language>/`（默认 `schinese`），抽取待译条目并跳过 `old`。
- 按配置预处理项目（脚本提取、`tl` 模板生成）。
- 构造带 glossary、macro setting、RAG 与可选 Story Memory 的请求。
- 完整 Batch 异步流程，以及 `check` / `apply` / `repair` 写回闸门（`apply` 默认只接受最近一次 `safe` check）。
- 订正、关键词候选提取，以及独立关系 / 语义分析工作流。

## 快速开始

### 1. 安装

发行版可从 [GitHub Releases](https://github.com/hu-border-collie/renpy-translation-lab/releases) 下载 `source.zip` 并按随附 SHA-256 校验；也可克隆对应 tag。

```bash
git clone https://github.com/hu-border-collie/renpy-translation-lab.git
cd renpy-translation-lab
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

可复现安装与哈希锁见 [依赖输入与哈希锁](docs/dependencies.md)。GUI 可选字体见 [GUI 工作台 · 可选字体](docs/gui_workbench.md#可选字体)。

### 2. 准备本地配置

仓库根目录：

- `api_keys.example.json` → `api_keys.json`
- `translator_config.example.json` → `translator_config.json`

目标游戏的 `work` 目录：

- `glossary.example.json` → `<work>/glossary.json`
- `macro_setting.example.md` → `<work>/macro_setting.md`（可选）

需提供 Gemini API key；`translator_config.json` 的 `game_root`（或环境变量 `GAME_ROOT` / `SA_GAME_ROOT`）应指向该 `work` 目录。详情见 [安装与本地配置](docs/setup.md)。

### 3. 项目目录

脚本默认处理某个游戏的 `work` 目录：

```text
Game_Example/
├─ original/
├─ work/
│  └─ game/
│     └─ tl/
│        └─ <language>/   # 默认 schinese
└─ build/
```

有第三方汉化或非标准 TL 时，先诊断（不调用 API、不写回 `.rpy`）：

```bash
python gemini_translate_batch.py doctor
```

### 4. 运行

**Batch（推荐）**：

```bash
python gemini_translate_batch.py doctor
python gemini_translate_batch.py bootstrap-rag            # 可选
python gemini_translate_batch.py bootstrap-source-index   # 可选
python gemini_translate_batch.py build
python gemini_translate_batch.py probe                    # 可选小样本
python gemini_translate_batch.py submit
python gemini_translate_batch.py status
python gemini_translate_batch.py download
python gemini_translate_batch.py check
python gemini_translate_batch.py apply                    # 仅在 check 为 safe 时
```

**同步**：

```bash
python gemini_translate.py
# 确认 preview.diff 后：
python gemini_translate.py --apply logs/sync_runs/<run>/manifest.json
```

要点：

- 同步默认只写 `logs/sync_runs/` 预览，不修改 `.rpy`；写回需显式 `--apply`。
- Batch 须带子命令；产物默认在 `logs/`。
- `check` 为干跑；`apply` 只接受最近一次 `safe` check。
- 子命令细节、订正/关键词、identity v2 见 [Batch 工作流与安全检查](docs/batch_workflows.md)。

显式指定 manifest 时的常用顺序：

```bash
python gemini_translate_batch.py doctor
python gemini_translate_batch.py build
python gemini_translate_batch.py submit logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py status logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py download logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py check logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py apply logs/batch_jobs/<package>/manifest.json
```

`warn` / `block` 时先看失败报告，再用 retry / repair / revision 等流程处理，不要强行 `apply`。

## 文档

完整导航见 [文档地图](docs/README.md)。

| 主题 | 文档 |
|---|---|
| 安装与配置 | [setup.md](docs/setup.md) |
| GUI 工作台 | [gui_workbench.md](docs/gui_workbench.md) |
| Batch 与写回安全 | [batch_workflows.md](docs/batch_workflows.md) |
| RAG / 索引 / Story Memory | [context_systems.md](docs/context_systems.md) |
| 边界、安全与烟测 | [project_notes.md](docs/project_notes.md) |
| 贡献与 CLI/GUI 同步 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 历史设计/审计稿 | [docs/archive/](docs/archive/README.md) |

## 安全提示

执行任何会修改项目文件的操作前，请先备份，并优先在副本上测试。不要把 API key、本地配置、私有游戏脚本、batch 结果或日志提交到公开仓库。详见 [项目说明 · 安全](docs/project_notes.md#安全说明)。
