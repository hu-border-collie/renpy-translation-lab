# GUI 工作台

文档地图：[docs/README.md](README.md)

本文档说明可选 PySide6 图形工作台能做什么、怎么装、以及和 CLI 的边界。GUI 主路径（#42）与同步翻译（#85）、关键词提取（#83）、订正（#84）模式均已实现。

## 定位

图形工作台是现有 CLI 和 JSON 配置之上的**可选外壳**：

- 普通用户走「选项目 → 配置 → 检查 → 翻译 → 写回」；高级信息集中在诊断页。
- 底层仍调用现有 CLI 脚本，不重写翻译核心：**批量翻译**走 `gemini_translate_batch.py`；**同步翻译**走 `gemini_translate.py`。
- 配置仍用 `api_keys.json`、`translator_config.json`；批量写回以 CLI 的 `check -> apply` 安全合约为准，同步模式按脚本规则直接写回。
- GUI 依赖在 `requirements-gui.txt`，不进入主 `requirements.txt`。**不装图形界面时，命令行工具可照常使用。**

界面用语尽量使用中文说明；诊断页「命令参考」仍保留可复制 CLI 子命令，供高级用户对照。

## 安装与启动

先安装核心依赖，再按需安装 GUI 依赖：

```powershell
pip install -r requirements.txt
pip install -r requirements-gui.txt
python -m gui_qt
```

如果未安装 PySide6，`python -m gui_qt` 会打印安装提示并退出；这不会影响 CLI。

### 字体（Git LFS）

GUI 字体位于 `gui_qt/resources/fonts/`，通过 **Git LFS** 存储（约 32 MB）：

- 界面正文：`HarmonyOS Sans SC`（见 `HarmonyOS_Sans_LICENSE.txt`）
- 等宽区域（项目路径、诊断日志、CLI 命令、任务记录、API Key 列表）：`LXGW WenKai Mono GB`（见 `LXGW_WenKai_OFL.txt`）

克隆仓库后请先安装并拉取 LFS 对象：

```powershell
git lfs install
git clone <repo-url>
cd renpy-translation-lab
git lfs pull
```

若已克隆过普通 git 仓库，补拉字体：

```powershell
git lfs install
git lfs pull
```

启动 GUI 时自动加载字体；若 LFS 对象缺失或加载失败，会回退到系统 `Segoe UI` 与 `Consolas`。

## 主流程

GUI 的普通主流程是：

```text
选择项目 -> 配置 API / 模型 -> 检查项目 -> 开始翻译 -> 检查结果 -> 写回翻译
```

对应的底层 CLI 仍是：

```text
doctor -> build -> submit -> status -> download -> check -> apply
```

主界面分为三个顶层 Tab：**工作台**、**配置**、**诊断日志**。

界面上的**灰色说明、状态段落和检查摘要**字号略大于普通正文，便于阅读；按钮、页眉和底部原始日志仍保持紧凑字号。

### 工作台

工作台页负责普通用户主流程：

- 顶部：**当前工作目录**选择与路径显示（也可先选项目根目录，存在 `work/` 时会自动切换）。
- **任务类型 + 子任务**（两列下拉，联动）：
  - **翻译**：批量翻译、同步翻译
  - **分析与准备**：批量关键词、同步关键词、预建记忆库、预建原文索引
  - **维护**：订正、同步订正
  子任务下方有固定高度的说明行；切换任务时布局不会跳动。说明会更新主按钮文案、进度 Tab 与写回 Tab 标签。
- 按钮行：环境检查、准备工作目录、开始翻译、继续任务、停止。
- **准备工作目录**：当 `work/` 不存在或为空且存在 `original/game` 时，把 `original/game` 复制到 `work/game/`；不生成 TL。成功后 GUI 会尝试把 `game_root` 自动改到 `work/`。
- 内层 Tab（标签随子任务变化，批量翻译默认为「翻译进度」）：
  - **环境检查**：`doctor` 的普通语言摘要，并显示记忆库 / 原文索引的启用状态、记录 / 片段数和存储路径。
  - **翻译进度**（或提取进度 / 订正进度等）：友好进度与任务事实行。
  - **写回**（或写回说明 / 订正写回 / 结果说明）：检查或预览摘要，以及写回相关按钮。

普通用户不需要在这一页理解任务记录（manifest）的内部结构；批量写回风险仍以 CLI 的 `check -> apply` 合约为准。

### 配置

配置页采用可滚动布局，自上而下为：

- **API Key**：读取 / 保存 `api_keys.json`；环境变量 Key 只读提示。
- **批量上下文**：记忆库、原文索引、build 时自动补建等开关。启用后需先保存配置，再到工作台「分析与准备」运行预建子任务。
- **模型**：同步 / 批量翻译模型、embedding model、批量 thinking level。
- **外观**：浅色 / 深色 / 跟随系统。
- **保存参数配置**：写回 `translator_config.json` 并保留未知字段。

预建库不会修改游戏源文件；若记忆库未启用就点预建记忆库，界面会提示先打开开关并保存。

### 诊断日志

诊断日志页面向开发者和高级用户，采用**可拖拽分割条**的上下布局：

- **默认**：上方任务上下文区域较高，下方原始 CLI 输出较窄，便于先看清任务记录、路径和命令。
- **任务运行时**：会自动切到此页，并临时放大下方日志区域，方便跟踪输出。

**上方内层 Tab**（`任务上下文` / `命令参考` / `任务记录`）：

- **任务上下文**：任务记录路径、翻译包、云端任务状态、最近检查结果、是否已写回；报告路径逐行展示并支持复制。
- **命令参考**：按当前任务记录生成可复制的手动命令（`doctor`、`submit`、`status`、`download`、`check`、`apply` 等）。当最近一次检查为「需处理」时，也会补出补译相关命令（底层为 `build-retry`、`merge-retry` 等）。
- **任务记录**：只读 JSON 预览（省略 `chunks` / `files` 大字段）。

**下方（原始输出）**：始终可见，显示 CLI 的 stdout/stderr。

工具栏提供「刷新上下文」与「清空日志」。切换到诊断 Tab 时会重新读取最近任务记录；流程进行中会优先展示当前活动的记录。

## 配置兼容性

GUI 不引入新的主配置系统。

- API Key 仍保存到 `api_keys.json` 的 `api_keys` 列表。
- 模型、embedding model、批量 thinking level、GUI 主题等写入 `translator_config.json`。
- 保存配置时应保留未知字段，避免破坏高级配置。
- 如果 API Key 来自 `GEMINI_API_KEY` 等环境变量，GUI 只提示只读状态，不强行写回文件。

## 翻译流程

点击「开始翻译」后，GUI 会编排基础批量流程：

```text
build -> submit -> status -> download -> check
```

如果批量任务仍在处理中，GUI 会停在等待状态，用户稍后可以点击「继续最新任务」。恢复逻辑会读取最近任务记录，并校验它是当前项目的基础翻译任务；如果记录属于其他项目或其他模式，会拒绝继续。

如果 build 已生成 package 但还没有 job，恢复会从 submit 继续，而不是错误地直接跑 status。

### 同步翻译

在工作台选择「翻译 · 同步翻译」后点击「开始翻译」，GUI 会调用 `gemini_translate.py`（无额外参数），配置仍来自 `translator_config.json` 的 `sync.*` 段。同步模式可能直接改项目文件；写回 Tab 仅作说明，不提供批量的 `check/apply` 闸门。运行前需已配置 API Key。

### 订正（批量与同步）

在工作台选择「维护 · 订正」后点击「生成订正预览」，GUI 会编排批量订正流程：

```text
build-revisions -> submit -> status -> download -> preview-revisions
```

对应 CLI 为：

```text
python gemini_translate_batch.py build-revisions
python gemini_translate_batch.py submit <manifest>
python gemini_translate_batch.py status <manifest>
python gemini_translate_batch.py download <manifest>
python gemini_translate_batch.py preview-revisions <manifest>
```

选择「维护 · 同步订正」时，GUI 只调用 `sync-revisions`（**不带 `--apply`**），生成预览报告后停止；适合小范围试跑。

确认预览后，批量与同步订正均通过「**写回订正**」按钮调用 `apply-revisions <manifest>`，对**已预览的任务记录**写回，而不是重新跑 `sync-revisions --apply`。

订正写回与普通翻译写回**分离**：

- 普通 `check -> apply` 的「可写回 / 需处理 / 禁止写回」闸门**不覆盖**订正任务记录。
- 订正写回走独立流程：`preview-revisions -> apply-revisions`，写回前会重新校验当前文件中的旧译文快照。
- GUI「订正写回」页在预览显示有可写回项且记录尚未写回过时，才启用「写回订正」；已写回过的记录不会再次启用。
- **不要在唯一原项目上直接整批写回**；请先在副本或备份上验证预览报告（`revision_preview.jsonl` / `revision_preview.md`）。

## 关键词提取

关键词模式只生成术语与剧情候选报告，**不会修改游戏脚本**。写回 Tab 仅展示结果说明，不提供翻译写回按钮。

### 批量关键词

在工作台选择「分析与准备 · 批量关键词」后点击「提取关键词」，GUI 会扫描翻译文本并编排批量提取；底层 CLI 为：

```text
build-keywords -> submit -> status -> download -> export-keywords
```

如果批量任务仍在处理中，GUI 会停在等待状态，用户稍后可以点击「继续提取」。恢复逻辑会读取最近任务记录，并校验它是当前项目的 `keyword_extraction` 任务。

如果 build 已生成 package 但还没有 job，恢复会从 submit 继续，而不是错误地直接跑 status。

`export-keywords` 完成后，界面会摘要候选数量与 JSONL / Markdown 报告路径，并把四个关键词报告复制到当前 `work` 目录上级的 `extracted_keywords/`；完整原始路径仍可在诊断页复制。诊断页对关键词任务记录会显示 `export-keywords` 等命令，而不是翻译的 `check/apply`。

### 同步关键词

在工作台选择「分析与准备 · 同步关键词」后点击「提取关键词」，GUI 会调用 `gemini_translate_batch.py sync-keywords`（无额外参数）。适合小范围即时生成报告；完成后同样会把关键词报告复制到 `extracted_keywords/`。不支持从任务记录恢复。运行前需已配置 API Key。

## 批量上下文预建

若项目已有一部分译文，或希望在 build 时检索相关剧情原文，可在配置页启用批量上下文并预建本地库。

配置页的「上下文库保存到游戏目录」会把默认 RAG / 原文索引 / 剧情图谱路径切到当前 `work` 同级的 `translation_context/`；关闭时仍使用工具项目内的 `logs/`。Batch manifest、检查失败报告、补译包等运行记录仍保存在工具日志目录。

预建流程：

```text
保存批量上下文开关 -> 工作台选「分析与准备」-> 预建记忆库 / 原文索引 -> 切回「翻译 · 批量翻译」开始翻译
```

对应 CLI 为：

```text
bootstrap-rag --skip-prepare
bootstrap-source-index
```

预建结果以普通语言摘要显示；失败细节可在诊断日志下半部分查看。

若开启了 build 时自动补建，后续 `build` 仍可能自动补建；图形预建入口适合在首次翻译前手动确认 store 状态。

## 检查与写回安全边界

GUI 界面用中文显示检查结果；与 CLI 的对应关系为：

| 界面显示 | CLI `check` 输出 |
|----------|------------------|
| 可写回 | `safe` |
| 需处理 | `warn` |
| 禁止写回 | `block` |

「写回翻译」按钮只在最近一次检查为**可写回**时启用。

- **可写回**：允许进入写回确认，并调用 `apply`。
- **需处理**：禁用写回。可先查看问题清单，必要时生成「补译包」并预览；处理完重新检查后，显示「可写回」才能写入项目。写回页提供「补救命令」入口，诊断页命令参考会给出补译相关可复制命令。
- **禁止写回**：禁用写回，要求修复源文件漂移或重新生成任务。
- 非零退出、未知状态、已写回过的任务记录：都不会启用写回。

GUI 不提供普通用户入口来运行 `apply --force`。`apply --force` 只用于绕过「已经写回过」的重复写回保护，不应被设计成确认「需处理」的入口。

写回前确认框会提醒用户先在副本或备份上验证。即使界面显示「可写回」，也不要在唯一原项目上直接整批写回。

## 当前限制

当前 GUI 仍不是完整发行产品：

- 还没有打包安装器。
- 还没有完整多项目管理。
- 还没有完整可视化 diff 编辑器。
- 补译包目前提供写回页按钮、预览对话框和诊断页命令参考；repair 仍作为独立高级 CLI 流程，不包装成同一个「需处理」解锁路径。
- 同步关键词 / 同步订正不支持从任务记录恢复（与批量模式不同）。

## 已覆盖范围

**GUI 工作台（#42）**：项目选择与配置、环境检查、一键批量翻译与长任务恢复、写回安全保护、主题、批量上下文预建、高级诊断页（任务上下文 / 命令参考 / 任务记录 + 原始日志）。

**后续模式（已合并）**：

- **#85 同步翻译**：`gemini_translate.py` 外壳、进度摘要、诊断页同步命令说明。
- **#83 关键词提取**：批量 / 同步关键词、结果说明 Tab、关键词任务记录的诊断命令。
- **#84 订正**：批量 / 同步订正预览、「写回订正」按钮、独立订正写回摘要。

仓库自动化测试也会验证「未安装图形依赖时 CLI 仍可运行」。

**真实项目烟测记录**：2026-06-19 使用一部约 3,300 待译行的真实 Ren'Py 项目跑通 GUI **批量翻译**主路径：预处理、原文索引预建、`build -> submit -> status`、排队后继续、`download -> check`、可写回状态下写回和游戏启动验证。该批次影响 6 个文件、约 3,300 处译文行；首轮检查为「需处理」时 GUI 正确禁用写回，问题经 CLI / 人工修正并重新检查为「可写回」后再写回。

**#85 / #83 / #84 验证口径**：同步翻译、关键词提取、订正模式已在 GUI 实现并有单元测试覆盖，但**尚未**各自完成与上述同等级别的真实项目烟测；使用前建议在副本项目上小范围试跑。

**有意延后**：repair 图形化编排、安装包、多项目管理、更完整的多游戏端到端手测矩阵等。