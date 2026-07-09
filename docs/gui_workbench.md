# GUI 工作台

文档地图：[docs/README.md](README.md)

本文档说明可选 PySide6 图形工作台能做什么、怎么装、以及和 CLI 的边界。GUI 主路径（#42）与同步翻译（#85）、关键词提取（#83）、订正（#84）、翻译 A/B 对比（#139）均已实现。

> **信息架构（Epic #157）**：**P0a–P1c（#158–#162）已合并**；本文描述的是当前界面。分阶段计划与剩余工作（P2 能力归位、P3 收尾）见 [GUI 信息架构重组计划](gui_ia_redesign.md)。

## 定位

图形工作台是现有 CLI 和 JSON 配置之上的**可选外壳**：

- 普通用户走「选项目 → 设置 → 检查 → 翻译 → 写回」；运行细节默认在工作台底部日志抽屉，完整任务上下文仍在诊断页。
- 底层仍调用现有 CLI 脚本，不重写翻译核心：**批量翻译**走 `gemini_translate_batch.py`；**同步翻译**走 `gemini_translate.py`。
- **开发与功能约定**：新能力须 CLI / GUI 同步交付，见根目录 [CONTRIBUTING.md](../CONTRIBUTING.md)。
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
选择项目 -> 设置 API / 模型 -> 检查项目 -> 开始翻译 -> 检查结果 -> 写回翻译
```

对应的底层 CLI 仍是：

```text
doctor -> build -> submit -> status -> download -> check -> apply
```

### 壳布局（当前）

```text
┌ 全局项目栏：当前 game_root · [切换项目] · [指定本地目录…] ──────────────┐
│ 顶层 Tab：工作台 | 设置 | 诊断与工具（诊断日志）                         │
│ ┌ 左导航 ──┐ ┌ 右侧任务内容 ─────────────────────────────────────────┐ │
│ │ 批量翻译 │ │ 模式说明 / 子模式（部分页）· 主按钮 · 阶段或进度/结果   │ │
│ │ 同步翻译 │ │ 底部：可折叠运行日志抽屉（与诊断原始输出同源）         │ │
│ │ 关键词   │ └──────────────────────────────────────────────────────┘ │
│ │ 订正     │                                                           │
│ │ 上下文库 │                                                           │
│ └──────────┘                                                           │
└────────────────────────────────────────────────────────────────────────┘
```

| 区域 | 作用 |
|------|------|
| **全局项目栏**（P0b / #159） | 任意顶层 Tab 可见当前 `game_root`；**切换项目**打开设置 → 工作区总表；**指定本地目录…** 走文件夹选择（切换前处理未保存设置）。任务运行中二者禁用。 |
| **工作台左导航**（P1a / #160） | 五个固定任务入口（见下）。切换时**保留各 WorkMode 会话**（进度摘要、写回 manifest、关键词 candidates、订正可写回态、批量阶段索引等）。运行中导航禁用。 |
| **工作台日志抽屉**（P0a / #158） | 从工作台启动的任务默认**留在工作台**并展开底部日志，**不强制跳到诊断**。与诊断页原始输出共享同一缓冲。 |
| **诊断页** | 任务上下文、命令参考、任务记录 + 原始日志；试跑 / 拆分 / A/B 等高级工具仍从这里启动，并在本页放大日志。 |

界面上的**灰色说明、状态段落和检查摘要**字号略大于普通正文，便于阅读；按钮、页眉和底部原始日志仍保持紧凑字号。

### 工作台 · 左导航任务页

| 导航项 | 主行动 | 结果区要点 |
|--------|--------|------------|
| **批量翻译** | 环境检查 · 准备工作目录 · 开始翻译 / 继续 · 停止 | 三阶段 **准备 → 执行 → 结果**；结果主按钮「写回翻译」；**问题处理**折叠（补译、修补、重新检查等） |
| **同步翻译** | 开始同步翻译 · 停止 | 风险提示（可能直接改项目文件）；无批量 `check/apply` 闸门 |
| **关键词 / 术语** | 模式：批量 \| 同步 · 提取关键词 · 继续（批量）· 停止 | 报告摘要；主入口 **合并到 glossary** |
| **订正** | 模式：批量 \| 同步 · 生成订正预览 · 继续（批量）· 停止 | 预览摘要；主入口 **写回订正**（与写回翻译分离） |
| **上下文库** | 记忆库 / 原文索引状态卡 · 预建 · 打开设置·上下文 | 开关须在设置中保存后才能预建 |

多模式页使用短标签「批量 / 同步」（关键词、订正）。**上下文库**用双状态卡代替子模式下拉（P1c / #162）。

往返规则：离开再回来会恢复该页最近结果与可执行按钮态（至少覆盖写回/manifest、关键词 merge 就绪、订正可写回、非 running 时的进度摘要）。仅「切换 game_root」或显式新任务会清空相关 session。

### 工作台 · 批量翻译三阶段（P1b / #161）

批量模式下显示阶段条，并隐藏下方平级三 Tab 栏（避免双导航）：

- **准备**：环境检查摘要与记忆库/原文索引状态；用于解锁翻译流程。
- **执行**：时间线、进度与任务事实；开始 / 继续 / 停止。
- **结果**：检查摘要；「写回翻译」；**问题处理**折叠（检查为「需处理」等时自动展开或角标提示）。

其他导航页仍使用「环境检查 / 进度 / 结果说明（或写回订正）」类标签，语义随任务变化。

### 设置

设置页采用左侧分区导航，右侧显示当前分区内容；底部固定提供 **重新加载**、**恢复推荐值**、**保存设置**。

- **工作区**：浏览 `games_registry.json` 中的全部项目，扫描/刷新总表、同步 `GAMES.md`、编辑项目详情，并**切换当前 `game_root`**（深度入口；全局栏「切换项目」会跳到此区）。切换成功后会自动打开「项目」分区。此分区的操作即时写入 registry，不使用底部「保存设置」。
- **密钥**：读取 / 保存 `api_keys.json`；环境变量 Key 只读提示。
- **项目**：术语表、翻译目录、include filters，以及准备流程的 source game、Ren'Py SDK、Python、launcher 和自定义命令。当前 `game_root` 只读展示；需换项目请用全局栏或「工作区」。
- **模型**：同步 / 批量翻译模型、embedding model、批量 thinking level。
- **上下文**：批量 RAG 记忆库、原文索引、build 时自动补建、上下文库保存位置等开关。启用后需先保存设置，再到工作台 **上下文库** 页预建。
- **外观**：浅色 / 深色 / 跟随系统。切换主题会立即预览，保存设置后才写入 `translator_config.json`。
- **高级**：翻译吞吐、retry、Batch safety settings、术语/风格、关键词提取、订正、RAG / 原文索引 / 剧情记忆的检索参数和 store 路径。数值字段会按 CLI 语义校验，路径类字段可留空表示使用默认路径；列表字段支持逐行填写，命令和 safety settings 支持 JSON。

> **P2 计划中**：同步 RAG / 剧情记忆等开关将进一步归位到「上下文」并保证单源写入；见 [计划 · P2b](gui_ia_redesign.md)。

预建库不会修改游戏源文件；若记忆库未启用就点预建记忆库，界面会提示先打开开关并保存。

### 诊断日志（诊断与工具）

诊断日志页面向开发者和高级用户，采用**可拖拽分割条**的上下布局：

- **默认**：上方任务上下文区域较高，下方原始 CLI 输出较窄。
- **工作台任务运行时**（P0a / #158）：默认**留在工作台**，展开底部**运行日志**抽屉；与诊断页原始输出共享同一日志缓冲。不会再为翻译/写回等主路径强制跳到本页。
- **从本页启动的工具**（试跑样本、拆分翻译包、翻译 A/B 对比）：仍在诊断页放大下方日志区域。
- 工作台抽屉提供「在诊断中打开」，可随时查看任务上下文与命令参考。

**上方内层 Tab**（`任务上下文` / `命令参考` / `任务记录`）：

- **任务上下文**：任务记录路径、翻译包、云端任务状态、最近检查结果、是否已写回；报告路径逐行展示并支持复制。
- **命令参考**：按当前任务记录生成可复制的手动命令（`doctor`、`submit`、`status`、`download`、`check`、`apply` 等）。批量翻译任务记录还会显示 `compare-variants` 试跑命令模板。当最近一次检查为「需处理」时，也会补出补译相关命令（底层为 `build-retry`、`merge-retry` 等）。
- **任务记录**：只读 JSON 预览（省略 `chunks` / `files` 大字段）。

**下方（原始输出）**：始终可见，显示 CLI 的 stdout/stderr。

工具栏提供「刷新上下文」「试跑样本请求」「翻译 A/B 对比」「拆分翻译包」与「清空日志」。切换到诊断 Tab 时会重新读取最近任务记录；流程进行中会优先展示当前活动的记录。

> **P2 计划中**：拆分与试跑将迁入批量·执行·高级入口；诊断工具条会进一步瘦身。当前二者仍在诊断页可用。

**试跑样本请求（probe）**：批量翻译模式下，若当前有可用的 translation manifest（version 1），可点击「试跑样本请求」对 package 内少量请求做同步 `generate_content` 冒烟测试（默认 `--limit 3`）。高级选项可调整 `--limit`、`--offset` 与 `--api-key-index`。该操作不会提交批量任务，也不会修改项目文件；摘要显示在任务上下文区，原始输出写入下方日志。适合在 `submit` 前排障 API、模型与请求格式。

**拆分翻译包（split）**：批量翻译模式下，若当前 translation manifest（version 1）含有可拆分的块，可点击「拆分翻译包」按上限生成多个子包（默认 `--max-chunks 600`）。高级选项可调整 `--max-items` 与 `--display-name-prefix`。拆分后 RAG 记忆库为静态快照，各子包需分别 submit，不会自动提交；子包路径会列在任务上下文区，命令参考会补充各子包的 `submit` 示例。详见 `docs/context_systems.md`。

## 翻译 A/B 对比

**翻译 A/B 对比（compare-variants）**用于在同一批 manifest chunk 上并排比较多个配置变体的同步译文，**不会写回** `.rpy` 或 `glossary.json`。适合评估 Story Memory、RAG、原文索引等上下文层对译文的影响。CLI 细节见 [Batch 工作流与安全检查](batch_workflows.md#翻译质量-ab-实验)。

### 何时可用

诊断页工具栏按钮在以下条件下启用：

- 当前诊断上下文绑定的是**批量翻译**任务记录（`mode=translation`）；
- 任务记录中有可采样的 chunk（完整 manifest 含 `chunks`，或 lite manifest 含 `summary.chunk_count > 0`）。

关键词、订正等非翻译任务记录会禁用该按钮，悬停提示会说明原因。

### 操作步骤

1. 完成至少一次批量 `build`（或从已有翻译包进入诊断页），确保诊断上下文已加载目标 manifest。
2. 点击「**翻译 A/B 对比**」，在对话框中配置对比项与采样参数。
3. 默认勾选「**仅试跑**」：只重建各变体 prompt 并生成报告，**不调用翻译 API**，适合先确认配置差异。
4. 试跑结果满意后，取消「仅试跑」再正式对比；建议保持较小采样块数（默认 `--limit 3`）。
5. 完成后在任务上下文区查看摘要，并从路径列表打开 `ab_report.md`；需要逐条结果时可打开 `ab_results.jsonl`。

### 对比变体（对话框）

GUI 不要求手写 JSON 变体文件。对话框以 **baseline + 可选覆盖变体** 方式生成临时变体列表：

- **baseline**：始终包含，对应当前 `translator_config.json` 中的批量配置。
- 每个维度一组单选（默认「不参与」），并显示该维度在配置中的**当前开/关状态**：
  - **Story Memory**
  - **RAG**
  - **原文索引**
- 「强制开启 / 强制关闭」会在 baseline 之外追加一个仅覆盖该开关的变体。

至少需为某一维度选择「强制开启」或「强制关闭」，否则会提示对比项不足。高级用户仍可在命令参考区复制 `compare-variants --variants-file <variants.json>` 模板，自行编写更复杂的变体（例如 macro setting、模型覆盖）；GUI 对话框暂不覆盖这些维度。

### 高级选项

| 选项 | 默认 | 说明 |
|------|------|------|
| 采样块数 (`--limit`) | 3 | 从 manifest 中取前 N 个 chunk 做对比 |
| 起始偏移 (`--offset`) | 0 | 跳过前若干 chunk 后再采样 |
| 输出目录 (`--output-dir`) | 空 | 留空则写入 `logs/experiments/<timestamp>_ab/` |
| 仅试跑 | 开启 | 等同 CLI `--dry-run` |
| API Key 索引 | 默认 | 等同 CLI `--api-key-index` |

### 结果解读

完成后任务上下文区会显示运行摘要。常见状态：

| 状态 | 含义 |
|------|------|
| 试跑 / 对比完成 | 报告已生成，可打开并排 Markdown 报告人工比较 |
| 需关注 | 部分变体在 `ab_results.jsonl` 中有 `error`，或变体数少于预期；报告可能不完整 |
| 失败 | 实验中断（如 `ab_settings.json` 含 `experiment_error`）或进程非零退出 |

路径列表通常包含：**A/B 报告**（`ab_report.md`）、**A/B 结果**（`ab_results.jsonl`）、**A/B 设置**（`ab_settings.json`）、**A/B 输出目录**。

### 成本与边界

- 正式对比（非试跑）的 API 用量约为 `采样块数 × 变体数` 次同步请求；先用小 `--limit` 与试跑确认无误。
- 试跑不会触发 RAG / 原文索引 / Story Memory 检索，也不会调用翻译 API。
- 实验产物不会写回游戏文件；调整配置后请在设置页保存，再重新跑对比。
- 与「试跑样本请求（probe）」不同：probe 验证单次请求能否成功；A/B 对比关注**同一批文本在不同配置下的译文差异**。

## 配置兼容性

GUI 不引入新的主配置系统。

- API Key 仍保存到 `api_keys.json` 的 `api_keys` 列表。
- 项目路径、准备流程、过滤器、模型、embedding model、批量 thinking level、GUI 主题、任务专用参数和上下文参数等写入 `translator_config.json`。
- 保存设置时应保留未知字段，避免破坏手写配置。
- 如果 API Key 来自 `GEMINI_API_KEY` 等环境变量，GUI 只提示只读状态，不强行写回文件。

## 翻译流程

在左导航选择 **批量翻译**，点击「开始翻译」后，GUI 会编排基础批量流程：

```text
build -> submit -> status -> download -> check
```

如果批量任务仍在处理中，GUI 会停在等待状态，用户稍后可以点击「继续最新任务」。恢复逻辑会读取最近任务记录，并校验它是当前项目的基础翻译任务；如果记录属于其他项目或其他模式，会拒绝继续。

如果 build 已生成 package 但还没有 job，恢复会从 submit 继续，而不是错误地直接跑 status。

### 同步翻译

在左导航选择 **同步翻译**。页内有备份/直接改文件的风险提示；主按钮为「**开始同步翻译**」。GUI 调用 `gemini_translate.py`（无额外参数），配置仍来自 `translator_config.json` 的 `sync.*` 段。同步模式可能直接改项目文件；结果区仅作说明，不提供批量的 `check/apply` 闸门。运行前需已配置 API Key。

### 订正（批量与同步）

在左导航选择 **订正**，用模式「批量 | 同步」切换。点击「生成订正预览」后：

**批量**编排：

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

**同步订正**只调用 `sync-revisions`（**不带 `--apply`**），生成预览报告后停止；适合小范围试跑。

确认预览后，批量与同步订正均通过「**写回订正**」按钮调用 `apply-revisions <manifest>`，对**已预览的任务记录**写回，而不是重新跑 `sync-revisions --apply`。

订正写回与普通翻译写回**分离**：

- 普通 `check -> apply` 的「可写回 / 需处理 / 禁止写回」闸门**不覆盖**订正任务记录。
- 订正写回走独立流程：`preview-revisions -> apply-revisions`，写回前会重新校验当前文件中的旧译文快照。
- GUI 在订正页结果区、预览显示有可写回项且记录尚未写回过时，才启用「写回订正」；已写回过的记录不会再次启用。
- **不要在唯一原项目上直接整批写回**；请先在副本或备份上验证预览报告（`revision_preview.jsonl` / `revision_preview.md`）。

## 关键词提取

在左导航选择 **关键词 / 术语**，用模式「批量 | 同步」切换。关键词模式只生成术语与剧情候选报告，**不会修改游戏脚本**。

### 批量关键词

点击「提取关键词」后，GUI 会扫描翻译文本并编排批量提取；底层 CLI 为：

```text
build-keywords -> submit -> status -> download -> export-keywords
```

如果批量任务仍在处理中，GUI 会停在等待状态，用户稍后可以点击「继续提取」。恢复逻辑会读取最近任务记录，并校验它是当前项目的 `keyword_extraction` 任务。

如果 build 已生成 package 但还没有 job，恢复会从 submit 继续，而不是错误地直接跑 status。

`export-keywords` 完成后，界面会摘要候选数量与 JSONL / Markdown 报告路径，并把四个关键词报告复制到当前 `work` 目录上级的 `extracted_keywords/`；完整原始路径仍可在诊断页复制。诊断页对关键词任务记录会显示 `export-keywords` 等命令，而不是翻译的 `check/apply`。

### 同步关键词

选择模式「同步」后点击「提取关键词」，GUI 会调用 `gemini_translate_batch.py sync-keywords`（无额外参数）。适合小范围即时生成报告；完成后同样会把关键词报告复制到 `extracted_keywords/`。不支持从任务记录恢复。运行前需已配置 API Key。

### 合并候选到 glossary

关键词报告生成后，可在 GUI 内把 `keyword_candidates.jsonl` 中经人工审核的候选写入当前项目的 `glossary.json`，无需复制 CLI 到终端逐条 `y/n` 确认。

**入口：**

- 工作台 **关键词 / 术语** 页结果区：**合并到 glossary**（主入口；candidates 就绪时可用）
- 诊断日志工具栏：**合并到 glossary**（诊断上下文能解析到候选 JSONL 时可用；P2 后可能进一步降级为次要入口）

若当前上下文没有候选文件，也可在对话框流程中手动选择 `.jsonl` 文件。

**交互要点：**

- 表格列出 `source`、`suggested_target`、`category`、`confidence`、计划写入分区（`preserve_terms` / `normalize_map`）与冲突提示。
- 默认**不勾选**疑似 Ren'Py 启动器 / UI 噪音项；与 `macro_setting` 或现有 glossary 冲突的条目会标红提示。
- 支持全选 / 全不选 / 反选；可勾选「覆盖已有 glossary 冲突项」后再写入。
- **预览写入**只生成摘要，不修改 glossary；**写入 glossary** 前会二次确认，并自动创建 `glossary.json.bak-<timestamp>` 备份。

合并目标 glossary 跟随 `translator_config.json` 的 `glossary_file`（或当前 `work/` 目录下的默认 `glossary.json`）。CLI 高级参数（`--min-confidence`、`--accept-confidence` 等）仍可在诊断页「命令参考」复制使用。

## 批量上下文预建

若项目已有一部分译文，或希望在 build 时检索相关剧情原文，可在设置页启用批量上下文并预建本地库。

设置页「上下文」里的「上下文库保存到游戏目录」会把默认 RAG / 原文索引 / 剧情图谱路径切到当前 `work` 同级的 `translation_context/`；关闭时仍使用工具项目内的 `logs/`。高级分区可显式覆盖 store 路径；Batch manifest、检查失败报告、补译包等运行记录仍保存在工具日志目录。

预建流程：

```text
设置 · 上下文 打开开关并保存
  -> 工作台 · 上下文库（查看状态卡）
  -> 预建记忆库 / 预建原文索引
  -> 左导航切回「批量翻译」开始翻译
```

对应 CLI 为：

```text
bootstrap-rag --skip-prepare
bootstrap-source-index
```

预建结果以普通语言摘要显示；失败细节可在工作台日志抽屉或诊断原始输出查看。任务运行中状态卡上的预建按钮会禁用，避免叠跑。

若开启了 build 时自动补建，后续 `build` 仍可能自动补建；图形预建入口适合在首次翻译前手动确认 store 状态。

## 检查与写回安全边界

GUI 界面用中文显示检查结果；与 CLI 的对应关系为：

| 界面显示 | CLI `check` 输出 |
|----------|------------------|
| 可写回 | `safe` |
| 需处理 | `warn` |
| 禁止写回 | `block` |

「写回翻译」按钮只在最近一次检查为**可写回**时启用（批量翻译 · 结果区主按钮）。

- **可写回**：允许进入写回确认，并调用 `apply`。
- **需处理**：禁用写回。展开 **问题处理**（默认折叠；需处理时自动展开或角标）。其中可：查看问题清单、生成/查看补译包、**继续补译**、**同步修补**、**重新检查**、查看写回失败报告、补救命令（高级回退）。
  - **继续补译**：预览确认后编排 `submit → status → download → check → merge-retry → check`；云端排队中可再次点击继续，或在工作台使用「继续翻译」从 `status` 恢复。
  - **同步修补**：对 repair 类问题调用 `gemini_translate_batch.py repair <jsonl>`（默认 `--batch-size 2`）；运行前会二次确认并提醒备份，完成后应 **重新检查**。
  - **补译** 与 **同步修补** 入口分开，不会合并为一键修复。
- **重新检查**：批量翻译模式下，只要已有任务记录（完成过至少一次检查），问题处理区会提供「重新检查」。该按钮调用 `gemini_translate_batch.py check <manifest>`；检查失败或非「可写回」时不会启用写回。
- **禁止写回**：禁用写回，要求修复源文件漂移或重新生成任务。
- 非零退出、未知状态、已写回过的任务记录：都不会启用写回。

GUI 不提供普通用户入口来运行 `apply --force`。`apply --force` 只用于绕过「已经写回过」的重复写回保护，不应被设计成确认「需处理」的入口。

写回前确认框会提醒用户先在副本或备份上验证。即使界面显示「可写回」，也不要在唯一原项目上直接整批写回。

## 当前限制

当前 GUI 仍不是完整发行产品：

- 还没有打包安装器。
- 还没有完整多项目管理（工作区总表可切换项目，但非 per-game 配置隔离）。
- 还没有完整可视化 diff 编辑器。
- 同步修补会直接改翻译文件，不提供内嵌 diff 编辑器；复杂问题仍需 CLI 或人工处理。
- 同步关键词 / 同步订正不支持从任务记录恢复（与批量模式不同）。
- 拆分 / 试跑 / A/B 仍主要在诊断页（P2 将部分迁入批量执行高级区）。

## 已覆盖范围

**GUI 工作台（#42 + Epic #157 P0–P1）**：全局项目栏、左导航五任务页、批量三阶段、页会话往返、工作台日志抽屉、环境检查、一键批量翻译与长任务恢复、写回安全保护与问题处理折叠、主题、批量上下文预建、高级诊断页。

**模式与能力（已合并）**：

- **#85 同步翻译**：独立导航页、风险提示、`gemini_translate.py` 外壳。
- **#83 关键词提取**：批量 / 同步子模式、结果说明、关键词任务记录的诊断命令。
- **#148 关键词合并 GUI**：勾选式合并对话框；关键词页与诊断页「合并到 glossary」。
- **#84 订正**：批量 / 同步子模式、「写回订正」、独立订正写回摘要。
- **#139 翻译 A/B 对比**：诊断页按钮、变体选项对话框、试跑 / 正式对比摘要与报告路径。

仓库 CI 将测试拆成三类：`unittest`（ubuntu/windows，安装 `requirements-gui.txt` 跑全量）、`cli-without-gui`（仅 `requirements.txt`，用 `tests/run_cli_tests.py` 排除 `test_gui_*`）、`gui`（ubuntu，`tests/run_gui_tests.py` + `QT_QPA_PLATFORM=offscreen`）。新增 GUI 测试请用 `tests/gui_test_support.py` 中的 `skip_unless_gui` 或现有 `skipIf` 惯例，避免在 CLI job 里顶层 `import PySide6`。

**真实项目烟测记录**：2026-06-19 使用一部约 3,300 待译行的真实 Ren'Py 项目跑通 GUI **批量翻译**主路径：预处理、原文索引预建、`build -> submit -> status`、排队后继续、`download -> check`、可写回状态下写回和游戏启动验证。该批次影响 6 个文件、约 3,300 处译文行；首轮检查为「需处理」时 GUI 正确禁用写回，问题经 CLI / 人工修正并重新检查为「可写回」后再写回。

**#85 / #83 / #84 / #139 验证口径**：同步翻译、关键词提取、订正、翻译 A/B 对比已在 GUI 实现并有单元测试覆盖，但**尚未**各自完成与上述同等级别的真实项目烟测；使用前建议在副本项目上小范围试跑。

**有意延后**：安装包、更完整的多游戏端到端手测矩阵、P2/P3 IA 收尾等。
