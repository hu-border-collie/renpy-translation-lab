# GUI 工作台

文档地图：[docs/README.md](README.md)

本文档说明可选 PySide6 图形工作台能做什么、怎么装、以及和 CLI 的边界。第一版 GUI 主路径已覆盖 issue #42 的核心验收；后续高级模式拆分到 #83 / #84 / #85。

## 定位

图形工作台是现有 CLI 和 JSON 配置之上的**可选外壳**：

- 普通用户走「选项目 → 配置 → 检查 → 翻译 → 写回」；高级信息集中在诊断页。
- 底层仍调用 `gemini_translate_batch.py`，不重写翻译核心。
- 配置仍用 `api_keys.json`、`translator_config.json`；写回仍以 CLI 的 `check -> apply` 安全合约为准。
- GUI 依赖在 `requirements-gui.txt`，不进入主 `requirements.txt`。**不装图形界面时，命令行工具可照常使用。**

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
- 等宽区域（项目路径、诊断日志、CLI 命令、Manifest、API Key 列表）：`LXGW WenKai Mono GB`（见 `LXGW_WenKai_OFL.txt`）

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

- 顶部：当前游戏 work 目录选择与路径显示。
- 按钮行：环境检查、开始翻译、继续任务、停止。
- 内层 Tab（默认打开「翻译进度」）：
  - **环境检查**：`doctor` 的普通语言摘要。
  - **翻译进度**：一键 Batch 流程的友好进度与任务事实行。
  - **写回**：`check` 安全状态摘要与「写回翻译」按钮（仅 `safe` 时启用）。

普通用户不需要在这一页理解 manifest 内部结构；写回风险仍以 CLI 的 `check -> apply` 合约为准。

### 配置

配置页采用可滚动布局，自上而下为：

- **API Key**：读取 / 保存 `api_keys.json`；环境变量 Key 只读提示。
- **Batch 上下文**：RAG、原文索引、build 时自动补建等开关。启用后需先保存配置，再运行预建按钮。
- **预建库**：
  - **预建 RAG 库**：扫描已有 TL 译文写入本地 history store。
  - **预建原文索引**：只索引 TL 模板原文，不修改 `.rpy`。
- **模型**：Sync / Batch 翻译模型、embedding model、Batch thinking level。
- **外观**：浅色 / 深色 / 跟随系统。
- **保存参数配置**：写回 `translator_config.json` 并保留未知字段。

预建库不会修改游戏源文件；若 RAG 未启用就点预建 RAG，界面会提示先打开开关并保存。

### 诊断日志

诊断日志页面向开发者和高级用户，采用**可拖拽分割条**的上下布局：

- **默认**：上方任务上下文区域较高，下方原始 CLI 输出较窄，便于先看清 manifest、路径和命令。
- **任务运行时**：会自动切到此页，并临时放大下方日志区域，方便跟踪输出。

**上方内层 Tab**（`任务上下文` / `CLI 命令` / `Manifest`）：

- **任务上下文**：manifest、package、job 状态、最近 check、是否已 apply；报告路径逐行展示并支持复制。
- **CLI 命令**：按当前 manifest 生成可复制的手动命令（`doctor`、`submit`、`status`、`download`、`check`、`apply` 等）。
- **Manifest**：只读 JSON 预览（省略 `chunks` / `files` 大字段）。

**下方（原始输出）**：始终可见，显示 CLI 的 stdout/stderr。

工具栏提供「刷新上下文」与「清空日志」。切换到诊断 Tab 时会重新读取 latest manifest；翻译流程进行中会优先展示当前活动的 manifest。

## 配置兼容性

GUI 不引入新的主配置系统。

- API Key 仍保存到 `api_keys.json` 的 `api_keys` 列表。
- 模型、embedding model、Batch thinking level、GUI 主题等写入 `translator_config.json`。
- 保存配置时应保留未知字段，避免破坏高级配置。
- 如果 API Key 来自 `GEMINI_API_KEY` 等环境变量，GUI 只提示只读状态，不强行写回文件。

## 翻译流程

点击「开始翻译」后，GUI 会编排基础 Batch 流程：

```text
build -> submit -> status -> download -> check
```

如果 Batch 任务仍在处理中，GUI 会停在等待状态，用户稍后可以点击「继续最新任务」。恢复逻辑会读取 latest manifest，并校验它是当前项目的基础翻译任务；如果 latest manifest 属于其他项目或其他模式，会拒绝继续。

如果 build 已生成 package 但还没有 job，恢复会从 submit 继续，而不是错误地直接跑 status。

## Batch 上下文预建

若项目已有一部分译文，或希望在 build 时检索相关剧情原文，可在配置页启用 Batch 上下文并预建本地库：

```text
保存 Batch 上下文开关 -> 预建 RAG 库 和/或 预建原文索引 -> 开始翻译
```

对应 CLI 为：

```text
bootstrap-rag --skip-prepare
bootstrap-source-index
```

预建结果以普通语言摘要显示；失败细节可在诊断日志下半部分查看。

若开启了 build 时自动补建，后续 `build` 仍可能自动补建；图形预建入口适合在首次翻译前手动确认 store 状态。

## Check 与 apply 安全边界

GUI 写回按钮只在最近一次 check 为 `safe` 时启用。

- `safe`：允许进入写回确认，并调用 `apply`。
- `warn`：禁用写回，引导用户查看问题、retry、repair 或重新检查。
- `block`：禁用写回，要求修复源文件漂移或重新生成任务。
- 非零退出、未知 safety、已 apply 的 manifest：都不会启用写回。

GUI 不提供普通用户入口来运行 `apply --force`。`apply --force` 只用于绕过「已经 apply 过」的重复写回保护，不应被设计成确认 `warn` 的入口。

写回前确认框会提醒用户先在副本或备份上验证。即使 GUI 显示 `safe`，也不要在唯一原项目上直接整批写回。

## 当前限制

当前 GUI 仍不是完整发行产品：

- 还没有打包安装器。
- 还没有完整多项目管理。
- 还没有完整可视化 diff 编辑器。
- 还没有 repair / retry / revision 的图形化编排入口（仍通过 CLI 或诊断页复制的命令操作）。

## 已覆盖范围与后续项

**第一版已覆盖（#42）**：项目选择与配置、环境检查、一键 Batch 翻译与长任务恢复、写回安全保护、主题、Batch 上下文预建、高级诊断页（任务上下文 / 命令 / Manifest + 原始日志）。仓库自动化测试也会验证「未安装图形依赖时 CLI 仍可运行」。

**有意延后或拆 follow-up**：同步补译模式（#85）、关键词提取模式（#83）、订正模式（#84）、在真实游戏工程上的端到端手测记录、repair/retry 图形入口、安装包与多项目管理等。