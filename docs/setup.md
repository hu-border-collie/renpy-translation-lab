# 安装与本地配置

文档地图：[docs/README.md](README.md)

本文档承接根目录 `README.md` 中会拖慢阅读节奏的本地配置、项目目录和 Ren'Py SDK 细节。

## 本地配置文件

工具级配置放在**工具仓库根目录**（与脚本同级；不会回退到上一级 `data/`）：

- `api_keys.example.json` -> `api_keys.json`
- `translator_config.example.json` -> `translator_config.json`

`game_root` / `workspace_root` 须显式配置；未配置时不会把工具父目录当成默认项目或工作区。Batch 日志在工具根下的 `logs/`。

项目资产放在当前 `game_root`（通常就是游戏的 `work` 目录）：

- `glossary.example.json` -> `<game_root>/glossary.json`
- `macro_setting.example.md` -> `<game_root>/macro_setting.md`

这些文件通常包含 API key、本地游戏路径、项目私有术语或剧情设定，不应提交到公开仓库。

说明：

- `api_keys.json` 保存 Gemini API Key；旧的 `batch_size/max_chars` 等字段仍兼容，但不再推荐写在这里。
- `translator_config.json` 保存**工具级**设置：可选 `workspace_root`（多项目工作区，须显式指定）、当前 `game_root`、模型、include 过滤、同步 / Batch 分块参数，以及 RAG/原文索引的**全局默认值**（当前项目尚未写过项目文件时使用）。
- **按项目生效**的批量上下文开关见下一节 `project_context_settings.json`。
- `glossary.json` 通常包含项目私有术语；不存在时脚本会退回内置默认术语规则。`translator_config.json` 的 `glossary_file` 应指向当前项目的文件。
- `macro_setting.md` 往往包含剧情、角色口吻、世界观约束，可供 Batch 的 `batch.macro_setting_file` 使用。
- GUI 切换项目时会把上述两个路径同步到当前 `work` 目录；纯 CLI 使用者若保留相对路径，请确保仓库根目录没有同名旧资产，或直接填写绝对路径，避免加载到其他项目的文件。

## 项目级上下文开关

批量 **启用 RAG**、**启用原文索引**、**build 时自动暖库** 跟随**当前游戏 work 目录**，不写进全局配置以免切换项目互相覆盖。

路径：

```text
<game_root>/project_context_settings.json
```

示例：

```json
{
  "schema_version": 1,
  "batch_rag_enabled": true,
  "batch_source_index_enabled": true,
  "batch_rag_bootstrap_on_build": true
}
```

行为：

- GUI「设置 · 上下文」保存时写入该文件；加载时按当前 `game_root` 读取。
- CLI 的 `load_batch_settings` 会在解析 `translator_config.json` 后套用该文件。
- 文件不存在时：回退到 `translator_config.json` 里 `batch.rag.enabled` / `batch.source_index.enabled` / `batch.rag.bootstrap_on_build`。
- 实现见 `project_context_settings.py`。
- 如果你不想使用 `api_keys.json`，也可以改用环境变量 `GEMINI_API_KEY`、`GEMINI_API_KEY_2`、`GEMINI_API_KEY_3`。
- 如果你不想使用 `translator_config.json`，也可以至少通过 `GAME_ROOT` 或 `SA_GAME_ROOT` 指向目标 `work` 目录。
- 也可以把 `.env.example` 复制为 `.env`，填写后供本地 shell/启动器加载；`.env` 只用于本机私有配置，不应提交。
- `.env` 不是必需文件；当前脚本不会自动加载 `.env`，默认复制并填写 `api_keys.json` / `translator_config.json` 即可使用。

## 游戏目录

脚本默认假设 `game_root` 指向某个游戏的 `work` 目录，典型结构如下：

```text
Game_Example/
├─ original/
├─ work/
│  └─ game/
│     └─ tl/
│        └─ <language>/   # 默认 schinese
└─ build/
```

默认目标语言是 `schinese`，可以通过 `translator_config.json` 里的 `tl_subdir` 和 `prepare.language` 调整。例如日语本地化可设为 `game/tl/japanese` 与 `prepare.language: "japanese"`；两者末段应保持一致。

`doctor`、`build` 与 `generate-template` 会在输出中显示当前 TL 路径与目标语言，便于确认配置是否生效。

推荐先使用 Ren'Py SDK 生成标准 `tl/<language>` 模板；如果启用了 `prepare`，脚本会尝试从 `original/game` 提取脚本并自动调用 Ren'Py 生成或刷新对应语言模板。自动模板生成需要 Ren'Py SDK 或目标游戏自带的 Ren'Py launcher；如果已经有可用 TL 文件，缺少 SDK 时仍可直接处理现有 TL。

不要用 Ren'Py 的 `--empty` 生成空模板。本工具的初译流程需要目标行保留原文，之后再把目标行或 `new` 行替换成中文。

**信任边界：** `translator_config.json` 是**可执行的本地配置**，不只是数据文件。`prepare.unpack_command` / `prepare.template_command` 会在准备阶段在本机运行。推荐使用 **argv 列表**；shell 字符串命令默认拒绝，只有显式设置 `prepare.allow_shell_commands: true` 后才允许（doctor / GUI 会标高风险）。不要加载来源不明的项目配置。

可以通过 `translator_config.json` 或环境变量 `RENPY_SDK_DIR` 指定 SDK 位置。**不会在加载配置或 prepare 时自动扫描磁盘**；只有用户主动操作才会搜索其它目录。

GUI **设置 → 项目 → Ren'Py SDK 目录** 提供：

- **查找 SDK**（主动）：在当前 `game_root`、已选工作区与工具附近扫描 `renpy-*-sdk` / `renpy.py`，找到后写入配置字段（多结果时选择）
- **浏览…**（主动）：手动选择目录

字段留空则 `prepare.renpy_sdk_dir` 为空，需要 SDK 的模板生成会提示配置。示例：

```json
{
  "game_root": "C:/games/Game_Example/work",
  "tl_subdir": "game/tl/schinese",
  "prepare": {
    "enabled": true,
    "generate_template": true,
    "refresh_existing_template": true,
    "language": "schinese",
    "renpy_sdk_dir": "C:/RenPy/renpy-8.5.2-sdk",
    "template_command": [
      "{python_exe}",
      "{launcher_py}",
      "{base_dir}",
      "translate",
      "{language}"
    ],
    "allow_shell_commands": false
  }
}
```

如果还没有 `work/` 工作目录，但项目根下已有 `original/game`，可以先初始化 work 副本：

```bash
python gemini_translate_batch.py bootstrap-work
```

该命令仅在 `work/` 不存在或为空时执行，会把 `original/game` 全量复制到 `work/game/`。它不会生成 TL 模板，也不会调用 Gemini。若 `game_root` 当时指向项目根目录，命令结束后会尝试把 `translator_config.json` 里的 `game_root` 更新为 `work/`；使用 `--no-update-game-root` 可跳过这一步。

如果已有第三方汉化或非标准 TL 文件，建议先跑诊断：

```bash
python gemini_translate_batch.py doctor
```

`doctor` 只检查配置、SDK/launcher 和 TL 文件形态，不调用 Gemini，也不会写回 `.rpy`。当检测到缺少 `work/` 或 TL 模板时，它还会在 `Recommendations` 段提示先运行 `bootstrap-work` 或 `build`。

## 运行模式

当前更推荐优先使用 Gemini Batch 模式；同步模式可显式选择 Gemini 原生调用或可选 LiteLLM 后端。同步模式的 `backend/model/chunk_size/max_source_chars/max_output_tokens` 和可选 RAG 滚动记忆都通过 `translator_config.json` 的 `sync` 配置读取。LiteLLM 未安装或未启用时，不影响 Gemini Batch、doctor 或 GUI 启动。

默认切块策略：

- 同步翻译默认每个 chunk 最多 40 条、`max_source_chars=12000`。
- 普通 Batch 翻译默认每个 chunk 最多 60 条、`max_source_chars=18000`。
- 两种模式都会继续按源文本长度提前切块，避免单个请求输出过长。

当前模型建议：

- 正式 Batch 默认优先使用 `gemini-3.1-flash-lite`。
- 该模型支持 Batch API、结构化输出和 Thinking，定位更适合高频、低延迟、低成本翻译任务。
- RAG 当前默认搭配 `gemini-embedding-001`。
