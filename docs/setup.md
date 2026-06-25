# 安装与本地配置

文档地图：[docs/README.md](README.md)

本文档承接根目录 `README.md` 中会拖慢阅读节奏的本地配置、项目目录和 Ren'Py SDK 细节。

## 本地配置文件

把示例文件复制为本地配置：

- `api_keys.example.json` -> `api_keys.json`
- `translator_config.example.json` -> `translator_config.json`
- `glossary.example.json` -> `glossary.json`
- `macro_setting.example.md` -> `macro_setting.md`

这些文件通常包含 API key、本地游戏路径、项目私有术语或剧情设定，不应提交到公开仓库。

说明：

- `api_keys.json` 保存 Gemini API Key；旧的 `batch_size/max_chars` 等字段仍兼容，但不再推荐写在这里。
- `translator_config.json` 保存本地游戏路径、模型、include 过滤和同步 / Batch 分块参数。
- `glossary.json` 通常包含项目私有术语；不存在时脚本会退回内置默认术语规则。
- `macro_setting.md` 往往包含剧情、角色口吻、世界观约束，可供 Batch 的 `macro_setting_file` 使用。
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
│        └─ schinese/
└─ build/
```

默认目标语言是 `schinese`，可以通过 `translator_config.json` 里的 `tl_subdir` 和 `prepare.language` 调整。

推荐先使用 Ren'Py SDK 生成标准 `tl/<language>` 模板；如果启用了 `prepare`，脚本会尝试从 `original/game` 提取脚本并自动调用 Ren'Py 生成或刷新 `tl/schinese` 模板。自动模板生成需要 Ren'Py SDK 或目标游戏自带的 Ren'Py launcher；如果已经有可用 TL 文件，缺少 SDK 时仍可直接处理现有 TL。

不要用 Ren'Py 的 `--empty` 生成空模板。本工具的初译流程需要目标行保留原文，之后再把目标行或 `new` 行替换成中文。

可以通过 `translator_config.json` 或环境变量 `RENPY_SDK_DIR` 指定 SDK 位置；如果没有配置，脚本会尝试在 `game_root` 附近和工具工作区里自动查找 `renpy-*-sdk`：

```json
{
  "game_root": "C:/games/Game_Example/work",
  "tl_subdir": "game/tl/schinese",
  "prepare": {
    "enabled": true,
    "generate_template": true,
    "refresh_existing_template": true,
    "language": "schinese",
    "renpy_sdk_dir": "C:/RenPy/renpy-8.5.2-sdk"
  }
}
```

如果还没有 `work/` 工作目录，但项目根下已有 `original/game`，可以先初始化 work 副本：

```bash
python gemini_translate_batch.py bootstrap-work
```

该命令仅在 `work/` 不存在或为空时执行，会把 `original/game` 全量复制到 `work/game/`。它不会生成 TL 模板，也不会调用 Gemini。若 `game_root` 当时指向项目根目录，命令结束后还会把 `translator_config.json` 里的 `game_root` 更新为 `work/`。

如果已有第三方汉化或非标准 TL 文件，建议先跑诊断：

```bash
python gemini_translate_batch.py doctor
```

`doctor` 只检查配置、SDK/launcher 和 TL 文件形态，不调用 Gemini，也不会写回 `.rpy`。当检测到缺少 `work/` 或 TL 模板时，它还会在 `Recommendations` 段提示先运行 `bootstrap-work` 或 `build`。

## 运行模式

当前更推荐优先使用 Batch 模式；同步模式仍可用，并且现在与 Batch 一样统一基于 `google-genai` SDK。同步模式的 `model/chunk_size/max_source_chars/max_output_tokens` 和可选 RAG 滚动记忆都通过 `translator_config.json` 的 `sync` 配置读取。

默认切块策略：

- 同步翻译默认每个 chunk 最多 40 条、`max_source_chars=12000`。
- 普通 Batch 翻译默认每个 chunk 最多 60 条、`max_source_chars=18000`。
- 两种模式都会继续按源文本长度提前切块，避免单个请求输出过长。

当前模型建议：

- 正式 Batch 默认优先使用 `gemini-3.1-flash-lite`。
- 该模型支持 Batch API、结构化输出和 Thinking，定位更适合高频、低延迟、低成本翻译任务。
- RAG 当前默认搭配 `gemini-embedding-001`。
