# Ren'Py Translation Lab

一个面向 Ren'Py 视觉小说的实验性翻译工具仓库，聚焦 Gemini 同步翻译、Batch 作业流、上下文增强和轻量 RAG 记忆层。

## 这是什么

这个仓库当前只保留 3 个非 GUI 核心脚本：

- `gemini_translate.py`
  - Ren'Py 主翻译脚本
  - 负责加载配置、预处理项目、扫描待翻译文本、调用 Gemini 并写回结果
- `gemini_translate_batch.py`
  - Batch 异步批处理脚本
  - 负责 `build / submit / status / download / check / apply / split / repair`
- `rag_memory.py`
  - 轻量 RAG / history store 模块
  - 提供本地 JSON 历史库存储、文本哈希和相似度检索

如果你想找的是 GUI、一键打包、面向普通用户的整套发行流程，这个仓库不是那个方向。

## 核心能力

- 扫描 `game/tl/schinese` 下的 `.rpy` 文件
- 抽取待翻译条目并跳过 `old`
- 构造带上下文的 Gemini 请求
- 自动预处理项目，包括提取脚本和生成 `tl/schinese` 模板
- 生成 Batch 请求包并执行完整的 `build / submit / status / download / check / apply / split / repair` 流程
- 使用本地 history store 做轻量 RAG 检索
- 将历史译文注入后续请求，提升术语和语气一致性

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

把示例文件复制为你自己的本地配置文件：

- `api_keys.example.json` -> `api_keys.json`
- `translator_config.example.json` -> `translator_config.json`
- `glossary.example.json` -> `glossary.json`
- `macro_setting.example.md` -> `macro_setting.md`（可选，供 Batch 的 `macro_setting_file` 使用）

说明：

- `api_keys.json` 保存 Gemini API Key，不要提交到公开仓库
- `translator_config.json` 保存本地游戏路径与批处理参数，不要提交到公开仓库
- `glossary.json` 通常包含项目私有术语，也建议本地维护
- `macro_setting.md` 往往包含剧情、角色口吻、世界观约束，也建议本地维护
- 如果你不想使用 `api_keys.json`，也可以改用环境变量 `GEMINI_API_KEY`、`GEMINI_API_KEY_2`、`GEMINI_API_KEY_3`
- 如果你不想使用 `translator_config.json`，也可以至少通过 `GAME_ROOT` 或 `SA_GAME_ROOT` 指向目标 `work` 目录
- `glossary.json` 是可选文件；不存在时脚本会退回内置默认术语规则

### 3. 准备项目目录

脚本默认假设你最终指向的是某个游戏的 `work` 目录，典型结构如下：

```text
Game_Example/
├─ original/
├─ work/
│  └─ game/
│     └─ tl/
│        └─ schinese/
└─ build/
```

说明：

- 默认目标语言是 `schinese`
- 可以通过 `translator_config.json` 里的 `tl_subdir` 和 `prepare.language` 调整
- 如果启用了 `prepare`，脚本会尝试从 `original/game` 提取脚本并自动生成 `tl/schinese` 模板

### 4. 运行

同步模式：

```bash
python gemini_translate.py
```

Batch 模式：

```bash
python gemini_translate_batch.py build
python gemini_translate_batch.py submit
python gemini_translate_batch.py status
python gemini_translate_batch.py download
python gemini_translate_batch.py check
python gemini_translate_batch.py apply
```

说明：

- 不带子命令直接运行 `gemini_translate_batch.py` 时，默认等价于 `submit`
- Batch 产物默认会写到本地 `logs/` 目录
- `check` 是干跑校验，不会修改 `.rpy`
- `apply` 只会写回通过校验的结果

## 环境要求

- Python 3.11+
- `google-generativeai`，供同步脚本使用
- `google-genai`，供 Batch 脚本使用
- 有效的 Gemini API Key
- Ren'Py 项目中的 `game/tl/schinese` 翻译目录

## 当前边界

目前项目更偏“核心引擎”，暂未重点覆盖：

- 图形界面（GUI）
- Excel / HTML 协作流
- 面向普通用户的零配置体验
- 完整的游戏解包 / 打包一体化发布流程

## 项目状态

这是一个仍在持续探索和改进中的个人实验项目。

- 它不是已经打磨完成、可直接开箱使用的正式产品
- 不保证在所有环境下稳定运行
- 更适合作为思路实现、代码快照和进一步改造的基础
- 项目开发过程中使用了 AI 辅助生成代码，整体方向、功能取舍、测试验证与集成决策由作者负责
- 目前不承诺及时处理 issue、兼容性问题或长期更新

执行任何会修改项目文件的操作前，请先备份，并优先在副本上测试。

## 安全说明

不要把以下内容提交到公开仓库：

- 真实 API key
- 你本地的 `api_keys.json`
- 你本地的 `translator_config.json`
- 你本地的 `glossary.json` / `glossary_*.json`
- 你本地的 `macro_setting.md`
- 私有游戏脚本
- 本地 batch 结果
- history / rag store
- 日志和缓存

## 适合谁使用

更适合下面这类使用者：

- 已经熟悉 Ren'Py 项目目录结构
- 能自行准备 `work/game/tl/schinese` 或理解 `prepare` 行为
- 能阅读 Python 脚本并按需修改本地配置
- 接受这是实验性工具，而不是稳定打包好的最终产品

## 目录结构（示例）

```text
.
├─ gemini_translate.py
├─ gemini_translate_batch.py
├─ rag_memory.py
├─ api_keys.example.json
├─ translator_config.example.json
├─ glossary.example.json
├─ macro_setting.example.md
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ docs/
│  └─ roadmap.md
```

## 路线图

详见 `docs/roadmap.md`。

## License

见 `LICENSE` 文件。
