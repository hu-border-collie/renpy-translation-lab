# Ren'Py Translation Lab

一个面向 Ren'Py 视觉小说的实验性翻译工具仓库，聚焦 Gemini 同步翻译、Batch 作业流、上下文增强和轻量 RAG 记忆层。

## 这是什么

这个仓库有几个主要入口/支持脚本，以及一个内部共享 runtime：

- `gemini_translate.py`
  - Ren'Py 主翻译脚本
  - 负责加载配置、预处理项目、扫描待翻译文本、调用 Gemini 并写回结果
- `gemini_translate_batch.py`
  - 当前更推荐的公开入口
  - Batch 异步批处理脚本
  - 负责 `build / submit / status / probe / download / check / apply / split / repair`
- `rag_memory.py`
  - 轻量 RAG / history store 模块
  - 提供本地 JSON 历史库存储、文本哈希和相似度检索
- `story_memory.py`
  - 可选的结构化剧情记忆模块
  - 从本地 `story_graph.json` 读取角色、关系、术语和场景摘要，并在启用后注入 `STORY MEMORY` prompt 块
- `translator_runtime.py`
  - 内部共享 runtime
  - 为同步脚本和 Batch 脚本提供共用的配置、SDK、校验、响应解析和文件处理逻辑

如果你想找的是 GUI、一键打包、面向普通用户的整套发行流程，这个仓库不是那个方向。

## 核心能力

- 扫描 `game/tl/schinese` 下的 `.rpy` 文件
- 抽取待翻译条目并跳过 `old`
- 构造带上下文的 Gemini 请求
- 自动预处理项目，包括提取脚本和生成 `tl/schinese` 模板
- 生成 Batch 请求包并执行完整的 `build / submit / status / probe / download / check / apply / split / repair` 流程
- 使用本地 history store 做轻量 RAG 检索
- 将历史译文注入后续请求，提升术语和语气一致性
- 可选注入结构化剧情记忆，补充角色、关系、场景和术语上下文
- 允许已知术语按规则保留英文，不再把这类结果一律判成失败

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

当前更推荐优先使用 Batch 模式；同步模式仍可用，并且现在与 Batch 一样统一基于 `google-genai` SDK。同步模式可以通过 `translator_config.json` 里的 `sync.rag.enabled=true` 启用可选 RAG 滚动记忆。

当前模型建议：

- 正式 Batch 默认优先使用 `gemini-2.5-flash`
- `gemini-3-flash-preview` 仍是目标模型，但 Batch 通道仍建议继续做小包稳定性验证
- RAG 当前默认搭配 `gemini-embedding-001`

同步模式：

```bash
python gemini_translate.py
```

Batch 模式：

```bash
python gemini_translate_batch.py bootstrap-rag
python gemini_translate_batch.py build
python gemini_translate_batch.py submit
python gemini_translate_batch.py status
python gemini_translate_batch.py probe
python gemini_translate_batch.py download
python gemini_translate_batch.py check
python gemini_translate_batch.py apply
```

说明：

- `python gemini_translate.py --help` 会显示同步脚本的最小 CLI 帮助
- 同步 RAG 启用后，每个成功写回的小批次会更新本地 history store，后续同步批次会在请求前重新检索并注入相关历史
- Structured Story Memory 默认关闭；需要分别通过 `batch.story_memory.enabled=true` 或 `sync.story_memory.enabled=true` 启用
- 不带子命令直接运行 `gemini_translate_batch.py` 时，默认等价于 `submit`
- Batch 产物默认会写到本地 `logs/` 目录
- `bootstrap-rag` 会扫描当前允许处理的全部 TL `.rpy` 文件，把已有译文预先写入本地 history store；适合在正式 `build / submit` 前先暖库
- `probe` 会用同步请求做最小 smoke test
- `check` 是干跑校验，不会修改 `.rpy`
- `apply` 只会写回通过校验的结果
- 当 `rag.enabled=true` 时，`split` 更接近“静态快照拆包”，不是动态波次式 RAG 工作流；后续包的回灌结果不会自动回流到已经 split 完的旧包
- 本地 RAG store 写入会使用 `.rag_store.lock` 和临时文件 + 原子替换保护 `history.jsonl` / `metadata.json`；如果另一个进程正在写同一个 store，后启动的进程会明确失败并显示锁持有者信息；同机写入进程崩溃后留下的 stale lock 会在确认 PID 已退出时自动回收
- 如果确认没有进程正在写入同一个 RAG store，可手动删除残留的 `.rag_store.lock` 或 `*.tmp.*` 文件来恢复写入；自动清理失败时会输出包含文件路径的 warning
- 加载 RAG store 时，损坏的 metadata 或坏 JSONL 行会输出 warning；可恢复的 history 记录会继续保留

### 可选：Batch RAG 预建库

如果项目里已经有一部分人工译文或旧译文，可以先运行：

```bash
python gemini_translate_batch.py bootstrap-rag
```

这个命令只刷新本地 RAG history store，不会创建 Batch package，也不会修改 `.rpy`。它会复用 `batch.rag` 配置，扫描 `game/tl/schinese` 下当前允许处理的 `.rpy` 文件，提取已有 `old/new` 译文记录并生成 source-only embedding。

典型流程：

```bash
python gemini_translate_batch.py bootstrap-rag
python gemini_translate_batch.py build
python gemini_translate_batch.py submit
```

命令输出会包含 `scan_scope`、`files_scanned`、`scanned`、`embedded`、`reused_embeddings`、`upserted` 和 history record 数量，方便确认预建库是否真的扫描并写入了内容。

注意：`bootstrap-rag` 解决的是“build 前先用已有译文暖库”的问题；它不会让已经 build / split 完的旧请求动态吃到后续 apply 的新结果。需要滚动回灌时，仍要按波次重新 build，或等待后续动态波次编排能力。

### 可选：Structured Story Memory

Structured Story Memory 是现有 glossary / translation-memory RAG 之外的可选上下文层。它不会调用额外 LLM 自动抽取图谱，也不依赖 Neo4j；启用后只读取本地 JSON，并把命中的结构化信息插入到 prompt 的 `STORY MEMORY` 分区。

配置示例见 `translator_config.example.json`：

```json
{
  "batch": {
    "story_memory": {
      "enabled": false,
      "graph_file": "logs/story_memory/story_graph.json",
      "max_context_chars": 1200,
      "top_k_relations": 6,
      "top_k_terms": 12,
      "include_scene_summary": true
    }
  },
  "sync": {
    "story_memory": {
      "enabled": false,
      "graph_file": "logs/story_memory/story_graph.json",
      "max_context_chars": 800,
      "top_k_relations": 4,
      "top_k_terms": 8,
      "include_scene_summary": true
    }
  }
}
```

`story_graph.json` 可以先手写或半自动维护，初版结构类似：

```json
{
  "characters": {
    "eileen": {
      "zh_name": "艾琳",
      "speaker_ids": ["eileen", "eileen_side"],
      "style": "语气轻快，常吐槽，但关键时刻认真。"
    }
  },
  "relations": [
    {
      "left": "eileen",
      "right": "noah",
      "type": "close_friend",
      "note": "两人关系亲近，可以使用自然熟悉的中文语气。",
      "confidence": 0.85
    }
  ],
  "terms": [
    {
      "source": "Void Gate",
      "target": "虚空门",
      "note": "世界观核心术语，必须统一。"
    }
  ],
  "scenes": [
    {
      "file_rel_path": "chapter1.rpy",
      "line_start": 120,
      "line_end": 220,
      "summary": "艾琳和诺亚在天台讨论是否进入危险区域。",
      "characters": ["eileen", "noah"]
    }
  ]
}
```

当前实现仍是 MVP：检索逻辑是轻量启发式，正式 schema 校验、`relation_analyzer` seed 导出、Neo4j 可视化导出和更完整 diagnostics 还属于后续工作。

## 角色关系 / 语义分析

仓库同时提供一个独立的剧本分析入口：

- `extract_relations.py`
  - 用于分析 `game/tl/schinese` 下的角色关系或语义接近度
  - 内部实现位于 `relation_analyzer/`

常见命令：

```bash
python extract_relations.py /path/to/game/tl/schinese
python extract_relations.py /path/to/game/tl/schinese --mode semantic
```

说明：

- 默认 `--mode relation`
  - 输出人物关系热力图、关系网络图和 `*_relations.csv`
- `--mode semantic`
  - 输出角色语义相似度热力图和网络图
- 不传 `--characters` 时，会自动选择主要说话人
- 可以用 `--auto-characters` 控制自动推断数量
- 可以用 `--portraits off` 禁用从 `archive.rpa` 自动读取头像
- `relation` 模式不需要 Gemini API
- `semantic` 模式需要有效的 Gemini API key

更具体的模块说明见 [relation_analyzer/README.md](relation_analyzer/README.md)。

## 环境要求

- Python 3.11+
- `google-genai`，供同步脚本和 Batch 脚本使用
- 有效的 Gemini API Key
- Ren'Py 项目中的 `game/tl/schinese` 翻译目录

## 当前边界

目前项目更偏“核心引擎”，暂未重点覆盖：

- 图形界面（GUI）
- Excel / HTML 协作流
- 面向普通用户的零配置体验
- 完整的游戏解包 / 打包一体化发布流程
- 面向超大项目的完整 RAG 生产工作流（例如严格的波次式回灌编排、多阶段调度策略）
- 完整的结构化剧情图谱生产工作流（例如 schema 校验、自动 seed 生成、Neo4j 可视化导出）

## 项目状态

这是一个仍在持续探索和改进中的个人实验项目。

- 日常实验和更新会先在作者本地工作区中进行；这个公开仓库只同步已经跑通、适合公开发布的版本
- 它不是已经打磨完成、可直接开箱使用的正式产品
- 不保证在所有环境下稳定运行
- 更适合作为思路实现、代码快照和进一步改造的基础
- 项目开发过程中使用了 AI 辅助生成代码，整体方向、功能取舍、测试验证与集成决策由作者负责
- 目前不承诺及时处理 issue、兼容性问题或长期更新
- 当前更推荐使用 Batch 脚本；同步脚本保留用于直接运行、补译、局部修复、smoke test，以及可选的 RAG 滚动记忆验证
- Batch / RAG 仍是主要验证方向；同步 RAG 更适合小批量即时反馈和局部精修，不是 Batch 吞吐流程的替代品
- 当前的 RAG 能力更适合“小包验证 + 逐步扩展”，还不应被表述为已经完成的大项目生产级方案
- Structured Story Memory 目前是可选 MVP，适合作为人工维护剧情上下文的 prompt 增强，不是完整自动图谱系统

执行任何会修改项目文件的操作前，请先备份，并优先在副本上测试。

## 安全说明

不要把以下内容提交到公开仓库：

- 真实 API key
- 你本地的 `api_keys.json`
- 你本地的 `translator_config.json`
- 你本地的 `glossary.json` / `glossary_*.json`
- 你本地的 `story_graph.json` / `story_graph.seed.json`
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
