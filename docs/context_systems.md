# 上下文系统

文档地图：[docs/README.md](README.md)

本文档说明 Batch / sync 可选上下文层：RAG history store、Batch source-only index 和 Structured Story Memory。

## 如何选择上下文层

- RAG history store：使用“已有原文 + 已有译文”保持术语、称呼和风格一致。
- Batch source-only index：只使用全文原文，为新项目或译文很少的项目提供远距离剧情上下文。
- Structured Story Memory：人工维护的结构化角色、关系、术语和场景图谱。

这些上下文层会进入不同 prompt 分区，避免把“以前的译法参考”和“相关剧情原文”混在一起。

**开关分层**：

- **是否启用**批量 RAG / 原文索引 / build 时暖库：优先读当前项目 `work/project_context_settings.json`（见 [setup.md](setup.md#项目级上下文开关)）；无项目文件时回退 `translator_config.json` 的 `batch.rag` / `batch.source_index`。
- **库文件放哪**：由下方 `context_storage` 与可选的 `store_dir` 决定（与开关是否按项目是两件独立的事）。

## 上下文文件存放位置

默认情况下，上下文库仍保存在工具项目的 `logs/` 下，保持旧行为：

```text
logs/rag_store/<project_slug>/
logs/source_index_store/<project_slug>/
logs/story_memory/story_graph.json
```

如果希望这些文件跟随具体游戏项目，可以在 `translator_config.json` 顶层设置：

```json
{
  "context_storage": {
    "location": "game",
    "game_dir_name": "translation_context"
  }
}
```

启用后，默认路径会改为当前 `work` 目录同级的游戏大目录：

```text
<GameProject>/translation_context/rag_store/
<GameProject>/translation_context/source_index_store/
<GameProject>/translation_context/story_memory/story_graph.json
```

`batch.rag.store_dir`、`batch.source_index.store_dir`、`sync.rag.store_dir` 和 `story_memory.graph_file` 仍然可以显式指定；一旦填了具体路径，就优先使用该路径，不再跟随 `context_storage.location`。

## Batch RAG 预建

如果项目里已经有一部分人工译文或旧译文，可以先运行：

```bash
python gemini_translate_batch.py bootstrap-rag
```

这个命令只刷新本地 RAG history store，不会创建 Batch package，也不会修改 `.rpy`。它会复用 `batch.rag` 配置，扫描当前配置的 TL 目录（默认 `game/tl/schinese`）下允许处理的 `.rpy` 文件，提取已有 `old/new` 译文记录并生成 source embedding。

典型流程：

```bash
python gemini_translate_batch.py bootstrap-rag
python gemini_translate_batch.py build
python gemini_translate_batch.py submit
```

也可以导入外部平行语料 JSONL 作为额外 seed：

```bash
python gemini_translate_batch.py bootstrap-rag --seed-jsonl parallel_corpus.jsonl
```

如果只想导入外部 seed，TL 目录可以暂时不存在；此时 TL 扫描数量会是 0，只导入 JSONL 中的有效记录。

JSONL 每行是一个对象，支持以下字段：

```json
{"source": "Aether Gate", "translation": "以太门", "file_rel_path": "external/memory.txt", "line": 1}
```

字段说明：

- `source` 或 `source_text`：原文。
- `translation`、`translated_text` 或 `target`：译文。
- `file_rel_path` / `file`、`line` / `line_start` / `line_end`：可选定位信息，用于生成稳定 memory id 和 diagnostics；如果未提供文件路径，会使用带 seed 文件内容指纹的默认来源名，避免多个同名 JSONL 相互覆盖，同时保持移动文件后的重复导入可去重。
- `memory_id`：可选；不提供时会根据来源、行号和原文生成。

空行会被忽略；坏 JSON 会计入 `external_seed_invalid_json`，缺少原文/译文、或原文和译文完全相同的有效行会计入 `external_seed_filtered`，两者合计为 `external_seed_skipped`。

命令输出会包含 `scan_scope`、`files_scanned`、`scanned`、`external_seed_records`、`external_seed_invalid_json`、`external_seed_filtered`、`external_seed_skipped`、`embedded`、`reused_embeddings`、`upserted` 和 history record 数量，方便确认预建库是否真的扫描并写入了内容。

注意：`bootstrap-rag` 解决的是“build 前先用已有译文暖库”的问题；它不会让已经 build / split 完的旧请求动态吃到后续 apply 的新结果。需要滚动回灌时，仍要按波次重新 build，或等待后续动态波次编排能力。

同步 RAG 启用后，每个成功写回的小批次会更新本地 history store，后续同步批次会在请求前重新检索并注入相关历史。

## RAG store 写入安全

本地 RAG store 写入会使用 `.rag_store.lock` 和临时文件 + 原子替换保护 `history.jsonl` / `metadata.json`。如果另一个进程正在写同一个 store，后启动的进程会明确失败并显示锁持有者信息；同机写入进程崩溃后留下的 stale lock 会在确认 PID 已退出时自动回收。

如果确认没有进程正在写同一个 RAG store，可手动删除残留的 `.rag_store.lock` 或 `*.tmp.*` 文件来恢复写入；自动清理失败时会输出包含文件路径的 warning。加载 RAG store 时，损坏的 metadata 或坏 JSONL 行会输出 warning；可恢复的 history 记录会继续保留。

## Batch 原文索引

Source-only index 不需要已有中文翻译。它扫描 TL 文件中的注释原文或可翻译字符串，写入独立的 source store：

```bash
python gemini_translate_batch.py bootstrap-source-index
```

默认存储位置取决于 `context_storage.location`：工具内部模式使用 `logs/source_index_store/<project_slug>/`，游戏目录模式使用 `<GameProject>/translation_context/source_index_store/`。

主要文件：

```text
source_metadata.json
source_segments.jsonl
```

每条 segment 会记录 `source_id`、`file_rel_path`、line span、`source_checksum`、`source_text`、embedding、embedding model/task/dim、created/updated time 等字段。该 store 不混入 translation memory history store。

命令会：

1. 扫描当前所有 `.rpy` 翻译模板文件，将原文分组为 source segments。
2. 对比本地数据库中已有的索引：
   - 如果对应位置的内容和模型配置未发生变化，则复用现有 embedding。
   - 如果发生变化或不存在，则分批调用 embedding 接口生成向量。
   - 识别并列出数据库中存在但当前项目中已失效的 stale segments。
3. 输出同步前统计信息。
4. 默认自动 prune stale 记录；如需保留，可传入 `--no-prune`。

Batch 构建时是否检索 source index 由 `batch.source_index.enabled` 控制。当前 source index 复用 `batch.rag` 中的 embedding 模型、query/document task type、output dimensionality 和 `segment_lines`；`batch.source_index` 自身控制检索开关、命中数、相似度阈值、单条原文截断预算和独立存储目录：

```json
{
  "batch": {
    "rag": {
      "embedding_model": "gemini-embedding-001",
      "query_task_type": "RETRIEVAL_QUERY",
      "document_task_type": "RETRIEVAL_DOCUMENT",
      "output_dimensionality": 768,
      "segment_lines": 4
    },
    "source_index": {
      "enabled": true,
      "top_k": 4,
      "min_similarity": 0.72,
      "char_limit": 220,
      "store_dir": ""
    }
  }
}
```

启用后，普通 Batch manifest 会写入：

- `source_index_store_path`：实际使用的 source-only store 路径。
- `source_index_settings`：schema version、`top_k`、`min_similarity`、`char_limit` 和每个 chunk 的字符预算。
- `source_index_summary`：命中 chunk 数、source hit 数、命中率、截断次数、实际注入字符数、字符预算、metadata/stale 过滤数、相似度过滤数、检索失败数和失败原因。
- 每个 chunk 的 `source_index_stats`：查询字符数、命中数、metadata 过滤明细、截断数、store schema version 和 store 路径。

Prompt 中 source index 命中会进入独立的 `RELATED PROJECT CONTEXT` 分区，只包含 `Source excerpt`，不会混入 `RETRIEVED MEMORY`，也不会携带译文。`batch.source_index.enabled=false` 时不会读取 source store，也不会增加该 prompt 分区。

## 结构化剧情记忆

Structured Story Memory 是 glossary / translation-memory RAG 之外的可选上下文层。它不会调用额外 LLM 自动抽取图谱，也不依赖 Neo4j；启用后只读取本地 JSON，并把命中的结构化信息插入到 prompt 的 `STORY MEMORY` 分区。

配置示例见 `translator_config.example.json`：

```json
{
  "batch": {
    "story_memory": {
      "enabled": false,
      "graph_file": "",
      "max_context_chars": 1200,
      "top_k_relations": 6,
      "top_k_terms": 12,
      "include_scene_summary": true
    }
  },
  "sync": {
    "story_memory": {
      "enabled": false,
      "graph_file": "",
      "max_context_chars": 800,
      "top_k_relations": 4,
      "top_k_terms": 8,
      "include_scene_summary": true
    }
  }
}
```

仓库提供了两个公开参考文件：

- `story_graph.schema.json`：正式的 JSON Schema，描述推荐的 `schema_version=1` 结构。
- `story_graph.example.json`：可复制后按项目修改的示例图谱。

`story_graph.json` 可以先手写或半自动维护，推荐结构类似：

```json
{
  "schema_version": 1,
  "characters": {
    "eileen": {
      "name": "Eileen",
      "zh_name": "艾琳",
      "speaker_ids": ["eileen", "eileen_side"],
      "aliases": ["Miss Eileen"],
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

为了兼容早期手写图谱，加载器仍接受 `terms` 对象映射、术语字符串条目，以及 `term` / `translation` 这类旧字段名；新图谱建议优先使用示例里的 `source` / `target` 写法。

加载 `story_graph.json` 时会做轻量基础校验：顶层集合类型、角色别名字段、关系 `left/right/confidence`、术语有效内容、场景行号与角色列表等明显问题会输出 warning。校验是非阻塞的；有效部分仍会被规范化后继续用于检索。

Batch `build` 生成的 `manifest.json` 会在 `story_memory_summary` 中记录 diagnostics，包括 `graph_file`、命中 chunk 数、命中率、characters / relations / terms / scenes 各类命中数量、总命中数、预算内格式化字符数，以及有多少个 `STORY MEMORY` 块会被 `max_context_chars` 截断。`split` 会按子包重新计算这些统计。

`repair` 在 `batch.story_memory.enabled=true` 时也会复用同一个 `story_memory.py` 读取和格式化路径，为 repair request 注入受 `max_context_chars` 限制的 `STORY MEMORY` 块，并在 `repair_summary.json` 记录 repair job 的 Story Memory 命中统计。`probe` 只探测当前 package 里的既有 `requests.jsonl`，不会重新读取或刷新 `story_graph.json`；如果这些请求是在 Story Memory 启用时 build 出来的，probe 会自然测试已经写入请求的 prompt。

`relation_analyzer` 可以额外导出 `story_graph.seed.json` 候选数据，帮助从 Ren'Py 剧本里半自动整理 `speaker_ids`、候选角色和候选关系：

```bash
python extract_relations.py /path/to/game/tl/<language> --mode relation --story-seed-output <GameProject>/translation_context/story_memory/story_graph.seed.json
```

seed 中的关系统一标记为 `candidate`，只包含共场景、对话往来、相互提及、来源文件和 speaker 统计等可审查信息；如果同一输入目录里存在 `define e = Character("Eileen")` 这类 Ren'Py 角色定义，seed 会优先用定义名作为 speaker 名称候选。它不会自动断言恋人、敌人、上下级等强语义关系。建议人工确认并编辑后，再作为正式 `story_graph.json` 使用。

当前实现仍是 MVP：检索逻辑是轻量启发式，Neo4j 可视化导出，以及 sync 运行日志里的更完整 Story Memory diagnostics 还属于后续工作。

## RAG store 性能基准

为了在大项目规模下评估本地 RAG Store（`JsonRagStore`）的性能极限并防止退化，项目中包含了一个离线性能基准脚本 [benchmark_rag_store.py](../benchmark_rag_store.py)。

### 运行方式

该脚本在本地离线运行，使用合成的随机向量记录，不依赖 Gemini Embedding API，也不需要真实的 API Key。

```bash
# 运行默认规模测试（100, 1000, 10000 条记录）
python benchmark_rag_store.py

# 快速 Smoke 测试或自定义参数
python benchmark_rag_store.py --sizes 10,100 --queries 5 --dim 768
```

### 测量指标说明

该基准工具在不同数据库规模（N）下测量并输出以下关键指标：

1. **Bulk Upsert (s)**：首次将 N 条记录批量写入空数据库并刷盘的时间。
2. **Load Store (s)**：清空内存后，重新从 `history.jsonl` 文件加载 N 条记录至内存的耗时。
3. **Zero-Hit Search (ms)**：针对 N 条记录，执行指定的随机向量查询（默认 `min_similarity=0.72`，基本无命中）的平均耗时。主要评估纯线性 Cosine Similarity 计算的开销。
4. **Cache-Hit Search (ms)**：针对 N 条记录，使用已存在于库中的记录向量进行查询（默认 `min_similarity=0.72`，保证至少 1 个匹配）的平均耗时。主要评估有命中时的排序、Slicing 和结果格式化开销。
5. **All-Match Search (ms)**：设置 `min_similarity=-1.0` 强行匹配全部记录时的平均查询耗时。主要评估全量参与排序和裁剪时的开销。
6. **Incremental Upsert (s)**：在已存在 N 条记录的数据库中，增量 Upsert 10 条新记录的耗时。由于当前实现需要全量重写 `history.jsonl`，该指标反映了频繁写入时的性能开销。
7. **File Size (MB)**：存储 N 条记录时 `history.jsonl` 在磁盘上的文件大小。

### 性能瓶颈与优化建议判定

运行结束后，脚本会根据实测数据动态提示优化建议：
- **Search 耗时阈值（50ms）**：若大规模下的 Average Search 超过 50ms，建议开启 **Norm Caching**（在内存中缓存向量范数，避免在 Cosine Loop 中重复计算）或引入 **NumPy 向量化**。
- **Upsert 耗时阈值（1.0s）**：若 Incremental Upsert 超过 1.0s，说明全量原子重写文件成为瓶颈，建议改用 **Append-only log + Compaction** 存储方案，或引入轻量数据库（如 **SQLite**）。

### 当前参考基线

以下结果来自 2026-06-16 在本地 Windows 开发环境运行默认命令：

```bash
python benchmark_rag_store.py
```

参数为 `--sizes 100,1000,10000 --queries 20 --dim 768 --seed 42`。这些数字不是测试断言，主要用于记录当前实现的大致量级；不同机器、磁盘和 Python 版本会有波动。

| Scale (N) | Bulk Upsert (s) | Load Store (s) | Zero-Hit Search (ms) | Cache-Hit Search (ms) | All-Match Search (ms) | Incremental Upsert (s) | File Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.0415 | 0.0261 | 10.87 | 10.88 | 10.83 | 0.0461 | 1.67 |
| 1000 | 0.3703 | 0.2719 | 133.87 | 158.99 | 159.50 | 0.5616 | 16.68 |
| 10000 | 5.1964 | 3.9065 | 1554.64 | 1569.88 | 1652.19 | 6.0499 | 166.84 |

这组基线显示：当前 JSONL + 内存字典实现可以可靠完成 10,000 条合成记录的离线 benchmark，但 768 维向量的纯 Python 线性检索和增量写入时的全量重写已经在 10,000 级别明显超过阈值。后续优先级应是 search 侧的 norm cache / NumPy vectorization，以及 write 侧的 append-only log + compaction 或 SQLite 迁移评估。
