# 关系与语义分析

文档地图：[docs/README.md](README.md)

仓库提供一个独立的剧本分析入口：

- `extract_relations.py`
  - 用于分析 `game/tl/schinese` 下的角色关系或语义接近度。
  - 内部实现位于 `relation_analyzer/`。

常见命令：

```bash
python extract_relations.py /path/to/game/tl/schinese
python extract_relations.py /path/to/game/tl/schinese --mode semantic
```

说明：

- 默认 `--mode relation`：输出人物关系热力图、关系网络图和 `*_relations.csv`。
- `--mode semantic`：输出角色语义相似度热力图和网络图。
- 不传 `--characters` 时，会自动选择主要说话人。
- 可以用 `--auto-characters` 控制自动推断数量。
- 可以用 `--portraits off` 禁用从 `archive.rpa` 自动读取头像。
- 可以用 `--story-seed-output <GameProject>/translation_context/story_memory/story_graph.seed.json` 在 relation 模式额外导出 Story Memory 候选 seed。
- `relation` 模式不需要 Gemini API。
- `semantic` 模式需要有效的 Gemini API key。

更具体的模块说明见 [`relation_analyzer/README.md`](../relation_analyzer/README.md)。
