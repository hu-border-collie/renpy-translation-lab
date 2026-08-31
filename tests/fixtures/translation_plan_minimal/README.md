# translation_plan_minimal golden fixture（issue #346）

P0 golden fixture 骨架 + P1 期望快照，用于冻结 Sync / Gemini Batch 共用的
`TranslationPlan` 语义合同。两种执行策略对同一输入构建的 plan 中，每个 request
的 `prompt_fingerprint` 必须一致（黄金等价的前身，P4 全量收口）。

## 目录结构

- `game/` — 手写的最小 Ren'Py 源文件（两个 label 形成 block 边界素材、说话人、
  `[CharacterA_name!t]` 插值、`{i}` 标签，以及术语命中：Sample Ensemble / Director B /
  setlist / B-side）。
- `inputs/file_jobs.json` — 从 `game/` 手工推导的 tasks（id 格式与生产
  `translation_core.build_identity_v2` 一致：`file:label:idx:sha1(text)[:8]`，内容
  派生；`line` 为与源文件 1-based 行号对应的 0-based 内部行号，与
  `TranslationUnit.line` 约定一致，测试断言每个 `line` 索引的源行包含该任务
  文本）。计划构建器消费的是这个文件；`.rpy` 是溯源与 P4 端到端测试的素材。
- `inputs/glossary.json` — D2 词法术语三件套（preserve/normalize/non-translatable）。
- `inputs/macro_setting.txt` — 项目层 Macro 文本。
- `inputs/retrieval_blocks.txt` / `inputs/analysis_blocks.txt` — 检索层与分析层的
  常量参考文本（真实检索逻辑属 P2/P3/#341，这里只冻结组装合同）。
- `inputs/config_snapshot.json` — 非敏感配置快照。
- `inputs/model_profile.json` — 固定 ModelProfile manifest（`model_profile.py`
  序列化形态；不含任何凭据值，credential 仅引用环境变量名）。
- `expected/plan.sync.json` / `expected/plan.gemini_batch.json` — 冻结的期望 plan
  （`translation_plan.build_translation_plan` 的 `to_dict()` 输出）。

注意：golden 构建使用缩小的 `ChunkPolicy(max_items=3)`，让 chapter01 在 label 内
和跨 label 都产生分块——冻结的 `context_window_spec` 锁住 D1 block 边界，冻结的
`user_prompt` 含块内 `CONTEXT BEFORE/AFTER`（带说话人标签的 dict 渲染）；生产
D4 默认（60/18000）由 `ChunkingTests` 单独断言。多行文本 fixture 由
`.gitattributes` 固定为 LF，测试读取时亦做 CRLF 归一化，指纹跨 checkout 稳定。

## 再生成期望快照

合同有意变更后（并确认 D1–D7 决策允许该变更）：

```bash
RTP_FIXTURE_UPDATE=1 python -m unittest tests.test_translation_plan -q
```

## 约束

- 夹具不得包含 API key、凭据值或敏感 header（`PlanPurityTests` 有扫描断言）。
- 相同输入 + 配置 + ModelProfile 必须字节一致地重建 plan（指纹全部内容派生，
  `run_id` 仅作审计标签、不参与 `plan_id` / `plan_fingerprint`）。
