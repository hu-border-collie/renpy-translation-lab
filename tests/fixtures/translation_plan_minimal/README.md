# translation_plan_minimal golden fixture（issue #346）

P0 golden fixture 骨架 + P1 期望快照，用于冻结 Sync / Gemini Batch 共用的
`TranslationPlan` 语义合同。两种执行策略对同一输入构建的 plan 中，每个 request
的 `prompt_fingerprint` 必须一致（黄金等价的前身，P4 全量收口）。

## 目录结构

- `game/` — 手写的最小 Ren'Py 源文件（两个 label 形成 block 边界素材、说话人、
  `[Gil_name!t]` 插值、`{i}` 标签，以及术语命中：Dawn Chorus / Mrs. Parker /
  setlist / B-side）。
- `inputs/file_jobs.json` — 从 `game/` 手工推导的 tasks（id 格式
  `file:label:idx:sha256(text)[:8]`，内容派生；`line` 为与源文件 1-based 行号
  对应的 0-based 内部行号，与 `translation_core` 的 `TranslationUnit.line`
  约定一致）。计划构建器消费的是这个文件；`.rpy` 是溯源与 P4 端到端测试的素材。
- `inputs/glossary.json` — D2 词法术语三件套（preserve/normalize/non-translatable）。
- `inputs/macro_setting.txt` — 项目层 Macro 文本。
- `inputs/retrieval_blocks.txt` / `inputs/analysis_blocks.txt` — 检索层与分析层的
  常量参考文本（真实检索逻辑属 P2/P3/#341，这里只冻结组装合同）。
- `inputs/config_snapshot.json` — 非敏感配置快照。
- `inputs/model_profile.json` — 固定 ModelProfile manifest（`model_profile.py`
  序列化形态；不含任何凭据值，credential 仅引用环境变量名）。
- `expected/plan.sync.json` / `expected/plan.gemini_batch.json` — 冻结的期望 plan
  （`translation_plan.build_translation_plan` 的 `to_dict()` 输出）。

注意：golden 构建使用缩小的 `ChunkPolicy(max_items=4)`，让 chapter01 跨两个
label 分块、冻结的 `context_window_spec` 真正锁住 D1 block 边界；生产 D4 默认
（60/18000）由 `ChunkingTests` 单独断言。

## 再生成期望快照

合同有意变更后（并确认 D1–D7 决策允许该变更）：

```bash
RTP_FIXTURE_UPDATE=1 python -m unittest tests.test_translation_plan -q
```

## 约束

- 夹具不得包含 API key、凭据值或敏感 header（`PlanPurityTests` 有扫描断言）。
- 相同输入 + 配置 + ModelProfile 必须字节一致地重建 plan（指纹全部内容派生，
  `run_id` 仅作审计标签、不参与 `plan_id` / `plan_fingerprint`）。
