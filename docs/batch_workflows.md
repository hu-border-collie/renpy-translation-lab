# Batch 工作流与安全检查

文档地图：[docs/README.md](README.md)

本文档记录 Batch 流程中偏内部或高级的部分。日常入口见根目录 `README.md`。

## 目标语言与 TL 路径

- 默认目标语言为 `schinese`，对应 TL 路径 `game/tl/schinese`。
- 通过 `translator_config.json` 的 `tl_subdir` 与 `prepare.language` 可改为 `japanese`、`korean` 等 Ren'Py 支持的语言目录。
- `doctor`、`build`、`generate-template` 与 Batch 启动 banner 会打印当前 `tl_subdir` 与目标语言；manifest 也会记录 `tl_subdir` / `target_language` 便于追溯。
- `prepare.language` 传给 Ren'Py 的 `translate` 命令；`tl_subdir` 决定脚本扫描与写回路径。两者末段应一致。

### 校验边界（当前版本）

`check` 默认要求译文包含中文字符（实现上检测 Unicode 中文范围，不包含日文假名或韩文），适用于以简体中文为目标的批次。以下情况允许保留非中文译文（无需改配置）：

- 术语表中的固定译法、保留英文名/缩写、玩家名比较行等启发式规则
- 特定 UI/制作人员名单文件路径上的静态文本（`batch.non_chinese_validation` 白名单，build 时写入 manifest `non_chinese_rules`）

可在 `translator_config.json` 的 `batch.non_chinese_validation` 中覆盖或追加路径；GUI 高级设置提供「非中文白名单追加路径」。默认白名单与当前 After Class / Glory Hounds 项目兼容。

若目标语言为日语、韩语等，`No Chinese characters` 失败可能属于预期行为，需要调整校验策略或白名单。换语言前请先跑 `doctor` 确认 TL 路径，并在 `check` 结果中逐项确认失败原因。

`allow_non_chinese_batch_translation` 在单次校验调用内会缓存 TL/source 文件读取，避免 short-circuit OR 链重复打开同一文件；pass/fail 结果与缓存前一致。

## 命令说明

- `gemini_translate_batch.py` 需要显式子命令；不带子命令会打印帮助并退出。
- Batch 产物默认写到 `logs/batch_jobs/<package>/`。
- `doctor` 只检查当前 `game_root` / `tl_subdir`、SDK/launcher、TL 模板和 `old/new` / 剧情块形态，不调用 Gemini，也不会写回 `.rpy`。
- `probe` 会用同步请求做最小 smoke test。
- `check` 是干跑校验，不会修改 `.rpy`；它会把当前 manifest、results、目标 item 形状和 check contract version 写入 `last_check_summary.check_fingerprint`，输出 `safe / warn / block` 安全等级，并在包目录写入 `check_failures.jsonl`。
- `apply` 默认要求最近一次 `check` 对应当前 manifest/results，且安全等级必须是 `safe`；未 check、results 变化、manifest item 变化、`warn` 或 `block` 都会拒绝写回。
- `--force` 只绕过“manifest 已经 apply 过”的重复写回保护，不会绕过 stale check、source snapshot 校验或 `block`。
- `apply` 写回前会再次校验当前源文本；如果 apply 阶段发现漂移，会拒绝写回并在包目录写入 `apply_failure_report.json` / `failures.jsonl`。
- 当 `rag.enabled=true` 时，`split` 更接近“静态快照拆包”，不是动态波次式 RAG 工作流；后续包的回灌结果不会自动回流到已经 split 完的旧包。

## 订正流程

订正模式扫描已有 `old/new` / TL 注释译文，先生成预览，显式 apply 后才写回当前译文行：

```bash
python gemini_translate_batch.py build-revisions
python gemini_translate_batch.py submit logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py status logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py download logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py preview-revisions logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py apply-revisions logs/batch_jobs/<package>/manifest.json
```

同步订正模式不走 Batch API；默认只写预览报告，传 `--apply` 才写回：

```bash
python gemini_translate_batch.py sync-revisions --limit 3
python gemini_translate_batch.py sync-revisions --apply
```

`build-revisions` 会复用 include 过滤、glossary、macro setting、可选 RAG / Story Memory，把已有原文和当前译文送入 Batch。`preview-revisions` 导出 `revision_preview.jsonl` 和 `revision_preview.md`，`apply-revisions` 会在写回前重新校验当前文件中的旧译文快照。

当前 `safe / warn / block` 强制闸门只覆盖普通 translation manifest 的 `check/apply`；订正写回仍走 `preview-revisions -> apply-revisions` 的独立快照校验。

`sync-revisions` 复用订正 prompt、schema、RAG / Story Memory 注入、预览报告和写回前源快照校验；默认只预览，传 `--apply` 才调用 `apply-revisions` 写回。

## 关键词提取流程

关键词提取模式只生成候选报告，不写回 `.rpy` / `glossary.json` / `story_graph.json`：

```bash
python gemini_translate_batch.py build-keywords
python gemini_translate_batch.py submit logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py status logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py download logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py export-keywords logs/batch_jobs/<package>/manifest.json
```

同步关键词提取模式不走 Batch API，直接生成候选报告：

```bash
python gemini_translate_batch.py sync-keywords --limit 3
```

`build-keywords` 会复用 include 过滤和 Batch manifest，默认不运行 prepare，按较大 chunk 扫描 TL 文本并要求模型输出 `candidates`、`chunk_summary`、`summary_evidence_item_ids`。候选项里包含 `source`、`suggested_target`、`category`、`confidence`、`evidence`、`source_item_ids`。如果确实要先刷新 TL 模板，可显式传 `--prepare`。

`export-keywords` 会导出去重后的 `keyword_candidates.jsonl` / `keyword_candidates.md`，并额外导出 chunk 级剧情概要 `keyword_chunk_summaries.jsonl` / `keyword_chunk_summaries.md`。报告会标出缺失 chunk row 或无法精确定位的候选 / 概要来源。

`sync-keywords` 复用关键词 prompt、schema、候选去重、chunk 概要和 JSONL / Markdown 导出逻辑，适合小范围即时跑报告。

订正 manifest 的 `mode=revision`，关键词 manifest 的 `mode=keyword_extraction`，普通 `check/apply` 会拒绝处理，避免把非翻译结果误写回 `.rpy`。

## Manifest 与 identity v2

Batch `build` 会生成：

```text
logs/batch_jobs/<package>/manifest.json
```

后续 `submit / status / download / check / apply` 默认都围绕这个 manifest 工作。新建的普通翻译、订正和关键词 manifest 会写入 `manifest_version=2` 与 `core_schema_version=2`。

普通翻译和订正 item 的 `id` 使用 identity v2：归一化后的文件相对路径、Ren'Py translate block 名、重复 block occurrence、block 内可翻译单元序号，以及原文 checksum。行号和列位置仍保存在 item 上，但它们只是当前写回 location hint，不再是唯一身份。

这个拆分的含义是：

- `identity` 用于跨 `build / check / apply / repair` 识别同一个翻译单元。
- `location` 是当前文件里的行号、列位置、translate block 等写回定位信息，可能因为插入空行、局部手改或模板刷新而漂移。
- `snapshot` 是写回前校验用的当前源文本或当前译文；即使 identity 能重定位，`check/apply` 和 `apply-revisions` 仍会复核快照，不会盲写。

v2 重定位覆盖普通 translation 和 revision manifest：`check`、`apply`、`preview-revisions` 和 `apply-revisions` 会在处理 v2 结果前重扫当前 TL 文件，用 v2 id 刷新行号和列位置。旧 manifest 保持兼容 fallback：`manifest_version` 缺失或为 `1` 时继续使用 manifest 内原始 location，不做 v2 重定位；`doctor` 会提示本地 `logs/batch_jobs` 中的旧 manifest。

RAG / history store 继续以 `memory_id` 关联记录。升级到 v2 后，已有旧 key 不会立即强制迁移；写入新记录时会先尝试按 `source_checksum` 复用旧记录的 source embedding，避免因为 id 升级就全量重算。如原文大量变动或文件结构重排，仍建议重新 `bootstrap-rag`。

## Golden corpus 测试

Golden corpus 测试使用离线 fixture 和 mock 模型结果验证格式合约，不调用 Gemini，也不需要真实 API key。

```bash
python -m unittest tests.test_batch_golden_corpus.BatchGoldenCorpusTests -q
python -m unittest tests.test_batch_golden_corpus.RevisionGoldenCorpusTests -q
python -m unittest tests.test_batch_golden_corpus.KeywordGoldenCorpusTests -q
```

fixture 位置：

- `tests/fixtures/golden_batch_minimal/`：普通 Batch 翻译的 `build -> check -> apply` 合约。
- `tests/fixtures/golden_revision_minimal/`：`build-revisions -> preview-revisions -> apply-revisions` 合约。
- `tests/fixtures/golden_keyword_minimal/`：`build-keywords -> export-keywords` 合约。

如果有意修改 prompt、manifest、schema 或写回行为，先确认差异合理，再更新 golden 输出：

```powershell
$env:UPDATE_GOLDEN_BATCH = "1"
python -m unittest tests.test_batch_golden_corpus.BatchGoldenCorpusTests -q
Remove-Item Env:UPDATE_GOLDEN_BATCH

$env:UPDATE_GOLDEN_REVISION = "1"
python -m unittest tests.test_batch_golden_corpus.RevisionGoldenCorpusTests -q
Remove-Item Env:UPDATE_GOLDEN_REVISION

$env:UPDATE_GOLDEN_KEYWORD = "1"
python -m unittest tests.test_batch_golden_corpus.KeywordGoldenCorpusTests -q
Remove-Item Env:UPDATE_GOLDEN_KEYWORD
```

CI 会在 Ubuntu 和 Windows 环境下自动运行单元测试，以验证跨平台路径、文件读写及数据格式合约。CI 中的测试仅使用离线 mock，不覆盖真实的 Ren'Py SDK 模板生成和 Gemini 网络请求。
