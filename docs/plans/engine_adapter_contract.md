# Engine Adapter P0：Ren'Py 当前调用链与合同设计

> 状态：#265 / #285 的 P0 设计基线；#265 P1 已按本文合同实现只读
> `RenPyAdapter`、coverage/review 产物，并迁移 sync 与普通 Batch translation
> build 的扫描入口；#265 P2 已实现 Ren'Py relocation、validation、声明式
> writeback plan 及 sync/Batch/revision 的公共 plan 消费；#330 实现 P3 的
> source-only ProjectSnapshot / GameVersion JSON/JSONL 与只读
> reconciliation/freshness；#354 / PR #358 实现 P4 译文复用候选与人工确认。
> 2026-08-24 已在真实 Ren'Py 项目副本上完成 P1–P4 门禁实测。P5–P6 尚未交付。
> 当前实现说明见 [Ren'Py Engine Adapter 与覆盖审计](../engine_adapter.md)。

## 1. 范围与硬性边界

P0 只交付两类内容：

1. 当前 Ren'Py 调用链与模块归属的代码审计；
2. 后续阶段要实现的 protocol、schema、digest、freshness 和接入边界。

以下内容不属于 P0：

- 不创建 adapter 生产骨架，不迁移任何运行时入口；
- 不新增命令、配置、manifest 字段或 GUI 状态；
- 不改变 translation unit identity v2；
- 不改变 Batch `check -> apply`、sync preview apply 或 revision apply；
- 不把 RAG `history.jsonl` 升格为 canonical translation asset；
- 不实现 reconciliation、跨版本 lineage 或第二引擎。

## 2. 当前代码事实

### 2.1 当前没有单一 Ren'Py pipeline

现有共享核心是 [`translation_core.py`](../../translation_core.py)：

- `TranslationUnit`、`ModelResult`、`WritebackAction`；
- sync / Batch / revision / keyword 的 legacy dict 转换；
- prompt、response schema 与 model result normalization。

但引擎语义仍分散在：

- [`translator_runtime.py`](../../translator_runtime.py)：项目/TL 路径、Ren'Py
  字符串扫描、speaker/source marker、格式验证、字面量渲染、sync progress 与
  sync RAG 回灌；
- [`gemini_translate_batch.py`](../../gemini_translate_batch.py)：Batch 文件发现、
  第二套 revision/keyword entry 扫描、identity span 绑定、v2 relocation、
  check/apply、Batch progress 与 Batch RAG 回灌；
- [`project_analysis_routes.py`](../../project_analysis_routes.py)：面向剧情结构的
  `.rpy` label/jump/call/menu 解析。它不是翻译单元扫描器。

因此“Ren'Py scan”目前至少有三种不同目的的扫描：

1. `collect_tasks_with_progress()`：待翻译 task 与已译计数；
2. `scan_all_translation_units()`：identity v2 到 live span 的重定位表；
3. revision、keyword、repair、RAG 的 source/translation pair 扫描；这里存在同名但
   不同语义的两套实现：
   - `translator_runtime.collect_translation_entries_from_lines(lines)` 服务 sync RAG，
     不绑定 identity v2；
   - `gemini_translate_batch.collect_translation_entries_from_lines(
     lines, file_rel_path="")` 服务 revision / Batch RAG，并通过
     `build_identity_v2_by_span()` 绑定 identity v2；
   - `gemini_translate_batch.collect_repair_entries_from_lines(lines)` 在后一套结果上
     追加 `translator_runtime.collect_tasks()` 找到的 pending task。

Project Analysis 另有第四种结构扫描，不能和翻译候选扫描互相证明完整。
两套同名 collector 的签名、source literal 解码、source marker 配对和附加字段并不
完全一致，这是现状债务。P1 的等价性主张只覆盖
`collect_tasks_with_progress()` 与 `scan_all_translation_units()`；不得只包装其中一套
pair collector 就宣称 revision/repair 已统一。

### 2.2 discover / scan / build / relocate / validate / render / check / apply

| 阶段 | 当前入口与调用链 | 当前输出 / 安全含义 |
|---|---|---|
| discover | `gemini_translate.py -> run_translation()` 或 `gemini_translate_batch.py -> dispatch_command()`；两者先经 `load_config() -> load_translator_settings()`，由 `game_root`、`resolve_effective_game_root()`、`tl_subdir` 得到 `BASE_DIR/TL_DIR`。sync 自行 `os.walk(TL_DIR)`；Batch 用 `collect_files_to_process()` 遍历 `.rpy` 并应用 include 过滤。 | 这是“已配置项目 + Ren'Py TL 目录发现”，不是引擎无关的项目探测。`game_ingest.detect_renpy_install_root()` 只服务导入流程，不在翻译调用链中。 |
| scan | sync：`run_translation() -> collect_tasks() -> collect_tasks_with_progress()`；Batch build：`create_batch_package() -> collect_pending_file_jobs() -> collect_tasks_with_progress()`。 | tokenizer/AST heuristic 识别字符串；translation file 中保护 `old`、keyword argument、Character display name 与 voice 行。宽泛解析异常当前会跳过。 |
| build | sync：task -> `process_batch_with_retry() -> process_batch()`；Batch：file jobs -> `build_chunks()` -> `translation_core.units_from_items()` -> request JSONL + manifest v2。 | prompt/result 是公共语义；文件、block、source marker、speaker、span 是 Ren'Py 语义。Batch manifest 仍为 `manifest_version=2`、`core_schema_version=2`。 |
| relocate | `collect_result_actions()` / `collect_revision_actions()` 调用 `relocate_v2_chunk_items()`；后者按文件调用 `scan_all_translation_units()` 并用 item `id` 更新 `line/start/end`。 | 仅 manifest v2 使用 identity relocation。v1 保持旧定位与后续 source validation 行为。sync preview 不做 item relocation，而以 whole-file snapshot/hash 拒绝漂移。 |
| validate | sync：`process_batch()` 后由 `RenPyAdapter.validate_translation()` 重检；Batch/revision：collect 的 source validation 后构建 adapter `ValidationResult`。 | adapter 输出稳定 reason codes；公共 plan consumer 继续负责 source snapshot、项目匹配、路径约束和 check safety。 |
| render | sync preview、Batch apply、revision apply 都先构建 adapter `WritebackPlan`，再由 `engine_adapters.writeback.render_writeback_plan()` 在内存中拼接最终 fragment。 | adapter 负责 Ren'Py prefix/quote/escape；common 不二次渲染，最后仍由现有 `atomic_write_many_lines()` 写入。 |
| check | `check_results() -> require_manifest_project_match() -> collect_result_actions(validate_sources=True) -> attach_check_contract()`。 | `CHECK_CONTRACT_VERSION=2`；fingerprint 绑定 manifest target shape、结果文件、项目 identity 和设置；reason codes 聚合为 `safe/warn/block`。 |
| apply | `apply_results()` 先恢复 transaction、要求最近一次 `safe` check，再重新收集结果、重新读取源文件和二次 source validation；之后 render 全部目标文件并用 `atomic_write_many_lines()` 事务写入。 | `--force` 只放宽“已经 apply”防重 guard，不绕过 stale check、project/source validation 或 `block`。 |
| sync apply | `apply_sync_translation_preview() -> sync_translation_preview.apply_sync_preview()`；先验证 project/TL identity、manifest/report/artifact fingerprint 和每个源文件 hash，再一次性原子写入。 | sync 没有 Batch check 命令，但 preview manifest 自带独立、不可变的 source/proposed snapshot 合同。 |

`resolve_manifest_file_path()`、`require_manifest_project_match()`、check fingerprint、
source re-read、路径 containment、transaction recovery 和 `atomic_write_many_lines()`
均不得下沉到 adapter。

### 2.3 identity、progress 与 RAG 回灌

#### identity v2

`translation_core.build_identity_v2()` 当前生成：

```text
<normalized file_rel_path>:<translate block[#occurrence]>:<block ordinal>:<source sha1 prefix>
```

`scan_all_translation_units()` 使用 source comment / `old` marker 作为 ID 的 source
证据；行号和列号只是 live location。重复 translate block 通过
`block_occurrence` 区分。这一事实由现有 manifest v2 和 relocation 消费，P0/P1
不得改写。

#### progress

- sync 使用 `translator_runtime.PROGRESS_LOG`，entry 为 `task:<line>:<column>` 或兼容
  的 `line:<line>`，只在 sync preview 成功 apply 后更新；
- Batch 使用 `gemini_translate_batch.PROGRESS_LOG`，按 `file_rel_path` 保存已 apply
  的行号；translation apply 与 revision apply 写入后才更新；
- progress 是工作流状态，不是 engine locator、translation identity 或覆盖证明。

adapter 可以在 P1 继续提供现有 `progress_entry` 兼容值，但 progress 的持久化与
“何时算成功”必须留在 workflow/common 层。

#### RAG 回灌

- sync apply callback：`maybe_update_sync_rag_store(full_file=True)` ->
  `translator_runtime.collect_translation_entries_from_lines()` -> segment ->
  embed/upsert，并清理该文件旧的 `file_scan` 记录；
- Batch translation/revision apply：收集已写入文件为 `rag_jobs` ->
  `sync_rag_store_for_jobs()` -> `collect_rag_seed_records_for_jobs()` ->
  `gemini_translate_batch.collect_translation_entries_from_lines()` ->
  embed/upsert；
- Batch build 可由 `prepare_rag_store(..., scan_all_files=True)` 预建派生索引。

当前 `memory_id` 仍绑定路径、行范围和原文，embedding 可重建，部分扫描还会替换或
清理旧记录。因此 RAG 是 adapter 之后的派生 consumer，不是 occurrence、lineage
或译文历史的事实来源。

### 2.4 revision、keyword、Project Analysis 与 Final Review

| 路径 | 当前扫描 / 消费方式 | P0 判断 |
|---|---|---|
| revision | `collect_revision_file_jobs() -> collect_translation_entries_from_lines()`；后者再用 `build_identity_v2_by_span() -> scan_all_translation_units(mode=revision)` 绑定 identity v2。preview/apply 继续走 source validation、adapter plan 和原子写回。 | 已译 occurrence 提取与 translation pending 扫描仍相邻但不同；P2 只收敛 relocation/validation/writeback，不把 keyword/Project Analysis 扩进本阶段。 |
| keyword | `collect_keyword_file_jobs() -> collect_repair_entries_from_lines()`，同时合并 source/translation pair 与 `legacy.collect_tasks()`。输出 glossary candidates，不写 `.rpy`。 | P1 保持原状；以后消费 candidate inventory 的“可分析文本投影”，不能把关键词任务等同翻译覆盖。 |
| Project Analysis | `project_analysis_generate.build_structure_drafts()` 调用 `project_analysis_routes.discover_script_files()` 扫 source `.rpy` 且跳过 `tl/`，再调用 `project_analysis_routes.build_route_graph()` 解析 label/jump/call/menu；content fingerprint 与 lineage 存在自己的 store。 | 保留独立剧情结构 parser。后续只把 fresh coverage digest/review status 作为 publish upstream gate；不把全部 UI 文本注入分析 prompt。 |
| Final Review | `create_final_review_package()` 以 `collect_pending_file_jobs()` 判断 pending，以 `collect_revision_file_jobs()` 取得已译 pairs；`evaluate_readiness()` 当前主要依赖 pending=0 与 review items>0，snapshot/context digest 冻结翻译与上下文。 | P1 保持原状；后续把 coverage dependency 加进 readiness 与 snapshot/upstream digest，不能只凭已识别 pending=0 宣称完成。 |

## 3. 目标模块归属

### 3.1 公共语义

| 语义 | 目标归属 | 说明 |
|---|---|---|
| `TranslationUnit`、`ModelResult`、prompt、response normalization | `translation_core.py` | 继续作为唯一翻译核心模型；不得创建 engine-specific 第二套 unit。 |
| adapter protocol、engine-neutral envelope/schema | 建议新增 `engine_adapters/contracts.py`（P1） | 可以依赖 `translation_core`；`translation_core` 不反向依赖 adapter，避免循环。 |
| candidate/report/review、digest、freshness、review policy | 建议新增 `engine_adapters/coverage.py`（P1） | 公共层生成并校验 artifacts；adapter 只提供候选和分类证据。 |
| project / manifest identity、check fingerprint、最近一次 safe check | common/workflow 层；P2 通过 adapter plan gate 接入现有 Batch/sync/revision 门禁 | adapter 不得决定是否允许 apply。 |
| target path containment、source snapshot re-read、plan 越界/重叠检查 | common writeback safety 层 | adapter locator 对公共层保持 opaque；writeback operation 必须另带公共层可校验的相对 target。 |
| atomic transaction、recovery、failure report、apply state | `atomic_io.py` 与 workflow 层 | adapter 不获得 writer、file handle、callback 或任意路径写权限。 |
| progress、CLI/GUI orchestration、model/provider、RAG/Source Index 回灌 | 现有 workflow/common 模块 | 都是 adapter 的 consumer，不属于引擎解析合同。 |

### 3.2 Ren'Py 专属语义

P1 建议新增 `engine_adapters/renpy.py`，逐步承接：

- Ren'Py project / source root / `game/tl/<language>` catalog 发现；
- `.rpy` candidate inventory 与 tokenizer/AST diagnostics；
- translate block、重复 block occurrence、block ordinal；
- source comment、`old/new`、voice 跳过、speaker id/display name；
- Ren'Py locator 的创建与解释；
- identity v2 兼容映射和 relocation；
- Ren'Py tags、fields、percent placeholders；
- string prefix/quote/escape 与声明式 span replacement plan。

以下内容即使处理 `.rpy` 也不归 `RenPyAdapter`：

- Project Analysis 的路线/剧情结构建模；
- glossary、target-language policy、prompt；
- manifest/check/apply、原子写回；
- RAG/Source Index/Story Memory；
- GUI/CLI 文案与任务状态。

## 4. Engine Adapter protocol v1

### 4.1 不变量

1. adapter 可以读取明确授予的 project roots；不能写文件。
2. adapter 只能返回数据对象，不返回可执行 callback、shell command、打开的 file
   handle 或任意绝对写入路径。
3. `inventory_candidates()` 必须先于 `extract_occurrences()`；不能用已提取 unit 集合
   反推“没有遗漏”。
4. `extract_occurrences()` 只接受已明确分类为 `translatable` 或
   `already_translated` 的 candidate。
5. adapter validation 只提供引擎规则结论；公共 validation/policy 仍可追加更严格的
   `block`。
6. `build_writeback_plan()` 只产生声明式 plan。common 层必须重新读取 source、校验
   project identity、coverage/review freshness、check fingerprint、span、非重叠和
   path containment 后，才可调用原子写入。
7. 未知 operation kind、reason code、schema major 或 locator schema 必须拒绝或只读
   展示，不能 best-effort 写入。

### 4.2 protocol 形状

以下是合同草图，不是 P0 生产 API：

```python
class EngineAdapter(Protocol):
    protocol_version: int
    engine: str
    adapter_version: str

    def capabilities(self) -> EngineCapabilities: ...

    def discover_project(
        self, request: ProjectDiscoveryRequest
    ) -> ProjectDiscovery: ...

    def inventory_candidates(
        self, project: ProjectDiscovery, policy: InventoryPolicy
    ) -> CandidateInventory: ...

    def audit_extraction(
        self, project: ProjectDiscovery, inventory: CandidateInventory
    ) -> CoverageReportDraft: ...

    def extract_occurrences(
        self,
        project: ProjectDiscovery,
        inventory: CandidateInventory,
        approved_candidate_ids: Sequence[str],
    ) -> Sequence[Occurrence]: ...

    def relocate_occurrences(
        self,
        project: ProjectDiscovery,
        occurrences: Sequence[Occurrence],
        live_sources: SourceSnapshotSet,
    ) -> RelocationResult: ...

    def validate_translation(
        self, occurrence: Occurrence, translated_text: str
    ) -> ValidationResult: ...

    def build_writeback_plan(
        self,
        project: ProjectDiscovery,
        validated: Sequence[ValidatedTranslation],
        live_sources: SourceSnapshotSet,
    ) -> WritebackPlan: ...
```

`coverage.py` 负责把 `CoverageReportDraft` 与 inventory 做一对一、不重复、reason code
和 digest 校验，最终生成 report/review package。adapter 自己不能签发 review
confirmation。

## 5. 核心 schema

### 5.1 版本集合

首版建议分别版本化，避免一个全局版本把互不相关的 artifact 一起升级：

| 名称 | 首版 | bump 条件 |
|---|---:|---|
| `engine_adapter_protocol_version` | 1 | 方法语义、必需 capability 或安全不变量改变 |
| `occurrence_schema_version` | 1 | occurrence 必需字段/identity 语义改变 |
| `content_fingerprint_schema_version` | 1 | 规范化或证据组合规则改变 |
| `locator_schema_version` | 每 engine 从 1 开始 | locator 解释或定位优先级改变 |
| `validation_schema_version` | 1 | status/reason/result 语义改变 |
| `writeback_plan_schema_version` | 1 | operation/precondition/plan digest 语义改变 |
| `candidate_schema_version` | 1 | candidate identity/classification 语义改变 |
| `coverage_schema_version` | 1 | report/status/digest 语义改变 |
| `coverage_review_schema_version` | 1 | reviewer/policy/findings/confirmation 语义改变 |
| `coverage_digest_schema_version` | 1 | canonicalization 或 digest payload 改变 |

同一 major 内只允许可选、可忽略的 additive 字段。新增必需字段、改变默认值或改变
digest canonicalization 必须 bump 对应整数版本。reader 对未知更新版本默认拒绝；
migration 必须显式、纯数据、保留原件，不能在 read 时静默覆写。

### 5.2 Occurrence：扩展而不复制 TranslationUnit

`Occurrence` 是现有 `TranslationUnit` 的版本/定位信封，不是第二套翻译模型：

```json
{
  "occurrence_schema_version": 1,
  "occurrence_id": "occ1:sha256...",
  "engine": "renpy",
  "project_snapshot_fingerprint": "sha256...",
  "content_fingerprint_schema_version": 1,
  "content_fingerprint": "sha256...",
  "candidate_id": "cand1:sha256...",
  "locator": {
    "engine": "renpy",
    "locator_schema_version": 1,
    "locator": {}
  },
  "translation_unit": {
    "core_schema_version": 2,
    "id": "script.rpy:chapter_1:42:abcd1234",
    "mode": "translation",
    "text": "..."
  }
}
```

`translation_unit.id` 这里只展示 identity v2 的字段形态，`abcd1234` 是示意值，
不是由示例文本计算出的真实 hash。真实值继续由
`translation_core.build_identity_v2()` 生成。

内存对象应直接持有 `TranslationUnit` 实例；artifact JSON 才使用
`translation_unit` 表示。P1 接入旧调用链时取 `occurrence.unit`，继续交给现有
`translation_core` prompt/result 逻辑。

三种 identity 必须分离：

- `TranslationUnit.id`：继续是现有 identity v2 兼容键，服务 manifest/check/repair/
  RAG fallback；
- `occurrence_id`：某一 project snapshot 内的出现位置，按
  `engine + locator canonical form + project_snapshot_fingerprint` 计算；
- `content_fingerprint`：原文、speaker 和有界局部上下文的匹配证据，不表示逻辑
  相同；
- `lineage_id`：P3 reconciliation 人工确认后的跨版本身份，P0 schema 不生成。

重复原文不得因 `content_fingerprint` 或 source hash 相同而合并。

P0/P1 不向现有 manifest v1/v2 序列化 occurrence 信封，因此 manifest shape 与
`TranslationUnit.id` 不变。未来若 occurrence/locator 成为 manifest 必需字段，必须
单独设计 manifest v3；不能把必需字段偷偷塞进 v2。

### 5.3 Ren'Py opaque locator v1

```json
{
  "engine": "renpy",
  "locator_schema_version": 1,
  "locator": {
    "file_rel_path": "chapter1.rpy",
    "translate_block": "chapter_1_abcd",
    "block_occurrence": 1,
    "ordinal": 42,
    "line_hint": 120,
    "start_col_hint": 8,
    "end_col_hint": 31,
    "source_marker_kind": "comment"
  }
}
```

必需字段为 `file_rel_path`、`translate_block`、`block_occurrence`、`ordinal`。
坐标约定为：

- `line_hint` 是 1-based 文件行号，面向诊断显示并与当前 `line_number` 口径一致；
- `start_col_hint` / `end_col_hint` 是该行 Python 字符串/tokenizer 的 0-based
  字符偏移，使用半开区间 `[start_col_hint, end_col_hint)`，不是 tab 展开后的视觉列；
- 三者都只是 adapter-private relocation hint，不是公共业务坐标。adapter 可以用它们
  缩小重定位候选，但不能只凭 hint 跳过 identity/source 校验。

identity v2 未命中时，Ren'Py content-evidence fallback 仅在同一 `file_rel_path`
内打分；最高分必须唯一且达到 `CONTENT_EVIDENCE_MIN_SCORE`（当前 140，排除仅
“同文件+同原文”的弱匹配 125；典型 stale-block 唯一回落约 140+）。同分并列或
低于最低分均返回 `common.locator.unresolved`，不得 fail-open 写回。

`source_marker_kind` 为 `comment`、`old_new` 或 `direct_source`。公共层只做 JSON
schema/version/size 检查，把 locator 原样交回同一 engine adapter；不得读取或校验
Ren'Py hint 字段作业务判断。

公共 writeback safety 需要的 `target_rel_path`、source hash 和 span 由
`WritebackPlan.operations` 单独暴露，不能通过偷看 locator 获得。

### 5.4 ValidationResult v1

```json
{
  "validation_schema_version": 1,
  "occurrence_id": "occ1:...",
  "engine": "renpy",
  "status": "pass",
  "reason_codes": [],
  "diagnostics": [],
  "source_constraints_digest": "sha256...",
  "translation_digest": "sha256...",
  "normalized_translation": null
}
```

`status` 只有 `pass|warn|block`。message 只供展示，机器语义只来自稳定 reason
code。common 层把 adapter result 与公共 policy result 合并，严重度取最大值。

首版 reason code namespaces：

- 公共：`common.translation.empty`、`common.preserve_term.missing`、
  `common.target_language.missing`、`common.locator.unresolved`；
- Ren'Py：`renpy.placeholder.missing`、`renpy.placeholder.added`、
  `renpy.tag.changed`、`renpy.field.changed`、`renpy.percent_token.changed`、
  `renpy.string_literal.unrenderable`；
- writeback preflight：`writeback.project_mismatch`、
  `writeback.source_snapshot_mismatch`、`writeback.span_mismatch`、
  `writeback.path_escape`、`writeback.overlap`、`writeback.plan_stale`。

当前 `translator_runtime.validate_translation()` 的 `(bool, message)` 到 v1 合同的
迁移草表如下；这是 P2 显式映射输入，不改变 P0 的返回值或用户文案：

| 当前 message 模式 | v1 reason code | 默认 status |
|---|---|---|
| `OK` | 无 reason code | `pass` |
| `Empty translation` | `common.translation.empty` | `block` |
| `Preserved terms missing: ...` | `common.preserve_term.missing` | `block` |
| `Ren'Py placeholders/tags changed: ...` | 按 token 差异拆为 `renpy.placeholder.missing` / `renpy.placeholder.added` / `renpy.tag.changed` / `renpy.field.changed` / `renpy.percent_token.changed` | `block` |
| `No Chinese characters` | `common.target_language.missing` | `block` |

现有 Batch `CHECK_WARN_REASON_CODES` / `CHECK_BLOCK_REASON_CODES` 保持原样；上述
adapter reason codes 在 P2 需要显式映射，不能通过错误字符串推断。

### 5.5 WritebackPlan v1

```json
{
  "writeback_plan_schema_version": 1,
  "engine": "renpy",
  "adapter_version": "x.y.z",
  "project_identity_digest": "sha256...",
  "source_snapshot_fingerprint": "sha256...",
  "coverage_digest": "sha256...",
  "coverage_review_digest": "sha256...",
  "operations": [
    {
      "operation_id": "op1:...",
      "kind": "text_span_replace",
      "occurrence_id": "occ1:...",
      "target_root": "localization_catalog",
      "target_rel_path": "chapter1.rpy",
      "expected_file_sha256": "sha256...",
      "line": 119,
      "start_col": 8,
      "end_col": 31,
      "expected_fragment_sha256": "sha256...",
      "expected_text_digest": "sha256...",
      "replacement_fragment": "\"译文\"",
      "validation_digest": "sha256..."
    }
  ],
  "plan_digest": "sha256..."
}
```

operation 坐标与 locator hint 不是同一权威层：

- `line` 是 0-based 文件行索引，与当前 `WritebackAction` / relocation 输出一致；
- `start_col` / `end_col` 是该行 Python 字符串/tokenizer 的 0-based 字符偏移，
  使用半开区间 `[start_col, end_col)`；
- common 层只验证 operation 上的 `line/start_col/end_col`、hash 和重叠关系，
  不得把 locator 的 `line_hint/*_col_hint` 当作 writeback span。

`replacement_fragment` 已是包含字符串 prefix、quote 与 escape 的最终 Ren'Py
fragment。adapter 负责按 Ren'Py renderer 语义生成它；common 层只做 span/hash/
validation 校验并拼接，不得再次调用 `render_replacement_lines()`、`quote_with()` 或
其他 engine renderer 二次渲染。

v1 只允许 common 层实现并注册的 operation kinds。Ren'Py 首个 kind 为
`text_span_replace`。plan：

- 不含绝对目标路径；
- 不含删除目录、重命名、shell、Python 表达式或 callback；
- 每个 operation 绑定 live file hash、span、expected fragment 和 validation；
- operation 按 `target_rel_path/line/start_col` canonical sort 后计算 plan digest；
- common 层拒绝重叠 span、重复 operation、未知 root/kind 和路径逃逸；
- common 层在第一笔写入前再次读取全部文件并复核；
- 全部通过后才生成 rendered files 并交给 `atomic_write_many_lines()`。

- `expected_fragment_sha256` 是 common consumer 对 live raw span 的校验；
- `expected_text_digest` 绑定 adapter 已解码的源文本与 operation/plan payload，
  但 common 层保持 engine-neutral，不重新解析 Ren'Py 字符串 literal；adapter
  构建 plan 时必须从同一已验证 occurrence 生成该 digest。

`apply --force` 不得跳过 plan freshness、source snapshot、coverage gate、`block`
validation 或公共安全检查。

### 5.6 LocalizationMode 与 native catalog freshness

`LocalizationMode` 是稳定 enum：

- `source_extraction`：从源脚本枚举并生成 occurrence；
- `native_catalog`：优先消费引擎官方 localization artifact；
- `hybrid`：源脚本负责 inventory/coverage，native catalog 负责稳定映射与译文。

capabilities 至少记录：

```json
{
  "engine_adapter_protocol_version": 1,
  "engine": "renpy",
  "adapter_version": "x.y.z",
  "supported_localization_modes": ["hybrid"],
  "selected_localization_mode": "hybrid",
  "source_inventory": true,
  "native_catalog": true,
  "relocation": true,
  "declarative_writeback": ["text_span_replace"],
  "native_catalog_required_for_writeback": true
}
```

`CatalogProvenance` 至少包含 catalog format/path digest、target language、
generator/tool、engine version、generation time、generation command digest、
recorded source fingerprint、live source fingerprint 与
`provenance_status=verified|inferred|missing`。

freshness 为 `fresh|stale|unknown|missing`：

- recorded source fingerprint 与 live source fingerprint 相等且 provenance 可验证：
  `fresh`；
- 明确不等：`stale`；
- artifact 存在但没有足够 provenance：`unknown`；
- artifact 不存在：`missing`。

Ren'Py 目标模式为 `hybrid`。fallback 规则：

1. source inventory/audit 可在 catalog missing/stale 时继续，以便报告缺口；
2. `unknown` provenance 至少产生 `attention`，不得宣称 catalog 完整；
3. `stale` 或 `missing` 且 adapter 声明 catalog 是 writeback 必需项时，translation
   build/writeback gate 为 `block`，建议重新运行 Ren'Py 官方模板生成；
4. 不允许退化为直接改写 source `.rpy`；
5. P1 为保持零用户行为回归，只产生只读 report，不启用上述新下游 gate。

## 6. Candidate inventory 与 Coverage 合同

### 6.1 固定流程

```text
discover
-> candidate inventory
-> automated extraction audit
-> coverage report + review package
-> independent human/agent review against raw scripts
-> coverage confirmation
-> extract TranslationUnit/Occurrence
-> downstream task
```

自动 audit 与独立 review 是两个 artifact。reviewer 必须能查看 raw scripts、文件
清单和 inventory；只看 adapter 输出、只检查 `unknown` 或让同一 extractor
“自证完整”均不满足合同。

### 6.2 `coverage_candidates.jsonl`

每行至少包含：

```json
{
  "candidate_schema_version": 1,
  "candidate_id": "cand1:sha256...",
  "engine": "renpy",
  "adapter_version": "x.y.z",
  "source_fingerprint": "sha256...",
  "locator": {},
  "raw_excerpt": "...",
  "structure_kind": "say_string",
  "classification": "translatable",
  "reason_codes": ["renpy.dialogue_string"],
  "translation_scope": "include",
  "analysis_scope": "include",
  "catalog_link": null,
  "evidence": {}
}
```

每个 candidate 必须恰好进入一个 classification：

- `translatable`
- `already_translated`
- `explicitly_excluded`
- `unsupported`
- `parse_error`
- `unknown`

`candidate_id` 绑定 project snapshot 与 locator；parse failure 也必须有稳定的
file/span/error-region candidate。一个位置重复出现、零分类、多分类或未知 reason
code 都使 report `block`。

translation 与 Project Analysis 是两个正交维度：

- `classification` 表示本地化提取结论；
- `translation_scope=include|exclude|unknown` 表示是否进入翻译覆盖；
- `analysis_scope=include|exclude|unknown` 表示是否进入剧情分析投影。

玩家可见 UI 可以 `translation_scope=include`、`analysis_scope=exclude`。Project
Analysis 只消费 analysis projection digest，不把所有 UI 文本注入 prompt。

### 6.3 reason code registry v1

候选分类首版保留以下稳定 codes：

- 识别：`renpy.dialogue_string`、`renpy.narration_string`、
  `renpy.translate_comment_pair`、`renpy.old_new_pair`、
  `renpy.catalog.translation_present`；
- 排除：`renpy.character_display_definition`、
  `renpy.keyword_argument`、`renpy.voice_asset`、`renpy.asset_path`、
  `renpy.non_player_visible_literal`、`project.explicit_exclusion`；
- 不支持/未知：`renpy.dynamic_string_expression`、
  `renpy.custom_statement_unsupported`、`renpy.visibility_unknown`；
- 解析：`renpy.tokenize_error`、`renpy.ast_parse_error`、
  `renpy.source_marker_unpaired`；
- catalog：`renpy.catalog.missing_entry`、`renpy.catalog.duplicate_entry`、
  `renpy.catalog.provenance_unknown`、`renpy.catalog.stale`；
- structured override：`project.extraction_override`。

review findings 使用独立 codes：

- `review.missed_candidate`
- `review.false_positive`
- `review.wrong_classification`
- `review.duplicate_candidate`
- `review.invalid_exclusion`

code 与中文/英文 message 分离。新增 code 只能 additive；改变 code 含义需要 bump
相关 schema/rules digest。

### 6.4 `coverage_report.json`

至少记录：

- `coverage_schema_version`、`coverage_digest_schema_version`；
- engine、protocol version、adapter version、adapter behavior digest；
- localization mode 与 catalog provenance/freshness；
- source fingerprint、inventory digest、classification rules digest、
  extraction overrides digest；
- automated audit reason codes 与 source-changed-during-scan 证据；
- files scanned、candidate count、六类 classification counts；
- translation/analysis scope counts；
- coverage status、reason counts、coverage digest、generated_at。

status 规则：

- `ready`：inventory invariant 成立，无 `unknown/parse_error/unsupported`，无 catalog
  或规则警告；
- `attention`：无 `unknown/parse_error`，但存在明确的 `unsupported`、需要重点复核的
  exclusion、弱 catalog provenance 等；可以进入独立 review；
- `block`：存在 `unknown/parse_error`、inventory invariant 失败、必需 catalog
  missing/corrupt，或 source 在扫描过程中变化；
- `stale`：已保存 report 的任一 freshness input 与 live input 不匹配。

“ready”只证明 adapter 声明范围中的候选均被分类；不表示数学意义上的 100% 文本
发现。

### 6.5 `coverage_review.md` 与 review record

`coverage_review.md` 是只读 review package，至少展示：

- source file manifest 与 hash；
- candidate 原文/上下文/locator/classification/reason；
- exclusions、unsupported、parse errors、unknown 的完整列表；
- 每种已支持结构的抽样清单；
- raw-script 对照步骤和 findings 填写说明。

机器可读 review record 至少包含：

```json
{
  "coverage_review_schema_version": 1,
  "source_fingerprint": "sha256...",
  "coverage_digest": "sha256...",
  "review_input_digest": "sha256...",
  "review_policy": "agent_or_human",
  "reviewer": {
    "type": "agent",
    "id": "tool-or-person-id",
    "tool": "codex",
    "model": "...",
    "session": "..."
  },
  "status": "agent_reviewed",
  "findings": [],
  "confirmed_at": "..."
}
```

`reviewer.type` 只有 `agent|human`。Agent 不得写成 human。status：

- `pending`
- `agent_reviewed`
- `human_reviewed`
- `changes_requested`
- `stale`

review policy：

- `agent_or_human`：fresh 的 `agent_reviewed` 或 `human_reviewed` 均可；
- `human_required`：只有 fresh `human_reviewed` 可通过。

人工结论可以 supersede Agent 结论，但旧 review 和 provenance 必须保留。review
不能直接覆写 report/candidate。发现漏项后必须：

1. 修 parser/adapter；或
2. 添加有 provenance 的结构化 project extraction override；或
3. 明确、可审计地排除；

然后重新执行 inventory -> audit -> review。禁止只在 review 文件手工补一条文本后
继续翻译。

## 7. Digest 与 freshness

### 7.1 canonicalization

所有 digest 使用 UTF-8、JSON object key 排序、紧凑分隔符和 `/` 相对路径。绝对
workspace 路径、时间戳、展示 message、reviewer display name 不进入稳定 digest。

source fingerprint 按相对路径排序，hash 原始文件 bytes：

```text
sha256(canonical_json([
  {"file_rel_path": "...", "size": N, "sha256": "..."},
  ...
]))
```

不得先做空白/换行文本规范化，否则 encoding 或控制字符变化可能逃过 freshness。

### 7.2 digest 组合

```text
inventory_digest = sha256(canonical candidates without timestamps/messages)

coverage_digest = sha256(canonical {
  coverage_digest_schema_version,
  engine,
  protocol_version,
  adapter_version,
  adapter_behavior_digest,
  localization_mode,
  source_fingerprint,
  catalog_digest_and_provenance,
  audit {
    reason_codes,
    source_changed_during_scan
  },
  candidate_schema_version,
  coverage_schema_version,
  inventory_digest,
  classification_rules_digest,
  extraction_overrides_digest
})

review_input_digest = sha256(canonical {
  coverage_review_schema_version,
  coverage_digest,
  inventory_digest,
  review_policy,
  review_package_template_version,
  sampling_plan
})
```

review record 本身另算 `coverage_review_digest`，包含 reviewer provenance、
recorded status、findings 与 resolution，不含展示 message。

### 7.3 stale 条件

以下任一变化使旧 report/review stale：

- source files/path set/content；
- native catalog content/provenance/freshness；
- engine、adapter version/behavior；
- protocol/candidate/coverage/review/digest schema；
- classification rules 或 project extraction overrides；
- coverage digest、review policy、review template/sampling plan；
- review 所绑定的 source/coverage/review input digest。

`stale` 应作为 live 计算的 effective status；原记录的 reviewer 结论和 provenance
仍保留，不能覆写成好像从未审过。

下游“文本范围已确认”门禁固定为：

```text
coverage_status in {ready, attention}
AND review_status satisfies review_policy
AND review.coverage_digest == live coverage_digest
AND review.review_input_digest == live review_input_digest
AND unresolved_findings == 0
```

## 8. 分阶段接入边界

| 路径 | P1 首阶段 | 后续边界 |
|---|---|---|
| sync translation preview build | 经 `RenPyAdapter` 的 discover/inventory/audit/extract 取得与当前完全等价的 units；保持 CLI/GUI/preview artifact 不变。 | P2 preview manifest 保存单文件 plan；apply 在首笔写入前用 source artifact 重建并重检 plan，project/source/atomic apply 仍由 common 层执行。 |
| Batch translation build | 与 sync 共用同一 adapter extraction；legacy item/manifest v2 shape、chunk/prompt/RAG 行为不变。 | P2 check 消费 adapter validation/plan；最近一次 safe check、fingerprint 与 apply re-read 保持公共层。 |
| revision | 保持 `collect_revision_file_jobs()` 的 pair 扫描边界，不把 Project Analysis/keyword 迁移偷带进来。 | P2 preview/apply 均在 source revalidation 后消费 adapter plan；不安全时拒绝写回。 |
| keyword | P1 保持现有 repair-entry scan 与输出。 | adapter inventory 稳定后再消费 analysis projection；仍只输出 glossary candidates。 |
| Project Analysis | P1 保持独立 route parser 和现有 publish 行为，不消费全部 occurrence。 | 后续将 fresh coverage/review digest 加入 publish gate 与 lineage upstream dependency；只消费 analysis projection digest。 |
| Final Review | P1 保持当前 pending/revision scan、readiness 和 snapshot shape。 | 后续 readiness 必须验证 coverage gate；coverage/review digest 纳入 snapshot/upstream dependency。 |
| progress | P1/P2 保持现有两个 progress logs 与 apply 后更新时机。 | 若未来统一，另开 issue；不能把 progress 当 occurrence identity。 |
| RAG / Source Index 回灌 | P1 保持现有 scan/upsert/prune 调用点。 | adapter extraction 稳定后可复用 occurrence 读取，但仍是 apply 后派生索引，不成为 canonical store。 |

P1 的最小切入顺序：

1. 增加 neutral contracts 与只读 Ren'Py adapter；
2. 用 characterization/golden tests 证明 adapter wrapper 与现有
   `collect_tasks_with_progress()` / `scan_all_translation_units()` 等价；
3. 先让 sync 与 Batch translation build 从同一 adapter 取得 units；
4. 导出只读 inventory/report/review package；
5. 不接 revision/writeback，不启用 coverage 下游 gate，不改 manifest。

## 9. 兼容策略与剩余风险

### 9.1 兼容策略

- manifest v1：保持当前无 v2 relocation、靠原定位 + source validation 的行为；
- manifest v2 / identity v2：ID、target shape、check fingerprint 输入不变；
- sync preview schema/version：P0/P1 不变；
- existing CLI/GUI/config：P0/P1 不新增 required input，不改变 stdout 文案；
- `TranslationUnit.metadata` 目前不是可靠的 manifest round-trip 通道，P1 不应依赖
  它偷偷持久化 locator；
- `WritebackAction` 在 P2 可作为 `text_span_replace` plan 的兼容输入，但 adapter
  plan 不能直接调用现有 writer。

### 9.2 P1 前必须正视的风险

1. 当前 scanner 的宽泛 `except Exception: continue` 会静默丢失 tokenizer/AST
   失败；P1 inventory 必须报告，但 extracted unit 集合仍需先保持等价。
2. pending scan、relocation scan、revision/repair scan 对 source marker、voice、
   keyword argument 的规则并非完全同一实现；P0 characterization 固定这些已有分支，
   但 P1 不能只包装其中一个就宣称统一。
3. Project Analysis 扫 source tree，翻译扫 TL catalog；两者 source roots 与
   fingerprint 口径不同，coverage dependency 必须显式桥接。
4. native Ren'Py catalog 当前没有统一记录 generation provenance；首次只能标记
   `inferred/unknown`，不能伪造 `fresh`。
5. existing `TranslationUnit.id` 不是跨游戏版本 lineage；P3 之前不得把
   occurrence/content fingerprint 当成自动复用确认。
6. current RAG history 会按派生扫描更新，不能承担旧版本译文保存与审计。

## 10. P0 characterization tests

[`tests/test_engine_adapter_p0_characterization.py`](../../tests/test_engine_adapter_p0_characterization.py)
直接固定：

- source marker、speaker、translated count 与重复 translate block 的扫描输出；
- identity v2 在行漂移后的稳定性和 duplicate block occurrence 区分；
- `relocate_v2_chunk_items()` 按 live identity 更新 span，并报告 unresolved item；
- `render_replacement_lines()` 的多 replacement 逆序应用、quote escaping 和输入不变；
- runtime / Batch 同名 pair collector 对 voice、dangling source marker、source
  literal 解码、speaker/identity 附加字段的当前分歧；
- repair collector 追加 direct pending task，同时跳过 voice statement 与 keyword
  argument 字符串的当前行为。

这些测试描述当前行为，不授权 P0 修改生产实现。P1 adapter 必须先在这些结果上做到
等价，再单独增加 candidate/error reporting。这里的“等价”对 pair collector 指
不得意外改变已固定分支，不表示两套 collector 已经统一。
