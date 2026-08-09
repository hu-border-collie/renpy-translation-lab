# Ren'Py Engine Adapter 与覆盖审计

文档地图：[docs/README.md](README.md)

当前已交付 #265 的 P1 与 P2，并由 #330 实现 P3 的只读项目版本快照与
reconciliation：同步翻译预览、普通 Batch translation build，以及 revision
preview/apply 通过 `RenPyAdapter` 的 relocation、validation 和声明式 writeback plan；
公共层重新校验 plan 后，仍由既有 workflow 执行 check→apply、路径约束、事务恢复和
atomic write。P3 在这条写回路径之外保存 source-only occurrence 证据并比较两个版本，
不会复用译文、确认 lineage 或修改游戏文件。

## 当前边界

| 层 | 当前职责 |
|---|---|
| `engine_adapters/contracts.py` | engine-neutral protocol、capability、candidate、occurrence、validation / writeback 的版本化信封 |
| `engine_adapters/renpy.py` | Ren'Py 项目发现、`.rpy` inventory、分类、source marker、speaker、translate block / occurrence / ordinal、只读 occurrence 提取 |
| `engine_adapters/coverage.py` | 独立校验 inventory invariant、生成稳定 digest、导出 coverage/review package、导入并校验人工或 Agent review |
| `engine_adapters/writeback.py` | 公共 plan schema、source snapshot、相对路径、文件 hash、span、重叠和 plan digest 校验；只在内存中渲染，不持有 writer |
| `engine_adapters/versioning.py` | P3 `GameVersion` / `ProjectSnapshot` / source-only `UnitOccurrenceRecord`、JSON/JSONL 导入导出、只读 reconciliation 与 freshness 校验 |
| `translation_core.py` | 唯一的 `TranslationUnit` / `ModelResult` 核心模型；adapter 不创建第二套翻译单元 |
| sync / Batch / revision workflow | 模型调用、prompt、progress、manifest、preview/check/apply、RAG / Source Index 回灌；atomic writer 仍在 workflow/common 层 |

P2 的 `relocate_occurrences()` 先按 identity v2，再在同一 localization 文件内按
source/context evidence 做唯一重定位；content evidence 还需达到最低结构分
（`CONTENT_EVIDENCE_MIN_SCORE`，默认 140，排除仅“同文件+同原文”的 125 分弱匹配），
同分并列时仍返回 `common.locator.unresolved`。`validate_translation()`
输出版本化 `ValidationResult`，`build_writeback_plan()` 只产生
`text_span_replace` 操作。公共消费者会在 check 和 apply 的二次源重读后再次校验
source snapshot、文件 hash、半开 span、非重叠、相对路径和 plan digest；adapter 没有
文件写入权限。keyword、Project Analysis 与 Final Review 的独立扫描入口不在本阶段
扩大范围。

## 扫描与等价性

一次 translation build 只建立一个不可变扫描快照：

1. `discover_project()` 按现有 include allowlist 读取目标语言目录下的 `.rpy`；
2. `inventory_candidates()` 独立枚举字符串、source marker 和解析错误区域；
3. 旧 `collect_tasks_with_progress()` 与 `scan_all_translation_units()` 作为 P1
   等价性来源，保留 task 集合、identity v2、speaker、source 和 span；
4. 公共 coverage 层校验每个 candidate 恰有一个分类；
5. 只有 `translatable` / `already_translated` candidate 可以变成
   `Occurrence[TranslationUnit]`。

合法但当前不支持的动态字符串进入 `unsupported`；未配对 source marker、字符串
tokenize 失败、AST/literal 解析失败进入可定位的 `parse_error`。这些项目不会再因为
旧扫描器的宽泛异常处理而被当作“没有文本”。

## Coverage 产物

同步预览写入：

```text
logs/sync_runs/<run>/coverage/
```

普通 Batch translation build 写入：

```text
logs/batch_jobs/<package>/coverage/
```

目录内包含：

| 文件 | 含义 |
|---|---|
| `coverage_candidates.jsonl` | 全部 candidate、opaque locator、分类、scope、reason code 与证据 |
| `coverage_report.json` | source / adapter / rules digest、分类计数、catalog provenance 与自动状态 |
| `coverage_review.md` | 供人工或 Agent 对照原脚本检查的只读清单 |
| `coverage_review_template.json` | 结构化 review 输入模板；自动报告不会把自己标记为已核对 |

这些文件不加入 manifest v2，也不改变现有 stdout/JSON 命令合同。adapter 本身没有
文件写入 API；产物由公共 coverage 层写到 workflow 已创建的日志包中，不会修改
`.rpy`。

自动状态含义：

- `ready`：inventory invariant 成立，且没有未知、解析失败、不支持或 provenance
  警告；
- `attention`：可以进入独立核对，但存在 unsupported、弱 catalog provenance 等；
- `block`：存在 `unknown` / `parse_error`、source 扫描中变化或 inventory invariant
  失败；
- `stale`：保存的 report/review 与当前 source、adapter、规则或 coverage digest
  不再一致。

Ren'Py P1 只能从现有 TL 脚本推断 catalog provenance，因此自动报告通常为
`attention`；P1 不把该状态接入新的 translation build/apply gate，以保持行为兼容。

## Review provenance

`coverage_review_template.json` 必须由核对者另存或填写：

- `reviewer.type` 只能是 `agent` 或 `human`；
- 已完成记录必须提供 reviewer ID、确认时间；Agent 还必须记录 tool 或 model；
- `agent_reviewed` 不能伪装为 `human_reviewed`；
- findings 使用稳定 code，并显式记录是否解决；
- `human_required` 策略不能由 Agent review 满足；
- source、adapter、rules 或 coverage digest 变化后，旧 review 校验结果为 `stale`。

发现漏项时应修 adapter/parser 或添加后续定义的结构化 extraction override，然后重新
执行 inventory → audit → review；不能只把漏掉的文本手工塞进 review 文件继续翻译。

## P3 项目版本快照

P3 提供两个离线命令：

```powershell
python gemini_translate_batch.py export-project-snapshot `
  --version-id 1.4.0 `
  --source-revision game-build-140 `
  --output-dir logs/project_snapshots/game-1.4.0

python gemini_translate_batch.py reconcile-project-snapshots `
  logs/project_snapshots/game-1.3.0/project_snapshot.json `
  logs/project_snapshots/game-1.4.0/project_snapshot.json `
  --output-dir logs/project_reconciliations/1.3.0-to-1.4.0
```

`export-project-snapshot` 复用一次完整 adapter discovery / inventory / coverage /
occurrence 扫描，输出：

- `project_snapshot.json`：`GameVersion`、engine/adapter/schema、source files、coverage
  digest、review digest/status、分类计数、snapshot digest；
- `unit_occurrences.jsonl`：opaque locator、source text、speaker、前后 occurrence
  context 和 content fingerprint。该文件不保存 `current_translation`，不是 P4 的译文
  历史存储。

可用 `--coverage-review <FILE>` 冻结已经完成并通过现有合同校验的人工 / Agent review；
未提供时，快照会明确冻结 `pending` review，而不是伪造覆盖确认。相同稳定输入与
`version-id` 产生相同 snapshot digest；`generated_at` 不参与 digest。

`reconcile-project-snapshots` 只读取两个已保存的快照，按以下证据优先级产生一对一
候选：

1. 已确认 lineage；
2. opaque locator 完全一致；
3. content fingerprint 完全一致；
4. 原文一致的移动项，以及可由 speaker / 区分性上下文唯一支持的重复原文；
5. 超过固定相似度与唯一性门槛的原文小改。

重复原文没有独立证据、多个候选同分或同一目标被竞争时会输出 `ambiguous`，不会按
原文哈希或文件顺序静默合并。报告另外列出 `added`、`deleted`、coverage 分类变化和
新增 unresolved 结构。`reconciliation_report.json` 保存摘要与输入 digest，
`reconciliation_items.jsonl` 保存逐项证据、置信度和候选来源。

快照、occurrence、reconciliation item 和报告均有独立 schema/digest 校验；JSONL
被修改、路径逃出 artifact 目录或数量/identity 不一致时导入失败。旧报告通过
`validate_reconciliation_freshness()` 对照当前两个 snapshot digest 与 coverage/review
dependency digest；任一依赖改变时状态为 `stale`。

P3 是按 #265 分阶段交付的高级 CLI 能力；GUI 当前只在「诊断与运行日志 → 命令参考」
提供命令模板。快照浏览、版本 diff、歧义处理和复用候选交互属于 P6。P4 之前任何
reconciliation 结论都不能直接进入 preview/check/apply。

完整 schema、P2 安全边界与后续阶段见
[Engine Adapter 合同设计](plans/engine_adapter_contract.md)。
