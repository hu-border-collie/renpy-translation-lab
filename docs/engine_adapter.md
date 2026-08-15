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

重复原文没有独立证据、多个候选同分或同一目标被竞争时会输出 base 侧
`ambiguous`，不会按原文哈希或文件顺序静默合并。每个未决 target 另有一条
`ambiguous_target` item；base/target item 通过稳定 ambiguity group ID 相连，因此
即使 base item 只保留 8 个候选样本，大歧义组中的每个目标仍可逐项追溯。报告另外
列出 `added`、`deleted`、coverage 分类变化和新增 unresolved 结构。
`reconciliation_report.json` 保存摘要与输入 digest，`reconciliation_items.jsonl`
保存逐项证据、置信度和候选来源。

快照、occurrence、reconciliation item 和报告均有独立 schema/digest 校验；枚举、
布尔值以及 manifest digest 列表与 JSONL 的逐项关系也会严格核对。JSONL 被修改、
路径逃出 artifact 目录或数量/identity 不一致时导入失败。旧报告通过
`validate_reconciliation_freshness()` 对照当前两个 snapshot digest 与 coverage/review
dependency digest；任一依赖改变时状态为 `stale`。

P3 是按 #265 分阶段交付的高级 CLI 能力；GUI 当前只在「诊断与运行日志 → 命令参考」
提供命令模板。快照浏览、版本 diff、歧义处理和复用候选交互属于 P6。P3 报告本身
没有任何写回入口；跨版本复用必须经过下面的 P4 流程。

## P4 译文复用候选与人工确认

P4 在 P3 只读 reconciliation 之上增加四个离线命令，形成
「冻结译文 → 生成候选 → 人工决策 → 结果导出」流程：

```powershell
python gemini_translate_batch.py build-translation-records <base-snapshot> <base-manifest> --output-dir <records-dir>
python gemini_translate_batch.py build-reuse-candidates <base-snapshot> <target-snapshot> <reconciliation> <base-records> --output-dir <reuse-dir>
python gemini_translate_batch.py import-reuse-decisions <reuse-report> <decisions-jsonl>
python gemini_translate_batch.py export-reuse-results <decided-reuse-report> <target-manifest>
```

- `build-translation-records` 只接受 translation 模式且已下载完成的 Batch 包；
  每个 unit 的译文经过现有响应合同校验后，连同来源（默认 `model_initial`）、
  provenance 和快照 occurrence 绑定写入 `translation_records.jsonl`。
- `build-reuse-candidates` 消费保存的 P3 快照、reconciliation 报告和译文记录，
  生成 `reuse_candidates.jsonl`、机器可读 `reuse_report.json`、人工 / Agent
  审核表 `reuse_review.md` 与决策模板 `reuse_decisions_template.jsonl`。
  候选分类固定为 `exact_reuse` / `moved_reuse` / `context_match` /
  `source_modified_reference` / `ambiguous`；locator 或 lineage 匹配但原文
  已变化的项一律降级为 reference-only，不会按位置身份直接复用旧译文。
- `import-reuse-decisions` 导入逐候选决策（accept / reject /
  override_translation / split_lineage / merge_lineage），必须显式填写
  `reviewer.type`（human / agent）与 `reviewer.name`；agent 决策不会伪装成人工。
  歧义候选的 accept 必须从候选目标中显式选定一个；同一目标被两个已接受候选
  竞争时导入失败。所有决策追加进候选 audit 记录并生成新 package。
- `split_lineage` / `merge_lineage` 按设计只写入候选包的 `lineage_decisions`
  审计记录，不会自动改写候选匹配或任何快照：跨版本 lineage 是 P3 快照 +
  reconciliation 的输入属性，人工拆分/合并结论需要导出后更新项目 lineage
  映射并重新导出快照 / reconciliation，再重建候选才会生效。P4 不提供跳过
  这条链路的暗道。
- `export-reuse-results` 是唯一的写回入口，而且只写 Batch 包内的
  `results.reuse_<time>.jsonl` 与 manifest 簿记：只有「已接受 + 输入 digest 全部
  fresh + 非 reference-only + 两版原文仍一致」的候选才会进入结果行，未覆盖的
  unit 会直接报错而不是静默跳过。之后必须照常运行 `check` 再 `apply`；
  复用结果不提供任何绕过安全层的捷径。

译文记录、候选、决策和报告均有独立 schema/digest 校验。任一输入
（reconciliation digest、两版 snapshot digest、译文记录 digest 或候选内记录
digest）变化时，`validate_reuse_freshness()` 会把候选标记为 `stale`，
决策导入与结果导出都会拒绝继续。高置信匹配同样以 pending 候选开始，没有免审
通道；reference-only 候选即使接受也只保留旧译文供参考，不会计入可写回的复用。

P4 不新增 GUI 界面；GUI 只在诊断命令参考提供模板（完整交互属 P6）。

完整 schema、P2 安全边界与后续阶段见
[Engine Adapter 合同设计](plans/engine_adapter_contract.md)。
