# #427 普通译文逐条审校索引与订正页合同（S1 核心 / CLI）

> **状态**：S1 只读索引 + 人工决定日志 + CLI 已实现；S2 GUI 订正页待做。
> **关联**：#427、#318、#320、#321、#348、#362、#363。
> **非目标**：不新增 SQLite / 第二翻译库；不重做质量或复用算法；不绕过
> `preview-revisions` / `apply-revisions`；不把 `ignored` 当作质量确认或复用接受；
> 不自动把所有条目改成「人工确认」。

## 1. 目标与数据流

普通译文（非 finding 专用条目）也需要逐条浏览、筛选、查看证据并形成可恢复的人工决定。
S1 建立可重建的派生索引，不改变现有写回授权：

```text
export-revision-corpus（#320）
        └─ revision_corpus.jsonl + manifest ─┐
check / preview 的 quality_findings.jsonl ──┼─> review-index-build
build-translation-records（可选 provenance）┘        │
                                                     ├─ review_index.jsonl        派生索引，可删除重建
                                                     ├─ review_index_manifest.json 输入/范围/诊断
                                                     └─ review_decisions_template.jsonl
review-decisions-import ──> review_decisions.jsonl   人工决定日志，与索引分离持久化
```

- `review_index.jsonl` 是**派生缓存**：删除后用同一 corpus / findings / decisions 重建必须得到相同
  `entry_id` 与 review 状态；它不是 source of truth，也不参与写回门禁。
- `review_decisions.jsonl` 是人工决定日志；索引重建只读取它，不覆盖它。
- 质量 finding 仍由 #363 / `quality_findings.jsonl` 提供，本模块只按稳定身份附着展示，
  不改变 finding 的 disposition 或 #362 的 quality acknowledgement。

## 2. 索引条目字段

每个 corpus occurrence 一条：

| 字段 | 说明 |
|---|---|
| `entry_id` | 由 project_id + occurrence_id + file/line + source/target/context/evidence digest 派生；内容变化会生成新 entry_id |
| `project` | `slug`、`tl_subdir`、`identity_digest`；与 #348 项目/任务身份约定对齐，不引入绝对路径 |
| `occurrence_id` / `identity_v2` | 复用 #320 的 occurrence 身份；重复原文不合并 |
| `file_rel_path` / `locator` / `display_line` / `speaker_id` | 定位与展示 |
| `source` / `current_translation` / `context` | 原文、当前译文与同文件前后上下文 |
| `snapshot_digest` / `binding` | `entry_id`、`snapshot_digest`、`source_digest`、`target_digest`、`context_digest`、`evidence_digest`；决定绑定用 |
| `quality_findings` / `quality_finding_ids` / `issue_count` | 附着 finding 摘要；未匹配 finding 进入 manifest diagnostics，不静默丢弃 |
| `translation_record` | 可选 `translation_records.jsonl` provenance 摘要（record_id / origin / status / revision_history 计数） |
| `review` | 当前生命周期、reviewer、note、decided_at、history、`needs_recheck` / `changed_bindings` |

finding 匹配顺序：`(file_rel_path, line)` → `item_id == occurrence_id`；匹配不到时在 manifest 写
`REVIEW_QUALITY_FINDING_UNMATCHED` 诊断。重复 occurrence 通过 occurrence_id 保持独立。

## 3. 人工决定与 needs_recheck

决定日志每行一条可审计动作（append-only）：

```json
{
  "schema_version": 1,
  "decision_id": "…",
  "occurrence_id": "occ-1",
  "lifecycle": "open | ignored | resolved",
  "reviewer": {"type": "human | agent", "name": "…", "run_id": ""},
  "binding": {"entry_id": "…", "snapshot_digest": "…", "source_digest": "…",
              "target_digest": "…", "context_digest": "…", "evidence_digest": "…"},
  "note": "",
  "decided_at": "2026-09-20T00:00:00+00:00",
  "supersedes": ""
}
```

- 生命周期只有 `open` / `ignored` / `resolved` / `needs_recheck`；`needs_recheck` 由绑定变化派生，
  不允许直接导入。`ignored` 只表示本次审校生命周期，不等于 quality-ack、reuse accept 或 apply 授权。
- 重建时按 `occurrence_id` 找最新决定；绑定完全一致才沿用生命周期；任一 binding digest 变化 →
  `needs_recheck`，记录 `changed_bindings` 与 `previous_lifecycle`。
- 决定历史按 occurrence 保存在 `review.history`，导入/导出不删除旧动作；决定记录不存在时默认 `open`。
- 找不到 occurrence 的决定写入 manifest `REVIEW_DECISION_ORPHANED` 诊断，决定本身不丢。

## 4. CLI

| 命令 | 说明 |
|---|---|
| `review-index-build --corpus PATH [--quality-findings PATH] [--translation-records PATH] [--decisions PATH] [--output-dir DIR]` | 读取 corpus manifest/JSONL/dir，构建/重建索引；输出 JSONL + manifest + Markdown；无 decisions 时写模板 |
| `review-index-status --index PATH` | 只读汇总条目数、finding 附着数、生命周期计数、needs_recheck 与诊断 |
| `review-decisions-export --index PATH --file PATH` | 导出当前决定日志；无决定时导出可编辑模板（reviewer.name 为 `TODO`） |
| `review-decisions-import --index PATH --file PATH` | 校验并追加决定；重复 `decision_id` 跳过，孤儿 occurrence 记诊断；随后刷新索引 review 状态，不覆盖输入决定文件 |

全部命令支持 `--output json`，并加入 `capabilities` / machine envelope 合同。S1 不修改
translator config、API key、glossary、quality acknowledgement 或任何 `.rpy`。

## 5. S2 GUI 订正页（后续）

在现有订正页内增加「逐条审校」工作区，不新增主页面：

- 表格/列表：按文件、生命周期、是否有 finding、severity、speaker、source/current 关键字筛选；
  默认只显示当前范围，支持跳转到下一未处理条目。
- 详情面板：原文、当前译文、前后上下文、finding 证据/suggestion、translation record provenance、
  缺失证据明确显示为 `unknown`/未附着，而不是空白通过。
- 编辑：生成建议译文草稿，走 #321 revision proposal / `import-revision-proposals` / 现有
  `preview-revisions` → `apply-revisions`；GUI 不直接写 `.rpy`，不把导入成功显示成游戏文件已修改。
- 决定：保存 `ignored` / `resolved` / 重新打开；展示 `needs_recheck` 与绑定变化；reviewer 身份、
  note、decided_at 与历史可恢复。
- 复用 #362 quality acknowledgement 与 #363 finding 展示，不互相授权；项目/任务切换按 #348
  identity 隔离，不跨项目复用 session 或决定。

## 6. 验收对照

| #427 验收项 | S1 现状 |
|---|---|
| 没有 finding 的普通译文可浏览与生成 proposal；CLI/GUI 使用同一核心结果 | 核心 + CLI 可浏览全部 corpus occurrence；proposal 生成属 S2 GUI |
| 重复原文、不同角色/场景和不同版本不会混淆，缺失证据有明确展示 | occurrence_id 保留重复；project identity + binding 隔离版本；未匹配 finding/孤儿决定进入 diagnostics |
| index 可重建，人工决定可恢复；source/target/context 变化进入 needs_recheck | 重建确定性测试；决定日志分离；binding 变化派生 needs_recheck |
| 条目审校状态、quality ack 和 reuse decision 各自持久化且不互相授权 | 独立 `review_decisions.jsonl`；不写 manifest quality ack / reuse decisions |
| 编辑只生成提案；invalid/stale/conflict 无法进入有效 preview/apply | S2 GUI 接 #321；S1 不触碰写回 |
| 切换项目/任务不复用旧 session；大语料加载不阻塞 UI，筛选只影响明确选中范围 | S2；S1 identity/binding 已隔离 |
| CLI JSON、中文 user copy、现行订正文档和 GUI 自动化同步；最小窗口可用 | CLI JSON + 中文文案已交付；GUI 与窗口属 S2 |

## 7. 诊断与 reason code

| code | 含义 |
|---|---|
| `REVIEW_QUALITY_FINDING_UNMATCHED` | quality finding 未能匹配任何 corpus occurrence；仍保留在 manifest diagnostics |
| `REVIEW_DECISION_ORPHANED` | 决定引用的 occurrence 不在当前索引；决定保留但不应用 |
| `REVIEW_QUALITY_FINDING_AMBIGUOUS` | 同一 occurrence 匹配到多个来源不明的 finding，需人工确认 |
| `REVIEW_CORPUS_ROW_MISSING_IDENTITY` | corpus row 缺 occurrence_id，跳过并显式诊断 |
| `REVIEW_INDEX_INPUT_MISSING` / `REVIEW_INDEX_INPUT_INVALID` | corpus / findings / decisions 输入缺失或不可解析 |
| `REVIEW_DECISION_INVALID` | 决定 lifecycle、reviewer、binding 或 note 不满足合同 |
| `REVIEW_INDEX_MISSING` | 指定 index 包不存在 |

## 8. 限制

- 索引是派生缓存；S1 不实现 GUI、不生成 revision proposal、不直接写 `.rpy`。
- 质量 finding 的匹配依赖 corpus row 的 file/line 与 item_id；路径规范化范围有限，
  未匹配会显式诊断，不会静默当作通过。
- 决定日志按 occurrence 的最新一条生效；时间戳用于展示/审计，不参与绑定 freshness。
- 真实大语料的分页、性能与窗口交互在 S2 测量；S1 只保证离线 JSONL 可复现。
