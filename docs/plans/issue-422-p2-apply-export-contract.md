# #422 P2：`apply --export-dir` 写回 + 导出双写事务合同

状态：实现前冻结稿（Refs #422）。
范围：Ren'Py Batch translation manifest 的 `apply --export-dir <PATH>`；P1 `--export-only` 行为不变。
非目标：累计发布补丁、历史文件合并/删除、rpa/rpyc 编译、引擎封包、安装器、durable Sync / revision 写回。

## 1. 事务边界

P2 把 **同一份已验证渲染结果** 同时提交到两个物理位置，并用一个可恢复 journal 保护：

| 输出 | 位置 | 来源 | 是否进入文件事务 |
|------|------|------|------------------|
| workspace 写回文件 | 清单 TL 根下的实际源文件路径 | `render_writeback_plan` 渲染结果（含源换行/BOM 合同） | 是 |
| export payload 文件 | 清单游戏根的相对路径映射到 `--export-dir` | 同一份渲染结果中相对源文件字节确有变化的文件 | 是 |
| P2 导出收据 `apply_export_record.json` | 清单 `_package_dir` | request / 文件摘要 / workspace 摘要 / 恢复计划 | 是 |
| journal `.apply_export_transaction.json` | 清单 `_package_dir` | staged/backup/内容摘要/元数据 | 事务自身 |
| manifest (`applied_at` / `apply_summary` / `export_summary`) | 清单路径 | 文件事务提交成功后的状态推进 | 否，receipt 为提交标记 |
| progress JSON | 当前 Batch 进度文件 | 已提交 workspace 行号 | 否，提交后按 receipt 计划幂等补记 |
| RAG store | 当前 RAG store | 已提交 workspace 文件 | 否，提交后幂等 upsert |
| next split / latest 游标 | 清单与 `latest_manifest.txt` | 现有 split 规则 | 否，提交后按 receipt 计划幂等补记 |

文件事务使用现有 `atomic_io.atomic_write_many_bytes`：所有 target 先 stage + backup，写
`prepared` journal，再逐个 `os.replace`，全部替换且 post-commit 验证通过后写 `committed`
journal 并清理。跨磁盘文件集合不声称瞬时全局原子性；合同是**失败可恢复、状态可解释**。

## 2. 提交顺序

1. **门禁复核**：manifest/项目/结果身份、最近匹配 check、stale check、源快照、
   TranslationPlan、adapter validation、声明式 WritebackPlan、`writeback_gate=allow`。
   `--force` 不绕过其中任何一项；`quality_gate` 沿用现有 blocker 策略。
2. **两侧预检**：
   - 目标目录必须不存在或为空，或与同一 `apply_identity` 的既有 P2 receipt 完全匹配；
   - workspace 路径必须在 TL 根内且与验证源路径同一；export 路径按真实源路径映射到游戏根；
   - 校验 `..`、绝对路径、Windows 大小写/重复目标、符号链接/目录联接、根重叠、路径逃逸；
   - 比对两侧渲染字节：每个 export payload 必须能在 workspace payload 中找到同一源路径、
     同一输出字节的对应项，否则拒绝（`apply_export.sides_mismatch`）；
   - 再次读取两侧源文件字节，和 check 时快照不一致则拒绝。
3. **文件事务**（journal：`.apply_export_transaction.json`，kind：`apply_export`）：
   1. 为 workspace payload 逐个检查 expected preimage（源快照 SHA-256）并 stage/backup；
   2. 为 export payload 逐个检查目标缺失（首次导出）或与既有 receipt 输出摘要一致后 stage/backup；
   3. stage P2 receipt 字节（包含 `state_advancement.status="pending"` 恢复计划）；
   4. 原子写 `prepared` journal（含每项 `staged_sha256`、`target_preimage_sha256` 和事务元数据）；
   5. 按 **workspace → export payload → receipt** 顺序 `os.replace`；
   6. post-commit validator 复读两侧落盘 SHA-256 与 export 目录树；失败视为提交失败并回滚；
   7. 原子写 `committed` journal，随后尽力清理 staged/backup/journal（清理失败可留待下次 recovery）。
4. **状态推进**（receipt 为提交标记，幂等）：
   1. 复读 receipt 中受管 workspace / export 树；发现事务外修改则 fail closed，
      不覆盖、不推进状态；
   2. `update_progress(file_key, line_numbers)`（集合 union，天然幂等）；
   3. RAG `sync_rag_store_for_jobs(..., quality_state="batch_applied")`（按 memory_id /
      source_checksum 幂等；失败只记录摘要 error，不伪报文件失败也不回滚已提交文件）；
   4. 写 manifest：`applied_at`、`apply_summary`、`export_summary`、`export_record_path`、
      `apply_state_advancement=complete`、必要的 next split 字段；清理 stale export/apply failure 字段；
   5. 按现有规则写 latest 游标（next split 优先，否则当前 manifest；sync execution 不更新）；
   6. 原子把 receipt 的 `state_advancement.status` 改为 `complete`。
5. 任一步失败：journal / receipt 状态必须能解释为 `prepared`（文件事务未提交）、
   `committed + state pending`（文件已提交，状态待补记）或 `recovery_required`
   （无法安全回滚/补记）。CLI 只在前述全部成功后报告 `applied_and_exported` / `no-op`。

## 3. 状态表

| 阶段 | journal | receipt.state_advancement | manifest.applied_at | workspace 内容 | export 树 | progress / latest / RAG | 下次运行行为 |
|------|---------|---------------------------|---------------------|----------------|-----------|--------------------------|--------------|
| 无操作 | 不存在 | 不存在 | 未设置 | 原始 | 原始/不存在 | 未推进 | 正常 apply |
| 预检失败 | 不存在 | 不存在 | 未设置 | 原始 | 原始/不存在 | 未推进 | 修复后重试，无副作用 |
| stage/backup 中崩溃 | 不存在或 `prepared` | 不存在 | 未设置 | 原始或部分回滚 | 原始/不存在 | 未推进 | 若 journal 存在：严格回滚；无 journal：仅清理 temp |
| 替换中崩溃（部分 target 已 replace） | `prepared` | 不存在 | 未设置 | 可能部分新 | 可能部分存在 | 未推进 | 严格检查事务外修改后全部回滚 |
| 全部 replace 后、committed journal 前崩溃 | `prepared` | 不存在 | 未设置 | 全部新 | 全部新 | 未推进 | 同上：整体回滚（未记录成功） |
| committed journal 后崩溃 | `committed` | 不存在 | 未设置 | 全部新 | 全部新 | 未推进 | 清理 journal；发现 receipt 后进入状态补记 |
| 文件事务成功、状态推进前崩溃 | 已清理 | `pending` | 未设置 | 全部新 | 全部新 | 未推进 | 复读 receipt，校验受管文件后幂等补记 |
| 状态推进部分完成 | 已清理 | `pending`（可带 `last_error`） | 可能未设置/旧值 | 全部新 | 全部新 | 可能部分推进 | 重试 union/upsert，不重复计数 |
| 状态推进完成、receipt 标记前崩溃 | 已清理 | `pending` | 已设置 | 全部新 | 全部新 | 已推进 | 幂等重放后标记 receipt complete |
| 完成 | 不存在 | `complete` | 已设置 | 全部新 | 全部新 | 已推进 | 幂等返回；不再次写文件 |
| 恢复发现事务外修改 | `prepared` 保留 | 不适用 | 未设置 | 保留外部修改 | 保留外部修改 | 未推进 | `recovery_required`，人工处置，禁止覆盖 |

## 4. Fail-closed 规则

- 任一 target 在 recovery 时无法与 journal 记录的状态匹配（已提交 target 不等于
  `staged_sha256`，未提交 target 不等于 `target_preimage_sha256`，或本不存在的 target
  被外部创建），recovery 必须整体拒绝并保留 journal/receipt，绝不覆盖外部修改。
- 已存在 export 树只有在 receipt 的 `apply_identity`、request fingerprint、受管文件摘要、
  目录集合全部匹配时才允许幂等返回；额外/缺失/修改文件、不同 request、不同 root 均拒绝。
- receipt 存在但 manifest 尚未记录 `applied_at` 时，只能按 receipt 计划补记状态；
  若当前 manifest 的 check/plan/result 身份与 receipt 的 `apply_identity` 不一致，
  按 stale recovery 拒绝，不能拿旧收据推进新结果。
- `--force` 只保留现有 `applied_at` guard 的语义；stale check、源快照、结构阻断、
  plan 绑定、导出目录冲突均不可被 `--force` 绕过。
- 一侧文件提交失败/校验失败时，不得输出 `applied`、`applied_and_exported` 或
  `exported` 的成功状态；只能输出结构化 `failed` / `recovery_required`。

## 5. 故障注入矩阵（骨架）

测试文件计划：`tests/test_batch_apply_export.py`（单元 + 故障注入）、
`tests/test_batch_golden_corpus.py`（端到端）、`tests/test_gemini_translate_batch_cli_contract.py`（机器输出）。

| ID | 注入点 | 期望 journal / receipt | 期望可见状态 | 期望重试/恢复 |
|----|--------|------------------------|--------------|----------------|
| FE-01 | workspace prefix 第一个文件 stage 失败 | 无 journal；无 receipt；temp 清理 | 两侧均原始 | 修复后正常重试 |
| FE-02 | workspace 第一个/中间/最后一个 target replace 失败 | `prepared` 保留 | 仅前缀可能新，随后严格回滚 | 下次 apply 恢复后重试 |
| FE-03 | export 第一个/中间/最后一个 target replace 失败 | `prepared` 保留 | workspace 已替换项随后回滚，export 原始 | 同上 |
| FE-04 | receipt `os.replace` 失败 | `prepared` 保留 | workspace/export 已替换项全部回滚 | 同上 |
| FE-05 | prepared journal 写入失败 | 无 journal，temp/backup 清理 | 两侧均原始 | 修复后正常重试 |
| FE-06 | committed journal 写入失败 | `prepared` 保留 | 全部新，但 recovery 判定未提交并整体回滚 | 下次 apply 回滚后重试 |
| FE-07 | post-commit validator 失败（外部改动/落盘不一致） | `prepared` 保留 | 严格回滚；若外部修改冲突则 `recovery_required` | 人工处置后重试 |
| FE-08 | journal 清理失败 | `committed` 残留 | 两侧均新，receipt pending | 下次运行清理 journal 后补记状态 |
| FE-09 | progress 写入失败 | receipt `pending` + `last_error` | 文件与 export 均新 | 下次 apply 幂等补 progress 后完成 |
| FE-10 | RAG upsert 失败 | receipt `pending`（或摘要 error 后 complete） | 文件与 export 均新 | 幂等 upsert 不重复计数 |
| FE-11 | manifest 保存失败 | receipt `pending` | 文件与 export 均新，progress 可能已补 | 下次 apply 重写 manifest 后完成 |
| FE-12 | receipt complete 标记失败 | receipt `pending`，manifest 已 applied | 文件与 export 均新 | 幂等重放状态推进后标记完成 |
| FE-13 | 恢复前外部修改 workspace 已提交文件 | `prepared` 保留，recovery 拒绝 | 保留外部修改 | `recovery_required`，不覆盖 |
| FE-14 | 恢复前外部修改 export 已提交文件 | 同上 | 保留外部修改 | 同上 |
| FE-15 | 重启后 committed journal + pending receipt | 清理 committed journal，然后状态补记 | 文件/export 保持新 | 成功返回 `applied_and_exported` |
| FE-16 | stale check / source drift / 结构阻断 / 非法 plan + `--force` | 不产生 journal/receipt | 两侧原始 | 结构化拒绝，无旁路 |

## 6. CLI / 机器输出

- `apply <manifest> --export-dir <PATH>` 与 `--export-only` 互斥、显式目标目录。
- P2 成功状态：`applied_and_exported`；零变化：`no-op`；提交失败：错误 envelope 带
  `mode="apply-export"`、`reason_code`、`recovery_state`（`none` / `recovered` /
  `recovery_required` / `state_pending`）、`journal_path`、`record_path`、`export_root`。
- `result.apply` 至少包含：`mode`、`export_root`、`record_path`、`exported_files`、
  `applied_files`、`actual_applied_files`、`applied_lines`、`recovery_state`、
  `state_advancement`。不能仅靠 `applied_at` 推断 P1/P2 成功状态。
- P1 `--export-only` 的 `mode`、状态与既有字段保持不变。

## 7. 验收映射

- P2 写回字节与导出字节一致：preflight `sides_mismatch` + post-commit validator。
- 故障注入覆盖 stage、首/中/末 replace、journal、manifest/状态、重启恢复：FE-01…FE-16。
- 恢复不覆盖事务外修改：strict target guard + FE-13/14。
- 重试不重复推进 progress/split/RAG：集合 union、memory_id upsert、receipt 计划幂等重放。
- 普通 apply / 不支持模式无回归：P1 测试 + durable/revision 拒绝导出选项 + `--export-only` 既有测试。
