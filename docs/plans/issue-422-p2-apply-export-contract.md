# #422 P2：`apply --export-dir` 写回 + 导出双写事务合同

状态：第二轮复审修复后的冻结修订稿（Refs #422）。
范围：Ren'Py Batch translation manifest 的 `apply --export-dir <PATH>`；P1 `--export-only` 行为不变。
非目标：累计发布补丁、历史文件合并/删除、rpa/rpyc 编译、引擎封包、安装器、durable Sync / revision 写回。

## 1. 事务边界

P2 把 **同一份已验证渲染结果** 同时提交到两个物理位置，并用一个可恢复 journal 保护：

| 输出 | 位置 | 来源 | 是否进入文件事务 |
|------|------|------|------------------|
| workspace 写回文件 | 清单 TL 根下的实际源文件路径 | `render_writeback_plan` 渲染结果（含源换行/BOM 合同） | 是 |
| export payload 文件 | 清单游戏根的相对路径映射到 `--export-dir` | 同一份渲染结果中相对源文件字节确有变化的文件 | 是 |
| P2 导出收据 `apply_export_record.json` | 清单 `_package_dir` | request / 文件摘要 / workspace 摘要 / 恢复计划 / pending 步骤 | 是（通常作为最后一个 replace target） |
| journal `.apply_export_transaction.json` | 清单 `_package_dir` | staged/backup/内容摘要/元数据/回滚阶段 | 事务自身 |
| manifest (`applied_at` / `apply_summary` / `export_summary` / `apply_state_advancement`) | 清单路径 | 文件事务提交成功后的状态推进 | 否，receipt 为提交标记 |
| progress JSON | 当前 Batch 进度文件 | 已提交 workspace 行号 | 否，提交后按 receipt 计划幂等补记 |
| RAG store | 当前 RAG store | 已提交 workspace 文件 | 否，提交后幂等 upsert；失败保持 pending |
| next split / latest 游标 | 清单与 `latest_manifest.txt` | 现有 split 规则 + 前值条件推进（共享锁 CAS） | 否，提交后按 receipt 计划幂等补记 |

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
   7. 原子写 `committed` journal，随后尽力清理 staged/backup/journal；清理结果见 FE-08。
4. **状态推进**（receipt 为提交标记，幂等；顺序与代码一致）：
   1. 复读 receipt 中受管 workspace / export 树；发现事务外修改、路径身份改变
      （link/junction/reparse）或字节漂移则 fail closed，不覆盖、不推进状态
      （稳定错误码 `APPLY_EXPORT_OUTPUT_CHANGED`）；
   2. `update_progress(file_key, line_numbers)`（集合 union，天然幂等）；
   3. 读取 receipt/manifest 中持久化的 `pending_steps`。`"rag"` 是权威待办：
      - RAG 开启且有 jobs：执行 `sync_rag_store_for_jobs`；成功才清除该待办；
      - RAG 关闭：不得当成完成，保持 pending 并报告 `rag_status=blocked_disabled`；
      - RAG 开启但无可执行 jobs：保持 pending 并报告 `rag_status=blocked_no_jobs`；
      - 返回 error 摘要或抛异常：保持 pending，记录 `last_error` / `pending_reason`，
        文件不回滚；
      - 本版本不提供静默豁免/waive 路径；关闭配置不会把 pending 改为 complete。
   4. 条件推进 latest 游标：读写都汇到共享模块 `atomic_io` 的 latest 写服务，
      统一使用 `<LATEST_MANIFEST_FILE>.lock`；GUI 与 CLI 物理路径一致（GUI
      `logs/batch_jobs/latest_manifest.txt`）。读、比较、写在同一临界区内完成。
      当前值等于 receipt 记录的前值时推进；等于目标时 `already_advanced`；
      属于其它更新操作时保留新值并记录 `retained_newer`，禁止覆盖；锁不可用/
      超时不得静默放行。latest 锁不自动抢占、遗弃锁仅人工清理，见 §6.1；
   5. 写 manifest：`applied_at`、`apply_summary`、`export_summary`、`export_record_path`、
      `apply_state_advancement=complete`、必要的 next split、latest 阶段记录；清理 stale 字段；
      任一 pending 步骤未完成时写 `apply_state_advancement.status=pending` + `pending_steps`
      + `pending_reason` / `last_error`，不写 `applied_at`，不推进 latest；
   6. 原子把 receipt 的 `state_advancement.status` 改为 `complete`，并写入 `rag_status`、
      `latest_cursor`；pending 路径则把 `pending_steps` / `pending_reason` / `last_error`
      持久化进 receipt 与 manifest。
5. 任一步失败：journal / receipt 状态必须能解释为 `prepared`（文件事务未提交）、
   `rolling_back` / `rolled_back`（回滚阶段可重放）、`committed + state pending`
   （文件已提交，状态待补记）或 `recovery_required`（无法安全回滚/补记）。
   CLI 只在前述全部成功后报告 `applied_and_exported` / `no-op`。

### 2.1 严格回滚阶段（D1）

严格 recovery（新 journal 均带 `staged_sha256` / `target_preimage_sha256`）把回滚本身也持久化：

1. 先对每个 entry 分类：`uncommitted` / `committed` / `rolled_back` / 非法外部状态。
   分类只看当前 target 是否等于 staged 字节或 preimage 字节，不信任“staged 缺失”单一信号。
2. 任何 target 既不等同 staged 也不等同 preimage：立即 fail closed，不改任何 target。
3. 合法 journal 先原子写 `state="rolling_back"`，并为每个 entry 写 `rollback_state`。
4. 逆序处理 `pending` entry：已提交的恢复 backup（或删除新建 target），随后立即把该 entry
   持久化为 `rollback_state="rolled_back"`；已回滚项直接跳过。
5. 全部完成后写 `state="rolled_back"`，再清理 staged/backup/journal。
6. 回滚中途进程停止或注入失败后重启：已回滚项被识别为合法状态；尚未回滚项继续回滚。
   已回滚 target 若被事务外再次修改，则仍按外部修改 fail closed，不覆盖。
7. **legacy 边界（兼容性边界钉住）**：严格 fail-closed 只适用于带
   `staged_sha256` / `target_preimage_sha256` 的新 journal。旧 prepared journal
   （无摘要）保留历史恢复语义，成功恢复后 journal 会被清理，恢复过程可能覆盖
   外部修改。本版本不升级旧格式；FE-25 与清理测试只钉住该行为边界，不是
   “旧格式回归必红”的证据。

## 2.2 共享 latest 写服务（F1）

- 物理写 latest 的唯一入口是 `atomic_io.write_latest_manifest_locked` 与
  `atomic_io.compare_and_swap_latest_manifest_locked`；两者共享
  `latest_manifest_lock_path(<latest>) = <latest>.lock`。
- CLI `remember_latest_manifest` / `remember_latest_manifest_if_unchanged` 与
  `save_manifest(update_latest=True)` 都是这些服务的薄包装；
  GUI `ProjectState.remember_latest_manifest_path` 直接调用同一服务。
- GUI 与 CLI 解析到同一物理 latest 文件（`tool_root/logs/batch_jobs/latest_manifest.txt`），
  因此 GUI writer 与 CLI CAS 不会交错成“CLI 按旧前值覆盖 GUI 新值并报 advanced”。
- 回归测试：真实 GUI writer 等待持锁 holder、GUI writer 先写后 CLI CAS 返回
  `retained_newer`、跨进程互斥，以及仓库源码扫描禁止 shared service 之外的代表性
  直接写模式（`write_text`/`write_bytes`/`atomic_write_text`、写模式 `open`、
  `os.replace`/`os.rename`、`shutil.copy*`、`.replace`/`.rename`）。该扫描是
  limited-pattern 检查，**不能证明不存在任意其它 writer**；这是验收辅助，不是
  完备性证明。

## 3. 状态表

| 阶段 | journal | receipt.state_advancement | manifest | workspace 内容 | export 树 | progress / latest / RAG | 下次运行行为 |
|------|---------|---------------------------|----------|----------------|-----------|--------------------------|--------------|
| 无操作 | 不存在 | 不存在 | 未应用 | 原始 | 原始/不存在 | 未推进 | 正常 apply |
| 预检失败 | 不存在 | 不存在 | 未应用 | 原始 | 原始/不存在 | 未推进 | 修复后重试，无副作用 |
| stage/backup 中崩溃 | 不存在或 `prepared` | 不存在 | 未应用 | 原始或部分 | 原始/不存在 | 未推进 | strict recovery；无 journal 时仅清理 temp |
| 替换中崩溃（部分 target 已 replace） | `prepared` | 可能存在 | 未应用 | 部分新/旧 | 可能部分存在 | 未推进 | 分类后整体回滚 |
| 全部 replace 后、committed journal 前崩溃 | `prepared` | receipt 通常已存在（最后一个 replace target） | 未应用 | 全部新 | 全部新 | 未推进 | 整体回滚（未记录成功） |
| committed journal 后崩溃 | `committed` | receipt 通常已存在 | 未应用 | 全部新 | 全部新 | 未推进 | 清理 journal；发现 receipt 后进入状态补记 |
| 回滚中断（D1） | `rolling_back` | 可能存在 | 未应用 | 部分已回滚、部分新 | 部分存在 | 未推进 | 识别 `rollback_state`，继续回滚并收敛 |
| 回滚完成、清理前崩溃 | `rolled_back` | 可能存在 | 未应用 | 全部 preimage | 无受管新文件 | 未推进 | 只清理 temp/journal |
| 文件事务成功、状态推进前崩溃 | 已清理 | `pending` | 未应用 | 全部新 | 全部新 | 未推进 | 复读 receipt，校验受管输出后幂等补记 |
| RAG 返回 error / URL 不可用（D4） | 已清理 | `pending` + `pending_steps=["rag"]` + `last_error` | `apply_state_advancement.status=pending`、`pending_steps=["rag"]`、无 `applied_at` | 全部新 | 全部新 | progress 可能已补；latest 未推进 | 下次 apply 重跑 RAG；成功后才 complete |
| RAG 配置关闭但 receipt 仍有 pending rag（D4） | 已清理 | `pending_steps=["rag"]`、`pending_reason=blocked_disabled`、`last_error` | pending + 同上 | 全部新 | 全部新 | 未推进 | 继续 pending；重新开启并成功后才 complete，配置变化不能抹除待办 |
| RAG 开启但无可执行 jobs | 已清理 | pending + `pending_reason=blocked_no_jobs` | pending | 全部新 | 全部新 | 未推进 | 修复 job 记录或恢复收据后重试 |
| latest 已被其它操作推进（D5） | 已清理 | pending → 重放后 complete | complete + `latest_cursor.status=retained_newer` | 全部新 | 全部新 | 保留更新游标 | 幂等完成，不回退游标 |
| 状态推进完成、receipt 标记前崩溃 | 已清理 | `pending` | complete | 全部新 | 全部新 | 已推进 | 幂等重放后标记 receipt complete |
| 完成 | 不存在 | `complete` | complete | 全部新 | 全部新 | 已推进 | 幂等返回；成功出口仍复核输出，不重复写文件 |
| 恢复发现事务外修改 / 路径身份改变 | `prepared`/`rolling_back` 保留 | 不适用 | 未应用 | 保留外部修改 | 保留外部修改 | 未推进 | `recovery_required` / `APPLY_EXPORT_OUTPUT_CHANGED`，禁止覆盖 |

## 4. 成功判定与 fail-closed 规则

- 任一 target 在 recovery 时无法与 journal 记录的状态匹配（已提交 target 不等于
  `staged_sha256`，未提交 target 不等于 `target_preimage_sha256`，或本不存在的 target
  被外部创建），recovery 必须整体拒绝并保留 journal/receipt，绝不覆盖外部修改。
- 所有幂等成功出口（fresh 双写、pending receipt 重放、complete receipt 重放、
  `--force` 快捷返回）统一执行 `batch_export.verify_apply_export_entry`：验证
  receipt、规范化路径、workspace target 父路径组件（link / junction / reparse）、
  export 受管树（缺失/修改/额外目录）与 workspace 字节摘要。
  失败返回稳定错误码 `APPLY_EXPORT_OUTPUT_CHANGED`，不得复用旧成功摘要。
- **F4 路径身份读取**：POSIX 且支持 `dir_fd` 时，workspace verifier 先用
  `O_NOFOLLOW` 打开 TL 根，并对根句柄做 `fstat` 与路径 `lstat` 的 dev/ino 身份比对；
  随后逐级 `O_NOFOLLOW` + `dir_fd` 打开组件，读文件句柄并比对 path stat。TL 根或
  父目录在检查后、打开前被替换为 link/junction 会在根/组件打开或身份比对时失败，
  统一映射为 `APPLY_EXPORT_OUTPUT_CHANGED` / `apply_export.workspace_conflict`。
  测试在两个位置注入：根 open 前替换 TL 根（证明根 `O_NOFOLLOW` 生效），以及在
  helper 最后一次预检后、组件 open 前替换父目录（证明 `dir_fd`/`fstat` 分支实际
  执行且读路径被拒绝）。Windows 等无 `dir_fd` 平台退化为 pre/post link-reparse
  检查 + realpath 比较；在“校验与读取之间”精确插入父目录替换的窄窗口仍存在残余
  TOCTOU 风险，严重度限于“可能把已提交字节的复核授权给错误路径身份”，不会覆盖
  文件；按已知限制记录，不宣称跨平台完全解决。
- 已存在 export 树只有在 receipt 的 `apply_identity`、request fingerprint、受管文件摘要、
  目录集合全部匹配时才允许幂等返回；额外/缺失/修改文件、不同 request、不同 root 均拒绝。
- receipt 存在但 manifest 尚未记录 `applied_at` 时，只能按 receipt 计划补记状态；
  若当前 manifest 的 check/plan/result 身份与 receipt 的 `apply_identity` 不一致，
  按 stale recovery 拒绝，不能拿旧收据推进新结果。
- **pending_steps 权威（F3）**：在判断 complete 之前先计算 receipt
  `state_advancement.pending_steps`、manifest `apply_state_advancement.pending_steps`
  以及 `apply_summary` / `export_summary` 镜像 pending 步骤的并集；**无论任一侧 status
  是否为 complete**，并集非空即不得视为 complete。当前 `RAG_ENABLED` / store 可用性
  只能决定“能否执行”，不能把 `"rag"` 抹成 complete；无法识别的待办步骤结构化拒绝为
  `unsupported_pending_steps`。本版本没有 waive 语义；要放弃待办只能修数据或明确
  重新生成任务，且不得以此伪造成功。
- **矛盾持久化状态证据边界**：receipt/manifest 一侧 complete、另一侧/自身仍带
  pending_steps 属于异常状态下的 fail-closed 加固；本轮未证明它会由正常提交或单次
  崩溃自然产生，不描述为“普通重试必现的 RAG 丢失”。
- **last_error best-effort**：RAG pending 状态会尽力把 `pending_steps` /
  `pending_reason` / `last_error` 写入 receipt、manifest 与 CLI error details；但
  receipt 重写是 best-effort。若 receipt 写入失败，CLI 仍在 details 中报告
  `receipt_updated=false`、manifest 保持 pending，不宣称三处一致落盘。
- P1 `--export-only` 发现同 package 的未决 P2 journal 或 pending receipt 时，
  必须结构化拒绝（`APPLY_EXPORT_RECOVERY_REQUIRED`，`recovery_state` 为
  `recovery_required` / `state_pending`），不得恢复 P2 文件事务、写进度、latest、
  `applied_at`，也不得创建用户请求的导出目录。
- `--force` 只保留现有 `applied_at` guard 的语义；stale check、源快照、结构阻断、
  plan 绑定、导出目录冲突、输出复核均不可被 `--force` 绕过。
- 一侧文件提交失败/校验失败、RAG 未成功、pending 步骤未清、latest 条件写入失败或锁不可用：
  不得输出 `applied`、`applied_and_exported` 或 `exported` 的成功状态；只能输出结构化
  `failed` / `recovery_required` / `state_pending`。

## 5. RAG pending 与 last_error 语义（D4）

| 问题 | 语义 |
|------|------|
| 权威待办 | receipt `state_advancement.pending_steps` 与 manifest `apply_state_advancement.pending_steps` 的并集；任一存在 `"rag"` 就不得 complete。 |
| 用户可见状态 | workspace 与 export 文件已提交；manifest `apply_state_advancement.status=pending`、`pending_steps=["rag"]`、`pending_reason`、`last_error`、无 `applied_at`；receipt `status=pending`、相同 `pending_steps` / `pending_reason` / `last_error`。 |
| CLI 结果 | `APPLY_EXPORT_STATE_PENDING`，`recovery_state=state_pending`、`files_committed=true`、`outputs_committed=true`、`pending_steps=["rag"]`、`rag_status`（`blocked_disabled` / `blocked_no_jobs` / `error` / `unsupported_pending_steps`）、`pending_reason`、retryable。 |
| last_error 落点 | 同时写入 manifest `apply_state_advancement.last_error`、receipt `state_advancement.last_error` 和 CLI error details；测试断言 receipt 与 manifest 都有该字段。 |
| 配置关闭 | RAG 关闭时不能执行，只能继续 pending；`rag_status=blocked_disabled`，不得写成 complete/waived。 |
| 重试入口 | 再次运行 `apply <manifest>`（plain 或同一 `--export-dir`）；pending receipt 分支在门禁判断前读取权威 pending 并尝试 RAG。 |
| 是否阻塞后续 apply | 是：该 package/manifest 上的新 apply/导出先解析 pending receipt；其它 package 不受影响。 |
| 完成条件 | `pending_steps` 清空且 RAG 步骤成功返回非 error 摘要后，才写 latest（条件推进）、manifest complete、receipt complete。 |

## 6. latest 游标条件推进语义与并发（D5 / F1 / F2）

- receipt 的 `state_advancement` 在文件事务提交前记录 `latest_previous_value`（事务前游标值）
  与 `latest_target`（本次计划值）；不记录或记录为空按“事务前不存在游标”处理。
- 所有 latest 写入口（CLI `remember_latest_manifest`、`save_manifest(update_latest=True)`、
  conditional replay，GUI `ProjectState.remember_latest_manifest_path`）共享同一把
  `atomic_io.exclusive_file_lock`，锁路径 `<latest>.lock`；GUI 与 CLI 指向同一物理
  文件。锁不可用/超时按现有失败分类抛出，不允许静默放行。
- F1 回归：GUI 与 CLI 写入口都调用 shared service；源码扫描禁止绕过 service 直接写
  latest；真实 GUI writer 与 CLI CAS 的确定性交错要么保留更晚 writer 的值，要么让
  CAS 返回 `retained_newer`，绝不出现“覆盖 + advanced”。

### 6.1 stale 抢占安全语义（F2）

- **latest 写服务关闭自动抢占**：`atomic_io.write_latest_manifest_locked` 与
  `compare_and_swap_latest_manifest_locked` 调用 `exclusive_file_lock` 时显式传
  `stale_after=-1`、`preempt_dead_owner=False`。因此 owner 已死、owner 记录损坏、
  PID 无法判断或锁文件年龄超过任何阈值，都不会触发自动删除；等待者到达 `timeout`
  后抛 `AtomicFileLockTimeoutError`，latest apply 侧统一转成
  `APPLY_EXPORT_STATE_PENDING` / `recovery_state=state_pending`。这是刻意的
  “绝不自动抢占”，因为“读 owner → 判断已死 → unlink”与替换锁之间无法只靠文件
  系统检查消除竞态。
- 人工清理路径（唯一恢复方式）：只有在确认没有 writer 正在使用 `<latest>.lock`、
  且能确认锁文件确为遗弃锁时，操作者才可显式删除该锁文件。不得在存活 owner 持锁
  期间删除。本版本不提供静默强制恢复；若后续增加 maintenance 命令，必须显式
  `--force` 并输出风险提示，不得作为普通写路径的自动行为。
- 通用 `exclusive_file_lock` / `config_store` / `sync_run_store` 仍保留
  `preempt_dead_owner=True` 的默认行为：owner JSON 含 `pid` / `token` / `created_at`，
  超过 `stale_after` 后先做 owner 存活检查：
  - POSIX：`os.kill(pid, 0)`；`ProcessLookupError` 才证明已死；`PermissionError` 视为存活；
    其它 `OSError` 视为无法判断。
  - Windows：`OpenProcess` + `GetExitCodeProcess`，只在明确无效 PID / 非活动进程时
    视为已死；`ERROR_ACCESS_DENIED` 视为存活；其它不可判断。
  - owner 记录损坏、PID 非正整数、或存活/死亡无法判断：一律不抢占，等到 `timeout`
    后抛 `AtomicFileLockTimeoutError`。
  - 抢占前会防御性重读 owner/token 与文件类型；token 校验仍能避免释放阶段删除替换
    锁，但**不能消除两个抢占者之间的 read-then-unlink 竞态**。该默认抢占路径不属于
    #422 的 latest 合同，作为范围外已知限制记录，后续独立 issue 收敛；latest 服务
    通过“从不自动抢占”完全绕开该竞态。
- 条件更新 helper（`remember_latest_manifest_if_unchanged`）在锁内完成读 → 比较 → 写：
  - 当前值 == 目标值：`already_advanced`，不再写入；
  - 当前值 == 记录的前值：写入目标并复核，阶段结果为 `advanced`；
  - 当前值既不是前值也不是目标值：判定为其它合法操作已推进，**保留新值**，阶段结果为
    `retained_newer`，在 manifest `apply_state_advancement.latest_cursor`、
    `apply_summary.latest_cursor` 与 receipt 完成记录中给出用户可见诊断，不覆盖。
- helper 内部不复用会再次上锁的 `remember_latest_manifest`（`exclusive_file_lock` 非重入）。
- `sync execution` 不更新 latest（`skipped`）；next split 存在时目标仍是 next split manifest。
- 正常重放只推进一次；并发/交错 writer 下不覆盖更晚的游标。

## 7. 故障注入矩阵

测试文件：`tests/test_batch_apply_export.py`（单元 + 故障注入）、
`tests/test_atomic_io.py`（严格回滚阶段 / legacy 边界）、
`tests/test_batch_golden_corpus.py`（端到端 + 并发锁）、
`tests/test_gemini_translate_batch_cli_contract.py`（机器输出）。

| ID | 注入点 | 期望 journal / receipt | 期望可见状态 | 期望重试/恢复 |
|----|--------|------------------------|--------------|----------------|
| FE-01 | workspace prefix 第一个文件 stage 失败 | 无 journal；无 receipt；temp 清理 | 两侧均原始 | 修复后正常重试 |
| FE-02 | workspace 第一个/中间/最后一个 target replace 失败 | `prepared`/回滚阶段 | 前缀可能新，随后严格回滚 | 下次 apply 恢复后重试 |
| FE-03 | export 第一个/中间/最后一个 target replace 失败 | `prepared`/回滚阶段 | workspace 已替换项随后回滚，export 原始 | 同上 |
| FE-04 | receipt `os.replace` 失败 | `prepared`/回滚阶段 | workspace/export 已替换项全部回滚 | 同上 |
| FE-05 | prepared journal 写入失败 | 无 journal，temp/backup 清理 | 两侧均原始 | 修复后正常重试 |
| FE-06 | committed journal 写入失败 | `prepared` 保留 | 全部新，但 recovery 判定未提交并整体回滚 | 下次 apply 回滚后重试 |
| FE-07 | post-commit validator 失败（外部改动/落盘不一致） | `prepared`/`rolling_back` 保留 | 严格回滚；若外部修改冲突则 `recovery_required` | 人工处置后重试 |
| FE-08a | entries/temp 清理失败但 journal 删除成功 | 无 journal 残留 | 提交有效，target 保持新 | 无 journal 可恢复；残留 temp 不影响成功判定 |
| FE-08b | entries/temp 清理失败且 journal 删除失败 | `committed` journal 残留 | 提交有效，target 保持新 | 下次 recovery 只清理 committed journal，不回滚 |
| FE-09 | progress 写入失败 | receipt `pending` + `last_error` | 文件与 export 均新 | 下次 apply 幂等补 progress 后完成 |
| FE-10 | RAG 返回 error 或抛异常 | receipt pending；manifest `pending_steps=["rag"]`、`last_error`、无 `applied_at` | 文件与 export 均新 | 下次 apply 重跑 RAG，成功后才 complete；不回滚文件 |
| FE-11 | manifest 保存失败 | receipt `pending` | 文件与 export 均新，progress 可能已补 | 下次 apply 重写 manifest 后完成 |
| FE-12 | receipt complete 标记失败 | receipt `pending`，manifest 已 applied | 文件与 export 均新 | 幂等重放状态推进后标记完成 |
| FE-13 | 恢复前外部修改 workspace 已提交文件 | `prepared`/`rolling_back` 保留，recovery 拒绝 | 保留外部修改 | `recovery_required`，不覆盖 |
| FE-14 | 恢复前外部修改 export 已提交文件 | 同上 | 保留外部修改 | 同上 |
| FE-15 | 重启后 committed journal + pending receipt | 清理 committed journal，然后状态补记 | 文件/export 保持新 | 成功返回 `applied_and_exported` |
| FE-16 | stale check / source drift / 结构阻断 / 非法 plan + `--force` | 不产生 journal/receipt | 两侧原始 | 结构化拒绝，无旁路 |
| FE-17 | 回滚恢复中途再次失败后重试（D1） | `rolling_back` + 逐项 `rollback_state` | 已回滚项保持 preimage；未回滚项保持 staged 新 | 重试识别已回滚项；无外部修改时收敛 |
| FE-18 | P2 prepared/pending → P1 `--export-only`（D2） | 保留 P2 journal/receipt | workspace/progress/latest/applied_at 零变化；请求目录不创建 | 结构化 `APPLY_EXPORT_RECOVERY_REQUIRED` |
| FE-19 | complete receipt 重放时删/改/增输出（D3） | receipt 不变并通过 manifest 指向原值 | 输出保持篡改后的状态 | `APPLY_EXPORT_OUTPUT_CHANGED`；不复用旧成功摘要 |
| FE-20 | RAG error → pending 与重试（D4） | receipt pending；manifest pending | 文件与 export 均新 | 重试再次调用 RAG；成功后才 complete |
| FE-21 | latest 已被后续操作推进后重放（D5） | receipt 最终 complete；阶段记录 | 保留更新后的游标 | `retained_newer`，不回退、不覆盖 |
| FE-22 | RAG pending 后关闭 RAG 再重试（D4） | receipt/manifest 保持 pending，`blocked_disabled` | 无 `applied_at`、latest 不推进 | 重新开启并成功后才 complete；无 waive 静默路径 |
| FE-23 | workspace 父目录被换成 symlink/junction，字节不变（D3） | receipt 不变 | target 路径身份改变 | `APPLY_EXPORT_OUTPUT_CHANGED` / workspace conflict |
| FE-24 | 两个 writer 并发 latest（D5） | 共享锁；CAS 记录阶段 | 保留更晚 writer 的值 | 锁内读-比较-写；不静默放行锁失败 |
| FE-25 | legacy prepared journal（无摘要） | 恢复后 journal 清理 | 历史恢复可能覆盖外部修改 | 兼容性边界钉住；不升级旧格式，不作为旧格式必红证据 |
| FE-26 | GUI latest writer + CLI CAS 交错（F1） | 同一 `<latest>.lock` | 保留更晚 writer 的值或 CAS `retained_newer` | 禁止“覆盖 + advanced”；所有物理写入口走 shared service |
| FE-27 | 存活 owner 超过 stale_after（F2） | 锁不被抢占 | 等待者 timeout → 结构化 pending | latest 服务从不自动抢占；generic lock 仅在可证明 owner 已死时抢占，未知不抢占 |
| FE-30 | 已死 owner 的 aged latest 锁（F2） | `<latest>.lock` 保留原 owner/token | latest 值不变，等待者 timeout → 结构化 pending | 必须人工确认无 writer 后删除锁；两个 reaper 并发也不能删除/双写 |
| FE-31 | 同级/后续 writer 替换锁后旧 owner 释放（F2） | 旧 owner 不删除替换锁 | 替换锁及其 writer 不受影响 | token 校验防误删；latest 路径因从不抢占不涉及抢占竞态 |
| FE-28 | complete/pending 矛盾 + pending_steps（F3） | pending 并集权威 | 不返回 applied、不写 applied_at、不推进 latest | 可执行步骤续跑；未知步骤结构化拒绝；配置关闭仍 pending |
| FE-29 | 检查后、读取前替换 workspace 父目录（F4） | verifier 句柄/路径身份校验 | POSIX 拒绝；Windows 窄窗口残余风险记录 | `APPLY_EXPORT_OUTPUT_CHANGED`；不宣称跨平台完全解决 |

## 8. CLI / 机器输出

- `apply <manifest> --export-dir <PATH>` 与 `--export-only` 互斥、显式目标目录。
- P2 成功状态：`applied_and_exported`；零变化：`no-op`；提交失败：错误 envelope 带
  `mode="apply-export"`、`reason_code`、`recovery_state`（`none` / `recovered` /
  `recovery_required` / `state_pending` / `state_conflict`）、`journal_path`、
  `record_path`、`export_root`。
- 稳定错误码：`APPLY_EXPORT_OUTPUT_CHANGED`（幂等重放输出/路径身份复核失败）、
  `APPLY_EXPORT_STATE_PENDING`（RAG/manifest/receipt 补记未完成；`pending_steps`、
  `pending_reason`、`rag_status`、`last_error`、`receipt_updated` 可诊断）、
  `APPLY_EXPORT_RECOVERY_REQUIRED`（journal/receipt 未决）。
- `result.apply` 至少包含：`mode`、`export_root`、`record_path`、`exported_files`、
  `applied_files`、`actual_applied_files`、`applied_lines`、`recovery_state`、
  `state_advancement`、`pending_steps`、`pending_reason`、`rag_status`、`latest_cursor`。
  不能仅靠 `applied_at` 推断 P1/P2 成功状态。
- P1 `--export-only` 的 `mode`、状态与既有字段保持不变；P1 遇到未决 P2 状态时只拒绝，
  不产生任何 P2 副作用。

## 9. 验收映射

- P2 写回字节与导出字节一致：preflight `sides_mismatch` + post-commit validator。
- 严格回滚可重放且不覆盖事务外修改：D1 回滚阶段 + FE-17/13/14；legacy 边界 FE-25。
- P1 不执行 P2 恢复/状态补记：D2 + FE-18。
- 所有幂等成功出口复核输出与路径身份，不伪报成功：D3 + FE-19/23/29。
- POSIX 根/父目录路径身份由句柄分支拒绝，并映射稳定错误码：F4 + FE-29。
- RAG 失败/配置变化可追踪、可重试、不 complete：D4 + FE-10/20/22。
- latest 条件推进、共享锁、不覆盖后续操作，且 latest 锁从不自动抢占：D5/F2 + FE-21/24/27/30/31。
- 普通 apply / 不支持模式无回归：P1 测试 + durable/revision 拒绝导出选项 + `--export-only` 既有测试。

## 10. 已知限制与范围外

- Windows 等无 `dir_fd` 平台仍存在“校验与读取之间”精确替换父目录的窄 TOCTOU 窗口；
  目前以 pre/post link/reparse + realpath 检查兜底，不宣称跨平台完全解决。
- `config_store` / `sync_run_store` 的 generic lock 默认抢占路径仍有概率极低但存在的
  两个抢占者 read-then-unlink 竞态；本 PR 未全局改变默认语义，作为范围外已知限制记录，
  后续独立 issue 收敛。latest 服务不受影响，因为它完全不自动抢占。
- latest 源码扫描是 limited-pattern 扫描，只覆盖代表性直接写模式，不能数学证明仓库
  中不存在任意其它物理 writer。
- reparse-only / junction unit test、FE-25 legacy prepared journal、清理阶段边界测试
  是兼容性或边界钉住，不声称在旧审计版本上必然失败；回归证据以明确的 fault-injection
  与并发交错测试为准。
- 真实进程终止/断电持久性、RAG upsert 在 kill 中途的注入未在本轮故障矩阵中模拟；
  文件事务边界按 `atomic_io` 已有语义与崩溃阶段矩阵验收。
