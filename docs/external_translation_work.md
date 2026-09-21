# 外部初译工作包（实验 CLI，#508）

本合同限定 Ren'Py 已准备好的原生翻译目录。宿主负责翻译、审校和分工；工具不调用
翻译、Embedding 或 Project Analysis 模型。它不运行 prepare 自定义命令，也不生成
Provider 请求或成功记录。真实章节质量、字体和游戏运行仍需另外验收。
生成目标沿用现行简体中文合同；`target_language` 表示原生目录语言，`generation_target`
表示译文语言（当前 `schinese`），两者分别绑定，不因修改目录名切换译文语言。

## 最短操作流程

在项目配置已指向隔离副本、原生 TL 模板已经准备好的前提下，从仓库根目录运行。
模型路由或 API Key 不是这些命令的前置条件；无需启用 prepare、RAG、原文索引或项目分析。
导出遵循当前 `include_files` / `include_prefixes` 和待译识别规则。

```powershell
python gemini_translate_batch.py work-export --output-dir outputs/chapter-work --output json --non-interactive
python gemini_translate_batch.py work-read outputs/chapter-work/manifest.json --remaining --limit 50 --output json
python gemini_translate_batch.py work-submit outputs/chapter-work/manifest.json outputs/chapter-submission.json --output json
python gemini_translate_batch.py work-status outputs/chapter-work/manifest.json --output json --strict-exit-codes
python gemini_translate_batch.py check outputs/chapter-work/manifest.json --output json --strict-exit-codes
python gemini_translate_batch.py work-preview outputs/chapter-work/manifest.json --output json
```

审阅 `work-preview` 返回目录中的完整文件 diff 和质量报告后，执行：

```powershell
python gemini_translate_batch.py work-apply outputs/chapter-work/manifest.json --output json --strict-exit-codes
```

`apply <manifest>` 同样识别外部工作包，并消费同一绑定预览；不支持 Batch 的文件树导出参数。
所有 `work-*` 命令支持现有机器 envelope、`--fields`、`--compact`、`--output-file`、
`--non-interactive` 与严格退出码。除了导出，target 均为必填，不读取 latest 回退。
工作包不推进 latest 指针；继续操作时使用导出返回的确切路径。`--output-dir` 必须为新目录，
且不能位于 TL 源目录内；省略时写到当前项目的 Batch 制品目录。

`work-read` 用 `--offset` 和 `--limit` 分页（上限 1000）；`--remaining` 只读取尚未收到成果的
条目。邻接上下文每侧最多两条，`context_truncated` 标明未附全文；需要更广上下文时可读取
其他分页、原生文件或显式参考。可根据已有包的 occurrence ID 重复传入 `--occurrence-id`
导出更小的范围。它固定范围，不能就地追加条目。

## 提交 JSON

从导出的 `work.json` 复制包级绑定，从对应 item 复制 occurrence 和快照摘要。下例中的
占位值必须替换；不要把 Batch Provider response 放进本合同。

```json
{
  "schema_version": 1,
  "kind": "external_translation_submission",
  "submission_id": "chapter-agent-a-001",
  "package_id": "COPY_FROM_WORK",
  "project_id": "COPY_FROM_WORK",
  "package_digest": "COPY_FROM_WORK",
  "reference_digest": "COPY_FROM_WORK",
  "producer": {"type": "agent", "name": "chapter-agent-a", "model": "unknown", "usage": "unknown"},
  "items": [{
    "occurrence_id": "COPY_FROM_ITEM",
    "snapshot_digest": "COPY_FROM_ITEM",
    "expected_candidate_digest": "",
    "translation": "[name]，不要跟着蓝光走。",
    "reason": "保留角色称呼和插值，按已提供的场景语气初译。",
    "review": {"status": "unreviewed"}
  }],
  "reference_proposals": []
}
```

Python 调用方可用 `external_translation_work.submission_template` 生成包级字段。
接收回执返回本次 accepted IDs、candidate digests、剩余 IDs 和 generation。订正时用新
`submission_id`，从 `work-read` 当前 candidate 中取 `candidate_digest` 填入
`expected_candidate_digest`，并给出订正原因。已审校项使用
`{"status": "reviewed", "reviewer": "具名审校者"}`；该声明不等同独立验收。

可选的 `reference_proposals` 每项需要 `kind=term|style`、建议 `text`、
`disposition=accepted|rejected|deferred` 和处置 `reason`。并行时先分配互斥 ID 列表，
让各执行者写独立 JSON，再由协调者串行 `work-submit`、检查全部剩余项并统一写回。

## 持久对象与版本

- 工作包是现有 `manifest.json` 的 `external_work` 扩展，`work.json` 是不可变的交换副本。
  schema 为 1；绑定随机 package ID、基于规范项目路径的 project ID、引擎及 Adapter
  版本、目标语言、原生目录内选定 occurrence、源/现译及文件快照、结构约束和参考版本。
  occurrence 同时携带 Adapter occurrence ID 和现有 unit identity，不按原文去重。
  引擎标识为 Ren'Py；本入口不启动 SDK，无法观察的引擎运行时版本记为 `unknown`。
  文件快照覆盖本次 discovery 范围内的原生 TL 文件（包含原文/现译），不覆盖游戏资源或
  所有原始剧情脚本；需要绑定后者时显式加入 `--reference-file`。
- 成果只存于 manifest 指向的规范化 results JSONL。每次接收先写不可变结果代，再原子
  替换 manifest 中的结果指针和回执。中断留下的未引用结果代不是已接收成果；重放可恢复。
  不新增 SQLite、成果主库或 Provider job。模型和用量不可观察时写 `unknown`。
- 提交绑定 `submission_id`、`package_id`、`project_id`、`package_digest`、
  `reference_digest`。逐条绑定 occurrence、`snapshot_digest` 和
  `expected_candidate_digest`；首次提交为空字符串，订正必须精确指向当前候选版本。
  接收为整次提交原子操作，可只提交包内部分条目。同 ID/同内容返回原回执；同 ID/异内容
  报冲突。竞争提交不覆盖；显式订正需要新提交 ID、当前候选摘要和原因。
- 候选的 `review` 仅记录外部审校证据，不创建 #427 的审校索引，也不等同质量确认或写回授权。
  审校者须具名。修改候选后必须重新表达审校状态。参考只能显式引用已审校候选版本；未写回
  也可引用。参考文件按内容哈希，候选参考按 occurrence 和候选版本绑定。相关参考变化
  使旧包 stale；无关的新候选不会使引用失效。重新导出派生包以采用新依据。
  `--reference-file` 可重复提供 UTF-8 文件，`--reference-work` 可重复提供同项目工作包。
  当前质量检查使用的 glossary 自动绑定为参考，包括“文件尚不存在”的状态；其路径、
  内容或存在性变化均使旧包 stale。check 与 preview 使用同一份 glossary。
- 术语/风格建议随提交回执记录明确的 `accepted` / `rejected` / `deferred` 处置和理由；
  不自动修改术语文件。采纳到共享文件后重新导出工作包。

## 状态与恢复

成果完整性（empty / partial / complete）、外部语义审校（待审 ID 集合）、写回
（not_applied / checked / previewed / recovery_required / applied）分别报告。
未知、跨项目或包外 occurrence 被拒绝；partial 永远列出剩余 ID。

源文件、现译、工作包、Adapter 版本、目标语言或相关参考变化为 stale；提交竞争为 conflict。
work.json、manifest 合同和当前结果代摘要不符属于完整性错误。拒绝不会覆盖已接收译文。
工作包不可原地修改范围；新范围使用新包。首版并行使用同一包内互斥 ID 列表、独立成果
文件、协调者串行接收及汇总写回；不提供领取租约或动态抢任务。

| 情况 | 稳定诊断或状态 | 恢复方式 |
|---|---|---|
| 未知、其他项目或范围外成果 | `WORK_UNKNOWN_OCCURRENCE` / `WORK_SUBMISSION_STALE` | 核对包级字段和分配范围，不能改 ID 猜测归属 |
| 同提交 ID 不同内容、候选版本竞争 | `WORK_SUBMISSION_CONFLICT` / `WORK_CANDIDATE_CONFLICT` | 查询持久冲突记录，用新提交 ID 和当前候选摘要显式处置；可重新确认保留原译文 |
| 尚有未决冲突 | `status=conflict` / `WORK_UNRESOLVED_CONFLICT` | 先处置冲突；旧检查与预览已撤销 |
| 源/现译、参考或包变化 | `WORK_SOURCE_STALE` / `WORK_REFERENCE_STALE` / `WORK_PACKAGE_CHANGED` | 重新导出并按新快照审阅成果；不原地修改摘要 |
| 结果代变化 | `WORK_RESULTS_CHANGED` | 查明非合同写入，使用可信成果重新导出/提交 |
| 缺项、结构错误或质量 blocker | 现有 `writeback_gate.decision=deny`；接收时结构错误为 `WORK_STRUCTURE_BLOCKED` | 补齐、订正后重新检查和预览 |
| 已写文件但状态未记完 | `writeback=recovery_required` | 核对当前事实后重放 `work-apply`；第三方后续改动不会被覆盖 |

严格模式下，成功接收、当前状态和写回返回 0；状态查询发现恢复待办返回 3、未决冲突返回 4、
stale 或合同不匹配返回 5。缺少 CLI 参数沿用 2。`check` 沿用原合同：allow 且无报警为 0，
allow 且有 warning 为 3，deny 为 4。`ok=true` 只表示查询成功，须同时读取状态及门禁。

## 唯一检查与写回边界

`check` 使用现有 Batch 检查服务，检查尝试开始即撤销旧授权。指纹额外绑定工作包、
结果代和提交版本。`work-preview` 要求最近一次 check=allow，再使用已有完整文件预览
服务生成源快照、diff、结构写回计划与质量报告。`apply` / `work-apply` 消费该绑定预览，
使用现有 `sync_translation_preview` / `atomic_io` 写回事务；没有外部专用文件 writer。

共享预览服务复核项目、最近 check、成果、参考、全部源快照、预览制品和写回计划。
事务绑定目标 preimage 并在提交边界复核输入；force 不绕过任何约束。质量 warning 默认
不阻断，配置提升为 blocker 的规则仍阻断。包内尚有缺项时禁止预览/写回。

文件已提交而状态补记失败时，状态查询根据绑定预览和实际文件摘要显示
`recovery_required`；重放 apply 只补记可验证事实。第三方随后修改文件时必须拒绝恢复，
不能重新覆盖。预备事务仍由既有事务恢复器校验并回滚。聊天记录不是恢复依据。

## 产品范围

这是 issue 明确允许的高级/脚本 CLI 例外。GUI 复用诊断命令参考与现有质量报告；
不增加工作包编辑器、第二套审校 GUI 或 #427 的新前置条件。当前写回后订正仍可使用
现有 `export-revision-corpus` / `import-revision-proposals` / `apply-revisions`。

## 原创 fixture 复现与工程结论

公开样本为 [chapter.rpy](../tests/fixtures/external_work/chapter.rpy)，6 个独立 occurrence，
含重复原文、角色区分、插值、格式标签和菜单。以下命令只在新输出目录创建隔离副本：

```powershell
python -B scripts/run_external_work_fixture.py --output-dir outputs/external-serial
python -B scripts/run_external_work_fixture.py --output-dir outputs/external-parallel --prepare-only
```

第二条输出互斥 ID 分配；两个独立执行者各准备一份上述 submission JSON 后接收：

```powershell
python -B scripts/run_external_work_fixture.py --output-dir outputs/external-parallel --resume --submission outputs/external-parallel/agent-a.json --submission outputs/external-parallel/agent-b.json
```

[复现脚本](../scripts/run_external_work_fixture.py) 通过真实 CLI dispatch 执行导出、部分接收、
状态/材料读取、重放、check、preview、apply 和 apply 重放；串行样例还做一次候选订正。
脚本拦截 socket 连接及项目模型入口，未观察到的宿主模型、用量和费用均为 `unknown`。
`run_report.json` 的 `cli_elapsed_seconds` 仅计脚本处理时间，不含代理生成、准备与人工审阅，
不能用来比较串行/并行翻译速度。运行结果位于被 Git 忽略的 `outputs/`，不提交成果 JSONL。

2026-09-21 的工程验证中，串行固定候选及两个独立子代理各 3 项的成果均收到 6/6，剩余 0；
重复原文没有合并，插值和标签保留，check 允许写回，写回重放没有重复改写。子代理候选的
语义审校仍标为未审；一个风格建议明确 deferred，未改共享术语文件。
这轮包括协调者准备材料、分配 ID、接收和检查；没有测量端到端人工介入成本。

[合同测试](../tests/test_external_translation_work.py) 另覆盖竞争提交、stale、接收回执中断、
事务中断、写回状态补记失败、第三方编辑保护、质量 warning/blocker、glossary 版本，以及
现有订正提案导入/写回的互操作。空 TL 槽的写回预镜像复用 Adapter 的实际 catalog 文本，
原文仅用于翻译输入；没有放宽原文或目标文件校验。

本次工程门禁（Python 3.14，Windows）结果：

| 验证 | 结果 |
|---|---|
| `python -B -m unittest tests.test_external_translation_work -q` | 33 项通过，已包含在完整 CLI 集合中 |
| `python -B tests/run_cli_tests.py -q` | 共 2650 项，运行成功（4 项跳过） |
| `python -B tests/run_gui_tests.py -q` | 1323 项通过 |
| `scripts/run_quality_gates.py all` 的全部门禁 | Ruff、mypy 与五份锁的 pip-audit 均通过，使用现有审计例外 |
| 本地 Markdown 链接与 `git diff --check` | 文件目标与差异空白检查通过 |

依赖审计的复跑仅在该进程指定工作区内缓存并对 PyPI 绕过本机代理，解决缓存权限及代理
断连；未修改依赖锁、审计例外或门禁策略。

**当前决定：继续保留实验 CLI，暂不扩大产品化范围。** 工程证据支持可持久化、可恢复并
复用现有安全门禁；尚不能证明减少真实章节遗漏、返工或改善翻译质量。
[设计 §9.2](plans/agent_translation_toolkit.md#92-产品实验) 的 A/B1/B2/C 对照仍待获准项目
副本、章节及跨章节片段、目标语言/风格、宿主与项目模型、额度上限。随后仍需独立清单、
匿名译文审阅、游戏运行/字体检查，以及人工介入、重试、遗漏和可得用量记录。
模型不一致时只能比较整条路线；宿主费用不可得时继续写 `unknown`。原创 fixture 不替代
真实质量结论，#508 的真实章节验收仍未完成。
