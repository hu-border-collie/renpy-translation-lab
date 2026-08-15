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

可在 `translator_config.json` 的 `batch.non_chinese_validation` 中覆盖或追加路径；GUI 高级设置提供「非中文白名单追加路径」。默认白名单已覆盖真实项目烟测中遇到的常见 UI / 制作人员名单路径。

若目标语言为日语、韩语等，`No Chinese characters` 失败可能属于预期行为，需要调整校验策略或白名单。换语言前请先跑 `doctor` 确认 TL 路径，并在 `check` 结果中逐项确认失败原因。

`allow_non_chinese_batch_translation` 在单次校验调用内会缓存 TL/source 文件读取，避免 short-circuit OR 链重复打开同一文件；pass/fail 结果与缓存前一致。

### 校验的语义边界（2026-08-06 记录）

`check` 的 `safe` 只表示**结构性写回安全**（源文本匹配、标签/占位符完整、含中文字符等），**不表示译文内容质量合格**。2026-08-06 在真实项目 Chapter 4（约 6,062 块）的审校中发现，初译直出即使 `check=safe` 仍存在系统性文本问题：

- `{w=...}` 标签插进中文词语中间，全文件约 241 处；
- 中文与英文/数字紧贴缺空格，全文件约 339 处；
- 已知机器错乱词残留，全文件 7 处实锤；
- Ren'Py 标签后的英文词缀被吞进中文（如法术名后残留的复数/过去式词尾），全文件 2 处。

结论与建议：

1. 直出文本只能作为草稿，交付前必须经过“机械质量门禁 + 人工/LLM 通读审校”两层处理。
2. 机械门禁建议新增：汉字+标签+汉字检测、CJK/拉丁字符间距检测、已知错乱词黑名单、`[[法术名]]` 英文词缀保护；全部归零才算通过。
3. 语义层（错译、反讽丢失、术语取舍、语气）无法由脚本保证，仍需逐块通读。

## 命令说明

普通用户推荐通过 GUI 执行这条流程；Agent、脚本和 CI 可在七个核心命令后追加 `--output json`，获得 `schema_version=1` 的统一结果 envelope：

```powershell
python gemini_translate_batch.py check logs/batch_jobs/<package>/manifest.json --output json
```

JSON 模式下 stdout 只包含结果文档，原有 banner、进度、prepare 子进程输出和诊断文本实时进入 stderr。`result` 提供业务摘要，`artifacts` 提供 manifest / results / 报告路径，`status` 表示 job state 或 `safe / warn / block` 等业务状态。文本模式和默认退出码保持兼容；仍须根据 `status` 判断是否可以继续写回。

### Parser-level 错误

如果原始参数中包含精确的 `--output json` 或 `--output=json`，但 argparse 在生成参数 Namespace 前失败（例如未知参数、缺少参数值或非法 choice），CLI 会在 stdout 输出一个 schema v1 的错误 envelope：`error.code=ARGUMENT_PARSE_ERROR`，退出码为 `2`，并把原生 argparse usage / error 诊断保留在 stderr。若已识别出合法子命令，envelope 的 `command` 使用该名称；最早期无法识别子命令时使用 `command=cli`。

这种失败发生在 workflow、`--output-file` 安全探测和字段投影之前，因此错误 envelope 始终回退到 stdout，不会尝试写入不完整参数中的输出路径。机器模式的识别边界是有意收窄的：未出现精确 JSON 输出标记（包括 `--strict-exit-codes` 单独出现）、`--output` 缺少值，或 `--output` 使用非 `json` 值时，继续使用普通 argparse 文本错误和退出码 `2`。扫描遇到 `--` 后停止；其后的参数全部按 positional 数据处理，不会再触发 JSON 或 `--compact` 识别。

Agent 可追加 `--strict-exit-codes`（必须与 `--output json` 同时使用），启用稳定的语义退出码：`0` 成功/继续轮询，`1` 未分类内部错误，`2` 用法错误，`3` 需要处理（例如 `warn`，或 reconciliation 的 `attention`），`4` 被安全门禁阻止或任务终止失败，`5` 输入/配置/状态失效，`6` 远端临时错误、可稍后重试。例如：

```powershell
python gemini_translate_batch.py check logs/batch_jobs/<package>/manifest.json --output json --strict-exit-codes
```

严格模式下也不能只看退出码：必须同时读取 envelope 的 `ok`、`status` 和 `error`。job pending/running 是成功查询，退出 `0`；`check` 只有 `safe` 才退出 `0` 并允许进入 `apply`。错误时优先使用稳定的 `error.code`、`retryable`、`suggested_action` 与权威的 `details.semantic_exit_code`，不要解析自然语言 `message`。


需要确定性调用时追加 `--non-interactive`。该选项保证核心命令不等待 stdin，并让 `submit / status / download / check / apply` 必须显式接收 manifest 或 package target；因此不会读取 latest manifest，`submit` 也不会隐式 build。缺少 target 时 JSON envelope 返回 `EXPLICIT_TARGET_REQUIRED`，配合 `--strict-exit-codes` 退出 `5`。

```powershell
python gemini_translate_batch.py apply logs/batch_jobs/<package>/manifest.json --output json --non-interactive --strict-exit-codes
```

只需关闭 target 回退时可单独使用 `--require-explicit-target`。默认模式保持现有 latest-manifest 和 submit-build 行为；`doctor / build` 不消费 manifest，不受显式 target 要求影响。
结构化输出当前覆盖 `doctor / build / submit / status / download / check / apply`；其它命令继续以各自帮助和落盘 JSON/JSONL 为准。

机器发现使用 `capabilities` 与 `schema <command>`；两者在加载项目配置前直接输出 JSON。`capabilities.commands` 已提供完整命令索引，因此不另设重复的 `commands`。单命令 schema 从当前 argparse action 动态生成，包含参数类型、required、repeatable、choices、默认值和帮助文本，避免文档与实际 parser 漂移。
核心 JSON 命令可用 `--compact` 压缩序列化、用 `--fields status result.check.safety_level` 按点路径保留必要字段，或用 `--output-file <path>` 将最终文档原子写入文件并保持 stdout 为空。三者只接受显式 `--output json`；裁剪不影响业务状态和严格退出码，文件结果会在未被投影掉时记录 `artifacts.output_file` 绝对路径。空路径或连续点等非法字段路径会在 workflow 执行前返回 `INVALID_FIELD_PATH` 和退出码 `2`。
输出文件会在 workflow 前进行可写性探测。若创建或原子替换失败，stderr 保留诊断，stdout 回退为未投影的 `OUTPUT_FILE_WRITE_FAILED` envelope；严格模式退出 `5`，兼容模式退出 `1`。必须同时读取 `error.details.workflow_started` 和 `error.details.command_completed`：分别区分 workflow 未启动、已启动但以错误结束、以及已成功完成后仅结果文件落盘失败；后两种状态都应按原命令可能已产生业务副作用处理。
`capabilities / schema` 也支持这三个选项；它们原生输出 JSON，因此无需 `--output json`。


Discovery schema 与核心结果 envelope 当前都使用 `schema_version=1`，但通过顶层 `type`（`capabilities` / `command_schema`）区分用途。

- `gemini_translate_batch.py` 需要显式子命令；不带子命令会打印帮助并退出。
- Batch 产物默认写到 `logs/batch_jobs/<package>/`。
- `doctor` 只检查当前 `game_root` / `tl_subdir`、SDK/launcher、TL 模板和 `old/new` / 剧情块形态，不调用 Gemini，也不会写回 `.rpy`。
- `probe` 会用同步请求做最小 smoke test；每个被抽样的 request row 必须能对应当前 manifest 中的非空 chunk，否则会在调用 Provider 前拒绝并提示重建 package，避免把过期或损坏的请求误判为成功。
- `check` 是干跑校验，不会修改 `.rpy`；它会把当前 manifest、results、目标 item 形状和 check contract version 写入 `last_check_summary.check_fingerprint`，输出 `safe / warn / block` 安全等级，并在包目录写入 `check_failures.jsonl`。
- `apply` 默认要求最近一次 `check` 对应当前 manifest/results，且安全等级必须是 `safe`；未 check、results 变化、manifest item 变化、`warn` 或 `block` 都会拒绝写回。
- `--force` 只绕过“manifest 已经 apply 过”的重复写回保护，不会绕过 stale check、source snapshot 校验或 `block`。
- `apply` 写回前会再次校验当前源文本；如果 apply 阶段发现漂移，会拒绝写回并在包目录写入 `apply_failure_report.json` / `failures.jsonl`。
- 当 `rag.enabled=true` 时，`split` 更接近“静态快照拆包”，不是动态波次式 RAG 工作流；后续包的回灌结果不会自动回流到已经 split 完的旧包。

### 提交前估算与异常恢复

`build` 会把当前定价配置下的估算写入 manifest；提交前也可显式复算并查看 token / 成本上限：

```powershell
python gemini_translate_batch.py estimate-cost logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py submit logs/batch_jobs/<package>/manifest.json --max-cost <上限>
```

`estimate-cost` 是估算，不是供应商最终账单；实际 usage 与可用成本应在下载后查看 [模型用量账本](model_usage_ledger.md)。`submit --max-cost` 在估算最大成本超过显式上限时拒绝提交。

提交进程若在“远端 job 已创建、manifest 尚未记入 job”之间中断，再次提交会检测 journal 并阻止可能的重复 job。按提示先恢复：

```powershell
python gemini_translate_batch.py recover-submit logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py status logs/batch_jobs/<package>/manifest.json
```

默认恢复会先验证远端 job，再把 journal 中已创建的 job 写回 manifest。只有已经独立核实远端状态、且明确接受跳过验证风险时才使用 `recover-submit --no-verify`。如果 journal 只记录到上传完成而尚未创建 job，则使用 `submit <manifest> --resume` 继续；`submit --force` 会开始新的提交尝试，不能作为不确定状态下的默认恢复手段。

## 实际模型用量

Batch 下载、同步关键词/订正、普通同步翻译、repair、probe、A/B 与项目分析会把 provider 返回的实际 usage metadata 汇总到当前项目的本地账本。该旁路统计不改变 Batch 状态，也不放宽 `check -> apply` 写回门禁。

```powershell
python gemini_translate_batch.py usage-import logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py usage-report
python gemini_translate_batch.py usage-report --provider gemini --group-by task,stage,model --json
```

- `usage-import` 只读取已下载结果，可重复执行；不会联网或重新调用模型。
- retry / split 使用 provider response identity 和结果 fingerprint 幂等去重；合并 retry 后会展开原始结果 lineage，不把本地合成行算作新调用。
- token 或 cost 未返回时保持未知，不会显示成 0；估算成本与 provider 报告的实际成本分列。
- 账本按 `game_root` 隔离，默认写到 `<game_root>/translation_usage/usage_ledger.json`。

完整字段、过滤器、成本与隐私语义见 [实际模型用量账本](model_usage_ledger.md)。

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

`build-revisions` 会复用 include 过滤、glossary、macro setting、可选 RAG / Story Memory，把已有原文和当前译文送入 Batch。`preview-revisions` 导出 `revision_preview.jsonl` 和 `revision_preview.md`，并把合同绑定写回 manifest：结果文件 SHA-256、manifest / 项目身份指纹、涉及源文件的快照指纹、preview schema 版本与生成时间。`apply-revisions` 必须找到有效且匹配的 preview 才允许写回；`--force` 只能绕过「已写回」守卫，不能绕过 preview 缺失、结果被替换、项目变化或源文件快照变化。

`apply-revisions` 的终态写在 manifest 的 `revision_apply_state`，固定区分：

- `applied`：全部可写回项已成功写回；
- `no_op`：有效 preview 没有需要修改的内容（不写 `revision_applied_at`）；
- `blocked`：没有发生写回且存在阻断（preview 缺失/过期、适配器写回计划不安全，或全部条目被跳过/源不匹配/校验失败，不写 `revision_applied_at`）；
- `partial`：部分写回、部分跳过/失败（仅真实写回的行计入 applied）。

`revision_applied_at` 只在 `applied` / `partial` 时写入；`no_op` / `blocked` 不会把 final-review finding 错误标记为已应用。

重新运行 `preview-revisions` 会清空旧的 apply 终态（blocked/no_op/partial）并重新打开写回闸门；若此前已真实写回过，旧写回记录会移到 `revision_apply_history`，`revision_applied_at` 不再拦截新的 preview/apply 流程。

阻断的退出语义：preview 校验类阻断（preview 缺失、结果/项目/源快照变化）与 `adapter_writeback_block` 以非零退出码结束；有效 preview 下全部条目被跳过/源不匹配/校验失败时的 `blocked` 以零退出码结束，但 machine envelope 的 `status=blocked` 且 manifest 写有 `revision_apply_blocked_reason`。依赖退出码的自动化应解析 `revision_apply_state` / `status`，不要仅凭零退出码判定写回成功。

当前 `safe / warn / block` 强制闸门只覆盖普通 translation manifest 的 `check/apply`；订正写回仍走 `preview-revisions -> apply-revisions` 的独立快照校验。

`sync-revisions` 复用订正 prompt、schema、RAG / Story Memory 注入、预览报告和写回前源快照校验；默认只预览，传 `--apply` 才调用 `apply-revisions` 写回。

## 最终审校（report-only → 人工选择 → 订正预览）

最终审校先以**独立 campaign** 批量发现问题，默认 **report-only**：不调用 autofix，也不直接写 `.rpy`。只有用户明确选择的 findings 才会转成普通 revision manifest，并强制走 `preview-revisions → apply-revisions`；模型不能声称问题已经修复或写回。

```bash
# 构建 campaign：完成度闸门 + 冻结上下文 digest + review units + requests.jsonl
python gemini_translate_batch.py final-review-build
python gemini_translate_batch.py submit logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py status logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py download logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py final-review-ingest-results logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py final-review-status logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py final-review-export logs/batch_jobs/<package>/manifest.json

# 续跑：跳过 digest 未变的 done unit；--force 全部重审
python gemini_translate_batch.py final-review-resume logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py submit logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py status logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py download logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py final-review-ingest-results logs/batch_jobs/<package>/manifest.json
python gemini_translate_batch.py final-review-status logs/batch_jobs/<package>/manifest.json
# 需要全部重审时，把第一条 resume 改为：final-review-resume <manifest> --force

# 将明确选择的问题转换为订正候选并立即生成预览（--finding-id 可重复）
python gemini_translate_batch.py final-review-create-revisions logs/batch_jobs/<package>/manifest.json --finding-id <finding-id>
# 确认预览后，仍使用现有安全写回入口
python gemini_translate_batch.py apply-revisions logs/batch_jobs/<revision-package>/manifest.json
```

通用 `status` 必须轮询到远端 job 成功后才能 `download`；pending/running 不是失败，也不能跳过轮询直接下载。`final-review-status` 查看的是 campaign 内 review unit / finding 生命周期，不能替代远端 job 的通用 `status`。`final-review-resume` 若报告 `Units to run: 0`，说明当前 digest 已是最新，不应再次 submit；若有待跑 unit，则必须完整执行 `submit -> status -> download -> final-review-ingest-results`，不能复用 resume 前的 `results.jsonl`。

### 启动闸门

- 范围内必须有可审校的已译条目（原文 + 当前译文）。
- 默认 `batch.final_review.require_zero_pending=true`：范围内若仍有未完成翻译，`final-review-build` **拒绝启动**并打印可操作原因（pending 条数、示例文件）。
- 可用 CLI `--allow-pending` 显式覆盖（不推荐；结果可能不完整）。
- 可用现有 include 过滤缩小审校范围。

### 快照与 digest

构建时冻结并哈希：

- 当前译文集合（identity + source + current_translation）
- glossary / macro setting（启用内容）
- 可选 Story Memory / Source Index（仅当启用）
- 可选已启用且可注入的 published Project Analysis fingerprint（无 PA 时允许仅靠其它上下文运行）

每个 review unit 保存 `input_digest`（本 unit 的 `items_digest` + **共享** `context_digest` + model + prompt schema）。`context_digest` 只绑定 glossary / macro / Story Memory / Source Index / 可注入 PA brief 等共享上下文；**不**把全项目 `translations_digest` 塞进 unit，因此改 A 文件不会让 B 文件的已完成 unit 变 stale。全量译文审计摘要仍记在 campaign 级 `snapshot_digest`。构建时会把可注入的简要上下文冻结到 `snapshot.prompt_context`（宏设定 / PA brief / glossary 词条），并写入每条 request 的 user prompt。

`final-review-resume` 会**重新采集 live 共享上下文**（而非只读冻结 snapshot）来判断 skip / stale，只为 pending / stale / failed 重建 `requests.jsonl`；`--force` 才重审全部。有待跑 unit 时会清空 `job_name` / 下载字段，并把旧 `results.jsonl` 改名为 `results.jsonl.pre_resume_*`，避免 `download` 短路复用上一轮结果。`final-review-ingest-results` 解析 Batch 结果：成功（含空 findings）→ `done`，并在成功时写回本次 live `input_digest`；解析失败 / 缺响应 → `failed`（**不会**记成「零问题 done」）。resume 之后若尚未重新 download，默认**拒绝**用 resume 前的 `results.jsonl`（可用 `--result` 或 `--allow-stale-results` 显式覆盖）。

当前 ingest 的固定 fixture 基线发现两个现行误接受边界：finding 缺 response schema 必填字段、完全重复 finding；另单列一个未来 completion receipt 数量不符的假设性探针。它们尚未进入生产 parser；本阶段也不启用宽松 JSON repair。现状对比、候选稳定码、unit 级 targeted resume 合约和 receipt 暂缓结论见 [Final Review 结果失败分类 fixture spike](plans/final_review_result_failure_spike.md)。

### 产物布局

```text
logs/batch_jobs/<ts>_<project>_final_review/
  manifest.json          # mode=final_review, report_only=true
  snapshot.json          # context_digest + snapshot_digest + prompt_context + 各层摘要
  review_units.jsonl     # unit 状态 / input_digest / items
  requests.jsonl         # Batch 请求（build / resume 写入）
  results.jsonl          # download 后的模型结果（resume 会作废上一份）
  findings.jsonl         # 审校发现（执行后填充；build 时为空）
  report.md              # 人类可读报告
```

### 选择、状态与 GUI

- 不自动修改 `.rpy`，无模型直接声称 `fixed` / `applied`。
- 不把回译抽检当作写回安全闸门。
- 同一译文上的多个 finding 若给出冲突建议，转换会拒绝，要求用户只保留一种建议。
- 转换时重新校验 campaign 完成状态、review unit digest、项目 identity、原文和当前译文；任一已变化就拒绝并要求先续跑最终审校。
- finding 状态只由真实生命周期推进：成功建包为 `candidate`，成功预览为 `previewed`，实际写入该 identity 后才是 `applied`。被二次校验跳过的条目不会误标已写回。
- GUI 位于「订正」页面的「终审」子模式，复用统一的开始 / 继续 / 停止、云端状态和写回交互。报告完成后用“选择问题并生成预览”表格人工勾选，不占用「上下文库」。

最终审校生成的 revision manifest 带有来源 campaign、snapshot digest 和 finding digest；后续预览 / 写回会验证这些 provenance，防止把被编辑或属于其他项目的报告状态写回。

相关配置见 `translator_config.example.json` 的 `batch.final_review`。

## 润色语料导出

`export-revision-corpus` 是只读导出命令（#318 P1 / #320）：把当前项目中全部
revision old/new 对照导出为 JSONL、Markdown 和 manifest，供人工线性通读或
Agent 分批起草润色提案：

```bash
python gemini_translate_batch.py export-revision-corpus
python gemini_translate_batch.py export-revision-corpus --output-dir C:/tmp/corpus
python gemini_translate_batch.py export-revision-corpus --output json
```

- 默认写入 `logs/batch_jobs/<timestamp>_<project>_revision_corpus/`；
  `--output-dir` 可指定其他目录。
- 导出范围与当前配置一致：应用 `include_files` / `include_prefixes` 过滤后
  的 TL 文件；不是无条件导出项目内全部 revision 对。
- 产物：`revision_corpus.jsonl`（权威结构化合同）、`revision_corpus.md`
  （线性通读报告）、`revision_corpus_manifest.json`（项目身份、scanner/schema、
  源快照 digest、范围计数）。
- 每个 item 携带 schema version、`identity_v2` occurrence、文件相对路径、
  locator（行号 / 起止 / ordinal）、speaker（可得时）、原文、现译和
  old/new 快照 digest；重复原文保留不同 occurrence，不按文本去重。
- 文件与 item 顺序跨运行稳定（文件按相对路径排序，item 保持文件内顺序）；
  相同输入、adapter 和配置产生相同 item 集合与 digest。
- 导出前后会重算 TL 文件 digest，扫描期间源文件变化会在 manifest 中标记
  `source_changed_during_scan`，而不是静默混入。
- 只读：不修改 `.rpy`、manifest、glossary 或 RAG；机器输出可用
  `--output json`（envelope 含三个产物路径与 item/file 计数）。

### 导入人工 / Agent 润色提案

权威输入必须是结构化 JSONL；自由格式 Markdown 只能作为伴生审阅报告。导入会把
通过校验的提案转换成标准 `mode=revision` 本地候选包，并立即运行
`preview-revisions`，自身绝不修改 `.rpy`：

```powershell
python gemini_translate_batch.py import-revision-proposals C:/review/proposals.jsonl
python gemini_translate_batch.py import-revision-proposals C:/review/proposals.jsonl `
  --corpus-manifest C:/review/revision_corpus_manifest.json `
  --output json --strict-exit-codes --non-interactive
```

每行 proposal schema v1 至少包含：

```json
{
  "schema_version": 1,
  "occurrence_id": "identity-v2-value",
  "identity_v2": "identity-v2-value",
  "file_rel_path": "chapter01/revisions.rpy",
  "source": "原文",
  "current_translation": "当前译文",
  "proposed_translation": "建议译文",
  "reason": "修改理由",
  "selected": true,
  "disposition": "accepted",
  "producer": {"type": "human", "tool": "optional", "model": "optional"},
  "snapshot_digest": "该条 source/current_translation 的导出摘要",
  "corpus_snapshot_digest": "revision_corpus_manifest.json 的 source.snapshot_digest"
}
```

`producer.type` 只接受 `human` / `agent`。若 proposal 同目录存在
`revision_corpus_manifest.json`，可省略逐行 `corpus_snapshot_digest`；也可用
`--corpus-manifest` 显式指定。未知/重复/冲突 identity、项目或语料快照 stale、
source/current translation 变化、空建议、Ren'Py 标签/变量破坏、adapter 校验失败或
不安全写回计划都不会获得写回资格。

状态固定区分 `imported`（内部导入阶段）、`previewed`、`no_op`、`partial`、
`blocked`、`stale`。机器 envelope 同时返回稳定 `status`、diagnostics、artifacts 和
`suggested_action`；严格退出码下 `partial` 为“需处理”，`blocked/stale` 为“阻断”。
只有 `previewed/no_op` manifest 能进入 `apply-revisions`，`--force` 不能绕过此闸门，
也不能绕过 preview/result hash、项目身份、源快照或 adapter writeback 校验。

## 项目版本快照与只读 reconciliation

本节对应 `#265` P3 / `#330`，增加两个不调用模型、不要求 API Key 的高级命令：

```bash
python gemini_translate_batch.py export-project-snapshot --version-id 1.4.0
python gemini_translate_batch.py reconcile-project-snapshots \
  C:/snapshots/1.3.0/project_snapshot.json \
  C:/snapshots/1.4.0/project_snapshot.json
```

- `export-project-snapshot` 复用当前 Ren'Py adapter 的 discovery / inventory /
  coverage / occurrence 结果，输出 `project_snapshot.json` 与
  `unit_occurrences.jsonl`；默认目录为 `logs/project_snapshots/`。
- 快照只保存原文 occurrence、opaque locator、speaker、上下文及 coverage/review
  dependency digest，不保存当前译文；可用 `--coverage-review` 导入并校验已有核对记录。
- `reconcile-project-snapshots` 只读取两个保存的 artifact，输出
  `reconciliation_report.json` 与 `reconciliation_items.jsonl`；默认目录为
  `logs/project_reconciliations/`。
- 匹配报告区分 confirmed lineage、locator/content exact、移动、上下文高置信、
  原文小改、新增、删除与歧义；每个未决目标会以 `ambiguous_target` 独立列出，并用
  共享 ambiguity group ID 连接对应的候选组，候选摘要截断时也不会丢失目标；歧义
  不会自动确认，也没有写回入口。
- 两个命令都支持 `--output json` 及通用机器输出裁剪参数。完整 schema、digest、
  stale 与 P4/P6 边界见 [Engine Adapter 与覆盖审计](engine_adapter.md#p3-项目版本快照)。

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

人工确认后，可用 `merge-keywords-to-glossary` 把 `keyword_candidates.jsonl` 中的候选追加进 `glossary.json`（默认写入 `normalize_map`；`source` 与 `suggested_target` 相同时写入 `preserve_terms`）：

```bash
python gemini_translate_batch.py merge-keywords-to-glossary logs/batch_jobs/<package>/keyword_candidates.jsonl
python gemini_translate_batch.py merge-keywords-to-glossary logs/batch_jobs/<package>/manifest.json --dry-run
python gemini_translate_batch.py merge-keywords-to-glossary logs/batch_jobs/<package>/manifest.json --accept-confidence 0.85 --yes
```

- 默认逐条 `y/n` 确认；`--accept-confidence` 可半自动接受高置信候选，`--yes` 跳过交互。
- `--min-confidence` 过滤低置信候选；已有 `source` 默认不覆盖，需 `--overwrite` 才改目标译法。
- `--dry-run` / `--preview` 只预览 diff；真实写入前会生成 `glossary.json.bak-<timestamp>` 备份（可用 `--no-backup` 关闭）。

订正 manifest 的 `mode=revision`，关键词 manifest 的 `mode=keyword_extraction`，普通 `check/apply` 会拒绝处理，避免把非翻译结果误写回 `.rpy`。

## 翻译质量 A/B 实验

`compare-variants` 用同一批 manifest chunk 在**同步模式**下跑多个配置变体，生成并排 Markdown 报告，**不会写回** `.rpy` 或 `glossary.json`。适合比较 Story Memory、RAG、macro setting 等上下文层对译文的影响。

图形界面入口见 [GUI 工作台 · 翻译 A/B 对比](gui_workbench.md#翻译-ab-对比)：在「诊断与运行日志」页工具栏打开，通过对话框选择 baseline 与 Story Memory / RAG / 原文索引的强制开/关变体，无需手写 `variants.json`。

```bash
python gemini_translate_batch.py compare-variants logs/batch_jobs/<package>/manifest.json \
  --variants-file experiment_variants.json \
  --limit 3 \
  --offset 0
```

`experiment_variants.json` 示例：

```json
[
  {
    "name": "baseline",
    "overrides": {
      "batch": {
        "story_memory": { "enabled": false }
      }
    }
  },
  {
    "name": "story_memory",
    "overrides": {
      "batch": {
        "story_memory": { "enabled": true }
      }
    }
  }
]
```

- 变体文件至少需要 **2 个**命名变体，以便并排比较。
- `--dry-run` 只重建各变体 prompt 并写报告，不调用翻译 API，也不会触发 RAG / source index / story memory 检索。
- 默认输出到 `logs/experiments/<timestamp>_ab/`，包含 `ab_report.md`、`ab_results.jsonl`、`ab_settings.json`。
- API 用量约为 `chunks × variants` 次同步请求；先用小 `--limit` 试跑。

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
