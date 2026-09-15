# #488 翻译预检最小增量字段合同

> **状态**：设计合同（待实现）。本文件冻结 #488 的“当前字段 + 最小新增字段 + 数据来源 + freshness / unknown 合同”；
> 实现与界面以代码和现行手册为准。
> **基线**：`main@1ddd0be`；研究来源：`docs/plans/github_localization_projects_research.md` §6.1.5、§8。
> **关联**：#488、#348、#424、#457、#364。

## 1. 范围

扩展已有 CLI `translate-preflight`（`gemini_translate_batch.py:22592`）与 GUI 启动确认，不新建第二套预检。
预检保持本地只读：不调用 provider / embedding、不执行 prepare、不改变游戏文件、配置、manifest 或检查授权状态。
本文件只冻结“最小增量字段”；字体检查、dashboard、实时网络定价、历史用量等不在范围内（见 §6）。

## 2. 当前 payload（基线）

`_run_translate_preflight()` 从 `gemini_translate_batch.py:22795` 起返回：

| 字段 | 现状 |
| --- | --- |
| `schema_version` / `status` / `strategy` / `plan_contract` | 固定合同字段；`status` 仅由 error 级 risk 决定 |
| `profile` | `id` / `label` / `model` / `adapter` / `provider`（已解析的 route profile） |
| `project` | `root` / `tl_dir` / `target_language` / `file_count` |
| `counts` | `files_with_pending` / `pending_items` / `chunks` |
| `chunk_policy` | `max_items` / `max_chars` |
| `source_snapshot` | engine / adapter_version / project identity digest / source fingerprint / file_count |
| `context_sources` | RAG / source index / story memory / PA brief / local context / macro file |
| `credential_available` / `risks` / `environment` | 凭据可用性、风险列表、运行环境 |

零待译 coverage 合同在 `:22761` 起：`COVERAGE_UNCONFIRMED`（error）、`COVERAGE_EVIDENCE_MISSING`（warning）、
`NO_PENDING_WORK`（info）。GUI 消费同一 payload（`gui_qt/app.py:11349`，文案 `gui_qt/user_copy.py:1207`）。

## 3. 最小新增字段

保持 `schema_version: 1`：三个新键均为**可选增量**，现有消费者忽略未知键即可；字段缺失不得被解释为“通过”或“零值”。

### 3.1 `cost`

```json
"cost": {
  "status": "known | unknown",
  "model": "gemini-3.5-flash",
  "strategy": "sync",
  "currency": "USD",
  "input_tokens": 12345,
  "output_tokens_max": 6000,
  "estimated_cost_min": 0.0123,
  "estimated_cost_max": 0.0456,
  "pricing_version": "…",
  "pricing_source": "translator_config | defaults",
  "scope": "current_plan",
  "excluded": ["provider 排队与重试", "未计入的实际输出"]
}
```

- **来源**：当前 TranslationPlan 的 request 文本字符数（复用 `batch_cost_estimate` 的 token 估算规则）+
  所选 strategy 的 `max_output_tokens`（sync 用 `legacy.MAX_OUTPUT_TOKENS`，batch 用 `BATCH_MAX_OUTPUT_TOKENS`）+
  `batch_cost_estimate.resolve_model_pricing(profile.model)` / `load_pricing_config`（`batch_cost_estimate.py:48/95`）。
- **known**：模型价格与 token 估算均可用；`estimated_cost_min` 只计输入，`estimated_cost_max` 含输入 + 最大输出。
- **unknown**：模型无价格、请求文本不可读或 token 估算缺失；此时不返回数字，**不得显示 0 或“免费”**。
- 只覆盖当前 plan 的初译请求，不含 final review / repair / embedding；不调用 provider、不执行 prepare。

### 3.2 `coverage`

```json
"coverage": {
  "status": "ready | attention | blocked | unknown",
  "completion": "confirmed | unconfirmed | unknown",
  "coverage_digest": "…",
  "candidate_count": 123,
  "classification_counts": {"translatable": 100, "unknown": 1},
  "reason_counts": {"…": 1},
  "unknown_count": 1,
  "parse_error_count": 0,
  "unsupported_count": 0,
  "reasons": ["…"],
  "gate": {"…": "…"},
  "review_status": "…",
  "scope": "current_scan"
}
```

- **来源**：同一次预检扫描的 adapter snapshot（`context.adapter_snapshot`），复用
  `summarize_coverage_for_doctor`（`gemini_translate_batch.py:3529`）与 `evaluate_coverage_completion`；
  不重新扫描、不另造放行语义。
- 零待译分支继续复用同一 gate（`:22761`）：阻断判定与摘要来自同一份证据。
- **unknown**：扫描证据缺失时 `status=unknown`、`completion=unconfirmed`；不得显示“覆盖完成”。
- `scope` 固定 `current_scan`，表示摘要与本次预检扫描同源；跨项目切换后必须重算，不得缓存复用。

### 3.3 `quality_summary`

```json
"quality_summary": {
  "status": "available | stale | not_available | unknown",
  "source": "latest_manifest | none",
  "manifest_path": "…",
  "report_path": "…",
  "finding_count": 0,
  "severity_counts": {"error": 0, "warning": 0, "info": 0},
  "reason_counts": {"…": 0},
  "generated_at": "…",
  "matched_by": ["project", "plan_fingerprint"]
}
```

- **来源**：已有质量报告的 manifest 引用（`last_quality_findings_path` / `last_check_report_path`）+
  `quality_report_export.load_quality_findings`（`quality_report_export.py:102`）。
- **available 必须同时满足**：
  1. manifest `mode == translation`；
  2. manifest `base_dir` / `tl_dir` 与当前项目一致；
  3. manifest `translation_plan.plan_fingerprint` == 当前 plan fingerprint（无 `translation_plan` 的旧 manifest 不匹配）；
  4. 报告文件存在；若 manifest 记录了 sha256，则实际内容需匹配。
- **stale**：找到 manifest 但任一匹配条件不满足；**not_available**：没有可用 manifest / 报告；
  **unknown**：读取失败或报告格式不可解析。
- 不启动模型审校、不把 warning 当作“质量通过”；初译前无报告时绝不显示“质量通过”。
- 仅按文件时间或 latest 指针不能认定匹配；项目、任务、源/结果或 profile 切换后不得沿用旧摘要。

## 4. freshness / identity 合同

| 摘要 | 证据来源 | fresh 判定 | 失效条件 |
| --- | --- | --- | --- |
| `cost` | 当前 plan + 当前 pricing config | 每次预检重新计算 | plan / profile / pricing config 变化即重算 |
| `coverage` | 当前只读扫描 | `scope=current_scan`，与零待译门禁同源 | 项目 / TL / include 范围变化即重算 |
| `quality_summary` | 已有 manifest 引用的报告 | 项目身份 + plan fingerprint + 报告存在（+ sha256） | 任一匹配条件不满足 → `stale` |

三条摘要都不跨请求缓存；payload 只描述“本次预检当前看到的状态”。

## 5. CLI / GUI 展示合同

- **CLI 文本**：在现有 facts 后增加 cost / coverage / quality 三行；`unknown` / `not_available` / `stale`
  使用明确文案，不显示 0 成本或“通过”。
- **CLI JSON**：新增 `cost` / `coverage` / `quality_summary` 三键；`--output json` 仍只输出 envelope。
- **GUI**：`TRANSLATION_PREFLIGHT_COPY.body` 增加对应行（同一 payload，不另算）；
  阻塞 / 确认 / 零待译逻辑保持不变；预检结论不是 `check` / `apply` 授权。
- 字段与文案由可复用核心生成，CLI 与 GUI 消费同一结果。

## 6. 非目标

- 不重做 doctor、价格系统或质量规则，不引入新的 readiness 权威；
- 不增加字体检查实现（另行 spike #487）、不执行编排、自动修复、实时网络定价或历史用量 dashboard；
- 不把预检扩展成全项目 dashboard，不扩大 `schema_version` 语义。

## 7. 实施切片

1. **S1 核心**：新增可复用 summary 函数（cost / coverage / quality）并接入 preflight payload；
   单测覆盖 known / unknown、available / stale / not_available、项目与 plan 切换。
2. **S2 界面**：CLI 文本 + GUI facts + `user_copy`，同步 argparse 帮助与 GUI 测试。
3. **S3 文档**：更新本文件的实现状态与现行文档（`docs/quickstart_agent.md` / `docs/gui_workbench.md` 相关段落）。

## 8. 验收映射

| #488 验收 | 本合同的落点 |
| --- | --- |
| 文档列出当前字段、最小新增字段、来源与 freshness/unknown 合同 | 本文 §2–§4 |
| CLI/GUI 一致展示有证据的成本 / coverage / 质量摘要 | §5 |
| 切换项目 / 任务 / profile 后不沿用旧摘要 | §4 |
| 离线无 Key 可运行；不调用 provider / embedding / prepare | §1、§3 |
| 现有预检风险与零待译 coverage 合同保持 | §2、§3.2 |
| 针对性测试覆盖已知 / 未知 / stale / 缺失 | §7 S1/S2 |
| 同步 argparse、GUI copy、现行文档 | §7 S2/S3 |
