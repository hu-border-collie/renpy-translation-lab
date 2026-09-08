# Final Review 结果失败分类与 targeted resume fixture spike（#309）

状态：P0 fixture spike，2026-08-09。本文记录现状，不实现激进 JSON repair、
unit 内 adaptive split 或完整恢复器。

## 范围与方法

固定 fixture 位于
[`tests/fixtures/final_review_result_cases.json`](../../tests/fixtures/final_review_result_cases.json)，
并由
[`tests/test_final_review_result_spike.py`](../../tests/test_final_review_result_spike.py)
直接走 `ingest_result_rows()`。每个 case 使用含两个 item 的同一 review unit，避免
把 parser 结果与不同 unit shape 混在一起。

“接受”指 unit 被记为 `done`；“拒绝”指 unit 被记为 `failed`。误接受 / 误拒绝只
评价决策，不表示当前自由文本 `error` 已提供稳定分类。

## Fixture 对比

| Fixture | 响应要点 | 当前行为 | 期望分类与决策 | 结果 |
|---|---|---|---|---|
| `truncated_json` | finding 内部截断，缺闭合括号 | `failed`；`failed_to_parse_model_json: ...` | `truncate`；拒绝 | 正确拒绝，但分类不稳定 |
| `missing_findings` | 根对象只有 `issues_count` | `failed`；提示缺 `findings` array | `missing_findings`；拒绝 | 正确拒绝 |
| `schema_missing_reason` | finding 缺 response schema 必填的 `reason` | `done`，生成 1 条空 reason finding | `schema`；拒绝 | **误接受** |
| `duplicate_item` | 同一 item 的同一 finding 完全重复 | `done`，保留 2 条同 ID finding | `duplicate_item`；拒绝 | **误接受** |
| `refusal` | provider `promptFeedback.blockReason=SAFETY`，无文本 | `failed`；`missing_response_text` | `refusal`；拒绝 | 正确拒绝，但拒答原因丢失 |
| `empty_ok` | `{"findings":[]}` | `done`，0 findings | `empty_ok`；接受 | 正确接受 |
| `reviewed_count_mismatch` | unit 有 2 项，假设回执声称只审 1 项 | `done`，忽略 `complete` / count | `schema`；若存在回执则拒绝矛盾值 | **假设性 receipt 缺口** |

现行合同的 6 类基线合计：2/6 误接受，0/6 误拒绝。截断、缺 `findings` 和无文本已经
fail closed；已有直接证据的收益集中在客户端 schema 校验和精确重复项校验，不在
宽松 JSON repair。第 7 个数量不符 case 是尚不存在的 receipt 合同探针，单独记录，
不计入当前 parser bug 比例。

## 候选稳定码

候选集合固定为：

- `truncate`：JSON 在可识别容器内部结束，语法外壳不完整；
- `missing_findings`：合法 JSON 对象没有规范的 `findings` 字段；
- `schema`：root / finding shape、必填字段、item 引用或完成数量违反 schema；
- `duplicate_item`：同一 item 的同一规范化 finding 完全重复；同一 item 的不同问题
  类型仍允许并存，不能只按 `item_id` 去重；
- `refusal`：provider 明确返回 block / refusal metadata；不建议只靠自然语言猜测；
- `empty_ok`：结构与完成合同都成立且 findings 为空，这是成功分类，不是失败。

数量不符映射为 `schema`，避免在尚未确定回执字段前新增只服务一种校验的错误码。
若后续进入实现，parser 应返回结构化分类与 detail；失败 unit artifact 可新增可选
`error_code`，保留现有 `error` 作为人类可读细节。`empty_ok` 是成功分类，不写入
`error_code`。与 #296 接入机器 envelope 时，失败可展示为
`final_review.<code>`，无需等待所有 final-review 命令迁移完成。

## Targeted resume 结论

现有 unit 级 resume **不需要新 artifact/schema 字段**。当前字段已经足够：

- `unit_id` 定位重跑单位；
- `status` 区分 `failed` / `done`；
- `items`、`items_digest`、`context_digest`、model、prompt schema 共同重算
  `input_digest`；
- digest 匹配的 `done` unit 被 skip，`failed` unit 单独写入新的 `requests.jsonl`；
- 未重跑 unit 的 findings 由 `merge_findings_preserve_selection()` 保留。

新增合约测试使用一个 matching-digest `done` unit 和一个 `failed` unit，证明只有失败
unit 进入 request；done unit 的状态、digest、完成时间、items、finding count 和 finding
均保持不变。resume 目前会给已跳过行附加 `live_input_digest` / `live_context_digest`
审计字段；这不改变上述完成语义，也不是 targeted resume 的新依赖。

## Completion receipt 结论

结论：**暂缓把 completion receipt 设为强制合同。**

证据是 `reviewed_count_mismatch` 与 `empty_ok` 在当前 ingest 中得到完全相同的
`done + 0 findings` 结果。仅靠 Batch row key、unit digest 和一个合法空 findings 数组，
无法证明模型实际看完 unit 的全部 item；但 `reviewed_item_count` 和 `complete` 仍然只是
同一个模型的自报值，也不能独立证明完成。该 fixture 证明“若响应已经带回执，内部数量
矛盾不能被忽略”，并没有证明“强制模型增加回执”能降低真实误接受率。

当前有直接 fixture 证据的收益是客户端 schema 与精确重复项校验，应先实现这两项。
只有收集到真实 provider 的“语法和 schema 均合法、但实际只覆盖部分 item”样本，并能
证明 receipt 与该失败相关时，再把 `complete: true` 与
`reviewed_item_count == unit.item_count` 设为 response schema 必填。届时必须升级
`prompt_schema_version`，让旧 done unit 按现有 digest 机制自动 stale；receipt 仍不是
targeted resume 的依赖，也不要求复制进 campaign manifest。

实施顺序应为：

1. 客户端按原 response schema 严格校验，拒绝缺必填字段和未知 item 引用；
2. 仅拒绝“同 item + 同规范化 finding”的精确重复，保留同 item 多类问题；
3. 收集真实 provider 的部分覆盖样本，再决定是否加入 completion receipt 并升级
   prompt schema；
4. 再评估是否需要仅修语法外壳的 JSON repair。

任何 repair 都必须在修复后重新通过相同 schema、重复项以及已启用的 receipt 校验；不得补造
`findings`、`reason`、`complete` 或数量字段。本 spike 不支持 unit 内 adaptive split；
失败仍按现有稳定 unit 整体重跑。

## 验证记录

2026-08-09 在独立 `codex/issue-309-final-review-spike` worktree 验证：

- `python -m unittest tests.test_final_review_result_spike tests.test_final_review_llm -q`：
  25 项通过；
- `python -B tests/run_cli_tests.py -q`：953 项通过，1 项跳过；
- `python -m unittest tests.test_gui_final_review_workflow -q`：8 项通过；
- `python scripts/run_quality_gates.py all`：ruff、mypy、pip-audit blocking gate 全通过；
- 39 个 Markdown 文件的 201 个本地链接目标均存在；
- `git diff --check` 通过。

## 风险与非目标

- fixture 是确定性的合成基线，不代表真实 provider 各类失败的发生率；
- `refusal` 的可靠分类需要保留 response metadata，纯文本启发式可能误判正常内容；
- response schema 当前未设置 `additionalProperties: false`；receipt 落地时应明确兼容
  策略，而不是静默接收拼错字段；
- 本 spike 故意钉住当前合同的 2 个误接受，并单列 1 个假设性 receipt 缺口；后续
  修复或引入新合同时必须同步更新 fixture 的 `current_*`、`evidence_scope` 与
  assessment，不能用放宽测试掩盖行为变化。
