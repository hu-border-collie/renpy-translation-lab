# Ren'Py coverage block 归因报告（#426 spike）

> **状态**：合成 fixture 归因完成；#460–#464 五条 follow-up 已全部实现并反映在本文；
> **未完成真实大型项目频率归因**，跨行 writeback span 仍未支持。
> **输入**：原创合成 fixture [`tests/fixtures/renpy_coverage_attribution/`](../../tests/fixtures/renpy_coverage_attribution/README.md)，
> adapter `renpy@1.1.7`，classification rules digest `c3db8146…300b`（#462 新增 `translation_present_without_marker`）。
> **运行方式**：[`scripts/coverage_block_attribution.py`](../../scripts/coverage_block_attribution.py)，只读；不写项目、不调用模型。
> **证据边界**：本环境没有获得授权的真实大型项目只读副本，因此本报告的频率与分布结论不得外推为真实项目归因。
> 公开材料只包含脱敏汇总与原创 fixture，不含私有游戏名、脚本、地图、路径、对白或 Batch 结果。

## 1. 摘要结论

同一份 fixture 上（`renpy@1.1.7`，含 #460–#464 修复），coverage 自动状态仍为 `block`，
16 个 candidate 的分类为：

| classification | candidates | 对 block 的作用 |
|---|---:|---|
| `already_translated` | 9 | 否 |
| `explicitly_excluded` | 3 | 否 |
| `translatable` | 1 | 否 |
| `unknown` | 0 | 否 |
| `unsupported` | 2 | 否（使状态为 attention 级风险；当前 fixture 的 block 来自 parse_error） |
| `parse_error` | 1 | 是（孤儿 `old` 行） |

自动归因得到的主要根因：

1. **跨行字符串 parser gap：已由 #460 修复**。多行 `"""…"""`（22–23 行）与行尾反斜杠续行（27–28 行）
   现在各产生 1 个带 `end_line_hint` / `multiline=true` 的 candidate：三引号保留换行原文，
   反斜杠续行按 Python 字符串语义拼接；TL 中 paired source marker + 中文译文分类为 `already_translated`。
   修复前它们是 4 个 `renpy.tokenize_error` 并连带 2 个 source marker parse_error。
2. **source marker 级联：已由 #461 修复**。marker pairing 不再依赖 candidate 是否为
   identity/legacy/multiline：动态 f-string 这类 unsupported/unknown candidate 上方合法的 `# "…"` 仍会与逻辑
   语句配对，不再产生额外 `renpy.source_marker_unpaired`。
3. **当前确认不支持的结构**：动态 f-string（`renpy.dynamic_string_expression`）与非标准 `old` 标记行
   （`renpy.custom_statement_unsupported`）各 1 个，均为 `unsupported`。
4. **缺少 source marker 的已译 TL 字符串：已由 #462 定义策略**。Ren'Py 8.5.3 lint 证据表明：单引号 say、
   三引号/续行字符串与动态 f-string 都是官方接受的语法；TL 文件缺少生成的 `# "…"` 注释不影响运行时按
   block identity 翻译。目标语言文本 + say/narration 结构现在走明确的
   `renpy.catalog.translation_present_without_marker`，classification 为 `already_translated`；
   证据不足的字符串仍保留 `unknown`，不静默清零。
5. **quoted comment / 孤儿 `old`：已由 #464 定义策略**。没有同 block 后继 target 的 quoted comment
   （39 行 TODO）按玩家不可见注释归入 `explicitly_excluded` + `renpy.non_player_visible_literal`；
   仍有后继 target 但未配对的 dangling marker 保持 parse_error。只有 `old` 没有 `new`（34 行）与
   `old` 行含额外 token 仍分别保持 parse_error / unsupported，不静默丢弃。
6. **合法排除与负控未被破坏**：`voice` / asset path 保持 `explicitly_excluded`；
   paired dialogue、speaker label、空 target pending 对白、含 asset 路径子串的可见对白仍正常。

## 2. 复现与聚合口径

```powershell
python scripts/coverage_block_attribution.py `
  tests/fixtures/renpy_coverage_attribution/game/tl/schinese `
  --project-root tests/fixtures/renpy_coverage_attribution/game `
  --target-language schinese `
  --generated-at 2026-09-11T00:00:00+00:00 `
  --json-output logs/attribution/coverage_attribution.json `
  --markdown-output logs/attribution/coverage_attribution.md
```

- 聚合按 `classification | structure_kind | reason_codes | review_flags` 分组；每个 group 保留**全部** candidate locator，
  `sum(group.count) == coverage candidate_count` 由模块内 invariant 拒绝不满足的情况。
- 输出分别记录自动证据（`automatic_evidence`）、人工判断（`human_judgment`，可选 `--decisions`）与
  `evidence_gaps`；没有人工判断时不会被伪装成已确认结论。
- `--generated-at` 固定时间戳后，JSON 与 Markdown 可跨进程逐字节复现；`aggregation_digest` 只覆盖
  稳定输入与聚合结果。
- 默认输出不含原文摘录；本地排查可用 `--include-excerpts`，但带摘录的输出不得公开。

输入指纹（fixture）。 除特别说明外，下表与分组表记录 Python 3.12 运行值；不同 Python tokenize 版本可能
改变个别 structure kind 的拆分与 `aggregation_digest`，但 classification / reason 总计数保持一致。

| 项目 | 值 |
|---|---|
| `source_fingerprint` | `e9a8e502173f4afef5b78b41f92666af5e679a40fbf0eee2b5ba6aea534c9b86` |
| `project_snapshot_fingerprint` | `eaa3def51f888454db52ef8ade9d93aa4b2f020017fcf0c76e4f0ad2b6d9d922` |
| `classification_rules_digest` | `3cce9a4dddf9664568ebaba04131984295a1d1667363d03493704abb084ff410` |
| `inventory_digest` | `ed8cf30562eeaed78407b3bbf0f6fe37c9b70b4a24f0094e71a75ab88cef56ef` |
| `coverage_digest` | `176a3c1754b8bb4b5cec5bfb82e2e751ee9745ed447aec7995fec1b864559e65` |
| `aggregation_digest`（2026-09-11 Python 3.12、`renpy@1.1.7` 固定运行） | `5d4cb84c0b0bf51337fb7a6976add5405555d7e199364967521bbe942223c622` |

## 3. 分类报告

| 归因类别 | 证据等级 | group / 数量 | 结论状态 |
|---|---|---|---|
| 跨行字符串（已修复 #460） | 自动（`already_translated` + `translate_comment_pair`） | `dialogue_string` 中 2 个 multiline locator（22→23、27→28） | 已提取为 candidate；writeback 仍按单行 span 处理 |
| 不支持结构（动态文本） | 自动（reason code 已登记） | `dynamic_string_expression` ×1 | 当前确认不支持；产品策略见 §5.3 |
| 不支持结构（非标准 `old`） | 自动（reason code 已登记） | `nonstandard_old_source_marker` ×1 | 当前确认不支持；见 §5.3 |
| quoted comment 无 target（已实现 #464） | 自动 + 语法证据 | `comment_without_target` ×1（39 行 TODO），reason `renpy.non_player_visible_literal` | 玩家不可见注释，合法排除 |
| dangling source marker | 自动 + fail-closed | 仍有后继 target 时保留 `source_comment` parse_error | 不静默丢弃可能缺失的译文 |
| 孤儿 `old` 行 | 自动 + fail-closed | `old_source_marker` ×1（34 行） | 仍 parse_error；Ren'Py lint 也报 previous string missing translation |
| 非标准 `old` 行 | 自动 | `nonstandard_old_source_marker` ×1 | 保持 unsupported；lint 报语法/翻译错误 |
| 未标记已译对白（已实现 #462） | 自动 + heuristic | `dialogue_string` ×1，reason `renpy.catalog.translation_present_without_marker` | 目标语言 say/narration 分类为 `already_translated`；非目标语言仍保持 `unknown` |
| 合法排除 | 自动 | `voice_statement` ×1、`asset_literal` ×1 | 不需修复 |
| 已译负控 | 自动 | 普通 dialogue ×6（含 2 个 multiline）、speaker label ×1、narration ×1 | 行为保持；含 asset 路径子串的可见对白未被过度排除 |
| 待译负控 | 自动 | empty target ×1 | 仍在 `translatable`，未因覆盖归因被排除 |

定位依据（fixture 相对路径 `game/tl/schinese/attribution_samples.rpy`）：

| 根因 | 行 |
|---|---|
| 多行三引号对白（已提取） | 22–23 |
| 反斜杠续行对白（已提取） | 27–28 |
| 动态 f-string 及 source marker | 17–18 |
| 非标准 `old` 标记 | 31 |
| 孤儿 `old` 行 | 34 |
| 尾部 quoted comment | 39 |
| 未标记单引号对白 | 43 |
| 合法排除 | 46–47 |
| 负控（paired / speaker label / pending / exclusions 后对白 / asset-lookalike） | 6、10、14、38、49、53 |

完整的 candidate ID、block occurrence、列号与 reason code 见脚本 JSON 输出；Markdown 默认每组列出最多
`--max-locators`（默认 10）条 locator。

### 3.1 漏扫确认

脚本另外用独立 quote-run state machine 扫描源文本，不调用 adapter tokenizer：

| 指标 | 值 |
|---|---:|
| 独立扫描 quote spans | 17 |
| 由非 `parse_error` candidate 覆盖 | 16 |
| 仅由 `parse_error` candidate 覆盖（文本不可提取） | 1（34 行孤儿 `old`） |
| 无任何 candidate 覆盖 | **0** |

#460 修复后，22–28 行的 4 个 quote span 由新的 multiline candidate 覆盖；当前唯一不可提取区域是孤儿
`old` 行。inventory 仍没有漏掉「完全没有 candidate」的字符串。扫描器本身是启发式（见 §6），
不能替代官方 parser 证据。

## 4. 自动证据 vs 人工判断 vs 未知

- **自动证据**：classification、structure kind、已登记 reason code、coverage 计数与独立扫描结果。
- **heuristic 提示**：`review_flags` 只提示复核方向，例如
  `comment_span_may_be_non_player_visible`、`orphan_old_row_needs_catalog_evidence`、
  `dynamic_text_may_be_player_visible`、`single_quote_literal_needs_official_parser_check`；
  它们不改变 classification，也不构成排除理由。
- **人工判断**：通过 `--decisions` 导入，必须提供 reviewer、category、rationale；
  group key 在当前输入中不存在时 fail closed，避免旧判断静默套用到新规则。
  本报告本身没有人工 judgement 输入，所以「归因类别」仍按自动/heuristic 标记。

## 5. 后续修复拆单草案

> 5.1 已由 issue #460 实现；其余仍是草案，也不代表运行时已经修复。
> 每个拆单都必须区分「归因完成」与「运行时修复完成」。

### 5.1 P1：支持跨行字符串的 inventory tokenization（已实现 / #460）

> **状态：已实现。** `_inventory_document()` 现在通过 `_tokenize_document_lines()` 把合法的
> triple-quote / 反斜杠续行区域与后续物理行合并 tokenize；multiline locator 增加
> `end_line_hint` 与 `multiline=true`，TL 中 paired source marker 的分类由新分支处理。
> 跨行 candidate 的 writeback 仍是单行 span 合同：`build_writeback_plan()` 会因 span 超出物理行而
> fail closed，不写坏文件，但完整的跨行写回需要另立 follow-up。
> #460 实现时版本为 `renpy@1.1.3`（当前 `renpy@1.1.4`，含 #461）；新增测试
> `tests/test_engine_adapter_multiline_strings.py`，并更新了本 fixture 的 characterization。

- **问题**：`_inventory_document()` 逐物理行 `tokenize.generate_tokens(io.StringIO(line))`；多行三引号与
  行尾反斜杠在行内是未闭合字符串，产生 `renpy.tokenize_error`，玩家可见对白没有 candidate/unit。
- **预期行为**：合法跨行字符串产生一个可定位（起始行 + 列、结束行 + 列）的 `dialogue_string` / `narration_string`
  candidate；译文原文保留换行语义；source marker pairing 仍按逻辑语句工作。
- **最小修复设计**：对文档使用整文件/逻辑行 tokenizer（保留每个 token 的 `(line, col)` 或映射回物理行），
  或在现有逐行路径显式累积未闭合的 triple-quote / 反斜杠续行区域；不得用全局正则替换 parser。
- **验收**：
  1. fixture 22–28 行不再产生 `renpy.tokenize_error`，且出现可提取 candidate；
  2. 更新 `tests/test_coverage_attribution.py` 的 characterization 期望；
  3. 保留非法未闭合字符串的 `parse_error` 行为；
  4. 现有 P1/P2 adapter 等价性测试通过；
  5. 不改变 writeback；若需要 adapter 行为版本升级，单独按既有规则处理。

### 5.2 P2：修复 source marker 级联 parse_error（已实现 / #461）

> **状态：已实现。** 所有 string/f-string 候选都会基于逻辑语句尝试 pairing：`old` marker 只配 `new` 行，
> voice 行不消费 comment marker，真正没有后继字符串的 quoted comment 仍保持
> `renpy.source_marker_unpaired`。没有新增 reason code，沿用 #420 allowlist。
> 当前实现版本 `renpy@1.1.4`，新增测试 `tests/test_engine_adapter_marker_pairing.py`。

- **问题**：字符串 candidate 一旦是 `unsupported` / `unknown` / tokenize 失败，上方合法 marker comment 不会被加入
  `paired_source_marker_lines`，随后作为 `renpy.source_marker_unpaired` 再记一个 `parse_error`，使 block 计数虚高。
- **预期行为**：marker comment 与紧随逻辑语句的字符串绑定应由语句结构决定；即使该字符串当前不可译，
  marker 也不应因为 candidate 分类而丢失 pairing。仍不为随机 quoted comment 自动生成可译文本。
- **验收**：
  1. fixture 中 3 个「marker + 跨行/动态字符串」不再产生额外 `source_marker_unpaired`；
  2. 尾部 `# TODO "…"` 仍有明确、可复核的 reason code / review 状态；
  3. reason code 必须先登记 allowlist（沿用 #420 合同）；
  4. P1/P2 测试与 coverage 计数对应关系保持。

### 5.3 P2：定义 TL 字符串缺少 source marker / 动态文本的分类策略（已实现 / #462）

> **官方 parser 证据（2026-09-11，Ren'Py 8.5.3.26051504）**：用最小项目执行
> `renpy.sh /tmp/project translate schinese` 生成 TL 文件后，删除某个 block 的 `# e "…"` source comment
> 并把目标改成单引号中文，再执行 `renpy.sh /tmp/project lint`：lint 不报语法错误，统计仍把该行计为
> schinese translation；单引号 say、三引号跨行、反斜杠续行与 say-position f-string 都被官方 parser 接受。
> 因此「TL + say/narration + 目标语言文本」有足够证据视为已译，而不是仅凭 `contains_chinese()` 的静默扩大。
>
> **策略**：新增登记 allowlist 的 `renpy.catalog.translation_present_without_marker`，只在前述窄条件下
> 把 candidate 标为 `already_translated`，并在 evidence / unit metadata 记录 `source_marker_missing=true`；
> dynamic f-string 仍保持 `unsupported`，按结构化支持计划处理，不自动翻译；非目标语言或证据不足的字符串
> 仍保留 `unknown` 供人工 review/repair。
> 当前实现版本 `renpy@1.1.6`，新增测试 `tests/test_engine_adapter_unmarked_translation.py`。

- **问题**：没有 paired comment 的已译 TL 字符串落到 `unknown / renpy.visibility_unknown`（本 fixture 的单引号对白）；
  动态 f-string 落到 `unsupported`。两者都会使 coverage `block`，但当前没有「官方 parser 接受性 + catalog provenance」
  的可执行判定。
- **预期行为**：先用 Ren'Py 官方 parser / 真实 catalog 证据确认结构和语义，再决定：
  - 有可靠 catalog/provenance 证据的已译字符串：走明确 reason code 的 `already_translated`；
  - 确属动态生成的玩家可见文本：进入结构化 unsupported 支持计划；
  - 证据不足：保持 `unknown`，但提供人工 review/repair 入口，不静默清零。
- **验收**：
  1. 文档记录 Ren'Py 版本与最小复现证据；
  2. 新 reason code 登记 allowlist，classification 变化同步 adapter version 规则与测试；
  3. 不允许仅凭 `contains_chinese()` 静默扩大 `already_translated`；
  4. 与 #416 speaker-label 规则不冲突。

### 5.4 P3：quoted comment / 孤儿 `old` 行的 false-positive 复核（已实现 / #464）

> **官方 parser 证据（Ren'Py 8.5.3.26051504）**：只有 `# TODO "…"` 注释的 strings block，lint 不报错；
> `old "…"` 没有 `new` 或 `old` 行含额外 token 时，lint 报 “previous string is missing a translation”。
> 因此策略是：
> - 没有同 block 后继字符串 target 的 quoted comment → `explicitly_excluded` +
>   `renpy.non_player_visible_literal`（合法排除，原因码原有且已登记 allowlist）；
> - 仍有后继 target 但未配对的 dangling marker → 保持 `renpy.source_marker_unpaired` parse_error；
> - 孤儿 `old` 行 → 保持 parse_error；`old` 行含额外 token → 保持 `renpy.custom_statement_unsupported`
>   unsupported。所有候选都保留在 inventory，不静默丢弃。
> 当前实现版本 `renpy@1.1.7`，新增测试 `tests/test_engine_adapter_comment_policy.py`。

- **问题**：尾部 `# TODO "…"` 与只有 `old` 没有 `new` 的行目前不能区分「开发者注释」与「真实 source marker
  缺失」，都会进入 parse_error 并 block。
- **预期行为**：基于 Ren'Py TL 语法与 catalog 证据给出明确分类；不满足 source marker 的结构不得被静默丢弃，
  `old` 行仍 fail closed，直到有可靠修复路径。
- **验收**：
  1. 至少覆盖 TODO 注释、`old` 无 `new`、`old` 行含额外 token 三类 fixture；
  2. 如改为合法排除，必须有有效 reason code 和人工 review 证据；
  3. 不把代码/断言/颜色/内部 ID 无条件移出 inventory。

### 5.5 P2：canonicalize parse-error evidence，恢复 coverage digest 可复现（已实现 / #463）

> **状态：已实现。** `_stable_error_text()` 会移除异常文本中的 `at 0x…` 等进程地址，再写入
> parse-error evidence / exception excerpt；相同输入、规则与 adapter 版本下 `inventory_digest` /
> `coverage_digest` 跨进程稳定。adapter version `renpy@1.1.5`，新增测试
> `tests/test_engine_adapter_digest_stability.py`；归因脚本重新输出两个 digest。
> 该修复只规范化错误文本，不改变 classification、reason code 或 writeback。

- **问题**：动态 f-string 的 `evidence["parse_error"]` 含 `<ast.JoinedStr object at 0x…>` 地址，
  导致相同输入的 `inventory_digest` / `coverage_digest` 在不同进程间漂移；本脚本才不得不省略这两个派生 digest。
- **预期行为**：parse-error evidence 序列化稳定（去掉对象地址等非确定内容），adapter inventory/coverage digest
  在同一输入同一规则下跨进程一致；不改变 classification、reason code 或 writeback。
- **验收**：
  1. 对同一 fixture 连续两次完整扫描得到相同 digest；
  2. 现有 adapter/coverage 测试通过；
  3. `scripts/coverage_block_attribution.py` 重新纳入 coverage/inventory digest 字段并保留 `--generated-at` 复现测试。

## 6. 未知与证据缺口

1. **真实大型项目频率未知**：没有授权副本，无法确认 unknown / unsupported / parse_error 的真实数量级与语法分布；
   本报告不能被称为「真实大型项目归因完成」。
2. **官方 parser 证据已有最小复现**：Ren'Py 8.5.3 接受单引号 say、三引号/续行、动态 f-string，且 TL 缺少
   source comment 不影响 runtime 翻译；但真实大型项目是否依赖其他 Ren'Py 版本/工具链仍需样本确认。
3. **独立扫描是启发式**：quote-run state machine 可能对嵌套 f-string、转义和注释过度/不足计数；
   `uncovered_span_count=0` 只说明本 fixture 没发现静默漏扫，不等于全局证明。
4. **digest 跨版本会变化是设计行为**：同一 adapter / rules / 输入下 digest 已可复现（#463）；
   升级 adapter version、classification rules 或 source 内容后，digest 变化用于 freshness 校验，不能跨版本比较。

## 7. 边界与仍未做的事

- #426 的 5 条 follow-up（#460–#464）已全部实现（当前 `renpy@1.1.7`）；跨行 writeback span 仍未支持，
  multiline candidate 在 `build_writeback_plan()` 仍 fail closed。
- 没有改变 GUI、doctor 或 TyranoScript adapter；真实大型项目频率归因仍缺授权样本。
- 按 #464 策略合法排除的只有「同 block 没有任何后继 target 的 quoted comment」；
  证据不足或结构异常仍保持 parse_error / unsupported，不伪装成 `ready`。
