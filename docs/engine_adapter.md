# Ren'Py Engine Adapter 与覆盖审计

文档地图：[docs/README.md](README.md)

当前已交付 #265 的 P1：同步翻译预览与普通 Batch translation build 通过同一个
`RenPyAdapter` 发现 `.rpy`、构建候选清单并提取 `TranslationUnit`。这一阶段是只读
扫描迁移，不改变现有命令、配置、manifest v1/v2、identity v2、终端文案或
`check -> apply` 写回合同。

## 当前边界

| 层 | 当前职责 |
|---|---|
| `engine_adapters/contracts.py` | engine-neutral protocol、capability、candidate、occurrence、validation / writeback 的版本化信封 |
| `engine_adapters/renpy.py` | Ren'Py 项目发现、`.rpy` inventory、分类、source marker、speaker、translate block / occurrence / ordinal、只读 occurrence 提取 |
| `engine_adapters/coverage.py` | 独立校验 inventory invariant、生成稳定 digest、导出 coverage/review package、导入并校验人工或 Agent review |
| `translation_core.py` | 唯一的 `TranslationUnit` / `ModelResult` 核心模型；adapter 不创建第二套翻译单元 |
| sync / Batch workflow | 模型调用、prompt、progress、manifest、preview/check/apply、RAG / Source Index 回灌 |

P1 的 `relocate_occurrences()`、`validate_translation()` 与
`build_writeback_plan()` 明确 fail closed；Ren'Py 重定位、格式校验和声明式写回属于
P2。revision、keyword、Project Analysis 与 Final Review 的扫描入口在 P1 保持原状。

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

完整 schema、P2 安全边界与后续阶段见
[Engine Adapter 合同设计](plans/engine_adapter_contract.md)。
