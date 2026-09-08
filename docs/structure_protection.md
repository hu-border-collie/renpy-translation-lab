# 结构 token 保护、恢复与校验

对应 [#425](https://github.com/hu-border-collie/renpy-translation-lab/issues/425)。新建的共享 TranslationPlan 翻译请求默认启用规则 v2；Sync、Gemini Batch 与新建 Ren'Py repair 请求使用同一个保护模块。repair 仍通过 Ren'Py 源行扫描建立 job，不扩展到 Tyrano；显式传入其他引擎的 repair job 会被拒绝。缺失或未知引擎身份会被拒绝，不猜测引擎。没有新增配置开关，CLI 与 GUI 的既有翻译入口自动获得此能力。

## 启用和兼容边界

- 只替换送模型的 TARGET 文本视图。canonical source、条目 ID、occurrence、locator、源快照和写回位置保持原值；上下文和术语文本不做强制占位替换。
- `transport_metadata.structure_protection` 保存规则版本、engine、request ID、语义 scope、每条源摘要、映射和映射摘要。请求 audit fingerprint 覆盖此元数据。相同语义输入的 Sync/Batch TARGET 一致；不同上下文或派生请求使用不同 scope。
- 没有该字段的历史请求是明确的 legacy 路径，读取时不生成映射；由历史请求派生的重试仍使用历史合同。新建任务不会原地迁移旧请求。已有的 legacy submit 限制、Sync 恢复 freshness 限制继续生效，不能恢复的旧任务应从当前源新建任务。
- 字段存在但版本旧于 v2、版本未知、映射损坏、条目或请求不符时拒绝恢复，不能退回 legacy。规则升级须提升保护版本；本次 check 合同提升到 v4，旧 check 需要重跑。
- 已持久化的 canonical 结果只做映射绑定和结构校验，不再做第二次替换。原始模型返回仍是原始返回，不能因为它像 canonical 文本就自动接受。

## 引擎规则 v2

规则由 `engine_adapters/structure_rules.py` 按 engine 显式选择，不通过文件内容猜测引擎。

| 引擎 | 保护范围 | 重排规则 |
|---|---|---|
| Ren'Py | `{...}` 标签、含索引/引号的 `[...]` 插值、百分号格式符、`[[` / `{{` / `%%` 转义、反斜线换行/制表转义和实际行边界 | 命名插值和命名百分号格式符可随语序移动；标签、位置格式符、转义和行边界保持相对顺序；已知成对标签校验栈嵌套 |
| Tyrano | 已支持 catalog 条目中的方括号控制标签（尊重属性引号）、反斜线转义和行边界 | 控制结构保持相对顺序；已有空值和控制字符 blocker 继续有效 |

同名重复变量按出现次数分别编号。原文已有 `__RTL_` 样式串会作为字面量保护，生成的命名空间也避开原文。占位符中只有 `_variable_` 类允许移动，其余类型由系统指令要求保序。首版采用保守结构顺序：不支持交换整个兄弟标签块；不会因变量重排而拒绝正常译文。

Ren'Py 未闭合方括号、花括号或不合法嵌套会拒绝生成或恢复。Tyrano parser 已将未转义控制标签与文本分开，提取的文本单元通过 `tyrano_literal_brackets` 元数据保留这一事实；其中的括号是字面文本。其他 Tyrano 文本中的未闭合括号按普通文字处理，已识别控制标签仍保序；本规则不承诺解析自定义标签的业务语义。它不新增引擎、不扩展 Tyrano 的可提取范围，也不允许直接改写 `.ks`。

## 制品与恢复

- Batch 的 manifest chunk 持久化映射，requests.jsonl 保存对应 prompt 和绑定指纹；download 仍原样、原子保存 provider 结果及摘要。check、重试分析和结果合并按对应 chunk 恢复，原始 response 不被覆盖。
- 包拆分保留完整请求和原映射；按条目重试重新生成子请求和子映射，并验证父映射。merge-retry 先用各自映射恢复，再合并 canonical 结果。
- 耐久 Sync 的 run store 保存请求映射、原始 response、normalized payload 和合同诊断；失败 attempt 也保存这些层，且不生成 item winner。重启使用已存请求，派生重试验证父请求绑定。初始及派生映射的 scope 均从当前完整 prompt、context、system 和 chunk 重算，不信任映射自带的 scope。
- 普通 Sync preview 的合同诊断保存 `protection_traces`，包含 request ID/fingerprint、映射、原始结构化返回、恢复后的结果和诊断；repair 的请求日志保存映射，结果日志保存对应原始正文与恢复结果。

这些是本地任务制品，可能包含完整私有文本，应与已有 manifest、模型结果和日志同等保护，不应提交到公共仓库。GUI 只展示原因码的中文解释，不展示映射值或 Provider 异常正文。

## 失败与写回

稳定原因码包括 `protection.missing_token`、`duplicate_token`、`extra_token`、`modified_token`、`mapping_mismatch`、`stale_mapping`、`structure_changed`、`structure_order` 和 `invalid_structure`（均使用 `protection.` 前缀）。CLI 的合同诊断与 GUI 的检查失败/重试报告消费同一组原因码。

失败条目不进入可写结果，只进入正常缺项重试或失败报告。不猜测位置、不追加缺失 token、不自动更换 Provider。结构失败保持 blocker；普通质量 warning 和配置提升的 quality blocker 保持原策略。最终写回仍必须满足 [Batch check → apply](batch_workflows.md) 或对应 Sync 预览、源复核与原子写回合同，`--force` 不绕过门禁。

## 离线验证

```powershell
python -m unittest tests.test_structure_protection tests.test_translation_plan tests.test_batch_golden_corpus -q
python -m unittest tests.test_sync_run_service tests.test_sync_run_store tests.test_durable_sync_executor -q
python -B tests/run_cli_tests.py -q
python -B tests/run_gui_tests.py -q
python scripts/run_quality_gates.py all
```

实现依据本仓库现有合同独立编写，未复制研究项目的第三方实现。
