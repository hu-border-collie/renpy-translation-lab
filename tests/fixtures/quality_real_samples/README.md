# quality_real_samples

Issue #364（真实项目质量规则校准）的离线回归样本。这里**不包含任何私有游戏文本**：所有样本都是从公开 issue #313 / #364 中已知问题示例重建、或人工编写的脱敏代表句。

## 运行

```bash
python -m unittest tests.test_quality_real_samples -v
```

测试对每个首版 reason code 断言：

- `positive`：必须被报告；
- `negative`：白名单 / 合法混排 / 已翻译证据，必须对该 reason code 保持沉默。

覆盖范围与 `translation_quality.ALL_REASON_CODES` 强制一致，新增首版规则时测试会提醒补样本。

真实项目校准基线的数量统计用 `scripts/quality_calibration_report.py` 生成，完整执行流程见
[#364 校准执行手册](../../../docs/plans/issue-364-calibration-runbook.md)。

## 格式

`samples.json` 的 `cases` 每个元素：

```json
{
  "rule_id": "cjk_latin_spacing",
  "reason_code": "quality.typography.cjk_latin_spacing",
  "positive": {
    "name": "…",
    "subject": {"translation": "这是iPhone手机"},
    "policy": {"rules": {"cjk_latin_spacing": "warning"}}
  },
  "negative": {
    "name": "…",
    "subject": {"translation": "这是VIP通道"},
    "policy": {"rules": {"cjk_latin_spacing": "warning"}}
  }
}
```

- `subject`：传给 `translation_quality.check_subject` 的机械质量主语；`glossary_map` 按需提供。
- `policy`：只写与运行时默认策略的差异，经 `normalize_policy` 合并，使样本独立于将来默认 disposition 的调整。
- `reason_code` 必须与 `rule_id` 对应，未知 code 会被测试拒绝。

## 校准后如何刷新

1. 在真实项目上运行 `check`，从 `quality_findings.jsonl` 抽样标注误报 / 漏报；
2. 摘录代表性样本时**改写或脱敏原文**，不得把私有游戏脚本原文直接提交到仓库；
3. 把标注为真实正例的句子写入对应 `positive`，把白名单 / 合法反例写入 `negative`；
4. 若抽样发现某 reason code 误报高，在默认 disposition 调整（issue #364 A2）之后重新运行本测试，确保「已知正例仍能报告」。
