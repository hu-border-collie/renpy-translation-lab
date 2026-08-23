# #364 真实项目质量规则校准执行手册

> **状态：进行中。** A1（离线回归语料）已随 PR #381 合入 main；
> A3（校准报告工具）就绪；A2 的白名单/disposition 取值与 B 线的真实项目
> 统计、人工标注、重跑验证仍待真实项目数据。
> 以 issue #364 正文、当前 checkout 和 `scripts/quality_calibration_report.py --help` 为准。

文档地图：[规划与设计草案](README.md) · [项目文档](../README.md) ·
[Batch 工作流与安全检查](../batch_workflows.md)

## 范围

对 #359 首版机械质量规则做真实项目校准，产出可复现的离线质量基线。
本手册只描述执行路径；规则细节见 `translation_quality.py`，首版 reason code
的离线正反例见 `tests/fixtures/quality_real_samples/`。

## 仓库内已完成（无需私有数据）

- A1：`tests/fixtures/quality_real_samples/samples.json` + `tests/test_quality_real_samples.py`，
  每个首版 reason code 一个脱敏正例和一个白名单/合法反例。
- A3：`scripts/quality_calibration_report.py`，从 `quality_findings.jsonl`
  生成 Markdown 校准基线。

## 生成校准报告

```bash
# 直接输出到 stdout
python scripts/quality_calibration_report.py /path/to/quality_findings.jsonl

# 写入基线文件，每个 reason code 最多带 10 条样本
python scripts/quality_calibration_report.py /path/to/quality_findings.jsonl \
  --sample-limit 10 -o docs/plans/quality_calibration_baseline.md
```

报告按 finding 数降序排列每个 reason code，并包含：

- Summary：reason code / rule_id / finding 数 / 文件数；
- File distribution：每个文件的 finding 数；
- Samples：文件、行号、原文、译文的代表性样本。

脚本以 `translation_quality.load_findings(strict=True)` 读取，报告文件格式
损坏会直接报错，避免静默产出失真的基线。

基线文件始终以 LF 换行写出；生成要提交或跨机器对比的基线时，用
`--generated-at` 固定头部时间戳（可沿用上一版基线的取值），保证重跑字节
一致，两版基线 diff 只反映 findings 变化。

## B 线：真实项目执行步骤

1. 在真实项目包上运行 `check`，从标准输出或 manifest 的
   `last_quality_findings_path` 找到 `quality_findings.jsonl`。
2. 用 A3 工具生成第一版基线报告，得到每个 reason code 的数量与文件分布。
3. 抽样人工标注，重点审查 issue #364 指定的
   `suspicious_english_residue`、`cjk_latin_spacing`、
   `english_suffix_adjacent`、`speaker_label_untranslated`。建议每类至少
   标注 20–30 条，按下面模板记录：

   | reason code | 判定 | 样本（文件:行 原文→译文） | 处理建议 |
   |---|---|---|---|
   | `quality.language.suspicious_english_residue` | 误报 | `dialogue.rpy:42 Hello → TA也说过` | 加白名单 `TA` |
   | `quality.language.english_suffix_adjacent` | 真实正例 | `skill.rpy:7 … → 迷踪步ping` | 保持 warning |

4. 把标注结论拆成两类动作：
   - 误报且确定属于合法专名/缩写/混排 → A2 加 `allowed_latin_tokens`；
   - 整条规则误报高、确定性低 → A2 把默认 disposition 改为 `off`，作为项目 opt-in；
   - 漏报的新规则需求 → 记录回 issue #313，不在此 issue 扩展规则集。
5. 落 A2 diff 并同步 `translator_config.example.json`、文档和测试。
6. 在真实项目上重跑 `check`，生成第二版基线报告，对比第一版证明误报下降、
   已知正例仍能报告。
7. 把代表性样本脱敏后回流 `tests/fixtures/quality_real_samples/samples.json`，
   重新跑 `test_quality_real_samples` 和 `test_quality_calibration_report`。

## 脱敏与提交边界

- 私有游戏脚本原文、Batch 产物和 `logs/` 不得进入仓库；
- 进入 fixture 的样本必须改写或脱敏，只保留规则可复现所需的最小文本；
- 质量基线报告若含私有文本，只在 issue 评论中给摘要，不直接提交原文样本。

## 验收映射

| #364 验收标准 | 落地位置 |
|---|---|
| 每个首版 reason code 有数量统计 | A3 基线报告 |
| 抽样标注结论记录 | 本手册第 3 步 + issue/文档 |
| 白名单/默认 disposition 调整有 diff 和测试 | A2 PR |
| 真实样本 fixture 可离线运行且不含私有文本 | A1 + 本手册第 7 步 |
| 重跑后误报下降且正例仍报告 | 两份基线报告对比 |
| 质量报告作为回归基线 | `docs/plans/quality_calibration_baseline.md` 或等价路径 |
