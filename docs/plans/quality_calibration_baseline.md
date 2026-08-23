# #364 真实项目机械质量校准基线

> **状态：已完成（2026-08-23）。** 本文只保存聚合统计和彻底脱敏后的
> 人工标注结论；真实脚本、finding 原文、Batch 产物和校准沙箱均不进入仓库。

## 校准口径

按 `Game_GloryHounds` → `Game_TemptationsBallad` → `Game_Dawntide` 的顺序，
在私有副本中恢复与历史 manifest 匹配的未翻译 TL 快照，再对复制的 Batch 包运行
现版 `check`。GloryHounds 的 687 个 identity 可由 live TL 的原文标记完整恢复；
另外两个项目使用 Ren'Py 8.5.2 在完整项目副本中重新生成 TL 模板。

所有进入统计的校准集都满足：预期 identity 100% 映射、`source_mismatch=0`、
`quality_unmatched_items=0`、结构失败 0，且 `writeback_gate.decision=allow`。
过程中从复制包排除的旧结构异常不计入质量规则精度：

| 项目 | 原包项 | 模板可映射 | 最终校准项 | 排除说明 |
|---|---:|---:|---:|---|
| GloryHounds | 687 | 687 | 681 | 4 个字体/Python 路径项；2 个被现行字段校验拒绝的旧结果 |
| TemptationsBallad | 21,307 | 21,268 | 20,953 | 39 个历史 identity；315 个现行结构校验失败 |
| Dawntide | 4,158 | 4,141 | 3,783 | 17 个历史 identity/字体项；358 个旧结构或不完整结果项 |

这些排除只发生在私有校准副本中；原始包和真实项目没有修改，也没有执行
`apply`。

## 调整前后统计

A2 将 `suspicious_english_residue` 和 `english_suffix_adjacent` 的默认 disposition
从 `warning` 改为 `off`；项目仍可显式设回 `warning` 或 `blocker`。其他规则及
内建 `allowed_latin_tokens` 不变。

| reason code | Glory 前→后 | Temptations 前→后 | Dawntide 前→后 | 合计前→后 |
|---|---:|---:|---:|---:|
| `quality.renpy.wait_tag_inside_cjk` | 0→0 | 902→902 | 28→28 | 930→930 |
| `quality.structure.unclosed_delimiters` | 0→0 | 161→161 | 0→0 | 161→161 |
| `quality.language.english_suffix_adjacent` | 2→0 | 127→0 | 43→0 | 172→0 |
| `quality.language.suspicious_english_residue` | 267→0 | 8,990→0 | 1,076→0 | 10,333→0 |
| `quality.typography.cjk_latin_spacing` | 35→35 | 1,301→1,301 | 283→283 | 1,619→1,619 |
| `quality.typography.halfwidth_punctuation` | 0→0 | 74→74 | 5→5 | 79→79 |
| `quality.typography.ascii_ellipsis` | 0→0 | 9→9 | 0→0 | 9→9 |
| `quality.glossary.term_not_applied` | 20→20 | 613→613 | 0→0 | 633→633 |
| `quality.speaker.label_untranslated` | 0→0 | 0→0 | 0→0 | 0→0 |
| `quality.completeness.interjection_untranslated` | 0→0 | 0→0 | 0→0 | 0→0 |
| `quality.garbled.known_bad_phrase` | 0→0 | 0→0 | 0→0 | 0→0 |
| **全部 findings** | **324→55** | **12,177→3,060** | **1,435→316** | **13,936→3,431** |

默认噪声下降 10,505 条（75.4%），同时 1,619 条 CJK/Latin 间距报警及其他默认
warning 规则保持不变。

## 人工标注结论

每类按文件顺序做等距抽样，目标 30 条/项目；不足时全量标注。样本只在私有
工作目录保存，本文仅记录判定数量。

| 规则 | GloryHounds | TemptationsBallad | Dawntide | 结论 |
|---|---:|---:|---:|---|
| `suspicious_english_residue` | 0 TP / 30 FP | 6 TP / 24 FP | 0 TP / 30 FP | 84/90 为合法人名、地名或代码文本，默认 warning 噪声过高 |
| `cjk_latin_spacing` | 30 TP / 0 FP | 29 TP / 1 FP | 28 TP / 2 FP | 87/90 为真实紧贴，继续默认 warning |
| `english_suffix_adjacent` | 0 TP / 2 FP（全量） | 0 TP / 30 FP | 0 TP / 30 FP | 完整专名因 `-er/-ly/-s` 等结尾误触发，默认关闭 |
| `speaker_label_untranslated` | 0 条 | 0 条 | 0 条 | 无真实 finding，证据不足，不改默认策略 |

英文词缀规则的脱敏正例（如中文词后残留 `ping` / `s`）仍由离线 fixture 以显式
`warning` 策略覆盖，证明项目 opt-in 后仍能报告。英文残留规则同样保留显式 opt-in
正例。

## 白名单决定与剩余风险

本轮不增加默认 `allowed_latin_tokens`。跨项目出现的 `Mrs`、`Miss`、`Master`
等词既可能是合法保留，也可能是真实未本地化；而当前白名单会同时屏蔽英文残留、
间距和词缀规则，把它们加入全局白名单会制造已观察到的间距漏报。

剩余风险：两条默认关闭的语言规则会降低开箱召回，需高要求项目显式 opt-in；
speaker label 在本矩阵中没有 finding，尚不能证明真实项目精度；历史包中被现行
结构合同拒绝的条目不属于本基线。若后续要提升英文残留/词缀召回，应在独立 issue
中先区分“专名白名单”和“排版边界白名单”，不在 #364 扩展新规则。
