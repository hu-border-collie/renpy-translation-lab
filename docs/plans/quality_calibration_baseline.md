# #364 脱敏质量校准基线

> **状态：已完成。** 本文仅保留聚合区间、比例和脱敏后的人工标注结论；
> 项目名称、精确规模、真实脚本、finding 原文、Batch 产物和校准沙箱均不进入仓库。

## 校准口径

在三个私有项目副本（Project A、Project B、Project C）中恢复与历史 manifest
匹配的未翻译 TL 快照，再对复制的 Batch 包运行现版 `check`。三个项目分别是
千行以内、两万行量级和数千行量级，用于覆盖不同规模。

所有进入统计的校准集都满足：预期 identity 100% 映射、`source_mismatch=0`、
`quality_unmatched_items=0`、结构失败 0，且 `writeback_gate.decision=allow`。
过程中从复制包排除的少量历史 identity、字体项、路径项和旧结构异常不计入
质量规则精度。排除只发生在私有校准副本中；原始包和真实项目没有修改，也
没有执行 `apply`。

## 调整前后结论

A2 将 `suspicious_english_residue` 和 `english_suffix_adjacent` 的默认 disposition
从 `warning` 改为 `off`；项目仍可显式设回 `warning` 或 `blocker`。其他规则及
内建 `allowed_latin_tokens` 不变。

跨三个项目汇总后，默认 findings 下降约四分之三；减少项几乎全部来自上述两条
高噪声语言规则。CJK/Latin 间距、未闭合分隔符、半角标点、ASCII 省略号和术语
未应用等默认 warning 规则保持不变。脱敏后的方向性结果如下：

| reason code | 调整结果 |
|---|---|
| `quality.renpy.wait_tag_inside_cjk` | 保持不变 |
| `quality.structure.unclosed_delimiters` | 保持不变 |
| `quality.language.english_suffix_adjacent` | 默认关闭；仍可显式 opt-in |
| `quality.language.suspicious_english_residue` | 默认关闭；仍可显式 opt-in |
| `quality.typography.cjk_latin_spacing` | 保持默认 warning |
| `quality.typography.halfwidth_punctuation` | 保持默认 warning |
| `quality.typography.ascii_ellipsis` | 保持默认 warning |
| `quality.glossary.term_not_applied` | 保持默认 warning |
| `quality.speaker.label_untranslated` | 样本证据不足，不改默认策略 |

## 人工标注结论

每类在三个项目中按文件顺序等距抽样，每个项目最多数十条；不足时全量标注。
样本只在私有工作目录保存，本文仅记录脱敏后的比例和判断。

| 规则 | 脱敏结论 |
|---|---|
| `suspicious_english_residue` | 超过九成样本为合法人名、地名或代码文本，默认 warning 噪声过高 |
| `cjk_latin_spacing` | 超过九成样本为真实紧贴，继续默认 warning |
| `english_suffix_adjacent` | 抽样未确认真实阳性，完整专名常因英文词尾误触发，默认关闭 |
| `speaker_label_untranslated` | 校准矩阵中没有足够 finding，不据此扩大规则 |

英文词缀规则的脱敏正例（如中文词后残留 `ping` / `s`）仍由离线 fixture 以显式
`warning` 策略覆盖，证明项目 opt-in 后仍能报告。英文残留规则同样保留显式
opt-in 正例。

## 白名单决定与剩余风险

本轮不增加默认 `allowed_latin_tokens`。跨项目出现的称谓既可能是合法保留，也
可能是真实未本地化；当前白名单会同时屏蔽英文残留、间距和词缀规则，把它们
加入全局白名单会制造已观察到的间距漏报。

剩余风险：两条默认关闭的语言规则会降低开箱召回，需高要求项目显式 opt-in；
speaker label 在本矩阵中证据不足；历史包中被现行结构合同拒绝的条目不属于本
基线。若后续要提升英文残留/词缀召回，应在独立 issue 中先区分“专名白名单”
和“排版边界白名单”，不在 #364 扩展新规则。
