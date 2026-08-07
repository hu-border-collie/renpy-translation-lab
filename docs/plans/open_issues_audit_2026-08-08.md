# 开放 Issues 审计（2026-08-08）

> **状态：时间点快照，不是长期事实来源。**
> 本文记录 `main@78ab050` 与 2026-08-08 GitHub 状态的交叉审计结果。
> Issue 的最新状态以 GitHub 正文、当前 checkout、现行文档和 CLI `--help` 为准。

文档地图：[规划与设计草案](README.md) · [项目文档](../README.md)

## 范围与方法

本轮覆盖当时全部 21 个开放 issues、唯一开放 PR #323，以及与开放项直接相关的
已关闭 issues / PR。判断同时核对：

- 当前 `main` 的代码、测试和现行文档；
- issue 正文、评论、checkbox、关闭状态与关联 PR；
- 尚未合并分支与 GitHub Actions 状态；
- #272 引用的主要官方本地化资料入口。

难度按一名熟悉仓库的维护者估算，包含实现、测试、文档和审阅：

| 等级 | 参考投入 |
| --- | --- |
| XS | 不超过 1 天 |
| S | 1–2 天 |
| M | 3–7 天 |
| L | 1–3 周 |
| XL | 多 PR，通常超过 3 周 |

## 总结

- **已实现、待合并**：#298、#315、#316，由 PR #323 统一收口。
- **Epic 记录明显失真**：#202、#265、#292。
- **Research 已有结论但状态未收口**：#305、#307、#308、#309。
- **其余有效待办**：#15、#147、#296、#297、#313、#314、#318、#320、
  #321、#322，以及被 #265 阻塞的 #272。

最需要立即纠正的记录是：

1. #265 P0–P2 已由 PR #286、#290、#291 合并，但 Epic 仍全部未勾选；
2. #292 的 #293、#294、#295、#299 已关闭；
3. #296 的机器输出覆盖已由 7/45 变成 8/45；
4. #202 的重定基线触发条件已经满足；
5. #298/#315/#316 不应再单独实施，应等待 PR #323 完成审阅后合并。

## 全量状态矩阵

| Issue | 审计结论 | 难度 | 下一步 |
| --- | --- | --- | --- |
| [#15 通用文本翻译](https://github.com/hu-border-collie/renpy-translation-lab/issues/15) | 需求有效，技术背景需按当前 Adapter / 安全写回重写 | XL | 拆成格式合同、TXT/JSONL、Markdown 三阶段；复杂格式首版排除 |
| [#147 Story Graph 试点](https://github.com/hu-border-collie/renpy-translation-lab/issues/147) | 核心代码已具备，真实采用状态无法仅由仓库验证 | S（试点） | 利用已完成 #139 的 `compare-variants` 做最小真实 A/B |
| [#202 Settings 页面化](https://github.com/hu-border-collie/renpy-translation-lab/issues/202) | 目标仍成立，旧 P0–P7 不可直接执行 | XL | 重定 10 页、lazy load、生命周期和测试基线；优先文档、导航/coordinator、LiteLLM |
| [#265 Engine Adapter / 版本化资产](https://github.com/hu-border-collie/renpy-translation-lab/issues/265) | P0–P2 已交付，checkbox 与最新评论失真；P3–P6 未完成 | XL（剩余） | 勾选 P0–P2，下一实施阶段从 ProjectSnapshot / reconciliation 开始 |
| [#272 第三引擎研究](https://github.com/hu-border-collie/renpy-translation-lab/issues/272) | 合理阻塞；矩阵仍只能视为初始假设 | M（研究） | 等 #265 P5/P6 后刷新官方资料并只选一个候选 |
| [#292 CLI/GUI 审核收口](https://github.com/hu-border-collie/renpy-translation-lab/issues/292) | 行政状态滞后；4/7 子项已关闭 | XS | 更新 checklist；PR #323 合并后仅余 #296/#297 |
| [#296 CLI 机器合同扩展](https://github.com/hu-border-collie/renpy-translation-lab/issues/296) | 方向有效；现为 8/45，parser 错误仍不是 JSON | M | 先做 parser P0，再选 revision/final-review 下一批命令 |
| [#297 GUI 生命周期](https://github.com/hu-border-collie/renpy-translation-lab/issues/297) | 未保存配置关闭保护已修，其余主要问题仍在 | XL | 先统一 shutdown/非阻塞 runner，再做 operation identity |
| [#298 Workbench 状态归属](https://github.com/hu-border-collie/renpy-translation-lab/issues/298) | PR #323 已实现 | XS | 等审阅检查完成后合并 |
| [#305 取证式 Review Agent Loop](https://github.com/hu-border-collie/renpy-translation-lab/issues/305) | 已决定暂缓，无当前实施入口 | L–XL（若重启） | 关闭为 not planned/parked；有真实静态审校基线后再开 |
| [#307 独立 Review Session](https://github.com/hu-border-collie/renpy-translation-lab/issues/307) | 已决定不建第二状态系统 | S–M（可选增量） | 关闭；若确有需要另开窄 `events.jsonl` issue |
| [#308 首次译法/历史 occurrence](https://github.com/hu-border-collie/renpy-translation-lab/issues/308) | 值得实现，Research 生命周期已结束 | M | P0 只做人类提示；复用 #320 corpus/occurrence API，不自动写 glossary |
| [#309 Review 解析恢复](https://github.com/hu-border-collie/renpy-translation-lab/issues/309) | 候选有效，但应缩成 fixture 与失败分类 | M–L | fixture spike → 结构化失败码 → targeted resume；receipt/repair 后评估 |
| [#313 写回门禁与质量报警](https://github.com/hu-border-collie/renpy-translation-lab/issues/313) | 高价值有效需求；核心仍把所有 warn 视为不可写回 | L–XL | 先版本化双门禁，再逐步增加高信号机械规则 |
| [#314 Settings 导航裁切](https://github.com/hu-border-collie/renpy-translation-lab/issues/314) | 当前代码仍可复现 | XS | 独立小 PR 修滚动/宽度并补 960×640 回归，不等待 #202 |
| [#315 空状态文案](https://github.com/hu-border-collie/renpy-translation-lab/issues/315) | PR #323 已实现 | XS | 随 #298 一起关闭 |
| [#316 首次使用 CTA](https://github.com/hu-border-collie/renpy-translation-lab/issues/316) | PR #323 已实现 | XS | 随 #298 一起关闭 |
| [#318 人工润色 Epic](https://github.com/hu-border-collie/renpy-translation-lab/issues/318) | P0 已完成，P1–P3 顺序准确 | L | #320 → #321 → #322；澄清 #313 是否为关闭硬依赖 |
| [#320 润色语料导出](https://github.com/hu-border-collie/renpy-translation-lab/issues/320) | 已具备开工条件，尚无运行时实现 | M | 先交付只读核心、CLI 与机器合同 |
| [#321 润色提案导入](https://github.com/hu-border-collie/renpy-translation-lab/issues/321) | 目标准确，硬依赖 #320 schema | L | 只生成 revision preview；继续复用 #294 后的安全 apply 合同 |
| [#322 润色 GUI](https://github.com/hu-border-collie/renpy-translation-lab/issues/322) | 目标准确，当前不应开工 | M–L | 等 #320/#321 与 PR #323；复用 #297 operation identity 合同 |

## 关键依赖

```text
#265 P0–P2 已完成 ─┬─> #320 ─> #321 ─> #322
                    ├─> #308（复用 occurrence / corpus 证据）
                    └─> #313（Adapter 提供结构规则）

#313 ─────────────────> #321 / #322（共享 quality finding，非硬阻塞）
#297 P0/P2 ───────────> #322、#202
PR #323 ──────────────> 关闭 #298、#315、#316
#202 ─────────────────> 可吸收 #314，但 #314 不应等待它
#265 P5/P6 ───────────> #272
#296 <────────────────> #309（CLI envelope 与业务 ingest 错误码）
```

必须避免把以下三类产物分别做成互相竞争的事实源：

- #265 的 ProjectSnapshot / 版本化翻译资产；
- #320 的 revision corpus；
- #308 的历史 occurrence 索引。

目标边界应是：ProjectSnapshot 作为未来版本事实，corpus 是只读交换视图，历史
occurrence 索引是可重建派生数据。

## 推荐执行顺序

1. 完成并合并 PR #323，关闭 #298/#315/#316；
2. 独立修复 #314；
3. 校准 #265、#292、#202、#296、#297 等 issue 记录；
4. 实施 #320 的只读语料导出；
5. 确定 #313 双门禁与 finding schema，再实施 #321；
6. 在 #322 前完成 #297 的最小 shutdown / operation identity 合同；
7. 让 #308 复用 #320 occurrence 数据，#309 先做 fixture spike；
8. 重定 #202 基线，并从 #265 P3 开启下一条战略主线；
9. 收档 #305/#307，保持 #147、#15、#272 为有明确触发条件的远期路线。

## Issue 维护规则

- 正文中的事实或 checkbox 错误时，应修改正文，并用一条日期评论记录依据；
- 仅有阶段性判断时只留评论，不把审计全文复制到每个 issue；
- 已得出“不做/暂缓”的 Research 应关闭或明确标记 parked；
- PR 已经覆盖的 issue 不再并行实施，等待 PR 合并关闭；
- 后续复核应生成新的日期快照，不回写本文伪装成当时事实。
