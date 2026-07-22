# 文译（Wenyi）参考对照与 Batch 主路径增强计划

> **状态：进行中**（2026-07-22 对照刷新；#256 起落地 Project Analysis 合同）
> **性质：设计与取舍记录，不是用户手册。**
> 用户可见行为写入现行文档（如 `batch_workflows.md`、`context_systems.md`）；实现落地后本计划可迁入 `docs/archive/`。

## 1. 背景

本地参考克隆：

- 路径：`C:\RenPy_Workspace\_ref\wenyi`（仓库外，不随 lab 提交）
- 上游：`https://github.com/BigDawnGhost/wenyi`（文译 / Wenyi，`trans-novel`）
- 本轮对照基线：本地 `main@b796e45`（已与 origin 对齐），发行标签至 **v0.3.4**

文译是**多语言小说 → 中文**的 CLI 工具：EPUB/FB2/TXT/Markdown/HTML/PDF 等，强调全书预扫、滚动上下文、动态术语、可选润色与可独立续跑的最终审校。
本仓库（Ren'Py Translation Lab）是**游戏 `tl/*.rpy` 翻译工作台**，稳定版主路径是 **Gemini Batch 作业流** + check/apply 写回闸门；同步 CLI/GUI 为辅。

**代码复用立场（默认）：**

- **默认不复制**文译源码；对照其阶段划分、可续跑状态与用量归因等**产品意图**，在 lab 内独立实现。
- 文译为 MIT；若将来复制 substantial portions，须保留 `Copyright (c) 2026 BigDawnGhost` 与 MIT 文本。仅 issue 致谢不构成许可证合规。
- 不把文译作为 submodule 或运行时依赖。

本文回答：

1. 文译哪些做法值得学、哪些不该搬；
2. 与 lab 已有能力（尤其**订正**、Batch 闸门、上下文层）如何区分；
3. 在 **Batch 为主** 的前提下，增强方向如何映射到已拆 GitHub issues。

## 2. 架构对照（决定取舍）

| 维度 | 文译（当前） | Lab（本仓库） |
|------|--------------|----------------|
| 翻译对象 | 长篇小说电子书 | Ren'Py `game/tl/<lang>/` |
| 主路径 | **同步在线** LLM 调用（进程内长跑） | **Batch 异步**（`build → submit → download → check → apply`） |
| 「batch」含义 | 切段大小（字符预算） | 远端 Batch 作业 + package/manifest |
| 全书理解 | `pipeline.book_understanding`：预扫 → 章梗概 + 全书概览，直接注入翻译 prompt | 规划中：**Project Analysis**（#253/#256–#254），draft → 人工 publish 后才注入 |
| 质量手段 | 多阶段 LLM（分析/梗概/译/润/审）+ 段数对齐 | glossary / macro / RAG / Story Memory + **规则闸门** + 订正/关键词 |
| 最终审校 | 独立 `review` 命令；可 `--force` / 可选 `autofix_severe`；默认关闭 | 规划中：#255 report-only campaign → 现有 revision preview/apply |
| 用量 | `state/.../usage.json`：provider-neutral，按 tier/stage 归因，跨续跑增量合并 | 规划中：#252 项目级账本；现有估算 + 分散在 results 的 raw usage |
| 写回 | 组装 EPUB 等到 `output/` | 写回 `.rpy`，强调 identity、快照、safe check |
| 状态目录 | `state/<book-slug>/`（manifest、chapters、context、glossary.db、usage…） | `logs/batch_jobs` manifest + 可选 `translation_context/` 上下文库 |
| 代码自称 | multi-agent（实为 role-specialized LLM stage） | 无 Agent 层；可复用模块 + CLI + 可选 GUI |

文译 `Agent` 实质是 **固定 prompt + 档位 + orchestrator 调度**，不是自治多智能体。命名可参考为「阶段/角色」，不宜引入 lab 的 Agent 抽象层。

### 2.1 文译近期能力（对照时注意）

相对 2026-07-21 初版对照，上游已更明显的部分：

| 能力 | 位置 / 说明 | Lab 映射 |
|------|-------------|----------|
| `prepare` 仅预扫分析 | 先生成风格指南与初始术语，再 `translate` | 对应「准备阶段」思路；lab 用 `bootstrap-*` / 未来 analysis 子命令，不塞进 translation submit |
| 独立 `review` 续跑 | `review` / `--force` / `--fix`；`REVIEW_*` 状态 | #255 最终审校 campaign；**默认无 autofix 写 `.rpy`** |
| 全书理解 map-reduce | `agents/synopsis.py` 分组归并章梗概 | #254 路线感知层级摘要；模型改为 Ren'Py route，非线性章 |
| 用量账本 | `llm/usage.py` + RunStore `usage.json` | #252；采集 Batch results + 同步 backend，项目隔离 |
| 模型分档 | `llm.tiers`: strong / cheap / fast | 后续阶段级路由；**先有 #252 真实数据** |
| 断点续跑 | 章/批状态写盘，再跑跳过已完成 | lab 已有 Batch package/manifest；审校另建 digest 模型（#255） |
| 术语作用域 | `glossary_scope: chapter` | 可在 build 时按 chunk 裁剪 glossary（P1a，与 epic 可并行） |

## 3. 明确不学 / 不搬

| 项 | 原因 |
|----|------|
| 整包 multi-agent 类结构 | 增加抽象、不提升 Batch 闸门安全性 |
| 默认全量「润色 pass」嵌进主译 | 成本高；对含标签/变量的 TL 风险大；与 Batch 物化请求模型不契合 |
| 线性「章节」作为唯一结构 | Ren'Py 是分支剧本，至少区分 file / label / scene / route / global |
| 自动把模型输出写入正式术语库 / 知识图 | lab 要求 draft → 人工确认 → published；不自动写 `glossary.json` / `story_graph.json` |
| `autofix_severe` 式审校直接改文 | 任何正文修改继续走 revision preview/apply + 源快照校验 |
| EPUB assemble / MinerU PDF 等 | 领域无关 |
| 以同步长跑替换 Batch 主路径 | 违背 lab 已定稳定范围与交付形态 |
| 回译抽检作为主质检 | 游戏短句 + code 混排更适合规则 check + 可选订正 |

## 4. 润色 vs 订正（已有能力边界）

| | 文译 · 润色 (polish) | Lab · 订正 (revision) |
|--|----------------------|------------------------|
| 意图 | 文学性二次加工，倾向整批改写 | **按需**修订：`should_update` 可 false |
| 输入 | 以当前中文为主（+ 风格/术语） | **原文 + 当前译文** + 上下文/RAG 等 |
| 流程位置 | 初译流水线内嵌一步 | **独立** `build-revisions → preview → apply` |
| 工程闸门 | 段数对齐失败则退回原文 | 预览报告 + 写回前旧译文快照校验 |
| 输出可解释性 | 弱 | `reason` + revision preview 表 |

**结论：** lab 的订正已经覆盖「改已有译文」的主需求，且更贴 Ren'Py。
**不要**再平行引入一个默认全量 polish 阶段去替代订正。若未来要「风格抛光」，应作为**可选、显式、可预览**的订正策略或独立 package 模式，而不是翻译 submit 的隐式附带步骤。

## 5. 值得学的方向（按 Batch 主路径改写）

下列意图来自文译，但实现必须落在 **build 时注入 / 多轮 package / 配置开关 / 独立子命令**，而不是「边译边改」的在线流水线。

### 5.1 开翻前「项目理解」强化 → Epic #253

**文译：** 预扫源文 → 分章梗概 + 全书概览，再注入翻译 prompt。
**Lab 落点（已拆 issue）：**

| 阶段 | Issue | 要点 |
|------|-------|------|
| 产物与依赖合同 | **#256**（当前实现） | schema / evidence / fingerprint / draft·published / 失效语义；不调模型 |
| 路线感知生成与发布 | #254 | chunk → label/scene → route → global brief；仅 published 注入 |
| 最终审校 campaign | #255 | 稳定上下文快照；report-only → revision candidates |

**与文译差异：**

- 不按线性章节建模；未解析动态跳转标 `unresolved`，不虚构单一路线
- 每项摘要保留 evidence item IDs / 行号 / checksum
- draft 不得静默进入 prompt；stale published 不得静默继续注入
- 产物集中在项目 `translation_context/project_analysis/`（或 tool 模式等价路径），不散落到 Batch package

### 5.2 术语：先抽后译 / 多轮，而非译中实时

**文译：** 译完一批立刻抽术语，下一批在线用上。
**Lab 现状：** glossary + `build-keywords` / 关键词合并；Batch **无法**在同一次作业内让后 chunk 看到前 chunk 新术语。
**Batch 落点：**

- 推荐节奏：`关键词/术语 pass` → 人工或半自动合并 glossary → 再 `build` 翻译
- 可选：大项目「译完 → 抽术语 → 订正统一术语」第二轮 package
- 可参考文译「本章作用域只注入相关词条」——在 **build 时**按 chunk 裁剪 glossary，省 token

### 5.3 模型分档 strong / cheap / fast

**文译：** 分析/润色用强档，梗概/审校用便宜档。
**Lab 落点：** 主译 Batch、关键词、订正、后续 analysis/review 可配置不同模型；**阶段级自动路由应基于 #252 真实用量**，不要在无数据时硬编码。

### 5.4 阶段开关与成本可见 → #252

**文译：** `pipeline.polish/review/...` 可关；`usage.json` 累计实际 token。
**Lab 落点：** 已有多子命令 + `batch_cost_estimate`；#252 建立 provider-neutral 实际用量账本（Batch results + 同步路径，可去重、可按 project/task/stage 聚合）。

### 5.5 状态与续跑清晰度

**文译：** `state/` + 再跑 `translate`/`review` 跳过已完成 digest。
**Lab：** Batch 作业以 package/manifest 为中枢；Project Analysis 与最终审校各自用 fingerprint / input digest（#256/#255），**不**为「更像文译」重做翻译主路径 store。

## 6. 建议优先级（与 issue 对齐）

| 优先级 | 主题 | Issue / 依赖 | 备注 |
|--------|------|--------------|------|
| P0 | 对照文档刷新 + issue 拆分 | 本文 + #252–#256 | 本轮完成对照刷新 |
| **P1** | Project Analysis 阶段 1 合同 | **#256** | 地基；不调模型 |
| P1 | 实际模型用量账本 | #252 | 可与 #256 并行，勿混 PR |
| P1 | 路线感知生成与发布 | #254 ← #256 | 仅 published 注入 |
| P1 | 最终审校 campaign | #255 ← #256 | report-only → revision |
| P1a | 术语：build 时按 chunk 裁剪 + 文档化「先 keyword 后翻译」 | 现有 keyword/glossary | 低风险、贴 Batch；可并行 |
| P2 | 任务类型模型分档 / 路由 | 配置 + GUI；宜参考 #252 | 跨 CLI/GUI |
| P2/P3 | 可选「风格向」订正策略 | 现有 revision | 显式，非默认 polish |
| 低 | Story Graph 试点 | #147 | 不被 #253 吞并 |
| 不做 | Agent 框架、默认同进程润色链、换主路径为同步、审校 autofix 写 rpy | — | 见 §3 |

## 7. 与现行文档/代码的衔接

| 主题 | 现行入口 |
|------|----------|
| Batch / 订正 / 关键词 | `docs/batch_workflows.md`，`gemini_translate_batch.py`，`translation_core.py` |
| RAG / 索引 / Story Memory / Project Analysis | `docs/context_systems.md` |
| CLI/GUI 同步 | `CONTRIBUTING.md`、`Agents.md` |
| 项目边界 | `docs/project_notes.md` |
| 原子写 / 指纹 | `atomic_io.py`；Batch check fingerprint 在 `gemini_translate_batch.py` |

落地功能时：

1. 先可复用核心与 CLI；
2. 再 GUI（CONTRIBUTING DoD）；
3. 更新现行手册，而不是只改本文。

## 8. 参考阅读（仓库外）

对照时以本地 `_ref/wenyi` 为准（实现前 `git pull`）：

- 流水线：`docs/zh/pipeline.md` / `docs/pipeline.md`
- 配置：`config.yaml`（`pipeline.*`、`llm.tiers`、`glossary_scope`）
- 全书理解：`trans_novel/agents/synopsis.py`
- 审校：`trans_novel/agents/reviewer.py` + orchestrator 中 review 阶段
- 状态与用量：`trans_novel/pipeline/runstore.py`、`trans_novel/llm/usage.py`
- 润色（仅对照边界）：`trans_novel/agents/polisher.py` ↔ lab `build_revision_*`

## 9. 开放问题

1. Project Analysis brief 的权威来源优先 TL 注释、解包原文，还是 source index / keyword chunk summaries？（#254 实现时定稿）
2. 术语「本章/本 chunk 裁剪」是否与 preserve_terms / identity 锁定词冲突，如何测？
3. 是否需要在 doctor 中提示「大批量前建议先 keywords → 合并 glossary」？
4. #252 账本与 #256 analysis fingerprint 是否共享 digest 工具函数（宜小公共模块，避免互相 import 重量级 batch 入口）？

## 10. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-07-21 | 初版：对照文译、Batch 主路径取舍、润色 vs 订正、候选优先级 |
| 2026-07-22 | 刷新本地 wenyi 至 `b796e45` / v0.3.4；补 prepare/review/usage/tiers；对齐 #252–#256；明确默认不复制源码 |
