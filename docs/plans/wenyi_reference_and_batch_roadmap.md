# 文译（Wenyi）参考对照与 Batch 主路径增强计划

> **状态：草案 / 规划中**（2026-07）  
> **性质：设计与取舍记录，不是用户手册。**  
> 实现落地后，应把「用户可见行为」写进现行文档（如 `batch_workflows.md`、`context_systems.md`），本文可迁入 `docs/archive/`。

## 1. 背景

本地参考克隆：

- 路径：`C:\RenPy_Workspace\_ref\wenyi`（仓库外，不随 lab 提交）
- 上游：`https://github.com/BigDawnGhost/wenyi`（文译 / Wenyi，`trans-novel`）

文译是**多语言小说 → 中文**的 CLI 工具：EPUB/FB2/TXT/HTML/PDF 等，强调全书预扫、滚动上下文、动态术语、可选润色与审校。  
本仓库（Ren'Py Translation Lab）是**游戏 `tl/*.rpy` 翻译工作台**，稳定版主路径是 **Gemini Batch 作业流** + check/apply 写回闸门；同步 CLI/GUI 为辅。

本文回答：

1. 文译哪些做法值得学、哪些不该搬；
2. 与 lab 已有能力（尤其**订正**）如何区分；
3. 在 **Batch 为主** 的前提下，若要增强，应拆成哪些方向（非实现规格）。

## 2. 架构对照（决定取舍）

| 维度 | 文译 | Lab（本仓库） |
|------|------|----------------|
| 翻译对象 | 长篇小说电子书 | Ren'Py `game/tl/<lang>/` |
| 主路径 | **同步在线** LLM 调用（进程内长跑） | **Batch 异步**（`build → submit → download → check → apply`） |
| 「batch」含义 | 切段大小（字符预算） | 远端 Batch 作业 + package/manifest |
| 质量手段 | 多阶段 LLM（分析/梗概/译/润/审） | glossary / macro / RAG / Story Memory + **规则闸门** + 订正/关键词 |
| 写回 | 组装 EPUB 等到 `output/` | 写回 `.rpy`，强调 identity、快照、safe check |
| 代码自称 | multi-agent | 无 Agent 层；CLI + 可选 GUI |

文译代码里的 `Agent` 实质是 **role-specialized LLM stage**（固定 prompt + 档位 + orchestrator 调度），**不是**自治多智能体。命名可参考为「阶段/角色」，不宜引入 lab 的 Agent 抽象层。

## 3. 明确不学 / 不搬

| 项 | 原因 |
|----|------|
| 整包 multi-agent 类结构 | 增加抽象、不提升 Batch 闸门安全性 |
| 默认全量「润色 pass」嵌进主译 | 成本高；对含标签/变量的 TL 风险大；与 Batch 物化请求模型不契合 |
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

下列意图来自文译，但实现必须落在 **build 时注入 / 多轮 package / 配置开关**，而不是「边译边改」的在线流水线。

### 5.1 开翻前「项目理解」强化（P1 候选）

**文译：** 预扫源文 → 分章梗概 + 全书概览，再注入翻译 prompt。  
**Lab 现状：** macro setting、Story Memory、source index、RAG 已有；缺的是「一键从 TL/原文生成可审 brief」的标准化步骤。  
**Batch 落点：**

- 独立准备阶段或 doctor/bootstrap 增强（**不**塞进同一次 translation submit）
- 产物写入 work 目录或 store，供后续 `build` 读入 system/user 固定前缀
- 可开关、可人工编辑后再 build

### 5.2 术语：先抽后译 / 多轮，而非译中实时（P1 候选）

**文译：** 译完一批立刻抽术语，下一批在线用上。  
**Lab 现状：** glossary + `build-keywords` / 关键词合并；Batch **无法**在同一次作业内让后 chunk 看到前 chunk 新术语。  
**Batch 落点：**

- 推荐节奏：`关键词/术语 pass` → 人工或半自动合并 glossary → 再 `build` 翻译
- 可选：大项目「译完 → 抽术语 → 订正统一术语」第二轮 package
- 可参考文译「本章作用域只注入相关词条」——在 **build 时**按 chunk 裁剪 glossary，省 token

### 5.3 模型分档 strong / cheap / fast（P2 候选）

**文译：** 分析/润色用强档，梗概/审校用便宜档。  
**Lab 落点：** 主译 Batch、关键词、订正、probe 等可配置不同模型或 tier；配置进 `translator_config` + GUI 设置同步（见 CONTRIBUTING）。

### 5.4 阶段开关与成本可见（P2 候选）

**文译：** `pipeline.polish/review/...` 可关。  
**Lab 落点：** 已有多子命令；可加强「推荐流水线说明 + 成本估算」一致性（`batch_cost_estimate` 等），避免暗示用户必须开满上下文层。

### 5.5 状态与续跑清晰度（P3 对照，非必须重写）

**文译：** `state/` + 再跑 `translate` 跳过已完成。  
**Lab：** `logs/batch_jobs` manifest 已是作业状态中枢。以对照改进文档与 doctor 提示为主，避免为「更像文译」重做 store。

## 6. 建议优先级（草案）

| 优先级 | 主题 | 依赖 | 备注 |
|--------|------|------|------|
| P0 | 本文档定稿 + 是否拆 GitHub issues | — | 无代码 |
| P1a | 术语：build 时按 chunk 裁剪注入 + 文档化「先 keyword 后翻译」 | 现有 keyword/glossary | 低风险、贴 Batch |
| P1b | 开翻前 brief/预扫产物（可选 bootstrap） | source index / story 可选 | 需防幻觉，产物可编辑 |
| P2 | 任务类型模型分档 | 配置与 GUI 同步 | 跨 CLI/GUI |
| P2/P3 | 可选「风格向」订正策略（显式，非默认 polish） | 现有 revision | 仅当用户反馈机翻腔严重 |
| 不做 | Agent 框架、默认同进程润色链、换主路径为同步 | — | 见 §3 |

## 7. 与现行文档/代码的衔接

| 主题 | 现行入口 |
|------|----------|
| Batch / 订正 / 关键词 | `docs/batch_workflows.md`，`gemini_translate_batch.py`，`translation_core.py` |
| RAG / 索引 / Story Memory | `docs/context_systems.md` |
| CLI/GUI 同步 | `CONTRIBUTING.md` |
| 项目边界 | `docs/project_notes.md` |

落地功能时：

1. 先可复用核心与 CLI；
2. 再 GUI（CONTRIBUTING DoD）；
3. 更新现行手册，而不是只改本文。

## 8. 参考阅读（仓库外）

- 文译流水线说明：`_ref/wenyi/docs/zh/pipeline.md`
- 文译配置：`_ref/wenyi/config.yaml`（`pipeline.*`、`llm.tiers`）
- 润色实现：`_ref/wenyi/trans_novel/agents/polisher.py`（对照 lab `build_revision_*`）

## 9. 开放问题

1. P1b brief 的权威来源优先 TL 注释、解包原文，还是 source index？
2. 术语「本章/本 chunk 裁剪」是否与 preserve_terms / identity 锁定词冲突，如何测？
3. 是否需要在 doctor 中提示「大批量前建议先 keywords → 合并 glossary」？

## 10. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-07-21 | 初版：对照文译、Batch 主路径取舍、润色 vs 订正、候选优先级 |
