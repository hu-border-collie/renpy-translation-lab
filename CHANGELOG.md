# 变更日志

本项目从 `v1.0.0` 开始记录面向使用者的发行变化。

## [Unreleased]

`main` 在 `v1.0.0` 之后的开发变化汇总如下；这些内容尚未对应新的稳定发行版，源码中的项目版本号仍会保持 `1.0.0`，直到下一次正式发版统一更新。

### 新增

- 为核心 Batch 命令增加版本化 JSON 结果 envelope、严格语义退出码、非交互显式 target、能力发现与命令 schema。
- 增加 Project Analysis 的生成、审查、发布与按场景注入生命周期，以及 report-only 最终审校 campaign。
- 增加按项目归集 Batch、同步、订正、关键词与分析调用的实际模型用量账本。
- 增加 Ren'Py Engine Adapter 的稳定扫描/occurrence/coverage 产物和写回计划校验。
- 增加提交恢复、成本估算、翻译 A/B 对比、同步预览写回与订正预览写回等完整工作流。
- 同步初译增加局部前后文（`sync.context_before` / `sync.context_after`，默认 30/10，限定同文件与 translate block 边界）、`sync.macro_setting_file` 风格设定注入，以及不依赖 RAG 开关、不受 `top_k_terms` 截断的术语命中（`normalize_map` / 保留词 / 不可翻译词，全部实际命中进入提示词）；上下文构造事实与 macro 指纹写入 manifest 并纳入预览指纹。

### 变更

- GUI 改为统一侧边导航与任务页自有状态，项目列表、上下文库、关键词、订正、同步翻译和批量翻译各自展示当前任务结果。
- 同步翻译默认只生成 diff 与 manifest 预览，必须显式确认后才写回；写回时重新校验项目、源快照和预览制品。
- 同步 manifest 增加 `prompt_context` 诊断（局部上下文设置、macro 身份、批次截断统计）；macro 文件的新增、删除或内容变化都会使旧预览无法写回。
- `doctor` 保持只读；初始化工作副本与生成 TL 模板分别使用 `bootstrap-work`、`generate-template`。
- Settings 增加项目列表、扩展、LiteLLM 供应商/凭据与模型目录等分区，并改进未保存设置和项目切换保护。
- CI 增加依赖锁、质量/类型/依赖审计、可选依赖安装、真实 Ren'Py SDK 和 provider 契约检查；PR-Agent 自动审查使用受限权限与固定 action commit。

### 安全与质量边界

- Batch `apply` 继续强制要求与当前 manifest/results 匹配的最近一次 `check=safe`，并在写回前重读源快照；`--force` 不绕过这些门禁。
- 订正与最终审校候选必须经过 `preview-revisions -> apply-revisions` 的独立 provenance、项目身份和源快照校验。
- 明确 `check=safe` 仅代表结构性可写回，不代表译文内容质量合格；正式交付前仍需机械质量检查与人工/LLM 语义审校。

## [1.0.0] - 2026-07-16

首个稳定源码发行版。

### 主要能力

- 提供 Gemini Batch 主工作流：`doctor -> build -> submit -> status -> download -> check -> apply`。
- 使用 manifest / identity v2，在写回前执行 `safe / warn / block` 分级与快照校验。
- 提供可选 PySide6 图形工作台，覆盖项目准备、批量翻译、同步翻译、关键词、订正、上下文库、设置与诊断日志。
- 提供本地 RAG、原文索引、可选 Story Memory、关键词提取与订正流程。
- 支持 Gemini 同步调用以及显式选择的 LiteLLM 同步后端。

### 验证范围

- 核心 Batch 链路已在约 11 万英文词规模的真实 Ren'Py 项目上完整跑通。
- GUI 批量翻译主路径已在约 3,300 待译行的真实项目副本上完成烟测。
- LiteLLM + DeepSeek 同步路径已完成小规模真实供应商烟测。
- 自动化测试覆盖 Windows 与 Ubuntu，并单独验证不安装 GUI 依赖时的 CLI 路径。

### 发行边界

- 本版本以源码 ZIP 交付，需要 Python 3.11+，不是零配置安装包。
- 不包含游戏解包、重新打包或完整游戏 QA。
- 批量写回前必须先执行 `check`，仅在结果为 `safe` 时执行 `apply`。
- 同步翻译会直接修改项目副本，不经过 Batch 的 `check -> apply` 闸门。

[Unreleased]: https://github.com/hu-border-collie/renpy-translation-lab/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/hu-border-collie/renpy-translation-lab/releases/tag/v1.0.0
