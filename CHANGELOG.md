# 变更日志

本项目从 `v1.0.0` 开始记录面向使用者的发行变化。

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

[1.0.0]: https://github.com/hu-border-collie/renpy-translation-lab/releases/tag/v1.0.0
