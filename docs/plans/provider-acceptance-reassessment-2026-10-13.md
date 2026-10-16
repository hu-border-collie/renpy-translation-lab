# Provider 实测与 issue 验收补证（2026-10-13）

本轮补充 #344 / #431 的真实连接、阶段路由和中断恢复证据，并对照 #508 已有章节交付。
这不是全部开放 issue 已通过的声明；不关闭 issue，不把结构安全、模型质量和游戏运行混为一项。

后续状态：本文修复已随 [PR #556](https://github.com/hu-border-collie/renpy-translation-lab/pull/556)
于 2026-10-15 合并。双连接、完全不带 Gemini Key 的五阶段与 #508 独立复审见
[10-15 续验记录](issue-431-508-acceptance-2026-10-15.md)；当日带有 Gemini Key 的环境未暴露
后续确认的同步入口凭据前置检查缺陷。下文保留 10-13 的证据与当时发布状态。

## 样本、环境与基线

- Windows、Python 3.14.0；独立 venv 的 `include-system-site-packages=false`，安装
  google-genai 1.57.0 / PySide6 6.11.1，运行时确认找不到 LiteLLM。
- 使用原创 Ren'Py 两句对白 fixture，包含独立源脚本、原生 TL 槽及固定 scope；没有新增私有
  游戏文本出站。写回只发生在这些隔离 fixture，正常经过 check、绑定 preview 和 apply。
- HEAD 为 `7e2cbd8a7f190ea300a5caf85c930644370cbbbc`；实测还包含本地未提交修复。
  最终 tracked Python 内容摘要为
  `cffd0c56745fd19a6d3e2762226f38234650e359408460cee9b3686ca15045ec`。
  初始源码副本、各次修复摘要、配置、命令、失败和回执均保留；这是实测时的工作树快照。
  2026-10-14 整理 PR 时，代码修复提交为 `a66426e`，与上述已测试 Python 内容逐字节一致；
  PR 从 `main` 建立，未包含原分支独立的 #272 研究提交。
- DeepSeek 使用 `adapter=openai_compatible`、官方 `api.deepseek.com`、`deepseek-flash`、
  `json_object`，输出上限 4096，超时 120 秒。只对实验进程绕过 DS 环境代理，未改系统代理。
  Gemini Sync 使用原生 `gemini` adapter、3.1 Flash-Lite、输出上限 2048。
- 凭据仅在进程内读取并传给子进程；公开记录不包含值、请求正文、私有游戏或本机完整路径。

## 真实结果

| 路径 | 证据 | 结论与边界 |
| --- | --- | --- |
| Gemini Sync 初译 | `sync-start` 接受 2/2；check=ready、writeback gate=allow；隔离 apply 1 文件/2 条 | 初译工作流通过；不同于此前的中性 schema 请求 |
| DS 直连目录 / probe | 真实 models GET、1 次结构化探测；auth/generation/JSON/usage/reasoning 均 pass | 无 LiteLLM 环境通过；没有把网络绕过代理当作 adapter 证明 |
| DS 直连初译 | 2/2，正式 check / preview / apply，质量报警 0 | 仅原创 fixture 的工作流与结构证据 |
| DS 术语 / 订正 | `sync-keywords` 有实际候选及概要；`sync-revisions` 生成 ready 预览 | 初次术语调用还发生一次定点重试；费用计入全部请求，未写回订正 |
| DS 项目分析 | 1 label、1 route、brief 的真实 map-reduce 全部完成 | 首次 brief 被 JSON 模式拒绝；修复后复测通过，保留原失败 |
| DS 最终审校 | 原创 scope 的 Agent coverage review 经正式 import；build → run-sync → status/findings 完成 1 unit | report-only，0 findings 不代表模型语义质量已达标 |
| GUI 五阶段 | offscreen QApplication、生产 CliRunner / Qt QProcess 及五个实际 workflow 完成上述能力 | 真实调用、无假 Provider；不是人工点击 MainWindow 的完整界面验收 |
| fresh PA 发布 / 消费 | 对真实生成草稿做明确 Agent 复核订正，正式 publish；新建原生 revision prompt 实际包含 published brief | 原创 fixture、0 次额外模型调用；不是生产项目人工 PA 验收或质量增益 A/B |
| 第二个直连地址 | Google 官方 OpenAI-compatible models GET 两次均 HTTP 400：地域不支持 | 没有提交生成；双地址成功、斜杠模型 ID 与真实跨连接隔离仍未完成 |

Google 兼容地址依据[官方说明](https://ai.google.dev/gemini-api/docs/openai)。失败保留为连接证据，
不通过改写成同一个供应商的两个 URL 来满足双连接验收。

GUI harness 的前两次尝试分别在生成前遇计数器错误、在初译完成后遇 Qt 对象生命周期崩溃。
修正 harness 并保留已生成成果后，从 check 继续；没有重新执行那次成功初译。
这两次失败不归因于翻译模型。测试环境初始 TEMP 权限错误也单独记录，随后使用可写的外部 TEMP。

## 中断恢复：两个边界分别验证

1. **T3 已提交后中断。** 两句分别为两个请求；第一条真实结果与 accepted winner 原子入库后
   测试子进程退出 88。租约到期后执行正式 `sync-resume`，只发送第二条。
   两个 request 均只有 ordinal=1，原第一条 attempt ID 保持，实际出站共 2 次，重复派发 0。
2. **响应返回、T3 前中断。** 一次真实 DS 请求已返回、合同已通过，子进程在入库前退出 86。
   恢复后保留 1 个 `outcome_unknown` attempt、0 accepted；出站仍为 1 次。
   `billing_unknown_attempts=1` 不会因实验旁路留有回执而自动清除，也没有授权重试未知请求。

中断钩子只位于实验子进程，调用生产 store/executor；短租约和退出位置明确记录。
“没有重复派发”由实际出站记录与原子状态证明；未核对供应商账单，不声称证明所有上游扣款行为。
另一次 Gemini 中断 smoke 未取得成功回执，报告 provider_error；未知费用保留，DS 通过不覆盖该失败。

## 实测发现的三处最小修复

1. `model_routing_reader.resolve_runtime_plan` 的全局 final_review=Batch 限制，错误阻断合法的
   显式 Sync 最终审校配置。允许它被其他阶段加载；Batch 专属命令仍按阶段拒绝 Sync 包。
   未显式配置的最终审校默认仍由原有合同决定。
2. 普通文本 Project Analysis 被 JSON-capable Profile 的 response format 强制约束。
   分析请求现在显式声明 `response_mime_type=text/plain`；直连后端遵守该任务合同。
   同时声明普通文本与 JSON schema 时调用前拒绝；需要 JSON 的任务保留原校验。
3. Gemini SDK `model_dump()` 回执使用 `usage_metadata` / `finish_reason`，共用提取器原先只读
   REST camelCase，导致已有用量漏记。本轮真实原始回执含 416 输入、98 输出、514 total 和 STOP。
   增加 SDK 字段读取兼容并离线重放确认；未改写历史回执或伪造一次新的 Provider 成功。

配置结构、GUI 参数入口和写回合同没有新增分支；修复落在共用 reader、请求构造与回执读取中。

## 调用、费用及证据限制

DS 真实生成 22 次：21 次 HTTP 成功、1 次 HTTP 400；其中 20 个不同成功回执有可追溯 usage，
probe 和被拒绝 brief 未记录到用量，继续留最高费用预留。Gemini 有 1 次初译成功及
1 次中断 smoke 失败；另有 DS models GET 与两个失败的 Google models GET。

按 2026-10-13 核对的[DeepSeek 官方价目](https://api-docs.deepseek.com/quick_start/pricing/)及
[Gemini 价目](https://ai.google.dev/gemini-api/docs/pricing)折算，已知用量为
USD **0.013225–0.026199**；另保留未知用量最高 USD **0.021579**。
连同此前全部未知用量预留，累计保守 USD **1.737764**，低于原 USD 2 上限。
计入缓存命中及全部 completion（包括 reasoning），不按主机时间猜实际扣款档，不把估算当账单。

私有审计目录使用忽略的 `logs/issue_acceptance_20261013/`：源码摘要、实际命令、Qt 输出、
原生回执、SQLite、出站计数、PA 发布消费、原失败和预算均在本地；不提交这些制品。
fixture 克隆带入的旧 usage ledger 不能作为新请求次数或费用，费用审计按不同 response ID
去重并使用新出站回执 / store 元数据；不会把复制的旧条目再次计费。

## Issue 对照与收口边界

| Issue | 本次可回填 | 仍缺证据 |
| --- | --- | --- |
| #344 | Gemini Batch 已于 10-11 实测；本轮 Gemini Sync、真实中断两个边界；原创 fresh PA 发布消费 | 固定发布提交上的完整四类 Provider matrix；生产 fresh PA 验收、非 Gemini embedding、原生高级上下文扩大 A/B |
| #431 | 无 LiteLLM 环境的 DS CLI / GUI 五阶段、目录 / probe / usage、发现缺陷后的修复复测 | 第二个成功 Base URL、斜杠模型 ID、双连接凭据隔离；代码修复待 PR 合并 |
| #508 | 已有 09-30 A/B1 与 10-08/09 的 552 项章节交付、匿名 Agent 审阅、恢复、安全写回、字体和引擎执行 | 修订后独立审阅、剩余视觉 / 交互、B2/C；本轮 Provider 小样本不替代路线实验 |
| #147 | 有关系误译筛查样本 | 本轮仍没有经确认图谱、开关 A/B 或非零命中 |
| #520 / #509 / #272 | 可复用部分基线；原有引擎研究仍按自身范围记录 | 本轮没有 OpenRouter Batch、Translator++ 往返或新引擎验收 |

不自动沿用 09-10 的 LiteLLM 两类 smoke 为本轮固定基线通过，也不把全文参考实验解释成
原生 RAG / Source Index / Story Memory / PA 的效果实测。高级上下文的默认决策仍需分别验证。

已将脱敏证据发布并读回复核到
[#344](https://github.com/hu-border-collie/renpy-translation-lab/issues/344#issuecomment-6144759028)、
[#431](https://github.com/hu-border-collie/renpy-translation-lab/issues/431#issuecomment-6144760340)、
[#508](https://github.com/hu-border-collie/renpy-translation-lab/issues/508#issuecomment-6144761576)。
三个 issue 均保持 OPEN，没有修改验收 checkbox 或把本地修复标为已合并。

## 验证

三处缺陷均有先失败、后通过的针对性回归。最终完整 CLI **2667 tests，5 skipped**、
完整 GUI **1552 tests** 通过；改动范围 Ruff 与 critical Ruff、mypy scoped 11 个源文件通过。
原实验分支 tracked Markdown 加本轮新增报告共 81 文件，533 个本地链接目标全部存在。
2026-10-14 排除独立 #272 提交、整理为 `main` 基线后，重新检查 80 文件、493 个本地链接目标，
全部存在，`git diff --check` 通过。
CLI source guard 使用当前 tracked Python 的逐字节副本，排除历史 untracked 工作树，
不放宽其断言；CLI / GUI 的 TEMP 位于可写外部目录，GUI 使用 offscreen。
本轮没有变更依赖，也没有重跑依赖审计，不宣称完整 `all` 门禁通过。
