# #344 固定版本 Provider 矩阵实测（2026-10-16）

> 归档于 2026-10-27：保留原日期、基线、结果与未验边界。现行操作见[文档地图](../../README.md)，当前工作见[计划索引](../../plans/README.md)。

本轮限定为四类 Provider 的 CLI 真实 smoke。首次结果为 2 类通过、2 类受环境阻断；
连接恢复后的 Gemini 重试已完成，当前为 **4 类执行、检查与隔离写回全部通过**。
#344 仍保持开放：生产 Project Analysis、非 Gemini embedding、扩大上下文 A/B 与新的
中断恢复实验未在本轮执行。10-16 发现的状态用量汇总差异已在 10-17 本地修复，
离线回放通过；修复与后续章节验收报告一并提交评审，详见下方续修记录。

## 基线与范围

- 发布基线为 `main@4612d8dd62128b2d97f686202f0902a710df0f4b`，文件树为
  `edf1c55c1f8105c564d90822fb96057dee313eb4`。四个独立工具副本的 708 个 tracked 文件
  在实测前冻结，结束后逐文件 SHA-256 核对一致；产品源码未改动。
- Windows、Python 3.14.0、google-genai 1.57.0、LiteLLM 1.83.7、httpx 0.28.1。
  本轮运行正式 CLI；不是新增的 GUI 或原生人工操作验收。
- 使用同一份原创 Ren'Py 两句对白与空 TL 模板，每类一个隔离副本、一个 chunk。
  prepare、自定义 shell 命令、模型与 Key 轮换、RAG、Source Index、Story Memory 和
  Project Analysis 自动注入均关闭。凭据只在内存与子进程环境中读取。
- 四类计划的 canonical prompt fingerprint 均为 `1f9a0fc510e39cc8`。
  请求身份随 profile / 执行策略变化，不要求 request fingerprint 相同；相同提示不代表
  不同模型或上游供应商的输出质量等价。

## 首次结果（保留失败记录）

| 组合 | 本轮实际结果 | 写回与边界 |
| --- | --- | --- |
| Gemini Sync，原生 `gemini` adapter | 正式 `sync-start` HTTP 400；从明确失败 run 正式派生一次诊断，仍为 HTTP 400 `FAILED_PRECONDITION`，明确报告地区不支持 | 两次各一个生成请求，接受 0/2，两个失败 run 均保留；无写回，usage 为 unknown |
| LiteLLM 内置 OpenRouter | `sync-start` 一次请求，接受 2/2；正式 `check` 的 `writeback_gate.decision=allow`，quality warning 0 | 正式 `apply` 修改隔离 TL 的 1 文件 / 2 条；独立核对源文件与目标槽外字节不变 |
| LiteLLM 自定义兼容 Provider | `matrix-router` 显式绑定 OpenRouter 地址，一次请求接受 2/2；正式 `check=allow`，quality warning 0 | 正式 `apply` 修改另一隔离 TL 的 1 文件 / 2 条；独立字节核对通过，不算作直连 adapter 验收 |
| Gemini Batch，原生 `gemini_batch` | 本地 `build` 为 1 请求 / 2 项；`submit` 在文件上传阶段 HTTP 400 `FAILED_PRECONDITION`，明确报告地区不支持 | manifest 为 `SUBMIT_FAILED`，没有远程 job name，也未提交模型生成；没有执行后续 download / check / apply |

首次 Gemini 两条路径的错误均为 `User location is not supported for the API use.`。
当时没有改变模型、地址或网络设置，也没有勾选四类 smoke 通过；后续重试见下一节。
10-11 / 10-13 的 Gemini 成功仍是各自日期、环境与基线的历史证据。

两条 LiteLLM 路径均显式选择 `deepseek/deepseek-v4-flash-0731`，传输观察与冻结 profile
分别记录 provider 前缀的保留 / 改写。OpenRouter 两次回执都报告 Baidu 上游；没有固定上游
或发起跨模型 fallback，本轮不推定其与其他服务的模型别名内部版本相同。
请求分别返回 955 和 938 total tokens（包含 reasoning），两份实际 TL 内容一致；
这两句样本只证明工作流与结构合同，不是翻译质量比较或性能结论。

## Gemini 恢复后的重试

用户确认连接条件已恢复后，沿用同一发布基线、冻结配置和原创两句范围重新执行。
旧失败 run、manifest、回执及首次审计汇总全部保留。

| 重试路径 | 当前结果 |
| --- | --- |
| Gemini Sync | 1 次生成接受 2/2，正式 check=allow、quality warning 0；apply 写入独立副本 1 文件 / 2 条 |
| Gemini Batch | 新包 1 请求 / 2 项；上传与 submit 成功，作业经 pending / running 至 succeeded；download、check=allow、apply 完成 1 文件 / 2 条，quality warning 0 |
| 独立核对 | 四类的 708 个冻结源码文件逐字节未变；每份实际 TL 两项非空，四份源文件与目标槽外字节保持一致；canonical prompt fingerprint 仍全部相同 |

Batch 第一次重试已取得上传初始化 HTTP 200，随后被 harness 的续传凭据断言误拦，未创建
模型作业。修正仅限验收脚本：在校验目标主机和初始 Key 后，允许该响应返回的精确授权续传 URL，
URL 仅在进程内保存、不写入记录。之后使用新包完成提交，既有失败包保留；产品代码没有改变。

本次重试新增两次模型执行（Sync 一次、Batch 一个请求），没有重复提交远程作业。
原始回执分别为 Sync 417 输入 / 98 输出 / 515 total，Batch 416 / 98 / 514。
四类累计生成尝试为 6 次，其中 4 次成功有 usage，旧的两次失败仍保留 unknown。
截至最终核对共观察到 14 次实际 HTTP 出站，包含上传、状态与下载；全部匹配目标 Key 或
由已验证服务器返回的精确续传授权，未保存授权 URL 的 query。

Gemini 费用按[官方价目](https://ai.google.dev/gemini-api/docs/pricing)估算：
Sync USD 0.00025125、Batch USD 0.0001255；与 OpenRouter 回执合计已知 / 估算费用
USD 0.000581351664。保留旧失败 USD 0.04，上传与查询等另预留 USD 0.02，加此前
USD 1.765823772，累计保守预留 **USD 1.826405123664**，仍低于原 USD 2 上限。
实际账单未核对，不将未知用量改成零。

本轮还观察到用量展示差异：成功 Sync 的 SQLite 与原始回执都完整保存 515 total tokens，
但状态摘要显示 unknown。[状态汇总](../../../execution/sync_run_store.py)读取 `total_tokens` /
`totalTokenCount`，遗漏 SDK 的 `total_token_count`。底层数据没有丢失；本次未修改产品代码
或历史 run，保留为后续修复项，不把四类执行链路通过扩大成所有展示行为已通过。

## 首次调用与费用（历史）

实际出站共 5 次：OpenRouter 生成 2 次、Gemini 生成 2 次（包含明确派生的诊断）、
Gemini 文件上传 1 次。5 次请求均核对凭据属于目标连接；没有自动重复派发。

OpenRouter 两个回执的已知费用合计 USD **0.000204601664**。Gemini 两次失败没有 usage，
分别保留 USD 0.02，上传另预留 USD 0.01。加此前保守预留 USD 1.765823772，
累计保守预留 USD **1.816028373664**，低于原 USD 2 上限。未核对实际账单；unknown
用量继续保留为 unknown。入场限制为最多 8 次生成、每次预留 USD 0.02，本轮实际生成尝试 4 次。

## 实验前置失败与独立核对

以下均为 harness / 配置问题，未作为 Provider 失败计数，原记录保留：

- `build` / `translate-preflight` 不接受文档旧示例的 `--json`，按当前 `--help` 改为
  `--output json`；durable Sync 命令仍保留 `--json` 别名。
- 实验 Batch 默认策略被错误继承到 Project Analysis；按现有合同显式设置该阶段为 `sync`，
  并保留修正前的配置与 freeze。首次真实请求发生在修正之后。
- `sync-derive` 需要 run ID，首次传入目录被前置拒绝；随后按帮助使用原失败 run ID。
- 初始 fixture 摘要使用 LF 规范化文本，Windows 实际文件为 CRLF。独立核对同时验证规范化
  摘要、四份物理源文件字节一致，并使用未写回的 Gemini 副本模板核对非目标字节，未改源文件
  或放宽产品门禁。

本轮没有产品代码或依赖变更，不把既有 CI、历史完整测试写成本轮新跑的完整门禁。
新增 / 修改文档检查全部 tracked 本地 Markdown 链接与 `git diff --check`。
原始 HTTP 观察、源码 / 配置冻结、命令回执、SQLite、fixture 和费用审计留在本地忽略目录，
公开结果仅包含脱敏汇总。

## 2026-10-17 续修：Sync 用量摘要

`SyncRunStore.build_snapshot` 改为复用模型用量账本的归一化函数，兼容 REST、SDK 与
OpenAI-compatible 计数字段。没有新增配置、命令、GUI 入口或文案常量；CLI JSON 和 GUI
运行事实共同消费修复后的快照，未知计数仍为 `null`，真实零不会被当成未知。

先以原创最小 fixture 复现失败，再验证字段兼容、思考 Token、零 / 缺失 / 无效值、
混合 Provider 的已知量汇总、重复读取不改写回执，以及 CLI JSON / GUI 展示。
另从 10-16 成功 run 的 SQLite 只读备份到新目录离线回放：旧摘要为 `null`，
原始 `total_token_count=515`，修复后快照为 515，GUI 显示「已知 Token：515」。
原数据库文件 SHA-256 与副本中的 attempt 回执均未改变，没有新增 Provider 请求或费用。

验证：store、用量账本、service、executor、结果导出、durable workflow、CLI 合同及两组 GUI
测试共 276 项通过；改动 Python 文件的 Ruff、82 份 Markdown / 506 个本地链接、9 组命令
示例解析和 `git diff --check` 通过。本次未重跑完整 CLI / GUI 套件或完整质量门禁。

此处源码修改属于 10-17 的续修，不改变前文 10-16 固定发布基线与四类实测的证据范围。
续修记录形成时，修复未提交、未推送、未合并。后续修复与章节验收报告一并提交 PR 评审，
合并及线上 issue 回填仍按关联 PR 核对，不因本地通过自动关闭 Epic。

## 后续与状态

- #431 已按首期范围完成并关闭；其无 LiteLLM 五阶段与双地址证据见
  [10-15 续验](issue-431-508-acceptance-2026-10-15.md)及
  [10-16 收口记录](https://github.com/hu-border-collie/renpy-translation-lab/issues/431#issuecomment-6218529358)。
- #344 四类执行链路矩阵已在同一固定基线上补齐；首次失败、续传 harness 失败和新成功结果
  分开保留。Sync 状态用量汇总已在 10-17 修复并提交评审，不自动关闭 Epic。
- [10-13 中断恢复与 PA 补证](provider-acceptance-reassessment-2026-10-13.md)保持原日期与范围。
  生产 fresh PA、非 Gemini embedding 和原生高级上下文扩大 A/B 仍独立待验收。
- #508 的路线实验、视觉 / 交互验收，以及 #147 / #272 / #509 / #520 均未在本轮启动。

现行命令与安全边界见 [Provider Smoke Matrix](../../provider_smoke_matrix.md)。

10-17 的真实章节 PA / 非 Gemini embedding / 扩大上下文 A/B 与新的恢复费用续验，
见[独立续验报告](issue-344-context-acceptance-2026-10-17.md)。该报告使用含本地修复的冻结副本，
不改变本文 10-16 固定发布基线和上述历史待办的日期。
