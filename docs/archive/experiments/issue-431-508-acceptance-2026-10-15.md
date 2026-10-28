# #431 / #508 续验与修复记录（2026-10-15）

> 归档于 2026-10-27：保留原日期、基线、结果与未验边界。现行操作见[文档地图](../../README.md)，当前工作见[计划索引](../../plans/README.md)。

本轮优先补 #431 的双连接与纯直连凭据证据，以及 #508 修订成果的独立语义复审。
它不关闭相关 issue，不代替 #344 的完整矩阵和高级上下文实验。

## 代码基线与证据范围

[PR #556](https://github.com/hu-border-collie/renpy-translation-lab/pull/556)
已于 2026-10-15 合并，`main=36bbc0850744b5831478963e5accd5fd9d26b847`。
该提交的文件树与本轮开始时的 checkout 一致；先冻结源码与原创两句 Ren'Py fixture，
再执行真实请求。本轮发现的凭据初始化与术语提示修复由本次变更交付，合并前不能记作 main 已交付。
初译、目录与 probe 先在发布基线上执行；其余阶段在有 SHA-256 记录的本地修复后复测。

环境为 Windows、Python 3.14.0、google-genai 1.57.0、PySide6 6.11.1。
真实 Provider 子进程来自 `include-system-site-packages=false` 的独立 venv，确认没有安装
LiteLLM；不设置 Gemini Key，也没有复制 `api_keys.json`。项目 prepare、模型/Key 轮换、
RAG、Source Index、Story Memory 和自动 Project Analysis 上下文均关闭。

## #431：双地址、凭据隔离与五阶段

| 路径 | 本轮结果 | 证据边界 |
| --- | --- | --- |
| DeepSeek 官方直连 | 目录、probe、初译 2/2、check=allow | `deepseek-flash`，JSON object |
| OpenRouter 显式连接 | 目录、probe、初译 2/2、check=allow | 模型 ID `deepseek/deepseek-v4-flash-0731` 未被改写 |
| 未内置的自定义 Provider ID | 初译 2/2、GUI 检查/绑定预览/隔离 apply | 同一 OpenRouter 地址，验证自定义连接路径；不把它算第三个地址 |
| 两套凭据 | 所有实际请求均匹配目标连接，另一套凭据未出现在 Authorization 中 | 密钥只在内存/子进程环境；不输出值或指纹 |
| 同进程切换 | DeepSeek → OpenRouter → DeepSeek 的三次目录请求全部成功 | 只读切换验证，不声称穷尽并发连接情形 |
| GUI 五阶段 | 初译检查/写回、术语、订正、项目分析、最终审校均完成 | offscreen QApplication、生产 CliRunner/QProcess 和实际 workflow；不是人工 MainWindow 点击验收 |

两个成功地址为 `api.deepseek.com` 和 `openrouter.ai/api/v1`。DeepSeek 仅在实验进程中
绕过环境代理；OpenRouter 沿用已有网络环境，没有改变系统代理或项目生产设置。
OpenRouter 回执报告上游 CoreWeave / DeepInfra；没有指定固定上游或发起模型 fallback，
本轮不据此确认二者与 DeepSeek 官方别名的内部模型版本等价。

五阶段分别通过正式 `sync-start`、`sync-keywords`、`sync-revisions`、
`project-analysis-generate` 和 `final-review-run-sync` 执行。GUI 从已完成的自定义初译 run
继续 check/apply，没有重复生成该次初译。订正只生成预览，最终审校保持 report-only；
项目分析覆盖 1 label、1 route 和 brief，不能替代生产项目分析或效果 A/B。

### 真实失败与最小修复

1. **不相关的 Gemini Key 前置检查。** 无 Gemini Key 时，术语命令在 Provider 调用前退出。
   原创离线回归进一步复现同样影响订正、项目分析生成和最终审校 Sync。
   CLI 现在先加载可用凭据，再由所选阶段/冻结 campaign 的 adapter 校验自身凭据。
   共用 Gemini 请求边界仍拒绝缺失 Gemini Key；远程 Batch 仍在入口要求 Gemini Key。
   GUI 继续使用同一 CLI，没有新增配置或另一套翻译语义。
2. **JSON object 模式缺少字段类型提示。** 术语生成和一次定点重试都返回字符串置信度，
   本地严格校验正确拒绝，GUI 报 warning，不能记为成功。该模式不向远端发送字段 schema；
   共享关键词 user prompt 现明确 `confidence` 为 0–1 的 JSON 数字。复测合同完整，
   没有把字符串标签猜成分数或放宽本地校验；Batch/Sync 使用同一提示构建函数。

另保留两类 harness 失败：最初未使用的 legacy Gemini profile 凭据类型配置不合法，在发送前
被拒绝；后续 harness 错将术语文本输出当作 JSON envelope。修复 harness 后沿用已完成结果，
没有覆盖失败记录。此前 10-13 环境带着 Gemini Key，未验证“完全没有 Gemini 凭据”这一条件。

### 调用与费用

实际生成 **13 次**，全部取得 usage：probe 2、三条连接配置的初译 3、术语 3（含原失败及
定点重试）、订正 1、项目分析 3、最终审校 1；另有目录 GET 5 次。
所有生成只发送本轮原创 fixture，没有新增私有游戏文本出站。

OpenRouter 回执报告合计 USD **0.00021771**。DeepSeek 共 4,149 prompt / 13,831 completion
tokens，completion 包含 reasoning；按当日[官方价格](https://api-docs.deepseek.com/quick_start/pricing/)
将全部 prompt 按 cache miss 计，估算 USD **0.00892095–0.01784190**。
本轮已知合计 USD **0.00913866–0.01805961**；另保留目录 USD 0.01 预留。
加此前含未知用量的保守预留，累计 USD **1.765823772**，低于原 USD 2 上限。
这是用量/价格估算与供应商回执，未核对实际账单。

## #508：独立复审、单项订正与新副本交付

已将实际交付文件与冻结源清单逐项核对，再生成只含背景、原文、现译和风格的独立审阅包。
用户授权的新上下文子代理逐项审阅 **552/552**（正文 529、关联背景 23），没有读取旧评分、
订正表或路线身份。审阅为 Agent、`human=false`；宿主 token、费用和活跃审阅总时长仍为
`unknown`，不是人类审稿或新路线比较。

确认 **0 major / 1 minor**：一处表示人的群体的词被译为船只编队。独立审阅者另对拟订正做
定点复核，未发现新增语义偏差。三个不确定读法保留，不强行改合理异译。
原全量审阅仍保留“需订正”的历史结论，定点通过不改写旧评分。

随后建立新的独立运行副本及工作包，以当前 occurrence、原文与槽位绑定接收全部成果，
经正式 `work-submit → check=allow → work-preview → work-apply` 写回。
实际文件复核为 **552 非空、551 项与前次交付一致、仅 1 项订正**；原文/定位/块绑定、
标签和插值、非目标字节均一致。成果仍为 `LOCAL_CANDIDATES`，不伪造 Provider 回执。

匹配引擎 Ren'Py 8.5.3.26051504 的原生 testcase 成功完成章节及关联背景。
外部 ledger 逐项确认 **538 个原生 TL 执行项 + 14 个实际 Text 显示项 = 552**，
每项与当前订正成果匹配，观察到章节 return 与原始背景 AST 边界。
测试压缩转场并自动推进，不是自然速度人工全程试玩；保存/加载、所有分辨率、动画和音频
体验仍未验收。旧运行副本 3,639 个文件的集合与 SHA-256 全部保持不变，冻结审阅输入未变。
新成果的实际字集为 1,102 字符，副本中的中文字体缺字为 0；这不代替所有画面的布局验收。

本轮支持继续限定于有独立范围核验的可恢复、安全交付工具。
**B2/C、长期多章节一致性和剩余视觉/交互验收仍缺；不宣称工具包带来质量或速度优势。**

## 验证与后续

- 两处产品缺陷均先复现再修复；针对性 CLI/core 157 项、请求/共享合同 73 项、GUI workflow
  63 项通过。关键词 golden 仅更新两处 user prompt 摘要，并在关闭更新模式后复核。
- 当前 Python 的逐字节独立副本完整 CLI **2671 tests，16 skipped**、GUI **1552 tests** 通过。
  初次 CLI 独立副本没有 Git index，隐私快照测试未能枚举 tracked files；建立只含同一份公开
  源码的临时 index 后完整重跑通过，没有放宽测试。
- 改动中无历史 lint 债务的文件 Ruff、全库 critical Ruff、mypy scoped 11 个源文件通过。
  未变更依赖，本轮未重跑依赖审计，不宣称完整 `all` 门禁通过。
- 81 个 tracked/本轮新增 Markdown 文件的 497 个本地链接目标有效，`git diff --check` 通过。
- 私有样本、运行副本、密钥、原始回执、子代理审阅文本与 Batch/Sync 制品不提交；本地审计
  位于 gitignored `logs/issue_acceptance_20261015/`。

#431 还需合入本轮产品修复并按证据维护验收清单；#508 保持开放，继续跟踪上述实验边界。
#344 的固定版本四类 Provider 矩阵、生产 fresh Project Analysis、非 Gemini embedding 和
原生高级上下文扩大 A/B 仍未完成。本轮未启动 #147 / #272 / #509 / #520 的实验或新产品接入。
