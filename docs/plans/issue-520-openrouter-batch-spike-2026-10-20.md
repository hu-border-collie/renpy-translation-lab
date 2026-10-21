# #520 OpenRouter Batch 有界可行性验证（2026-10-20）

后续：用户要求更现代模型并排除 GPT/Claude 后，已启动
[DeepSeek V4.1 Flash 续验](issue-520-openrouter-batch-deepseek-2026-10-20.md)。
10-21 已读回该路线的 Sync 与 Batch 成功结果（Batch 2/2 请求、6/6 item）；最终仍暂缓
生产接入，详见续验报告。以下保留第一轮条件及负结果，不能将其 400
推广为 OpenRouter Batch 整体不可用。

**首轮决定：暂缓生产接入。** 当时账号下，选定的 OpenRouter / OpenAI Batch 请求被明确
HTTP 400 拒绝，目录声明与实际提交冲突；同模型 Sync 首条又被 HTTP 403 TOS 限制拒绝。
没有 OpenRouter batch ID 或成功生成结果，不能据公开目录或 mock 宣布可用。
已有架构足以复用语义层，但不足以证明新增策略有实际收益。

本次已交付[版本化合同 v1](issue-520-openrouter-batch-contract-v1.md)、
[原创 fixture](../../tests/fixtures/openrouter_batch_spike/source.json)、
[离线 harness](../../scripts/openrouter_batch_spike.py)及负例测试；真实调用也已执行。
没有修改生产 CLI/GUI、策略枚举、默认配置、依赖或写回事务。没有重跑 #508 或使用其预算/材料。

## 基线、材料与实验条件

- 刷新时 `main=origin/main=b964723f8f0e81ffdc7f6abaf3b5cd951b6fe482`，工作区干净；
  [#520](https://github.com/hu-border-collie/renpy-translation-lab/issues/520) OPEN、无评论，开放 PR 为 0。
- 执行环境：Windows、Python 3.14.0、标准库 OpenRouter HTTP；Gemini 对照使用本机已有
  `google-genai 1.57.0`，`attempts=1`。未升级依赖或修改系统 Python/锁文件。
  这不是此前 Python 3.11 锁定环境的重复验收。
- 原创 6 句、两场景、每条请求 3 句；包含变量、标签、停顿和反讽。无私有游戏正文、历史
  Batch 结果或 #508 参考材料。只有这一份 fixture，无收费失败矩阵或追加试模。
- 由现有 `TranslationPlan` 冻结 system/user prompt、ID schema、结构保护映射、source
  snapshot 和 generation config。研究使用 `execution_strategy=sync` 作为现有语义基线，
  没有伪造一个生产可用的 `openrouter_batch` 枚举。
- 两条请求 temperature=0.2、max output=1024；OpenRouter 上游固定
  `provider.only=[openai]`、`allow_fallbacks=false`，不配置 web search、tool 或缓存写入。
  Sync body 与 Batch body 由同一现有 serializer 生成。

| 冻结事实 | 值 |
| --- | --- |
| 研究合同 | v1，SHA-256 `0535796aa9890506f3ebfa8546f00017e513256e89f7f1a101da4275711d79e1` |
| 原有 plan fingerprint | `0b65c615bac733df` |
| source snapshot | `96d412b7e7d37036702082deea7c5b52aae805e771b647e526db0210455de11f` |
| request ID | `40c31ec9497e780d` / `c1e911c1b01c84fa` |
| prompt fingerprint | `ae27e2085c039d5b` / `b87ef50539f9f5ca` |
| request fingerprint | `be04e9c0f744b058` / `80ccc0b369fcff02` |
| OpenRouter wire | 2 requests，5,180 UTF-8 bytes；body 继承顶层基础 model slug |

## 官方声明与本次实际可用性

查阅日期均为 2026-10-20。目录是当日快照，不是未来可用性承诺。

| 项目 | 官方声明 / 目录记录 | 本次实测与限制 |
| --- | --- | --- |
| 文本模型与上游 | `openai/gpt-4.1-mini:batch`；endpoint `OpenAI \| openai/gpt-4.1-mini-2025-04-14:batch`，tag=`openai` | 公共及同账号认证 GET 均返回该 variant；实际 POST 基础 slug 被 400 拒绝 |
| 结构化输出 | variant endpoint 列出 `response_format`、`structured_outputs`、temperature、max_tokens；OpenAI 模型页也声明支持 Batch/结构化输出 | 请求冻结 strict JSON schema；没有远端成功结果，不能验证 Batch 的 schema enforcement |
| 上下文 / 输出 | 目录：1,047,576 context / 32,768 completion tokens | 本次输出上限 1,024，无限额压力测试 |
| 请求 / 文件限额 | OpenAI 原生 Batch 为每批 50,000 请求、200 MB；排队 token 随账号层级 | OpenRouter inline 前端的独立硬上限未在查阅页找到，不能把原生文件限额直接当作 OR 合同；本次仅 2 请求、5,180 bytes |
| 价格 / 百万 token | OR 选定 endpoint Batch `$0.20/$0.80` 输入/输出，cache read `$0.05`；同上游 Sync `$0.40/$1.60`，cache read `$0.10` | 目录 50% token 差价成立；OR 无生成 usage/cost，不能报告实测节省 |
| 普通额度 / BYOK | 普通额度 `usage.cost` 为 OR 收费；BYOK 中它只是 OR fee，上游另计并提供 cost_details 估算 | 本次无 `is_byok` 回执，实际资金路线未知；没有改账号 BYOK 设置或添加上游凭据 |
| 窗口 | `completion_window` 只接受 `24h`，`expires_at` 按上游窗口；OpenAI 为 24h | 未创建 OR 作业，无实测完成/过期时间；其他上游不能统一为 24h，例如 Vertex 96h、AI Studio 48h |
| 输入/结果保留 | OR GCS 制品从创建起 30 天；过期后结果 GET 410；上游另按其政策 | 未创建 OR 输入/结果制品，保留/410 均未实测 |
| 删除 | 终态 DELETE 清理 OR 制品；OpenAI 原生 batch record 不支持此删除，但 OR 声明删除其上传的输入及 output/error files；计费/生成/审计记录保留 | 未执行删除；不能把 OR 200 删除解释成 OpenAI batch record、账单和所有日志都被删除 |
| 上游隐私 | OpenAI `/v1/batches`：不用于训练，通常 abuse retention 30 天，application state until deleted，非 ZDR eligible；原生 Batch output file 完成后 30 天自动删除 | 未核对账号合同例外；OR 的 30 天与上游状态保留不是同一个承诺 |

依据：[OR variant endpoint](https://openrouter.ai/api/v1/models/openai/gpt-4.1-mini:batch/endpoints)、
[Sync endpoint](https://openrouter.ai/api/v1/models/openai/gpt-4.1-mini/endpoints)、
[Batch 文档](https://openrouter.ai/docs/batch-quickstart)、
[OpenAI GPT-4.1 Mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)、
[OpenAI 原生 Batch](https://developers.openai.com/api/docs/guides/batch)、
[OpenAI 数据控制](https://developers.openai.com/api/docs/guides/your-data)。

BYOK 的现行免费 allowance 按计划和 list-price inference cost 衡量，不能沿用旧的
“每月一百万次请求免费”。Pay-as-you-go / Standard 与 Business 的公开额度为每月
USD 25,000，超额 OR fee 为同请求 list price 的 5%；Enterprise 依合同。
普通额度的 credit purchase fee 与逐请求 inference charge 也要分开。本次没有充值。
依据：[当前 BYOK](https://openrouter.ai/docs/guides/overview/auth/byok)、
[当前定价](https://openrouter.ai/pricing)。

## 真实执行记录与同口径比较

时间口径为北京时间 2026-10-20。所有提交在本单明确预算授权之后；每个 attempt 先落盘
dispatch marker、冻结摘要、payload 摘要和预留，无自动 POST 重试。

| 路线 | 请求形状 / 结果 | 失败率、费用与耗时 |
| --- | --- | --- |
| OR Batch，GPT-4.1 Mini / 固定 OpenAI | 一次 POST `/batches`、2 个 custom_id；HTTP 400：模型没有 `:batch` endpoint。无 ID，无生成回执 | 作业提交拒绝 1/1；逐条生成失败率、质量、总完成时长不适用；费用未报告 |
| OR Sync，同模型/同上游 | 计划 2 条；首条 POST Chat Completions 返回 HTTP 403，provider Terms Of Service 限制，第二条停止 | 已尝试 HTTP 请求拒绝 1/1，另 1 条未执行；无实际 Provider/model/usage 回执；首条往返约 1.33 秒，不是生成耗时 |
| Gemini Batch，2.5 Flash-Lite / Google Developer API | 同一冻结 prompt/schema/输出上限；inline src 的 2 请求，唯一作业成功，原始 REST `metadata.key` 关联 2/2 | 远端生成成功 2/2；严格内容合同失败 2/2，合法 item 0/6；955 input + 237 output token；远端创建至完成 107.81 秒 |

OpenRouter Batch 的错误摘要为：`Model 'openai/gpt-4.1-mini' does not have a :batch endpoint.`
这与提交后同账号 endpoint GET 的记录仍然冲突。随后只读 list（Unix created_after，
`has_more=false`）未找到本次模型的作业。没有猜测追加 `:batch`、重提、换上游或换模型。
Sync 的 403 是明确的 Provider TOS 拒绝；具体账号/区域/上游政策原因未确认，不能宣称
OpenRouter 全局不支持，也不能把 Batch 400 与 Sync 403 推断为同一根因。

本地工具的两项只读查询错误保留在检查点：ISO 时间 query 中的 `+` 未编码导致 list 400，
改为官方支持的 Unix timestamp 后 GET 200；临时 Gemini Client 提前释放导致查询报
client closed，保持 Client 引用后查询成功。二者都没有重提生成，不计入 Provider 生成失败率。

Gemini 创建时间 `21:15:24.240620`，完成 `21:17:12.053642`，提交 HTTP/SDK 返回约 6.08 秒。
原始 REST metadata 为 `BATCH_STATE_SUCCEEDED`，统计 successfulRequestCount=2。
SDK 1.57.0 的 `InlinedResponse` 丢弃 metadata；只读取得原始 REST 回执后按 `metadata.key`
关联，未按结果位置猜测。远端 job 名只留在忽略目录；公开 ID 摘要为
`f16576c4b6e29a139d41d17752754af51cd2a35fade9ce9bc381268887b95218`，
原始 REST JSON 的规范化摘要为 `9d97805e2ef6f8c751d0db0e903ffdb077da922607214d20f2ad32cb4aa127da`。

两条 Gemini 返回的 `translations` 都是字符串数组，缺少对象 item 的 `id`；现有
`validate_model_response(..., allow_legacy=false)` 各报 3 个 `result_item_not_object`
与 3 个 `response_missing_expected_id`。本地复查 SDK serializer 保留了两份
`responseJsonSchema`；未抓取实际发送 wire，故不能确定是远端 schema enforcement、
SDK/inline 传输还是其他条件造成。**不把这个小 harness 结果归因于现有生产 Gemini JSONL 路径。**

本次没有合法候选，未运行 check/apply、未写游戏文件，也未做按位置补 ID 的修复。
fixture 的独立质量量表已冻结，但缺少可接受的 OR/Sync 结果和独立审阅者，独立语义质量
对照未执行；不把单 Agent 阅读、格式检查或不同模型差异冒充策略质量结论。
Gemini 模型与传输形状和生产 JSONL 路线存在差异，故三臂比较并未齐全。

复查冻结 prompt 还发现一个 fixture 条件问题：macro 中的 `Mira=米拉 / Oren=奥伦`
与现有默认 system instruction“人名保留英文”冲突。它不解释或改变 OR 的 HTTP 拒绝，
也不能据此确定 Gemini 缺 ID 的原因，但会进一步干扰名字/风格质量比较。
本次保留已经发送的原始冻结输入和摘要，不事后改 prompt 或付费重跑；公开 harness 用于
复现该日期的条件，不能作为干净的质量 benchmark。未来如获准新实验，先消除这一冲突并
冻结新版本，再提交；本单不从这些输出得出模型质量或策略优势结论。

## 独立费用账本与未知

授权上限 **USD 0.10**，来源是用户本单明确回复，不沿用 #508 的任何授权。
计划是 OR Batch 2 + OR Sync 2 + Gemini Batch 2；实际只发送 3 个生成 POST 操作：
OR Batch（包含 2 请求）被拒、OR Sync 首条被拒、Gemini Batch（包含 2 请求）完成。
未执行 Sync 第二条，未新增修复调用。提交结果未知数为 **0**；没有运行中作业。

| 账本项目 | 金额 / 状态 |
| --- | --- |
| OR Batch 拒绝 attempt | 未返回 usage/cost，预留 USD 0.010，不能将缺少费用字段记为已确认零扣款 |
| OR Sync 首条拒绝 attempt | 未返回 usage/cost，预留 USD 0.015；第二条未发送、未占用预留 |
| Gemini 成功 attempt | token 已知；USD 实扣未报告，保留 USD 0.005 |
| 合计未释放预留 | **USD 0.030**；授权其余 USD 0.070 未使用，不代表允许扩大请求数 |
| Gemini 目录估算 | Batch `$0.05/$0.20` 每百万 input/output：`955×0.05/1e6 + 237×0.20/1e6 = USD 0.00009515` |
| 保守计费参考 | 原始 usage 报 serviceTier=`standard`；按标准 `$0.10/$0.40` 折算为 USD 0.00019030。它也不是实扣，预留覆盖两者 |
| Provider 确认的美元实扣 / 最终账单 | **未知**；不能把目录估算写成实际支出。未核对月度账单 |

Gemini 价格依据：[官方价格表](https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash-lite)；
inline 提交及状态依据：[官方 Batch 文档](https://ai.google.dev/gemini-api/docs/batch-api)。
OR 拒绝没有 batch/generation ID 可用于逐条 cost 查询；保留预留只是一项本地保守账务规则，
不表示已发生这些扣款。BYOK/普通额度未得到实际 OR usage 验证。

## 验证、验收与下一次触发条件

离线研究覆盖：冻结/重建、模型不加后缀、wire 字段顺序、精确 prompt/schema、乱序、
重复/未知/缺失 custom_id、错误 batch/route/result model、逐条错误、截断/拒答/错误 JSON、
错误 item ID/空译/缺占位符、计数冲突、非 completed 部分计数、未知提交不重提。
合成回执只验证本地合同，不冒充真实远端能力。

交付验证结果：

- `python -B -m unittest tests.test_openrouter_batch_spike tests.test_translation_plan tests.test_translation_plan_p4 tests.test_translation_core -q`：**130 tests 通过**，含新增 11 tests；首次使用沙箱默认 TEMP 的文件权限失败保留记录，改用仓库内 TEMP 后同一组全部通过。
- `python -m ruff check scripts/openrouter_batch_spike.py tests/test_openrouter_batch_spike.py`：通过。
- `python -m mypy scripts/openrouter_batch_spike.py --follow-imports skip`：通过。
- CLI `freeze` 重建摘要不变；`inspect` 处理明确标记的合成回执得到 2 candidates、`writeback_authorized=false`。
- 全部 tracked Markdown 加本次新增文档共 86 文件，检查 533 个本地文件目标，无缺失；不包括联网 URL 或 heading anchor 的可达性证明。
- `git diff --check` 及新增文件的行尾空白检查：通过。

完整命令与输出保存在忽略的检查点目录。未改产品代码或依赖，未运行全量 CLI/GUI 或依赖审计；
不以这些未执行的检查冒充新增覆盖。

| #520 原验收 | 本次状态 |
| --- | --- |
| 版本化请求/结果/状态/错误映射 | 已交付 v1；标清可复用核心和新增传输边界 |
| 原创 fixture 提交、轮询、逐条关联 | OR 真实提交明确失败，无可轮询 ID，成功链路**未满足**；Gemini 对照真实完成与关联不替代 OR 验收 |
| 可用性、费用、保留及同口径比较 | 官方研究和当日负结果已记录；OR 实扣/BYOK/成功完成/删除未实测，三臂质量和耗时比较不完整 |
| 实施/暂缓决定 | **暂缓**；没有生产实施任务、PR、推送或合并 |

重新评估的最小前置是：在合法可用账号/上游条件下，解释目录与基础 slug 提交的冲突，
确认可接受的请求模型合同，并获得一个固定模型/Provider 的原创小批次真实成功回执。
如需新的模型、Provider、请求次数或预算，先冻结新的审阅方案；不绕过本次 TOS 拒绝。
然后才能判断省下的费用是否值得新增持久状态、恢复及 CLI/GUI 维护成本。
生产接入最小范围已经写入合同，当前不启动，也不建设通用远程调度器。

私有执行目录是仓库忽略的 `tmp/issue-520/`：checkpoint、attempt marker、冻结输入、
原始回执、验证摘要和账本均在其中。后续核对只读已有文件；没有遗留批次需要重提或继续轮询。
