# #520 现代模型续验：DeepSeek V4.1 Flash（2026-10-20）

> 归档于 2026-10-27：保留原日期、基线、结果与未验边界。现行操作见[文档地图](../../README.md)，当前工作见[计划索引](../../plans/README.md)。

最终结论（2026-10-21）：**DeepSeek / DeepInfra 的 Sync 与 Batch 均已真实完成；Batch 两条结果、6 个 item 的精确关联及结构保护通过。生产接入暂缓。**
首轮 GPT-4.1 Mini 的 400 只说明当时该路线失败，不能推广为整个 OpenRouter Batch 不可用。
本页追加新条件及终态证据，不改写[首轮负结果](issue-520-openrouter-batch-spike-2026-10-20.md)。

## 授权与固定条件

用户后续明确要求换更现代的模型，并说明账号可能无法使用 GPT / Claude，因此本轮排除
这两家。Sonnet 5.5 曾做离线准备，但没有付费提交；也没有新增 Gemini 生成。
沿用本单 **USD 0.10 总上限**，旧 USD 0.030 费用预留继续占用，本轮最多新增 USD 0.015，
合计最高预留 USD 0.045；没有借用 #508 预算或释放其未知费用。

| 条件 | v2 冻结值 |
| --- | --- |
| 模型 | `deepseek/deepseek-v4.1-flash`；2026-09-10 发布的模型系列 |
| 目录 variant | `deepseek/deepseek-v4.1-flash:batch` |
| Provider / endpoint | DeepInfra，`deepinfra/fp8`；only 精确固定该 tag，allow_fallbacks=false |
| 接受回执实际模型 | `deepseek/deepseek-v4.1-flash-20260910`，为目录已声明的 snapshot，而非未知模型轮换 |
| 请求数 | 1 Batch × 2 条，另做相同 prompt/schema/参数的 Sync 2 条；无自动重试 |
| 输入 | 仍是原创 6 句 / 2 场景；人名约定改为 Mira/Oren 保留英文，消除首轮冲突 |
| 生成参数 | temperature=0.2，max output=1024，`reasoning.enabled=false`；两种传输一致 |
| 冻结摘要 | `d7be2b940d2945675379ae8629d52b1a4380c3a77fb16041aae21d65b8a43583` |
| Batch wire | 5,258 UTF-8 bytes，2 个唯一 custom_id；模型使用基础 slug，未追加 `:batch` |

reasoning 控制是本次研究传输显式添加的字段，并进入冻结参数和 wire 摘要。现有生产
OpenAI-compatible Sync serializer 尚不转发这个控制；本次不改生产 API，不声称现有
profiles 已支持它。除这个有记录的扩展外，继续复用现有 serializer、TranslationPlan、
core ID 验证及结构保护恢复；未来生产接入需覆盖该参数保真边界。

## 目录与上游资料

2026-10-20 查询的固定 endpoint 声明：JSON Schema / response_format、temperature、
reasoning 和 max_tokens 均支持；context=1,048,576，max completion=131,072。
未测试容量边界，所有请求只使用原创文本。

| 每百万 token | Batch | 同一 DeepInfra FP8 Sync |
| --- | --- | --- |
| Input | USD 0.112 | USD 0.14 |
| Output | USD 0.336 | USD 0.42 |
| Cache read | USD 0.00336 | USD 0.0042 |

在该日期、该 endpoint 上目录差价为 **20%**；不能用“所有 Batch 都五折”，也不能与
其他上游的全局最低 Sync 价格混算。已有目录 `discount=0.3`，上表直接使用最终 pricing
字段，不再次乘折扣。依据：[Batch endpoint](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash:batch/endpoints)、
[Sync endpoint](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash/endpoints)。

DeepInfra 官方确认该模型以 FP8 提供；其原生 ID 为 `deepseek-ai/DeepSeek-V4.1-Flash`。
本次不使用 DeepSeek 官方 API 的价格作为 DeepInfra 计费依据。
来源：[DeepInfra 模型说明](https://deepinfra.com/blog/deepseek-v4-1-flash-deepinfra)、
[OpenRouter 模型页](https://openrouter.ai/deepseek/deepseek-v4.1-flash)。

保留语义仍需分层：OpenRouter 声明输入/结果制品创建后 30 天删除，终态 DELETE 可提前
清理，其文档也列出 DeepInfra 上传文件清理；本次尚未验证删除。
DeepInfra 的[官方隐私说明](https://docs.deepinfra.com/account/data-privacy)明确区分普通
推理与 bulk：bulk 可能加密落盘，结果返回后经过短暂保留再删除，但没有给出精确时长。
因此不能把普通 Sync 的 zero retention 声明套到本次远程 Batch。上游精确保留期限仍是缺口。

## 真实执行与终态

Batch 创建返回 HTTP 202，固定 Provider=DeepInfra，resolved model 为上述 20260910 snapshot。
已持久化 batch ID、请求/冻结摘要、预留和回执，随后 GET 观察到 `in_progress`。
`expires_at-created_at=86,400` 秒，即这个实际作业的过期窗口为 24 小时。
初次观察 completed=0 / failed=0，results 和 usage 为 null；当时保留 unknown。
原始 ID 与回执只在忽略的 `tmp/issue-520/deepseek-v2/` 中保存。
首轮观察约 10 分钟，最后仍为 `in_progress`，没有终态错误。观察结束时保存了续查
命令与下一次安全查询时间；用户随后明确要求收尾，已另设每 15 分钟的本地只读续查，
只查询同一 batch，未变更收费提交与预算边界。过期时间为北京时间 2026-10-21 21:56:44；
这只是过期窗口，不是预测完成时间。2026-10-21 13:53（北京时间）GET 取得
`completed` 回执；远端 `finalized_at` 为 **2026-10-20 23:59:29**，从创建到终态为
**7,365 秒（2 小时 2 分 45 秒）**。回执时间表示远端已完成时间，不能把本地读回时间
当作生成耗时，也不声称夜间持续执行了本地查询。

终态 counts=2 completed / 0 failed，两条 response 均 HTTP 200、finish_reason=stop，
实际 Provider 保持 DeepInfra，顶层 model 为冻结白名单中的 20260910 snapshot，单条
model 为基础 slug。返回数组与提交顺序相反；按 `custom_id` 而非位置映射，仍恢复到
两个原 request/chunk/prompt 身份和 **6/6 精确 item ID**。冻结摘要、变量/标签/停顿恢复
均通过，没有结果错误、漏 ID 或重复 ID。乱序关联已有真实回执证据，其余错误/部分完成
分支仍只有离线覆盖，不冒充真实故障注入。

两条 Sync 都返回 HTTP 200、实际 Provider=DeepInfra、实际 model=基础模型 slug。
共同 core 验证得到 2 个完整请求、6/6 个合法 item，变量/标签/停顿的结构保护通过；
没有生成写回授权，也没有执行 check/apply 或写游戏文件。

| Sync 回执 | 请求 1 | 请求 2 | 合计 |
| --- | --- | --- | --- |
| Input token | 482 | 438 | 920 |
| Output token | 139 | 97 | 236 |
| Cached input token | 0 | 256 | 256 |
| Reasoning token | 0 | 0 | 0 |
| Provider 报告 USD cost | 0.00012586 | 0.0000672952 | **0.0001931552** |
| 单请求往返秒 | 2.2584 | 1.9421 | 串行往返合计 4.2005 |

两条均 `usage.is_byok=false`，证明本次使用普通 OpenRouter 额度；这些是请求级费用回执，
尚未核对最终月度账单。Batch 终态同样 `usage.is_byok=false`，且第二条也命中 256 个
缓存 token；本次两臂均使用普通额度，缓存计数相同。

| 同模型、同 Provider、同冻结 prompt/schema/参数 | Batch | Sync |
| --- | --- | --- |
| 成功请求 / 合法 item / 结构保护 | 2/2 · 6/6 · 通过 | 2/2 · 6/6 · 通过 |
| Input / cached / output / reasoning token | 920 / 256 / 237 / 0 | 920 / 256 / 236 / 0 |
| Provider 报告 USD cost | **0.00015486**（批次汇总） | **0.0001931552**（两条合计） |
| 观察到的时间口径 | 创建至终态 7,365 秒，包含排队/处理 | 两次串行 HTTP 往返合计 4.2005 秒 |
| 新生成重试 / 路由回退 | 0 / 0 | 0 / 0 |

Batch 原始费用较 Sync 低约 **19.8261%**，两臂缓存计数相同，输出差 1 token。
按 10-20 冻结目录价格，将 Batch 的同一组 token 按 Sync 价格重算为 USD 0.0001935752，
按 Batch 价格为 USD 0.00015486016，与实报的 8 位小数值一致。因此本例支持该固定
endpoint 的约 20% 单价差，不能直接推广到其他 Provider、不同缓存条件或大样本吞吐。
单条 Batch usage 没有 dollar cost；不把批次总价猜分给单条请求。

现代两臂已知实报费用共 **USD 0.0003480152**。首轮 OpenRouter 拒绝调用及 Gemini
实际 dollar 费用仍未知，整个 #520 的实际总扣费保持 unknown；保留旧 USD 0.030，
总预留 USD 0.045 未释放或复用。本次收集只 GET 既有作业，没有新增生成。

既有机械 quality checker 在豁免人名后返回 0 findings，但单 Agent 语义抽查发现
两臂 `harbor:3` 均保留了标签内的英文 `quiet`，未满足冻结 rubric 的完整中文/反差要求。
其余五句的基本事件、反讽和逆风方向在单 Agent 复核中保留；这不是独立审校或质量评分。
没有按位置补 ID、修改远端译文、付费修复或扩大样本。规则通过不等于语义质量通过。
同模型两臂只在地图措辞、停顿空格等处有小差别，未见 Batch 修复该漏译；本例不支持
Batch 的语义质量优势。首轮 Gemini 远端 2/2 生成、严格 ID 合同 0/2，耗时约 107.81 秒，
Batch 目录估价 USD 0.00009515（standard 保守估价 USD 0.00019030），实报 dollar 缺失。
其旧 prompt 人名约定与 v2 不同，不能与现代两臂组成完整同口径三臂因果比较。

## 持久化与最终决定

`checkpoint.json` / `plan.json` / `attempt-*/` 保存新条件与安全恢复信息；旧实验保持原样。
原始终态、规范候选、机械检查和脱敏比较摘要只在忽略目录保存；不会提交原始远端结果。
该 Batch 已结束，不再轮询或重提；已知 Sync 结果也不再生成。

最终决定：**暂缓生产接入**。成功的端到端 smoke 证明本账号可用这条非 GPT/Claude
路线，20% 单价差也有本例实报支持；仅 2 请求不足以证明可靠性、吞吐或质量收益，
本例延迟明显长于 Sync，并保留共同漏译、Gemini 无法完整同口径对照、首轮费用未知、
上游 bulk 精确保留时长及删除未验证的缺口。现有任务没有必须新增生产策略的证据。

原验收中的提交/轮询/逐条关联可勾选；费用/数据保留和完整同口径比较仅部分满足，
保留未勾选项，以 `not_planned` 关闭研究 issue，明确不是全部验收通过。后续如有真实
需求，应先澄清 bulk 保留/清理并制定新的有界验证，再单独决定生产任务；不在本单新增
实验、策略、CLI/GUI 或默认配置。终态收尾后 heartbeat 已停用。本次 PR 只发布研究
材料，不包含凭据、原始回执、日志或私有游戏材料。

10-21 终态收集及文档更新后，针对性测试共 **132 项通过**；ruff/mypy 通过。
87 份 tracked / 本次新增 Markdown 的 540 个本地文件目标无缺失，`git diff --check`
及新增文件行尾空白检查通过。首轮冻结摘要保持不变；公开材料未保存凭据或原始远端结果。
