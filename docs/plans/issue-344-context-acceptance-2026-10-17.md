# #344 真实章节上下文与恢复验收（2026-10-17）

本次续验承接 [10-16 固定版本四类矩阵](issue-344-provider-matrix-2026-10-16.md)，
处理 #344 尚缺的真实计费恢复、fresh published Project Analysis（PA）、非 Gemini
embedding 和扩大上下文 A/B。四类矩阵保持原日期与发布基线；本次使用含本地修复的冻结副本，
修复与报告一并提交 PR 评审；代码合并和线上 issue 回填状态以关联 PR 与 #344 的记录为准。

四项续验已在以下限定范围取得可追溯结果。上下文链路可工作，但本次语义对照没有支持
默认开启高级上下文；不因实验完成而宣称完整游戏质量通过。

## 范围与可复现边界

- 使用获用户明确授权的匿名真实项目章节副本：529 条原文，预先固定 128 条
  待译目标、保留其余 401 条已审阅历史译文。不是完整游戏，也不是生产目录交付。
- 目标为 8 组各 16 条，起点索引为 0、72、144、232、296、312、400、480；在生成前冻结，
  两组使用同一源快照、目标集合、32 项 / chunk、4 个 chunk、本地上下文前 5 / 后 3。
- 生成模型为 OpenRouter `deepseek/deepseek-v4-flash-0731`，温度 0.2，A/B 输出上限同为
  24576。固定请求模型与地址，不强制 OpenRouter 的上游供应商，因此不能推定底层供应商
  或模型内部版本完全相同。没有配置跨模型 fallback。
- embedding 为 OpenRouter `qwen/qwen3-embedding-8b`，4096 维；用户在模型变更前单独授权。
  原计划的 OpenAI embedding 明确返回 403 服务条款拒绝，保留失败，不重试该路径。
- 基础组只用相同的本地上下文；上下文组启用原生 RAG、Source Index 和 fresh published PA，
  两组 Story Memory 均关闭。组合实验不分离各层的因果贡献，检索阈值 0.30 是显式实验参数。
- Windows、Python 3.14.0、google-genai 1.57.0、LiteLLM 1.83.7、httpx 0.28.1；不升级依赖。
  从当前工作树复制 708 个 tracked 文件，在每次命令前核对文件与配置 SHA-256。实验副本内
  的核心修复先于相关真实执行；后续最终源码另在干净 tracked 副本运行完整测试。
- 仅原生 CLI 执行业务逻辑。传输观察器限定 OpenRouter 主机、认证、模型、路径、请求体大小
  和累计 USD 2 预算；它不伪造模型响应、不修改写回门禁。原文、译文、请求、SQLite、向量库、
  凭据及完整失败回执不入库，公开报告只保留脱敏汇总。

## 真实中断与请求费用核对

以原创两句 fixture 各跑两种边界，每句独立请求，模型和连接与章节实验一致：

| 中断边界 | 恢复后的原生状态 | 传输与账本证据 |
| --- | --- | --- |
| 首个成功结果已提交 SQLite（T3 后）再退出 | 恢复仅执行第二句，最终接受 2/2；两条 attempt 均为首次派发 | 2 次真实生成、2 个不同上游回执；已完成的首次 attempt 不变，无重复派发 |
| 上游响应已返回、SQLite 成功提交前（T3 前）退出 | 首个请求保持 `outcome_unknown`，恢复只执行另一条 pending；最终接受 1/2、unknown 1 | 仍只有 2 次真实生成；外部已返回的收费回执保留，但不猜测回填或自动重发 unknown |

四次请求均通过 OpenRouter 的
[请求级费用元数据接口](https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation)
读回对应回执与 `total_cost`。首次短暂 404 也保留，后续元数据可用时再核对；没有补发生成。
成功 run 的原生已知 Token 总数为 1238，`actual_cost=0.00015838064`；unknown run 仅汇总
已提交的 580 Token / `actual_cost=0.000035553664`，不把未入库的另一条当作零费用。

该结果证明本连接下两个具体边界的派发与请求级计费可对应；没有核对月度发票，不能承诺
所有供应商、网络中断点或显式重发都绝不重复计费。

## 非 Gemini embedding 的真实预建与身份

在没有 Gemini / Google Key 的进程中，用原生 `bootstrap-rag` 和 `bootstrap-source-index`
分别生成 101 个历史记录、133 个原文段，真实 document embedding 共 7 + 9 次 HTTP 200。
4 行聚合，因此数量不等于 401 / 529 条原文数量。

两库记录 `backend=openai_compatible`、绑定 OpenRouter 端点的 Provider identity、模型
`openrouter/qwen/qwen3-embedding-8b`、4096 维及 document task 指纹。真实 store API 接受
匹配的 query identity；分别改变模型、维度或 Provider 时拒绝复用，测试不调用 Provider、
不重写原库。查询实际使用及上下文注入统计在下文 A/B 中核对。

首次真实 embedding 还暴露了两处产品接线缺陷，均先以最小测试复现再修复：

1. 两个 bootstrap 命令原来在解析所选 embedding 后端前强制要求 Gemini Key。现在只对
   Gemini 后端保留进入 store 前的 Key 检查，其他后端由所选连接校验凭据。GUI 继续调用
   同一 CLI，不新增第二套预建行为。
2. OpenAI-compatible adapter 原来未指定编码；OpenRouter 拒绝 SDK 传来的不明确格式。
   两种传输默认显式请求 `encoding_format=float`；PR 复审后补充 LiteLLM 本地明确拒绝该
   参数时的一次兼容处理，只省略编码参数，保留维度、连接和数值向量 / 身份校验。

实验配置也保留了两次无出站的前置修正：bootstrap 需要有效的 legacy Batch profile 绑定，
但不会因此选择 Gemini 生成；LiteLLM 的 OpenAI 命名空间不支持本模型的 dimensions 参数，
改用它的原生 OpenRouter 模型命名空间后才发出 Qwen 请求。所有旧配置 / freeze 均保留。

## 真实章节 PA 的生成、审阅与发布

先只构建结构所生成的摘要缺乏剧情事实，未作为生产内容验收；该阶段已发生 3 次真实生成，
费用与后续有内容的 PA 生成分开保留。随后尝试原生同步关键词
摘要：5 个初始请求覆盖词法筛选后的 522 条，发生 10 次真实生成、9 次有响应，最后一次
在父进程超时后结果未知。未获得可供导入的完整原生摘要文件，保留包和费用，不伪造其成功，
不从截断输出猜测还原，也不自动重发未知请求。

改由本次 Agent 阅读冻结的全部 529 条可信原文，编写 5 份有明确原始脚本坐标和来源证据的
章节事实摘要，再走原生 `project-analysis-ingest-keywords` → `project-analysis-build-structure`
→ `project-analysis-generate` → `project-analysis-publish`。
摘要审阅身份是实际 Agent，`human_review=false`；这是一份经 Agent 整理输入的 PA 验收，
不代表关键词模型阶段已成功或人工审阅。

- coverage 独立核对 543 个候选：128 待译、401 已译、14 个明确排除的旧原文 marker 重复；
  无 parse error / unknown / unsupported。按现有 Agent-or-human 策略导入 review。
- 结构为 1 个 label / 1 条 route，5 个上游摘要 ID 全部进入 label / route / brief 链路。
- 内容生成共 3 次真实请求，输入 17752、completion 1872（其中 reasoning 1193）、
  total 19624 Token。已启用的原文索引实际参与生成输入，来源保留在原生 manifest。
- Agent 对最终全局 brief 复核章节范围、因果措辞、别名与未展示付款事实，保存原始生成文本
  和审阅差异，再用公开 store API 保存。没有伪造 human-reviewed 时间戳。
- 原生发布无 `--force`、无门禁绕过；fingerprint 为
  `c3dab50ed3732c03890f15b52c59819e7a2b3823cea3e0d8db698f57df2fb81f`，
  运行时实际可注入全局 brief 755 字符。源指纹过期、草稿和缺失发布文件的独立副本均被原生
  loader 拒绝，原 PA store 字节未改变。

PA 的本地价格表估算曾显示 0；本报告不以它作为实际免费证据，费用采用真实 Provider
usage 和请求级回读，未知请求继续保守预留。价格表补齐未纳入本轮修复。

## 原生上下文 A/B 与隔离写回

| 事实 | 基础组 | 上下文组 |
| --- | --- | --- |
| 原生执行 | 4 请求，128/128 接受；unknown / failed 均 0 | 4 请求，128/128 接受；unknown / failed 均 0 |
| 原生 check | 128 safe，writeback=allow | 128 safe，writeback=allow |
| 质量规则 | 19 warning：18 中英间距、1 半角标点；0 blocker | 0 warning / blocker |
| 输入 / completion / total Token | 9508 / 47537 / 57045 | 14736 / 45461 / 60197 |
| 其中 reasoning Token | 41640 | 39876 |
| 生成回执费用 USD | 0.009673499648 | 0.009003983232 |
| 本次启动额外 query embedding | 无 | 8 次 HTTP 200，回执 USD 0.00005626 |
| 观察到的整次启动耗时 | 222.17 秒 | 324.57 秒，含冻结检索 |
| 原生隔离 apply | 1 文件 / 128 条 | 1 文件 / 128 条 |

两组的源 identity、全部 chunk 和目标 ID 逐项相同。直接读取耐久 SQLite 中冻结的请求，
核对上下文组每 chunk 实际有 3 个历史记忆命中、3 个 Source Index 命中（各共 12）；
4 个 query identity 均报告 compatible，无 stale / 身份过滤错误。8 次查询分别供两种检索
使用，不是另一个 Gemini 后备调用。

4 个请求都实际携带当前发布 brief 的 755 字符正文，指纹与 native loader 一致。
前三个 chunk 还带 label / route 局部摘要，analysis 层各 1525 字符；第四个没有可靠的局部
坐标匹配，只保留全局层，analysis 共 825 字符。检索层依次为 1771、1778、1655、1736
字符；基础组的检索与 PA 均为空。未把一次成功的全局注入误写成全部局部节点均命中。

两组均走正式 `check → apply`，没有 `--force`。先保存并审计基础组输出，再只把本次自建
隔离章节副本的 TL 从精确备份恢复到冻结输入，重新 check 后 apply 上下文组；不是恢复或修改
原始游戏。独立校验两次各 128 个写入值与结果精确一致，401 个非目标译文、目标字面量外
所有字节、原始源脚本和历史交付文件均未变。隔离输出保留，不作为新的游戏交付。

语义审阅读取全部 128 对输出与可信章节上下文：先随机以 X/Y 命名，仍由执行本次实验的
同一 Agent 审阅，**不是独立盲测或人工验收**。复核冻结的词表后发现 `Right` 被共同标为
Non-translatable，撤销首轮将其算作漏译的误判；旧审阅保留，最终修订已在揭示组别后进行，
不声称最终评分具有盲测效力。

最终探索性审阅偏好基础组 9 对、上下文组 2 对、其余 117 对打平（其中 24 对完全相同）。
按本次 rubric 记录基础组 3 个需处理条目、上下文组 10 个；这是单 Agent 的复核结果，
不是穷尽错误数或统计显著性结论。具体可追溯发现：

- 上下文组 6 条把人名中文化，违反两组共有的保留英文人名指令；旧历史译文含中文人名，
  属于本次复用配置的重要限制，不能直接推断是哪一上下文层导致。
- 上下文组有 1 条把“说话者向当局解释”变成“当局所说”，颠倒陈述主体；另有习语过于
  字面化。基础组也有标题关系弱化、增加未经证实地点等小问题。
- 预先选定的宝石术语误译在两组都出现；原文简称在当前英文人名政策下均保留，不误计为
  必须本地化的别名错误；过去未实现愿望的时态在两组均保留。

结论是机制和写回合同通过、排版 warning 减少，语义收益未得到支持。不同上游供应商、
reasoning 波动、仅一次串行重复、单章 128 项、Agent 整理 PA 输入及组合上下文均限制推断；
不据费用 / 耗时的单次差异作性能结论，不更改默认策略、上下文开关或原项目配置。

## 代码与自动化验证

另将 10-16 发现的 Sync Token 汇总改为复用共享用量归一化，兼容 SDK snake_case。
CLI 和 GUI 共用公开 snapshot；未知仍是 `null`，真实零仍是零，读取不改写历史 receipt。
历史 Gemini run 的只读副本回放从 unknown 恢复为 515，原数据库未变、无新增计费调用。

首次验收中，adapter、runtime、bootstrap 凭据、上下文和 GUI 包装层的回归通过；当时源码在
独立的 708 文件 tracked 副本、仓库外临时目录执行完整 CLI / GUI：

- CLI 套件运行 2677 项，4 项 skipped、其余通过；GUI offscreen 1554 项通过。
- `lint-critical` 与 11 文件 typecheck 通过；改动处没有新增 advisory Ruff 诊断。
  大型历史文件原有 47 条 advisory 仍存在，不称为全量 Ruff 无告警。
- 初次直接在工作区跑 CLI 有 11 项失败：历史 untracked 工具副本被源码守卫扫描，以及临时
  路径位于仓库内被安全测试拒绝。保留该次结果，未弱化守卫；干净副本与外部临时目录复跑
  同套件通过。GUI 为自动化 offscreen，未新增原生人工点击或游戏视觉验收。

83 份 Markdown 的 511 个本地链接目标和 11 组 CLI 示例解析通过，`git diff --check`
通过。完整质量门禁 `all` 和依赖审计未重跑；本轮无依赖变更。

## 费用、限制与发布状态

本轮实际计费 POST 共 55 次：28 次生成（27 次返回 usage、1 次超时后结果未知），
27 次 embedding（25 次成功、2 次 HTTP 400 / 403 且 usage 未报告）。生成分项为恢复 4 次、
早期结构 PA 3 次、有内容的 PA 3 次、关键词 10 次及 A/B 8 次。25 次成功 embedding
为预建 16 次、PA 查询 1 次及 A/B 查询 8 次。全部目标认证与配置连接匹配，没有 Gemini
调用。27 个不同的成功生成回执均与官方请求级费用元数据对应，没有重复 receipt ID。

已知回执费用合计 USD **0.04745642404**，包含所有成功响应和失败关键词阶段的已知部分；
生成的官方请求级 `total_cost` 合计 USD **0.047222749**，embedding 费用来自响应 usage，
二者口径不同，不能直接相加成总价。未知关键词请求仍保留 USD 0.02，两次 embedding
错误各保留 USD 0.001，其余未逐请求核对的 embedding 继续使用入场预留；已匹配的生成
按回执与官方费用较大值加 2% 余量。未把错误 / unknown 改成零。

加原有保守预留 USD 1.826405123664，最终累计保守预留 **USD 1.9215723327848**，
低于原 USD 2 上限。仍未核对月度发票；Agent 工作 / 审阅时间和宿主模型费用不在该 Provider
实验账本内，不能据此宣称端到端总成本下降。

最终四个工具副本各 708 个冻结文件与配置 SHA-256 均未变，恢复 run 的原数据库经只读审计
未改变。原生写回、PA 注入、store identity、请求回执、评分修订及预算均有本地忽略制品。
私有正文和凭据不提交。完整 CLI / GUI、关键 lint、typecheck、文档本地链接与
`git diff --check` 均通过；未执行远程 CI、原生人工 GUI 或游戏视觉 / 交互验收。

本轮确认的 Token 展示、bootstrap 凭据与向量编码修复随本报告提交评审。#344 的限定实验
缺口已有可审阅结果；完成 PR 审查、远程 CI、合并和验收记录回填前，不将线上 Epic 标为
已完成或关闭。
#508 的剩余路线 / 游戏验收及 #147、#272、#509、#520 均未扩入本轮。

## PR #558 复审修订

复审后的兼容修复与以下离线验证不改变前文真实实验的冻结范围：

- 仅在 LiteLLM 的 `UnsupportedParamsError` 明确指出 `encoding_format` 不受支持时，去掉
  编码参数再调用一次；模型、端点、维度、认证和 identity 不变。服务端错误、鉴权失败、
  超时或只涉及其他参数的错误不触发此处理；第二次失败直接保留原有错误分类，返回向量仍完整校验。
- 新回归先在 `f818fdb` 上复现失败，再验证兼容处理；另用真实 LiteLLM 1.83.7 参数校验和
  模拟 HTTP 证明 Ollama embedding 可到达传输层。OpenRouter 所需的 float 默认请求仍保留，
  没有发起真实 Provider 请求。
- GUI 回归改用合法的 legacy embedding 配置，实际加载设置并调用 CLI 分发，只替换最终
  store 预建入口。新测试在原 main `4612d8d` 的 Gemini Key 前置检查处失败，在修复后通过。
  Gemini 缺 Key 用例保留为已有行为的兼容守卫，不当作新缺陷的失败复现。
- 当前报告使用匿名项目与隔离副本称呼；现行手册补充 legacy 配置回退，#431 首期关闭状态、
  后续 #344 记录入口、GFM 表格和 PA 命令名称已同步。28 次生成的分项补齐，费用总数不变。
- 最终 Python 文件与两份独立的 710 文件 tracked 验证副本一致；完整 CLI 2681 项（4 项
  skipped、其余通过）、offscreen GUI 1554 项、针对性 296 项均通过。关键 lint、改动核心
  文件 Ruff、11 文件 typecheck、83 份 Markdown / 511 个本地链接目标、11 组命令示例及
  `git diff --check` 通过。依赖和原实验配置未改，没有新增 Provider 费用。
