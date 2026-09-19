# 七类规划方向重新评估

> **状态（2026-09-19）：评估已完成，两项后续实现已登记为 [#512](https://github.com/hu-border-collie/renpy-translation-lab/issues/512) / [#513](https://github.com/hu-border-collie/renpy-translation-lab/issues/513)，尚未交付。** 其他方向按已有任务或本文触发条件推进。全量计划状态整理仍由 [#510](https://github.com/hu-border-collie/renpy-translation-lab/issues/510) 跟踪。

评估日期：2026-09-19。固定代码基线：[main@a94ff89](https://github.com/hu-border-collie/renpy-translation-lab/commit/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb)。评估验证阶段在该提交的临时源码副本上检查和运行离线验证，未修改产品代码或执行真实模型调用。

入库前再次核对：主干为 [main@3ddca7a](https://github.com/hu-border-collie/renpy-translation-lab/commit/3ddca7aa17b12b9c88aaa0927b3b39fcc65dddb1)，相对评估基线只增加 PR [#507](https://github.com/hu-border-collie/renpy-translation-lab/pull/507) 的验证准备文档。字体只读 spike 已有开放 PR [#511](https://github.com/hu-border-collie/renpy-translation-lab/pull/511)，尚未合并；本文没有复验该 PR 的运行结果。以下源码证据与测试数字仍对应固定评估基线，任务状态为此次日期快照。

本报告区分已验证事实、工程判断和仍需实验的数据。它是当前取舍建议，不把原研究中的全部建议转成开发承诺。

## 结论与顺序

| 方向 | 本轮决定 | 下一步 |
| --- | --- | --- |
| 两阶段翻译与上下文编排 | 先保留为可选使用策略；暂不新增固定流水线 | 在 [#508](https://github.com/hu-border-collie/renpy-translation-lab/issues/508) 的章节实验中观察真正缺失的工具能力；场景边界只接入有来源、可版本化的证据 |
| 第三引擎、Translator++、多媒体 | 继续窄实验；暂不启动完整接入或多引擎并行实现 | [#509](https://github.com/hu-border-collie/renpy-translation-lab/issues/509) 验证指定版本往返；[#272](https://github.com/hu-border-collie/renpy-translation-lab/issues/272) 保留原生引擎选型，Naninovel 仍有良好适配条件 |
| 复用、恢复和回滚 GUI | 拆开处理；复用决策 GUI 已具备立项条件，成功写回撤销还需要新合同 | 已建 [#512](https://github.com/hu-border-collie/renpy-translation-lab/issues/512)，只包装已有复用核心；成功写回撤销暂不开发 |
| 扫描、术语及质量规则 | 白名单拆分具备立项依据；通用过滤、术语强制占位和自动断行暂缓 | 已建 [#513](https://github.com/hu-border-collie/renpy-translation-lab/issues/513)，保留旧配置语义；其他改动等待具体样例与误报评估 |
| LiteLLM、新协议与默认策略 | 保留可选 LiteLLM；先补现有直连的真实验证 | [#431](https://github.com/hu-border-collie/renpy-translation-lab/issues/431)/[#344](https://github.com/hu-border-collie/renpy-translation-lab/issues/344)；按实际 Provider / embedding 需求再拆协议或迁移单 |
| 字体检查产品接入 | 明确值得做，但先完成只读能力的支持范围 | 跟踪 [#487](https://github.com/hu-border-collie/renpy-translation-lab/issues/487) / 待审 PR [#511](https://github.com/hu-border-collie/renpy-translation-lab/pull/511)，之后再决定 doctor/preflight/GUI 接入范围 |
| Story Graph、终审 Agent Loop、RAG | 质量架构扩展暂缓；RAG 存在可测量性能成本，但尚不支持换库决定 | [#147](https://github.com/hu-border-collie/renpy-translation-lab/issues/147) 收集质量收益；[#305](https://github.com/hu-border-collie/renpy-translation-lab/issues/305) 保持触发式方向；真实任务中记录 RAG 的耗时占比 |

本轮评估后新增的实现任务只有两项：

1. [#512](https://github.com/hu-border-collie/renpy-translation-lab/issues/512)：**在现有复用候选界面提交审校决定并导出复用结果**。
2. [#513](https://github.com/hu-border-collie/renpy-translation-lab/issues/513)：**区分语言允许词与排版豁免词，保留旧白名单语义**。

二者已进入待办，但不抢占 [#508](https://github.com/hu-border-collie/renpy-translation-lab/issues/508) 的主线实验与已有 [#427](https://github.com/hu-border-collie/renpy-translation-lab/issues/427)/[#487](https://github.com/hu-border-collie/renpy-translation-lab/issues/487)。RAG 读侧优化是低优先级候选：只有实际使用这条路径并确认准备/检索耗时明显时，再拆小优化单；不能把合成基准阈值直接当成整个产品的性能事故。

## 1. 两阶段翻译与上下文编排

**已验证事实。** 现有 [关键词到术语表合并](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/keyword_glossary_merge.py#L842) 已支持人工审定后的导入。没有必要为了先审术语、再译正文重新开发一整套流水线。现有 [局部上下文窗口](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/translation_plan.py#L629) 已识别显式 scene_boundary/route_id，并区分真正的共享 block 与每句唯一的翻译 ID；缺失可靠场景信息时明确退回文件顺序。源码检索未发现把实际 scene_boundary 写入 TranslationUnit 的生产提取流程，Project Analysis 的 route_id 也不是直接接入该条目边界的证据。

合成探针：三条同文件记录、各自具有唯一 block ID，以中间记录为目标时，缺少场景元数据会读到前后两条并报告 scene_boundary_unknown=true；给前一条不同的 scene_boundary 后，前文被截断、同场景后文保留。这证明消费能力有效，不能证明真实场景自动识别已经完成。

**工程判断。** 两阶段顺序可先由 Agent 或现有 workflow 组织，项目继续提供可靠材料与术语版本。自动生成场景、自动重置记忆或把所有系统词条强制放到正文之前，尚缺质量收益证据；剧情依赖和角色称呼也未必可由“系统/正文”二分表达。

**可触发实施的证据。** [#508](https://github.com/hu-border-collie/renpy-translation-lab/issues/508) 中出现可复现的跨场景污染、旧术语影响定位困难或大量手工范围标注；独立审阅能指出工具缺口。届时优先增加可审查、带源版本的显式边界/范围输入，不直接开发推断整个控制流图的调度器。

## 2. 第三引擎、Translator++ 与多媒体

**官方资料复核。**

- [Naninovel 1.21 本地化指南](https://naninovel.com/guide/localization) 提供带 ID 的独立文本制品、注解、表格往返及社区本地化导出路径。这使它仍适合作为原生 Adapter 候选，但合并片段、内联命令和实际游戏版本仍需 fixture 验证。
- [Dialogic 2 官方翻译说明](https://docs.dialogic.pro/translation.html) 以时间线翻译 ID 与 CSV 工作流为主；[Godot 导入说明](https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/importing_translations.html) 区分 CSV 导入与 PO 支持。读取表格不等于验证资源导入和运行时生效。
- Translator++ 官方[格式参考](https://dreamsavior.net/docs/translator-developers-guide/trans-format/format-reference/)展示多列 data 与 context；[按上下文翻译](https://dreamsavior.net/docs/translator/getting-started/translation-by-context/)允许同文不同译；[staging 文档](https://dreamsavior.net/docs/translator/staging-files/)明确缓存缺失会妨碍导出。格式参考本身是较早的文档，不能替代目标版本实测。

**工程判断。** [#509](https://github.com/hu-border-collie/renpy-translation-lab/issues/509) 的无修改/少量人工译文往返顺序正确，继续执行。后续先比较 Agent 直接使用 Translator++ 与加入本项目工具后的净收益，再决定 bridge。MCP 只优化操作步骤，不能解决身份映射和导出合同，故后置。原生 Adapter 与外部工具路线应按真实项目引擎需求选择，不对不同引擎的翻译耗时作伪 A/B。

**暂缓。** 不同时实现 Naninovel、Dialogic、RPG Maker；不把图像/音频资产强塞进现有文本写回合同。多媒体首先需要可授权输入、资产身份、尺寸/格式约束、人工审阅和游戏加载验收，当前没有足以冻结这些边界的任务证据。

**证据限制。** 本轮未安装或运行 Translator++、Naninovel 或 Dialogic，也没有确认具体安装包许可。部分 Translator++ 发行页访问失败；不宣称已核实其最新发行版或 MCP 实际工具列表。上述缺口由 [#509](https://github.com/hu-border-collie/renpy-translation-lab/issues/509)/[#272](https://github.com/hu-border-collie/renpy-translation-lab/issues/272) 的实际实验承担。

## 3. 复用、恢复与回滚 GUI

**复用决策 GUI：已登记 [#512](https://github.com/hu-border-collie/renpy-translation-lab/issues/512)。** [复用核心](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/engine_adapters/reuse.py#L1361) 和 [现行合同](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/docs/engine_adapter.md#L224) 已存在，CLI 可导入带 reviewer 的决定，校验 freshness/冲突，再导出复用结果进入 check/apply。现有 [GUI](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/gui_qt/engine_snapshot_dialog.py#L188) 只有读取、查看和打开报告，表格禁用编辑。两个端到端复用测试已通过。

首个增量可以限定为已有候选包上的 accept/reject、审阅者与备注、人工选择歧义目标、调用既有导入服务生成新包，以及导出符合条件的复用结果。保留 stale 拒绝、同一目标竞争拒绝、reference-only 不进入可写结果、Agent 不确认歧义等规则。高级 override/lineage 操作可保持 CLI；应明确首版范围，不自动按相似度接受，也不与 [#427](https://github.com/hu-border-collie/renpy-translation-lab/issues/427) 的普通条目审校状态混用。

**失败恢复与成功撤销：必须拆开。** [事务恢复](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/atomic_io.py#L1054) 用于未完成事务回滚；[成功提交清理](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/atomic_io.py#L831) 会删除临时备份和日志。探针成功写入后目录仅剩目标文件，恢复函数返回 false。项目 source-only 版本快照也不是可恢复全部文件内容的持久备份。

因此“撤销一次成功 apply”需要持久 preimage、当前内容是否仍为该次写回结果的检查、多次 apply/外部编辑的冲突规则，以及任务状态、复用资产/RAG 等后续影响的处理。当前证据支持先设计该合同，不能直接给恢复函数加一个“一键撤销”按钮。若没有明确用户场景，不启动全量快照管理器。

## 4. 扫描、术语保护与质量规则

**白名单拆分：已登记 [#513](https://github.com/hu-border-collie/renpy-translation-lab/issues/513)。** [语言残留](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/translation_quality.py#L630)、[中英文间距](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/translation_quality.py#L574) 等规则共用 allowed_latin_tokens。以原创句子“My name is Alice.”→“我叫Alice。”作探针并启用英文残留 warning：

| 设置 | 实际 finding |
| --- | --- |
| 不额外允许 Alice | suspicious_english_residue、cjk_latin_spacing |
| allowed_latin_tokens 加入 Alice | 无 |

这确认了配置表达能力的耦合：允许保留外文专名时，不能单独保留该专名的排版检查。它符合当前实现和旧配置语义，属于可改进能力，不作为新回归或普遍翻译错误报告。

建议提供作用域明确的语言允许词和排版豁免词，准确字段名在实现合同中冻结；旧字段继续维持现有含义，已有 manifest 继续消费冻结策略，不静默迁移旧任务。同步 runtime、示例配置、GUI schema/中文说明与回归；保留 [#364](https://github.com/hu-border-collie/renpy-translation-lab/issues/364) 对高噪声规则默认关闭的决定，不顺手重开全部语言规则。

**其他方向暂缓。** [Ren'Py 分类器](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/engine_adapters/renpy.py#L1138) 已区分动态表达式、解析错误、关键字参数、voice/asset 等；[结构保护](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/docs/structure_protection.md#L1) 已交付。更宽泛的“像代码/ID/颜色就过滤”须先有具体语法证据，保留 inventory 与 reason code，不能静默删对白。术语强制占位、词形恢复、自动断行涉及语言与布局，不沿用结构 token 的正确性证明；只有出现真实样例并明确可接受误报时才拆单。真实扫描频率证据仍归 [#470](https://github.com/hu-border-collie/renpy-translation-lab/issues/470)。

## 5. LiteLLM、新协议与默认策略

**决定：保留可选 LiteLLM，优先完成现有验证。** S1–S3 已覆盖直接 Chat Completions 生成、目录/探测与显式 sync final_review；这不等于全部依赖用途消失。[迁移矩阵](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/docs/plans/issue-431-litellm-migration-matrix.md#L10) 仍列出 embedding、非兼容协议、旧配置语义、目录/安装等用途，当前 [embedding transport](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/translator_runtime.py#L3442) 仍实际导入并调用 litellm.embedding。

[#431](https://github.com/hu-border-collie/renpy-translation-lab/issues/431)/[#344](https://github.com/hu-border-collie/renpy-translation-lab/issues/344) 的真实验证应覆盖可取得的连接、未安装 LiteLLM 的生成路径、实际结构化输出能力、凭据隔离、失败/恢复与 final_review。PR [#507](https://github.com/hu-border-collie/renpy-translation-lab/pull/507) 已合并，交付的是准备验证说明，不是已经通过的 Provider 证据。

如果真实任务需要独立 embedding，可另拆小 transport 接入，复用 provider-neutral embedding identity 与持久 store 合同；不以“彻底移除 LiteLLM”为首个验收。只有指定目标模型/协议确实无法由现有入口满足时才增加 Responses/Anthropic 等协议。默认路径继续维持现行决定，新的切换必须有真实 smoke 和可比 A/B。

## 6. 字体检查正式产品接入

**决定：值得做，继续跟踪 [#487](https://github.com/hu-border-collie/renpy-translation-lab/issues/487)；生产接入等待其结果。** 在固定评估基线中，字体工作主要服务工具 GUI，未找到游戏实际译文字集的覆盖检查实现。入库时 PR [#511](https://github.com/hu-border-collie/renpy-translation-lab/pull/511) 已提交只读覆盖 spike，仍处于开放待审状态；应先审查并验证其支持范围，不能重复安排为尚未开工，也不能视为已交付。[fontTools 官方接口](https://fonttools.readthedocs.io/en/latest/ttLib/ttFont.html)可读取字体映射，本机只读探针也验证：Arial 对 A/é 有映射、对“中”无映射；Microsoft YaHei 第一个字面对三者都有映射。本轮没有分发字体或将该环境中已安装的库视为项目已声明依赖。

这只验证静态 cmap 查询的可行性。[Ren'Py FontGroup](https://www.renpy.org/doc/html/text.html#font-groups)可按范围、顺序和映射选择字体，样式、字体集合索引、替换、动态配置和实际渲染还会影响结果。

首版只承诺：可解析的静态字体引用、实际译文字符集、确定缺失和无法判定的 checked/missing/unknown 证据。unknown 不等于通过；静态覆盖也不等于 shaping、溢出或游戏显示通过。[#487](https://github.com/hu-border-collie/renpy-translation-lab/issues/487) 应产出可再分发 fixture、明确支持范围与成本预算，然后才拆 doctor/preflight/GUI 接入；不同时做自动注入字体或通用游戏解释器。

## 7. Story Graph、终审 Agent Loop 与 RAG

**Story Graph：继续低优先级试点。** [现有实现](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/story_memory.py#L511) 已能使用角色 style、关系和 scene summary；示例配置仍默认关闭。静态存在这些能力不能证明质量提升，也不能据此要求所有项目维护完整图谱。先在 [#147](https://github.com/hu-border-collie/renpy-translation-lab/issues/147) 用最小确认图谱对关系/人称错误高发片段做盲评，并计入维护耗时；收益不足继续关闭。自动图谱、图数据库和更复杂 scene 检索暂缓。

**终审 Agent Loop：暂不重开 [#305](https://github.com/hu-border-collie/renpy-translation-lab/issues/305)。** 最新 issue 决定仍是 NOT_PLANNED。静态最终审校的 schema/精确重复校验已实现，S3 的 sync 执行也已交付；新循环需先证明动态取证能改善静态审校的误报/漏报或人工确认成本。模型自报 completion receipt 不能独立证明完整审阅；没有真实部分覆盖样本时不据此增加强制合同或宽松 JSON 修复。

**RAG：性能成本已确认，但先做实际任务归因。** 当前 [JsonRagStore.search_history](https://github.com/hu-border-collie/renpy-translation-lab/blob/a94ff891a1bb2353d50fc9de8f00a622f3d6b2eb/rag_memory.py#L595) 对每条向量重复算范数并全量排序；历史写入会重写 JSONL。Source Index 已有不同的候选/范数处理，不能将两个 store 混为同一个未优化实现。

现有基准在本机运行，768 维、每类查询 5 次、seed=42：

| 历史条数 | 三类查询平均耗时范围 | 追加 10 条 | 历史文件 |
| --- | --- | --- | --- |
| 100 | 8.15–8.60 ms | 0.0440 s | 1.67 MB |
| 1,000 | 79.40–81.61 ms | 0.3064 s | 16.68 MB |
| 5,000 | 397.08–404.26 ms | 1.5067 s | 83.42 MB |

这是本机合成数据结果，不是生产任务的延迟承诺或真实翻译性能结论。现有路径在构建请求时按 chunk 检索，所以频繁使用大 store 的项目可能受益；是否值得优先开发还取决于该阶段在完整任务中的占比。

若真实任务证实该成本，应先考虑查询范数/记录范数缓存、减少无必要排序，并验证更新/删除后的失效、相同结果次序、embedding identity、零向量及阈值边界。不要直接换成 SQLite、append-only store 或增加数据库依赖；那会引入持久化、并发、恢复和迁移的新合同。

## 验证记录与限制

在 Windows、Python 3.14.0 的固定源码副本执行；以下为评估阶段记录，不是对 PR [#511](https://github.com/hu-border-collie/renpy-translation-lab/pull/511) 的测试：

- `python -B -m unittest tests.test_translation_plan tests.test_translation_quality tests.test_quality_real_samples tests.test_translation_reuse_workflow -q`：125 项通过。
- `python -B -m unittest tests.test_atomic_io.AtomicIoHelperTests tests.test_atomic_io.AtomicWriteStrictRecoveryGuardTests tests.test_atomic_io.AtomicWriteRollbackPhaseTests tests.test_atomic_io.AtomicWriteLegacyJournalCompatibilityTests -q`：20 项，其中 19 通过、1 项按 Windows 平台跳过 POSIX mode bits。
- `python -B benchmark_rag_store.py --sizes 100,1000,5000 --queries 5 --dim 768 --seed 42`：完成；表中为实际输出。
- 独立合成探针验证白名单耦合、显式场景边界和成功事务清理；本机字体只读探针验证 cmap 层可行性。

没有执行真实 Provider 计费调用、真实游戏翻译、盲评或游戏启动，也没有完成 [#508](https://github.com/hu-border-collie/renpy-translation-lab/issues/508)/[#509](https://github.com/hu-border-collie/renpy-translation-lab/issues/509)/[#487](https://github.com/hu-border-collie/renpy-translation-lab/issues/487) 的正式验收。评估结论已经给出；这些真实结果是后续实施或扩大范围的触发条件，而非本轮已经获得的证据。
