# #272 有限研究归档、#565 调查衔接与 #147 暂缓

> 决策日期：**2026-10-22**。本轮只归档研究与路线判断，不启动新实验或实现。
> 后续（2026-10-24）：#565 离线调查与 PR #568 公共 JSON 核心已进入 main；[#567](https://github.com/hu-border-collie/renpy-translation-lab/issues/567) 承接 MV 产品与 Komorebi 交付。下文保留 10-22 研究决定及草案，不视为当前实现状态；最新映射见[计划索引](README.md)。
> 首次复核基线：`09de6dcfd25ea8e21fe00d22251bb590a917b88e`；发布前按 `main@1a220214072e2fc44632c43cc0459cdb149287a1` 与新 issue #565 更新，在独立文档分支交付，保留原工作区。
> 关联：[#272](https://github.com/hu-border-collie/renpy-translation-lab/issues/272) · [#565 Komorebi 调查](https://github.com/hu-border-collie/renpy-translation-lab/issues/565) · [#147](https://github.com/hu-border-collie/renpy-translation-lab/issues/147) · [#509 实测报告](issue-509-translatorpp-roundtrip-2026-10-21.md) · [能力矩阵](visual_novel_localization_matrix.md)

## 1. 当前决定与历史材料

**#272 保持开放，第三 Adapter 的选型由 #565《Komorebi》真实游戏调查提供依据，生产实现待调查结论后另拆。Translator++ 保留独立使用、生产 bridge 暂缓；Story Graph 试点维持低优先级。** #565 已满足“具体游戏/真实需求”的研究触发条件，但游戏引擎、版本、插件与文本范围尚未核实，不把它当作 Adapter 已立项或已支持。

本轮复核 #272、#147、#509 的正文与评论及已合并 [PR #501](https://github.com/hu-border-collie/renpy-translation-lab/pull/501) / [PR #562](https://github.com/hu-border-collie/renpy-translation-lab/pull/562)，发布前又读取新建的 #565。2026-10-22 最新开放 issue 为 #565 / #272 / #147；#509 已关闭，#562 已合并为 `14456923968b80bc56b0bcb80cbd904b1087a8bd`。此前“文档合并后按 not_planned 关闭 #272”的草案已撤回；文档合并不关闭这三个 issue，也不勾选未完成验收。

#565 仅交付真实版本、插件/文本分布、独立试点清单及继续/收缩/暂缓决定；本轮文档不执行该调查，不调用模型或修改游戏。它要求先在实际副本确认版本，不能套用 #509 的 MV 1.5.1。后续公共写回、原生 Adapter、耐久 Sync 与 CLI/GUI 接线按调查结果另拆，TPP bridge / MCP 不自动恢复。

旧分支的 2026-10-09 候选合同报告用 `git show 7e2cbd8a7f190ea300a5caf85c930644370cbbbc:docs/plans/issue-272-candidate-contract-review-2026-10-09.md` 阅读，未整体 cherry-pick，也不假定该本地提交已公开。保留其“条件候选、索引不是稳定身份、纯文本不等于完整运行链路、公共写回仍有缺口”的判断；其中 **MV 1.6.3 / TPP 8.4.15B 是当时的准备方案**，不是后来实测版本。该报告的合成探针和验证计数属于历史记录，本轮没有重跑或当作游戏往返证据。

本次仅刷新 Naninovel 关键身份/工具链事实、MV/MZ 版本区分和 #509 结果；其他引擎资料沿用各文档原核对日期。旧 A+/B+ 评级及第三引擎顺序不再作为当前排名或支持承诺。

## 2. 证据分类与有限复核

| 标记 | 可支持的结论 |
| --- | --- |
| `[仓库]` | 当前源码与现行文档的静态核对；本轮未执行候选引擎集成测试。 |
| `[实测]` | #509 已合并报告记载的固定版本实验；本轮复核报告，没有重新运行工具或游戏。 |
| `[官方]` | 2026-10-22 实际打开的上游原始资料；说明上游能力，不表示本项目已经接入。 |
| `[推断]` / `[待验证]` | 路线判断与缺口；不能升级为产品能力或完成验收。 |

| 对象 | 本轮复核与结论 |
| --- | --- |
| Naninovel 身份 `[官方]` | [v1.21 Text Identification](https://naninovel.com/guide/scenario-scripting#text-identification) 说明默认按内容 hash 关联，源文改动会破坏关联；Text Identifier 可写显式 `\|#id\|`。显式 ID 仍须核对源与语境，不能当跨工程 GUID。 |
| Naninovel 制品 `[官方]` | [Localization](https://naninovel.com/guide/localization) 描述先生成目录、再编辑或 CSV 交换；源未改条目保留译文，joined IDs 需保留片段关系。[Managed Text](https://naninovel.com/guide/managed-text) 是独立键值制品。离线处理已生成文本可行；生成、导入、构建和游戏验收仍依赖官方工具链。 |
| Naninovel 窗口与样例 `[官方 / 待验证]` | [v1.21 兼容性](https://naninovel.com/guide/compatibility) 列 Unity 6.0 / 6.3 LTS 最新 patch；[Samples](https://naninovel.com/guide/samples) 指向账户 release archive，并要求 Unity Editor。未取得目标工程或逐资产再分发许可，不称 MIT / CI fixture 就绪。缺失文档、空目标、字体及运行 fallback 仍未实测；不混入 preview 行为。 |
| MV 版本 `[官方 / 实测]` | [官方下载页](https://rpgmakerofficial.com/product/mv/special/tkool-mv-trial.html) 仍标 `RPGMV_163J.zip`，不能证明包内编辑器/core 实际是 1.6.3。#509 通过 About、实际 core 与工程确认的是 **1.5.1**，不覆盖旧准备方案的 1.6.3。 |
| MZ `[官方 / 待验证]` | [1.10.0 发行公告（2026-01-30）](https://rpgtkool.hatenablog.com/entry/2026/01/30/130101) 将编辑器更新和工程 core 更新分开；[MZ Messages](https://rpgmakerofficial.com/product/MZ_help-en/01_10_01.html) 含 Name 与额外控制码。未取得目标 MZ 工程/core，不继承 MV 的布局或成功/失败结论。 |
| Translator++ `[官方 / 实测]` | [8.8.9 Release](https://github.com/dreamsavior/TranslatorPlusPlus/releases/tag/8.8.9) 与 [Translation by context](https://dreamsavior.net/docs/translator/getting-started/translation-by-context/) 仅提供版本/机制依据；实际选列、context 与 staging 以 #509 为准。Staging 官方页本轮抓取超时，其说明保留为历史资料，失败及恢复则有 #509 实测。 |

## 3. #509 已交付证据与失败

固定窗口是 **MV 编辑器/core 1.5.1 → TPP 8.8.9 → RMMVJS parser 2.19**；新建项目入口的 rmrgss **3.3** 不是实际委派 parser 版本。下表均为 `[实测]`，细节和本地证据目录以[报告](issue-509-translatorpp-roundtrip-2026-10-21.md)为准。

| 项目 | 保留的结果与限制 |
| --- | --- |
| 分母 | 完整原件 1,100 文件；独立清单 25 参数位置（19 条 401、3 个 102 元素、3 个 402 镜像），主要文本位置 22、唯一原文 19。网格行、context、唯一原文与参数位置不能互换。 |
| 无修改导出 | 15 份 JSON 仅格式变化、零 JSON 语义变化；`js/plugins.js` 生成注释/换行改变，由 **87 → 84 字节**。严格非目标文件保护检查仍 **fail**，没有因可运行而改记通过。 |
| 限定译文与空值 | 人工译文、staging 恢复重试、独立 null 保存/重开/导出均精确命中 **19 个显式字符串叶位置**；混合 null/空串回退原文。其他 6 个清单参数保留原文，不是全游戏翻译率 100%。 |
| 插入命令漂移 | 前插 code 108 注释后，一个 context 译文未迁移；期望 **19 / 实际 18**，缺失路径 `/events/1/pages/0/list/3/parameters/0` 保留英文。未观察到错写其他 context，捕获日志未提供该处专门迁移诊断；不推断所有流程都没有诊断。 |
| 非法/缺失与恢复 | 畸形地图、缺失公共事件被排除，捕获日志未明确说明遗漏；独立 coverage 仍必要。staging 丢失实际失败，恢复后重试通过，不能只保存 `.trans` 就假定可复现导出。 |
| 运行与质量 | 部分原生 NW.js / HTML5 路线、HTML5 存读档有证据；原生路线未完整遍历。字体 fallback 文件未追踪，构造译文不是正式汉化，语义/口吻质量没有完成结论。 |

这些结果支持独立工具在该受限窗口内使用，也暴露身份迁移、遗漏核对与导出恢复成本。没有同口径净收益 A/B，不给节省百分比；既不证明 MZ、Naninovel 或其他 parser 已可用，也不证明它们不可用。

## 4. 对本项目合同的复核与剩余验收

| 当前事实 `[仓库]` | 尚待候选实验解决的问题 |
| --- | --- |
| [contracts.py](../../engine_adapters/contracts.py)、[coverage.py](../../engine_adapters/coverage.py)、[versioning.py](../../engine_adapters/versioning.py)、[reuse.py](../../engine_adapters/reuse.py) 已有 occurrence、候选清单、快照/对账与复用。 | Naninovel ID/source、joined 片段和 RPG 事件/页/context 如何持久化并绑定摘要；同文不合并 occurrence。数组坐标只在冻结快照内有效。 |
| [writeback.py](../../engine_adapters/writeback.py) 的 `json_catalog_set` 限 `localization_catalog`，只遍历非空字符串字典键，要求已有字符串叶子与非空替换；已有非空单行/跨行 span 操作。 | 不能直接写 RPG 源树或 `.trans` 数组/null；首次空目标、受限根与重复目标等需定义和验证公共合同。不得用私有 JSON writer 绕过。 |
| [structure_rules.py](../../engine_adapters/structure_rules.py) 只认 `renpy` / `tyrano`；[外部初译工作包](../external_translation_work.md) 仍限 Ren'Py。 | 新引擎规则、覆盖分母及 CLI/GUI 接线尚不存在，通用对象或离线 corpus 不构成产品支持。 |
| [Batch 合同](../batch_workflows.md) 要求与当前 manifest/results 匹配的最近一次 `writeback_gate.decision=allow`。 | 第三引擎仍须绑定源快照、结构和写前复核；普通质量报警默认不阻断，配置提升为 blocker 时阻断，force 不豁免 stale/源快照/结构。外部工具导出、官方导入/构建与本项目许可分别验收。 |

#272 原 P0/P1/P2 未完成项保持未完成：其余候选的一手资料/版本/许可、所有候选的实际制品与集成对照、跨版本重复/移动关联、唯一候选及实现范围选择，均未被本次有限复核代替。本轮不勾选原清单，不以关闭研究 issue 表示全部研究完成。

## 5. 真实需求触发与实施准入

#565 已登记真实需求，#272 的下一步为其只读调查与选型回填；以下条件是后续生产实现的准入要求，不再作为推迟只读调查的理由：

1. 一个具体游戏或真实翻译需求，能说明现有 Ren'Py / Tyrano 路线或独立工具不能解决的目标。
2. 确定的引擎、编辑器/core、parser/插件版本与合法运行条件；MV 与 MZ 分开。
3. 可合法使用、再分发边界清楚的最小素材，记录原创/第三方数据、引擎、字体和资源各自边界；私有游戏不自动成为公开 fixture。
4. 可界定的输入、覆盖分母、排除项和安全写回验收，包含无修改/少量译文、重复/context 漂移、失败恢复、结构副作用及游戏验收；未知不计为零。

#565 先确认版本、范围与未知项，选择 2–3 个可到达场景，独立核对 10–20 个文本位置，并界定后续约 100–300 occurrence 的小样本；未取得样本时如实记录缺口。游戏文件、对白、存档和商业资源留在本地，公开只提交脱敏结论和许可明确的原创 fixture。调查回填 #272 后再决定生产实现，不因队列清空或文档合并而直接开工；不为选出唯一候选虚构已验证排名。

#147 的 [Story Memory MVP](../context_systems.md#结构化剧情记忆) 与 [示例配置](../../translator_config.example.json) 仍在，batch/sync 默认关闭。本次不生成图谱、不启用配置、不虚构人工确认关系。issue 正文“没有生产项目启用”是 2026-07 快照，[2026-08-08 评论](https://github.com/hu-border-collie/renpy-translation-lab/issues/147#issuecomment-5220024353)已指出其验证边界；本轮不审计所有私人项目。

只有真实翻译反复出现 glossary/source_index 难解决的关系、称呼或口吻错误，才重新启动：由人工确认 **至少 5 角色、10 条关系**，保留来源与确认记录；选小文件/片段作同条件开关对照，manifest `story_memory_summary` 必须非零命中，并记录质量、额外费用/时间、图谱整理与更新维护成本。候选 seed 不等于确认关系，命中不等于质量收益；没有收益就保持关闭。原试点验收仍未完成，不先做维护工具。

## 6. Issue 状态说明（待发布）

### #272

> 本轮归档能力矩阵、候选合同研究与 #509 实测。新 #565 已登记《Komorebi》的真实翻译需求，先调查实际引擎版本、插件、文本范围及独立最小试点，再向本 issue 回填候选优先级和选型依据。生产 Adapter 与公共合同实现待调查决定后另拆，当前保持 open，本轮文档合并不触发关闭。
>
> #509 固定 MV 编辑器/core 1.5.1、TPP 8.8.9、实际 RMMVJS 2.19（入口 3.3）；严格文件保护仍 fail，漂移后译文期望 19 / 实际 18。它不证明 Komorebi 使用该版本，不泛化为 MZ、Naninovel 等引擎结论；Translator++ 保留独立使用，生产 bridge / MCP 不自动恢复。
>
> 有限研究不代表全部 P0/P1/P2 完成，原未完成项不勾选。后续实现须明确版本、合法素材及再分发边界、输入/覆盖范围与安全写回验收。待 #565 回填后再判断 #272 的范围和收口方式，不沿用先前的 not_planned 关闭草案。

### #147

> Story Memory MVP 已存在且默认关闭，本轮暂缓真实图谱试点与维护工具。原文“没有生产项目启用”是历史快照，不代表今天所有私人项目；本轮没有生成正式图谱或完成人工关系确认，原试点验收保持未完成。
>
> 仅在真实翻译反复出现 glossary/source_index 难解决的关系、称呼或口吻问题时重启：人工确认最小图谱（≥5 角色、≥10 关系），做小范围同条件开关对照，记录 manifest 非零命中、质量收益、费用/时间及整理更新维护成本；seed 不等于已确认关系，命中不等于质量改善。
>
> 当前保持低优先级 open，本轮不关闭。若以后决定将此试点移出计划，应按 **not_planned** 记录，不能将未完成试点标为 completed；#565 的引擎调查不自动触发 Story Graph 试点。

## 7. 本轮文档验收与交付状态

Python **3.14.0** 满足 3.11+ 要求。先阅读 `scripts/run_quality_gates.py` 与 `--help`，确认其为 lint/typecheck/依赖审计入口，没有 Markdown 链接子命令，因此使用一次性只读检查器；脚本与结果留在 gitignored `artifacts/issue-272-closeout-2026-10-22/`。

- 首次本地草稿：89 份 tracked Markdown / 562 个本地目标为无断链基线；加本报告后为 90 份 / 582 个目标、37 个章节锚点，检查通过。这是 #565 纳入前的历史计数。
- 2026-10-23 发布前在 `main@1a22021` 的隔离文档分支重查：全部 tracked Markdown 加本报告共 **91 份、584 个本地文件目标、37 个章节锚点**，无断链；基线 36 个锚点也均有效。
- `git diff --check` 与新增文件 whitespace 检查通过；提交前还需核对暂存区仅有六份文档且无 whitespace 错误。
- 仅文档修改，未运行无关完整 CLI/GUI 或候选引擎测试，未执行新模型/付费/私有游戏实验。

本轮范围为两份索引、能力矩阵、生态研究、Translator++ 评估和本报告，共六份文档。独立分支基于 PR #564 已合并后的 main；原工作区及既有改动保留，未切换其分支或暂存其内容。提交/PR/CI/合并以实际记录为准；本文不声称已合并，§6 仍为未发布的 issue 文字，不关闭 #272 / #147 / #565。
