# 多引擎本地化适配与候选引擎开源生态研究

> 归档于 2026-10-27：保留原日期、基线、结果与未验边界。现行操作见[文档地图](../../README.md)，当前工作见[计划索引](../../plans/README.md)。

> **状态**：公开仓库源码与工具生态级研究记录，对接 [#265](https://github.com/hu-border-collie/renpy-translation-lab/issues/265) 与 [#272](https://github.com/hu-border-collie/renpy-translation-lab/issues/272)。
> **核对日期**：2026-09-06（原生态研究）；2026-09-16 专项刷新 Translator++ 接口、接入边界及 #265 / #272 状态，其余工具与引擎未做全量重新审计。
> **研究对象**：跨引擎本地化抽象框架（VNTextPatch、Translator++、translate-toolkit / Weblate）以及候选目标引擎（Naninovel、Godot+Dialogic 2、RPG Maker MV/MZ、KiriKiri）的原生生态与工具链。
> **当前路线（2026-10-22）**：新 [#565](https://github.com/hu-border-collie/renpy-translation-lab/issues/565)以 Komorebi 真实需求启动只读调查，#272 保持 open 等待版本/文本范围与选型回填；生产实现待调查后另拆。见[研究归档与调查衔接](issue-272-research-closeout-2026-10-22.md)。本轮仅复核 Naninovel 关键合同、MV/MZ 版本与 #509；其他工具/引擎仍为历史研究，不改变既有安全门禁，也不把目标游戏当可再分发 fixture。
> **专项评估**：[Translator++ 多引擎接入评估](translatorpp_integration_assessment.md)记录当前 issues、官方证据、公共合同缺口与最小实验；接入尚未实现。
> **#509 续验补记（2026-10-22）**：[单版本往返报告](../experiments/issue-509-translatorpp-roundtrip-2026-10-21.md)固定 TPP **8.8.9** / 实际 RMMVJS **2.19**（新建入口3.3）和 MV 编辑器/core **1.5.1**。已有实际GUI工程、限定19处译文导出、独立null保存/重开/导出、部分原生/HTML5运行及staging失败恢复；无修改全资源导出改写插件注释，源前插命令后缺一处context译文，畸形/缺失源文件被排除。限定spike证据已齐，决定保留独立工具使用、生产bridge暂缓。下方其他工具/引擎仍按原研究日期解释。

---

## 1. 摘要与核心结论

本研究按公开源码、官方接口和本仓库实现比较可复用机制；不同工具的证据深度并不相同，不能据此给出全面优劣排名。

1. **外部提取与导出能力值得复用，但须逐条验证合同**：
   * VNTextPatch 的格式接口与重定位机制、Translator++ 的 parser 和统一工程制品可以减少部分引擎的重复开发，不能把所有工具归为同一种 Span 实现；
   * Translator++ 可作为可选外部格式后端。导入后仍须建立与本项目 snapshot、occurrence、结果和写回门禁的绑定；公开资料未证明双方合同等价，也不足以断言上游必然没有版本或安全保护。
2. **工业级本地化（如 Weblate / translate-toolkit）长于格式对账，但短于剧情拓扑**：
   * 通用软件本地化标准（PO, XLIFF）具备成熟的三向合并（3-way merge）与 Fuzzy 状态机；
   * 但它们无法原生表达视觉小说的**角色说话人（Speaker）、前后对白滑动窗口（Context Window）与代码/排版控制标签嵌套**。
3. **候选生态提供机制线索，尚不足以确定第三 Adapter**：
   * **Naninovel (Unity)**：官方目录与 CSV 可作原生候选；默认 ID 是内容 hash，显式 ID 才可维持编辑后的关联，仍须源/语境核对。公共写回适配、空目标与运行未验证，取消无条件 A+ 首选结论；
   * **Godot + Dialogic 2**：Alpha 20 是原研究窗口，本轮未刷新；版本与导出合同须在真实需求出现后重查，不把历史成熟度当当前排名；
   * **RPG Maker MV/MZ**：叙事范围不能消除插件、分支和数组定位风险。#509 新增 MV 1.5.1 原创覆盖层与 TPP 有限实测，但保留严格文件审计失败及漂移缺译；原生 Adapter 暂缓，MZ 不继承结论。旧 B+/C- 仅作历史假设。
4. **对本项目的战略启示**：
   * 本项目已建立的 `ProjectSnapshot`、`ReconciliationReport`、`TranslationRecord` 与声明式 `WritebackPlan` 可作为接入基础；新增格式仍需要公共合同扩展、产品接线和独立验收；
   * Translator++ 的价值不只在数据网格交互，还在可复用的提取/导出与自动化接口。应比较“Agent + Translator++”和“再加入本项目工具包”的实际收益，同时保留 Naninovel 官方离线目录路线，避免重复建设已有能力。

---

## 2. 对标维度与研究方法

针对 #265 与 #272 的核心诉求，本研究从以下两大维度展开源码级与生态级对标：

### 2.1 #265 引擎适配架构对标维度
* **适配器解耦度（Adapter Abstraction）**：引擎特有逻辑与通用流水线是否彻底分离；
* **写回安全性（Writeback Safety）**：是破坏性覆写、运行时 Hook 替换，还是基于 SHA-256 校验的声明式原子写入；
* **版本差分与对账（Reconciliation & Diff）**：上游脚本修改后能否精确识别新增、移动、修改与删除；
* **资产溯源与复用（Provenance & Reuse）**：历史译文是简单字符串命中，还是带有快照指纹与上下文校验的候选审查。

### 2.2 #272 目标引擎原生机制对标维度
* **原生本地化制品形态**：纯文本脚本、CSV 表格、Gettext PO 还是单体大 JSON；
* **运行时依赖边界**：是否需要启动游戏引擎运行时（Unity/Godot/Web）才能完成写回；
* **稳定 ID 与重定位机制**：是否有行号无关的稳定文本 Key（Text Identifier）；
* **生态工具与破坏性经验**：社区已有工具采用何种方案，存在哪些崩溃与维护陷阱。

---

## 3. 架构级对标项目源码与接口分析（面向 #265）

### 3.1 VNTextPatch / VNTranslationTools：跨引擎脚本重定位的标杆

* **关键仓库**：[arcusmaximus/VNTranslationTools](https://github.com/arcusmaximus/VNTranslationTools) · [rafael-vasconcellos/VNTextPatch-net8](https://github.com/rafael-vasconcellos/VNTextPatch-net8)
* **技术底座**：C# / .NET，涵盖 30+ 种商业与开源视觉小说引擎（Ren'Py, Kirikiri, CatSystem2, Majiro, BGI, Circus 等）。

#### 核心机制剖析
1. **清晰的抽象接口契约**：
   上游 [IScript](https://github.com/arcusmaximus/VNTranslationTools/blob/main/VNTextPatch.Shared/IScript.cs) 声明了 `Extension` 属性与以下方法（2026-09-07 源码复核；fork 须单独核对）：
   * `void Load(ScriptLocation location)`：加载脚本；
   * `IEnumerable<ScriptString> GetStrings()`：枚举待译字符串；
   * `void WritePatched(IEnumerable<ScriptString> strings, ScriptLocation location)`：写出补丁脚本。
2. **字节与字符 Span 偏移重定位（Offset Relocation）**：
   在写回变长目标语言文本时，工具会解析脚本的字节码或控制流，自动修正所有跳转指令（Jumps / Labels / Pointers）的绝对偏移量，并支持在写回阶段按字符像素宽度进行自动折行（Word Wrapping）。

#### 对本项目的启发与边界
* ✅ **汲取**：加载、枚举、写出三个职责的分离值得参考。[ScriptString](https://github.com/arcusmaximus/VNTranslationTools/blob/main/VNTextPatch.Shared/ScriptString.cs) 仅包含 `Text` 与 `Type`，不自带角色名字段、位置标识或版本溯源，不能等同于本项目的 `TranslationUnit`。格式专用的重定位机制可作为研究对象，不能直接替代公共写回门禁。
* 🛑 **防范**：该工具完全面向“单次补丁制作”，**不具备任何版本快照（Snapshot）机制**。一旦游戏官方发布补丁更改了脚本行数或分支，原有的翻译文档将发生错位；本项目必须坚守 #265 的 `ProjectSnapshot` 对账红线。

---

### 3.2 Translator++：全流程多引擎工作台的产品化形态

* **关键参考**：[官方格式参考](https://dreamsavior.net/docs/translator-developers-guide/trans-format/format-reference/) · [8.9.2 发行说明](https://dreamsavior.net/update-info-translator-ver-8-9-2/) · [8.9.9 发行说明](https://dreamsavior.net/update-info-translator-ver-8-9-9/)
* **证据范围**：9 月官方接口/发行说明为历史快照；10 月 #509 已完成 TPP 8.8.9 / RMMVJS 2.19 / MV 1.5.1 有限往返，不能概括为其他 Beta、parser 或引擎已验证。

#### 核心机制剖析

1. **统一工程制品**：`.trans` 是 JSON；`project.files[*].data` 为原文/多译文列网格，允许空值；`context` 是逐行上下文列表。格式参考页较旧，当前布局与有效译文列需实测。
2. **上下文与去重**：同一行可能对应多个真实位置，并有按 context 指定译文的机制；context 的定位语义由 parser 决定，不能自动推导统一 speaker、稳定 occurrence 或人工审核状态。[Context](https://dreamsavior.net/docs/translator/getting-started/context/) · [按上下文翻译](https://dreamsavior.net/docs/translator/getting-started/translation-by-context/)
3. **导出依赖**：`.trans` 不等于完整自包含工程，生成补丁依赖 staging；版本冻结与备份需覆盖实际导出输入。[Staging files](https://dreamsavior.net/docs/translator/staging-files/)
4. **自动化接口**：8.9.2 Beta 已公告 `/mcp` Streamable HTTP 和工具发现/调用；8.9.9 Beta 列出 RPG Maker MV/MZ Parser 2.21、Unity binary translator 0.1 等更新。接口可用性、具体 parser 覆盖和恢复行为仍须按实际安装版本验证。

#### 对本项目的启发与边界

* **当前决定**：#509 结果支持独立使用，生产 bridge 暂缓；旧“先文件、后自动化”只作将来满足重启条件后的实验顺序，不是已排期实现。
* **保留的责任**：本项目继续负责结果身份、完整性、冲突/stale、源引擎结构规则及受控写回。`.trans` 的数组/空值、同文多 context 和 staging 都使接入不止是字段转换；当前公共 JSON writer 还不支持这些数组路径。
* **证据边界**：不再以未经验证的“缺乏哈希”“经常静默错位”评价上游。是否能满足本项目合同、能节省多少工作，按[专项评估](translatorpp_integration_assessment.md)的对照实验和验收清单判断；上游模型/Batch 能力与本项目也有重合。

---

### 3.3 translate-toolkit / Weblate：工业级本地化与格式对账底座

* **关键仓库**：[translate/translate](https://github.com/translate/translate) · [WeblateOrg/weblate](https://github.com/WeblateOrg/weblate)
* **技术底座**：Python，全球自由软件与多语言工程的事实标准底座。

#### 核心机制剖析
1. **严格的 Unit / Store 抽象**：
   每个可翻译条目（`TranslationUnit`）天然绑定 `id`、`source`、`target`、`locations`、`context`，并具备完整的标志位（`approved`、`fuzzy / needs-review`、`translated`）。
2. **三向合并与版本漂移防御（3-way Merge & Fuzzy Matching）**：
   当上游源文件更新时（类似 `msgmerge` 流程）：
   * 原文完全一致 -> 自动保留 `approved` 译文；
   * 原文发生微小修改（拼写修正、标点变化） -> 自动迁移旧译文，但**强制降级为 `fuzzy`（待人工复查），并阻止其直接打包发布**；
   * 原文删除 -> 归档为无用条目，不破坏现有数据库。

#### 对本项目的启发与边界
* ✅ **汲取**：这种“源文本发生任何变动即降级为待复核，绝不自动放行”的哲学，与本项目 `#265 P3 / P4` 完全一致。可直接吸收其 Fuzzy 状态机设计，优化 GUI 中的复用候选审查体验。
* 🛑 **防范**：面向通用文本格式（PO/JSON/XLIFF），缺乏对游戏特有的剧情树上下文（Dialogue Scene History）和角色说话人属性的深度支持，不能直接作为游戏剧本提取器使用。

---

## 4. 候选目标引擎生态与落地实践（面向 #272）

### 4.1 Naninovel (Unity)：原生目录条件候选

* **官方体系**：Unity 商业插件，官方文档见 [Naninovel Localization Guide](https://naninovel.com/guide/localization.html)。

#### 关键源码事实与生态核验
1. **离线纯文本与独立 Locale 目录**：
   * Naninovel 原生设计了 `Resources/Naninovel/Localization/{Locale}/` 架构；
   * 运行官方 `Localization Tool` 后，在 `Localization/{Locale}/Text/Scripts/` 生成按文本 ID 组织的本地化文档，不是源 `.nani` 的逐行镜像；Managed Text 位于 `Text/`；
   * 按 [Naninovel 1.21 官方格式](https://naninovel.com/guide/localization#scripts-localization)，条目由 `# ID`、`;` 原文注释和译文组成：
     ```text
     # 8c2d5a3b
     ; Original source sentence here.
     这里是译文。
     ```
   * 译文允许占多行，直到下一个 `# ID`；Join Lines 会生成 `# id1|id2|id3` 与 `|` 分隔的片段，`; >` 注解不得进入译文。解析与 fixture 必须覆盖这些结构，不能用物理行号一一对应。
2. **Spreadsheet CSV 交换工具**：
   * 官方内置 `Spreadsheet Tool`；须先生成本地化数据，再导出 `.csv`，翻译后通过官方工具导入。编辑 CSV 本身不等于完成资源导入。
3. **离线编辑与官方工具链分开**：
   * 经核实，Naninovel 的本地化文件在打包前就是项目 Assets 目录下的普通文本文件（UTF-8）；
   * Adapter 可离线修改已生成的本地化文档；生成、CSV 导入、构建与运行时验证仍属于官方工具链。公共 `text_span_replace` 与 `multiline_text_span_replace` 已支持单行/跨行非空源片段替换（#471），空源片段插入仍不支持；Naninovel 条目如何形成合法 plan 仍须设计与夹具验证，不能由“纯文本”推定已接入。

#### 与 Unity 通用翻译工具的边界

* 社区常有使用 [XUnity.AutoTranslator](https://github.com/bbepis/XUnity.AutoTranslator) 汉化 Unity 游戏的先例；
* 运行时捕获/替换与消费 Naninovel 官方 locale 制品是不同路径，不能据此推定相同的文本 ID、覆盖范围或导出合同；
* Translator++ 已公告 Unity binary translator 0.1，不能继续把其 Unity 能力仅描述为 XUnity 路线，但本研究尚无 Naninovel 专项往返证据；
* **条件判断**：若需求触发 Naninovel 实验，先核对官方目录与 CSV；外部 Unity parser 另验。没有已确定的第三 Adapter，也没有 Naninovel 专项实测。

---

### 4.2 Godot + Dialogic 2：下一阶段评估对象的生态状态

* **关键仓库**：[coppolaemilio/dialogic](https://github.com/coppolaemilio/dialogic) · [dannygaray60/godot-localization-editor](https://github.com/dannygaray60/godot-localization-editor)

#### 关键源码事实与生态核验
1. **Godot 原生 TranslationServer**：
   * Godot 引擎原生支持 CSV 矩阵文件（`keys,en,zh_CN,...`）和 Gettext `.po` 文件；
   * CSV 由 Godot 编辑器的资源导入器生成 `.translation` 资源，游戏运行时加载导入后的资源；Gettext `.po` 另有资源加载支持，不能统一描述为启动时编译。离线修改 CSV 后须重新执行 Godot 资源导入，才能验证游戏中的新译文。参见 [Godot 翻译导入说明](https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/importing_translations.html)。
2. **Dialogic 2.0 的本地化出口**：
   * Dialogic 2 专门提供了针对 Godot 本地化系统的导出通道，但目前**仅官方原生支持导出 CSV**；
   * 其 Timeline 行文本与角色名会被提取为带稳定路径的 key（形如 `Text/1/text`）。
3. **版本成熟度阻断**：
   * 原研究记录为 **2.0 Alpha 20**；本轮未刷新发行状态，版本变更风险需届时核对；
   * **裁定**：技术方案高度清晰（消费 Dialogic CSV 矩阵），但必须严格维持 #272 的决策——**在 Dialogic 2 达到 Beta/RC 且 API 冻结前，坚决不进入正式开发**。

---

### 4.3 RPG Maker MV/MZ：单体大 JSON 的写回风险印证

* **关键仓库**：[Lord0fTurk/RPGMLocalizer](https://github.com/Lord0fTurk/RPGMLocalizer) · [dazedanon/DazedMTLTool](https://github.com/dazedanon/DazedMTLTool)

#### 关键源码事实与写回灾难分析
1. **Monolithic JSON 架构**：
   * 本研究的 RPG Maker 叙事范围以 `data/MapXXX.json` 与 `data/CommonEvents.json` 为主；系统、数据库、插件和动态文本另计，不声称这些文件覆盖所有剧情；
   * 对白散落在嵌套极深的事件指令列表（`list` 数组）中，对白事件代码为 `401`，选项为 `102`。
2. **破坏性写回的高危性**：
   * 整体重写 JSON 时，需要核对字段、数组、数值/类型与非目标资源；仅凭解析成功不能证明安全。
   * `[推断]` 非标准插件字段、编码与结构变化可能造成副作用；本轮没有重新审计上述社区工具，不将风险推断写成其必然损毁地图的实测结论。
3. **缺乏官方原生多语言**：
   * 引擎原生没有类似 Ren'Py `tl/` 或 Naninovel `Localization/` 的概念，必须依赖第三方插件（如 DK_Localization）才能在运行时切换语言；
   * **裁定**：对**未收敛的全量 RPG Maker**（战斗数值、复杂插件、粗暴 JSON 覆写），#272 原 **C- / 不予泛化支持** 的判断仍然成立。
   * **收敛后的叙事模式**：仍只是候选范围，需事件/context 身份、独立 coverage、声明式写回与运行验收；#509 不证明本项目原生写回成立，不再以旧 B+ 表示当前排名。
   * **公共合同缺口**：现有 `json_catalog_set` 是字典键路径的写回操作，不是 Adapter 模式，也不支持事件树数组索引。须先完成矩阵 §5.4 所列公共合同扩展与拒绝路径验证，不能将叙事模式写回标为已实现。

---

## 5. 跨项目架构矩阵与关键机制对照

| 对标项目 | 核心定位 | 引擎模式 | 写回与安全保障 | 版本对账与差分能力 | 对本项目的关键借鉴与警示 |
|---|---|:---:|---|---|---|
| **VNTextPatch** | 跨 30+ 引擎提取/补丁器 | 源码解析与二进制重写 | 字节/字符 Span 重定位；自动 Word Wrapping | ❌ 无快照，上游更新易错位 | 借鉴抽象接口与 Span 重定位；坚守版本快照。 |
| **Translator++** | 跨引擎工作台 / 可选外部格式后端 | `.trans` 网格 + parser / staging | 专用 parser 导出；与本项目门禁的等价性待验证 | 须验证上下文、版本变化与复用语义，不能据行号认定稳定身份 | 优先评估文件 bridge，保留公共结果/写回责任；详见[专项评估](translatorpp_integration_assessment.md)。 |
| **translate-toolkit (Weblate)** | 工业级本地化底座 | 通用格式 (PO/JSON/CSV) | 格式无损往返读写 (Roundtrip) | ✅ 成熟的三向合并与 Fuzzy 降级 | 吸收 Fuzzy 降级状态机；弥补游戏剧情上下文。 |
| **Naninovel Built-in** | Unity 视觉小说插件 | 原生纯文本目录 + CSV | 已生成文本可离线处理；生成/导入/构建/运行依赖官方工具链 | 默认内容 hash；显式 ID 与源对照仍必要 | 原生目录条件候选，非最佳排名或已接入证明。 |
| **Dialogic 2 (Godot)** | Godot 剧情对白系统 | 原生 CSV 矩阵导出 | 离线改写 CSV 后由 Godot 编辑器重新导入；运行时加载资源 | ⚠️ Alpha 阶段频繁重构 | 技术路线可行；严格等待 Beta/RC 稳定。 |
| **RPGMLocalizer** | RPG Maker JSON 翻译工具（历史研究） | `MapXXX.json` 等 | 本轮未重新审计，不能从整体序列化推定必然损坏 | 目标版本的身份/迁移能力待核对 | 本项目候选仍须节点身份、快照与公共安全写回，见矩阵文档。 |
| **renpy-translation-lab** | VN 本地化工作台 | 抽象 Adapter + 混合双轨 | 声明式 WritebackPlan、源快照校验与写回事务 | ProjectSnapshot + Reconciliation | 复用现有核心；第三引擎与外部制品仍需合同扩展、产品接线和独立验收。 |

---

## 6. 对 #265 与 #272 的演进路线建议

### 6.1 Issue #265 已交付边界

#265 P5/P6 已交付，Epic 已于 2026-09-13 关闭，不再把旧研究建议列作 P6 待办。当前 Ren'Py 产品化、coverage、版本对账/复用、CLI/GUI 与 Tyrano 离线 corpus 的边界，以 [Engine Adapter 现行说明](../../engine_adapter.md)为准。

后续借用外部编辑界面仍须保持复用决定、质量确认、审校状态与写回许可分离。自动折行、字体或新引擎结构规则须单独定义适用范围，不能作为格式 bridge 的隐式副作用。

### 6.2 对 Issue #272 的条件候选预研规范

Naninovel 保留为原生目录条件候选；#565 因具体需求优先调查 RPG Maker，未预定第三 Adapter。当前研究衔接与生产准入见[矩阵 §6.1](visual_novel_localization_matrix.md#61-最终决策结论)。将来若选择 Naninovel，须刷新版本/素材边界，并核对以下假设：

1. **架构模式归属**：
   * 采用 `native_catalog` / `hybrid` 模式：优先读取并操作 `Resources/Naninovel/Localization/{Locale}/` 下的纯文本脚本；
   * 辅助支持官方 Spreadsheet CSV 格式的双轨对账。
2. **限定离线处理范围**：
   * Adapter 核心按纯 Python 离线文件处理器设计；输入须已由官方工具生成。CSV 导入、资源构建与运行时验收仍依赖官方工具链，应在工作流中明确记录。
3. **对齐版本快照与写回门禁**：
   * 按矩阵 §4.1 / §5.2，将 Naninovel 文本 ID 作为 opaque locator 与 lineage 的候选证据；组合 ID 须保留片段关系，不能直接把 `# id` 同时当作 occurrence 与 lineage 身份；
   * 写回计划必须遵循声明式 `WritebackPlan`，在验证目标文件快照哈希与内容校验无误后方可执行写入。

### 6.3 Translator++ 外部格式路线

将“自行实现引擎 parser”与“接入外部格式后端”分开决策。#509 已给出单版本位置映射、选列和 staging 的有限证据及失败，当前保留独立工具使用，生产 bridge 暂缓；未来满足重启条件才评估公共工作包的净收益。生成 `.trans`、最终游戏导出与本项目写回许可须分别验收。

历史 issue 影响、格式合同缺口与停止条件见 [Translator++ 多引擎接入评估](translatorpp_integration_assessment.md)。[现有外部初译工作包](../../external_translation_work.md)限 Ren'Py，不自动接通第三引擎。

---

## 7. 参考资料

### 外部项目源码与官方文档

- [VNTextPatch (arcusmaximus)](https://github.com/arcusmaximus/VNTranslationTools)：跨引擎视觉小说文本提取与补丁工具集。
- [VNTextPatch-net8 (rafael-vasconcellos)](https://github.com/rafael-vasconcellos/VNTextPatch-net8)：现代跨平台 .NET 8 移植版。
- [Translator++ (Dreamsavior)](https://dreamsavior.net/)：通用跨引擎游戏本地化工作台。
- [translate-toolkit](https://github.com/translate/translate)：Python 工业级本地化转换与对账工具包。
- [Naninovel 官方本地化指南](https://naninovel.com/guide/localization.html)：Unity 官方本地化目录与表格工具说明。
- [Dialogic 2 (Godot)](https://github.com/coppolaemilio/dialogic)：Godot 对白系统与 CSV 本地化模块。
- [godot-localization-editor](https://github.com/dannygaray60/godot-localization-editor)：Godot 官方 CSV 翻译编辑器插件。
- [RPGMLocalizer (Lord0fTurk)](https://github.com/Lord0fTurk/RPGMLocalizer)：RPG Maker MV/MZ 数据文件解析与翻译工具。

### 本仓库现行核心文档与实现

- [视觉小说引擎本地化能力矩阵与后续 Adapter 路线](visual_novel_localization_matrix.md)（#272 决策文档）
- [Translator++ 多引擎接入评估](translatorpp_integration_assessment.md)（2026-09-16 官方接口、issue 影响与待实验接入合同）
- [Engine Adapter P0：Ren'Py 当前调用链与合同设计](../../plans/engine_adapter_contract.md)（#265 P0 契约文档）
- [GitHub 视觉小说本地化工具源码研究](github_localization_projects_research.md)（跨项目对比与迁移规范）
- [Engine Adapter、覆盖审计与安全写回](../../engine_adapter.md)（P1–P6 现行实现与验证边界）
- [engine_adapters/writeback.py](../../../engine_adapters/writeback.py)（声明式原子写回内核）
- [engine_adapters/versioning.py](../../../engine_adapters/versioning.py)（项目版本快照与对账）
- [engine_adapters/reuse.py](../../../engine_adapters/reuse.py)（版本化翻译记录与复用）
- [engine_adapters/tyrano.py](../../../engine_adapters/tyrano.py)（第二引擎 TyranoScript V600+ 验证 Adapter）
