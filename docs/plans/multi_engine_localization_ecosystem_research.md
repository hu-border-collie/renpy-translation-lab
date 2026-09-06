# 多引擎本地化适配与候选引擎开源生态研究

> **状态**：公开仓库源码与工具生态级研究记录，对接 [#265](https://github.com/hu-border-collie/renpy-translation-lab/issues/265) 与 [#272](https://github.com/hu-border-collie/renpy-translation-lab/issues/272)。
> **核对日期**：2026-09-06。
> **研究对象**：跨引擎本地化抽象框架（VNTextPatch、Translator++、translate-toolkit / Weblate）以及候选目标引擎（Naninovel、Godot+Dialogic 2、RPG Maker MV/MZ、KiriKiri）的原生生态与工具链。
> **前置约定**：本研究服务于 #265（引擎适配边界与版本化翻译资产）的产品化收口，以及 #272（能力矩阵与后续路线）中第三引擎立项决策；不违背既有的 `check -> apply`、声明式 `WritebackPlan` 与 Fail-Closed 门禁。

---

## 1. 摘要与核心结论

视觉小说与泛游戏本地化工具在开源与民间生态中存在深厚的积累。然而，通过源码级审视这些工具与引擎底层机制后，可以得出以下核心结论：

1. **“跨引擎提取”常见，但“版本化生命周期管理”极度匮乏**：
   * 绝大多数知名跨引擎工具（如 VNTextPatch、Translator++）的核心竞争力在于针对 30+ 种引擎文件格式的**逆向解包与静态字节/字符 Span 重定位（Offset Relocation）**；
   * 但它们普遍缺乏**版本快照（Snapshot）、文本演进对账（Reconciliation）与机器质量门禁（Quality Gate）**。一旦上游游戏更新源码，现有的翻译产物往往面临大面积错位或损坏。
2. **工业级本地化（如 Weblate / translate-toolkit）长于格式对账，但短于剧情拓扑**：
   * 通用软件本地化标准（PO, XLIFF）具备成熟的三向合并（3-way merge）与 Fuzzy 状态机；
   * 但它们无法原生表达视觉小说的**角色说话人（Speaker）、前后对白滑动窗口（Context Window）与代码/排版控制标签嵌套**。
3. **#272 对候选引擎的评估结论得到外部生态的强力验证**：
   * **Naninovel (Unity)**：官方原生内置了纯文本的 `Resources/Naninovel/Localization/{Locale}/` 目录与 CSV 交换机制。**完全不需要挂载 Unity 运行时即可实现离线原子写回**，印证了其作为唯一推荐第三引擎（评级 A+）的科学性；
   * **Godot + Dialogic 2**：生态中已有成熟的 CSV 翻译插件，但 Dialogic 2.0 目前处于 Alpha 20，API 频繁重构，过早立项必然背负巨大的接口废弃维护债务；
   * **RPG Maker MV/MZ**：社区工具（如 RPGMLocalizer）普遍采用直接原位重写数十兆的单体 `MapXXX.json`，极易导致工程损毁，且原生缺乏官方多语言支持，印证了其“评级 C- / 异构压力测试候选”的定位。
4. **对本项目的战略启示**：
   * 本项目在 `#265` 中建立的 `ProjectSnapshot`、`ReconciliationReport`、`TranslationRecord` 与声明式 `WritebackPlan`，在**数据严密性与防漂移安全性上明显超越了同类开源游戏翻译工具**；
   * 本项目应积极借鉴 **VNTextPatch 的字符 Span 重定位算法**、**Weblate 的 Fuzzy 微改审核状态**、**Translator++ 的数据网格交互直觉**，以及 **Naninovel 官方离线文本目录的直接消费模式**，完成从“文本翻译脚本”到“工业级多引擎工作台”的跃迁。

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

## 3. 架构级对标项目源码分析（面向 #265）

### 3.1 VNTextPatch / VNTranslationTools：跨引擎脚本重定位的标杆

* **关键仓库**：[arcusmaximus/VNTranslationTools](https://github.com/arcusmaximus/VNTranslationTools) · [rafael-vasconcellos/VNTextPatch-net8](https://github.com/rafael-vasconcellos/VNTextPatch-net8)
* **技术底座**：C# / .NET，涵盖 30+ 种商业与开源视觉小说引擎（Ren'Py, Kirikiri, CatSystem2, Majiro, BGI, Circus 等）。

#### 核心机制剖析
1. **清晰的抽象接口契约**：
   定义了通用的 `IScriptFile` / `IScriptFormat` 接口，每个引擎只需实现三个核心生命周期：
   * `Load(Stream stream)`：加载特定格式脚本或编译二进制；
   * `List<ScriptText> ExtractText()`：返回抽取出的结构化文本片段列表；
   * `void PatchText(List<ScriptText> texts)`：将翻译后的文本重写回脚本。
2. **字节与字符 Span 偏移重定位（Offset Relocation）**：
   在写回变长目标语言文本时，工具会解析脚本的字节码或控制流，自动修正所有跳转指令（Jumps / Labels / Pointers）的绝对偏移量，并支持在写回阶段按字符像素宽度进行自动折行（Word Wrapping）。

#### 对本项目的启发与边界
* ✅ **汲取**：其 `ScriptText`（含文本、角色名、位置标识）的设计高度契合本项目的 `TranslationUnit`；其针对不同脚本格式的 Span 替换与语法重定位思路，对完善本项目的 `text_span_replace` 算子很有参考价值。
* 🛑 **防范**：该工具完全面向“单次补丁制作”，**不具备任何版本快照（Snapshot）机制**。一旦游戏官方发布补丁更改了脚本行数或分支，原有的翻译文档将发生错位；本项目必须坚守 #265 的 `ProjectSnapshot` 对账红线。

---

### 3.2 Translator++：全流程多引擎工作台的产品化形态

* **关键参考**：[Dreamsavior Translator++](https://dreamsavior.net/) · [TranslatorPlusPlusChineseWiki](https://github.com/zyf722/TranslatorPlusPlusChineseWiki)
* **技术底座**：Node.js / Electron / NW.js，支持 RPG Maker (全部版本)、Wolf RPG Editor、Ren'Py、TyranoScript、Unity 等。

#### 核心机制剖析
1. **统一的中间数据网格（TransData Grid）**：
   无论导入哪种引擎，全部转化为标准网格模型：
   * `Tag / Context`：记录对白来源（地图、事件编号、代码行）；
   * `Speaker`：说话人识别；
   * `Source / Target`：原文与当前译文；
   * `Status`：翻译状态（未译、机器生成、人工验证）。
2. **启发式工程更新（Update Project）**：
   允许用户在游戏更新版本后，载入新版游戏并与旧翻译工程做 Diff：通过上下文标签与原文相似度，将已翻译文本自动迁移到新工程中。

#### 对本项目的启发与边界
* ✅ **汲取**：其网格数据模型的直观性与多引擎插件架构值得 GUI 参考，尤其是面向人工审校的表格化呈现。
* 🛑 **防范**：其增量更新主要依赖字符串与行号启发式匹配，**缺乏基于源文件哈希与片段 SHA-256 的严格门禁**。在复杂分支剧情中，经常出现“由于上下文微调导致旧译文被错误塞入新对白”的静默灾难；本项目 `#265 P4` 的 `ReuseCandidate` 必须保持严格的人工确认门禁。

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

### 4.1 Naninovel (Unity)：首选推荐候选的生态印证

* **官方体系**：Unity 商业插件，官方文档见 [Naninovel Localization Guide](https://naninovel.com/guide/localization.html)。

#### 关键源码事实与生态核验
1. **离线纯文本与独立 Locale 目录**：
   * Naninovel 原生设计了 `Resources/Naninovel/Localization/{Locale}/` 架构；
   * 运行官方 `Localization Tool` 后，会在该目录下生成与源脚本**同名且行级镜像对应的纯文本脚本**；
   * 每个待译行上方均带有由引擎生成的全局稳定文本标识符：
     ```text
     # 8c2d5a3b
     Original Japanese sentence here.
     ```
2. **Spreadsheet CSV 交换工具**：
   * 官方内置 `Spreadsheet Tool`，可直接将工程所有待译文本导出为标准 `.csv` 表格，翻译后再通过官方工具重新导入生成本地化脚本。
3. **完全无需 Unity 运行时的写回能力**：
   * 经核实，Naninovel 的本地化文件在打包前就是项目 Assets 目录下的普通文本文件（UTF-8）；
   * 本项目的 `EngineAdapter` **完全不需要启动 Unity，也不需要碰复杂的 C# 反射或 IL2CPP 编译注入**，直接作为离线文件适配器对 `Localization/{Locale}` 文本或 CSV 执行声明式写回即可。

#### 对比反思：为什么坚决不用 XUnity.AutoTranslator？
* 社区常有使用 [XUnity.AutoTranslator](https://github.com/bbepis/XUnity.AutoTranslator) 汉化 Unity 游戏的先例；
* 但 XUnity 依赖 BepInEx/MelonLoader 在游戏运行时动态 Hook 拦截 `TextMeshPro` 渲染，面对包含动态变量插值（Interpolation）的对白极易发生正则击穿或内存泄漏，无法生成供官方或民间直接发布的标准本地化包；
* **裁定**：本项目支持 Naninovel 时，必须坚定走官方 `Localization` 文本目录与 CSV 的离线静态写回通道。

---

### 4.2 Godot + Dialogic 2：下一阶段评估对象的生态状态

* **关键仓库**：[coppolaemilio/dialogic](https://github.com/coppolaemilio/dialogic) · [dannygaray60/godot-localization-editor](https://github.com/dannygaray60/godot-localization-editor)

#### 关键源码事实与生态核验
1. **Godot 原生 TranslationServer**：
   * Godot 引擎原生支持 CSV 矩阵文件（`keys,en,zh_CN,...`）和 Gettext `.po` 文件；
   * 游戏启动时，引擎会自动将 CSV / PO 编译为紧凑的二进制 `.translation` 资源。
2. **Dialogic 2.0 的本地化出口**：
   * Dialogic 2 专门提供了针对 Godot 本地化系统的导出通道，但目前**仅官方原生支持导出 CSV**；
   * 其 Timeline 行文本与角色名会被提取为带稳定路径的 key（形如 `Text/1/text`）。
3. **版本成熟度阻断**：
   * 截至目前，Dialogic 2 依然处于 **2.0 Alpha 20**；其数据存储格式与 API 在每个 Alpha 版本间都会发生剧烈破坏性变更（Breaking Changes）；
   * **裁定**：技术方案高度清晰（消费 Dialogic CSV 矩阵），但必须严格维持 #272 的决策——**在 Dialogic 2 达到 Beta/RC 且 API 冻结前，坚决不进入正式开发**。

---

### 4.3 RPG Maker MV/MZ：单体大 JSON 的写回风险印证

* **关键仓库**：[Lord0fTurk/RPGMLocalizer](https://github.com/Lord0fTurk/RPGMLocalizer) · [dazedanon/DazedMTLTool](https://github.com/dazedanon/DazedMTLTool)

#### 关键源码事实与写回灾难分析
1. **Monolithic JSON 架构**：
   * RPG Maker MV/MZ 的所有剧情、事件、对白全部存储在 `data/MapXXX.json`、`data/CommonEvents.json` 与 `data/System.json` 中；
   * 对白散落在嵌套极深的事件指令列表（`list` 数组）中，对白事件代码为 `401`，选项为 `102`。
2. **破坏性写回的高危性**：
   * 外部工具（如 RPGMLocalizer）在汉化时，必须反序列化庞大的单体 JSON，原地修改字符串后再全量序列化落盘；
   * 一旦在写回过程中遇到非标准插件注入的数据字段、编码异常或结构微调，**会导致整张地图数据损毁、游戏彻底无法加载**。
3. **缺乏官方原生多语言**：
   * 引擎原生没有类似 Ren'Py `tl/` 或 Naninovel `Localization/` 的概念，必须依赖第三方插件（如 DK_Localization）才能在运行时切换语言；
   * **裁定**：完全印证了 #272 将其定为 **“评级 C-（异构压力测试候选）”** 的判断——不具备原生目录写回安全边界，当前阶段不予支持。

---

## 5. 跨项目架构矩阵与关键机制对照

| 对标项目 | 核心定位 | 引擎模式 | 写回与安全保障 | 版本对账与差分能力 | 对本项目的关键借鉴与警示 |
|---|---|:---:|---|---|---|
| **VNTextPatch** | 跨 30+ 引擎提取/补丁器 | 源码解析与二进制重写 | 字节/字符 Span 重定位；自动 Word Wrapping | ❌ 无快照，上游更新易错位 | 借鉴抽象接口与 Span 重定位；坚守版本快照。 |
| **Translator++** | 通用跨引擎本地化工作台 | 统一中间网格 (TransData) | 引擎专用插件导出；缺乏原子哈希校验 | ⚠️ 启发式字符串匹配，缺乏门禁 | 借鉴数据网格交互；防范无门禁的静默合并。 |
| **translate-toolkit (Weblate)** | 工业级本地化底座 | 通用格式 (PO/JSON/CSV) | 格式无损往返读写 (Roundtrip) | ✅ 成熟的三向合并与 Fuzzy 降级 | 吸收 Fuzzy 降级状态机；弥补游戏剧情上下文。 |
| **Naninovel Built-in** | Unity 顶级视觉小说插件 | 原生纯文本目录 + CSV | 离线纯文本写回，无需 Unity 运行时 | ✅ 官方文本 ID 增量更新与保留 | **证实为最佳第三引擎候选**；直接消费离线目录。 |
| **Dialogic 2 (Godot)** | Godot 剧情对白系统 | 原生 CSV 矩阵导出 | 依赖 Godot 启动时编译 `.translation` | ⚠️ Alpha 阶段频繁重构 | 技术路线可行；严格等待 Beta/RC 稳定。 |
| **RPGMLocalizer** | RPG Maker JSON 批翻译器 | 暴力解析 `MapXXX.json` | 危险的原位 Monolithic JSON 覆写 | ❌ 极差，重新生成易破坏事件树 | **印证高危预警**；严禁引入全量破坏性写回。 |
| **renpy-translation-lab** | **工业级 VN 本地化工作台** | **抽象 Adapter + 混合双轨** | **声明式 WritebackPlan + 原子 Span SHA-256 门禁** | **ProjectSnapshot + Reconciliation 严格对账** | **坚守架构领先优势，补齐产品化交互。** |

---

## 6. 对 #265 与 #272 的演进路线建议

### 6.1 对 Issue #265 的产品化收尾（P6）建议

1. **GUI 接入快照对账与复用审查**：
   * 参考 Translator++ 的直观网格设计，在 GUI 中将 `ProjectSnapshot` 对账结果与 `ReuseCandidate` 呈现为直观的可视化卡片；
   * 严格遵循 Weblate 规范：对源文本微改的条目默认标记为 `needs_review`，不经人工确认绝不放行。
2. **强化 Span 替换与排版防御**：
   * 吸收 VNTextPatch 经验，在 `engine_adapters/writeback.py` 中增强长文本自动换行与字符集安全检查。
3. **固化 Fail-Closed 机器退出契约**：
   * 确保 CLI 在机器调用模式（`--output json --strict-exit-codes`）下，任何门禁阻断都能输出不可篡改的结构化 Envelope。

### 6.2 对 Issue #272 的第三引擎（Naninovel）实施预研规范

当 #265 P6 正式交付并关闭后，正式启动 Naninovel Adapter 时必须遵守以下规范：

1. **架构模式归属**：
   * 采用 `native_catalog` / `hybrid` 模式：优先读取并操作 `Resources/Naninovel/Localization/{Locale}/` 下的纯文本脚本；
   * 辅助支持官方 Spreadsheet CSV 格式的双轨对账。
2. **严格保持无运行时依赖**：
   * Naninovel Adapter 必须作为纯 Python 离线文件处理器实现，严禁引入对 Unity Editor 或 Mono/IL2CPP 环境的强依赖。
3. **对齐版本快照与写回门禁**：
   * 将 Naninovel 的行级 `# id` 映射为稳定的 `TranslationUnit.identity`；
   * 写回计划必须遵循声明式 `WritebackPlan`，在验证目标文件快照哈希与内容校验无误后方可执行写入。

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
- [Engine Adapter P0：Ren'Py 当前调用链与合同设计](engine_adapter_contract.md)（#265 P0 契约文档）
- [GitHub 视觉小说本地化工具源码研究](github_localization_projects_research.md)（跨项目对比与迁移规范）
- [Engine Adapter、覆盖审计与安全写回](../engine_adapter.md)（P1–P5 现行实现说明）
- [engine_adapters/writeback.py](../../engine_adapters/writeback.py)（声明式原子写回内核）
- [engine_adapters/versioning.py](../../engine_adapters/versioning.py)（项目版本快照与对账）
- [engine_adapters/reuse.py](../../engine_adapters/reuse.py)（版本化翻译记录与复用）
- [engine_adapters/tyrano.py](../../engine_adapters/tyrano.py)（第二引擎 TyranoScript V600+ 验证 Adapter）
