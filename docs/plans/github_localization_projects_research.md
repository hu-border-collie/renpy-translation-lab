# GitHub 视觉小说本地化工具源码研究

> **状态**：公开仓库源码级研究记录，不是生产实现方案。
> **核对日期**：2026-09-06。
> **研究对象**：[renpy-translation-lab](https://github.com/hu-border-collie/renpy-translation-lab) 及 GitHub 上定位相近的视觉小说 / 游戏本地化工具。
> **证据边界**：本轮阅读了公开仓库的 README、目录、关键源码和部分测试；没有把所有外部项目完整安装并运行，因此动态兼容性不能由本文单独确认。
> **本地状态**：研究时 checkout 为 `main@5ea0ddd`；本地 `origin/main` tracking ref 领先 3 个提交，差异主要是 Ren'Py speaker label sibling detection。刷新本文时应重新核对当前 checkout 和远端默认分支。

## 1. 摘要结论

这些项目在表面功能上确实与本项目高度重合：扫描文本、批量翻译、术语表、翻译记忆、质量检查、审校、导出和多引擎支持都很常见。

但源码级对照后，项目之间存在明显的层次差异：

1. 有些项目是**玩家侧运行时补丁工具**，重点是尽快看到可玩的译文；
2. 有些项目是**简单的文本批处理器**，主要依赖正则、字典和原地重写；
3. 有些项目是**编辑器 / 翻译工作台**，重点是逐条审校、角色映射和状态保存；
4. 只有少数项目同时处理了**版本溯源、结构安全、任务恢复和可审计写回**。

本项目当前最有竞争力的部分，不是“能不能调用模型”，而是：

- `TranslationUnit`、TranslationPlan 和结构化模型响应合同；
- `check -> apply` 写回门禁；
- 源快照、预期片段哈希、声明式 `WritebackPlan`；
- EngineAdapter、coverage、版本快照和 provenance-carrying reuse；
- Ren'Py / Tyrano 的受限写回边界；
- 质量 finding、revision preview 和 final review 生命周期。

最值得补强的部分则是产品层：

- 模型调用前后的统一 placeholder / tag protection；
- 面向普通条目的逐条审校工作区；
- 带 provenance 的翻译记忆建议；
- 将现有 doctor、coverage、成本估算和质量报告聚合成可见的 preflight；
- 备份、恢复和回滚的 GUI 体验。

总体建议是：

> 借鉴 RenPyTranslator 的工作区边界、RenLocalizer 的结构保护、Dialogue Visual Editor 的审校状态、2R-Tools 的旧译复用思路和 GameStringer 的产品化交互；保留本项目现有的 manifest、snapshot、adapter 和 fail-closed 写回内核。

## 2. 研究方法与评价标准

### 2.1 本轮实际检查的内容

对重点项目进行了以下源码级检查：

- 核心数据模型：文本单元、角色、源文、译文、状态、版本和 provenance；
- 提取路径：正则、脚本解析、JSON / CSV / 原生 catalog 读取；
- 翻译路径：批处理、上下文、术语、占位符、失败重试和 checkpoint；
- 质量路径：结构检查、placeholder 检查、术语检查、评分和警告处理；
- 写回路径：是否原地修改、是否备份、是否先校验、是否允许错误输出；
- 审校路径：逐条编辑、状态保存、角色映射、控制码差异和人工确认；
- 测试结构：fixture、单元测试、契约测试和失败边界。

### 2.2 评价维度

| 维度 | 关注问题 |
|---|---|
| 数据模型 | 是否只有 `source -> target`，还是包含定位、版本、角色、上下文和来源？ |
| 结构安全 | 标签、变量、控制码和换行是否在模型调用前被保护？ |
| 增量能力 | 源文本变化后能否识别新增、删除、移动和修改？ |
| 复用可信度 | 旧译文是直接复用，还是只能生成待审核候选？ |
| 写回安全 | 是否有源快照、目标校验、原子写入和失败恢复？ |
| 审校体验 | 是否有长期保存的逐条状态，而不只是一次性报告？ |
| 引擎边界 | 是否明确版本、原生 catalog 和排除项？ |
| 测试证据 | 是否有最小工程和结构损坏 fixture？ |

本文使用以下证据标记：

- **源码事实**：直接来自公开源码或测试；
- **工程判断**：根据源码路径推断出的风险或适配价值；
- **待运行验证**：需要安装外部项目或建立本仓库 fixture 后才能确认。

## 3. 项目总览

### 3.1 外部项目定位与得失对照

| 项目 | 源码定位 | 最值得借鉴 | 不应直接照搬 |
|---|---|---|---|
| [RenLocalizer](https://github.com/Lord0fTurk/RenLocalizer) | Ren'Py 翻译器、语法保护和运行时相关工具 | placeholder / tag protection、术语长度降序替换、恢复分层 | 自动追加补回缺失 token 不能直接进入安全 `apply` |
| [Dialogue Visual Editor](https://github.com/vadondaniel/dialogue-visual-editor) | RPG Maker MV/MZ 与 TyranoScript 桌面审校编辑器 | 逐条模型、版本状态、角色映射、控制码审计、交互式 prompt/apply 工作流 | 其编辑器强绑 SQLite 写回模型不能替代本项目 Adapter 合同 |
| [RenPyTranslator](https://github.com/basil520/RenPyTranslator) | Ren'Py 工作区、Ollama 翻译、审校和导出 | 隔离工作区、断点 checkpoint、review 状态机、显式导出 | `rmtree + copytree` 式覆写导出不应成为本项目核心写回机制 |
| [translate-renpy](https://github.com/teo-lin/translate-renpy) | 分阶段 PowerShell / Python Ren'Py 工具链 | `.parsed.yaml` / `.tags.yaml` 文本与代码结构解耦分层 | 合并发现语法校验错误后仍强制输出，不符合 fail-closed |
| [2R-Tools](https://github.com/Phoenix525/2R-Tools) | Ren'Py 与 RPG Maker 多 API 桌面工具 | `TransLib.json` 历史译文沉淀与工作区概念 | 原文大写后作为无状态全局字典 key、原地重写缺少溯源与快照 |
| [GameStringer](https://github.com/rouges78/GameStringer) | 多引擎 Web 产品、TM、QA、patch 和成本估算 | 预检（preflight）、进度/质量仪表盘、分级 TM 建议和工作流产品化 | 简单 fuzzy TM 与评分不能替代 manifest 级写回门禁 |
| [rpgmaker-translator](https://github.com/MoriTranslates/rpgmaker-translator) | 多引擎（RPG Maker/Ren'Py 等）本地 LLM 高保真翻译器 | 两阶段管线（DB生成术语表再翻译对话）、事件边界上下文隔离、代词/性别推断 | 深度依赖 RPG Maker 事件树结构，写回缺乏 AST 级原子门禁 |
| [DeepRenPyTrans](https://github.com/Danko-Novak/DeepRenPyTrans) | Ren'Py 黑盒字典注入与质量审计工具 | 运行时未命中日志（`untranslated.log`）反哺提取、字典清洗与代码垃圾过滤 | 纯运行时 `config.replace_text` 容易破坏动态插值且缺乏工程版本化 |
| [RenForge](https://github.com/foulnike/RenForge) | Ren'Py 综合汉化 Mod 构建与多媒体管理器 | Text/Images/Audio 全要素资产面板、中文字体缺失/方块自动降级修补 | 运行时 Mod 补丁封包与动态替换不可作为长期源码维护方案 |
| [varo](https://github.com/rag91560-spec/varo) | 本地优先桌面泛游戏 AI 本地化平台 | 现代解耦架构（FastAPI+Next.js）、透明引擎 readiness（Stable/Beta）状态机、一键快照回滚 | 底层仍偏通用字符串替换，缺少针对特定引擎的 AST 契约校验 |

### 3.2 跨项目 8 大能力维度横向矩阵

对全部 10 个外部项目及本项目在 8 个关键技术维度进行横向对比（★ 极弱/缺失，★★ 基础/较弱，★★★ 中等/实用，★★★★ 优秀/先进，★★★★★ 行业标杆）：

| 项目 | 数据模型与溯源 | 结构与Tag保护 | 上下文边界管理 | 增量与复用可信度 | 写回安全与原子性 | 审校工作台体验 | 引擎边界清晰度 | 工程与产品化完备度 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RenLocalizer** | ★★ | ★★★★ | ★★ | ★★ | ★★ | ★★ | ★★★ | ★★★ |
| **Dialogue Visual Editor** | ★★★★ | ★★★ | ★★★ | ★★★ | ★★ | ★★★★★ | ★★★ | ★★★ |
| **RenPyTranslator** | ★★★ | ★★★ | ★★ | ★★★ | ★★ | ★★★★ | ★★★ | ★★★ |
| **translate-renpy** | ★★★ | ★★★★ | ★★ | ★★ | ★ | ★★ | ★★★ | ★★ |
| **2R-Tools** | ★ | ★★ | ★ | ★★ | ★ | ★★ | ★★ | ★★ |
| **GameStringer** | ★★★ | ★★★ | ★★★ | ★★★ | ★★ | ★★★★ | ★★★★ | ★★★★★ |
| **rpgmaker-translator** | ★★★ | ★★★ | ★★★★ | ★★★ | ★★ | ★★★ | ★★★★ | ★★★★ |
| **DeepRenPyTrans** | ★★ | ★★ | ★★★ | ★★★ | ★ (Hook) | ★★ | ★★★ | ★★★ |
| **RenForge** | ★★ | ★★ | ★★ | ★★ | ★ (Hook) | ★★★ | ★★★ | ★★★ |
| **varo** | ★★★ | ★★★ | ★★ | ★★★ | ★★★ | ★★★★ | ★★★★ | ★★★★ |
| **renpy-translation-lab (本项目)** | **★★★★★** | ★★★ | ★★★★ | **★★★★★** | **★★★★★** | ★★★ | ★★★★ | ★★★★ |

## 4. 源码级项目分析

### 4.1 RenLocalizer：最强的是 provider-facing 结构保护

关键源码：

- [glossary_manager.py](https://github.com/Lord0fTurk/RenLocalizer/blob/main/src/core/glossary_manager.py)
- [syntax_guard.py](https://github.com/Lord0fTurk/RenLocalizer/blob/main/src/core/syntax_guard.py)

#### 实际实现

`GlossaryManager` 不只是把术语附加到 prompt：

- 术语按源词长度降序匹配，避免短词先吞掉长词；
- `preserve_case()` 根据源词大小写调整目标术语；
- `protect_terms()` 在翻译前把术语替换为带随机命名空间的 placeholder；
- 同时支持 XML `<ph>` 形式和 Unicode bracket token 形式；
- 翻译后再应用目标术语。

`syntax_guard.py` 则把 Ren'Py 结构保护做成独立模块：

- 识别 `{tag}`、`{/tag}`、`[variable]`、格式符、转义结构和 ruby 等内容；
- 使用带命名空间的 token，降低模型改写 token 名称的概率；
- 恢复阶段处理 Unicode bracket 被剥离、token 被转写、token 周围空格变化和旧 token 格式；
- 对 Ren'Py 标签嵌套做有限修复；
- 检查缺失 placeholder，并提供 `inject_missing_placeholders()` 形式的恢复。

#### 对本项目的启发

真正值得迁移的是边界，而不是具体 token 格式：

```text
source text
    -> extract structural tokens
    -> protect glossary / tags / variables
    -> build provider request
    -> parse model response
    -> restore original structures
    -> validate exact structure
    -> produce translation result
```

本项目应该将这层放在 `TranslationUnit` / `TranslationPlan` 与 provider 之间，让 CLI、GUI、Batch 和同步翻译共用。

#### 不应照搬的部分

自动把缺失 token 按位置插回或追加到译文尾部，最多只能生成 repair preview：

- 位置可能已经发生语义变化；
- token 虽然存在，但可能落在错误的角色、标签或句子中；
- 自动修复会掩盖模型实际没有遵守结构合同这一事实。

因此，`restore` 可以用于生成候选，`apply` 仍必须由当前的结构检查、源快照和写回计划决定。

### 4.2 Dialogue Visual Editor：最值得借鉴的是“翻译数据编辑器”模型

关键源码：

- [models.py](https://github.com/vadondaniel/dialogue-visual-editor/blob/main/helpers/core/models.py)
- [version_state_db.py](https://github.com/vadondaniel/dialogue-visual-editor/blob/main/helpers/core/version_state_db.py)
- [translation_state_mixin.py](https://github.com/vadondaniel/dialogue-visual-editor/blob/main/helpers/mixins/translation_state_mixin.py)
- [persistence_export_mixin.py](https://github.com/vadondaniel/dialogue-visual-editor/blob/main/helpers/mixins/persistence_export_mixin.py)
- [mass_translate_dialog.py](https://github.com/vadondaniel/dialogue-visual-editor/blob/main/helpers/ui/mass_translate_dialog.py)

#### 实际实现

它的 `DialogueSegment` 不只是字符串对：

- `source_lines`、`translation_lines`；
- speaker / actor 信息；
- 控制码和脚本 entry 模板；
- 原始译文快照；
- 合并 segment 和 translation-only segment；
- 角色名翻译和推断开关。

它还维护 SQLite 版本状态：

- `original`、`working`、`translated` 版本；
- 多个 translated profile；
- 当前 applied version / profile；
- 文件级保存时间和状态。

`translation_state_mixin.py` 使用源文本、speaker 等信息生成 hash，并在旧状态无法直接匹配时使用受限的文本相似度恢复。这解决了编辑器最常见的问题：源文件插入或移动几行后，原有人工译文不应全部丢失。

审计层还提供：

- 控制码数量和位置差异；
- 术语使用检查；
- 名称一致性；
- 清理和规范化；
- 每文件覆盖率。

批量翻译对话框则采用“复制 prompt → 外部模型处理 → 粘贴 JSON → 审查 warning → 应用”的流程。应用前会检查行数和控制码差异，并允许跳过有问题的条目。

#### 对本项目的启发

本项目不需要复制一个独立 SQLite 编辑器，但应补一个与现有 manifest 体系兼容的审校视图：

- 每条 `TranslationUnit` 有稳定 identity；
- 展示源文本、当前译文、角色、上下文和结构 finding；
- 状态至少区分 `open`、`ignored`、`resolved`、`needs_recheck`；
- 人工修改先生成 revision proposal / preview，不直接改源文件；
- 角色映射和控制码检查复用现有质量 finding。

这会比继续增加一个“开始翻译”按钮更能补齐当前产品差距。

### 4.3 RenPyTranslator：工作区和恢复体验有参考价值

关键源码：

- [project.py](https://github.com/basil520/RenPyTranslator/blob/main/src/renpy_translator/core/project.py)
- [translation_pipeline.py](https://github.com/basil520/RenPyTranslator/blob/main/src/renpy_translator/core/translation_pipeline.py)
- [review.py](https://github.com/basil520/RenPyTranslator/blob/main/src/renpy_translator/core/review.py)
- [export.py](https://github.com/basil520/RenPyTranslator/blob/main/src/renpy_translator/core/export.py)

#### 实际实现

项目创建后会在游戏目录下维护 `.renpy-translator/`，其中包含：

- `workspace.json`；
- `source_root`；
- 翻译目录；
- `logs`；
- `state`；
- 项目索引和归档信息。

翻译管线具备：

- 批处理；
- token 保护 / 恢复；
- provider 返回校验；
- batch 失败后拆分重试；
- 按文件或时间写 checkpoint；
- `failures.jsonl`；
- 翻译进度统计。

审校模块将问题保存为独立状态，支持 `open`、`ignored`、`resolved`、`needs_recheck`，并允许对具体翻译条目进行更新。

#### 写回风险

它的导出流程会：

1. 生成导出预览；
2. 校验 source hash manifest；
3. 将旧语言目录复制到备份；
4. 删除目标目录；
5. 复制工作区内容到原游戏；
6. 处理额外的适配文件和危险备份。

这对普通用户来说非常直观，但 `rmtree` 后 `copytree` 的过程并不是声明式的原子 span / catalog 写回。它更适合作为“工作区导出体验”的参考，不适合作为本项目 `WritebackPlan` 消费器的替代方案。

### 4.4 translate-renpy：中间产物分层很清楚，但门禁较宽松

关键源码：

- [src/extract.py](https://github.com/teo-lin/translate-renpy/blob/main/src/extract.py)
- [src/merge.py](https://github.com/teo-lin/translate-renpy/blob/main/src/merge.py)
- [scripts/correct.py](https://github.com/teo-lin/translate-renpy/blob/main/scripts/correct.py)
- [scripts/translate.py](https://github.com/teo-lin/translate-renpy/blob/main/scripts/translate.py)

#### 实际实现

提取阶段把一个 Ren'Py 翻译文件拆成两类 YAML：

- `*.parsed.yaml`：供人工或模型编辑的清洁文本；
- `*.tags.yaml`：标签、位置、模板、角色变量和原始结构。

合并阶段根据 block ID 和 tag metadata 恢复标签、原文、译文和脚本模板，并检查：

- 引号；
- 花括号；
- 方括号；
- 变量集合；
- 角色变量。

订正阶段还将模式修复和 LLM 修复分开，并提供 `--dry-run`、`--patterns-only` 和 `--llm-only`。

#### 关键限制

`merge.py` 的实现即使发现 validation errors，也会继续写出 `.translated.rpy`，并提示用户之后复核。这种策略对于“生成一个供人工检查的副本”可以接受，但不符合本项目的：

```text
stale / source drift / structure block / invalid plan
    -> reject writeback
```

因此，本项目应该借鉴它的**数据分层**，而不是它的**错误后继续输出策略**。

### 4.5 2R-Tools：简单而实用的旧译文库，但不是版本化资产系统

关键源码：

- [translated_txt_lib.py](https://github.com/Phoenix525/2R-Tools/blob/main/app/controllers/translated_txt_lib.py)
- [renpy_translation.py](https://github.com/Phoenix525/2R-Tools/blob/main/app/controllers/renpy_translation.py)
- [renpy_update.py](https://github.com/Phoenix525/2R-Tools/blob/main/app/controllers/renpy_update.py)

#### 实际实现

项目将可复用译文存入 `TRANS_LIBS/TransLib.json`，并区分：

- 当前翻译工作区；
- 等待处理文本；
- 旧版本译文；
- RPG Maker 的默认译文库。

Ren'Py 更新流程主要以原文和 identifier 作为 key；部分路径会把源文本转成大写后查找缓存，并对空译文或 TODO 译文进行回填。

#### 工程判断

这对小型项目很有效，但存在明显限制：

- 同一句话在不同场景、角色和语言环境下可能发生冲突；
- key 没有源文件快照和内容 digest；
- 不能区分“模型生成”“人工确认”“最终写回”；
- 原地文件重写无法表达 source drift 和 plan stale。

本项目已有更强的 `TranslationRecord` 和 `ReuseCandidate`，不应退回到全局字符串字典。可以借鉴的只是“将历史译文单独沉淀为可复用资产”这一产品概念。

### 4.6 GameStringer：产品化最完整，但核心数据仍偏轻量

关键源码：

- [placeholder-guard.ts](https://github.com/rouges78/GameStringer/blob/main/lib/ai/placeholder-guard.ts)
- [quality-gates.ts](https://github.com/rouges78/GameStringer/blob/main/lib/quality/quality-gates.ts)
- [translation-validator.ts](https://github.com/rouges78/GameStringer/blob/main/lib/quality/translation-validator.ts)
- [translation-memory.ts](https://github.com/rouges78/GameStringer/blob/main/lib/translation-memory.ts)
- [patch-generator.ts](https://github.com/rouges78/GameStringer/blob/main/lib/patch-generator.ts)
- [chain-cost.ts](https://github.com/rouges78/GameStringer/blob/main/lib/translation/chain-cost.ts)

#### 实际实现

它的 Translation Memory 记录：

- source text；
- target text；
- source / target language；
- game ID；
- provider；
- confidence；
- verified；
- usage count 和时间戳。

查找策略是：

- 语言对内的 hash 精确查找；
- Levenshtein 相似度；
- 精确或高相似度结果优先；
- verified 结果优先排序；
- 70–94% 的模糊结果作为 prompt 上下文；
- 约 95% 以上的结果可能直接复用。

它的 placeholder guard 使用 multiset 比较缺失和多余 token，并提供自动修复：相同数量但 token 不同则位置替换，缺失 token 则追加。这个策略能提高玩家侧“最终能运行”的概率，但会降低译文 provenance 的可信度。

质量模块包含：

- 长度比例；
- glossary；
- placeholder；
- 末尾标点；
- 未翻译文本；
- 语气；
- 数字和单位；
- 综合质量分数。

成本模块把 provider 真实价格和 token 估算接到 chain preset 上；patch generator 能生成 Ren'Py 原生 TL 文件及多种通用格式。

#### 对本项目的启发

GameStringer 更适合提供：

- 翻译前 dry-run；
- provider / chain 成本预览；
- 进度、质量和 confidence 可视化；
- TM 建议卡片；
- patch / export 的用户流程。

但本项目的自动复用必须保持更严格的层次：

```text
exact + provenance + fresh snapshot -> 可进入直接复用候选
moved / context match             -> 需要人工确认
fuzzy similarity                  -> 只能作为 prompt 参考
source modified                   -> 只能作为 reference，不能自动写回
```

### 4.7 rpgmaker-translator：两阶段术语沉淀与事件边界上下文隔离

关键源码与机制：

- 两阶段翻译工作流（`Ctrl+D` Batch DB -> `Ctrl+T` Batch Dialogue）：先翻译游戏底层数据库（角色、道具、技能、敌人、属性），并将译文自动提取沉淀为项目级动态术语库（Project Glossary），在第二阶段翻译正文剧本时按需动态匹配注入；
- 事件分组翻译与上下文隔离（Event-grouped Translation）：每个 Worker 顺序翻译完整的单个事件（Event），在事件内部维护最近 10 轮 User/Assistant 历史，确保同场景角色语气、前后称谓与回调一致；在跨越事件边界时严格重置对话历史，从根源上阻断了跨场景、跨分支的上下文记忆污染；
- 角色代词与控制码映射（Pronoun & Actor Gender System）：主动扫描角色性别，结合对话 Header 识别说话人，并将 `\N[n]` 等控制码反解为具体角色名与性别标签传递给 LLM，有效降低日语/无主语句子中代词错乱的概率；
- 批量自动保存与断点续跑：每 25 条自动写入 Checkpoint，遇到模型服务宕机（30 秒内 5 次连接错误）自动暂停并等待重试。

#### 对本项目的启发

- **两阶段翻译流水线拓扑**：视觉小说通常包含系统菜单、角色设定、道具清单等元数据，先译元数据并反哺生成专用 Glossary，再翻译正文，能大幅提升术语一致性；
- **场景/文件边界上下文重置**：可与本项目的 ContextAssembler 及滑动窗口机制结合，在跨越文件或重大 Label 边界时强制截断历史窗口，避免上下文漂移；
- **代词消解**：角色感知与代词提示可纳入 TranslationPlan 上下文组装。

#### 不应照搬的部分

- 该项目强绑定 RPG Maker 事件树与数据结构；写回阶段虽然支持软链接（Directory Junctions）联动官方编辑器进行测试，但在 Ren'Py / Tyrano 上的写回缺乏声明式 AST/Span 级原子门禁。

### 4.8 DeepRenPyTrans：未翻译日志回流与字典垃圾修剪

关键源码与机制：

- 纯运行时注入（Runtime Injection via `config.replace_text`）：完全不触碰游戏原始 `.rpy` 脚本，也不生成原生 `tl/` 目录，而是通过注入 `hooks.rpy` 挂载 Ren'Py 原生文本渲染钩子，运行时动态查表替换；
- 漏译条目自动回流（`untranslated.log`）：游戏运行过程中，凡是未能命中字典的文本都会被自动追加到日志；后续运行 `extract --include-log` 即可将动态拼接的漏网文本反哺合并进待译池；
- 字典审计与垃圾修剪（Audit & Clean）：内置启发式规则，自动检测并修剪字典中的 Python 报错信息、内部 ID、hex 颜色、代码片段以及由于脚本更新导致的孤儿词条（Orphaned strings）。

#### 对本项目的启发

- **扫描与提取期的垃圾过滤（Junk/Code Filter）**：在 Adapter 扫描时借鉴其启发式规则，主动过滤非文本的调试断言、纯变量赋值与代码残留，提升待译条目纯净度；
- **漏译感知机制**：可作为非破坏性验证工具，辅助测试阶段验证翻译覆盖率。

#### 不应照搬的部分

- 纯运行时 Hook 是典型的“玩家向免解包外挂补丁”思路：当原句包含动态变量插值或特殊格式化时，字符串级别替换极易失效或导致渲染崩溃；完全丧失了工程级版本溯源、覆盖率审计与多语言发行能力。

### 4.9 RenForge：全要素资产管理与字体渲染防护

关键源码与机制：

- 全要素资产本地化面板：将本地化工作区分流为 Text、Images、Audio 三大模块，统一管理游戏脚本、汉化标题图/UI 贴图以及配音素材；
- 字体缺失与乱码防御（Font Fixing）：针对英/日原版游戏在渲染中文或特殊字符集时容易出现方块乱码（“▯▯▯▯”）的问题，提供中文字体自动注入与 `gui.text_font` 覆写能力；
- 运行时 Mod 打包：将所有翻译资源与字体封装为独立 Mod，便于普通玩家一键解压使用。

#### 对本项目的启发

- **字体与渲染防御（Font Fallback Defense）**：本地化不仅是文本翻译，字符集渲染是交付的最后一公里。可在 doctor 或 preflight 中增加目标语言字体有效性校验与回退提示；
- **多媒体资产扩展**：为后续多媒体本地化 Adapter 的资产边界规划提供参考。

#### 不应照搬的部分

- 本项目坚持以官方规范的 `translate` 块及原生 catalog 为唯一写回路径，不采用破坏脚本运行机制的非标准 Mod 封装。

### 4.10 varo：本地优先产品架构与透明引擎准备度

关键源码与机制：

- 现代解耦桌面架构：采用 Electron + Next.js + TypeScript + FastAPI，前端界面交互流畅，后端通过 Python 处理多引擎解析；
- 本地优先（Local-first）：内置离线 NLLB 模型作为零成本基线，同时支持按需配置商业 API；
- 透明的引擎成熟度矩阵（Engine Readiness Matrix）：在产品界面与文档中明确区分各引擎的支持等级（Stable vs Beta），例如 RPG Maker 标为 Stable，而 TyranoScript / Kirikiri 明确标为 Beta，不作虚假全能承诺；
- 快照安全与一键回滚（Rollback）：严格管理工作区副本与游戏原始状态，支持用户一键安全回退。

#### 对本项目的启发

- **引擎成熟度透明化**：在 CLI `doctor` 和 GUI 仪表盘中，明确标识当前 Engine Adapter 的成熟度（如 Ren'Py: Stable, Tyrano: Catalog-only Stable / KS-patch: Unsupported），建立用户信任；
- **备份与回滚的可视化**：将安全写回底层已有的事务保障抽象为用户可见的一键快照与回滚操作。

#### 不应照搬的部分

- 底层实现仍偏向通用的文本提取与整包写回，缺乏像本项目 `engine_adapters/writeback.py` 那样细粒度到字符 Span SHA-256 和语义 catalog 类型的双重防御。

## 5. 与当前项目源码的逐项对照

### 5.1 当前已经领先的部分

当前仓库相关核心文件：

- [translation_core.py](../../translation_core.py)
- [translation_plan.py](../../translation_plan.py)
- [engine_adapters/reuse.py](../../engine_adapters/reuse.py)
- [engine_adapters/writeback.py](../../engine_adapters/writeback.py)
- [engine_adapters/tyrano.py](../../engine_adapters/tyrano.py)
- [translation_quality.py](../../translation_quality.py)
- [Batch 工作流与安全检查](../batch_workflows.md)
- [Engine Adapter、覆盖审计与安全写回](../engine_adapter.md)

#### A. 复用数据模型已经不是简单 TM

`engine_adapters/reuse.py` 的 `TranslationRecord` 至少绑定：

- snapshot digest；
- source text；
- target language；
- provenance；
- status；
- record digest。

`ReuseCandidate` 还绑定 base / target version、reconciliation digest、base record digest 和候选状态。这比 2R-Tools 的全局字符串字典以及 GameStringer 的轻量 TM 更适合维护长期翻译资产。

#### B. 写回合同更严格

`engine_adapters/writeback.py` 会校验：

- plan schema；
- engine / adapter version；
- source snapshot fingerprint；
- 目标路径不能逃逸；
- 预期片段 SHA-256；
- JSON catalog path 和类型；
- duplicate / overlap 操作；
- replacement 内容形状。

这类检查在外部项目中通常只出现一部分，较少形成从 manifest 到 apply 的完整链路。

#### C. Tyrano 边界更清晰

`TyranoAdapter` 是 hybrid source/catalog adapter：

- 扫描 `.ks` 场景；
- 读取原生语言 JSON catalog；
- 对 catalog 中存在且通过 coverage 审计的条目生成 `json_catalog_set`；
- 直接修改 `.ks` 源脚本保持不支持。

这比“正则找到文本就直接写回 `.ks`”更适合长期维护，也更适合作为其他引擎 Adapter 的安全基线。

### 5.2 当前最明显的产品缺口

#### A. 结构保护尚未形成独立公共层

当前 prompt 会要求模型保留 Ren'Py 标签、变量、占位符和格式符，响应合同、quality rules 和写回检查也会发现一部分问题。但从当前核心路径看，尚未形成 RenLocalizer 那样独立的：

```text
extract -> protect -> provider -> restore -> exact validate
```

这意味着结构安全仍然分散在 prompt、response parser、quality check 和 adapter 中。建议建立一个共享模块，但不得改变 `check -> apply` 的最终权威性。

#### B. 普通翻译条目缺少长期审校视图

当前 GUI 已有：

- revision selection；
- quality findings dialog；
- final review campaign；
- preview / apply 分离。

但它们主要围绕“finding、revision 和任务”工作，还不是 Dialogue Visual Editor 那种长期保存的逐条语料编辑器。普通译文缺少一个统一视图来查看：

- source / target；
- speaker；
- 前后文；
- glossary 命中；
- control-code 差异；
- 翻译来源；
- 人工状态；
- 可直接生成的 revision proposal。

#### C. 现有成本、coverage、quality 能力还可以进一步产品化

CLI 已经有成本估算、usage ledger、doctor、coverage 和 quality report。下一步价值不在于再做一套估算算法，而在于把这些结果聚合为一个只读 preflight：

- 当前项目和引擎识别与成熟度状态（Stable vs Beta）；
- 可提取 / 不可提取 / 需注意条目；
- 待译量和上下文准备状态；
- 预计 token 和成本；
- 质量与结构风险；
- 预计需要人工审校的范围；
- 推荐的下一步命令或 GUI 操作。

#### D. 系统词条与剧情文本尚未形成两阶段拓扑

当前系统将提取出的所有 `TranslationUnit` 视为平级集合送入 TranslationPlan。然而视觉小说中，系统菜单、设定术语、道具/技能等词条如果与正文对话并发翻译，极易产生术语不一致。尚未借鉴 `rpgmaker-translator` 那样清晰的“两阶段依赖拓扑”：
- 阶段一：优先提取并翻译系统级/元数据条目；
- 阶段二：将阶段一的审核译文沉淀为项目动态术语表（Project Glossary），在正文剧情翻译时强制注入。

#### E. 缺乏扫描期代码垃圾过滤与目标语言渲染回退防御

在提取阶段，部分未遵循严格命名规范的脚本容易残留 Python 表达式、断言、内部 ID 等无效条目（类似 `DeepRenPyTrans` 识别的代码垃圾）；在交付阶段，虽然文本写回成功，但若原游戏缺省中文字体，玩家侧会遭遇方块乱码（类似 `RenForge` 针对的字体缺失痛点）。这两端属于“最前置的输入纯化”和“最后 100 米的渲染可读性防御”，当前尚未纳入自动化审计范围。

## 6. 可迁移设计与禁止照搬项

### 6.1 推荐吸收

#### 1. Provider-facing Protection

新增统一结构保护层，建议至少支持：

- Ren'Py `{tag}` / `{/tag}`；
- Ren'Py `[variable]`；
- `%s`、`%d` 和其他格式符；
- Tyrano / 通用控制码；
- glossary 术语 token；
- 原始换行和多段文本边界。

它只负责降低模型损坏概率和生成 finding，不拥有写回权限。

#### 2. Review Index

从现有 manifest / translation records / quality findings 派生一个可审校索引。每条记录建议包含：

```json
{
  "unit_id": "...",
  "source_text": "...",
  "target_text": "...",
  "speaker": "...",
  "context_digest": "...",
  "provenance": {},
  "quality_finding_ids": [],
  "review_state": "open"
}
```

它应当是可重新生成的派生产物，不替代 manifest、results 或 writeback plan。

#### 3. Provenance-aware Translation Memory

继续扩展现有 `TranslationRecord`，而不是引入一个平行的字符串字典。建议将以下内容用于排序和门禁：

- exact / moved / context match / fuzzy；
- base / target snapshot；
- source digest；
- provider / model / prompt schema；
- quality state；
- human confirmation；
- record digest。

模糊匹配只能提供参考上下文或人工候选，不能自动改变写回目标。

#### 4. Staged Artifacts

参考 translate-renpy 的分层思想，但保持本项目的版本化合同：

```text
source inventory
    -> protected request
    -> raw provider result
    -> normalized TranslationUnit result
    -> quality findings
    -> review / revision proposal
    -> validated WritebackPlan
    -> apply
```

每一层都应能通过 digest 或显式 parent identity 追溯到上一层。

#### 5. Preflight UX

把已有的 `doctor`、coverage、成本、usage、quality 和 split recommendation 组合成一个只读报告。报告可以同时导出 JSON 和 Markdown，便于 Agent、GUI 和人工决策共用。

#### 6. Two-stage Translation Pipeline

借鉴 `rpgmaker-translator` 的两阶段拓扑思想：
- 阶段一：优先识别并翻译游戏底层数据库、菜单系统条目、角色与道具属性；
- 阶段二：将阶段一的审核结果自动提取沉淀为项目动态术语库（Project Glossary），在第二阶段翻译正文剧情时作为强制约束注入 Prompt。

#### 7. Scene Boundary Context Isolation

借鉴 `rpgmaker-translator` 的事件边界重置策略：
- 在单个场景或事件内部维护短期多轮对话上下文（如 5–10 轮），保持说话人语气与人称称谓一致；
- 在遇到文件切换、宏观 Label 跳转或场景分界时，强制清空并重置历史对话窗口，阻断跨场景记忆污染与逻辑幻觉。

#### 8. Pre-filtering & Font Fallback Defense

- **扫描端纯化**：参考 `DeepRenPyTrans` 的 `audit & clean` 启发式规则，在扫描期自动跳过代码断言、内部系统 ID、十六进制颜色代码等非可读文本；
- **交付端防御**：参考 `RenForge` 解决汉化方块乱码（“▯▯▯▯”）的痛点，在 preflight / doctor 中增加目标语言字符集字体有效性检查，并提供字体覆写配置建议。

#### 9. Transparent Readiness & One-click Rollback

- 参考 `varo` 的设计，在 CLI 与 GUI 中公开各个 Engine Adapter 的成熟度等级（Stable vs Beta）；
- 将 `WritebackPlan` 底层保障转化为普通用户可见的“改动快照创建”与“一键安全回滚（Rollback）”交互。

### 6.2 不建议吸收

1. **模型返回后无条件自动追加缺失 token**：可能把结构错误隐藏起来。
2. **以大写或规范化后的原文作为唯一 key**：会制造角色、场景和版本冲突。
3. **发现校验错误后仍自动生成可导入产物**：应改成 preview-only 或 blocked。
4. **直接删除整个目标目录再复制工作区**：不适合作为声明式写回内核。
5. **用运行时 `replace_text` 或 hook 代替源 / catalog 写回**：纯黑盒补丁在面对动态变量插值时脆弱且不可维护，会掩盖 source drift 和 coverage 缺口。
6. **为了产品宣传一次性声称支持所有引擎**：每个引擎都必须锁定版本、原生制品、coverage 和排除项，并明确标定 Stable / Beta 状态。
7. **无界滑动窗口跨场景无脑携带上下文**：会导致严重的跨剧情记忆污染与角色指代漂移。
8. **忽视字符集字体与渲染直接写回**：只交付文本而不管目标语言字体缺失，会导致终态游戏展示为方块乱码。

## 7. 建议路线图

### P0：结构保护、代码纯化与审校索引

- 新增共享 `protect / restore / validate` 核心模块（支持长度降序匹配与带命名空间 Token）；
- 扫描期增加垃圾代码、断言与系统 ID 预过滤规则（Junk/Code Filter）；
- 将 missing / extra / reordered token 统一成稳定 reason code；
- 从现有 manifest 派生 review index；
- 在 GUI 中增加逐条查看和筛选，不直接绕过 revision preview；
- 为 Ren'Py 和 Tyrano 各增加结构损坏 fixture。

### P1：两阶段流水线、复用与 preflight 产品化

- 引入两阶段翻译流水线（系统元数据优先 -> 自动提取 Project Glossary -> 对话翻译）；
- 引入场景与文件边界的上下文清理与重置机制（防跨场景上下文污染）；
- 将 `ReuseCandidate` 变成 GUI 可读的候选列表，exact / moved / context / fuzzy 使用不同的默认动作；
- preflight 聚合引擎成熟度（Stable/Beta）、coverage、待译量、成本、质量和风险；
- 增加目标语言字体有效性与渲染回退检查（Font Fallback Check）；
- 将人工确认、忽略、待复查和已解决状态纳入 review provenance；
- 将备份 / 恢复 / 回滚路径完整暴露在 GUI。

### P2：上下文、多媒体与引擎扩展

- speaker profile 与 scene metadata 接入 RAG / Story Memory；
- 参考 Dialogue Visual Editor 增加角色名和控制码专用审计；
- 参考 RenForge 探索多媒体（图像/音频）本地化 Adapter，但不扩大当前文本写回合同；
- 在 Tyrano 和 Ren'Py 稳定后，基于透明的 Stable / Beta 矩阵评估第三个 Engine Adapter（如 RPG Maker 或 Godot）。

## 8. 最终判断

本项目不需要把自己变成“功能最多的游戏翻译器”。更有价值的定位是：

> 面向视觉小说源项目的、可复现、可审计、可恢复的 AI 翻译工作台。

外部项目最值得借鉴的是局部能力与工程巧思：

- **RenLocalizer**：模型调用前后的结构保护与术语长度降序替换；
- **Dialogue Visual Editor**：逐条语料编辑、审校状态机与控制码差异审计；
- **RenPyTranslator**：隔离工作区、断点续跑 checkpoint 与显式导出；
- **translate-renpy**：中间产物（纯文本 vs 代码结构）解耦分层；
- **2R-Tools**：历史译文沉淀的产品直觉；
- **GameStringer**：preflight、成本精算、质量打分和分级 TM 建议；
- **rpgmaker-translator**：两阶段术语反哺管线、事件边界上下文隔离与代词消解；
- **DeepRenPyTrans**：扫描期代码垃圾过滤与漏译条目回流感知；
- **RenForge**：全要素多媒体资产管理与字体渲染回退防御；
- **varo**：本地优先现代桌面架构与透明的引擎 readiness（Stable/Beta）状态机。

但最终写回语义仍应由本项目现有的 manifest、source snapshot、coverage、quality gate、EngineAdapter 和 `check -> apply` 合同决定。

## 9. 参考资料

### 外部项目源码

- [RenLocalizer](https://github.com/Lord0fTurk/RenLocalizer)：Ren'Py 翻译器、语法保护和运行时相关工具。
- [Dialogue Visual Editor](https://github.com/vadondaniel/dialogue-visual-editor)：RPG Maker MV/MZ 与 TyranoScript 桌面审校编辑器。
- [RenPyTranslator](https://github.com/basil520/RenPyTranslator)：Ren'Py 工作区、Ollama 翻译、审校和导出。
- [translate-renpy](https://github.com/teo-lin/translate-renpy)：分阶段 PowerShell / Python Ren'Py 工具链。
- [2R-Tools](https://github.com/Phoenix525/2R-Tools)：Ren'Py 与 RPG Maker 多 API 桌面工具。
- [GameStringer](https://github.com/rouges78/GameStringer)：多引擎 Web 产品、TM、QA、patch 和成本估算。
- [rpgmaker-translator](https://github.com/MoriTranslates/rpgmaker-translator)：多引擎本地 LLM 高保真翻译器，两阶段术语管线与事件边界上下文。
- [DeepRenPyTrans](https://github.com/Danko-Novak/DeepRenPyTrans)：Ren'Py 黑盒字典注入、未翻译日志回流与字典垃圾清洗。
- [RenForge](https://github.com/foulnike/RenForge)：Ren'Py 综合汉化 Mod 构建、多媒体资产面板与中文字体修补。
- [varo](https://github.com/rag91560-spec/varo)：本地优先泛游戏 AI 本地化平台，现代解耦桌面架构与引擎 readiness 状态机。

### 本仓库现行文档

- [Batch 工作流与安全检查](../batch_workflows.md)
- [Engine Adapter、覆盖审计与安全写回](../engine_adapter.md)
- [GUI 工作台](../gui_workbench.md)
- [上下文系统](../context_systems.md)
- [视觉小说引擎本地化能力矩阵与后续 Adapter 路线](visual_novel_localization_matrix.md)
