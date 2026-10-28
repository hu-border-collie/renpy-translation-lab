# RPG Maker MV：耐久 Sync 与预处理

这是 #567 / #572 的耐久 Sync 产品路线，限定在[MV Adapter](engine_adapter.md#rpgmaker-mv-a2)声明的核心／插件窗口。默认支持地图和公共事件的 401 消息、102 选项及其 402 镜像；可显式启用下述系统术语与地图显示名。数据库、角色名修改、MapInfos 名称回退、插件 UI、动态脚本、滚动文本及图片仍留在后续范围；MZ 不在此路线。

工程回归使用原创固定响应材料。A0 原生补证、A4 的真实 101 + 7 试点、人类／Agent 独立语义审阅、字体、显示、分支与存读档分别验收，不能从本页的产品接线推断“正式支持已验收”或“游戏已汉化完成”。

## 配置源、目标和支持窗口

原始源与目标必须是不同且不互相嵌套的内容目录，均直接包含 `data/` 和 `js/`。发布包有 `www/` 时工具会解析到其内容根；GUI 也可选择标准 `original/work` 项目根，自动定位目标 `work/www`。
先建立可信原始材料和受控目标副本；「准备 MV 项目」引导选择原始源与已核对的兼容性清单，保存后运行同一 doctor。它不执行 Ren'Py prepare，也不批准支持窗口或覆盖审阅。

在现有 `translator_config.json` 中设置：

```json
{
  "engine": "rpgmaker_mv",
  "game_root": "C:/Games/ChineseCopy/www",
  "rpgmaker_mv": {
    "source_root": "C:/Games/Original/www",
    "support_window_file": "C:/TranslationProject/approved-window.json",
    "occurrence_scope_file": "",
    "display_fields": [],
    "reviewed_history_file": ""
  },
  "prepare": {"enabled": false},
  "include_files": ["data/Map001.json", "data/CommonEvents.json"],
  "sync": {
    "rag": {"enabled": false},
    "source_index": {"enabled": false},
    "story_memory": {"enabled": false},
    "project_analysis": {"inject_published_brief": false}
  }
}
```

这是配置片段，保留已配置的 Provider、ModelProfile、Sync 超时、分块、术语和风格设置。模型凭据继续使用现行安全存储／环境变量。缺少 `engine` 时可由配置的 `game_root` 自动识别 MV；无 MV 标记时保持 Ren'Py 行为。显式未知、空或非法值会拒绝执行。CLI doctor 解析不改写文件，GUI 选择目录时保存识别结果。MV 的 `game_root` 解析为受控目标内容根，`tl_subdir` 不参与 MV 路径解析；相对源、目标及窗口路径均相对于配置文件。原外壳目录下显式配置的相对 glossary／Macro 路径会保留原锚点。

Sync 风格文件可位于目标内容目录，或可识别为同一目标的游戏外壳／标准 `original/work` 项目根。自动定位 `www/` 后仍会加载这些既有文件；其他项目或无关目录中的文件继续拒绝加载，符号链接也按实际路径检查。

支持窗口文件必须由可信调查明确批准，工具不会把现场摘要自动当成兼容性批准。格式为：

- `schema_version`：整数 `1`。
- `material_sha256`：相对路径到真实 64 位十六进制 SHA-256 的对象，必须包含 `js/rpg_core.js`、`js/rpg_managers.js`、`js/rpg_objects.js`、`js/rpg_scenes.js`、`js/rpg_sprites.js`、`js/rpg_windows.js`、`js/plugins.js`，以及注册表中每个插件的 `js/plugins/<name>.js`，包括已禁用插件。
- `message_styles`：布尔值；只有在该窗口明确批准并启用 GALV_MessageStyles 时设为 `true`，仅开放 Adapter 已实现的数值 popup 语法。

未知核心／插件字节、缺失文件、MV/MZ 混用和结构变化会阻断。`include_files` / `include_prefixes` 只筛选本轮请求，使用内容根内的相对路径；覆盖审计和漂移检查仍扫描完整材料。目标中已有非空、结构通过且镜像一致的不同文本视为现译；空白、结构失败或选项镜像不一致仍进入待处理，不能据现译计数推断语义质量。

### 精确位置范围（可选）

小样本只包含地图中的部分事件页或对白时，可在 CLI 配置或 GUI「设置 → 项目」的
「MV 精确位置范围文件」填写 `rpgmaker_mv.occurrence_scope_file`。路径相对于配置文件；
留空继续沿用文件筛选。独立清单必须来自原始源核对，格式如下（SHA-256 须换成实际值）：

```json
{
  "schema_version": 1,
  "source_files": {"data/Map001.json": "完整原始文件的64位SHA-256"},
  "occurrences": [
    {"file_rel_path": "data/Map001.json", "json_path": ["events", 1, "pages", 0, "list", 1, "parameters", 0], "source_sha256": "原始字符串UTF-8字节的64位SHA-256"}
  ]
}
```

可列主要 401 / 102 及已显式启用的显示字段位置，402 镜像由同一 Adapter 自动关联。重复、空清单、布尔下标、
未知／镜像位置、源文件或字串摘要不符及与文件筛选冲突均拒绝执行。清单字节摘要绑定
耐久计划；改动清单会使派发、检查和写回失效，GUI 旧检查结果也会失效。
该清单仅缩小请求，全部源／目标和核心／插件仍参加覆盖审计及漂移校验；样本外的
未知控制码或结构阻断仍会阻止执行，不能用清单隐藏问题。

### 系统术语与地图显示名（显式启用）

`rpgmaker_mv.display_fields` 默认为 `[]`，保持原叙事范围。可填写
`["system_terms", "map_display_name"]`，也可只启用其中一项；未知值、重复值和错误类型会拒绝。
CLI 读取该配置；GUI「设置 → 项目 → MV 附加显示字段」逐行填写相同标识。
两个入口使用同一字段规则，不按游戏名称、原文或数量推断范围。

| 配置值 | 允许的既有字符串叶子 | 保留边界 |
| --- | --- | --- |
| `system_terms` | `data/System.json` 的 `terms.basic[0..9]`、`terms.params[0..7]`、`terms.commands[0..19,21,22,24,25]`，以及 `terms.messages` 的固定 core 消费键 | 20 / 23 等保留或扩展索引、未知 message 键、其他 terms 成员、游戏标题、货币和类型名不开放 |
| `map_display_name` | `data/Map数字.json` 顶层 `displayName` | 不修改地图事件名、note、资源引用或 `MapInfos.json`；空显示名不填补，也不自动翻译编辑器名称回退 |

固定 messages 键为：`alwaysDash commandRemember bgmVolume bgsVolume meVolume seVolume possession
expTotal expNext saveMessage loadMessage file partyName emerge preemptive surprise escapeStart escapeFailure
victory defeat obtainExp obtainGold obtainItem levelUp obtainSkill useItem criticalToEnemy criticalToActor
actorDamage actorRecovery actorGain actorLoss actorDrain actorNoDamage actorNoHit enemyDamage enemyRecovery
enemyGain enemyLoss enemyDrain enemyNoDamage enemyNoHit evasion magicEvasion magicReflection counterAttack
substitute buffAdd debuffAdd buffRemove actionFailure`。
权威规则在[字段模块](../engine_adapters/rpgmaker_mv_fields.py)，对应
[MV TextManager](https://github.com/rpgtkoolmv/corescript/blob/master/js/rpg_managers/TextManager.js) 的消费路径；
这不是任意 core／插件版本兼容的证明，仍需原支持窗口。

启用系统术语要求 `System.json` 和 `terms` 对象存在。缺失的可选成员或 null 叶子保留，
已存在的所选容器／叶子类型不符会阻断。空字符串保持原样，不创建键或数组成员。
新增字段仍参加完整 inventory 和独立 raw-leaf audit，未开放的 System 内容仍显示为范围外。

若只翻译特定存读档词或地点名，继续使用 schema 1 occurrence 范围文件。例如位置可以是
`["terms", "messages", "saveMessage"]`、`["terms", "commands", 9]` 或 `["displayName"]`，
绑定各自完整源文件与原文字串摘要。`include_files` 可选择已启用的 `data/System.json`；
它与精确位置文件必须一致。没有精确位置文件时，启用组内全部非空字段都可能进入待译，
不要把它当作“只选择当前可见 UI”的自动判断。

显示字段的 `%1`、`%2` 等参数按
[MV format 规则](https://github.com/rpgtkoolmv/corescript/blob/master/js/rpg_core/JsExtensions.js)
保护：编号原拼写与次数不变，可随中文语序移动；缺失、重复或新增参数阻断。
已识别 core 控制码的大小写、参数与相对顺序不变，未知转义、物理换行和 popup 均拒绝。
模型只见绑定请求／词条／字段角色的保护标记；恢复后 Adapter 再检查一次。

Adapter 0.3.0 / locator 2 将字段策略、启用组、角色、源／目标、支持窗口、配置、范围及结果身份
纳入现有绑定。0.2 的任务和预览可留作查询，但旧检查或审阅不能在新版本下继续授权写回；
升级或修改字段选择后须重新导出覆盖材料、独立审阅并建新计划，不能编辑旧制品解除 stale。
先前翻译的显示字段须继续启用，关闭后它们与源的差异会被识别为范围外目标变化。

## 独立覆盖审阅

支持窗口审核与文本范围审核是两个独立输入。先导出覆盖材料：

```powershell
python gemini_translate_batch.py coverage-status --export-dir C:/TranslationProject/coverage
```

核对 `coverage_candidates.jsonl`、`coverage_report.json` 和 `coverage_review.md`，按 `coverage_review_template.json` 完成独立审阅。记录真实 reviewer 类型、身份、Agent 工具／模型（适用时）、时间及 findings，遵守模板 policy；不要把待填写模板改名当作批准。导入完成的记录：

```powershell
python gemini_translate_batch.py coverage-review-import --file C:/TranslationProject/review.json
python gemini_translate_batch.py doctor
python gemini_translate_batch.py translate-preflight --strategy sync
```

记录保存在目标内容根的 `translation_context/coverage_review.json`。缺失、pending、非法、过期、policy 不满足或存在未解决 findings 均阻止生成执行计划。缺失记录时 GUI 显示待覆盖核对并提供导出／导入入口，保留真实计数；底层读取错误留在完整详情。当前文本范围内审阅通过不会消除报告中其他 unsupported 范围，也不会使全游戏 coverage 变成 confirmed；GUI 仍在摘要中单列未支持候选，不计入待译数。doctor 的 `start_pending_sync` / `mv_scope_no_pending` 只描述当前已审阅文本范围；材料不可读时计数为未知。

## 同步关键词、原文索引与已审阅历史

三个预处理入口复用现有关键词及上下文核心。它们使用当前已审阅范围中的真实源 occurrence，
包含显式启用的显示字段；同文异处保留不同 occurrence ID 和类型化 JSON 路径，不伪造 `.rpy` 行号。
原始源、支持窗口、字段选择、范围或覆盖审阅变化后，先重新审计／审阅，再重建预处理。

```powershell
python gemini_translate_batch.py sync-keywords --limit 3 --output json
python gemini_translate_batch.py bootstrap-source-index --skip-prepare --output json
python gemini_translate_batch.py bootstrap-rag --skip-prepare --output json
```

`sync-keywords` 使用冻结的 Sync keyword 阶段路由、Sync Macro 与本地词表，复用既有 prompt、
schema、候选去重、证据校验和四份 JSONL / Markdown 报告。候选的 `source_locations`、
概要的 `source_locations` / `evidence_locations` 指向实际文件与 JSON 路径。它只生成报告，
`check/apply` 拒绝关键词 manifest；要改词表，仍须显式使用
[`merge-keywords-to-glossary`](batch_workflows.md#关键词提取流程) 的审核与备份流程。
MV 关键词不运行 Ren'Py prepare，也不开放 `build-keywords` Batch 路线。

`bootstrap-source-index` 每个选中源位置建立一条 source-only 记录，绝不把目标中文存入原文索引。
`bootstrap-rag` 只读显式提供的已审阅双语 JSONL：可在 `rpgmaker_mv.reviewed_history_file`
填写路径（相对配置文件），或用 `--seed-jsonl <FILE>` 补充输入（相对当前目录）。
历史库还绑定解析后的配置语料路径，包括未选择语料的状态。切换或清空
`reviewed_history_file` 后，旧库会标为 stale，重新 bootstrap 前不再注入旧译文；
原文索引不因这项选择变化失效。显式 seed 输入继续独立按路径和摘要核验，
未配置语料时仍可用 `--seed-jsonl` 建库。缺少这项绑定的旧历史库须显式重新 bootstrap。
没有输入、文件缺失、仅有 pending/rejected 记录时返回 `prepared=false`，不会把目标中的现译、
UI／图片草稿、结构检查通过或 apply 成功推断为语义批准。apply 不会自动将 MV 现译加入历史库。

历史输入每行是独立 JSON 对象，下例仅示意待审阅材料，不能直接当作已批准输入：

```json
{"schema_version":1,"engine":"rpgmaker_mv","adapter_version":"0.3.0","support_window_digest":"<当前支持窗口摘要>","file_rel_path":"data/Map001.json","json_path":["events",1,"pages",0,"list",1,"parameters",0],"source":"Keep walking.","source_file_sha256":"<完整源文件SHA-256>","translation":"继续前行。","review":{"status":"pending","reviewed_at":"","reviewer":{"type":"human","id":""}}}
```

真正纳入历史库时，`review.status` 必须为 `approved`，记录实际审阅者的 `type=human|agent`、
非空 `id` 及带时区的 ISO `reviewed_at`；保留其他实际审阅信息也可。每条绑定当前 adapter、
支持窗口、完整源文件摘要、精确源文本与类型化路径，译文还须通过相同结构保护。
已批准但过期、越界、重复或结构不符的记录会拒绝 bootstrap；未批准记录只计数，不采用。
空输入文件可以明确生成 0 条的已准备库，与未配置、缺失和未准备分开报告。

两种库沿用 `JsonSourceIndexStore` / `JsonRagStore` 的文件格式、锁和原子替换，目录独立且不能
互相嵌套，也不能放入冻结原始源或目标 `data/`、`js/`。路径遵循 `context_storage`；
可用 `sync.source_index.store_dir` 与 `sync.rag.store_dir` 显式覆盖（相对目标内容根）。
只有 bootstrap 会建立或重建向量。源记录带 occurrence 身份、源摘要和路径；历史记录另带
独立 review 与输入摘要。默认 prune；`--no-prune` 留下的旧范围记录不会进入新范围检索。

启用 `sync.source_index.enabled` / `sync.rag.enabled` 后再建立相应库；原文索引另遵循现有
项目级同步索引开关，历史使用既有的「启用同步 RAG」开关。
Embedding 使用 **Sync 的 embedding 绑定**；v1 配置由 Sync 所用 ModelProfile 引用 embedding
profile，旧配置读取 `sync.rag` 的 embedding 字段。来源、模型、endpoint、维度与 document/query
task 身份必须兼容；修改身份需要显式重新 bootstrap，不会静默混用旧向量。
准备和非空库检索会调用 embedding Provider，`sync-keywords` 会调用生成 Provider。

耐久 Sync 通过相同 ContextAssembler 消费已准备的库，保留既有 top-k、相似度、单条字符与
Source Index 独立总预算，并把命中、诊断及最终 prompt 冻结进请求；恢复不重新检索旧请求。
Macro 和词法 glossary 不依赖 embedding。禁用、缺失、过期、身份不兼容、失败、
真正的 0 条库及检索零命中分别报告；可选库不可用时降级为空上下文，不把它们显示为 ready。
`doctor` 和 `translate-preflight` 只读准备状态，不发翻译或 embedding 请求。
bootstrap 的 JSON `status` 与 `result.prepared` 描述实际结果；加 `--strict-exit-codes` 时，
ready 为 0，disabled／missing／stale／incompatible 为 3（需处理），failed 为 4（阻断）。
StoryMemory 与 Project Analysis 注入仍未接入 MV，开启会阻断，不能按 Ren'Py 扫描 JSON。

预处理输入也进入新任务材料身份；升级前的旧任务仍可查看，但缺少该身份的旧检查／计划
不能在新版下继续授权写回，应重建计划。该工程能力不表示任何真实游戏已经完成预处理或验收。

## 运行、恢复和写回

使用现有[耐久 Sync](sync_workflow.md)流程：

```powershell
python gemini_translate_batch.py sync-start
python gemini_translate_batch.py sync-status --latest
python gemini_translate_batch.py sync-resume <RUN>
python gemini_translate_batch.py check <RUN>
python gemini_translate_batch.py apply <RUN>
```

`sync-start` 会实际调用配置的 Provider，应在范围、模型和费用安排明确后执行。`doctor`、`translate-preflight` 和覆盖导出／导入不会调用翻译或 embedding Provider。任务 ID、剩余项、取消、派生、unknown Provider 结果与费用事实沿用既有耐久合同；unknown 不会因 resume 自动重发。

原文始终来自独立源，目标中文不会成为下一轮英文原文。每个 occurrence 独立，重复文本不合并；多行消息保持各字符串叶子的物理位置，一个选项同步修改 102 与对应 402。结构保护先隐藏控制 token，恢复后再由同一 Adapter 校验。

运行前和每次派发前重新校验材料。check 绑定完整源／目标、支持窗口、配置、审阅、结果与质量策略，只有匹配最近一次 `writeback_gate.decision=allow` 才能生成可用预览。apply 消费 schema 2 双快照，复用原子事务、当前目标 preimage 校验及失败／崩溃恢复；紧邻提交再检查现场，变化会拒绝或回滚。重复 apply 只容许与该次预览完全一致的已应用字节。`--force` 不能绕过这些条件。

源、事件位置、核心、插件、配置、审阅或目标发生合法更新后，重新审计／审阅并建立新计划；不要编辑旧 artifacts 来解除 stale。运行制品包含私有源与目标快照，应留在本地。

## GUI

1. 在「项目列表」选择独立目标内容目录；在「设置 → 项目」填写 `rpgmaker_mv`、原始源目录和支持窗口文件。需要时填写「MV 附加显示字段」与精确位置文件，再保存。
2. 在「项目与环境」使用「导出覆盖材料」，完成独立审阅后用「导入覆盖核对」导入，再运行环境检查。
3. 需要历史时，在「设置 → 项目 → MV 已审阅双语历史文件」选择上述独立审阅 JSONL。
   在「上下文库」启用所需层，使用「初始化记忆库」「初始化原文索引」调用同一 bootstrap，
   按实际状态确认是否准备完成；「关键词 / 术语 → 同步」生成候选与概要报告。
4. 进入「同步翻译」，选择可用模型并启动；运行、继续、检查和应用均调用上述 CLI 合同。

项目／配置／材料／模型／库制品／任务代次变化会使相应旧状态回调失效。GUI 不会重新实现 MV 翻译语义。Batch、外部 Agent 工作包、旧 `gemini_translate.py` 同步入口、修订和 Ren'Py prepare 当前不支持 MV，会在准备／付费派发前拒绝；不支持的 GUI 模式不能启动。
