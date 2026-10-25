# RPG Maker MV 叙事模式：耐久 Sync

这是 #567 / #572 的 A3 产品路线，限定在[MV Adapter](engine_adapter.md#rpgmaker-mv-a2)声明的核心／插件窗口与叙事事件范围。支持地图和公共事件的 401 消息、102 选项及其 402 镜像。数据库、角色名修改、插件 UI、动态脚本、滚动文本及图片等范围仍由 B0 核查；MZ 不在此路线。

工程回归使用原创固定响应材料。A0 原生补证、A4 的真实 101 + 7 试点、人类／Agent 独立语义审阅、字体、显示、分支与存读档分别验收，不能从本页的产品接线推断“正式支持已验收”或“游戏已汉化完成”。

## 配置源、目标和支持窗口

原始源与目标必须是不同且不互相嵌套的内容目录，均直接包含 `data/` 和 `js/`。发布包有 `www/` 时选择各自的 `www/`。先自行建立可信原始材料和受控目标副本；工具不会执行 Ren'Py prepare 来生成 MV 副本。

在现有 `translator_config.json` 中设置：

```json
{
  "engine": "rpgmaker_mv",
  "game_root": "C:/Games/ChineseCopy/www",
  "rpgmaker_mv": {
    "source_root": "C:/Games/Original/www",
    "support_window_file": "C:/TranslationProject/approved-window.json",
    "occurrence_scope_file": ""
  },
  "prepare": {"enabled": false},
  "include_files": ["data/Map001.json", "data/CommonEvents.json"],
  "sync": {
    "rag_enabled": false,
    "source_index": {"enabled": false},
    "story_memory": {"enabled": false},
    "project_analysis": {"inject_published_brief": false}
  }
}
```

这是配置片段，保留已配置的 Provider、ModelProfile、Sync 超时、分块、术语和风格设置。模型凭据继续使用现行安全存储／环境变量。缺少 `engine` 时保持 Ren'Py 行为；未知、空或非法值会拒绝执行。MV 的 `game_root` 就是受控目标内容根，`tl_subdir` 不参与 MV 路径解析；相对源、目标及窗口路径均相对于配置文件。

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

只列主要 401 / 102 位置，402 镜像由同一 Adapter 自动关联。重复、空清单、布尔下标、
未知／镜像位置、源文件或字串摘要不符及与文件筛选冲突均拒绝执行。清单字节摘要绑定
耐久计划；改动清单会使派发、检查和写回失效，GUI 旧检查结果也会失效。
该清单仅缩小请求，全部源／目标和核心／插件仍参加覆盖审计及漂移校验；样本外的
未知控制码或结构阻断仍会阻止执行，不能用清单隐藏问题。

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

记录保存在目标内容根的 `translation_context/coverage_review.json`。缺失、pending、非法、过期、policy 不满足或存在未解决 findings 均阻止生成执行计划。叙事范围内审阅通过不会消除报告中其他 unsupported 范围，也不会使全游戏 coverage 变成 confirmed。doctor 的 `start_pending_sync` / `mv_scope_no_pending` 只描述当前已审阅叙事范围；材料不可读时计数为未知。

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

1. 在「项目列表」选择独立目标内容目录；在「设置 → 项目」填写 `rpgmaker_mv`、原始源目录和支持窗口文件，保存。
2. 在「项目与环境」使用「导出覆盖材料」，完成独立审阅后用「导入覆盖核对」导入，再运行环境检查。
3. 进入「同步翻译」，选择可用模型并启动；运行、继续、检查和应用均调用上述 CLI 合同。

项目／配置／材料／模型／任务代次变化会使旧环境检查或预检回调失效。GUI 不会重新实现 MV 翻译语义。Batch、外部 Agent 工作包、旧 `gemini_translate.py` 同步入口、关键词、修订和 Ren'Py prepare 当前不支持 MV，会在准备／付费派发前拒绝；不支持的 GUI 模式不能启动。
