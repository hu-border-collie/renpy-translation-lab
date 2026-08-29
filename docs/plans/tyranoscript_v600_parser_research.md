# TyranoScript V600+ P5 parser 调研与 fixture 基线

> **状态**：#265 P5 预研资料，用于冻结 parser 行为与离线 golden fixture。
> 本文不引入生产行为；实现边界以 `engine_adapters/contracts.py` 和 #265 正文为准。
> 基线：`main@114b376`。调研日期：2026-08-29。

## 1. 结论摘要

1. TyranoScript V600+ 的官方翻译运行时代价是 `data/others/lang/<lang>.json`；
   `[lang_set name="<lang>"]` 加载该 JSON 后用同一 parser 重新解析当前 scenario 并替换
   text / `chara_ptext` / 已注册 tag 参数。**官方编辑器里的 CSV 只是编辑交换格式，
   运行时只读 JSON。**
2. 官方解析器是宽松的：未闭合引号、多个无引号参数、未闭合行内 tag 经常被静默
   “补偿”成可执行节点。P5 adapter 不能把官方解析结果当作正确性证明，必须引入
   自己的候选审计和 parse_error 分类。
3. catalog key 是**官方 parser 归一化后的值**，不是源码里的原始字面量。例如
   `KeepSpaceInParameterValue=2` 时，`text="Start Game"` 的 catalog key 是
   `Start Game`；带引号的 `]` 和未闭合 quote 会触发官方 parser 的补偿路径。
4. Studio V603 的翻译 UI 只提取：text 节点（`[iscript]` 内除外）、
   `chara_ptext` 的 `name`、`map_tags` 中注册的 tag 参数；未注册 tag、宏调用和
   动态 `&sf.*` 表达式不会成为翻译行，但也不报错——这正是 P5 必须补上的
   unknown / unsupported 来源。
5. 官方 JSON 除运行时消费的 `scenes` / `charas` 外，Studio 还会写入 `systems`
   （`tyrano/lang.js` 的系统文案）和 `tags`（tag 注册表本身）。运行时
   `convertLang` 只消费 `scenes` / `charas`；`systems` 的运行时消费点尚未在
   V602c runtime 中找到，写回前需单独验证。

## 2. 调研材料

| 材料 | 版本 / 位置 | 用途 |
|------|--------------|------|
| 官方 Localization 页面 | <https://tyranoscript.com/usage/advance/translate> | V600+ 功能范围、默认 glink/ptext、CSV 介绍 |
| 官方 runtime 开源仓库 | `ShikemokuMK/tyranoscript`，branch `tyrano6` @ `6f45d046029943a756f8bd10004b2a1e8089fa25` | `kag.parser.js`、`kag.js`、`kag.tag_system.js` 可读实现 |
| 官方 runtime 包 | `tyranoscript_v602c_en.zip`（SHA256 `7f07ffe04dd40eb88d3660f54dd4efd892889746ad5a4c77e1fc274d0c9b1803`） | `expected/parser_nodes.json` 的生成来源 |
| TyranoStudio V603 en（Windows） | `TyranoStudio_win_std_v603_en.zip` 内 `resources/app/src/js/view/studio/Translate.js` | 官方编辑器提取、保存、CSV 行为 |

调研时对 Studio `Translate.js` 做了字符串反混淆核对；仓库不复制第三方代码，
只冻结行为结论。

## 3. 官方项目布局

- 源脚本：`data/scenario/**/*`；Studio 递归读取 `data/scenario/` 下所有文件。
  scenario 的 catalog key 是相对于 `data/scenario/` 的路径并保留 `.ks`，
  例如 `scene1.ks` 或 `sub/extra.ks`。
- 原生 catalog：`data/others/lang/<lang>.json`；`<lang>` 是 `[lang_set name="..."]`
  中的语言代码。
- 解析配置：`data/system/Config.tjs` 的 `KeepSpaceInParameterValue`（V600 模板
  默认 `2`）会改变带空格参数值和 catalog key，必须纳入 source fingerprint。
- `[lang_set]` 实现：`kag.tag_system.js` 的 `tag.lang_set`（约 L4279），
  调用 `kag.loadLang()`；`loadLang()`（`kag.js` L3720）加载
  `./data/others/lang/<name>.json`，清空 scenario 缓存并重新 `loadScenario`
  当前文件。因此同一 build 内可多次切换语言。

### lang_set / catalog 缺失语义（重要）

`loadLang` 在 JSON 不存在或解析失败时只会 `console.log` 错误，把 `kag.lang`
置空、`map_lang` 清空，然后继续运行；`convertLang` 遇到不存在的
`map_lang["scenes"][scenario]` 也会直接返回原文。也就是说：

- 官方 runtime **静默回退原文**，不会阻止打包或运行；
- P5 的 `check` / 发布门禁不能依赖 runtime 报错，必须自己检查 catalog 存在性、
  `lang_set` 静态目标语言、scenario section 和 row 完整性。

## 4. 官方 parser 行为

来源：`kag.parser.js` `parseScenario`（L64）与 `makeTag`（L309）。

### 4.1 行级语法

- 每行先 `trim`；空行跳过。
- 整行 `;` 开头是注释；只有独占一行的 `/*` 与 `*/` 才是块注释边界，
  `/* ... */` 与行尾注释不是注释。
- `#name` 或 `#name:face` 生成 `chara_ptext` 节点，`name` 是 `#` 后原始文本；
  名字解析在 `chara_ptext` tag 执行时才做 `jcharas` 反查。
- `*label|display` 生成 `label` 节点；`label_obj` 没有顶层 `line`，
  行号只在 `pm.line`。本仓库 fixture 生成 `parser_nodes.json` 时已归一化。
- `@tag params` 把整行余下部分交给 `makeTag`，等价于行内 `[tag params]`。
- 其他行按字符扫描，遇到 `[` 进入 inline-tag 扫描；扫描到的 text 片段被拆成
  独立 `text` 节点，每个片段有 `pm.val` 和 `val`。

### 4.2 行内 tag 扫描

- 扫描支持嵌套 `[` / `]` 深度，但只有最外层 `]` 闭合 tag；`exp` 属性里的数组
  下标靠深度计数存活。
- 引号 `"` / `'` / `` ` `` 成对出现期间，`]` 按字面量保留在参数值内。
- 行尾若仍在 tag 扫描中：
  - 若 `tag_str` 以 `]` 结尾，官方 parser 认为少了一个引号，**截掉最后的 `]`**
    并产生 `compensate_missing_quart` warning，继续 `makeTag`；
  - 若不以 `]` 结尾，也直接 `makeTag`，**不报 unclosed tag**。
- 反斜杠转义只保留一个字符；文本节点遇到 `\[` 时不会进入 tag 扫描。

### 4.3 makeTag 参数语义

- tag 名后是 `key=value` 序列；`value` 可用三种引号、也可无引号（无引号时以
  空白结束，后面的 token 会成为下一个空值参数）。
- 参数值按 `Config.tjs` 的 `KeepSpaceInParameterValue` 处理：
  - `1`：删除值内全部半角空格；
  - `2`：`trim` 两端空格、保留内部空格（V600 模板默认）；
  - `3`：完全保留。
- 字面量 `undefined` 会归一化为空字符串。
- 未闭合引号不产生 parser error；`makeTag` 在串尾把已读内容 `trim` 后写入参数。
- `iscript` / `endscript` 会切换 `flag_script`；`flag_script` 是 parser 实例级
  状态，官方 `parseScenario` 不在文件开头重置它。fixture 生成时逐文件重置，
  未来 adapter 也必须逐文件隔离。

### 4.4 if / macro / iscript

- `if` / `elsif` / `else` / `endif` 由 `makeTag` 维护 `deep_if`，不匹配时
  `parseScenario` 报 `if_and_endif_do_not_match`。
- `[iscript]` 内所有字符（包括其中的 JS 字符串）都作为 text 节点进入
  `array_s`，`convertLang` 用 `is_script` 标志跳过替换；官方 Studio 也不显示
  这些行。P5 inventory 应把它们标为 `explicitly_excluded`。
- `[macro]` 定义体内的 text 不会被特殊排除；官方 Studio 会像普通 text 一样
  显示。P5 首版若要保守处理 macro 定义体，应在 adapter 层标记
  `structure_kind` 并降级为 attention / unsupported，而不是静默跳过。

## 5. 原生 catalog 运行时合同

`kag.js` `convertLang`（L3644）消费的结构：

```json
{
  "scenes": {
    "<scenario_rel_path>": {
      "scenario": { "<parser_text_value>": "<translation>" },
      "tag": {
        "<tag_name>": {
          "<param_name>": { "<parser_param_value>": "<translation>" }
        }
      }
    }
  },
  "charas": { "<chara_ptext_pm_name>": "<display_translation>" },
  "systems": { "<lang.js word key>": "<translation>" },
  "tags": { "<tag_name>": ["<param_name>"] }
}
```

替换语义：

- `scenario`：对 `[iscript]` 外的每个 `text` 节点，用 `pm.val` 作 key 精确
  查找；命中且译文为 truthy 时替换。**空字符串译文不会应用。**
- `tag`：只对 `tags` 注册表中存在的 `tag_name` / `param_name` 查找；源码
  `pm[param]` 为 key，值必须 truthy 才替换。
- `charas`：`chara_ptext` 的 `pm.name` 为 key。若存在 `[chara_new]` 定义，
  翻译写入 `stat.charas[name].jname`；否则直接替换 `chara_ptext` 的 name。
  因此官方字符翻译的源 key 是 **`#` 后的原文**，不是 `[chara_new]` 的 `jname`。
- `systems` / `tags`：runtime `convertLang` 不消费；它们是 Studio 翻译工程数据。
- 不存在 scenario section、tag / param section、row 时均静默保留原文。
- 同一 scenario 内重复的相同文本/参数值共享一个 catalog row；不同 occurrence
  仍应由 locator（文件 + 行 + node_index）区分。

## 6. TyranoStudio V603 官方翻译工作流（反混淆核实）

`Translate.js` 的方法行为：

- `loadTyranoProject()`：递归遍历 `data/scenario/`，跳过目录，用 parser 同步
  解析每个文件；`map_scene` 的 key 是去掉 `data/scenario/` 前缀后的相对路径。
- `initSystemLang()`：扫描所有已解析 `chara_ptext` 节点，把 `pm.name` 加入
  `map_charas`；再从 `tyrano/lang.js` 的 `word` map 读入 `map_systems`。
- `updateLangTable()`：对选中 scenario 的 `array_s` 逐节点处理：
  - `iscript` / `endscript` 翻转 `is_script`；
  - `text` 且不在 `iscript` 内 -> type `text` 行；
  - 其他 tag 只处理 `map_tags[tag.name]` 中注册的参数，且参数值 truthy；
  - `&sf.*` 表达式被当作普通字符串显示出来，**官方 UI 不标记动态表达式**。
- 默认注册表：`{"glink": ["text"], "ptext": ["text"]}`；用户在设置页新增的
  条目保存在语言 JSON 的 `tags` 字段中。
- `saveCurrentTable()`：只把非空译文写回 `scenario` / `tag` 行；空译文删除行。
  `saveCharaTable()` / `saveSystemTable()` 会连空值一起保存。
- `saveLangFile()`：`JSON.stringify(map_trans, null, 2)` 写到
  `data/others/lang/<lang>.json`。
- CSV：
  - scenario CSV 的字段为 `type`、`original_text`、`trans_text`（UI 记录还带
    `recid`，import 忽略）；
  - chara / system CSV 为 `original_text`、`trans_text`；
  - import 使用 `delimiter=','`、`comment='#'`、`quote='"'`、`columns=true`，
    只更新当前源文件中仍然存在的 `original_text`，未知行被忽略；
  - 空译文在 scenario / tag import 时同样不写回。

## 7. 对 #265 P5 的 parser 合同建议

以下 schema 是 fixture 已经冻结的候选形状；实现 `TyranoAdapter` 时应沿用。

### 7.1 candidate / classification

- `translatable`：text 节点或已注册 tag 参数，无可用译文。
- `already_translated`：catalog 存在对应 row 且译文非空。
- `explicitly_excluded`：注释、label / `nw` 等无文本控制 tag、`lang_set`、
  `iscript` 边界与 `iscript` 内容、`chara_new` 定义。
- `unsupported`：已知 tag 但参数未注册（如 `[ruby text=...]`）、动态参数
  `&sf.*` / `&tf.*`。
- `unknown`：未注册宏调用或无法判断玩家可见性的结构。
- `parse_error`：未闭合引号、未闭合行内 tag、多个无引号参数序列等官方 parser
  会静默接受的缺陷。

fixture 冻结的 reason code 词表见
`tests/fixtures/tyranoscript_v600/expected/inventory.json` 和
`tests/test_tyranoscript_p5_fixtures.py`；实现阶段若扩词表，先更新三处：
inventory、测试常量、本文。

### 7.2 locator（建议 v1）

```json
{
  "engine": "tyrano",
  "locator_schema_version": 1,
  "locator": {
    "file_rel_path": "scene1.ks",
    "scenario": "scene1.ks",
    "line": 6,
    "node_index": 3,
    "kind": "text",
    "parser_value": "Hello, world!"
  }
}
```

- `line` 使用 0-based 源文件行号，与现有 `TranslationUnit.line` 约定一致。
- `node_index` 是 parser `array_s` 中的顺序；文本重复时由 line/node_index 区分
  occurrence，不按原文全局合并。
- tag candidate 增加 `tag_name` / `param_name`；comment candidate 使用
  `kind="comment"`、`node_index=null`。
- catalog row 只能定位到 `parser_value`，不能反推唯一 source span；写回计划
  必须回到 parser 节点/源行，不能在 catalog 上直接 patch 行号。

### 7.3 freshness / 写回前置条件

- source fingerprint 至少覆盖 `data/scenario/**/*.ks`、`Config.tjs` 的
  `KeepSpaceInParameterValue`、catalog 文件与 tag registry。
- `check` 至少拒绝：
  - `lang_set` 静态目标语言与 catalog 文件名不一致；
  - catalog 缺失 scenario section / text row / 已注册 tag row；
  - catalog 存在源中已不存在的 stale row（由 snapshot reconciliation 判定）；
  - 空字符串译文；
  - 任何 `unknown` / `parse_error` / 未解决 coverage finding 试图进入写回。
- 运行时不会替我们做上述任何检查，所以这些必须是公共层门禁，不能依赖
  TyranoScript 引擎报错。

## 8. Fixture 基线

`tests/fixtures/tyranoscript_v600/` 已冻结：

| 文件 | 覆盖点 |
|------|--------|
| `scene1.ks` | 对白 text、`#chara:face`、默认注册 `ptext` / `glink`、未注册 `ruby`、`lang_set` |
| `choices.ks` | 行内 tag 拆分 text、转义引号、注册自定义 tag `mymacro`、未注册宏、动态 `&sf.*`、`iscript` |
| `broken.ks` | 未闭合引号、多个无引号参数、未闭合行内 tag、块注释 |
| `ch.json` | 官方 JSON 形状、`systems` / `tags`、一个有意缺失的待译 text row |
| `expected/parser_nodes.json` | 官方 V602c parser 的宽松输出快照 |
| `expected/inventory.json` | P5 adapter 应达到的严格 candidate 输出合同 |
| `negative_cases.json` | missing row / stale extra / missing scenario / empty translation / stale tag registry 变异 |

表征测试：`python -m unittest tests.test_tyranoscript_p5_fixtures`（runner 中为
`python -B tests/run_cli_tests.py -q` 的一部分）。测试不调用模型、不运行 Node、
不写 `work/` / `build/`。

## 9. 已知风险与后续验证

1. **CSV 列顺序与 `recid`**：官方 Studio 通过 `csv_stringify(header=true)` 导出，
   实际列顺序未用真机 TyranoStudio 生成 golden；实现导入时应按列名读取，
   不得依赖顺序。
2. **`systems` 的运行时去向**：V602c runtime `convertLang` 不消费 `systems`；
   可能由 Studio 打包时替换 `tyrano/lang.js`。P5 首版先只读报告，不写 `systems`。
3. **tag 注册表的双存储**：`tags` 既在项目编辑数据中，也会随语言 JSON 保存；
   runtime 不需要它。以语言 JSON 的 `tags` 为 catalog provenance 的一部分，
   项目编辑文件若存在应做一致性检查。
4. **动态 `lang_set`**：`[lang_set name="&sf.select_lang"]` 是官方推荐用法；
   静态单 catalog 检查不能误报为 block，应标记 attention / dynamic。
5. **官方 parser 状态泄漏**：`flag_script` 跨文件不重置；adapter 必须逐文件
   重置并记录 behavior digest 版本。
6. **宏定义体文本**：官方 Studio 会提取，但 P5 首版建议单独标记
   `tyrano.macro_definition_text`，等真实项目 review 后再决定默认分类。
