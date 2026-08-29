# tyranoscript_v600 fixture（#265 P5 parser 预研基线）

本目录是 TyranoScript V600+ Adapter 的**离线 parser / catalog / candidate 基线**。
当前不包含生产 adapter 实现；`tests/test_tyranoscript_p5_fixtures.py` 只校验夹具
自洽性和手写期望，不要求 Node 或 TyranoScript 运行时。

## 目录结构

- `data/system/Config.tjs` — 最小项目配置；冻结 `KeepSpaceInParameterValue = 2`
  （官方 V600 模板默认值，影响所有带空格参数值和 catalog key）。
- `data/scenario/scene1.ks` — 对白、`#chara_ptext`、官方默认注册的
  `[ptext text=...]` / `[glink text=...]`、未注册的 `[ruby]`、`[lang_set]`。
- `data/scenario/choices.ks` — 行内 tag 切分文本、注册的自定义 tag
  `[mymacro value=...]`、未注册宏、动态参数 `&sf.button_label`、`[iscript]` 排除。
- `data/scenario/broken.ks` — 官方解析器会“补偿”或静默接受的三类缺陷：
  未闭合引号、多个无引号参数、未闭合行内 tag。P5 adapter 必须将它们升级为
  `parse_error`，不能沿用官方解析器的宽松结果。
- `data/others/lang/ch.json` — 按官方 V603 Studio 生成规则手写的原生 catalog
  （运行时消费 `scenes` / `charas`；`systems` / `tags` 由 Studio 写入同一文件）。
- `expected/parser_nodes.json` — 官方 V602c runtime `kag.parser.js` 对本夹具的
  冻结解析输出（`KeepSpaceInParameterValue = 2`，逐文件重置 `flag_script`）。
  这是 characterization 的“旧行为快照”，不是 P5 adapter 的期望输出。
- `expected/inventory.json` — 手写的 P5 candidate inventory 合同：每个 parser
  node / comment line 恰有一个候选，包含分类、reason code、parser 归一化值、
  catalog 行路径和 evidence。
- `negative_cases.json` — 对基线 catalog 的变异规格，用于后续 stale / missing /
  empty translation 负路径测试；当前测试只校验变异目标在基线中存在。

## 维护约束

- 测试通过 universal newline 读取夹具；Windows checkout 产生的 CRLF 会被归一化，不参与断言。
- `expected/parser_nodes.json` 不得手改；若官方 parser 行为重新核实有变化，先更新
  `docs/plans/tyranoscript_v600_parser_research.md` 中的材料引用，再整体重生成。
- 修改 `expected/inventory.json` 时必须同时更新 fixture 测试中的原因码集合和
  研究文档的 candidate 合同。
