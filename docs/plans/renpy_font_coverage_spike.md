# Ren'Py 目标译文字集与字体字形覆盖只读 spike（#487）

> **状态**：spike 已实现；离线 API/脚本、fixture、报告与生产接入方案已落地。
> **关联**：#487、#364、#424、#272、#265、#426。
> **非目标**：不启动游戏、不下载/替换字体、不执行游戏 Python、不实现生产 doctor/GUI 功能、
> 不把「字体存在」当作「渲染验收通过」。

## 1. 问题与范围

文本结构检查和 `apply` 成功不能证明游戏内目标语言可正常显示。当前已有 GUI 字体下载/设置
能力，但没有面向游戏译文字集的字体字形覆盖检查。本 spike 只做 Ren'Py 范围的只读验证：

- 解析静态 `font` 引用（style 属性、`gui.*_font` 赋值、字符串字面量）。
- 解析字体 `cmap`，对目标译文字集做 `checked / missing / unknown` 报告。
- 对动态表达式、`FontGroup` / fallback 链、TTC 多 face 等无法可靠确定的配置标记 unknown。
- 区分「字体文件缺失 / 损坏」「字体确定缺字」「实际渲染字体链不确定」。

## 2. 实现

| 层 | 位置 | 说明 |
|---|---|---|
| 核心 API | `font_coverage.py` | `load_font_face` / `scan_font_references` / `extract_translation_strings` / `analyze_font_coverage` / `format_report_markdown` |
| CLI | `scripts/font_coverage_report.py` | `--game-root`、可选 `--tl-dir` / `--text-file` / `--text`；`--output markdown|json`；可原子 `--output-file` |
| 离线测试 | `tests/test_font_coverage.py` | 正反例、无副作用、路径边界、CLI JSON |
| Fixture | `tests/fixtures/font_coverage_minimal/` | 原创最小 Ren'Py 脚本、可再分发 Noto 子集字体、损坏字体、译文/文本样本 |

核心 API 不导入 Qt、不导入 Provider SDK，只用标准库读取 `.rpy` 文本与字体表。

## 3. 字体解析边界

- 支持 sfnt：TrueType `0x00010000`、`OTTO`、`true`、`typ1`；TTC 只读取第一个 face。
- 支持 cmap format 12（UCS-4）、4（BMP）、6、0；按 Unicode 子表优先级选择。
- 读取 `maxp.numGlyphs` 与 `name` family 作为证据；字体文件上限 64 MiB。
- 字体引用路径必须解析在 `game_root` 内；越界引用标记
  `font.path_outside_game`，不读取任意本机文件。
- 损坏/缺表/不支持 cmap 都返回稳定 reason code，不抛 Provider 风格异常。

## 4. 文本与样式解析边界

- `.rpy` 只按行解析 `style <name>:` 块内的 `font` 属性与 `define gui.*_font = ...` 赋值；
  不执行 `init python`、`$` 或任意表达式。
- 只有**完整字符串字面量**会解析为静态字体路径；含 `+` / `%` / 变量 / 下标 /
  方法调用的表达式标记 `font.dynamic_expression`。
- `FontGroup(...)` / fallback 链标记 `font.group_unsupported`。
- `None` 或空字符串标记 `font.not_declared`；没有任何字体声明时整体报告 unknown。
- 文本样本来源：显式 `--text`、`--text-file`（每行一条）、`--tl-dir`（best-effort 提取
  `new "..."` 与译文对话行；`old` 原文跳过）。`{tag}` 会从字形检查中移除，
  `[name]` 动态插值会让样本计入 `dynamic_sample_count`，不会冒充静态字集。
- `unknown` 不代表通过；报告不会把动态配置或未知 fallback 当作 checked。

## 5. 报告字段与 reason code

JSON 顶层：

```json
{
  "schema_version": 1,
  "spike": "font_coverage",
  "status": "checked | missing | unknown",
  "text": {"sample_count": 3, "unique_char_count": 31, "dynamic_sample_count": 0, "sources": ["tl", "cjk_covered.txt"]},
  "fonts": [
    {
      "script": "scripts/styles.rpy", "line": 3, "style": "default",
      "expression": "\"fonts/test_cjk_subset.ttf\"", "kind": "static",
      "status": "checked", "reason": "", "font_path": "fonts/test_cjk_subset.ttf",
      "cmap_format": 4, "glyph_count": 34, "missing_char_count": 0,
      "missing_chars": [], "evidence": "scripts/styles.rpy:3 font=... cmap=4 glyphs=34"
    }
  ],
  "checked": ["..."], "missing": ["..."], "unknown": ["..."],
  "limits": {"max_missing_chars": 200, "max_font_bytes": 67108864},
  "limitations": ["..."]
}
```

稳定 reason code：

| code | 含义 | 建议处理 |
|---|---|---|
| `font.file_missing` | 静态字体路径不存在 | doctor/preflight warning；提示补充字体或改为正确路径 |
| `font.invalid` | 字体文件损坏 / 不是可解析 sfnt | warning；不自动替换 |
| `font.cmap_unsupported` | 缺少受支持的 Unicode cmap 子表 | warning；必要时人工确认 |
| `font.glyph_missing` | 字体确定缺少目标字符 | warning（不默认 blocker）；提示换字体/子集 |
| `font.dynamic_expression` | 字体引用是动态表达式 | unknown；不猜运行时值 |
| `font.group_unsupported` | FontGroup / fallback 链不可静态解析 | unknown；不宣称必然缺字 |
| `font.not_declared` | 未找到静态字体声明 | unknown；提示显式配置或人工确认 |
| `font.expression_unparsed` | 表达式无法安全解析 | unknown |
| `font.path_outside_game` | 引用越出 `game_root` 输入边界 | unknown；拒绝读取 |
| `text.no_samples` | 没有显式/文件/TL 文本样本 | unknown；不把空字集当通过 |

`unknown` 不当作通过，也不默认作为结构 blocker；`missing` 的确定性缺字可以在
doctor/preflight 中作为 warning，是否升级为 blocked 由后续产品决策决定。

## 6. Fixture 与许可证

`tests/fixtures/font_coverage_minimal/` 只包含原创最小 Ren'Py 文本、原创损坏字节与
Noto 字体子集：

- `game/fonts/test_cjk_subset.ttf`：`NotoSansCJK-Regular.ttc` face 2（Noto Sans CJK SC）
  子集，覆盖 fixture 的中文/标点样本。
- `game/fonts/test_latin_subset.ttf`：`NotoMono-Regular.ttf` 子集，故意缺少 CJK 字形，
  用于确定缺字负例。
- 上游：Google Noto fonts；许可：SIL Open Font License 1.1，完整文本与 SHA-256 见
  `tests/fixtures/font_coverage_minimal/game/fonts/README.md` 与 `LICENSE-OFL-1.1.txt`。
- `corrupt.ttf` 为项目生成的非字体字节；不提交任何私有游戏字体、脚本、地图或对白。

## 7. 离线验证覆盖

- 完整字集覆盖：CJK 子集 + `texts/cjk_covered.txt` + `tl/schinese` → `checked`。
- 确定缺字：Latin 子集 + 同一中文文本 → `font.glyph_missing` 且列出缺字。
- 字体缺失 / 损坏：`fonts/not_here.ttf`、`corrupt.ttf` → 稳定 reason code。
- 动态表达式 / FontGroup：`gui.dynamic_font`、`FontGroup().add(...)` → `unknown`。
- 无字体声明：只有普通 `.rpy` 的临时 game_root → `font.not_declared`。
- 输入边界：`../outside.ttf` → `font.path_outside_game`，不读取外部字体。
- 无副作用：fixture 中的 `init python` 会写 `FONT_COVERAGE_SIDE_EFFECT.txt`；
  扫描/CLI 运行后该文件必须不存在。
- CLI：`--output json` 输出稳定 envelope 形状，不修改项目。

## 8. 生产接入建议（后续切片）

建议只在 doctor / `translate-preflight` 增加**只读 warning**，不进入写回授权：

- doctor 字段：`font_coverage.status`、`font_coverage.checked` / `missing` / `unknown`、
  `font_coverage.reason_counts`、`font_coverage.evidence`。
- preflight 字段：把 `missing` 的缺字数与涉及文件加入 readiness reasons；
  `unknown` 单独展示，不参与 blocker 计算。
- 中文文案建议：
  - 缺字：`目标译文字符集有 {count} 个字符未被声明的字体覆盖；请确认字体或 fallback。`
  - unknown：`无法静态确定实际渲染字体链（动态表达式 / FontGroup）；需要人工或运行时验证。`
  - 字体缺失：`字体文件不存在：{path}；当前不会自动替换或下载。`
- GUI 展示建议：诊断页按字体引用逐行显示 `checked/missing/unknown`、证据路径与缺字样例；
  不提供一键替换字体按钮，不把 unknown 显示成“通过”。
- 性能预算：只扫描 `.rpy` 文本与静态字体文件；单字体上限 64 MiB；不启动 Ren'Py、
  不加载 Qt、不执行游戏代码。真实大型项目的覆盖率/耗时需要在接入前重新测量。
- 误报风险：TTC face 选择、FontGroup/fallback、运行时 style 覆盖、shaping/排版、
  动态插值文本；这些在 spike 中全部进入 unknown 或限制说明，不宣称游戏必然缺字/必然可显示。

## 9. 结论

- 该只读检查在 Ren'Py 范围内可行，且能在不执行游戏的前提下给出确定缺字与 unknown 证据。
- 生产接入价值成立，但必须保持 warning/unknown 语义：字形覆盖不等于排版和运行时验收通过。
- 后续接入 doctor/preflight 时，建议把本 spike 的 reason code 与报告字段作为合同输入，
  再决定是否升级为 blocker；自动替换/注入字体不在本 spike 范围。
