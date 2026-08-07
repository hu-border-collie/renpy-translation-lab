# RenPy Polishing Toolkit 原型归档

> **状态：历史原型，仅供设计参考。**
> 本目录不属于 Ren'Py Translation Lab 的稳定 CLI、GUI 或公共 API；不要直接运行其中脚本处理正式项目。

## 归档来源

- 原始目录：`C:\RenPy_Workspace\RenPy_Polishing_Toolkit`
- 归档日期：2026-08-07（Asia/Shanghai）
- 原目录未包含独立 Git 元数据、测试目录或许可证文件。
- 本次只保存文本快照，没有归档 `__pycache__/`。
- 原始文件的 SHA-256 记录在 [SHA256SUMS](SHA256SUMS)；归档文本已逐项执行换行归一化后的内容等价检查。

脚本统一保存为 `*.py.txt`，避免它们被误认为受支持的可执行入口。源码发行包会包含 tracked 文件，因此不得把这些副本恢复为可直接运行的 `.py` 文件。
原型原文含有行尾空格；为保持快照内容不变，`.gitattributes` 只对本目录关闭 `trailing-space` diff 报警，正式代码与其他文档仍使用原有检查规则。

## 归档内容

| 原文件 | 归档副本 | 原型意图 |
|---|---|---|
| `extract_dialogue_pairs.py` | [source/extract_dialogue_pairs.py.txt](source/extract_dialogue_pairs.py.txt) | 导出原文、现译、文件与行号供线性通读 |
| `apply_polish_batch.py` | [source/apply_polish_batch.py.txt](source/apply_polish_batch.py.txt) | 从 Markdown 润色提案批量写回 |
| `verify_polish_status.py` | [source/verify_polish_status.py.txt](source/verify_polish_status.py.txt) | 检查提案译文是否已经出现 |
| `apply_glossary.py` | [source/apply_glossary.py.txt](source/apply_glossary.py.txt) | 按项目术语映射批量替换 |
| `verify_glossary.py` | [source/verify_glossary.py.txt](source/verify_glossary.py.txt) | 扫描遗留源词 |
| `deep_polish_engine.py` | [source/deep_polish_engine.py.txt](source/deep_polish_engine.py.txt) | 使用固定中文替换规则机械润色 |
| `README.md` | [original_README.md.txt](original_README.md.txt) | 原始迭代润色工作流说明 |
| `polishing_guide.md` | [polishing_guide.original.md.txt](polishing_guide.original.md.txt) | 原始润色与校对 SOP |

## 采用结论

值得正式吸收的是工作流设计：

```text
导出原文/现译对照
→ 人工或 Agent 起草可审计提案
→ 明确审核
→ 安全预览
→ 显式写回
→ 质量审计与 Ren'Py lint
```

正式实现必须落到现有 revision / final-review / engine-adapter 架构：

- 使用稳定 occurrence / identity，而不是按译文字符串模糊查找；
- 提案绑定原文、当前译文和源文件快照；
- 导入只生成 revision package 和预览，不直接修改 `.rpy`；
- 最终写回只能经过 `preview-revisions → apply-revisions`；
- glossary 与机械质量问题生成可定位 finding 或 revision candidate，不做无审计全局替换；
- CLI、GUI、配置、中文文案、现行文档和测试同步交付。

## 明确不采用

- 不移植 `str.replace()` 式全文件写回。
- 不接受去掉省略号后的部分字符串作为写回定位依据。
- 不把“建议译文在任意文件中出现”当成成功验证。
- 不引入无上下文固定词表润色；例如原型会把“我紧紧地抱住他”改成不通顺的“我抓紧不放抱住他”。
- 不把项目专用硬编码路径或术语并入默认配置。

## 跟踪关系

- [Epic #318：人工润色提案工作流](https://github.com/hu-border-collie/renpy-translation-lab/issues/318)
- [#319：归档原型与设计评估](https://github.com/hu-border-collie/renpy-translation-lab/issues/319)
- [#320：导出带稳定 ID 的原文／现译润色语料](https://github.com/hu-border-collie/renpy-translation-lab/issues/320)
- [#321：导入人工／Agent 润色提案并接入安全预览写回](https://github.com/hu-border-collie/renpy-translation-lab/issues/321)
- [#322：订正页 GUI 产品化](https://github.com/hu-border-collie/renpy-translation-lab/issues/322)
- 机械质量检查、glossary 一致性与写回/质量状态拆分已由 [GitHub #313](https://github.com/hu-border-collie/renpy-translation-lab/issues/313) 跟踪。

本目录只作为设计证据。日常使用以根目录 `README.md`、`docs/batch_workflows.md`、现行 CLI `--help` 和 GUI 为准。
