# 润色工具原型设计摘录

> **状态：历史设计摘录，仅供参考。**
> 本目录不属于 Ren'Py Translation Lab 的稳定 CLI、GUI 或公共 API。

## 脱敏说明

早期原型曾包含项目专用术语、角色名和本机绝对路径。当前公开快照不再保存
这些原始正文、脚本副本或内容哈希，只保留下方与具体项目无关的设计结论。
需要考证旧实现时应在非公开、权限受控的副本中进行，不要从公开仓库恢复后
直接运行。

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

本目录只作为设计摘录。日常使用以根目录 `README.md`、`docs/batch_workflows.md`、现行 CLI `--help` 和 GUI 为准。
