# 实际模型用量账本

文档地图：[docs/README.md](README.md)

实际模型用量账本把 Batch、同步 Gemini、LiteLLM 及分析流程返回的 usage metadata 归一到当前游戏项目。它只增加旁路统计，不改变翻译、检查、预览或写回语义；Batch 仍必须遵守 `check -> apply` 合约。

## 项目隔离与文件位置

每个 `game_root` 使用自己的账本：

```text
<game_root>/translation_usage/usage_ledger.json
```

文件中的 `project.game_root` 是规范化绝对路径，`project.project_id` 是该路径的 SHA-256 identity。读取和写入都会核对 identity，不允许把两个游戏项目的记录混入同一文件。写入使用同目录临时文件和原子替换；GUI 只读取该公共账本，不维护第二份统计状态。

账本包含逐响应记录，主要字段为：

- `operation_id` / `run_id` / `manifest_id`
- `task_mode`：`translation`、`revision`、`keyword`、`repair`、`analysis`
- `stage`：例如 `batch_translation`、`sync_revision`、`compare_variants`、`label`、`route`、`brief`
- `provider` / `model` / `thinking_level` / `execution_mode`
- `calls`
- `prompt_tokens` / `completion_tokens` / `total_tokens` / `thoughts_tokens` / `cached_tokens`
- `provider_usage`：provider 返回的原始 usage 摘要，不保存响应正文
- `estimated_cost` 与 `actual_cost`
- `recorded_at` / `dedupe_key` / 响应与来源 identity

缺失 token 或 cost 保持 `null`。聚合报告同时给出 `*_known_records` 与 `*_unknown_records`，不会把未知值伪造成 0。

## 数据来源与自动记录

| 路径 | 记录时点 | 说明 |
|---|---|---|
| Gemini Batch | `download` 得到或复用完整 `results.jsonl` 后 | 只读取本地结果，不重新调用 API |
| 同步关键词 / 订正 | 同步 `results.jsonl` 原子落盘后 | Gemini 与 LiteLLM 使用同一导入器 |
| 普通同步翻译 | 每个成功响应返回后 | 即使后续 JSON 解析或译文校验失败，已发生的调用仍会记录 |
| repair / probe / A/B | 每个成功响应返回后 | 分别归入 `repair` 或 `analysis` 阶段 |
| Project Analysis | label、route、brief 成功响应后 | map-reduce 通过 stage-aware recorder 复用同一核心接口 |

自动记录失败只产生警告，不会把一次已经安全完成的下载、翻译或分析改成失败。需要排查或补录时使用显式 CLI。

### 幂等与 retry / split

有 provider response id 时，去重键优先绑定该 identity；没有 id 时使用 provider、model、请求行 key 与响应摘要 fingerprint。重复执行 `download` 或 `usage-import` 不会重复累计同一响应。

`split` 子包和完整包之间复制同一响应时会命中同一去重键。`merge-retry` 会先导入已验证的 parent 与 retry 原始结果；之后对已合并 manifest 执行 `usage-import` 时，也会展开到原始结果 lineage，不把本地合成行当作新的 provider 调用。实际重新发出的 retry / repair 请求仍是新的计费调用，应单独记录。

## CLI

### 离线导入

```powershell
python gemini_translate_batch.py usage-import C:\path\to\manifest.json
```

省略 target 时沿用现有 latest-manifest 规则。该命令只读取 manifest 与已经落盘的结果；不会创建 client、提交任务或请求 provider。

机器可读摘要：

```powershell
python gemini_translate_batch.py usage-import C:\path\to\manifest.json --json
```

输出包含扫描行数、候选记录数、新增/重复记录数、账本路径及当前项目聚合报告。

### 查询与聚合

```powershell
python gemini_translate_batch.py usage-report
python gemini_translate_batch.py usage-report --task revision --provider gemini
python gemini_translate_batch.py usage-report --stage brief --model gemini-3.1-flash-lite
python gemini_translate_batch.py usage-report --group-by task,stage,provider,model --json
```

可用过滤器：

- `--task`
- `--stage`
- `--provider`
- `--model`

`--group-by` 支持 `task`、`stage`、`provider`、`model`、`run`、`operation`、`execution`，默认按 task / stage / provider / model 分组。`usage-import` / `usage-report` 的机器输出开关是独立的 `--json`；它们是只读/本地账本命令，不使用 Batch workflow 的 `--output json` envelope。

## GUI

「诊断与运行日志」的任务上下文会显示：

- 当前项目累计调用数和已知 total token
- token 不完整时的未知记录数
- 最近一次运行的 task / stage / provider / model 与 token 摘要
- 存在时的估算成本和 provider 报告成本

命令参考提供「导入当前结果用量」与「查看项目模型用量」。切换 `game_root` 后读取新的项目账本；账本 mtime 参与刷新键，因此同一任务记录下新增用量也会刷新。

## 成本语义

成本字段严格区分：

- `estimated_cost`：使用当前配置价格表和已知 token 计算，`estimated_cost_basis=configured_pricing`；只是估算，不是 provider 账单。
- `actual_cost`：只有 usage 或响应 metadata 明确给出 cost 时才记录，并保留 `actual_cost_source`。
- 没有匹配价格、缺少必要 token 或 provider 未返回 cost 时，值为未知，不显示 0 成本。

当前 `batch.pricing` 是 Batch 估算价格表，因此自动导入的 Batch 响应可以生成估算成本；同步路径不会擅自把 Batch 费率当作同步账单。Project Analysis 延续其现有配置费率估算，但仍标记为 estimate。

## 隐私与版本控制

账本保存 provider usage 摘要、模型/阶段 identity 和响应 fingerprint，不保存 prompt、译文或完整响应正文。它仍属于项目本地运行数据；不要把真实游戏项目账本、Batch 结果或日志提交到公开仓库。

## 设计来源

provider-neutral tracker、按阶段归因与跨续跑增量合并的方向参考了 [BigDawnGhost/wenyi](https://github.com/BigDawnGhost/wenyi)（MIT License）的设计思路。本实现依据本项目现有 manifest、同步 backend、GUI 与安全写回边界独立编写，没有复制 Wenyi 源代码或实质性代码片段。
