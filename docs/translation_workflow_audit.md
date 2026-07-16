# Ren'Py Translation Lab 翻译全生命周期与影响因子全景审计报告

文档地图：[docs/README.md](README.md)

> ## 历史审计快照（非现行用户手册）
>
> 本文是对流水线、上下文与 CLI/GUI 门禁的**代码审计笔记**，便于理解设计动机。
> 实现细节可能已演进（例如 GUI 信息架构、项目级上下文开关、doctor 建议并列输出）。
> **现行操作说明**见 [Batch 工作流](batch_workflows.md)、[上下文系统](context_systems.md)、[GUI 工作台](gui_workbench.md)。

本报告对照审计时点的代码，梳理 `Ren'Py Translation Lab` 的**核心翻译流水线**、**提示词与上下文记忆系统**、**辅助工作流（订正与关键词）**以及 **CLI 与 GUI 同步门禁设计**。

---

## 1. 核心翻译流水线生命周期 (Build ➔ Apply)

整个批量翻译生命周期由 6 个高内聚、低耦合的阶段组成，并通过多重数字指纹和快照验证确保翻译的写回安全。

```text
 [ Build ] ──> [ Submit ] ──> [ Status ] ──> [ Download ] ──> [ Check ] ──> [ Apply ]
    │             │              │               │              │             │
 提取任务      上传云端并     轮询作业状态     下载翻译结果    relocation与  二次复核快照
 导出指纹      写入日志记录   (Quota轮换)     导出results.jsonl 干跑校验安全  安全写回脚本
```

### 六阶段生命周期详细审计

| 阶段 (Stage) | 输入依赖与物理文件 | 核心操作与算法逻辑 | 关键校验与防线 |
| :--- | :--- | :--- | :--- |
| **1. Build** | `.rpy` 脚本、`glossary.json`、`macro_setting.md`、RAG 数据库。 | 1. 过滤纯数字、URL 等非翻译行；<br>2. 整合 RAG 与剧情图谱上下文；<br>3. 按 `batch_target_size` 与字符上限拆分为 Requests 块。 | 写入 **`manifest_version=2`**（核心使用 **Identity v2** 独立标识符 `file:label:idx:checksum`），生成本地 `requests.jsonl` 和 `manifest.json`。 |
| **2. Submit** | `requests.jsonl`、`manifest.json`、`api_keys.json`。 | 1. 计算 max cost 估算；<br>2. 将请求打包提交至 Gemini API 批处理网关；<br>3. 写入操作流水日志 (`batch_submit_recovery`) 以防断电或超时损坏。 | 1. **哈希校验**：续传时比对 `requests.jsonl` 的 SHA-256 哈希值，防篡改；<br>2. **熔断器**：超限 `--max-cost` 时自动熔断阻止提交；<br>3. **Quota 轮换**：遇限制时自动尝试轮换 API key。 |
| **3. Status** | `manifest.json` 里的 `job_name`。 | 轮询 Gemini 批处理作业状态。 | **多 key 容错**：在多 API key 配置下，若当前 key 报 "Job Not Found"，自动遍历后续 key 查找。 |
| **4. Download**| `manifest.json`。 | 从 Gemini API 下载翻译结果，导出 `results.jsonl`。 | 状态门禁：校验 `job_state` 必须为 `JOB_STATE_SUCCEEDED`。 |
| **5. Check** | `results.jsonl`、`manifest.json`、物理 `.rpy` 脚本。 | 1. **V2 Relocation**：根据 V2 identity 扫描 live 脚本，重新定位发生偏移行数的翻译行；<br>2. **干跑校验**：验证括号、格式变量配对、中文占比（非白名单路径下）。 | 1. **漂移校验**：读取物理文件行与 manifest 保存的 source 原始快照对比，若有漂移触发 `Block` 级警告；<br>2. **指纹生成**：根据结果文件和配置哈希计算 `check_fingerprint`，锁定该次 Check 状态。 |
| **6. Apply** | `manifest.json`、`results.jsonl`、物理 `.rpy` 脚本。 | 1. 比对最新的文件指纹以防止“过时写回” (`stale_check_fingerprint`)；<br>2. 将翻译数据执行物理替换写入 `.rpy`；<br>3. **RAG 回灌**：写回成功后同步更新 RAG 本地库。 | 1. **指纹双向校验**：要求最近一次 Check 产生的指纹与当前内容无漂移；<br> 2. **二次复核**：写回前瞬间再次比对文本物理行，发生冲突立刻拒绝写回并输出失败报告。 |

---

## 2. 提示词与上下文记忆注入系统 (Prompt Context)

翻译质量的核心取决于提示词（Prompt）中上下文的准确度和容量控制。

```text
┌─────────────────────────────────────────────────────────────┐
│ System Instruction                                          │
│  - Style: [从 macro_setting.md 载入的翻译口吻与风格指引]     │
│  - Task: [翻译/订正指令与 LOCKED TERMS 词条复制约束]         │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ User Prompt (Payload)                                       │
│  1. LOCKED TERMS: 术语精准匹配 [Glossary]                    │
│  2. RETRIEVED MEMORY: 检索 RAG 记忆库 (Cosine Similarity >= 0.72)│
│  3. RELATED PROJECT CONTEXT: 原文索引库上下文                 │
│  4. STORY MEMORY: 剧情图谱 (登场角色打分, 限制最大 1200 字符)  │
│  5. CONTEXT BEFORE & TARGET (JSON Array) & CONTEXT AFTER    │
└─────────────────────────────────────────────────────────────┘
```

### 五大上下文系统的加载、打分与注入逻辑：

1. **术语锁定机制 (`glossary.json`)**
   - **载入**：在 `translator_runtime.py` 统一解析为 `PRESERVE_TERMS`（保留原英文词）和 `NORMALIZE_TRANSLATION_MAP`（强制替换词）。
   - **注入**：扫描 Target 行，提取匹配到的 `top_k_terms`（默认 5），并在 System Instruction 中声明复制约束，在 User Prompt 中以 `LOCKED TERMS` 列表强制执行。
2. **风格设定机制 (`macro_setting.md`)**
   - **载入**：检测项目根目录下的 `macro_setting.md` 物理文件，读入为全局 `BATCH_MACRO_SETTING`。
   - **注入**：直接注入在 System Instruction 提示词最上方，作为翻译全局风格的“最高宪法”。
3. **记忆检索检索 (`history.jsonl` 与锁机制)**
   - **文件锁机制**：读写 RAG 库时创建 `.rag_store.lock`。在 Windows 下使用 `ctypes` 读取进程 PID，若锁进程已死或超过 3600 秒未释放，则作过期自动清理。
   - **打分模型**：将最近的 2 行历史对话 + 当前 Target 组合转化为 Embedding 向量，与库中向量做余弦相似度计算。
   - **相似度与排序公式**：
     $$\text{Cosine Similarity} \ge 0.72$$
     满足阈值的候选记录，按 **余弦分值值 (Cosine Score) ＞ 质量等级 (Quality Rank) ＞ 时间戳 (Timestamp)** 进行三级降序排序，取 Top 4。
     - *质量等级权重*：`manual_polished` (手动订正) = 3 ＞ `seed` (种子数据) = 2 ＞ `applied` (普通翻译写回) = 1。
4. **剧情记忆检索 (`story_graph.json`)**
   - **第一步：提取当前角色**：比对当前文本和上下文，找出场景内活跃的 Speaker（匹配 Alias 别名和名字）。
   - **第二步：场景打分排序**：扫描剧情图谱中的 `scenes` 结构，计算场景推荐分数：
     - *文件名相符*：+50 分；
     - *行数重叠*：重叠时为 `100 + overlap`；不重叠时通过距离做衰减衰退 `max(0, 40 - distance)`；
     - *活跃角色出现在该场景*：每个匹配角色 + 8 分。
   - **第三步：关系与术语过滤**：提取 Top 3 场景，并将双方都是当前活跃角色的 Relations 关系，以及出现在文本中的 Terms 条目过滤提取。
   - **第四步：Token 预算拦截**：剧情记忆转换为文本注入时，有严格的字符截断限制 `STORY_MEMORY_MAX_CONTEXT_CHARS`（默认 1200 字符），防止膨胀挤占翻译窗口。

---

## 3. 辅助流程分析（订正、关键词与合并）

除普通翻译外，订正与关键词提取形成了独立的资产闭环：

### 辅助工作流对比矩阵

| 特性 / 流程 | 批量订正 (Revisions) | 关键词提取与合并 (Keywords & Merge) |
| :--- | :--- | :--- |
| **主要命令** | `build-revisions` $\rightarrow$ `preview-revisions` $\rightarrow$ `apply-revisions` | `build-keywords` $\rightarrow$ `export-keywords` $\rightarrow$ `merge-keywords-to-glossary` |
| **输入数据单元** | 对比 `.rpy` 已有的 `old/new` 和注释译文。 | 将源文本以较大的 `KEYWORD_CHUNK_SIZE`（便于提取宏观剧情与高频术语）分块提取。 |
| **产出数据单元** | 带有纠正/润色建议的译文行。 | 返回 `candidates` 候选词及 `chunk_summary` 场景大纲。 |
| **写回安全闸门** | `preview-revisions` 检查非中文及格式；`apply-revisions` 进行行快照一致性比对。 | **不修改游戏脚本**。生成 `keyword_candidates.jsonl` 和 `keyword_chunk_summaries.jsonl` 报告。 |
| **合并逻辑** | 无合并，直接应用到游戏脚本。 | 通过 `keyword_glossary_merge.py` 将候选词并入 `glossary.json`：<br>- 英文词并入 `preserve_terms`；<br>- 有译文词并入 `normalize_map`。 |
| **CLI / GUI 对齐** | GUI 通过 `QProcess` 包装 CLI 子进程执行生成和预览。 | **GUI 直接作为模块导入调用**。在 `KeywordMergeDialog` 表格中显示冲突，允许用户手动筛选、去重与强行覆写。 |

---

## 4. CLI 与 GUI 的同步设计与门禁安全边界

GUI 保持与 CLI 的强一致性，主要依靠**进程隔离**与**选择性门禁策略**。

### 1. 进程隔离与 SSOT (单一事实源)
- 耗时且计算繁重的流水线子任务（Build, Submit, Status, Download, Check, Apply, Revision）统一通过 `QProcess` 子进程拉起 CLI 脚本。这保证了底层在文本解析、API 交互、RAG 向量检索时**使用完全相同的 Prompt 模板和库文件**。
- CLI 的 `Check` 会把 `last_check_summary` 和基于 SHA-256 的完整性指纹 `check_fingerprint` 写入 manifest；该指纹不是加密。GUI 主要读取 `last_check_summary.safety_level` 展示检查结果，真正的强制指纹复核由 CLI `apply` 预检执行：重新计算后的指纹不一致时以 `stale_check_fingerprint` 拒绝写回。

### 2. 精准的门禁放行策略 (Gating Policy)
- 在审计时点的 `gui_qt/app.py` 中，GUI 根据不同的工作模式应用不同的环境检查（Doctor Check）门禁；现行入口与页面归属请以 [GUI 工作台](gui_workbench.md) 和当前代码为准：
  - **翻译类模式**（`BATCH_TRANSLATION`, `SYNC_TRANSLATION`）: 强制卡口。必须先通过 Doctor 环境检查，且结果不能有 `Blocked` 报错，翻译按钮才会被允许激活。
  - **准备/维护类模式**（`KEYWORD_EXTRACTION`, `BOOTSTRAP_RAG`, `REVISION`）: 忽略卡口门禁。因为它们是**只读工具**或**准备性工具**，它们自己就是解决项目异常的方法（例如在“预建记忆库”时，虽然 RAG 会报未就绪警告，但按钮决不能禁用，否则用户将陷入无法预建的死循环）。
