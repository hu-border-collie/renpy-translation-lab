# 环境检查智能建议状态关联矩阵审核报告

文档地图：[docs/README.md](README.md) · 建议规则说明：[doctor_recommendations.md](doctor_recommendations.md)

> 本文是 **doctor 派生字段与决策漏斗的开发对照表**。用户可见建议文案与 GUI 入口名称见 [doctor_recommendations.md](doctor_recommendations.md) 与 [gui_workbench.md](gui_workbench.md)。
> 布局就绪后，上下文相关**必需准备与可选建议可并列输出**（见 `collect_doctor_recommendations`）。

---

## 1. 维度一：文件与目录结构状态 (Layout Status)

这是环境检查的最底层，决定项目所处的物理阶段，也是判定是否存在“阻断级”异常的基础。

| 评估字段 / 派生事实 | 校验逻辑与物理路径 | 对应含义 |
| :--- | :--- | :--- |
| **`is_work_root`** | 检查当前工作路径 `base_dir` 的最后一级文件夹名是否为 `work` (大小写不敏感)。 | 判断用户是否在隔离的沙盒工作区 (`work`) 下运行。 |
| **`work_exists`** | 检测 `work_dir`（默认指向 `base_dir/work`）文件夹在物理磁盘上是否存在。 | 判断工作目录是否存在。 |
| **`work_empty`** | 调用 `legacy.is_work_dir_empty()`，校验 `work/game/` 是否为空或不存在。 | 判断工作区是否为空（即尚未进行任何 bootstrap 复制）。 |
| **`original_game_exists`** | `resolve_original_game_dir()` 优先接受已配置且存在的 `SOURCE_GAME_DIR`，否则只检查 `<project>/original/game`。 | 判断原始游戏目录是否可用。 |
| **`work_bootstrap_allowed`** | `work_dir_bootstrap_allowed()` 只判断 work 目录是否不存在或为空；生成 `BOOTSTRAP_WORK` 建议时还会另外要求 `original_game_dir` 存在。 | 区分“work 可写入”与“已有来源可复制”两个事实。 |
| **`tl_exists`** | `os.path.isdir(tl_dir)` 判定翻译语言子目录是否存在。 | 判定翻译输出目录（默认 `game/tl/schinese`）是否存在。 |

### ➔ 派生的 `layout_status` 四大基线状态：
- **`switch_to_work`**：当前在原游戏目录，但工作区 `work` 已有内容或原游戏有可处理资源。
- **`failed`**：没有 `work` 目录，且找不到 `original/game` 资源，处于无法初始化的损坏状态。
- **`attention`**：处于 `work` 目录，但没有检测到翻译文件，需要先做准备目录或生成模板。
- **`ready`**：处于 `work` 目录，且翻译模板就绪，可以直接执行翻译。

---

## 2. 维度二：翻译数据与计数状态 (Metrics & Counts)

当目录结构正常时，环境检查将深度扫描翻译文件夹中的 `.rpy` 脚本以计算进度与漏译统计。

| 评估字段 / 派生事实 | 扫描的正则与算法依据 | 对应含义 |
| :--- | :--- | :--- |
| **`rpy_files`** | 统计 `counts['rpy_files']` (翻译目录下所有的 `.rpy` 脚本数量)。 | 模板文件数。为 0 时表示未生成任何模板。 |
| **`translate_blocks`** | 匹配以 `translate <language> <label>:` 开头的对话翻译块。 | 对话翻译块总数（剧情句子的总行数基准）。 |
| **`commented_original_lines`** | 统计对话翻译块内部，以 `#` 开头的原文备份注释行。 | 作为重配对、RAG 或订正的上下文对照基准线。 |
| **`old_lines`** / **`new_lines`** | 匹配界面字符串块中的 `old "..."` / `new "..."` 数量。 | 界面文本原文/译文行数（如果不等，会触发“行数不一致”的格式警报）。 |
| **`has_existing_translations`** | 派生：`counts['old_lines'] > 0` 或 `translate_blocks > pending_task_count`。 | 判定当前项目是“全新初译”还是“已有旧译（增量补译）”。 |
| **`pending_task_count`** | `collect_pending_file_jobs()` 提取出的所有待翻译任务数。 | 过滤掉纯数字、纯标点、URL 等无效翻译后，剩余需提交翻译的行数。 |
| **`pending_is_minor`** | 派生：`pending < 50` 或 `pending / baseline < 0.01` | 剩余翻译量是否已经微乎其微（对应项目基本译完）。 |

---

## 3. 维度三：Ren'Py SDK 与模板生成能力状态 (Template Mode)

当项目缺少翻译模板时，决定如何指导用户完成模板生成环境配置。

| 配置文件关联项 | 校验逻辑与物理检测 | 对应建议与 UI 模式影响 |
| :--- | :--- | :--- |
| **`prepare.enabled`** | `translator_config.json` 中的 `"prepare": { "enabled": ... }` 开关。 | 准备工作（包括 RPA 解包、生成 tl 模板等）是否被允许执行。 |
| **`prepare.renpy_sdk_dir`** | 检测配置路径下是否有 `renpy.exe` 或启动器脚本。 | SDK 路径配置。 |
| **`can_generate_template`** | 检查 SDK 目录、或者系统 PATH 里的 `renpy`、或者 custom 命令是否可用。 | 派生：当前系统环境是否具备自动生成模板的执行能力。 |
| **`mode` (检查模式)** | 派生结果：<br>1. **`existing_tl_only`**（有翻译文件，不管 SDK 在不在）<br>2. **`can_generate_template`**（无翻译文件但 SDK 可用）<br>3. **`blocked_missing_template`**（无翻译文件且无法生成） | 决定项目在模板生成能力上的分类状态。若为 `can_generate_template`，GUI 翻译按钮会自动变为 **“生成翻译模板”** 并重定向点击目标。 |

---

## 4. 维度四：上下文存储库状态 (RAG & Source Index Context)

确保翻译时的检索增强功能和原文索引功能处于健康、可用状态。

| 配置文件关联项 | 校验逻辑与物理文件 | 派生状态与建议影响 |
| :--- | :--- | :--- |
| **`batch.rag.enabled`** | 控制是否使用历史翻译记忆库。 | RAG 模块开关。 |
| **`batch.rag.store_dir`** | 检测该路径下是否存在 `history.jsonl` 和 `metadata.json`。 | RAG 记忆库物理文件存在性。 |
| **`rag_needs_bootstrap`** | `enabled` 开启，且物理库不存在，或历史记录数 `history_records <= 0`。 | **必需准备**：记忆库未就绪。根据 `bootstrap_on_build` 配置，建议手动预建或提示会在 build 时自动暖库。 |
| **`batch.source_index.enabled`**| 控制是否使用原文上下文索引。 | 原文索引开关。 |
| **`source_index_needs_bootstrap`** | `enabled` 开启，且物理库不存在，或索引片段数为 0，或索引数少于预期数量。 | **必需准备**：原文索引未建立（`missing`）或建立不完整（`incomplete`）。 |

---

## 5. 维度五：项目资产关联与冲突状态 (Assets & Conflicts)

校验术语表、翻译规范和角色记忆是否存在路径偏移、缺失或内容冲突。

| 资产文件 | 检测逻辑与物理位置 | 对应报警/处理机制 |
| :--- | :--- | :--- |
| **术语表 (`glossary.json`)** | 1. 检测文件是否存在；<br>2. 用 `paths_match_project()` 检测路径是否与当前 `base_dir` 严格匹配。 | 1. 路径非当前项目：报**配置路径偏移警告**；<br>2. 物理缺失：报**缺失警告**（降级使用默认保留词）。 |
| **风格规范 (`macro_setting.md`)**| 1. 检测文件是否存在；<br>2. 用 `paths_match_project()` 检测。 | 同上。偏移报**偏移警告**；物理缺失报**缺少风格指引警告**。 |
| **术语/记忆冲突** | 对比术语表 `normalize_map` 和剧情记忆 `story_graph.json` 实体表。 | **术语冲突警告**：如果两者对同一个词给出了不同的翻译设定，会报出冲突，要求人工统一。 |

---

## 6. 决策流（基于优先级的漏斗抑制逻辑）

决策分两段：布局 / 模板阻断仍通过漏斗提前返回；布局就绪后，上下文建议改为累计收集，并按“必需在前、可选在后”输出：

```text
========================================================================================
 优先级 1: 路径与隔离阻断 (Blocking)
----------------------------------------------------------------------------------------
 [条件] 1. layout_status == 'switch_to_work' 
        ──> 触发建议：SWITCH_TO_WORK 且若允许则同时触发 BOOTSTRAP_WORK。
 [条件] 2. work_bootstrap_allowed 
        ──> 触发建议：BOOTSTRAP_WORK。
        ➔ 拦截返回：不继续往下运行。

========================================================================================
 优先级 2: 模板生成阻断 (Blocking)
----------------------------------------------------------------------------------------
 [条件] 1. 无 rpy_files 且 prepare_enabled 为假
        ──> 触发建议：ENABLE_PREPARE。
 [条件] 2. 无 rpy_files 且 can_generate_template 为真
        ──> 触发建议：GENERATE_TEMPLATE。
 [条件] 3. 无 rpy_files 且 属于 blocked_missing_template
        ──> 触发建议：INSTALL_SDK_GENERATE_TEMPLATE。
        ➔ 拦截返回：不继续往下运行。

========================================================================================
 优先级 3: 辅助索引与记忆库未预建 (Required)
----------------------------------------------------------------------------------------
 [条件] 1. source_index_needs_bootstrap == 'missing' / 'incomplete'
        ──> 触发建议：BOOTSTRAP_SOURCE_INDEX / BOOTSTRAP_SOURCE_INDEX_INCOMPLETE。
        ➔ 加入必需建议集合，继续评估其他上下文条件。
 [条件] 2. rag_needs_bootstrap 
        ──> 触发建议：BOOTSTRAP_RAG 或 BOOTSTRAP_RAG_OR_WARM_ON_BUILD。
        ➔ 加入必需建议集合，继续评估可选质量建议。

========================================================================================
 优先级 4: 可选质量优化建议 (Optional)
----------------------------------------------------------------------------------------
 [条件] 1. 补译量大 (pending >= 150) 且有旧译，但未开启 RAG
        ──> 触发建议：ENABLE_RAG_FOR_CONSISTENCY。
 [条件] 2. 全新项目 (has_tl 且无旧译) 且未开启原文索引
        ──> 触发建议：ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT。
        ➔ 与优先级 3 的累计结果合并；输出时必需建议在前、可选建议在后。

========================================================================================
 优先级 5: 环境已完全就绪 (Ready)
----------------------------------------------------------------------------------------
 [条件] 没有任何待处理项 
        ──> 建议列表返回空 []，GUI 显示环境就绪状态。
========================================================================================
```
