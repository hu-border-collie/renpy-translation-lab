# 环境检查智能建议机制

文档地图：[docs/README.md](README.md)

本文档说明 `doctor` / GUI「环境检查」应如何根据项目事实生成建议，包括设计原则、建议等级、优先级、数据结构和测试要求。

## 目标

环境检查只在用户确实需要处理某件事时显示建议：

- 阻断问题应说明为什么不能继续，以及首先要做什么。
- 已启用功能缺少前置数据时，应给出可执行的准备动作。
- 可选质量优化必须明确标记为可选，不能伪装成错误。
- 「可以开始翻译」「已经基本译完」「当前无待译条目」属于状态，不包装成建议。

`doctor` 仍是只读检查：它负责报告事实和建议，不生成翻译文件、不预建上下文库，也不调用 Gemini。建议对应的动作由用户在 GUI 或 CLI 中显式触发。

GUI 文案入口名称（须与界面一致）：

- 预建记忆库 / 原文索引 → 左侧 **「上下文库」**
- 启用开关 → **「设置 · 上下文」** 后 **保存设置**
- 开始批量翻译 → 左侧 **「批量翻译」**
- API 密钥 → **「设置 · 密钥」**

## 不穷举所有状态组合

建议系统不应为每一种原始字段组合编写独立分支。例如，无须创建一个名为「work 正常 + 模板存在 + 240 条待译 + 有旧译 + 索引完整 + RAG 关闭」的状态。

正确做法是分三层处理：

```text
原始检查结果
  -> 派生事实
  -> 按优先级匹配有限的建议规则
  -> 输出建议，或在没有待处理事项时返回空列表
```

需要穷举的是有限的「处理动作」和规则冲突，而不是所有字段的笛卡尔积。

## 输入事实与派生事实

建议规则可以使用以下原始事实：

- 布局：当前路径、work 目录、original/game 是否存在。
- 模板：TL 目录、翻译文件、Ren'Py SDK、自定义模板命令是否可用。
- 工作量：待译条目数、待译文件数、翻译块和原文注释数量。
- 翻译阶段：是否已有旧译、是否属于全新初译或增量补译。
- 上下文系统：RAG、原文索引是否启用，store 是否存在，记录或片段是否完整。
- 项目资产：术语表和风格设定是否存在、路径是否匹配当前项目。
- 运行条件：API 密钥是否配置。

原始字段应先归一化为有业务含义的派生事实：

```python
work_ready
template_ready
has_pending_translation
has_existing_translations
pending_is_minor
source_index_needs_bootstrap
rag_needs_bootstrap
```

派生事实集中计算，避免不同建议规则分别解释同一个计数或路径字段。

## 建议等级与优先级

### 1. 阻断问题

不处理就无法继续当前流程，例如：

- 当前路径应切换到 work 目录。
- work 目录不存在或为空。
- 缺少翻译模板，且当前无法生成。

阻断问题使用最高优先级，并给出唯一、明确的首要动作。

### 2. 必需准备

目标功能已经启用，但其前置数据未准备完成，例如：

- 原文索引已启用，但尚未建立或预建不完整。
- RAG 已启用，但记忆库为空，且不会在 build 时自动补建。

这表示按当前配置继续前需要处理，但不等同于项目布局损坏。

### 3. 可选优化

不影响流程正确性，但可能改善翻译质量，例如：

- 补译量较大且已有历史译文，可以考虑启用 RAG 以提高术语一致性。
- 全新项目可以考虑启用原文索引以增加剧情上下文。

文案必须包含「可选」或等价表述。可选建议不能把绿色检查结果升级成警告。

### 4. 正常状态

以下结果只作为状态和下一步说明，不进入建议列表：

- 翻译环境已就绪，可以开始批量翻译。
- 增量补译环境已就绪。
- 项目已基本译完，剩余待译量很小。
- 当前没有待译条目。

推荐优先级为：

```text
阻断问题 > 必需准备 > 可选优化 > 正常状态
```

布局 / 模板类阻断问题应抑制上下文类建议（例如 work 目录不存在时，不应同时要求预建原文索引）。
在布局与模板已就绪后，**必需准备与可选优化并列输出**（先必需、后可选），不再因「索引未建」而吞掉「建议启用 RAG」等可选提示。

## 建议结构

当前实现使用稳定的 `code` 和 `params` 表示可执行建议，并用独立的 `workflow_state` 表示正常翻译阶段；GUI 通过代码集合区分必需建议与可选建议。后续若需要把严重程度、原因和动作写入协议，可以演进为：

```json
{
  "code": "bootstrap_source_index",
  "severity": "required",
  "priority": 80,
  "reason": "source_index_enabled_but_missing",
  "params": {},
  "action": "bootstrap-source-index"
}
```

字段含义：

- `code`：稳定机器标识，业务判断不能依赖展示文案。
- `severity`：`blocking`、`required` 或 `optional`。
- `priority`：同一次检查命中多个规则时的排序依据。
- `reason`：触发建议的事实，便于日志、测试和问题定位。
- `params`：路径等动态数据。
- `action`：GUI 操作或 CLI 子命令的稳定标识。

用户文案由展示层根据 `code` 和参数生成。CLI 不应把英文句子当协议，GUI 也不应通过搜索文案关键词判断严重程度。

## 匹配示例

假设检查结果为：

```text
work 目录：正常
翻译模板：存在
待译条目：240 条
已有历史译文：是
原文索引：已启用且完整
RAG：未启用
```

规则可以写成：

```python
# 布局 / 模板阻断：仍只处理并返回（无法继续时不堆上下文建议）
if not work_ready:
    recommend("bootstrap_work", severity="blocking", priority=100)
    return
if not template_ready:
    recommend("generate_template", severity="blocking", priority=90)
    return

# 上下文层：必需准备与可选建议并列输出（必需在前，可选在后）
if source_index_needs_bootstrap:
    recommend("bootstrap_source_index", severity="required", priority=80)
if rag_needs_bootstrap:
    recommend("bootstrap_rag", severity="required", priority=70)
elif has_existing_translations and pending_count >= 150 and not rag_enabled:
    recommend("enable_rag_for_consistency", severity="optional", priority=30)
if new_project and not source_index_enabled:
    recommend("enable_source_index_for_new_project", severity="optional", priority=20)
```

示例：work 正常、模板就绪、待译 240、有旧译、原文索引已启用但未建、RAG 未启用时，可同时得到：

1. 必需准备：预建原文索引
2. 可选优化：启用记忆库以提高术语一致性

GUI 主文案通常取**第一条**（最高优先的必需准备）；其余建议仍出现在事实列表中。

如果用户随后启用了 RAG 但尚未预建，则在索引已就绪的前提下会出现：

> 需要准备：记忆库已启用但尚未建立，请先预建记忆库。

如果所有必需条件均满足且无可选提示，建议列表为空，摘要只显示环境已就绪（`workflow_state`）。

## 当前代码边界

相关代码职责如下：

- `gemini_translate_batch.py::collect_doctor_recommendations()`：根据 doctor report 选择建议代码。
- `gemini_translate_batch.py::collect_doctor_workflow_state()`：生成不进入建议列表的正常流程状态。
- `doctor_recommendations.py`：稳定建议代码、参数和 CLI 序列化兼容。
- `gui_qt/doctor_report.py`：把 report 转换成 GUI 摘要、事实和状态。
- `gui_qt/user_copy.py`：根据稳定代码生成中文文案和严重程度表现。

`start_pending_batch`、`start_incremental_batch`、`substantially_complete` 和 `no_pending_lines` 现在作为 `workflow_state` 输出，不进入建议列表；对旧版 CLI 建议行仍保留解析兼容。

当存在**必需准备**建议（例如 `bootstrap_source_index` / `bootstrap_rag`）时，`workflow_state` 应留空，避免 CLI 同时出现「可开始翻译」与「必须先准备」。可选优化（`bootstrap_rag_or_warm_on_build`、`enable_rag_for_consistency`、`enable_source_index_for_new_project`）不抑制 `workflow_state`。

## 文案要求

每条建议应回答三个问题：

1. 检测到了什么事实？
2. 这是否会阻止继续？
3. 用户接下来可以执行什么动作？

避免以下问题：

- 只说「建议优化」，但不说明原因。
- 对可选功能使用「必须」「错误」等阻断措辞。
- 在摘要和事实列表中重复同一句建议。
- 检查已经通过，仍固定显示一条「建议开始翻译」。
- 文案指向不存在的 GUI 按钮或错误的 CLI 子命令。

## 测试要求

不需要覆盖全部字段组合，但必须覆盖：

- 每条规则至少一个命中用例和一个不命中用例。
- 数量阈值两侧及等于阈值时的边界值。
- 多条规则同时满足时的优先级和抑制关系。
- work、模板、索引、RAG 等关键阶段的独立回归。
- 所有必需条件满足时，建议列表必须为空。
- 可选优化不能把 `ready` 状态升级为警告。
- CLI 序列化与 GUI 本地化使用相同稳定代码。
- 未识别的历史建议代码能够安全降级。

建议矩阵应优先测试业务不变量，而不只是具体字符串：

```text
返回 blocking 建议时，不得同时显示“可以开始翻译”。
返回 bootstrap_source_index 时，source index 必须已启用且未就绪。
没有真实待处理动作时，recommendations 必须为空。
```

## 验收标准

- 建议与当前项目事实一致。
- 最先显示的是用户当前确实可以执行的动作。
- 必需项和可选项在状态、颜色与文案上可区分。
- 正常状态不会为了填充界面强行生成建议。
- 同一事实不会在摘要和详情中机械重复。
- CLI、GUI 和自动化测试对建议代码的解释一致。
