# 架构概览

本文描述当前 `main` 的维护者边界，按 2026-10-24 的 `main@b5a3f14` 核对。
操作步骤见[文档地图](README.md)，函数与测试入口见[代码路径索引](code_paths.md)；
历史设计及实施状态见[规划、合同与研究索引](plans/README.md)。

## 分层

### 核心包与入口

PR #563 将根目录核心模块归入五个包；根目录继续保留 CLI、runtime 和兼容入口。
CLI 保持行为事实来源，GUI 调用同一核心或同一 CLI。核心与 CLI 不依赖 GUI，
不安装 PySide6 仍可运行核心流程。

| 位置 | 职责 |
| --- | --- |
| `gemini_translate_batch.py`、`gemini_translate.py` | Batch / 耐久 Sync 命令与兼容同步入口 |
| `translator_runtime.py` | runtime 配置快照、SDK、计划和生产请求接线 |
| `foundation/` | 原子 I/O、配置、路径安全、doctor 和基础合同 |
| `context_engine/` | glossary、原文索引、RAG、Story Memory 与 embedding runtime |
| `providers/` | Provider 连接、ModelProfile、模型路由、迁移、能力与成本 |
| `quality_review/` | TranslationPlan、翻译单元、结构 / 质量检查、审校、提案与 Project Analysis |
| `execution/` | Batch 导出、耐久 Sync、外部工作包、用量与工作区管理 |
| `engine_adapters/` | 引擎发现、候选、coverage、occurrence、重定位与声明式写回 |
| `relation_analyzer/` | 可选关系与语义分析 |

### GUI 与测试目录

PR #564 按职责组织 GUI；启动入口仍为 `python -m gui_qt`。

| 目录 | 职责 |
| --- | --- |
| `gui_qt/settings/`、`gui_qt/workbench/` | 设置页、工作台页面及其协调与动作 |
| `gui_qt/workflows/` | 任务步骤、CLI 参数与运行结果编排 |
| `gui_qt/reports/` | 检查、进度、失败与订正结果的展示投影 |
| `gui_qt/dialogs/`、`gui_qt/widgets/` | 对话框与复用控件 |
| `gui_qt/workers/` | 后台操作与结果信号 |
| `gui_qt/runtime/` | CLI 子进程、安装、异步操作身份与生命周期 |
| `gui_qt/appearance/` | 主题、字体和图标 |

`gui_qt/app.py` 保留应用壳和宿主协调；项目 / 会话状态、共享文案、路径与诊断参考
仍在 GUI 包根部。模块迁移没有改变命令或配置格式。

`tests/` 按核心职责分包，GUI 测试位于 `tests/gui/`；共享辅助位于
`tests/support/`，原创夹具位于 `tests/fixtures/`。两套 runner 递归发现并以完整模块名
加载；GUI / CLI 分流沿用文件命名规则。运行方式见[测试目录](../tests/README.md)。

## 配置事实来源

- `translator_config.json` 保存非敏感项目 / 翻译设置；`foundation/config_store.py`
  提供原始 JSON、共用写锁与原子替换，GUI 通过 `ProjectState` 委托读取 / 保存。
- `model_routing` 存在时，`providers/model_routing_reader.py` 整体验证并使用该 section；
  无效或不支持的 schema 拒绝读取，不逐字段回退到旧配置。
- 缺少 `model_routing` 时，保留 `sync.*` / `batch.*` 的兼容路径。
  新任务冻结解析后的 profile / strategy / route，旧任务优先消费原冻结路由。
- 显式文件迁移由 `providers/model_config_migration.py` 的 preview / migrate / rollback
  完成。GUI 编辑迁移后的配置；存在旧字段时引导离线迁移，不建立第二套 GUI 迁移事务。
- API Key 由 `api_keys.json`、环境变量或 keyring 提供；项目配置只保存适用的凭据引用。
- 项目级 `project_context_settings.json` 保存 RAG / 索引 / 项目分析等覆盖值；全局默认
  仍在 `translator_config.json`，读取由 `foundation/project_context_settings.py` 统一处理。

迁移和兼容用法见[模型配置迁移](model_config_migration.md)；
schema 的设计边界见[#348 合同](plans/issue-348-model-routing-config-contract.md)。

## 请求计划与耐久执行

`providers/model_profile.py` 与路由 reader 解析模型、能力、策略与阶段，
`quality_review/translation_plan.py` 冻结 source、context、prompt / schema、路由和预算。
Sync 与 Gemini Batch 消费相同计划合同，传输与调度生命周期分别实现。

- 耐久 Sync：CLI `sync-start / resume / status / cancel / derive` 调用
  `execution/sync_run_service.py`；状态与机器错误由 `execution/sync_run_contracts.py`
  定义，SQLite 细节不暴露给 GUI。
- GUI 的 `gui_qt/workflows/sync_translation_workflow.py` 消费同一 CLI 信封；
  统一翻译页选择 profile 和执行方式，配置编辑由 `gui_qt/settings/profiles_page.py` 承接。
- `execution/sync_request.py` 执行共享请求与响应提取；service 不通过导入 CLI 执行请求，
  `gemini_translate_batch.run_sync_request` 保留依赖装配与单向转发。
- `SyncRequestRuntime` 在进程内绑定请求超时、Provider 配置和凭据回调。Key 不进入
  plan、数据库或 repr；resume 重新装配当前凭据，并保留 frozen route 与 freshness 检查。
- durable attempt 固定 `retry_attempts=1`、`allow_credential_rotation=False`；
  重试由外层执行器记录为新 attempt。新 run 冻结 executor policy，resume 使用已有 policy，
  非 durable 兼容路径保留自己的轮换边界。

操作与证据见[同步工作流](sync_workflow.md)及[Provider Smoke Matrix](provider_smoke_matrix.md)。

## 检查、预览与写回

Engine Adapter 提供 discovery、candidate inventory、coverage、occurrence、relocation、
translation validation 和声明式 writeback plan。公共层复核材料 / 结果 / 项目身份、
源快照、目标现值及计划，再交给既有原子事务；adapter 自身不写文件。

只有与当前 manifest / results 匹配的最近检查给出 `writeback_gate.decision=allow`
才能写回；需要绑定预览的路径还须复核 preview。普通质量报警默认不阻断，项目配置
提升为 blocker 的规则必须阻断。`--force` 不能绕过 stale check、源快照、结构 blocker
或 adapter plan 验证。检查与 apply 在各执行路径中的实现仍分别维护，不能只凭模型完成状态授权。

公共源 JSON schema 2 由 PR #568 合并，支持独立源 / 目标快照与数组路径。
`RpgMakerMvAdapter` 在显式材料窗口上提供 MV 叙事共享核心，复用该计划消费者与结构保护；
`execution/mv_sync.py` 将其接入共享 TranslationPlan、耐久 Sync 和现有双快照原子写回；
`engine_adapters/project_config.py` 负责自动识别、显式引擎／源目标／支持窗口解析，
`engine_adapters/project_setup.py` 提供 CLI / GUI 共用的准备状态与配置编辑，GUI 浏览表单调用后仍走同一 doctor，
`engine_adapters/rpgmaker_mv_fields.py` 管理默认关闭的系统术语与地图显示名白名单。
`execution/mv_preprocessing.py` 把同一源 occurrence 投影为关键词材料和既有 Source Index
记录，只将显式独立审阅的双语输入投影为既有 history 记录。它复用两种 JSON store 的
多步写锁／原子替换、embedding 身份和排名；当前材料／输入／数据摘要决定是否已准备。
耐久 Sync 继续由 ContextAssembler 裁剪并冻结检索；目标现译与 apply 不自动成为审阅批准。
CLI、GUI、doctor 和 preflight 消费同一准备状态；StoryMemory / Project Analysis 注入未开放。
字段角色与选择绑定同一 Adapter、审阅和任务身份。CLI、GUI 和 doctor
复用同一合同，Batch / 外部工作包保持未支持，真实原生验收仍单列。产品操作见
[MV 耐久 Sync](rpgmaker_mv_sync.md)，核心边界见
[Engine Adapter 手册](engine_adapter.md#rpgmaker-mv-a2)。

## Settings 页面与保存所有权

`gui_qt/settings/registry.py` 统一登记当前 11 个分类、页面身份、配置键和 lazy 映射。
`SettingsCoordinator` 负责补建、切页、load / collect / validate / reset、错误定位、
任务锁、dirty 基线和离开提示；它不依赖 Qt，也不持有文件写盘事务。

页面拥有控件、字段映射、局部校验和局部 worker；通过显式 callback 请求宿主执行动作，
不读取其他页面控件。关键落点如下：

| 位置 | 所有权 |
| --- | --- |
| `gui_qt/settings/page_contract.py` | `SettingsPage`、`SettingsIssue` 与页面动作协议 |
| `gui_qt/settings/profiles_page.py` | 统一模型 / Provider / strategy / route 编辑，调用 `providers/model_profiles_editor.py` |
| `gui_qt/settings/settings_schema.py`、`gui_qt/settings/field_widgets.py`、`gui_qt/settings/page_chrome.py` | 字段定义、控件工厂和共用布局 |
| `gui_qt/settings/save_apply.py` | 将收集快照应用到原始 JSON，保留未建模字段，不写盘 |
| `gui_qt/settings/leave_guard.py` | 启动任务、切项目、关闭与离开设置的保护文案 |
| `gui_qt/app.py` | 全局 route / header / sidebar、配置加载、两文件持久化与失败回滚、Qt 对话框和 shutdown |

普通配置由宿主执行 collect → validate → persist，并同步 dirty 基线。
API Key、keyring、工作区总表、`game_root` / `workspace_root` 和安装操作有各自即时
持久化边界；主题只即时预览，保存后才持久化。首次加载或切换分类不应覆盖未保存编辑。

### #528 Settings 内部兼容清理

LiteLLM worker / 目录 / 编辑状态和 Models thinking 状态由各自页面持有；
无生产调用者的宿主事件转发与平行副本已删除。保留接口仍有明确消费者：

- 控件别名与 `_settings_widget` 供加载、快照恢复、推荐值和布局使用；只读取已创建控件
  的查询不触发补建。
- LiteLLM / Models 薄委托服务保存、安装后刷新、密钥管理和关闭流程。
- `_litellm_cache` 由宿主注入并供密钥页读取；未加载 LiteLLM 页时使用已保存配置，
  不同步导入 LiteLLM。
- save extras 读取页面真实修改标志，后台线程清单读取页面实际 worker。

### 异步任务与历史合同

worker、QProcess 和线程池操作复用 `gui_qt/runtime/lifecycle.py` 与
`gui_qt/runtime/operation_identity.py`：结果绑定项目 / 配置 / 任务身份及代次，旧结果不得覆盖新任务。
取消后保留 worker 所有权到真实终态，GUI 线程不得用固定等待阻塞退出。

#202 的页面迁移与 #348 P3 集成都已交付。两文件写盘保留在宿主是当前所有权边界；
历史 Phase / target 描述见[#202 合同](plans/issue-202-settings-page-contract.md)和
[#348 合同](plans/issue-348-model-routing-config-contract.md)，不作为新增待办。
