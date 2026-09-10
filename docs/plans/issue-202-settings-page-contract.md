# #202 Phase A：Settings 页面契约与现状基线

> 状态：Phase A 文档已合并（PR #433，merge `773014b`）；Phase B 已合并（PR #434，
> merge `3db29ab`）。Phase C 已将 LiteLLM 页迁到独立 `SettingsPage`（PR #436）。
> Phase D 已将 10 个 Settings 页迁到独立 `SettingsPage`；dirty 基线、离开保护文案与 collect→persist
> 已由 coordinator 持有；collected 值 apply 在 Qt-free `save_apply.py`；legacy adapter 与旧 builder 已删除。
> 两文件写盘仍在 `MainWindow`。Epic 在全量回归与人工烟测完成前保持打开。
> 本文既是 Phase B 接入合同，也是实现索引；as-is 与 target 的差异逐项标注。
>
> 核验基线：Phase A 于 `main@2b93e43`；Phase B 基于 `main@773014b`，合并于 `main@3db29ab`。
>
> 关联：[#202 Epic](https://github.com/hu-border-collie/renpy-translation-lab/issues/202)、
> [#348 配置与迁移合同](issue-348-model-routing-config-contract.md)、
> [架构概览](../architecture.md)、[代码路径索引](../code_paths.md)。

## Phase A 范围（已交付）

- 只补维护者文档与说明性 docstring，不改变运行行为。
- 冻结 Phase B 需要的最小页面接口、字段所有权和宿主事件边界。
- 记录当前 10 个 Settings 页面、字段/即时持久化所有权、局部异步任务、lazy 构建与测试入口。
- 不定义 #348 的 `model_routing` schema、迁移语义或生产 resolver；Phase A 不实现任何页面。
- Phase B 的 as-built 状态见下文「Phase B 实现状态」；页面迁移见 Phase C/D 实现状态。

## 当前基线（as-is）

### 页面身份与 lazy 构建

事实来源是 `gui_qt/app.py`：

- `_SETTINGS_PAGE_SPECS`：10 页的 `(key, nav_label, builder)` 清单。
- `_SETTINGS_CONFIG_PAGE_KEYS`：参与 `translator_config` load/save/dirty 的页面集合。
- `_CONFIG_SNAPSHOT_KEYS_BY_PAGE`：各页面当前贡献的扁平 dirty 键。
- `_SETTINGS_LAZY_ATTR_TO_PAGE`：属性名到页面的 lazy 映射；`MainWindow.__getattr__` 只在测试或
  代码直接访问控件时补建对应页面。
- `_ensure_settings_page(key, populate=...)`：普通切页只补建目标页；目标页属于配置页时调用
  `_load_config_to_ui(pages={key})`。
- `_ensure_settings_pages_for_config()`：保存/重载/恢复默认值时补建所有配置页；已打开页面的
  编辑通过 `_current_config_ui_snapshot()` + `_restore_config_ui_snapshot()` 保留。

全局 `settings` shell route、应用 header/sidebar、任务锁和主题应用仍由 `MainWindow` 持有；
页内导航由 `settings_nav` + `settings_stack` 直接管理。

### 10 页现状与所有权

| key | 页签 | 页面实际读写的字段 | 持久化时机 | 当前 dirty 键来源 | 页面局部异步任务 |
|---|---|---|---|---|---|
| `workspace` | 项目列表 | 无 `translator_config` 字段；工作区总表与当前项目切换 | `workspace_root` / `game_root` 由 `ProjectState` 即时写入 `translator_config.json`；`games_registry.json` 由 `games_registry.py` 即时写入 | 无（不计入 Settings dirty） | `RegistryRefreshWorker`、`RegistryIngestWorker`（QThread）；工作区初始化对话框内 `SdkInstallWorker` |
| `project` | 项目 | `ADVANCED_SETTING_FIELDS` 的「项目与资源」「准备流程」两组（`game_root` 除外） | 点「保存设置」写入 `translator_config.json` | 当前无独立项；字段键落入 `advanced` 键集合（见 as-is 缺口） | 无 |
| `api_keys` | 密钥 | Gemini Key（`api_keys.json`）与 LiteLLM Provider Key（系统 keyring） | 对话框确认后即时写入，不进入 dirty | 无 | 无（keyring 调用同步执行） |
| `models` | 模型 | `sync.model` / `sync.litellm_model`（经 `write_sync_backend_models`）、`sync.rag.embedding_model`、`batch.model`、`batch.rag.embedding_model`、`batch.thinking_level` | 点「保存设置」写入 `translator_config.json` | `_CONFIG_SNAPSHOT_KEYS_BY_PAGE["models"]` | 无（下拉目录为同步内存数据） |
| `litellm` | LiteLLM | `sync.backend`、`sync.litellm_model`、`sync.custom_litellm_providers` | 配置项随保存写入；Provider Key 即时写 keyring；目录缓存即时写用户缓存；安装即时执行 | `_CONFIG_SNAPSHOT_KEYS_BY_PAGE["litellm"]` | `LiteLLMProviderCatalogWorker`、`LiteLLMModelCatalogWorker`、`LiteLLMVersionWorker`、`LiteLLMConnectionTestWorker`（QThread）；`OptionalFeatureInstallController`（QProcess pip） |
| `extensions` | 扩展 | 无 `translator_config` 字段；可选能力安装状态来自当前 Python 环境 | 安装完成即生效 | 无 | `OptionalFeatureInstallController`（关系分析器，QProcess pip） |
| `context` | 上下文 | `project_context_settings.json` 的项目级开关；`context_storage.location`；上下文/项目分析 advanced 字段 | 项目级开关与全局配置在同一次保存中写入；项目文件失败会回滚已写全局文件 | `_CONFIG_SNAPSHOT_KEYS_BY_PAGE["context"]`（含 4 个项目分析键） | 无（上下文库状态 Job 属于工作台 Context 页，不属于 Settings 页） |
| `appearance` | 外观 | `gui.theme` | 切换主题立即预览但 `persist=False`；点「保存设置」才写入；字体下载/安装即时 | `_CONFIG_SNAPSHOT_KEYS_BY_PAGE["appearance"]` | `FontInstallWorker`（QThread + 子进程） |
| `shortcuts` | 快捷键 | 只读目录 | 无 | 无 | 无 |
| `advanced` | 高级 | `ADVANCED_SETTING_FIELDS` 中除「项目与资源」「准备流程」和上下文主开关外的字段；模型目录扩展经 `write_model_catalog_extras` 写入 | 点「保存设置」写入 `translator_config.json`；`validate_advanced_settings` 失败时不写盘 | `_CONFIG_SNAPSHOT_KEYS_BY_PAGE["advanced"]`（当前为除 `game_root` 外的全部 advanced 键） | 无 |

即时持久化与普通配置保存的边界：

- **即时**：`api_keys.json`、LiteLLM keyring、`games_registry.json`、`workspace_root` /
  `game_root`、字体与可选能力安装、主题预览。
- **随「保存设置」提交**：`translator_config.json` 的模型/上下文/外观/高级字段，以及
  当前项目的 `project_context_settings.json`。
- `game_root` 虽然由 advanced 字段渲染，但实际由工作区/项目列表即时管理，不进入 dirty 基线。

### 当前加载、保存、dirty 与离开保护链

**加载**：`_load_config_to_ui(pages=...)` 每次调用 `ProjectState.load_translator_config()` 读取原始
JSON，按页面过滤后填充控件，并调用 `_update_config_ui_saved_snapshot(pages=...)`；部分加载只更新
所加载页面的 baseline 键。`_ensure_settings_pages_for_config()` 在保存/重载前补建全部配置页，
先把已打开页面的 UI 快照保存下来，全量加载后再恢复，避免补建其他页面覆盖未保存编辑。

**保存**：`_on_save_config()` 仍是入口；apply 已抽到 Qt-free `apply_collected_settings`：

1. 要求已有 `game_root`，补建配置页并 flush LiteLLM 下拉的延迟保存；
2. 有 coordinator 时 `SettingsCoordinator.save()` → host `persist(collect())`；`MainWindow.__new__`
   helper 走 `_widget_settings_collect()` + 同一 persist；
3. persist 读取原始 JSON 并深拷贝，再调用 `apply_collected_settings`（含
   `validate_advanced_settings`）；失败则聚焦对应页并中止，不写任何文件；
4. `ProjectState.save_translator_config(config)` 经 `config_store.write_json_object` 加共用写锁并原子
   替换；调用方已在原始对象上合并已知修改，未知字段保留；
5. `save_project_context_settings(game_root, flags)` 写当前项目的
   `project_context_settings.json`；项目文件失败时用原始配置回滚全局文件；
6. 可能经 `_sync_state_game_root_from_settings` 触发项目切换，最后刷新 UI 并把 dirty 基线
   更新为当前控件快照。

**dirty 与离开保护**：`_current_config_ui_snapshot()` 生成扁平键值快照，`_config_tab_has_unsaved_changes()`
与 `_config_ui_saved_snapshot` 比较；`_confirm_unsaved_config_before_workflow()`、
`_confirm_unsaved_config_before_registry_switch()`、`_confirm_unsaved_config_before_close()` 和
`_confirm_leave_config_tab()` 分别覆盖启动任务、切换项目、关闭窗口和离开设置页的
保存/放弃/取消三态。`_on_reload_config()` 是当前丢弃未保存修改的入口。

### 已知 as-is 缺口与处理状态

- **项目页单独打开不会填充字段**：Phase B 已修。`_load_config_to_ui` 现按 registry 键加载
  `project` 页的 advanced 字段；首次 materialize 即填入保存值。
- **dirty 键所有权不完整**：Phase B 已修。registry 强制一键一主；`advanced` 不再包含
  context 主开关，`project` 字段归 project 页。
- **补建页面被 preserve/restore 覆盖**：Phase B 已修。只保留已加载页面的快照，且恢复只回写
  快照中存在的 advanced 键。
- **页面不可独立构造**：已收敛。10 个 Settings 页均为独立 `SettingsPage`；legacy adapter 与
  `_build_settings_*_page` 已删除。
- **保存编排集中**：部分收敛。coordinator 已提供 load/collect/validate/reset/dirty 基线/离开保护文案/
  collect→persist；apply 在 `save_apply.py`；两文件写盘与 Qt 对话框仍在 `MainWindow`。
- **局部 worker 归属整窗**：部分收敛。LiteLLM worker 已由页面持有；字体 / 安装 / registry
  worker 仍由 `MainWindow` 属性和私有回调管理。
- **说明性 docstring 过时**：Phase A 已修。`gui_qt/__init__.py` / `gui_qt/app.py` 已区分
  QProcess、QThread/QThreadPool 与 GUI 本地动作。

## Phase B 接入合同（已冻结并落地最小接线；页面迁移属于 C/D）

目标目录与职责：

- `gui_qt/settings/page_contract.py`：`SettingsPage` Protocol、`SettingsIssue`、`SettingsPageActions`。
- `gui_qt/settings/registry.py`：页面 key / label / builder / 字段所有权登记，强制一键一主。
- `gui_qt/settings/coordinator.py`：页内导航、lazy materialize、load/collect/validate/dirty/save、
  离开保护、错误聚焦、任务锁记录与建页补发、项目切换失效。
- `MainWindow`：只保留应用壳、全局 `settings` route、header/sidebar、全局任务锁、runner/log、
  主题应用、顶层装配和 shutdown 协调。

### Page adapter 合同

```python
@dataclass(frozen=True)
class SettingsIssue:
    page_key: str
    field_key: str
    message: str
    severity: Literal["error", "warning"] = "error"

class SettingsPage(Protocol):
    page_key: str
    nav_label: str
    config_keys: frozenset[str]          # 本页唯一拥有的扁平 dirty 键
    immediate_action_ids: frozenset[str] # 声明不进入 dirty 的即时动作

    def load(self, snapshot: Mapping[str, object]) -> None: ...
    def collect(self) -> Mapping[str, object]: ...
    def validate(self) -> Sequence[SettingsIssue]: ...
    def reset(self) -> None: ...
    def focus_issue(self, issue: SettingsIssue) -> bool: ...
    def set_task_running(self, running: bool) -> None: ...

@dataclass(frozen=True)
class SettingsPageActions:
    """Coordinator-owned callbacks injected into a page (no MainWindow access)."""

    save: Callable[[], None] | None = None
    reload: Callable[[], None] | None = None
    navigate: Callable[[str], None] | None = None
    run_immediate: Callable[[str, Mapping[str, object]], bool] | None = None
    show_status: Callable[[str], None] | None = None
```

语义冻结：

- `load(snapshot)`：只把 `snapshot` 中属于 `config_keys` 的值推入本页控件；不读磁盘、不读其他页
  控件、不修改配置对象、不弹窗。首次 materialize 必须调用一次。
- `collect()`：只返回本页拥有的键值；不写盘、不弹窗。coordinator 对未声明键或重复所有者报错。
- `validate()`：纯页面校验，返回 `SettingsIssue` 列表；禁止弹窗、禁止写盘。共享字段
  （advanced 范围、URL、数值范围等）由 coordinator 调 `validate_advanced_settings` 复核。
- `reset()`：丢弃本页未保存编辑，回到最近一次成功 `load` / 保存的基线；不写盘。
- `focus_issue(issue)`：聚焦、滚动或标注 `field_key` 对应控件，成功返回 `True`；coordinator 在
  校验失败后先切到 `issue.page_key` 再调用它。
- `set_task_running(running)`：响应全局任务锁，禁用会改变配置或触发长任务的控件；不管理锁本身。
  Coordinator 记录当前 `running`，并在 `ensure_page` 新建页面后立即补发，避免任务运行中
  首次打开的设置页控件仍可编辑。
- 有局部 worker 的页面必须复用 #297 设施：operation identity、retired ownership 到真实
  `finished`、`ShutdownParticipant` 注册、禁止 GUI 线程固定等待、丢弃 stale result。Phase B
  先用 adapter 兼容未迁移页面；Phase C 首个 worker 页面必须落地生命周期接入，不允许另建框架。

### 宿主事件边界

Coordinator → 页面（只通过上述方法）：

- `load`：配置加载、保存成功、项目切换后的重新填充。
- `reset`：放弃修改、重载、项目切换或页面销毁前。
- `set_task_running`：全局任务锁变化；新建页面时补发当前锁。
- `focus_issue`：校验失败后的错误聚焦。

页面 → 宿主（`SettingsPageActions`，由 coordinator 注入；页面不得直接调用 `MainWindow` 私有方法）：

- `save()` / `reload()`：请求唯一保存事务或丢弃重载。
- `navigate(page_key)`：跨页跳转（如项目页跳项目列表）。
- `run_immediate(action_id, payload)`：执行已声明的即时动作（密钥、keyring、registry、安装、主题预览）。
- `show_status(message)`：非阻塞反馈；错误对话框仍由宿主统一处理。

页面禁止：读取其他页控件、直接调用 `ProjectState.save_translator_config` / `config_store`、
在 `load` / `collect` / `validate` 中弹窗、自建任务锁或第二套 dirty/save 状态。

### Coordinator 职责与保存事务

- 注册与导航：registry 记录页面身份和唯一字段所有权；`activate(key)` 只构建目标页；普通切页
  不触发全量重载或主题重应用。
- `load(pages=None)`：一次读取原始 JSON，构造扁平快照，只调用选中页面的 `load`，并只更新这些
  页面拥有的 baseline 键。
- `collect()` / `dirty()`：合并各页 `collect()`；与 baseline 比较时只比较页面拥有的键。
- `save()`：沿用当前唯一事务语义（读取原始对象 → 收集/校验 → 原子写全局配置 → 写项目级开关 →
  失败回滚 → 更新 baseline），页面不得自行整份保存配置。
- `reset()`：调用页面 `reset()` 后按需重新 `load`，作为“放弃修改”的唯一实现。
- `project_changed`：统一失效旧项目状态，调用页面 `reset()`；页面局部 worker 通过 operation
  identity 判断 stale，旧结果不得覆盖新项目。
- 离开保护：保存 / 放弃 / 取消三态文案与行为保持现状；全局 `settings` route 仍属于应用壳。

## #202 / #348 所有权

| 关注点 | 唯一所有者 | 消费者 |
|---|---|---|
| Settings page contract / registry / coordinator | #202（`gui_qt/settings/`） | 各 Settings 页面、#348 |
| 页面控件、字段读写映射、局部校验与动作 | #202 各页面 | coordinator |
| load/collect/validate/dirty/save/离开保护编排 | #202 coordinator | 页面 adapter、#348 |
| 原始 JSON、unknown-field preservation、原子保存、共用写锁 | `ProjectState` / `config_store` | #202 coordinator、#348 migrator |
| `model_routing` schema、validator、migrator、rollback | #348 | CLI、GUI Settings |
| ModelProfile / ExecutionStrategy / TaskRoute 解析 | #348 | TranslationPlan、Sync/Batch 服务 |
| Model Profiles 产品流程与统一翻译页 | #348 | 复用 #202 contract；不得另建配置缓存或保存入口 |
| run 状态与恢复 | #347 `SyncRunService` / RunSnapshot | #348 P3 统一页 |
| `check → bound preview → apply` | 现有公共安全层 | 所有翻译写回入口 |

完整边界见 [#348 P0 合同的「#202 / #348 所有权」](issue-348-model-routing-config-contract.md#202--348-所有权)。
`#202` 不定义第二套模型迁移语义；`#348` 不复制 `MainWindow._on_save_config`。

## Phase B 实现状态（已合并）

已落地：

- `gui_qt/settings/page_contract.py`：`SettingsPage` Protocol、`SettingsIssue`、`SettingsPageActions`。
- `gui_qt/settings/registry.py`：10 页 `SettingsPageSpec`、唯一配置键所有权、lazy 属性映射；
  `build_default_registry()` 在 `MainWindow` 构造时校验一键一主。
- `gui_qt/settings/coordinator.py`：`ensure_page` / `activate` / `load` / `collect` / `validate` /
  `reset` / `focus_issue` / `set_task_running`（含新建页补发） / `has_unsaved_changes`，不依赖 Qt。
- `gui_qt/settings/legacy.py`：已删除。十页均经 `_create_*_settings_page` 构造独立 `SettingsPage`。
- `MainWindow`：页面清单、dirty 键和 lazy 映射改由 registry 提供；`_ensure_settings_page` /
  `_on_settings_nav_row_changed` 经 coordinator 补建与切换；项目页首次打开会加载其 owned 字段；
  `_ensure_settings_pages_for_config` 只保留已加载页面的快照，且 `_restore_config_ui_snapshot`
  只回写快照中存在的 advanced 键，避免补建页面被推荐值覆盖。
- 测试：`tests.test_settings_page_contract`、`tests.test_settings_registry`、
  `tests.test_settings_coordinator`（纯 Python）与 `tests.test_gui_settings_coordinator`（GUI 集成）。

仍未落地（Phase D 收口）：

- 10 页均已迁出独立 `SettingsPage`；dirty 基线、离开保护文案、collect→persist 与 Qt-free apply 已落地；
  旧 builder / legacy adapter 已删除；两文件写盘与 Qt 对话框仍在 `MainWindow`。

Phase C 已落地：

- `gui_qt/settings/litellm_page.py`：`LiteLLMSettingsPage` 可脱离 `MainWindow` 构造，实现
  `load/collect/validate/reset/focus_issue/set_task_running`。磁盘上的
  `custom_litellm_providers` 完整交给 `custom_provider_registry` 校验，不得先过滤非法条目再当
  作有效配置保存；`collect()` / `reset()` 基线是键值对元组快照，加载时先还原成对象，且
  `reset()` 仍清除 modified；`load(..., restore=True)` 恢复未保存编辑时保留 `modified`，避免只打开
  LiteLLM 页删除最后一项后首次保存仍写回磁盘旧列表。
- 目录/版本/连接测试/warmup worker 由页面持有；取消后 retired 到真实 `finished`，warmup 仍使用
  模块级 retired set。连接测试继续用 `litellm_connection_identity` 丢弃 stale result。
- 凭据对话框、LiteLLM 安装控制器、密钥页下拉与模型页 Gemini 下拉 gating 仍由宿主回调提供。
- 测试：`tests.test_settings_litellm_page`（独立构造）与既有 `test_gui_litellm_*`。

Phase D（进行中，十页已迁出；apply 已抽到 `save_apply.py`，写盘仍待完全收口）：

- `gui_qt/settings/page_chrome.py`：迁移页共用的 Settings 滚动页/表单 chrome。
- `gui_qt/settings/field_widgets.py`：bool/int/float/str/text/list/json 字段控件工厂。
- `gui_qt/settings/gemini_catalog_widgets.py`：Gemini 目录 extras 与轮换 checklist 控件。
- `gui_qt/settings/models_page.py`：`ModelsSettingsPage` 可脱离 `MainWindow` 构造，拥有 Gemini
  同步/批量模型与思考程度下拉；目录 extras 仍由宿主 `set_catalog` 注入，保存仍走
  `MainWindow._on_save_config`。保存补建其他页时须保留 `_batch_thinking_user_changed`，否则显式
  「不启用」思考在首次保存时不会写入空 `thinking_level`。
- `gui_qt/settings/project_page.py`：`ProjectSettingsPage` 可脱离 `MainWindow` 构造，拥有
  「项目与资源」「准备流程」字段；`game_root` 只读展示，SDK 浏览/查找/下载仍由宿主对话框与
  `SdkInstallWorker` 执行。
- `gui_qt/settings/context_page.py`：`ContextSettingsPage` 可脱离 `MainWindow` 构造，拥有项目级
  RAG/索引/分析开关与上下文主开关键；保存仍走 `MainWindow._on_save_config`。
- `gui_qt/settings/advanced_page.py`：`AdvancedSettingsPage` 可脱离 `MainWindow` 构造，拥有
  剩余 advanced 字段、Gemini 目录扩展与模型轮换清单；保存与 `validate_advanced_settings`
  仍走 `MainWindow._on_save_config`。
- `gui_qt/settings/appearance_page.py`：`AppearanceSettingsPage` 可脱离 `MainWindow` 构造，拥有
  主题下拉；切换立即预览但 `persist=False`，字体下载/安装仍由宿主 `FontInstallWorker` 执行。
- `gui_qt/settings/shortcuts_page.py`：`ShortcutsSettingsPage` 可脱离 `MainWindow` 构造；只读目录，
  无 translator_config 字段，导航行由宿主 shell IA 注入。
- `gui_qt/settings/api_keys_page.py`：`ApiKeysSettingsPage` 可脱离 `MainWindow` 构造；Gemini /
  LiteLLM 密钥 chrome 由页面持有，无 translator_config 字段；对话框、keyring 写入与状态刷新
  仍由宿主回调提供。
- `gui_qt/settings/extensions_page.py`：`ExtensionsSettingsPage` 可脱离 `MainWindow` 构造；
  关系分析器 chrome 由页面持有，无 translator_config 字段；`OptionalFeatureInstallController`
  与 pip 安装仍由宿主执行。
- `gui_qt/settings/workspace_page.py`：`WorkspaceSettingsPage` 可脱离 `MainWindow` 构造；
  嵌入 `GamesRegistryPanel`，无 translator_config 字段；刷新/导入 worker 仍在面板内，
  切换项目与 `workspace_root` 写入仍由宿主回调提供。
- `gui_qt/settings/leave_guard.py`：未保存离开保护四套文案；coordinator 持有 dirty 基线并在 dirty 时返回 prompt，宿主弹出 Qt 对话框。
- `gui_qt/settings/save_apply.py`：Qt-free `apply_collected_settings`；coordinator `save()` 经 host persist 调用，两文件写盘仍在 `MainWindow`。
- 旧 `_build_settings_*_page`、`_legacy_settings_page_*`、MainWindow chrome helpers 与
  `gui_qt/settings/legacy.py` 已删除；registry `builder_name` 指向 `_create_*_settings_page`。
- LiteLLM 选中时禁用 Gemini 同步模型下拉的跨页 gating 仍由宿主调用
  `set_gemini_sync_allowed`，避免 `set_task_running(False)` 把该控件重新点亮。
- 测试：`tests.test_settings_models_page`、`tests.test_settings_project_page`、
  `tests.test_settings_context_page`、`tests.test_settings_advanced_page`、
  `tests.test_settings_appearance_page`、`tests.test_settings_shortcuts_page`、
  `tests.test_settings_api_keys_page`、`tests.test_settings_extensions_page`、
  `tests.test_settings_workspace_page`（独立构造）与既有
  `test_gui_app_config` / `test_gui_settings_coordinator` / `test_gui_settings_context_primary` /
  `tests.test_settings_save_apply`。

## Phase B 验收映射

- [x] `gui_qt/settings/` 下的 contract / registry / coordinator 可脱离 `MainWindow` 测试。
- [x] 10 页登记后一键一主；`project`、`api_keys` 的字段/即时动作所有权明确，advanced 与 context
      不再共享上下文主开关键。
- [x] 普通切页只构建并填充目标页；其他页面 materialize 不覆盖已打开页面的未保存编辑。
- [x] coordinator 的 load/collect/validate/reset/dirty/save/错误聚焦有独立测试。
- [x] 过渡期保存仍只有一份总事务；页面不得直接写 `translator_config.json` 或
      `project_context_settings.json`。
- [x] #348 可用一个最小 Model Profiles adapter 示例或测试接入 `load/collect/validate/reset`。
- [x] 全局 shell route、键盘/焦点可达性和导航防裁切行为无回归。

## 相关测试入口

- contract / registry / coordinator（纯 Python）：`tests.test_settings_page_contract`、
  `tests.test_settings_registry`、`tests.test_settings_coordinator`。
- GUI 集成：`tests.test_gui_settings_coordinator`。
- 布局 / 导航 / 配置：`tests.test_gui_settings_layout`、`tests.test_gui_shell_navigation`、
  `tests.test_gui_app_config`、`tests.test_gui_settings_context_primary`、
  `tests.test_gui_settings_schema`。
- LiteLLM：`tests.test_settings_litellm_page`（独立构造）、`tests.test_gui_litellm_settings_page`、
  `tests.test_gui_litellm_settings`、`tests.test_gui_litellm_worker`、`tests.test_gui_litellm_install`、
  `tests.test_litellm_catalog_cache`、`tests.test_litellm_provider_config`。
- 模型页：`tests.test_settings_models_page`（独立构造）、`tests.test_gui_settings_coordinator`、
  `tests.test_gui_app_config`。
- 项目页：`tests.test_settings_project_page`（独立构造）、`tests.test_gui_settings_coordinator`。
- 上下文页：`tests.test_settings_context_page`（独立构造）、`tests.test_gui_settings_context_primary`、
  `tests.test_gui_settings_coordinator`。
- 高级页：`tests.test_settings_advanced_page`（独立构造）、`tests.test_gui_settings_coordinator`。
- 外观页：`tests.test_settings_appearance_page`（独立构造）、`tests.test_gui_settings_coordinator`、
  `tests.test_gui_app_config`。
- 快捷键页：`tests.test_settings_shortcuts_page`（独立构造）、`tests.test_gui_settings_coordinator`、
  `tests.test_gui_settings_layout`。
- 密钥页：`tests.test_settings_api_keys_page`（独立构造）、`tests.test_gui_settings_coordinator`、
  `tests.test_gui_litellm_settings_page`。
- 扩展页：`tests.test_settings_extensions_page`（独立构造）、`tests.test_gui_settings_coordinator`、
  `tests.test_gui_optional_feature_install`。
- 项目列表页：`tests.test_settings_workspace_page`（独立构造）、`tests.test_gui_settings_coordinator`、
  `tests.test_gui_games_registry_panel_layout`、`tests.test_gui_startup_perf`。
- 生命周期与 worker：`tests.test_gui_lifecycle`、`tests.test_gui_operation_identity`、
  `tests.test_gui_optional_feature_install`、`tests.test_gui_font_worker`、
  `tests.test_gui_games_registry_worker`。
- 配置持久化：`tests.test_gui_project_state`、`tests.test_project_context_settings`、
  `tests.test_gui_theme_helpers`。
- 全量 GUI / CLI 回归：`python -B tests/run_gui_tests.py -q`、`python -B tests/run_cli_tests.py -q`。

## 非目标

- Phase A 未创建 `gui_qt/settings/`；Phase B 只创建 contract/registry/coordinator/legacy adapter，
  不迁移任何页面。
- Phase B 不把保存事务、dirty 基线与离开保护搬出 `MainWindow`（Phase D 收口）。
- 不改变 CLI/workflow、manifest、默认模型、执行策略或 `check → bound preview → apply` 合同。
