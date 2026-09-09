# #202 Phase A：Settings 页面契约与现状基线

> 状态：Phase A 文档已合并（PR #433，merge `773014b`）；Phase B 已合并（PR #434，
> merge `3db29ab`）。Phase C 已将 LiteLLM 页迁到独立 `SettingsPage`；Phase D 的其余页面
> 迁移与保存/dirty/离开保护收口仍未开始，`MainWindow` 仍持有唯一保存事务。
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
- Phase B 的 as-built 状态见下文「Phase B 实现状态」；页面迁移（C/D）仍未开始。

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

**保存**：`_on_save_config()` 是当前唯一总事务入口：

1. 要求已有 `game_root`，补建配置页并 flush LiteLLM 下拉的延迟保存；
2. `state.load_translator_config()` 取原始对象并深拷贝一份原始配置；
3. 从控件收集模型、后端、上下文、主题与 advanced 值；
4. `validate_advanced_settings(...)` 失败则聚焦高级页并中止，不写任何文件；
5. `ProjectState.save_translator_config(config)` 经 `config_store.write_json_object` 加共用写锁并原子
   替换；调用方已在原始对象上合并已知修改，未知字段保留；
6. `save_project_context_settings(game_root, flags)` 写当前项目的
   `project_context_settings.json`；项目文件失败时用原始配置回滚全局文件；
7. 可能经 `_sync_state_game_root_from_settings` 触发项目切换，最后刷新 UI 并把
   `_config_ui_saved_snapshot` 更新为当前控件快照。

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
- **页面不可独立构造**：仍未消除。10 页当前仍经 `LegacySettingsPageAdapter` 调用
  `MainWindow` builder，页面本身依赖整窗私有状态；Phase C 起逐页迁移为独立 `SettingsPage`。
- **保存编排集中**：部分收敛。coordinator 已提供 load/collect/validate/reset/dirty/错误聚焦合同，
  但唯一保存事务、dirty 基线与离开保护仍在 `MainWindow`；Phase D 继续收口。
- **局部 worker 归属整窗**：仍未消除。LiteLLM / 字体 / 安装 / registry worker 仍由 `MainWindow`
  属性和私有回调管理；Phase C 起按 #297 合同迁到页面。
- **说明性 docstring 过时**：Phase A 已修。`gui_qt/__init__.py` / `gui_qt/app.py` 已区分
  QProcess、QThread/QThreadPool 与 GUI 本地动作。

## Phase B 接入合同（已冻结并落地最小接线；页面迁移属于 C/D）

目标目录与职责：

- `gui_qt/settings/page_contract.py`：`SettingsPage` Protocol、`SettingsIssue`、`SettingsPageActions`。
- `gui_qt/settings/registry.py`：页面 key / label / builder / 字段所有权登记，强制一键一主。
- `gui_qt/settings/coordinator.py`：页内导航、lazy materialize、load/collect/validate/dirty/save、
  离开保护、错误聚焦与项目切换失效。
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
- 有局部 worker 的页面必须复用 #297 设施：operation identity、retired ownership 到真实
  `finished`、`ShutdownParticipant` 注册、禁止 GUI 线程固定等待、丢弃 stale result。Phase B
  先用 adapter 兼容未迁移页面；Phase C 首个 worker 页面必须落地生命周期接入，不允许另建框架。

### 宿主事件边界

Coordinator → 页面（只通过上述方法）：

- `load`：配置加载、保存成功、项目切换后的重新填充。
- `reset`：放弃修改、重载、项目切换或页面销毁前。
- `set_task_running`：全局任务锁变化。
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
  `reset` / `focus_issue` / `set_task_running` / `has_unsaved_changes`，不依赖 Qt。
- `gui_qt/settings/legacy.py`：`LegacySettingsPageAdapter` 把未迁移页面接入 coordinator；
  页面仍由 `MainWindow` builder 构建，字段读写委托宿主，旧保存事务不变。
  `coordinator.load(snapshot)` 经 `_restore_config_ui_snapshot` 应用内存快照（不读盘）；
  `reset()` 后页面仍视为已加载；`focus_issue()` 会先补填未访问页面再聚焦。
- `MainWindow`：页面清单、dirty 键和 lazy 映射改由 registry 提供；`_ensure_settings_page` /
  `_on_settings_nav_row_changed` 经 coordinator 补建与切换；项目页首次打开会加载其 owned 字段；
  `_ensure_settings_pages_for_config` 只保留已加载页面的快照，且 `_restore_config_ui_snapshot`
  只回写快照中存在的 advanced 键，避免补建页面被推荐值覆盖。
- 测试：`tests.test_settings_page_contract`、`tests.test_settings_registry`、
  `tests.test_settings_coordinator`（纯 Python）与 `tests.test_gui_settings_coordinator`（GUI 集成）。

仍未落地（Phase D）：

- 其余 9 页仍以 `LegacySettingsPageAdapter` 运行。
- `MainWindow` 仍持有唯一保存事务、dirty 基线与离开保护；coordinator 只提供合同并委托宿主。

Phase C 已落地：

- `gui_qt/settings/litellm_page.py`：`LiteLLMSettingsPage` 可脱离 `MainWindow` 构造，实现
  `load/collect/validate/reset/focus_issue/set_task_running`。
- 目录/版本/连接测试/warmup worker 由页面持有；取消后 retired 到真实 `finished`，warmup 仍使用
  模块级 retired set。连接测试继续用 `litellm_connection_identity` 丢弃 stale result。
- 凭据对话框、LiteLLM 安装控制器、密钥页下拉与模型页 Gemini 下拉 gating 仍由宿主回调提供。
- 测试：`tests.test_settings_litellm_page`（独立构造）与既有 `test_gui_litellm_*`。

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
