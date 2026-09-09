# 架构概览

本文描述当前 `main` 的维护者边界。面向使用者的操作步骤以快速开始和工作流文档为准；
规划中的结构以 `docs/plans/` 为准。

> 2026-09-09（#202 Phase A）核对：`gui_qt/settings/` 与 `SettingsCoordinator` **尚未实现**。
> 本文明确区分 as-is 与 target；target 结构只作为 Phase B 接入合同，不代表已交付。

## 分层

1. **CLI 与 GUI 入口**
   - `gemini_translate.py`、`gemini_translate_batch.py` 提供 CLI 和机器 JSON 合同。
   - `gui_qt/app.py` 是当前桌面壳；工作台页面与 workflow 负责把用户动作映射到共用服务或 CLI。
   - CLI 核心保持可脱离 GUI 导入；CLI 不得反向导入 GUI，GUI 测试显式导入并在缺少 PySide6 时跳过。
2. **项目与配置**
   - `translator_runtime.py` 从代码默认值、`api_keys.json`、环境变量和
     `translator_config.json` 构造运行时快照。
   - `gui_qt/project_state.py` 负责 GUI 项目路径，原始 JSON 读取/保存委托给共用 `config_store.py`。
   - `gui_qt/settings_schema.py` 描述高级设置字段；当前 Settings 总体 load/save/dirty
     编排仍位于 `gui_qt/app.py`。
   - 当前项目的 `project_context_settings.json` 保存 RAG / 原文索引 / 项目分析开关；
     全局默认值仍在 `translator_config.json`。
3. **模型路由与请求计划**
   - `model_profile.py` 把现有 `sync.*` / `batch.*` 只读解析为 provider-neutral
     ModelRoutingPlan。它的 `primary`/`batch` 等 ID 是兼容 slot，不是长期用户 ID。
   - `translation_plan.py` 冻结 source snapshot、模型路由、上下文、prompt/schema 和预算。
   - #348 的 schema-v1 合同位于 `model_routing_config.py`，目前尚未接入生产读取。
   - P1 的 `model_routing_reader.py` / `model_routing_migration.py` 提供离线兼容读取和迁移候选，
     `model_routing_migration_store.py` 通过共用 config store 完成显式暂存/回滚；生产激活属于 P2。
4. **执行与耐久状态**
   - CLI 耐久路径：`sync-start` → `sync_run_service.py` / `SyncRunService`；状态机和机器错误位于
     `sync_run_contracts.py`，SQLite 细节不暴露给 GUI。
   - GUI Sync 页仍由 `gui_qt/sync_translation_workflow.py` 启动 `gemini_translate.py`，
     不持久化可恢复 run；统一页接入 #347 服务属于 #348 P3。
   - Gemini Batch 继续使用 Batch 生命周期，但与 Sync 消费相同 TranslationPlan 合同。
5. **检查与写回**
   - 模型结果先规范化并检查，再生成绑定 preview，最后由公共 apply 安全层写回。
   - `--force` 不得绕过 stale check、源快照、结构 blocker 或 adapter 写回计划验证。
6. **引擎边界**
   - Engine adapter 负责 discovery、candidate inventory、coverage、occurrence、relocation、
     translation validation 和声明式 writeback plan。
   - 公共层负责 manifest/result/project identity、实时源复核和原子事务写入。

## 配置事实来源

当前生产事实仍是旧 `translator_config.json`：

- `sync.backend/model/models/custom_litellm_providers` 描述同步模型；
- `batch.model` 及阶段 `model` 描述 Gemini Batch 和少数阶段覆盖；
- GUI 从原始 JSON 加载并在原对象上修改，以保留未建模字段；
- API Key 值只来自 `api_keys.json`、环境变量或系统 keyring，不属于项目配置。

`#348` 将以版本化 `model_routing` 替代模型相关旧字段。迁移期必须选择完整的新或旧事实
来源，不允许逐字段混合。合同见
[Model Routing 配置与迁移合同](plans/issue-348-model-routing-config-contract.md)。

## Settings：as-is 与 target

### as-is（当前 `main`）

`MainWindow` 仍是 Settings 的实际所有者：

- 10 个页面由 `_SETTINGS_PAGE_SPECS` 登记，按 `_ensure_settings_page` / `__getattr__` lazy materialize；
  普通切页只补建目标页，保存/重载才经 `_ensure_settings_pages_for_config` 补建全部配置页。
- `_load_config_to_ui` 负责把 `translator_config.json` 推入控件，`_current_config_ui_snapshot` /
  `_config_ui_saved_snapshot` 负责 dirty 基线，`_on_save_config` 是唯一保存总事务。
- `_confirm_unsaved_config_before_workflow` / `_confirm_unsaved_config_before_registry_switch` /
  `_confirm_unsaved_config_before_close` / `_confirm_leave_config_tab` 覆盖启动任务、切换项目、
  关闭窗口和离开设置页的保护。
- 即时持久化（API Key、LiteLLM keyring、工作区总表、`game_root`/`workspace_root`、字体与可选能力
  安装）与普通项目配置保存边界不同；主题只做即时预览，保存后才写入 `gui.theme`。
- LiteLLM、字体、安装、工作区刷新等局部任务由 `MainWindow` 属性与私有回调持有，页面尚不能
  脱离整窗构造或测试。

已核对的 as-is 缺口（Phase B 处理，不在 Phase A 修行为）：

- `_load_config_to_ui(pages={"project"})` 不填充项目页 advanced 字段，首次单独打开会看到默认值；
- `_CONFIG_SNAPSHOT_KEYS_BY_PAGE` 缺少 `project` / `api_keys`，而 `advanced` 键集合包含上下文主开关；
- builder 直接写 `self.<widget>` 与共享 `_advanced_setting_widgets`，页面依赖整窗私有属性；
- load/collect/validate/dirty/save/错误聚焦没有页面级 adapter 接口；
- 局部 worker 的取消、stale result 与关闭语义仍以整窗为单位。

完整页面清单、字段/即时持久化所有权与测试入口见
[#202 Phase A 契约与现状基线](plans/issue-202-settings-page-contract.md)。

### target（#202 Phase B 接入合同，尚未实现）

目标目录：

- `gui_qt/settings/page_contract.py`：`SettingsPage` Protocol、`SettingsIssue`、`SettingsPageActions`；
  冻结 `load/collect/validate/reset/focus_issue/set_task_running` 与 `config_keys` 单一所有权。
- `gui_qt/settings/registry.py`：页面登记与 lazy builder；强制一键一主，禁止两页复制 dirty/save。
- `gui_qt/settings/coordinator.py`：页内导航、load/collect/validate/dirty/save、离开保护、
  错误聚焦、项目切换失效与 worker 生命周期接线。
- `MainWindow`：收敛为应用壳与顶层装配，保留全局 `settings` route、header/sidebar、全局任务锁、
  runner/log、主题应用和 shutdown 协调。

页面只拥有控件、字段读写映射、局部校验和动作；页面通过显式 callback 与宿主交互，不读取其他
页面控件，也不依赖整窗私有属性才能独立构造。过渡期允许旧页面 adapter，但同一行为只能有一个
实际状态所有者。`gui_qt/workbench/page_contract.py` 与 `gui_qt/workbench/coordinator.py` 是已交付的
工作台页面化参考实现，Settings 合同复用其“页面报告意图、宿主持有执行与生命周期”的边界。

## #202 / #348 所有权

- #202 负责 Settings page contract/coordinator、页面局部生命周期与 `MainWindow` 收壳。
- #348 负责 `model_routing` schema、validator、migrator、rollback 与 ModelProfile/ExecutionStrategy/
  TaskRoute 解析和 Model Profiles 产品流程。
- 双方共用唯一配置 reader/persistence 边界：原始 JSON、unknown-field preservation、原子保存与
  共用写锁归 `ProjectState` / `config_store`；dirty snapshot/save 与迁移备份不得出现平行实现。
- #348 P2 可独立推进；P3 Settings 集成依赖本单 Phase B 的最小 contract/coordinator 接线。

逐项所有权表与 Phase B 验收映射见
[#348 合同](plans/issue-348-model-routing-config-contract.md#202--348-所有权) 与
[#202 Phase A 契约](plans/issue-202-settings-page-contract.md#202--348-所有权)。

## 调用链索引

doctor、Batch、同步翻译、关键词、订正写回、项目切换和 Settings 保存/加载/dirty 的可追踪入口、
关键函数与测试入口见 [代码路径索引](code_paths.md)。
