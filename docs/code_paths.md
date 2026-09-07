# 代码路径索引

本文列出维护当前配置、模型路由、执行与安全写回时需要核对的主要入口。它是调用链
索引，不替代模块 docstring、测试或用户手册。

## `translator_config.json`

- `translator_runtime.py`
  - `load_translator_settings()`：项目路径、prepare、Sync 与上下文配置。
  - `load_config()`：`api_keys.json`、环境变量和兼容模型/吞吐字段。
  - `load_runtime_config()`：长生命周期宿主使用的 defaults → credentials → project config
    聚合入口。
- `gui_qt/project_state.py`
  - `ProjectState.load_translator_config()`：GUI 原始 JSON object 读取。
  - `ProjectState.save_translator_config()`：路径规范化后写回；保存者需传入已保留未知字段的对象。
- `gui_qt/settings_schema.py`
  - `SettingField` / `ADVANCED_SETTING_FIELDS`：高级字段定义。
  - `read_advanced_settings()` / `apply_advanced_settings()`：高级字段读取、验证与合并。
- `gui_qt/app.py`
  - `_load_config_to_ui()`：当前 Settings 多页面加载协调。
  - `_current_config_ui_snapshot()`：dirty baseline 的扁平快照。
  - `_on_save_config()`：当前 collect/validate/save 总事务；#202 收缩前的事实入口。

## 模型与 Provider

- `litellm_provider_config.py`
  - 自定义 Provider ID、URL、环境变量、catalog 与 keyring 约定。
- `model_profile.py`
  - `build_profile_registry()`：旧 `sync.*` / `batch.*` 到内部 profile slot 的只读适配。
  - `resolve_routing_plan()`：冻结 ModelRoutingPlan 和阶段路由。
  - `validate_routing_plan()`：能力、策略和凭据引用检查。
  - `build_sync_backend()`：从已解析 profile 构造实际 Sync adapter。
- `model_routing_config.py`
  - `validate_model_routing_section()`：#348 schema-v1 纯验证器；当前未接入生产。
  - `LEGACY_FIELD_MAPPINGS` / `legacy_fields_present()`：P1 migrator 的冻结输入清单。

## TranslationPlan 与任务执行

- `translation_plan.py`：计划构造、序列化、fingerprint 和 source/context/model 快照。
- `translator_runtime.py`：生产 Sync plan/build 与 backend 接线。
- `gemini_translate_batch.py`：Batch build/submit/status/download/check/apply 与阶段请求入口。
- `sync_run_service.py`：CLI `sync-start` 等命令使用的耐久 Sync start/resume/status/cancel/derive 服务。
- `gui_qt/sync_translation_workflow.py` → `gemini_translate.py`：当前 GUI Sync 入口；
  尚不持久化可恢复 run，#348 P3 再接入 #347 服务。
- `sync_run_contracts.py`：状态、退出码、错误码和合法转换。
- `sync_run_store.py`：SQLite/WAL 持久层；产品 UI 不应直接读取。

## Engine adapter 与写回

- `engine_adapters/`：项目发现、候选、coverage、occurrence、relocation、
  validation、声明式 writeback plan。
- `translation_core.py`：共享 TranslationUnit/ModelResult/WritebackAction 与结果规范化。
- Batch 与 durable Sync 的 check/preview/apply 测试是写回安全事实；新增入口必须复用，
  不得仅凭模型任务完成状态授权写回。

## GUI 工作台

- `gui_qt/app.py`：应用壳、导航、Settings 总协调和遗留 glue。
- `gui_qt/workbench/`：真实工作台页面及页面 contract 的参考实现。
- `gui_qt/*_workflow.py`：GUI 动作到 CLI/服务调用的包装。
- `gui_qt/diagnostics_context.py`：CLI 命令参考与诊断入口。
- `gui_qt/user_copy.py`：共享用户文案；新增产品能力须同步。

## 修改检查表

- 配置结构：同步 example、runtime reader、Settings schema/page、迁移和文档。
- CLI 命令/输出：同步 argparse、workflow、诊断命令参考和机器合同测试。
- GUI 入口/用语：同步 `user_copy.py`、GUI 文档和 GUI 测试。
- 写回：保持 check → bound preview → apply、source snapshot 和 blocker 合同。
- Model routing：run 开始时冻结 resolved profile/strategy/route；禁止静默跨模型回退。
