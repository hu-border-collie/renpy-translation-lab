# 代码路径索引

本文列出维护当前配置、模型路由、执行、写回和 GUI 页面时需要核对的主要入口。它是调用链
索引，不替代模块 docstring、测试或用户手册。2026-09-09（#202 Phase A）补齐了 doctor、Batch、
同步翻译、关键词、订正写回、项目切换和 Settings 保存的可追踪入口。

## `translator_config.json`

- `translator_runtime.py`
  - `load_translator_settings()`：项目路径、prepare、Sync 与上下文配置。
  - `load_config()`：`api_keys.json`、环境变量和兼容模型/吞吐字段。
  - `load_runtime_config()`：长生命周期宿主使用的 defaults → credentials → project config
    聚合入口。
- `gui_qt/project_state.py`
  - `ProjectState.load_translator_config()`：GUI 原始 JSON object 读取。
  - `ProjectState.save_translator_config()`：路径规范化后写回；保存者需传入已保留未知字段的对象。
  - `ProjectState.set_game_root()` / `set_workspace_root()`：项目/工作区切换的即时持久化入口。
- `config_store.py`：`read_json_object()` / `write_json_object()`，共用写锁与原子替换；
  GUI 配置、`api_keys.json` 和 #348 离线迁移都经这里。
- `gui_qt/settings_schema.py`
  - `SettingField` / `ADVANCED_SETTING_FIELDS`：高级字段定义。
  - `read_advanced_settings()` / `validate_advanced_settings()` / `apply_advanced_settings()`：
    高级字段读取、验证与合并。
- `project_context_settings.py`
  - `load_project_context_settings()` / `save_project_context_settings()`：当前项目
    `<work>/project_context_settings.json`。
  - `read_batch_context_flags()`：项目级覆盖优先、全局默认兜底。
- `gui_qt/app.py`
  - `_load_config_to_ui()`：当前 Settings 多页面加载协调。
  - `_current_config_ui_snapshot()`：dirty baseline 的扁平快照。
  - `_on_save_config()`：保存入口（ensure pages / flush LiteLLM / coordinator.save 或 widget collect）。
  - `_persist_collected_settings()`：调用 Qt-free apply 后执行两文件写盘与回滚。

## 模型与 Provider

- `litellm_provider_config.py`
  - 自定义 Provider ID、URL、环境变量、catalog 与 keyring 约定。
- `model_profile.py`
  - `build_profile_registry()`：旧 `sync.*` / `batch.*` 到内部 profile slot 的只读适配。
  - `resolve_routing_plan()`：冻结 ModelRoutingPlan 和阶段路由。
  - `validate_routing_plan()`：能力、策略和凭据引用检查。
  - `build_sync_backend()`：从已解析 profile 构造实际 Sync adapter。
- `model_routing_config.py`
  - `validate_model_routing_section()`：#348 schema-v1 纯验证器。
  - `LEGACY_FIELD_MAPPINGS` / `legacy_fields_present()`：P1 migrator 的冻结输入清单。
- `model_routing_reader.py`
  - `runtime_settings_view()` / `read_routing_plan()`：P2 生产兼容读取；新任务使用 v1，旧配置保持原读取，已有任务优先冻结路由。
- `sync_model_backend.py` / `litellm_sync_backend.py`：Sync 请求适配与模型目录解析。

## TranslationPlan 与任务执行

- `translation_plan.py`：计划构造、序列化、fingerprint 和 source/context/model 快照。
- `translator_runtime.py`：生产 Sync plan/build 与 backend 接线。
- `gemini_translate_batch.py`：Batch build/submit/status/download/check/apply 与阶段请求入口。
- `sync_run_service.py`：CLI `sync-start` 等命令使用的耐久 Sync start/resume/status/cancel/derive 服务。
- `gui_qt/sync_translation_workflow.py` → `gemini_translate_batch.py sync-*`：当前 GUI Sync
  入口；步骤状态机只消费 schema-v1 envelope，运行/恢复/取消均经 #347 服务。
- `sync_run_contracts.py`：状态、退出码、错误码和合法转换。
- `sync_run_store.py`：SQLite/WAL 持久层；产品 UI 不应直接读取。

## Engine adapter 与写回

- `engine_adapters/`：项目发现、候选、coverage、occurrence、relocation、
  validation、声明式 writeback plan。
- `translation_core.py`：共享 TranslationUnit/ModelResult/WritebackAction 与结果规范化。
- Batch 与 durable Sync 的 check/preview/apply 测试是写回安全事实；新增入口必须复用，
  不得仅凭模型任务完成状态授权写回。

## GUI 工作台与 Settings

- `gui_qt/app.py`：应用壳、导航、Settings 总协调和遗留 glue。
- `gui_qt/workbench/`：真实工作台页面及页面 contract 的参考实现
  （`page_contract.py` / `coordinator.py`）。
- `gui_qt/*_workflow.py`：GUI 动作到 CLI/服务调用的包装。
- `gui_qt/diagnostics_context.py`：CLI 命令参考与诊断入口。
- `gui_qt/user_copy.py`：共享用户文案；新增产品能力须同步。
- `gui_qt/settings/page_contract.py`：`SettingsPage` Protocol、`SettingsIssue`、`SettingsPageActions`。
- `gui_qt/settings/registry.py`：`SettingsPageSpec` / `SettingsPageRegistry` / `build_default_registry`；
  10 页身份、唯一配置键所有权与 lazy 属性映射的唯一来源。
- `gui_qt/settings/coordinator.py`：`SettingsCoordinator` 的 `ensure_page` / `activate` / `load` /
  `collect` / `validate` / `reset` / `focus_issue` / `set_task_running`（记录并在建页时补发） /
  `set_baseline` / `is_dirty` / `leave_guard_prompt` / `save`（经 host `persist`）；不依赖 Qt。
- `gui_qt/settings/leave_guard.py`：未保存离开保护文案。
- `gui_qt/settings/save_apply.py`：`apply_collected_settings` 把 collect() 快照应用到原始 JSON 对象；
  extras 由宿主注入，写盘仍走 `MainWindow._persist_collected_settings`。
- `gui_qt/settings/litellm_page.py`：Phase C 迁出的 LiteLLM `SettingsPage`；局部 worker 复用
  #297 的取消、operation identity、retired ownership 与 shutdown 合同。
- `gui_qt/settings/page_chrome.py`：迁移页共用 chrome。
- `gui_qt/settings/models_page.py`：Phase D 迁出的「模型」`SettingsPage`；无局部 worker，目录
  extras 由宿主注入，保存仍走 `MainWindow._on_save_config`。
- `gui_qt/settings/field_widgets.py`：基础 SettingField 控件工厂。
- `gui_qt/settings/project_page.py`：Phase D 迁出的「项目」`SettingsPage`；SDK 浏览/查找/下载
  仍由宿主对话框与 `SdkInstallWorker` 执行。
- `gui_qt/settings/context_page.py`：Phase D 迁出的「上下文」`SettingsPage`；保存仍走
  `MainWindow._on_save_config`。
- `gui_qt/settings/advanced_page.py`：Phase D 迁出的「高级」`SettingsPage`；Gemini 目录扩展与
  模型轮换清单由页面持有，保存与 `validate_advanced_settings` 仍走 `MainWindow._on_save_config`。
- `gui_qt/settings/gemini_catalog_widgets.py`：Gemini 目录 extras 与轮换 checklist 控件。
- `gui_qt/settings/appearance_page.py`：Phase D 迁出的「外观」`SettingsPage`；主题预览 `persist=False`，
  字体下载仍走宿主 `FontInstallWorker`，保存仍走 `MainWindow._on_save_config`。
- `gui_qt/settings/shortcuts_page.py`：Phase D 迁出的「快捷键」`SettingsPage`；只读目录，无配置字段。
- `gui_qt/settings/api_keys_page.py`：Phase D 迁出的「密钥」`SettingsPage`；对话框与 keyring 写入仍由宿主执行，无 translator_config 字段。
- `gui_qt/settings/extensions_page.py`：Phase D 迁出的「扩展」`SettingsPage`；关系分析器安装仍走宿主 `OptionalFeatureInstallController`，无 translator_config 字段。
- `gui_qt/settings/workspace_page.py`：Phase D 迁出的「项目列表」`SettingsPage`；嵌入 `GamesRegistryPanel`，刷新/导入 worker 仍在面板内，切换项目由宿主回调执行。
- Settings 页面/字段/异步任务现状与 Phase B 接入合同见
  [#202 Phase A 契约与现状基线](plans/issue-202-settings-page-contract.md)。

## 环境检查 doctor

- CLI：`gemini_translate_batch.py doctor`（只读，`--output json` 返回机器 envelope）
  → `legacy.load_translator_settings(persist_corrected_game_root=False)` → `load_glossary()` /
  `load_batch_settings()` → `collect_doctor_report()` → `print_doctor_report()`。
- 采集与建议：`gemini_translate_batch.py` 的 `collect_doctor_translation_progress()`、
  `collect_tl_doctor_counts()`、`collect_doctor_layout_context()` / `assess_doctor_layout_status()`、
  `collect_doctor_workflow_state()`、`collect_doctor_recommendations()` →
  `doctor_recommendations.py` 的稳定 code 与 `gui_qt/doctor_report.py` 的用户摘要。
- GUI：`doctor_btn` / `Ctrl+D` → `MainWindow._on_doctor_button_clicked()` → `_on_run_doctor()` →
  `_snapshot_runtime_config_for_job(persist_corrected_game_root=False)` → `gui_qt/doctor_worker.py` 的
  `DoctorWorker`（QThread）→ `run_doctor_check_in_subprocess()`（spawn 子进程，测试/启动失败回退
  进程内）→ `run_doctor_check()` → `summarize_doctor_report()` → `_on_doctor_completed()`。
  工作区总表对照由 `compare_registry_with_doctor_report()` 提供。
- 取消/身份：`_invalidate_doctor_worker()`、`_retire_doctor_worker()`、
  `_wire_doctor_worker_terminal()`；doctor 不占用全局任务锁的导航冻结。
- 测试：`tests.test_gui_doctor_worker`、`tests.test_gui_doctor_report`、
  `tests.test_gui_doctor_gating`、`tests.test_doctor_recommendations`、
  `tests.test_doctor_layout_status`、`tests.test_gui_games_registry_doctor_compare`。

## Batch 翻译主路径

- CLI：`build` → `submit` → `status` → `download` → `check` → `apply`；辅助入口包括
  `estimate-cost`、`recover-submit`、`build-retry` / `merge-retry`、`quality-*`、`split` 相关命令。
  `check_results()` 写入 `last_check_summary` / `last_check_report_path` 与 writeback gate；
  `apply_results()` 先经 `require_safe_check_for_apply()` / `fail_apply_preflight()` 复核最近一次 check
  的 writeback gate，再重新校验 source snapshot、结构 blocker 和结果/预览制品；`--force`
  只能绕过“已写回”保护，不能绕过这些检查。
- GUI 页面：`gui_qt/workbench/batch_translation_page.py`（`BatchTranslationPage`，只渲染与回调）
  → `MainWindow._on_batch_translation_page_action()` / `_on_start_translation()` →
  `gui_qt/workflow_factory.py` 的 `create_workflow()` → `gui_qt/translation_workflow.py` 的
  `TranslationWorkflow`（`start_new()` 生成 `build`/`submit`/`status`，`current_step()` /
  `_args_for_step()` 规划命令，`complete_current_step()` 解析输出）。
- 执行与收尾：`_run_workflow_current_step()` → `_start_cli_command()` → `gui_qt/cli_runner.py` 的
  `CliRunner`（QProcess，参数列表、无 shell）→ `gemini_translate_batch.py`；
  `_on_workflow_step_finished()` → `_update_writeback_from_check()` →
  `gui_qt/check_report.py` 的 `summarize_check_envelope()` / `summarize_check_output()` 生成
  `WritebackSummary`；用户点击写回后 `_on_apply_writeback()` 才执行
  `apply <manifest> --output json --non-interactive`。
- 拆分/补译：`gui_qt/split_batch_workflow.py` 的 `SplitBatchQueueWorkflow`、
  `gui_qt/retry_workflow.py` 的 `create_retry_followup_workflow()`；
  不确定提交状态由 `gui_qt/batch_workflow_support.py` 的
  `plan_unsubmitted_workflow_steps()` / `build_submit_cli_args()` / `build_recover_submit_cli_args()` 规划。
- 测试：`tests.test_gui_translation_workflow`、`tests.test_gui_batch_workflow_support`、
  `tests.test_gui_workflow_factory`、`tests.test_gui_workflow_guards`、
  `tests.test_gui_task_pages`、`tests.test_gui_task_controls`、`tests.test_gui_split_batch`、
  `tests.test_gui_retry_workflow`、`tests.test_gemini_translate_batch_cli_contract`、
  `tests.test_batch_golden_corpus`。

## 同步翻译

- GUI 耐久运行：`gui_qt/workbench/sync_translation_page.py`（`SyncTranslationPage`）→
  `_on_start_translation()` / `_on_resume_durable_sync()` / `_on_cancel_durable_sync()` →
  `gui_qt/sync_translation_workflow.py` 的 `SyncTranslationWorkflow` 步骤（`sync-start`、
  `sync-status`、`sync-resume`、`sync-cancel`、`sync-derive`、`check`、`apply`）→
  `CliRunner` → `gemini_translate_batch.py --output json --non-interactive` →
  `run_durable_sync_command()` → `sync_run_service.SyncRunService` →
  `durable_sync_executor.DurableSyncExecutor` → `sync_run_store.SyncRunStore`（SQLite/WAL）。
- GUI 只渲染公开 snapshot（进度、`next_action`、usage、制品路径）；不读数据库、不自行重试，
  也不改写 freshness / 写回判定。停止本机 worker 不会取消 run；取消是单独的 `sync-cancel`。
- GUI 写回：`check <RUN>` 在 `writeback_gate=allow` 时生成绑定预览，用户确认后
  `_on_apply_sync_translation()` → `apply <RUN>`，由 `apply_durable_sync_results()` 复核
  run/results/targets/check 制品哈希、源快照与预览绑定；重复 apply 为无副作用
  `already_applied`。旧 `gemini_translate.py` 入口仍保留为 CLI 兼容路径，不再是 GUI 主流程。
- 测试：`tests.test_gui_sync_translation_workflow`、`tests.test_gui_durable_sync_page`、
  `tests.test_gui_sync_translation_report`、`tests.test_sync_run_service`、
  `tests.test_durable_sync_workflow`、`tests.test_durable_sync_executor`、
  `tests.test_durable_sync_subprocess_recovery`、`tests.test_sync_translation_preview`。

## 关键词提取与合并

- Batch 提取：CLI `build-keywords` → `submit` → `status` → `download` → `export-keywords`。
  GUI `KeywordsPage` → `create_workflow(KEYWORD_EXTRACTION)` →
  `gui_qt/keyword_workflow.py` 的 `KeywordBatchWorkflow`；导出完成后
  `_copy_keyword_reports_to_game_parent()` 与 `keyword_merge_candidates_path_from_manifest()` 记录候选路径。
- 同步提取：`gui_qt/sync_keyword_workflow.py` 的 `SyncKeywordWorkflow` →
  `gemini_translate_batch.py sync-keywords` → `keyword_merge_candidates_path_from_sync_output()`。
- 人工合并：工作台 Keywords 页写回按钮 → `_on_open_keyword_merge()` →
  `gui_qt/keyword_merge_report.py` 的 `keyword_merge_ready()` / `load_keyword_merge_context()` →
  `gui_qt/keyword_merge_dialog.py` 的 `KeywordMergeDialog`（先 `preview_selected_merge_actions()`，
  再 `merge_selected_candidates()`）→ `keyword_glossary_merge.py` 写 `glossary.json`（写入前备份）。
  CLI 等价入口为 `merge-keywords-to-glossary`，命令模板由
  `keyword_glossary_merge.build_merge_keywords_cli_command()` 生成。
- 历史证据：`keyword_history.py` 只读匹配已有 old/new 译文，产出 consistent/conflict/ambiguous
  等证据状态，不直接改 glossary 或脚本。
- 测试：`tests.test_gui_keyword_workflow`、`tests.test_gui_sync_keyword_workflow`、
  `tests.test_gui_keyword_merge_dialog`、`tests.test_gui_keyword_merge_report`、
  `tests.test_keyword_glossary_merge`、`tests.test_keyword_history`、
  `tests.test_keyword_history_corpus`。

## 订正与写回

- Batch 订正：CLI `build-revisions` → `submit` → `status` → `download` → `preview-revisions`。
  `preview_revisions()` 写入 `last_revision_preview` / `last_revision_preview_at` 与 preview
  JSONL/Markdown；用户确认后 `apply-revisions` → `apply_revisions()` 重新校验
  `_require_valid_revision_preview()`、manifest/project identity、result JSONL hash、源文件快照、
  质量规则/策略和结构 blocker。`--force` 只绕过 `revision_applied_at`，不绕过这些检查。
- Sync 订正：`gui_qt/sync_revision_workflow.py` 的 `SyncRevisionWorkflow` →
  `gemini_translate_batch.py sync-revisions` 生成 preview 报告，仍经同一 `apply-revisions` 门禁。
- 订正提案：CLI `export-revision-corpus` → 外部/人工编辑 → `import-revision-proposals` →
  `gui_qt/revision_selection_dialog.py` + `revision_selection.py` 的 staged selection →
  `confirm-revision-proposals` → `preview-revisions` / `apply-revisions`。最终审校的
  `final-review-create-revisions` 也汇入同一门禁。
- GUI 编排：`gui_qt/workbench/revision_page.py`（`RevisionPage`）→
  `gui_qt/revision_workflow.py` 的 `RevisionBatchWorkflow` / `RevisionProposalImportWorkflow` /
  `RevisionProposalConfirmWorkflow` 与 `RevisionCorpusExportWorkflow`；
  写回按钮 `_on_apply_revision()` → `apply-revisions <manifest>`，摘要由
  `gui_qt/revision_writeback_report.py` / `gui_qt/revision_report.py` 生成。
- 纯核心：`revision_corpus.py`（只读导出）、`revision_proposals.py`（导入校验）、
  `revision_selection.py`（选择/摘要绑定）、`final_review_revision.py`（审校联动）。
- 测试：`tests.test_gui_revision_workflow`、`tests.test_gui_sync_revision_workflow`、
  `tests.test_gui_revision_selection`、`tests.test_gui_revision_writeback_report`、
  `tests.test_gui_revision_report`、`tests.test_gui_revision_corpus`、
  `tests.test_revision_proposals`、`tests.test_revision_corpus`、
  `tests.test_final_review_revision`、`tests.test_batch_golden_corpus`。

## 项目切换与工作区

- 入口：全局项目栏「切换项目」`_on_global_switch_project()` 只导航到 Settings·项目列表；
  「指定本地目录…」`_on_select_project()` 经 dirty 保护后直接切换；
  项目列表 `GamesRegistryPanel` 的「切换到此项目」回调 `_on_registry_switch_project()`；
  advanced `game_root` 保存时由 `_sync_state_game_root_from_settings()` 触发。
- 切换链：`_on_registry_switch_project()` → `_confirm_unsaved_config_before_registry_switch()` →
  `_switch_game_root()`：取消 runner/doctor worker → `ProjectState.set_game_root()` →
  `_save_game_root_to_config()`（`sync_project_asset_paths_in_config()` + `config_store` 原子写）→
  `_refresh_project_label()` / `_load_config_to_ui()` → `_clear_all_mode_sessions()` 并
  `reset_project()` 上下文库/Sync/关键词/订正页 → 清空 workflow、writeback、manifest、doctor、
  revision corpus/proposal 状态 → `_apply_work_mode_ui()` → `_refresh_diagnostics_context()`。
- 工作区：`_on_workspace_changed()` → `ProjectState.set_workspace_root()`；
  项目列表维护经 `gui_qt/games_registry_actions.py` → `games_registry.py` 的
  `load_registry()` / `save_registry()` / `refresh_all()` / `merge_discovered_projects()` /
  `record_batch()` / `apply_workspace_setup()`，后台执行由 `RegistryRefreshWorker` /
  `RegistryIngestWorker` 承担。
- 测试：`tests.test_gui_project_state`、`tests.test_gui_project_bar`、
  `tests.test_gui_games_registry_panel_layout`、`tests.test_gui_games_registry_worker`、
  `tests.test_gui_games_registry_actions`、`tests.test_gui_games_registry_view`、
  `tests.test_games_registry`、`tests.test_gui_workspace_setup_dialog`。

## Settings 加载、保存、dirty 与离开保护

- 页面身份：`gui_qt/settings/registry.py` 的 `SettingsPageSpec` / `SettingsPageRegistry` /
  `build_default_registry()`；`MainWindow._settings_registry` 在构造时校验一键一主。
  `gui_qt/app.py` 的 `_SETTINGS_PAGE_SPECS` / `_SETTINGS_CONFIG_PAGE_KEYS` /
  `_SETTINGS_LAZY_ATTR_TO_PAGE` 现在只是 registry 的兼容别名。
- lazy 构建：`MainWindow.__getattr__` / `_ensure_settings_page()` →
  `SettingsCoordinator.ensure_page()` → `_create_*_settings_page()`；
  `_on_settings_nav_row_changed()` → `SettingsCoordinator.activate()`。
  普通切页只构建目标页；保存/重载经 `_ensure_settings_pages_for_config()` 补建全部配置页，
  并只保留已加载页面的编辑快照。
- 加载：`_load_config_to_ui()` → `ProjectState.load_translator_config()` → 按 registry 键填充
  `models` / `litellm` / `context` / `appearance` / `advanced` / `project`（项目页首次
  materialize 即加载 owned advanced 字段）→ `_update_config_ui_saved_snapshot()`。
  `_current_config_ui_snapshot()` / `_restore_config_ui_snapshot()` 负责 dirty 基线与补建时恢复；
  恢复只回写快照中存在的 advanced 键，避免把新补建页面覆盖成推荐值。
- 保存：coordinator 的 `SettingsPageActions.save` 委托
  `MainWindow._request_settings_save()` → `_on_save_config()`；
  `_on_save_config()` → `_ensure_settings_pages_for_config()` →
  `_flush_litellm_model_selection_save()` → `SettingsCoordinator.save()` →
  host `persist(collect())` → `apply_collected_settings()`（Qt-free）→
  `state.save_translator_config()`（`config_store`）→ `save_project_context_settings()`；
  项目文件失败时回滚全局文件；最后 `_sync_settings_dirty_baseline()`。
  `__new__` helper 无 coordinator 时走 `_widget_settings_collect()` + 同一 persist。
  页面 `collect()` 只返回值，均不写盘。
- dirty/离开保护：coordinator `is_dirty` / `leave_guard_prompt` 与 `_config_ui_saved_snapshot`
  同步；宿主仍弹出 Qt 对话框。
  `_confirm_unsaved_config_before_workflow()`、`_confirm_unsaved_config_before_registry_switch()`、
  `_confirm_unsaved_config_before_close()`、`_confirm_leave_config_tab()`；
  丢弃修改走 `_on_reload_config()`，恢复推荐值走 `_on_restore_recommended_config()`。
- 即时持久化边界：`_on_manage_api_keys()` → `ProjectState.save_api_keys()`；
  `_open_litellm_provider_key_dialog()` → `litellm_provider_config.store_provider_key_store()`；
  项目列表 → `games_registry.save_registry()`；`ProjectState.set_game_root()` /
  `set_workspace_root()`；字体 `_on_download_recommended_fonts()` → `FontInstallWorker`；
  扩展 `_on_install_relation_analyzer()` → `OptionalFeatureInstallController`；
  主题 `_on_theme_changed()` → `_set_theme_preference(persist=False)` 仅预览。
- 测试：`tests.test_settings_page_contract`、`tests.test_settings_registry`、
  `tests.test_settings_coordinator`、`tests.test_settings_save_apply`、`tests.test_gui_settings_coordinator`、
  `tests.test_gui_settings_layout`、`tests.test_gui_shell_navigation`、
  `tests.test_gui_app_config`、`tests.test_gui_settings_context_primary`、
  `tests.test_gui_settings_schema`、`tests.test_settings_litellm_page`、
  `tests.test_settings_models_page`、`tests.test_settings_project_page`、`tests.test_settings_context_page`、`tests.test_settings_advanced_page`、`tests.test_settings_appearance_page`、`tests.test_settings_shortcuts_page`、`tests.test_settings_api_keys_page`、`tests.test_settings_extensions_page`、`tests.test_settings_workspace_page`、`tests.test_gui_litellm_settings_page`、
  `tests.test_gui_litellm_settings`、`tests.test_gui_litellm_worker`、
  `tests.test_litellm_catalog_cache`、`tests.test_project_context_settings`。

## GUI 局部异步任务与生命周期

- CLI 长任务：`gui_qt/cli_runner.py` 的 `CliRunner`（QProcess，参数列表，流式 stdout/stderr，
  可 kill）；`MainWindow._start_cli_command()` / `_on_workflow_step_finished()` 管理任务槽。
- QThread worker：`doctor_worker.DoctorWorker`、`litellm_worker.*`、`font_worker.FontInstallWorker`、
  `games_registry_worker.RegistryRefreshWorker` / `RegistryIngestWorker`、
  `sdk_install_worker.SdkInstallWorker`。
- QThreadPool：`context_library_worker.ContextLibraryStatusJob`（QRunnable）由
  `QThreadPool.globalInstance().start(job, -1)` 调度。
- QProcess 本地动作：`gui_qt/optional_feature_install.py` 的 `OptionalFeatureInstallController`
  串行执行 pip 安装。
- 生命周期设施（#297）：`gui_qt/lifecycle.py` 的 `ShutdownParticipant` /
  `CallbackShutdownParticipant`，`gui_qt/operation_identity.py` 的 operation identity 摘要；
  worker 取消后保留 retired ownership 到真实 `finished`，GUI 线程不得 `wait()` 固定等待，
  stale 结果不得覆盖新项目/新请求。Settings 页面化必须复用这些设施，不另建生命周期框架。
- 测试：`tests.test_gui_lifecycle`、`tests.test_gui_operation_identity`、
  `tests.test_gui_optional_feature_install`、`tests.test_gui_font_worker`、
  `tests.test_gui_games_registry_worker`、`tests.test_gui_cli_runner`、
  `tests.test_gui_context_library_worker`。

## P1/P2 配置迁移与生产读取

`model_config_migration.py` → `model_routing_migration_store.py` →
`model_routing_migration.preview_migration` → schema validator / 兼容 reader →
`config_store` 备份、报告和原子替换。GUI 的原始 JSON 保存也委托该 store 并共用写锁。
P2 起 Sync/Batch 兼容入口消费合法 `model_routing`；步骤与边界见 [迁移说明](model_config_migration.md)。

## 测试入口速查

- Settings / 导航 / 配置：`tests.test_gui_settings_layout`、`tests.test_gui_shell_navigation`、
  `tests.test_gui_app_config`、`tests.test_gui_settings_context_primary`、
  `tests.test_gui_settings_schema`、`tests.test_gui_project_state`、
  `tests.test_project_context_settings`。
- doctor / Batch / 同步：`tests.test_gui_doctor_worker`、`tests.test_gui_doctor_report`、
  `tests.test_gui_translation_workflow`、`tests.test_gui_sync_translation_workflow`、
  `tests.test_gemini_translate_batch_cli_contract`、`tests.test_sync_run_service`。
- 关键词 / 订正：`tests.test_gui_keyword_workflow`、`tests.test_keyword_glossary_merge`、
  `tests.test_gui_revision_workflow`、`tests.test_gui_revision_selection`、
  `tests.test_revision_proposals`。
- 工作区 / 项目切换：`tests.test_gui_project_bar`、`tests.test_gui_games_registry_panel_layout`、
  `tests.test_gui_games_registry_worker`、`tests.test_games_registry`。
- 生命周期 / worker：`tests.test_gui_lifecycle`、`tests.test_gui_operation_identity`、
  `tests.test_gui_optional_feature_install`、`tests.test_gui_litellm_worker`。
- 全量：`python -B tests/run_gui_tests.py -q`、`python -B tests/run_cli_tests.py -q`、
  `python scripts/run_quality_gates.py all`。

## 修改检查表

- 配置结构：同步 example、runtime reader、Settings schema/page、迁移和文档。
- CLI 命令/输出：同步 argparse、workflow、诊断命令参考和机器合同测试。
- GUI 入口/用语：同步 `user_copy.py`、GUI 文档和 GUI 测试。
- Settings 结构/导航：同步 `_SETTINGS_PAGE_SPECS`、lazy 映射、dirty 键所有权、
  `gui_qt/settings/` 合同（Phase B 起）和相关 GUI 测试。
- 写回：保持 check → bound preview → apply、source snapshot 和 blocker 合同。
- Model routing：run 开始时冻结 resolved profile/strategy/route；禁止静默跨模型回退。
