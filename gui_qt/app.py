"""Main GUI application for the optional workbench.

This is the first version shell (per #42):
- Pure PySide6 with tabbed layout (workbench / config / diagnostics)
- Delegates everything to the existing CLI via QProcess
- Workbench tab: project selection, doctor + translation workflow status
"""
from __future__ import annotations

import copy
import importlib.util
import os
import sys
import re
import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QProcess, QProcessEnvironment, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QGuiApplication,
    QKeySequence,
    QPalette,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QGroupBox,
    QLineEdit,
    QCheckBox,
    QFrame,
    QFormLayout,
    QScrollArea,
    QSplitter,
    QLayout,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
    QListView,
    QStackedWidget,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QToolButton,
)
from project_version import __version__

from .path_utils import canonical_abs_path, normalize_context_storage_location
from .responsive_layout import FlowButtonBar, ResponsiveActionPanel
from .empty_state import EmptyStateWidget
from .api_key_dialog import ApiKeyDialog
from .api_key_helpers import mask_api_key
from .bootstrap_report import (
    BootstrapProgressState,
    BootstrapProgressTracker,
    BootstrapSummary,
    create_bootstrap_progress_state,
    create_bootstrap_progress_tracker,
    format_bootstrap_progress_bar_label,
    format_bootstrap_progress_facts,
    read_batch_context_flags,
    running_bootstrap_summary,
    stale_bootstrap_summary,
    summarize_rag_bootstrap_output,
    summarize_source_index_bootstrap_output,
    update_bootstrap_progress_from_line,
)
from .apply_failure_dialog import ApplyFailureDialog
from .apply_failure_report import (
    apply_failure_report_available,
    build_apply_failure_report,
)
from .check_failures_report import build_check_issues_report
from .check_issues_dialog import CheckIssuesDialog
from .check_report import (
    WritebackSummary,
    build_recheck_cli_args,
    idle_writeback_summary,
    idle_writeback_summary_for_work_mode,
    recheck_writeback_ready,
    running_writeback_summary,
    stale_writeback_summary,
    summarize_apply_output,
    summarize_check_output,
    summarize_manifest_writeback,
)
from .diagnostics_context import (
    DiagnosticsContext,
    build_diagnostics_context,
    existing_retry_manifest_path,
    sync_diagnostics_context,
)
from .probe_report import (
    build_probe_cli_args,
    probe_summary_to_diagnostics_context,
    running_probe_summary,
    summarize_probe_output,
    translation_probe_ready,
)
from .ab_experiment_report import (
    AB_VARIANT_CHOICE_FORCE_OFF,
    AB_VARIANT_CHOICE_FORCE_ON,
    AB_VARIANT_CHOICE_SKIP,
    AB_VARIANT_DIMENSIONS,
    ab_experiment_summary_to_diagnostics_context,
    build_compare_variants_cli_args,
    build_variants_from_gui_selection,
    format_variant_names,
    read_ab_dimension_enabled_states,
    running_ab_experiment_summary,
    summarize_compare_variants_output,
    translation_ab_experiment_ready,
    validate_ab_experiment_variants,
    write_variants_to_temp_file,
)
from .keyword_merge_dialog import KeywordMergeDialog
from .keyword_merge_report import (
    keyword_merge_candidates_path_from_manifest,
    keyword_merge_candidates_path_from_sync_output,
    keyword_merge_ready,
    load_keyword_merge_context,
    summarize_keyword_merge_result,
)
from .keyword_report import summarize_keyword_result_from_manifest
from .revision_report import summarize_revision_apply_output
from .revision_writeback_report import (
    summarize_revision_writeback_from_manifest,
    summarize_revision_writeback_from_preview_output,
)
from .split_batch import (
    SplitManifestEntry,
    load_split_manifest_entries,
    summarize_split_entries,
)
from .split_report import (
    build_split_cli_args,
    running_split_summary,
    split_summary_to_diagnostics_context,
    summarize_split_output,
    translation_split_ready,
)
from .split_batch_workflow import SplitBatchQueueWorkflow
from .sync_translation_workflow import SyncTranslationWorkflow
from .split_status_delegate import SPLIT_ACTION_DATA_ROLE, SplitStatusActionDelegate
from .split_status_table_helpers import is_split_action_column, split_action_item_payload
from .retry_preview_dialog import RetryPreviewDialog
from .retry_report import (
    assess_retry_eligibility,
    build_retry_cli_args,
    parse_build_retry_output,
    retry_followup_allowed,
    summarize_retry_manifest,
)
from .repair_report import (
    RepairReportCandidate,
    assess_repair_eligibility,
    build_repair_cli_args,
    discover_repair_report_candidates,
    parse_repair_output,
    repair_action_ready,
    repair_summary_to_diagnostics_context,
    running_repair_summary,
    summarize_repair_output,
)
from .retry_workflow import (
    create_retry_followup_workflow,
    describe_retry_followup_button,
    retry_followup_workflow_ready,
)
from .cli_runner import CliRunner
from .doctor_report import (
    DoctorSummary,
    idle_summary,
    cancelled_summary,
    running_summary,
    stale_summary,
    summarize_doctor_output,
    summarize_doctor_report,
)
from .doctor_worker import DoctorWorker, DoctorWorkerResult
from .font_helpers import optional_fonts_installed, user_fonts_dir
from .font_worker import FontInstallResult, FontInstallWorker

from .games_registry_actions import handle_post_apply_registry_update
from .games_registry_panel import GamesRegistryPanel
from .games_registry_doctor_compare import compare_registry_with_doctor_report

from .manifest_resume_summary import (
    ManifestWorkflowDisplay,
    build_manifest_workflow_display,
    completed_manifest_entry_fact,
)
from .template_generation_report import (
    running_template_generation_summary,
    summarize_template_generation_output,
    template_generation_to_doctor_summary,
)
from .work_bootstrap_report import (
    running_work_bootstrap_summary,
    summarize_work_bootstrap_output,
    with_game_root_persist_warning,
    work_bootstrap_to_doctor_summary,
)
from .litellm_worker import (
    LiteLLMConnectionTestWorker,
    LiteLLMModelCatalogWorker,
    LiteLLMVersionWorker,
)
from .litellm_settings import (
    provider_credential_status,
    read_sync_backend_models,
    write_sync_backend_models,
)
from gemini_model_catalog import (
    BUILTIN_GEMINI_EMBEDDING_MODELS,
    BUILTIN_GEMINI_TRANSLATION_MODELS,
    merge_model_lists,
    resolve_gemini_embedding_models,
    resolve_gemini_translation_models,
    write_model_catalog_extras,
)
from litellm_provider_config import (
    DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
    catalog_source_label,
    installed_litellm_version,
    version_key,
    ProviderCredentialStoreError,
    delete_provider_api_key,
    load_provider_api_key,
    provider_from_model,
    store_provider_api_key,
)
from optional_feature import FeatureInstallState, FeatureStatus
from .optional_feature_install import (
    OptionalFeatureInstallController,
    action_enabled_for_status,
    build_litellm_controller,
    build_relation_analyzer_controller,
)
from .project_state import ProjectState
from .theme import apply_theme, system_prefers_dark
from .icon_provider import set_tabler_button_icon
from .theme_helpers import (
    DEFAULT_THEME_PREFERENCE,
    THEME_DARK,
    THEME_SYSTEM,
    normalize_theme_preference,
    read_gui_theme_from_config,
    resolve_effective_theme,
    write_gui_theme_to_config,
)
from .settings_schema import (
    ADVANCED_SETTING_FIELD_BY_KEY,
    ADVANCED_SETTING_FIELDS,
    CONTEXT_PRIMARY_SETTING_CATEGORY,
    CONTEXT_PRIMARY_SETTING_KEYS,
    context_primary_setting_fields,
    BASIC_RECOMMENDED_VALUES,
    SettingField,
    allowed_gemini_rotation_models,
    apply_advanced_settings,
    grouped_advanced_fields,
    read_advanced_settings,
    recommended_advanced_settings,
    validate_advanced_settings,
)

# Selected on 设置 → 工作区; shown read-only on 设置 → 项目.
_SETTINGS_WORKSPACE_MANAGED_KEYS = frozenset({"game_root"})
from .translation_workflow import WorkflowUpdate
from .user_copy import (
    format_job_fact,
    format_job_state_fact,
    format_manifest_path_fact,
)
from .work_modes import (
    WORKBENCH_NAV_ORDER,
    WorkMode,
    WorkbenchNavItem,
    bootstrap_disabled_message,
    default_work_mode_for_nav,
    normalize_work_mode,
    work_mode_hint_texts,
    work_mode_spec,
    workbench_nav_for_work_mode,
    workbench_nav_spec,
)
from .workbench import WorkbenchPageActions
from .workbench.coordinator import WorkbenchPageCoordinator
from .workbench.batch_translation_page import BatchActionState, BatchTranslationPage
from .workbench.context_library_page import ContextLibraryPage
from .workbench.keywords_page import KeywordsPage
from .workbench.revision_page import RevisionPage
from .workbench.sync_translation_page import SyncTranslationPage
from .workbench_session import WorkbenchModeSession
from .batch_workflow_support import resolve_submit_max_cost
from .workflow_factory import create_workflow, resume_workflow
from .workflow_progress import (
    WorkflowProgressState,
    create_workflow_progress_state,
    update_workflow_progress_from_line,
)
from .widget_helpers import (
    NoWheelComboBox,
    NoWheelTabWidget,
    add_editable_combo_popup_action,
)
from .wizard_timeline import WizardTimeline
from .log_highlighter import LogHighlighter
from .status_icons import StatusBadge
from .toast_widget import ToastNotification

# Diagnostics splitter: idle favors task context; running tasks expand the log.
_DIAGNOSTICS_IDLE_CONTEXT_PX = 420
_DIAGNOSTICS_IDLE_LOG_PX = 180
_DIAGNOSTICS_RUNNING_CONTEXT_RATIO = 0.32

# Dedicated status-page indices; task sessions update these values off-surface.
_BATCH_STAGE_PREPARE = 0  # 环境检查
_BATCH_STAGE_EXECUTE = 1  # 进度
_BATCH_STAGE_RESULT = 2  # 写回 / 结果

_LOG_FLUSH_INTERVAL_MS = 80
_LAYOUT_SYNC_DEBOUNCE_MS = 32
_UI_PROGRESS_FLUSH_INTERVAL_MS = 100

# Unified application-shell routes. The legacy tab/list widgets remain as
# internal state adapters, while these semantic destinations drive the only
# visible primary navigation.
_SHELL_ROUTE_PROJECT_PREPARE = "project_prepare"
_SHELL_ROUTE_SETTINGS = "settings"
_SHELL_ROUTE_DIAGNOSTICS = "diagnostics"
_SHELL_WORKBENCH_PREFIX = "workbench:"
_SHELL_TASK_ROUTES = tuple(
    f"{_SHELL_WORKBENCH_PREFIX}{item.value}" for item in WORKBENCH_NAV_ORDER
)

# Settings sections: (key, nav label, builder method name).
# Pages are built on first visit (or when config load/save needs them).
_SETTINGS_PAGE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("workspace", "项目列表", "_build_settings_workspace_page"),
    ("project", "项目", "_build_settings_project_page"),
    ("api_keys", "密钥", "_build_settings_api_keys_page"),
    ("models", "模型", "_build_settings_models_page"),
    ("litellm", "LiteLLM", "_build_settings_litellm_page"),
    ("extensions", "扩展", "_build_settings_extensions_page"),
    ("context", "上下文", "_build_settings_context_page"),
    ("appearance", "外观", "_build_settings_appearance_page"),
    ("shortcuts", "快捷键", "_build_settings_shortcuts_page"),
    ("advanced", "高级", "_build_settings_advanced_page"),
)
# Sections that hold translator_config / dirty-snapshot fields.
_SETTINGS_CONFIG_PAGE_KEYS = frozenset(
    {
        "project",
        "api_keys",
        "models",
        "litellm",
        "context",
        "appearance",
        "advanced",
    }
)
# Attribute → settings section for lazy materialization (tests + direct access).
_SETTINGS_LAZY_ATTR_TO_PAGE: dict[str, str] = {
    "_games_registry_panel": "workspace",
    "api_status_label": "api_keys",
    "api_btn": "api_keys",
    "settings_project_root_value": "project",
    "settings_go_workspace_btn": "project",
    "rag_enabled_cb": "context",
    "source_index_enabled_cb": "context",
    "bootstrap_on_build_cb": "context",
    "context_storage_game_cb": "context",
    "sync_model_combo": "models",
    "sync_embedding_combo": "models",
    "batch_model_combo": "models",
    "batch_embedding_combo": "models",
    "batch_thinking_combo": "models",
    "sync_backend_combo": "litellm",
    "litellm_provider_combo": "litellm",
    "litellm_model_combo": "litellm",
    "litellm_refresh_models_btn": "litellm",
    "litellm_catalog_status_label": "litellm",
    "sync_backend_hint": "litellm",
    "litellm_version_label": "litellm",
    "litellm_check_version_btn": "litellm",
    "install_litellm_btn": "litellm",
    "litellm_install_progress": "litellm",
    "litellm_provider_label": "litellm",
    "litellm_api_key_edit": "litellm",
    "litellm_save_key_btn": "litellm",
    "litellm_delete_key_btn": "litellm",
    "litellm_credential_status_label": "litellm",
    "litellm_test_connection_btn": "litellm",
    "litellm_connection_status_label": "litellm",
    "relation_analyzer_status_label": "extensions",
    "relation_analyzer_failure_label": "extensions",
    "relation_analyzer_install_btn": "extensions",
    "relation_analyzer_docs_btn": "extensions",
    "relation_analyzer_install_progress": "extensions",
    "theme_combo": "appearance",
    "font_install_status_label": "appearance",
    "download_fonts_btn": "appearance",
    "font_install_progress": "appearance",
}


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        qt_app: QApplication | None = None,
        resources_dir: Path | None = None,
        project_state: ProjectState | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Ren'Py Translation Lab - 图形工作台")
        self.setMinimumSize(960, 640)
        self.resize(1180, 780)

        self.state = project_state or ProjectState()
        self._diagnostics_context_fingerprint: tuple[object, ...] | None = None
        self._diagnostics_refresh_input_key: tuple[object, ...] | None = None
        self._main_tab_enter_generation = 0
        self._pending_log_lines: list[str] = []
        self._workflow_progress_dirty = False
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setSingleShot(True)
        self._log_flush_timer.setInterval(_LOG_FLUSH_INTERVAL_MS)
        self._log_flush_timer.timeout.connect(self._flush_pending_log_lines)
        self._progress_flush_timer = QTimer(self)
        self._progress_flush_timer.setSingleShot(True)
        self._progress_flush_timer.setInterval(_UI_PROGRESS_FLUSH_INTERVAL_MS)
        self._progress_flush_timer.timeout.connect(self._flush_throttled_progress_ui)
        self._layout_sync_timer = QTimer(self)
        self._layout_sync_timer.setSingleShot(True)
        self._layout_sync_timer.setInterval(_LAYOUT_SYNC_DEBOUNCE_MS)
        self._layout_sync_timer.timeout.connect(self._sync_layout_sizes)
        self._qt_app = qt_app
        self._resources_dir = resources_dir or Path(__file__).resolve().parent / "resources"
        self._theme_preference = DEFAULT_THEME_PREFERENCE
        self.runner = CliRunner()
        self._loading_config_to_ui = False
        self._loading_theme_to_ui = False
        self._updating_batch_thinking_combo = False
        self._batch_thinking_config_has_key = False
        self._batch_thinking_user_changed = False
        self._active_command = ""
        self._doctor_output_lines: list[str] = []
        self._workflow = None
        self._split_status_entries: list[SplitManifestEntry] = []
        self._split_status_selected_manifest_path = ""
        self._completed_manifest_snapshot: dict[str, object] | None = None
        self._viewing_completed_manifest = False
        self._work_mode = WorkMode.BATCH_TRANSLATION
        self._mode_sessions: dict[WorkMode, WorkbenchModeSession] = {}
        self._workbench_nav_item = WorkbenchNavItem.BATCH_TRANSLATION
        # Session emptiness is not navigation state: a user can choose 同步
        # before it has produced any workflow or manifest to save.
        self._last_mode_by_nav: dict[WorkbenchNavItem, WorkMode] = {
            item: default_work_mode_for_nav(item) for item in WORKBENCH_NAV_ORDER
        }
        self._workflow_step_output_lines: list[str] = []
        self._apply_output_lines: list[str] = []
        self._recheck_output_lines: list[str] = []
        self._probe_output_lines: list[str] = []
        self._compare_variants_output_lines: list[str] = []
        self._compare_variants_names = ""
        self._compare_variants_temp_file = ""
        self._keyword_merge_candidates_path = ""
        self._split_output_lines: list[str] = []
        self._repair_output_lines: list[str] = []
        self._apply_revision_output_lines: list[str] = []
        self._bootstrap_output_lines: list[str] = []
        self._bootstrap_progress: BootstrapProgressState | None = None
        self._bootstrap_progress_tracker: BootstrapProgressTracker | None = None
        self._workflow_progress: WorkflowProgressState | None = None
        self._workflow_progress_base_facts: list[str] = []
        self._bootstrap_progress_eta_timer = QTimer(self)
        self._bootstrap_progress_eta_timer.setInterval(1000)
        self._bootstrap_progress_eta_timer.timeout.connect(
            self._on_bootstrap_progress_eta_tick
        )
        self._work_bootstrap_output_lines: list[str] = []
        self._template_generation_output_lines: list[str] = []
        self._doctor_summary_mode = ""
        self._doctor_summary_status = ""
        self._doctor_check_completed = False
        self._doctor_worker: DoctorWorker | None = None
        self._last_doctor_report: dict | None = None
        self._last_doctor_report_game_root = ""
        self._build_retry_output_lines: list[str] = []
        self._retry_followup_confirmed: set[str] = set()
        self._writeback_manifest_path = ""
        self._config_ui_saved_snapshot: dict[str, object] = {}
        self._advanced_setting_widgets: dict[str, QWidget] = {}
        self._advanced_setting_error_labels: dict[str, QLabel] = {}
        self._settings_nav_rows: dict[str, int] = {}
        self._shell_nav_rows: dict[str, int] = {}
        self._shell_section_items: list[QListWidgetItem] = []
        self._shell_nav_dispatching = False
        self._shell_special_route: str | None = None
        self._shell_task_status_index: int | None = None
        self._last_main_tab_index = 0
        self._handling_config_tab_leave = False
        self._task_running = False
        self._litellm_catalog_worker: LiteLLMModelCatalogWorker | None = None
        self._litellm_version_worker: LiteLLMVersionWorker | None = None
        self._litellm_latest_version = ""
        self._litellm_latest_compatible_version = ""
        self._litellm_latest_requires_python = ""
        self._litellm_catalog_source = ""
        self._litellm_connection_worker: LiteLLMConnectionTestWorker | None = None
        self._font_install_worker: FontInstallWorker | None = None
        self._litellm_catalog_models: dict[str, tuple[str, ...]] = {}
        self._updating_litellm_provider = False
        # _games_registry_panel is intentionally NOT set here so attribute access
        # triggers __getattr__ lazy materialization of 设置 · 项目列表.
        self._litellm_install: OptionalFeatureInstallController | None = None
        self._relation_analyzer_install: OptionalFeatureInstallController | None = None
        self._optional_feature_last_failed: set[str] = set()
        self._settings_pages_built: set[str] = set()
        self._settings_models_signals_wired = False
        self._settings_lazy_resolving = False

        central = QWidget()
        central.setObjectName("app_shell")
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_app_sidebar())

        main = QWidget()
        main.setObjectName("app_main")
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._build_page_header())
        root_layout.addWidget(main, 1)

        self.tab_widget = NoWheelTabWidget()
        self.tab_widget.setObjectName("main_tabs")
        main_layout.addWidget(self.tab_widget, 1)

        self._build_workbench_tab()
        self._build_config_tab()
        self._build_log_tab()
        self.tab_widget.tabBar().hide()
        self._populate_shell_nav()
        self._setup_shell_status_bar()
        self._refresh_action_icons()

        # batch_model / thinking signals are wired when the models settings page
        # is first built (lazy settings pages).
        self._last_main_tab_index = self.tab_widget.currentIndex()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self._sync_shell_nav_selection()

        # Connect runner
        self.runner.line_ready.connect(self._on_cli_line_ready)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_runner_error)

        self._refresh_project_label()
        self._show_pending_game_root_redirect_notice()
        # Settings widgets are lazy: skip API label / full config→UI until a
        # settings section is materialized (or save/reload needs it). Cold start
        # only paints the workbench shell; settings pages build on first visit.
        # Idle chrome only — skip resume/manifest disk walks until deferred
        # startup refresh (see _deferred_startup_refresh).
        self._set_doctor_summary(idle_summary(), resume_available=(False, ""))
        QTimer.singleShot(0, self._deferred_startup_refresh)
        QTimer.singleShot(0, self._sync_work_mode_hint_height)

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Status
        self.statusBar().showMessage(
            "图形界面是可选组件；核心命令行不受影响。"
        )

    def _shell_nav_shortcut_entries(self) -> list[tuple[str, str]]:
        """Canonical sidebar destinations for navigation shortcuts.

        Fixed order matching ``_populate_shell_nav`` so the Settings catalog can
        be built before the shell list is populated, and stays aligned with IA.
        """
        return [
            (_SHELL_ROUTE_PROJECT_PREPARE, "项目与环境"),
            (
                f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.BATCH_TRANSLATION.value}",
                workbench_nav_spec(WorkbenchNavItem.BATCH_TRANSLATION).label,
            ),
            (
                f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.SYNC_TRANSLATION.value}",
                workbench_nav_spec(WorkbenchNavItem.SYNC_TRANSLATION).label,
            ),
            (
                f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.KEYWORDS.value}",
                "关键词 / 术语",
            ),
            (
                f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.REVISION.value}",
                workbench_nav_spec(WorkbenchNavItem.REVISION).label,
            ),
            (
                f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.CONTEXT.value}",
                workbench_nav_spec(WorkbenchNavItem.CONTEXT).label,
            ),
            (_SHELL_ROUTE_SETTINGS, "设置"),
        ]

    def _shortcut_catalog(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Human-readable shortcut groups for the Settings · 快捷键 page.

        Mirrors the current shell IA (sidebar routes + page-header diagnostics),
        not the legacy hidden main-tab indices.
        """
        nav_rows: list[tuple[str, str]] = []
        for index, (_route, label) in enumerate(
            self._shell_nav_shortcut_entries(),
            start=1,
        ):
            if index > 9:
                break
            nav_rows.append((f"Ctrl+{index}", f"打开「{label}」"))
        nav_rows.append(("Ctrl+0", "打开「诊断与运行日志」"))

        return [
            (
                "任务",
                [
                    ("Ctrl+D", "环境检查（运行中变为停止检查）"),
                    (
                        "Ctrl+T",
                        "开始当前任务（翻译 / 生成模板 / 提取等，取决于当前页）",
                    ),
                    (
                        "Ctrl+K",
                        "停止当前任务（含环境检查、准备工作目录、生成模板、预建）",
                    ),
                ],
            ),
            (
                "导航",
                nav_rows,
            ),
            (
                "日志与设置",
                [
                    ("Ctrl+L", "打开「诊断与运行日志」"),
                    ("Ctrl+Shift+L", "清空诊断日志输出"),
                    ("Ctrl+S", "保存设置（仅在「设置」页有效）"),
                ],
            ),
        ]

    def _setup_shortcuts(self) -> None:
        """Bind global keyboard shortcuts aligned with the shell information architecture."""
        self._doctor_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        # Use the project-bar dispatcher so Ctrl+D can also cancel doctor / template.
        self._doctor_shortcut.activated.connect(self._on_doctor_button_clicked)

        self._translate_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        self._translate_shortcut.activated.connect(self._on_start_translation)

        self._kill_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self._kill_shortcut.activated.connect(self._on_kill)

        self._open_diagnostics_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self._open_diagnostics_shortcut.activated.connect(
            lambda: self._activate_shell_route(_SHELL_ROUTE_DIAGNOSTICS)
        )

        self._clear_log_shortcut = QShortcut(QKeySequence("Ctrl+Shift+L"), self)
        self._clear_log_shortcut.activated.connect(self._on_clear_log)

        # Config save — only active when config tab is shown
        self._save_config_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self._save_config_shortcut.activated.connect(self._shortcut_save_config)

        # Primary shell destinations (sidebar order), not legacy main-tab indices.
        self._shell_route_shortcuts: list[QShortcut] = []
        for index, (route, _label) in enumerate(
            self._shell_nav_shortcut_entries(),
            start=1,
        ):
            if index > 9:
                break
            sc = QShortcut(QKeySequence(f"Ctrl+{index}"), self)
            sc.activated.connect(
                lambda checked=False, target=route: self._activate_shell_route(target)
            )
            self._shell_route_shortcuts.append(sc)

        self._diagnostics_nav_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        self._diagnostics_nav_shortcut.activated.connect(
            lambda: self._activate_shell_route(_SHELL_ROUTE_DIAGNOSTICS)
        )

        # Button tooltips with shortcut hints
        if hasattr(self, "doctor_btn"):
            self.doctor_btn.setToolTip("环境检查 (Ctrl+D)")
        if hasattr(self, "translate_btn"):
            self.translate_btn.setToolTip("开始当前任务 (Ctrl+T)")
        if hasattr(self, "kill_btn"):
            self.kill_btn.setToolTip("停止任务 (Ctrl+K)")
        if hasattr(self, "header_log_btn"):
            self.header_log_btn.setToolTip("诊断与运行日志 (Ctrl+L / Ctrl+0)")
        if hasattr(self, "clear_log_btn"):
            self.clear_log_btn.setToolTip("清空日志 (Ctrl+Shift+L)")
        if hasattr(self, "save_config_btn"):
            self.save_config_btn.setToolTip("保存设置 (Ctrl+S)")

    def _shortcut_save_config(self) -> None:
        """Handle Ctrl+S: only save when the config tab is active."""
        if hasattr(self, "save_config_btn") and not self.save_config_btn.isEnabled():
            return
        if hasattr(self, "tab_widget") and hasattr(self, "_config_tab"):
            if self.tab_widget.currentWidget() is self._config_tab:
                self._on_save_config()

    def _deferred_startup_refresh(self) -> None:
        self._apply_work_mode_ui(
            refresh_manifest_writeback=True,
            refresh_diagnostics=False,
        )
        QTimer.singleShot(0, self._deferred_startup_diagnostics_refresh)

    def _deferred_startup_diagnostics_refresh(self) -> None:
        self._refresh_manifest_derived_ui(refresh_diagnostics=True)

    def eventFilter(self, watched: Any, event: QEvent) -> bool:
        if watched is self.work_mode_hint_label and event.type() == QEvent.Type.Resize:
            self._sync_work_mode_hint_height()
        if hasattr(self, "split_status_table") and watched is self.split_status_table.viewport():
            if event.type() == QEvent.Type.Resize:
                self._sync_split_status_table_columns()
            elif event.type() in {QEvent.Type.Leave, QEvent.Type.MouseMove}:
                delegate = getattr(self, "_split_status_action_delegate", None)
                if delegate is not None:
                    if event.type() == QEvent.Type.Leave:
                        delegate.clear_hover_state()
                        delegate.clear_pressed_state()
                    else:
                        index = self.split_status_table.indexAt(event.position().toPoint())
                        if not index.isValid() or not is_split_action_column(index.column()):
                            delegate.clear_hover_state()
                            delegate.clear_pressed_state()
        return super().eventFilter(watched, event)

    def _sync_work_mode_hint_height(self) -> None:
        label = self.work_mode_hint_label
        width = label.width()
        if width <= 0:
            frame = getattr(self, "_mode_frame", None)
            if frame is not None:
                margins = frame.layout().contentsMargins()
                width = frame.width() - margins.left() - margins.right()
        if width <= 0:
            width = 320

        metrics = label.fontMetrics()
        max_height = metrics.lineSpacing() * 2
        for text in work_mode_hint_texts():
            rect = metrics.boundingRect(
                0,
                0,
                width,
                0,
                Qt.TextFlag.TextWordWrap.value,
                text,
            )
            max_height = max(max_height, rect.height())
        label.setMinimumHeight(max_height + 4)

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        self._layout_sync_timer.start()

    def _sync_layout_sizes(self) -> None:
        self._reflow_button_bars()
        if not hasattr(self, "doctor_message_label") or not hasattr(self, "workbench_status_tabs"):
            return

        width = self.workbench_status_tabs.width()
        workflow_scroll = getattr(self, "workflow_scroll", None)
        if workflow_scroll is not None:
            workflow_viewport = workflow_scroll.viewport()
            if workflow_viewport is not None and workflow_viewport.width() > 0:
                width = workflow_viewport.width()
        if width <= 0:
            width = 400
        label_width = max(100, width - 24)
        changed = False

        def sync_label(label: QLabel) -> None:
            nonlocal changed
            if not label or not label.text() or not label.isVisible():
                target_height = 0
            else:
                metrics = label.fontMetrics()
                rect = metrics.boundingRect(
                    0,
                    0,
                    label_width,
                    9999,
                    Qt.TextFlag.TextWordWrap.value,
                    label.text(),
                )
                target_height = rect.height() + 4
            if label and label.minimumHeight() != target_height:
                label.setMinimumHeight(target_height)
                changed = True

        # The project check pane is scrollable, so Qt can size those labels
        # lazily inside the scroll area. Measuring long doctor output here is
        # synchronous on the GUI thread and makes checks/tab switches feel slow.
        sync_label(self.workflow_message_label)
        sync_label(self.workflow_facts_label)
        sync_label(self.writeback_message_label)
        sync_label(self.writeback_facts_label)
        sync_label(self.writeback_details_label)

        if hasattr(self, "split_status_table") and self.split_status_table.isVisible():
            self._sync_split_status_table_columns()

        if changed:
            current = self.workbench_status_tabs.currentWidget()
            if current is not None and current.layout() is not None:
                current.layout().invalidate()
            if workflow_scroll is not None and workflow_scroll.widget() is not None:
                workflow_content_layout = workflow_scroll.widget().layout()
                if workflow_content_layout is not None:
                    workflow_content_layout.invalidate()
                workflow_scroll.widget().updateGeometry()
            self.workbench_status_tabs.updateGeometry()

    def _build_app_sidebar(self) -> QFrame:
        """Build the persistent product and primary-navigation rail."""
        sidebar = QFrame()
        sidebar.setObjectName("app_sidebar")
        sidebar.setFixedWidth(224)
        self.app_sidebar = sidebar

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 16, 14, 12)
        layout.setSpacing(12)

        brand = QWidget()
        brand_copy_layout = QVBoxLayout(brand)
        brand_copy_layout.setContentsMargins(0, 0, 0, 0)
        brand_copy_layout.setSpacing(0)
        self.sidebar_brand_title = QLabel("Ren'Py Translation Lab")
        self.sidebar_brand_title.setObjectName("sidebar_brand_title")
        self.sidebar_brand_subtitle = QLabel("TRANSLATION WORKBENCH")
        self.sidebar_brand_subtitle.setObjectName("sidebar_brand_subtitle")
        brand_copy_layout.addWidget(self.sidebar_brand_title)
        brand_copy_layout.addWidget(self.sidebar_brand_subtitle)
        layout.addWidget(brand)

        self.shell_nav = QListWidget()
        self.shell_nav.setObjectName("shell_nav")
        self.shell_nav.setFrameShape(QFrame.Shape.NoFrame)
        self.shell_nav.setSpacing(1)
        self.shell_nav.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.shell_nav.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.shell_nav.currentRowChanged.connect(self._on_shell_nav_row_changed)
        layout.addWidget(self.shell_nav, 1)

        return sidebar

    def _build_page_header(self) -> QFrame:
        """Build the shared page-level header used by every shell destination."""
        header = QFrame()
        header.setObjectName("page_header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 14, 20, 14)
        layout.setSpacing(16)

        copy = QVBoxLayout()
        copy.setContentsMargins(0, 0, 0, 0)
        copy.setSpacing(2)
        self.shell_breadcrumb_label = QLabel("工作流 / 批量翻译")
        self.shell_breadcrumb_label.setObjectName("shell_breadcrumb_label")
        self.shell_page_title = QLabel("批量翻译")
        self.shell_page_title.setObjectName("shell_page_title")
        self.shell_page_subtitle = QLabel(
            "配置批量任务、跟踪进度，并在检查通过后安全写回。"
        )
        self.shell_page_subtitle.setObjectName("shell_page_subtitle")
        self.shell_page_subtitle.setWordWrap(True)
        copy.addWidget(self.shell_breadcrumb_label)
        copy.addWidget(self.shell_page_title)
        copy.addWidget(self.shell_page_subtitle)
        layout.addLayout(copy, 1)

        self.header_task_status_label = QLabel("待命")
        self.header_task_status_label.setObjectName("header_task_status_label")
        self.header_task_status_label.setProperty("status", "idle")
        layout.addWidget(
            self.header_task_status_label,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )

        self.header_log_btn = QPushButton("运行日志")
        self.header_log_btn.setObjectName("header_log_btn")
        self.header_log_btn.setCheckable(True)
        self.header_log_btn.clicked.connect(self._on_header_log_clicked)
        layout.addWidget(
            self.header_log_btn,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        return header

    def _on_header_log_clicked(self) -> None:
        """Open diagnostics and restore the checked state on repeated clicks."""
        self._expand_diagnostics_log(switch_tab=True)
        # When diagnostics is already current, QTabWidget emits no change signal;
        # mirror the semantic route explicitly so a checkable button cannot drift.
        self._sync_shell_nav_selection()

    def _add_shell_section(self, label: str) -> None:
        item = QListWidgetItem(label)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setSizeHint(QSize(0, 30))
        font = item.font()
        font.setBold(True)
        font.setPointSize(9)
        item.setFont(font)
        self.shell_nav.addItem(item)
        self._shell_section_items.append(item)

    def _add_shell_route(
        self,
        route: str,
        label: str,
    ) -> None:
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, route)
        item.setToolTip(label)
        self.shell_nav.addItem(item)
        self._shell_nav_rows[route] = self.shell_nav.row(item)

    def _populate_shell_nav(self) -> None:
        """Populate the selected third-concept information architecture."""
        self.shell_nav.clear()
        self._shell_nav_rows = {}
        self._shell_section_items = []

        self._add_shell_section("工作流")
        self._add_shell_route(
            _SHELL_ROUTE_PROJECT_PREPARE,
            "项目与环境",
        )
        self._add_shell_route(
            f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.BATCH_TRANSLATION.value}",
            workbench_nav_spec(WorkbenchNavItem.BATCH_TRANSLATION).label,
        )
        self._add_shell_route(
            f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.SYNC_TRANSLATION.value}",
            workbench_nav_spec(WorkbenchNavItem.SYNC_TRANSLATION).label,
        )

        self._add_shell_section("翻译资产")
        self._add_shell_route(
            f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.KEYWORDS.value}",
            "关键词 / 术语",
        )
        self._add_shell_route(
            f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.REVISION.value}",
            workbench_nav_spec(WorkbenchNavItem.REVISION).label,
        )
        self._add_shell_route(
            f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.CONTEXT.value}",
            workbench_nav_spec(WorkbenchNavItem.CONTEXT).label,
        )

        self._add_shell_section("系统")
        self._add_shell_route(
            _SHELL_ROUTE_SETTINGS,
            "设置",
        )
        self._set_shell_nav_task_lock(False)

    def _on_shell_nav_row_changed(self, row: int) -> None:
        if row < 0 or self._shell_nav_dispatching:
            return
        item = self.shell_nav.item(row)
        route = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if route:
            self._activate_shell_route(str(route))

    def _set_shell_special_route(self, route: str | None) -> None:
        """Isolate a special status-page tab from the active task session."""
        previous = getattr(self, "_shell_special_route", None)
        tabs = getattr(self, "workbench_status_tabs", None)

        if route is not None:
            if previous is None and tabs is not None:
                self._shell_task_status_index = self._current_batch_stage_index()
            self._shell_special_route = route
            return

        self._shell_special_route = None
        if previous is None:
            return

        # Reveal the task-owned tabs before restoring their saved index; Qt will
        # not reliably select a tab while it is still hidden by the project route.
        self._sync_workbench_status_surface()

        saved_index = getattr(self, "_shell_task_status_index", None)
        self._shell_task_status_index = None
        if (
            tabs is not None
            and saved_index is not None
            and 0 <= int(saved_index) < tabs.count()
        ):
            self._focus_workbench_status_tab(int(saved_index))

    def _activate_shell_route(self, route: str) -> None:
        """Navigate through existing state-aware entry points, then mirror selection."""
        if not hasattr(self, "tab_widget"):
            return

        if route.startswith(_SHELL_WORKBENCH_PREFIX):
            nav_value = route[len(_SHELL_WORKBENCH_PREFIX):]
            nav_item = WorkbenchNavItem(nav_value)
            if self._context_switching_locked():
                if nav_item != workbench_nav_for_work_mode(self._current_work_mode()):
                    self._sync_shell_nav_selection()
                    return
            self.tab_widget.setCurrentWidget(self._workbench_tab)
            if self.tab_widget.currentWidget() is not self._workbench_tab:
                self._sync_shell_nav_selection()
                return
            self._set_shell_special_route(None)
            self.workbench_stack.show()
            target_mode = self._last_mode_by_nav.get(
                nav_item,
                default_work_mode_for_nav(nav_item),
            )
            if target_mode != self._current_work_mode():
                # During 环境检查, restore from session snapshots only. Disk
                # manifest reloads on the UI thread compete with doctor work and
                # make page switches feel frozen.
                refresh_disk = not self._is_doctor_active()
                self._set_work_mode(
                    target_mode,
                    refresh_manifest_writeback=refresh_disk,
                )
            else:
                self._sync_task_selectors_from_work_mode()
        elif route == _SHELL_ROUTE_PROJECT_PREPARE:
            self.tab_widget.setCurrentWidget(self._workbench_tab)
            if self.tab_widget.currentWidget() is not self._workbench_tab:
                self._sync_shell_nav_selection()
                return
            self._set_shell_special_route(route)
            self.workbench_stack.hide()
            self.workbench_status_tabs.setCurrentIndex(_BATCH_STAGE_PREPARE)
        elif route == _SHELL_ROUTE_SETTINGS:
            self.tab_widget.setCurrentWidget(self._config_tab)
            if self.tab_widget.currentWidget() is not self._config_tab:
                self._sync_shell_nav_selection()
                return
            self._set_shell_special_route(None)
            self.workbench_stack.show()
        elif route == _SHELL_ROUTE_DIAGNOSTICS:
            self.tab_widget.setCurrentWidget(self._diagnostics_tab)
            if self.tab_widget.currentWidget() is not self._diagnostics_tab:
                self._sync_shell_nav_selection()
                return
            self._set_shell_special_route(None)
            self.workbench_stack.show()
        self._sync_shell_nav_selection()

    def _current_shell_route(self) -> str:
        if not hasattr(self, "tab_widget"):
            return (
                f"{_SHELL_WORKBENCH_PREFIX}"
                f"{WorkbenchNavItem.BATCH_TRANSLATION.value}"
            )
        current = self.tab_widget.currentWidget()
        if current is getattr(self, "_config_tab", None):
            return _SHELL_ROUTE_SETTINGS
        if current is getattr(self, "_diagnostics_tab", None):
            return _SHELL_ROUTE_DIAGNOSTICS
        if current is getattr(self, "_workbench_tab", None):
            special = getattr(self, "_shell_special_route", None)
            if special == _SHELL_ROUTE_PROJECT_PREPARE:
                return special
            nav = workbench_nav_for_work_mode(self._current_work_mode())
            return f"{_SHELL_WORKBENCH_PREFIX}{nav.value}"
        return (
            f"{_SHELL_WORKBENCH_PREFIX}"
            f"{WorkbenchNavItem.BATCH_TRANSLATION.value}"
        )

    def _sync_shell_nav_selection(self) -> None:
        nav = getattr(self, "shell_nav", None)
        if nav is None:
            return
        route = self._current_shell_route()
        self._shell_nav_dispatching = True
        blocked = nav.blockSignals(True)
        try:
            row = self._shell_nav_rows.get(route)
            if row is None:
                nav.setCurrentRow(-1)
                nav.clearSelection()
            else:
                nav.setCurrentRow(row)
        finally:
            nav.blockSignals(blocked)
            self._shell_nav_dispatching = False
        self._update_shell_header(route)
        self.header_log_btn.setChecked(route == _SHELL_ROUTE_DIAGNOSTICS)
        self._sync_workbench_status_surface(route)

    def _update_shell_header(self, route: str) -> None:
        descriptions = {
            _SHELL_ROUTE_PROJECT_PREPARE: (
                "工作流 / 项目与环境",
                "项目与环境",
                "选择项目、检查环境并准备 work/game 工作目录。",
            ),
            f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.BATCH_TRANSLATION.value}": (
                "工作流 / 批量翻译",
                "批量翻译",
                "配置批量任务、跟踪进度，并在检查通过后安全写回。",
            ),
            f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.SYNC_TRANSLATION.value}": (
                "工作流 / 同步翻译",
                "同步翻译",
                "直接运行同步翻译，并在同一工作台查看状态与结果。",
            ),
            f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.KEYWORDS.value}": (
                "翻译资产 / 关键词与术语",
                "关键词 / 术语",
                "提取与合并关键词资产，保留批量和同步两种执行方式。",
            ),
            f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.REVISION.value}": (
                "翻译资产 / 订正",
                "订正",
                "生成、检查并安全写回译文订正结果。",
            ),
            f"{_SHELL_WORKBENCH_PREFIX}{WorkbenchNavItem.CONTEXT.value}": (
                "翻译资产 / 上下文库",
                "上下文库",
                "管理 RAG 与源码索引，为翻译任务提供项目语境。",
            ),
            _SHELL_ROUTE_SETTINGS: (
                "系统 / 设置",
                "设置",
                "管理项目列表、当前项目参数、模型、上下文、外观与高级参数。",
            ),
            _SHELL_ROUTE_DIAGNOSTICS: (
                "系统 / 运行日志",
                "诊断与运行日志",
                "查看任务上下文、命令参考、任务记录和原始输出。",
            ),
        }
        breadcrumb, title, subtitle = descriptions.get(
            route,
            descriptions[
                f"{_SHELL_WORKBENCH_PREFIX}"
                f"{WorkbenchNavItem.BATCH_TRANSLATION.value}"
            ],
        )
        if route == _SHELL_ROUTE_SETTINGS and hasattr(self, "settings_nav"):
            current = self.settings_nav.currentItem()
            if current is not None:
                breadcrumb = f"系统 / 设置 / {current.text()}"
        self.shell_breadcrumb_label.setText(breadcrumb)
        self.shell_page_title.setText(title)
        self.shell_page_subtitle.setText(subtitle)
        self._refresh_shell_status()

    def _is_doctor_active(self) -> bool:
        """True while an environment check owns the global task slot."""
        if getattr(self, "_active_command", "") == "doctor":
            return True
        return self._is_doctor_running()

    def _context_switching_locked(self) -> bool:
        """Whether a running job should freeze workbench context switches.

        Environment check (doctor) is a read-only prep diagnostic: keep shell
        navigation free so users can browse other pages while it finishes.
        """
        if not bool(getattr(self, "_task_running", False)):
            return False
        if self._is_doctor_active():
            return False
        return True

    def _task_page_running_chrome(self) -> bool:
        """Whether task pages should show page-owned stop / running chrome.

        Doctor is a project-prep job. Translation and asset pages must not look
        like they own it (which previously enabled mismatched stop buttons).
        """
        if not bool(getattr(self, "_task_running", False)):
            return False
        if self._is_doctor_active():
            return False
        return True

    def _is_bootstrap_work_active(self) -> bool:
        """True while 准备工作目录 owns the global task slot."""
        return getattr(self, "_active_command", "") == "bootstrap_work"

    def _is_generate_template_active(self) -> bool:
        return getattr(self, "_active_command", "") == "generate_template"

    def _task_stop_button_label(self) -> str:
        """Contextual stop label for page-local and legacy kill controls."""
        command = getattr(self, "_active_command", "")
        if command == "generate_template":
            return "停止生成"
        if command in {"bootstrap_rag", "bootstrap_source_index"}:
            return "停止预建"
        if command == "bootstrap_work":
            return "停止准备"
        if command == "doctor":
            return "停止检查"
        return "停止"

    def _sync_doctor_prep_button_chrome(self) -> None:
        """Project-bar doctor control: run check / cancel doctor or template gen."""
        if not hasattr(self, "doctor_btn"):
            return
        # generate-template results land in the doctor summary surface; reuse the
        # project-bar control so cancel stays visible when status is focused there.
        if self._is_generate_template_active():
            self.doctor_btn.setText("停止生成")
            self.doctor_btn.setObjectName("doctor_stop_btn")
            self.doctor_btn.setEnabled(True)
            self.doctor_btn.setToolTip("停止生成翻译模板 (Ctrl+K)")
            self._repolish_widget(self.doctor_btn)
            self._apply_doctor_prep_button_icon(stopping=True)
            return
        if self._is_doctor_active():
            self.doctor_btn.setText("停止检查")
            self.doctor_btn.setObjectName("doctor_stop_btn")
            self.doctor_btn.setEnabled(True)
            self.doctor_btn.setToolTip("停止环境检查 (Ctrl+K)")
            self._repolish_widget(self.doctor_btn)
            self._apply_doctor_prep_button_icon(stopping=True)
            return
        running = bool(getattr(self, "_task_running", False))
        self.doctor_btn.setText("环境检查")
        self.doctor_btn.setObjectName("secondary_btn")
        self.doctor_btn.setEnabled(not running)
        self.doctor_btn.setToolTip("环境检查 (Ctrl+D)")
        self._repolish_widget(self.doctor_btn)
        self._apply_doctor_prep_button_icon(stopping=False)

    def _apply_doctor_prep_button_icon(self, *, stopping: bool) -> None:
        """Use the red stop icon while cancelling 环境检查; stethoscope otherwise."""
        if not hasattr(self, "doctor_btn") or not hasattr(self, "_resources_dir"):
            return
        set_tabler_button_icon(
            self.doctor_btn,
            self._resources_dir,
            "player-stop" if stopping else "stethoscope",
            dark=self._effective_theme_is_dark(),
            role="danger" if stopping else "default",
        )

    def _sync_bootstrap_prep_button_chrome(self) -> None:
        """Project-bar bootstrap control: prepare work dir, or cancel while active.

        Legacy kill_btn lives on a hidden action panel; without this chrome the
        user has no visible cancel on 项目与环境 during 准备工作目录.
        """
        if not hasattr(self, "bootstrap_work_btn"):
            return
        if self._is_bootstrap_work_active():
            self.bootstrap_work_btn.setText("停止准备")
            self.bootstrap_work_btn.setObjectName("bootstrap_stop_btn")
            self.bootstrap_work_btn.setEnabled(True)
            self.bootstrap_work_btn.setToolTip("停止准备工作目录 (Ctrl+K)")
            self._repolish_widget(self.bootstrap_work_btn)
            self._apply_bootstrap_prep_button_icon(stopping=True)
            return
        running = bool(getattr(self, "_task_running", False))
        self.bootstrap_work_btn.setText("准备工作目录")
        self.bootstrap_work_btn.setObjectName("secondary_btn")
        self.bootstrap_work_btn.setEnabled(not running)
        self.bootstrap_work_btn.setToolTip(
            "从 original/game 复制到 work/game（若已存在则按 CLI 规则跳过或更新）。"
        )
        self._repolish_widget(self.bootstrap_work_btn)
        self._apply_bootstrap_prep_button_icon(stopping=False)

    def _apply_bootstrap_prep_button_icon(self, *, stopping: bool) -> None:
        if not hasattr(self, "bootstrap_work_btn") or not hasattr(self, "_resources_dir"):
            return
        set_tabler_button_icon(
            self.bootstrap_work_btn,
            self._resources_dir,
            "player-stop" if stopping else "folder-cog",
            dark=self._effective_theme_is_dark(),
            role="danger" if stopping else "default",
        )

    def _set_shell_nav_task_lock(self, running: bool) -> None:
        nav = getattr(self, "shell_nav", None)
        if nav is None:
            return
        lock_task_routes = running and self._context_switching_locked()
        active_task_route = (
            f"{_SHELL_WORKBENCH_PREFIX}"
            f"{workbench_nav_for_work_mode(self._current_work_mode()).value}"
        )
        for route, row in self._shell_nav_rows.items():
            item = nav.item(row)
            if item is None:
                continue
            enabled = True
            if lock_task_routes and route in _SHELL_TASK_ROUTES:
                enabled = route == active_task_route
            flags = Qt.ItemFlag.ItemIsSelectable
            if enabled:
                flags |= Qt.ItemFlag.ItemIsEnabled
            item.setFlags(flags)


    def _work_mode_has_writeback_surface(self, mode: WorkMode | None = None) -> bool:
        """Return whether the active workflow owns a meaningful writeback page."""
        resolved = mode or self._current_work_mode()
        spec = work_mode_spec(resolved)
        return bool(
            spec.supports_translation_writeback
            or self._uses_revision_writeback(spec.mode)
            or spec.mode
            in {WorkMode.KEYWORD_EXTRACTION, WorkMode.SYNC_KEYWORD_EXTRACTION}
        )

    def _sync_workbench_status_surface(
        self,
        route: str | None = None,
        *,
        refresh_readiness: bool = True,
    ) -> None:
        """Compose project prep or workflow-owned status pages for the route."""
        card = getattr(self, "workbench_status_card", None)
        tabs = getattr(self, "workbench_status_tabs", None)
        if card is None or tabs is None:
            return

        current_route = route or self._current_shell_route()
        on_workbench = (
            current_route == _SHELL_ROUTE_PROJECT_PREPARE
            or current_route.startswith(_SHELL_WORKBENCH_PREFIX)
        )
        project_route = current_route == _SHELL_ROUTE_PROJECT_PREPARE
        writeback_visible = (
            not project_route and self._work_mode_has_writeback_surface()
        )

        project_bar = getattr(self, "global_project_bar", None)
        if project_bar is not None:
            project_bar.setVisible(project_route)
        tabs.setTabVisible(_BATCH_STAGE_PREPARE, project_route)
        tabs.setTabVisible(_BATCH_STAGE_EXECUTE, not project_route)
        tabs.setTabVisible(_BATCH_STAGE_RESULT, writeback_visible)

        target_index = tabs.currentIndex()
        if project_route:
            target_index = _BATCH_STAGE_PREPARE
        elif target_index == _BATCH_STAGE_PREPARE or (
            target_index == _BATCH_STAGE_RESULT and not writeback_visible
        ):
            target_index = _BATCH_STAGE_EXECUTE
        if tabs.currentIndex() != target_index:
            blocked = tabs.blockSignals(True)
            tabs.setCurrentIndex(target_index)
            tabs.blockSignals(blocked)
            self._sync_workbench_status_chrome(
                stage_index=target_index,
                refresh_readiness=refresh_readiness,
            )

        card.setVisible(on_workbench)
        tabs.setVisible(on_workbench)
        card.updateGeometry()
        primary = getattr(self, "workbench_primary", None)
        if primary is not None and primary.layout() is not None:
            primary.layout().invalidate()
            primary.layout().activate()
        self._layout_sync_timer.start()

    def _setup_shell_status_bar(self) -> None:
        status_bar = self.statusBar()
        status_bar.setObjectName("app_status_bar")
        self.safety_status_label = QLabel()
        self.safety_status_label.setObjectName("safety_status_badge")
        status_bar.addPermanentWidget(self.safety_status_label)
        self._refresh_shell_status()

    @staticmethod
    def _repolish_widget(widget: QWidget) -> None:
        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)
        widget.update()

    def _refresh_shell_status(self) -> None:
        running = bool(getattr(self, "_task_running", False))
        header_status = getattr(self, "header_task_status_label", None)
        if header_status is not None:
            if not running:
                status_text = "待命"
            elif self._is_doctor_active():
                status_text = "环境检查中"
            elif self._is_bootstrap_work_active():
                status_text = "准备工作目录中"
            elif self._is_generate_template_active():
                status_text = "生成模板中"
            elif getattr(self, "_active_command", "") in {
                "bootstrap_rag",
                "bootstrap_source_index",
            }:
                status_text = "预建中"
            else:
                status_text = "任务运行中"
            header_status.setText(status_text)
            header_status.setProperty("status", "running" if running else "idle")
            self._repolish_widget(header_status)

        badge = getattr(self, "safety_status_label", None)
        if badge is None:
            return
        summary = self._current_writeback_summary()
        status = summary.status
        if status == "safe":
            if summary.can_apply:
                kind, text = "safe", "写回门禁 · 可安全写回"
            else:
                kind, text = "safe", "写回门禁 · 安全完成，无需写回"
        elif status == "applied":
            kind, text = "safe", "写回门禁 · 已完成写回"
        elif status == "running":
            kind, text = "running", "写回门禁 · 检查中"
        elif status in {"warn", "stale"}:
            kind, text = "warning", "写回门禁 · 需要处理或重查"
        elif status in {"block", "failed", "unknown"}:
            kind, text = "danger", "写回门禁 · 禁止写回"
        else:
            kind, text = "idle", "写回门禁 · 尚未检查"
        badge.setText(text)
        badge.setProperty("status", kind)
        self._repolish_widget(badge)

    def _build_global_project_bar(self) -> QFrame:
        """Always-visible project path + switch entries (GUI IA P0b / #159)."""
        bar = QFrame()
        bar.setObjectName("global_project_bar")
        self.global_project_bar = bar
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(6)

        title = QLabel("当前项目")
        title.setObjectName("global_project_bar_label")
        self.global_project_bar_label = title
        outer.addWidget(title)

        self.global_project_path_edit = QLineEdit("尚未选择项目")
        self.global_project_path_edit.setReadOnly(True)
        self.global_project_path_edit.setObjectName("global_project_path_edit")
        outer.addWidget(self.global_project_path_edit)
        # Keep legacy objectName for mono-font QSS / tests that still look up project_path_edit.
        self.project_path_edit = self.global_project_path_edit
        # Buttons wrap under the path on narrow windows instead of colliding.
        self.global_project_actions = FlowButtonBar(spacing=8)
        self.global_project_actions.setObjectName("global_project_actions")
        self.global_switch_project_btn = QPushButton("切换项目")
        self.global_switch_project_btn.setObjectName("secondary_btn")
        self.global_switch_project_btn.setToolTip(
            "打开设置 → 项目列表，从项目总表选择并切换当前 game_root。"
        )
        self.global_switch_project_btn.clicked.connect(self._on_global_switch_project)
        self.global_project_actions.add_widget(self.global_switch_project_btn, min_width=108)

        self.global_browse_project_btn = QPushButton("指定本地目录…")
        self.global_browse_project_btn.setObjectName("secondary_btn")
        self.global_browse_project_btn.setToolTip(
            "通过文件夹对话框指定本地路径（可与总表无关）；会立即写入 game_root。"
        )
        self.global_browse_project_btn.clicked.connect(self._on_select_project)
        self.global_project_actions.add_widget(self.global_browse_project_btn, min_width=108)
        # Alias for existing enable/disable paths that still reference select_btn.
        self.select_btn = self.global_browse_project_btn

        # Project-level prep lives with directory switch (not per-task action row).
        self.doctor_btn = QPushButton("环境检查")
        self.doctor_btn.setObjectName("secondary_btn")
        self.doctor_btn.setToolTip("环境检查 (Ctrl+D)")
        self.doctor_btn.clicked.connect(self._on_doctor_button_clicked)
        self.global_project_actions.add_widget(self.doctor_btn, min_width=108)

        self.bootstrap_work_btn = QPushButton("准备工作目录")
        self.bootstrap_work_btn.setObjectName("secondary_btn")
        self.bootstrap_work_btn.setToolTip(
            "从 original/game 复制到 work/game（若已存在则按 CLI 规则跳过或更新）。"
        )
        self.bootstrap_work_btn.clicked.connect(self._on_bootstrap_button_clicked)
        self.global_project_actions.add_widget(self.bootstrap_work_btn, min_width=108)

        self.global_project_actions.finish_setup()
        outer.addWidget(self.global_project_actions)

        return bar

    def _build_workbench_tab(self) -> None:
        tab = QWidget()
        tab.setObjectName("workbench_tab")
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Left fixed task navigation (GUI IA P1a / #160).
        self.workbench_nav = QListWidget()
        self.workbench_nav.setObjectName("workbench_nav")
        self.workbench_nav.setFixedWidth(148)
        self.workbench_nav.setSpacing(2)
        self.workbench_nav.setFrameShape(QFrame.Shape.NoFrame)
        for nav_item in WORKBENCH_NAV_ORDER:
            item = QListWidgetItem(workbench_nav_spec(nav_item).label)
            item.setData(Qt.ItemDataRole.UserRole, nav_item.value)
            self.workbench_nav.addItem(item)
        blocked_nav = self.workbench_nav.blockSignals(True)
        self.workbench_nav.setCurrentRow(0)
        self.workbench_nav.blockSignals(blocked_nav)
        self.workbench_nav.currentRowChanged.connect(self._on_workbench_nav_changed)
        outer.addWidget(self.workbench_nav)
        self.workbench_nav.hide()

        # Scroll the workbench body so dense chrome (stacked actions + advanced +
        # stages + writeback) never crushes buttons into each other on short windows.
        right_scroll = QScrollArea()
        right_scroll.setObjectName("workbench_content_scroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._style_themed_surface(right_scroll)
        right_scroll.viewport().setObjectName("workbench_content_viewport")
        self._style_themed_surface(right_scroll.viewport())
        right = QWidget()
        right.setObjectName("workbench_content")
        self._style_themed_surface(right)
        layout = QVBoxLayout(right)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(14)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        right_scroll.setWidget(right)
        outer.addWidget(right_scroll, 1)

        # P1: context is the first real, persistent stack page. Other entries
        # remain migration placeholders until their own phase.
        self.workbench_stack = QStackedWidget()
        self.workbench_stack.setObjectName("workbench_stack")
        self.workbench_stack.setMinimumWidth(340)
        self._workbench_stack_pages: dict[WorkbenchNavItem, QWidget] = {}
        for nav_item in WORKBENCH_NAV_ORDER:
            if nav_item == WorkbenchNavItem.BATCH_TRANSLATION:
                page = BatchTranslationPage()
                page.set_action_callbacks(
                    WorkbenchPageActions(action=self._on_batch_translation_page_action)
                )
                self.batch_translation_page = page
            elif nav_item == WorkbenchNavItem.CONTEXT:
                page = ContextLibraryPage()
                page.set_action_callbacks(
                    WorkbenchPageActions(
                        prebuild=self._on_context_bootstrap_clicked,
                        open_settings=self._on_open_context_settings,
                        stop=self._on_kill,
                    )
                )
                self.context_library_page = page
                self.context_library_panel = page
                self.context_rag_status_label = page.rag_status_label
                self.context_source_index_status_label = page.source_index_status_label
                self.context_bootstrap_rag_btn = page.bootstrap_rag_btn
                self.context_bootstrap_source_index_btn = page.bootstrap_source_index_btn
                self.context_open_settings_btn = page.open_settings_btn
            elif nav_item == WorkbenchNavItem.SYNC_TRANSLATION:
                page = SyncTranslationPage()
                page.set_action_callbacks(
                    WorkbenchPageActions(
                        start=self._on_start_translation,
                        stop=self._on_kill,
                        writeback=self._on_apply_sync_translation,
                    )
                )
                self.sync_translation_page = page
            elif nav_item == WorkbenchNavItem.KEYWORDS:
                page = KeywordsPage()
                page.set_action_callbacks(
                    WorkbenchPageActions(
                        start=self._on_start_translation,
                        resume=self._on_resume_translation,
                        stop=self._on_kill,
                        writeback=self._on_open_keyword_merge,
                        select_mode=self._on_keywords_page_mode_selected,
                    )
                )
                self.keywords_page = page
            elif nav_item == WorkbenchNavItem.REVISION:
                page = RevisionPage()
                page.set_action_callbacks(
                    WorkbenchPageActions(
                        start=self._on_start_translation,
                        resume=self._on_resume_translation,
                        stop=self._on_kill,
                        writeback=self._on_apply_revision,
                        select_mode=self._on_revision_page_mode_selected,
                    )
                )
                self.revision_page = page
            else:
                page = QWidget()
                page.setObjectName(f"workbench_page_{nav_item.value}")
                page_layout = QVBoxLayout(page)
                page_layout.setContentsMargins(0, 0, 0, 0)
                page_layout.setSpacing(0)
            self._workbench_stack_pages[nav_item] = page
            self.workbench_stack.addWidget(page)
        self._workbench_coordinator = WorkbenchPageCoordinator(
            self.workbench_stack, self._workbench_stack_pages
        )
        self.workbench_primary = QFrame()
        self.workbench_primary.setObjectName("workbench_primary")
        primary_layout = QVBoxLayout(self.workbench_primary)
        primary_layout.setContentsMargins(0, 0, 0, 0)
        primary_layout.setSpacing(14)
        self.project_environment_bar = self._build_global_project_bar()
        primary_layout.addWidget(self.project_environment_bar)
        primary_layout.addWidget(
            self.workbench_stack,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        layout.addWidget(self.workbench_primary, 1)

        # Keep redirect notice near the project/workflow content.
        self.project_redirect_label = QLabel()
        self.project_redirect_label.setWordWrap(True)
        self.project_redirect_label.setObjectName("config_hint_label")
        self.project_redirect_label.setVisible(False)
        layout.addWidget(self.project_redirect_label)

        mode_frame = QFrame()
        mode_frame.setObjectName("mode_frame")
        self._mode_frame = mode_frame
        mode_outer = QVBoxLayout(mode_frame)
        mode_outer.setContentsMargins(12, 8, 12, 8)
        mode_outer.setSpacing(6)

        self.work_mode_hint_label = QLabel()
        self.work_mode_hint_label.setWordWrap(True)
        self.work_mode_hint_label.setObjectName("work_mode_hint_label")
        self.work_mode_hint_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.work_mode_hint_label.installEventFilter(self)
        mode_outer.addWidget(self.work_mode_hint_label)

        # Sync-only risk banner (P1c / #162 · §3.2.2).
        self.sync_mode_warning = QLabel(
            "同步翻译默认只生成差异预览；确认预览后才会修改项目脚本。"
        )
        self.sync_mode_warning.setObjectName("sync_mode_warning")
        self.sync_mode_warning.setWordWrap(True)
        self.sync_mode_warning.setVisible(False)
        mode_outer.addWidget(self.sync_mode_warning)
        layout.addWidget(mode_frame)

        action_frame = QFrame()
        action_frame.setObjectName("action_frame")
        self._action_frame = action_frame
        action_outer = QVBoxLayout(action_frame)
        action_outer.setContentsMargins(12, 10, 12, 10)
        action_outer.setSpacing(8)
        action_outer.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Task actions only (project prep is on the global project bar).
        self.action_panel = ResponsiveActionPanel(compact_width=640)
        self.translate_group_label = self.action_panel.translate_label
        self.translate_btn = self.action_panel.add_translate_button(QPushButton("开始翻译"))
        self.translate_btn.setObjectName("translate_btn")
        self.translate_btn.clicked.connect(self._on_start_translation)
        self.resume_btn = self.action_panel.add_translate_button(QPushButton("继续翻译"))
        self.resume_btn.setObjectName("secondary_btn")
        self.resume_btn.clicked.connect(self._on_resume_translation)
        self.split_submit_btn = self.action_panel.add_translate_button(QPushButton("提交剩余包"))
        self.split_submit_btn.setObjectName("secondary_btn")
        self.split_submit_btn.setToolTip("提交当前拆分组中尚未提交的包")
        self.split_submit_btn.clicked.connect(self._on_submit_remaining_split_packages)
        self.split_submit_btn.setVisible(False)
        self.kill_btn = self.action_panel.add_translate_trailing(QPushButton("停止"))
        self.kill_btn.setObjectName("kill_btn")
        self.kill_btn.clicked.connect(self._on_kill)
        self.kill_btn.setEnabled(False)
        self.action_panel.finish_setup()
        action_outer.addWidget(self.action_panel)
        # When the panel stacks to two rows it needs vertical room; don't let the
        # parent VBox crush it under the advanced-tools strip (overlap bug).
        action_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        # Keep main actions + batch advanced tools in one column so stacked
        # action height always pushes probe/split down (no sibling-layout race).
        actions_column = QWidget()
        actions_column.setObjectName("workbench_actions_column")
        self._workbench_actions_column = actions_column
        actions_column_layout = QVBoxLayout(actions_column)
        actions_column_layout.setContentsMargins(0, 0, 0, 0)
        actions_column_layout.setSpacing(10)
        actions_column_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        actions_column.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        actions_column_layout.addWidget(action_frame)
        # Batch · 翻译进度 · 高级：试跑 / 拆分 (P2a / #164); 提交剩余包仍在主按钮行。
        actions_column_layout.addWidget(self._build_batch_advanced_tools_bar())
        layout.addWidget(actions_column)

        self.timeline = WizardTimeline()
        self.timeline.setObjectName("workbench_timeline")
        self.timeline.setVisible(False)
        layout.addWidget(self.timeline)

        # Route-owned status surface: environment on the project page, progress
        # and optional writeback below each workflow page.
        status_card = QFrame()
        status_card.setObjectName("workbench_status_card")
        self.workbench_status_card = status_card
        self._style_themed_surface(status_card)
        status_card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        status_card.setMinimumWidth(300)
        status_card_layout = QVBoxLayout(status_card)
        status_card_layout.setContentsMargins(0, 0, 0, 0)
        status_card_layout.setSpacing(0)

        self.workbench_status_tabs = NoWheelTabWidget()
        self.workbench_status_tabs.setObjectName("workbench_status_tabs")
        self.workbench_status_tabs.setDocumentMode(True)
        self.workbench_status_tabs.tabBar().setExpanding(True)
        self.workbench_status_tabs.tabBar().setUsesScrollButtons(False)
        self.workbench_status_tabs.setMinimumHeight(200)
        self.workbench_status_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        # Connect currentChanged after all status tabs exist and the initial
        # index is set — addTab/setCurrentIndex would otherwise fire chrome
        # sync (manifest history walk) during cold construction.
        status_card_layout.addWidget(self.workbench_status_tabs)

        doctor_tab = QWidget()
        doctor_tab.setObjectName("workbench_doctor_page")
        self._style_themed_surface(doctor_tab)
        doctor_layout = QVBoxLayout(doctor_tab)
        doctor_layout.setContentsMargins(12, 12, 12, 12)
        doctor_layout.setSpacing(6)
        # Stack empty CTA vs summary so they never share the same column height
        # (VBox + dual stretch caused multi-line empty description to collapse).
        self.doctor_page_stack = QStackedWidget()
        self.doctor_page_stack.setObjectName("doctor_page_stack")
        doctor_layout.addWidget(self.doctor_page_stack, 1)

        doctor_summary_page = QWidget()
        doctor_summary_page.setObjectName("doctor_summary_page")
        self._style_themed_surface(doctor_summary_page)
        doctor_summary_layout = QVBoxLayout(doctor_summary_page)
        doctor_summary_layout.setContentsMargins(0, 0, 0, 0)
        doctor_summary_layout.setSpacing(6)
        self.doctor_status_label = StatusBadge("doctor_status_label")
        doctor_summary_layout.addWidget(self.doctor_status_label)
        self.doctor_summary_scroll = QScrollArea()
        self.doctor_summary_scroll.setObjectName("doctor_summary_scroll")
        self._style_themed_surface(self.doctor_summary_scroll)
        self.doctor_summary_scroll.setWidgetResizable(True)
        self.doctor_summary_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.doctor_summary_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        doctor_viewport = self.doctor_summary_scroll.viewport()
        doctor_viewport.setObjectName("doctor_summary_viewport")
        self._style_themed_surface(doctor_viewport)
        doctor_content = QWidget()
        doctor_content_layout = QVBoxLayout(doctor_content)
        doctor_content_layout.setContentsMargins(0, 0, 0, 0)
        doctor_content_layout.setSpacing(6)
        self.doctor_message_label = QLabel()
        self.doctor_message_label.setWordWrap(True)
        self.doctor_message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.doctor_message_label.setObjectName("summary_body_label")
        doctor_content_layout.addWidget(self.doctor_message_label)
        self.doctor_facts_label = QLabel()
        self.doctor_facts_label.setWordWrap(True)
        self.doctor_facts_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.doctor_facts_label.setObjectName("doctor_facts_label")
        doctor_content_layout.addWidget(self.doctor_facts_label)
        # Flat disclosure control (not checkable — checked QToolButtons pick up
        # selected-button chrome and change width when the label flips).
        self._doctor_details_expanded = False
        self.doctor_details_toggle = QToolButton()
        self.doctor_details_toggle.setObjectName("doctor_details_toggle")
        self.doctor_details_toggle.setCheckable(False)
        self.doctor_details_toggle.setAutoRaise(True)
        self.doctor_details_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.doctor_details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.doctor_details_toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.doctor_details_toggle.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        # Pin width so ▸/▾ expand/collapse does not nudge layout.
        # Measure only the stable Chinese label — never pass ▾/▸ to
        # QFontMetrics.horizontalAdvance at cold start: the first resolve of
        # those decorative glyphs can cost hundreds of ms via font fallback.
        _toggle_fm = self.doctor_details_toggle.fontMetrics()
        _label_w = _toggle_fm.horizontalAdvance("更多详情")
        _marker_w = max(_toggle_fm.averageCharWidth() * 2, 16)
        self.doctor_details_toggle.setFixedWidth(max(_label_w + _marker_w + 8, 96))
        self.doctor_details_toggle.setVisible(False)
        self.doctor_details_toggle.clicked.connect(self._on_doctor_details_clicked)
        doctor_content_layout.addWidget(
            self.doctor_details_toggle, 0, Qt.AlignmentFlag.AlignLeft
        )
        self.doctor_details_label = QLabel()
        self.doctor_details_label.setWordWrap(True)
        self.doctor_details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.doctor_details_label.setObjectName("doctor_details_label")
        self.doctor_details_label.setVisible(False)
        doctor_content_layout.addWidget(self.doctor_details_label)
        self._sync_doctor_details_toggle_chrome()
        doctor_content_layout.addStretch()
        self.doctor_summary_scroll.setWidget(doctor_content)
        doctor_summary_layout.addWidget(self.doctor_summary_scroll, 1)
        self.doctor_page_stack.addWidget(doctor_summary_page)

        # P3 / #166: empty CTA before the first environment check.
        self.doctor_empty_state = EmptyStateWidget(
            "",
            "尚未运行环境检查",
            "完成检查后这里会显示项目就绪状态与建议下一步。",
            action_text="运行环境检查",
        )
        self.doctor_empty_state.setObjectName("doctor_empty_state")
        self.doctor_empty_state.action_clicked.connect(self._on_run_doctor)
        self.doctor_page_stack.addWidget(self.doctor_empty_state)
        self.doctor_page_stack.setCurrentWidget(self.doctor_empty_state)
        self.workbench_status_tabs.addTab(doctor_tab, "环境检查")

        workflow_tab = QWidget()
        workflow_tab.setObjectName("workbench_workflow_page")
        self._style_themed_surface(workflow_tab)
        workflow_outer_layout = QVBoxLayout(workflow_tab)
        workflow_outer_layout.setContentsMargins(0, 0, 0, 0)
        workflow_outer_layout.setSpacing(0)
        self.workflow_scroll = QScrollArea()
        self.workflow_scroll.setObjectName("workflow_summary_scroll")
        self._style_themed_surface(self.workflow_scroll)
        self.workflow_scroll.setWidgetResizable(True)
        self.workflow_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.workflow_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        workflow_viewport = self.workflow_scroll.viewport()
        workflow_viewport.setObjectName("workflow_summary_viewport")
        self._style_themed_surface(workflow_viewport)
        workflow_content = QWidget()
        workflow_content.setObjectName("workflow_summary_content")
        self._style_themed_surface(workflow_content)
        workflow_layout = QVBoxLayout(workflow_content)
        workflow_layout.setContentsMargins(12, 12, 12, 12)
        workflow_layout.setSpacing(6)
        self.workflow_status_label = StatusBadge("workflow_status_label")
        workflow_layout.addWidget(self.workflow_status_label)
        # P3 / #166: empty CTA when there is no project / no resumable progress yet.
        self.workflow_empty_state = EmptyStateWidget(
            "",
            "还没有任务进度",
            "选择项目并开始翻译后，这里会显示时间线与任务事实。"
            "若已有未完成的批量任务，可点「继续」。",
            action_text="去环境检查",
        )
        self.workflow_empty_state.setObjectName("workflow_empty_state")
        # Environment check lives on the project-prepare shell route; the
        # prepare status tab is hidden on task pages, so tab-index focus alone
        # would be remapped to 进度 and appear to do nothing.
        self.workflow_empty_state.action_clicked.connect(
            lambda: self._activate_shell_route(_SHELL_ROUTE_PROJECT_PREPARE)
        )
        workflow_layout.addWidget(self.workflow_empty_state)
        self.workflow_empty_state.setVisible(False)
        self.workflow_message_label = QLabel()
        self.workflow_message_label.setWordWrap(True)
        self.workflow_message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.workflow_message_label.setObjectName("summary_body_label")
        workflow_layout.addWidget(self.workflow_message_label)
        self.workflow_progress_bar = QProgressBar()
        self.workflow_progress_bar.setObjectName("workflow_progress_bar")
        self.workflow_progress_bar.setVisible(False)
        self.workflow_progress_bar.setTextVisible(True)
        workflow_layout.addWidget(self.workflow_progress_bar)
        self.workflow_facts_label = QLabel()
        self.workflow_facts_label.setWordWrap(True)
        self.workflow_facts_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.workflow_facts_label.setObjectName("workflow_facts_label")
        workflow_layout.addWidget(self.workflow_facts_label)
        completed_actions = QHBoxLayout()
        completed_actions.setSpacing(8)
        self.view_last_completed_btn = QPushButton("查看上次已完成任务")
        self.view_last_completed_btn.setObjectName("secondary_btn")
        self.view_last_completed_btn.setVisible(False)
        self.view_last_completed_btn.clicked.connect(self._on_view_last_completed_task)
        completed_actions.addWidget(self.view_last_completed_btn)
        self.hide_completed_view_btn = QPushButton("返回概览")
        self.hide_completed_view_btn.setObjectName("secondary_btn")
        self.hide_completed_view_btn.setVisible(False)
        self.hide_completed_view_btn.clicked.connect(self._on_hide_completed_manifest_view)
        completed_actions.addWidget(self.hide_completed_view_btn)
        completed_actions.addStretch()
        workflow_layout.addLayout(completed_actions)
        self.split_status_title = QLabel("拆分包状态")
        self.split_status_title.setObjectName("diagnostics_section_label")
        self.split_status_title.setVisible(False)
        workflow_layout.addWidget(self.split_status_title)
        self.split_status_table = QTableWidget(0, 6)
        self.split_status_table.setObjectName("split_status_table")
        self.split_status_table.setHorizontalHeaderLabels(["包", "状态", "项", "块", "云端任务", "操作"])
        self.split_status_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.split_status_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.split_status_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.split_status_table.setAlternatingRowColors(True)
        self.split_status_table.setWordWrap(False)
        self.split_status_table.setMinimumHeight(260)
        self.split_status_table.setMaximumHeight(360)
        self.split_status_table.verticalHeader().setVisible(False)
        self._configure_split_status_table_hover_palette()
        self.split_status_table.viewport().installEventFilter(self)
        self._split_status_action_delegate = SplitStatusActionDelegate(self.split_status_table)
        self._split_status_action_delegate.select_requested.connect(self._select_split_manifest)
        split_header = self.split_status_table.horizontalHeader()
        split_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        split_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        split_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        split_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        split_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        split_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self._configure_split_status_table_columns()
        self.split_status_table.setVisible(False)
        workflow_layout.addWidget(self.split_status_table)
        workflow_layout.addStretch()
        self.workflow_scroll.setWidget(workflow_content)
        workflow_outer_layout.addWidget(self.workflow_scroll, 1)
        self.workbench_status_tabs.addTab(workflow_tab, "翻译进度")

        writeback_tab = QWidget()
        writeback_tab.setObjectName("workbench_writeback_page")
        self._style_themed_surface(writeback_tab)
        writeback_layout = QVBoxLayout(writeback_tab)
        writeback_layout.setContentsMargins(12, 12, 12, 12)
        writeback_layout.setSpacing(10)
        writeback_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.writeback_status_label = StatusBadge("writeback_status_label")
        writeback_layout.addWidget(self.writeback_status_label)
        writeback_scroll = QScrollArea()
        writeback_scroll.setObjectName("writeback_summary_scroll")
        self._style_themed_surface(writeback_scroll)
        writeback_scroll.setWidgetResizable(True)
        writeback_scroll.setFrameShape(QFrame.Shape.NoFrame)
        writeback_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        writeback_viewport = writeback_scroll.viewport()
        writeback_viewport.setObjectName("writeback_summary_viewport")
        self._style_themed_surface(writeback_viewport)
        writeback_content = QWidget()
        writeback_content.setObjectName("writeback_summary_content")
        self._style_themed_surface(writeback_content)
        writeback_content_layout = QVBoxLayout(writeback_content)
        writeback_content_layout.setContentsMargins(0, 0, 0, 0)
        writeback_content_layout.setSpacing(6)
        self.writeback_message_label = QLabel()
        self.writeback_message_label.setWordWrap(True)
        self.writeback_message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.writeback_message_label.setObjectName("summary_body_label")
        writeback_content_layout.addWidget(self.writeback_message_label)
        self.writeback_facts_label = QLabel()
        self.writeback_facts_label.setWordWrap(True)
        self.writeback_facts_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.writeback_facts_label.setObjectName("writeback_facts_label")
        writeback_content_layout.addWidget(self.writeback_facts_label)
        self.writeback_details_label = QLabel()
        self.writeback_details_label.setWordWrap(True)
        self.writeback_details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.writeback_details_label.setObjectName("config_hint_label")
        self.writeback_details_label.setVisible(False)
        writeback_content_layout.addWidget(self.writeback_details_label)
        writeback_content_layout.addStretch()
        writeback_scroll.setWidget(writeback_content)
        writeback_layout.addWidget(writeback_scroll, 1)

        # Primary writeback row: main action(s) + 「问题处理」 toggle on one strip
        # so fullscreen result view is not a vertical stack of separate toolbars.
        primary_row = QHBoxLayout()
        primary_row.setSpacing(8)
        primary_row.setContentsMargins(0, 0, 0, 0)

        self.writeback_primary_bar = FlowButtonBar(spacing=8)
        self.writeback_primary_bar.setObjectName("writeback_primary_bar")
        self.apply_btn = QPushButton("写回翻译")
        self.apply_btn.setObjectName("apply_btn")
        self.apply_btn.clicked.connect(self._on_apply_writeback)
        self.apply_btn.setEnabled(False)
        self.writeback_primary_bar.add_widget(self.apply_btn, min_width=96)
        self.apply_revision_btn = QPushButton("写回订正")
        self.apply_revision_btn.setObjectName("apply_revision_btn")
        self.apply_revision_btn.clicked.connect(self._on_apply_revision)
        self.apply_revision_btn.setEnabled(False)
        self.apply_revision_btn.setVisible(False)
        self.writeback_primary_bar.add_widget(self.apply_revision_btn, min_width=96)
        self.keyword_merge_writeback_btn = QPushButton("合并到 glossary")
        self.keyword_merge_writeback_btn.setObjectName("secondary_btn")
        self.keyword_merge_writeback_btn.setToolTip(
            "勾选审核关键词候选并写入 glossary.json；不会修改 .rpy 脚本。"
        )
        self.keyword_merge_writeback_btn.clicked.connect(self._on_open_keyword_merge)
        self.keyword_merge_writeback_btn.setEnabled(False)
        self.keyword_merge_writeback_btn.setVisible(False)
        self.writeback_primary_bar.add_widget(self.keyword_merge_writeback_btn, min_width=120)
        self.writeback_primary_bar.finish_setup()
        primary_row.addWidget(self.writeback_primary_bar, 1)

        # Recovery tools collapse under 「问题处理」 (GUI IA P0b / #159; 重新检查 included).
        self.writeback_issues_toggle_btn = QPushButton("问题处理 ▸")
        self.writeback_issues_toggle_btn.setObjectName("secondary_btn")
        self.writeback_issues_toggle_btn.setToolTip(
            "展开补译、修补、问题清单、重新检查等恢复操作；检查为「需处理」时会自动提示。"
        )
        self.writeback_issues_toggle_btn.clicked.connect(self._toggle_writeback_issues_panel)
        self.writeback_issues_toggle_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self.writeback_issues_toggle_btn.setMinimumWidth(100)
        primary_row.addWidget(self.writeback_issues_toggle_btn, 0)
        self.writeback_issues_badge = QLabel("")
        self.writeback_issues_badge.setObjectName("writeback_issues_badge")
        self.writeback_issues_badge.setVisible(False)
        primary_row.addWidget(self.writeback_issues_badge, 0)
        writeback_layout.addLayout(primary_row)

        # Right-align under the primary-row 「问题处理」 toggle.
        self.writeback_issues_panel = FlowButtonBar(
            spacing=8, row_spacing=8, align="right"
        )
        self.writeback_issues_panel.setObjectName("writeback_issues_panel")
        self.check_issues_btn = QPushButton("查看问题清单")
        self.check_issues_btn.setObjectName("secondary_btn")
        self.check_issues_btn.clicked.connect(self._open_check_issues)
        self.check_issues_btn.setEnabled(False)
        self.writeback_issues_panel.add_widget(self.check_issues_btn, min_width=100)
        self.retry_btn = QPushButton("生成补译包")
        self.retry_btn.setObjectName("secondary_btn")
        self.retry_btn.clicked.connect(self._on_retry_action)
        self.retry_btn.setEnabled(False)
        self.retry_btn.setVisible(False)
        self.writeback_issues_panel.add_widget(self.retry_btn, min_width=96)
        self.retry_followup_btn = QPushButton("继续补译")
        self.retry_followup_btn.setObjectName("secondary_btn")
        self.retry_followup_btn.clicked.connect(self._on_retry_followup_action)
        self.retry_followup_btn.setEnabled(False)
        self.retry_followup_btn.setVisible(False)
        self.writeback_issues_panel.add_widget(self.retry_followup_btn, min_width=88)
        self.repair_btn = QPushButton("同步修补")
        self.repair_btn.setObjectName("secondary_btn")
        self.repair_btn.setToolTip(
            "对 repair 类问题执行同步修补；会直接修改翻译文件，请先备份。"
        )
        self.repair_btn.clicked.connect(self._on_run_repair)
        self.repair_btn.setEnabled(False)
        self.repair_btn.setVisible(False)
        self.writeback_issues_panel.add_widget(self.repair_btn, min_width=88)
        # Design SSOT: 重新检查 lives under 「问题处理」, not next to 写回翻译.
        self.recheck_btn = QPushButton("重新检查")
        self.recheck_btn.setObjectName("secondary_btn")
        self.recheck_btn.clicked.connect(self._on_recheck_writeback)
        self.recheck_btn.setEnabled(False)
        self.recheck_btn.setVisible(False)
        self.writeback_issues_panel.add_widget(self.recheck_btn, min_width=88)
        self.apply_failure_btn = QPushButton("查看写回失败报告")
        self.apply_failure_btn.setObjectName("secondary_btn")
        self.apply_failure_btn.clicked.connect(self._open_apply_failure_report)
        self.apply_failure_btn.setEnabled(False)
        self.apply_failure_btn.setVisible(False)
        self.writeback_issues_panel.add_widget(self.apply_failure_btn, min_width=120)
        self.remediation_btn = QPushButton("补救命令")
        self.remediation_btn.setObjectName("secondary_btn")
        self.remediation_btn.clicked.connect(self._open_remediation_commands)
        self.remediation_btn.setEnabled(False)
        self.writeback_issues_panel.add_widget(self.remediation_btn, min_width=88)
        self.writeback_issues_panel.finish_setup()
        writeback_layout.addWidget(self.writeback_issues_panel)

        self._writeback_issues_expanded = False
        self._set_writeback_issues_expanded(False)

        writeback_layout.addStretch()
        self.workbench_status_tabs.addTab(writeback_tab, "写回")

        # Select default stage before wiring currentChanged so construction does
        # not pay for per-tab chrome + manifest readiness scans.
        blocked_status = self.workbench_status_tabs.blockSignals(True)
        self.workbench_status_tabs.setCurrentIndex(_BATCH_STAGE_EXECUTE)
        self.workbench_status_tabs.blockSignals(blocked_status)
        self.workbench_status_tabs.currentChanged.connect(
            self._on_workbench_status_tab_changed
        )
        primary_layout.addWidget(self.workbench_status_card)

        # Real pages own all visible task actions. The old action widgets remain
        # internal readiness adapters until their workflow state is made widget-free.
        self._mode_frame.hide()
        self._workbench_actions_column.hide()

        self._workbench_tab = tab
        self.tab_widget.addTab(tab, "工作台")
        # Visibility/layout only — probe/split readiness waits for deferred startup.
        self._sync_workbench_status_chrome(refresh_readiness=False)
        self._sync_workbench_status_surface(refresh_readiness=False)

    def _build_batch_advanced_tools_bar(self) -> QFrame:
        """Batch execute advanced strip: probe + split (P2a / #164)."""
        frame = QFrame()
        frame.setObjectName("batch_advanced_frame")
        self.batch_advanced_frame = frame
        frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)

        title = QLabel("高级工具")
        title.setObjectName("action_group_label")
        outer.addWidget(title)

        self.batch_advanced_bar = FlowButtonBar(spacing=8)
        self.batch_advanced_bar.setObjectName("batch_advanced_bar")
        self.probe_btn = QPushButton("试跑样本请求")
        self.probe_btn.setObjectName("secondary_btn")
        self.probe_btn.setToolTip(
            "对当前翻译包执行少量同步请求，提交批量任务前验证 API 与请求格式。"
        )
        self.probe_btn.clicked.connect(self._on_run_probe)
        self.probe_btn.setEnabled(False)
        self.batch_advanced_bar.add_widget(self.probe_btn, min_width=108)

        self.split_btn = QPushButton("拆分翻译包")
        self.split_btn.setObjectName("secondary_btn")
        self.split_btn.setToolTip(
            "将过大的翻译包拆成多个子包；拆分后需分别提交，RAG 为静态快照。"
        )
        self.split_btn.clicked.connect(self._on_run_split)
        self.split_btn.setEnabled(False)
        self.batch_advanced_bar.add_widget(self.split_btn, min_width=96)
        self.batch_advanced_bar.finish_setup()
        outer.addWidget(self.batch_advanced_bar)

        frame.setVisible(False)
        return frame

    def _on_open_context_settings(self) -> None:
        self._focus_settings_section("context")

    def _on_context_bootstrap_clicked(self, kind: str) -> None:
        target = (
            WorkMode.BOOTSTRAP_RAG
            if kind == "rag"
            else WorkMode.BOOTSTRAP_SOURCE_INDEX
        )
        if self._current_work_mode() != target:
            self._set_work_mode(target, refresh_manifest_writeback=False)
        self._start_bootstrap_task(kind)

    def _refresh_context_library_panel(self, *, running: bool | None = None) -> None:
        page = getattr(self, "context_library_page", None)
        if page is not None:
            flags = self._saved_batch_context_flags()
            rag_on = bool(flags.get("rag_enabled"))
            idx_on = bool(flags.get("source_index_enabled"))
            game_root = self.state.get_game_root() if hasattr(self, "state") else None
            if running is None:
                running = self._task_page_running_chrome()
            page.set_context_status(
                rag_enabled=rag_on,
                source_index_enabled=idx_on,
                game_root=str(game_root or ""),
            )
            page.set_task_running(running)
            return

    def _refresh_active_workbench_page(self, spec) -> None:
        """Refresh the active real page and hide internal readiness adapters."""
        nav = workbench_nav_for_work_mode(spec.mode)
        self._workbench_coordinator.resize(nav)
        if nav == WorkbenchNavItem.CONTEXT:
            self._refresh_context_library_panel()

        self.translate_btn.setVisible(False)
        self.resume_btn.setVisible(False)
        self.work_mode_hint_label.setVisible(False)
        if nav == WorkbenchNavItem.KEYWORDS:
            self.keyword_merge_writeback_btn.setVisible(False)
            self.keyword_merge_writeback_btn.setEnabled(False)
        if nav == WorkbenchNavItem.REVISION:
            self.apply_revision_btn.setVisible(False)
            self.apply_revision_btn.setEnabled(False)
    def _style_themed_surface(self, widget: QWidget) -> None:
        widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def _build_config_tab(self) -> None:
        tab = QWidget()
        tab.setObjectName("config_tab")
        self._style_themed_surface(tab)
        self._config_tab = tab
        self._advanced_setting_widgets = {}
        self._advanced_setting_error_labels = {}
        self._settings_nav_rows = {}
        self._settings_page_bodies: dict[str, QWidget] = {}

        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.settings_nav = QListWidget()
        self.settings_nav.setObjectName("settings_nav")
        self.settings_nav.setFixedHeight(50)
        self.settings_nav.setSpacing(2)
        self.settings_nav.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_nav.setFlow(QListView.Flow.LeftToRight)
        self.settings_nav.setMovement(QListView.Movement.Static)
        self.settings_nav.setResizeMode(QListView.ResizeMode.Adjust)
        self.settings_nav.setWrapping(False)
        self.settings_nav.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.settings_nav.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        outer_layout.addWidget(self.settings_nav)

        right_panel = QWidget()
        right_panel.setObjectName("settings_right_panel")
        self._style_themed_surface(right_panel)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        outer_layout.addWidget(right_panel, 1)

        self.settings_stack = QStackedWidget()
        self.settings_stack.setObjectName("settings_stack")
        right_layout.addWidget(self.settings_stack, 1)

        self._settings_pages_built = set()
        self._settings_page_builders: dict[str, str] = {
            key: builder for key, _label, builder in _SETTINGS_PAGE_SPECS
        }
        # Placeholder stack pages — real content replaces them on first visit.
        for index, (key, label, _builder) in enumerate(_SETTINGS_PAGE_SPECS):
            self._settings_nav_rows[key] = index
            self.settings_nav.addItem(label)
            placeholder = QWidget()
            placeholder.setObjectName(f"settings_{key}_placeholder")
            self._style_themed_surface(placeholder)
            self.settings_stack.addWidget(placeholder)
        self.settings_nav.currentRowChanged.connect(self._on_settings_nav_row_changed)

        action_bar = QFrame()
        action_bar.setObjectName("settings_action_bar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(12, 10, 12, 10)
        action_layout.setSpacing(8)
        action_layout.addStretch()
        self.reload_config_btn = QPushButton("重新加载")
        self.reload_config_btn.setObjectName("secondary_btn")
        self.reload_config_btn.setToolTip(
            "从 translator_config.json 重新载入设置页字段。"
            "「项目列表」操作即时写入 registry，不受此按钮影响。"
        )
        self.reload_config_btn.clicked.connect(self._on_reload_config)
        action_layout.addWidget(self.reload_config_btn)
        self.restore_defaults_btn = QPushButton("恢复推荐值")
        self.restore_defaults_btn.setObjectName("secondary_btn")
        self.restore_defaults_btn.setToolTip(
            "将模型、上下文主开关、高级参数等恢复为推荐值（仅填入界面，需再点保存）。"
            "不改写项目列表 registry。"
        )
        self.restore_defaults_btn.clicked.connect(self._on_restore_recommended_config)
        action_layout.addWidget(self.restore_defaults_btn)
        self.save_config_btn = QPushButton("保存设置")
        self.save_config_btn.setObjectName("save_config_btn")
        self.save_config_btn.setToolTip(
            "写入 translator_config.json（项目/模型/上下文/高级/外观等）。"
            "「项目列表」扫描与切换项目即时生效，无需此按钮。"
        )
        self.save_config_btn.clicked.connect(self._on_save_config)
        action_layout.addWidget(self.save_config_btn)
        right_layout.addWidget(action_bar)

        # Select first section without building it yet (signals blocked).
        blocked = self.settings_nav.blockSignals(True)
        self.settings_nav.setCurrentRow(0)
        self.settings_nav.blockSignals(blocked)
        self.settings_stack.setCurrentIndex(0)

        self.tab_widget.addTab(tab, "设置")

    def __getattr__(self, name: str) -> Any:
        """Materialize lazy settings pages when tests/code access their widgets."""
        if name.startswith("_") and name not in _SETTINGS_LAZY_ATTR_TO_PAGE:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            )
        page_key = _SETTINGS_LAZY_ATTR_TO_PAGE.get(name)
        if page_key is None or getattr(self, "_settings_lazy_resolving", False):
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            )
        self._settings_lazy_resolving = True
        try:
            if page_key in _SETTINGS_CONFIG_PAGE_KEYS:
                self._ensure_settings_pages_for_config()
            else:
                self._ensure_settings_page(page_key)
        finally:
            self._settings_lazy_resolving = False
        try:
            return object.__getattribute__(self, name)
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            ) from exc

    def _settings_lazy_ready(self) -> bool:
        """True when the full settings shell exists (not a MainWindow.__new__ stub)."""
        return bool(
            getattr(self, "_settings_page_builders", None)
            and getattr(self, "settings_stack", None) is not None
            and "state" in self.__dict__
        )

    def _ensure_settings_page(self, key: str, *, populate: bool = True) -> None:
        """Build one settings section and swap it into the settings stack."""
        if not self._settings_lazy_ready():
            return
        built = getattr(self, "_settings_pages_built", None)
        if built is None:
            return
        if key in built:
            return
        if populate and key in _SETTINGS_CONFIG_PAGE_KEYS:
            self._ensure_settings_pages_for_config()
            return
        builders = getattr(self, "_settings_page_builders", {})
        builder_name = builders.get(key)
        rows = getattr(self, "_settings_nav_rows", {})
        index = rows.get(key)
        stack = getattr(self, "settings_stack", None)
        if builder_name is None or index is None or stack is None:
            return
        builder = getattr(self, builder_name, None)
        if not callable(builder):
            return
        page = builder()
        old = stack.widget(index)
        current_index = stack.currentIndex()
        if old is not None:
            stack.removeWidget(old)
            old.deleteLater()
        stack.insertWidget(index, page)
        if current_index == index:
            stack.setCurrentIndex(index)
        built.add(key)
        if key == "models":
            self._wire_settings_models_signals()
        if key == "workspace":
            # Inherit current task gate if panel was built after a run started.
            panel = self.__dict__.get("_games_registry_panel")
            if panel is not None and bool(getattr(self, "_task_running", False)):
                set_gate = getattr(panel, "set_host_task_running", None)
                if callable(set_gate):
                    set_gate(True)

    def _ensure_settings_pages_for_config(self) -> None:
        """Ensure every section that participates in config load/save/dirty."""
        if not self._settings_lazy_ready():
            return
        built = getattr(self, "_settings_pages_built", set())
        pending = [
            key
            for key, _label, _builder in _SETTINGS_PAGE_SPECS
            if key in _SETTINGS_CONFIG_PAGE_KEYS and key not in built
        ]
        for key in pending:
            self._ensure_settings_page(key, populate=False)
        if pending and not getattr(self, "_loading_config_to_ui", False):
            self._load_config_to_ui(refresh_task_gates=False)
            if hasattr(self, "api_status_label"):
                self._refresh_api_status()

    def _wire_settings_models_signals(self) -> None:
        if getattr(self, "_settings_models_signals_wired", False):
            return
        if not hasattr(self, "batch_model_combo") or not hasattr(
            self, "batch_thinking_combo"
        ):
            return
        self.batch_model_combo.currentTextChanged.connect(self._on_batch_model_changed)
        self.batch_thinking_combo.currentIndexChanged.connect(
            self._on_batch_thinking_changed
        )
        self._settings_models_signals_wired = True

    def _settings_page(self, object_name: str) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setObjectName(f"{object_name}_scroll")
        self._style_themed_surface(scroll)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        viewport = scroll.viewport()
        viewport.setObjectName(f"{object_name}_viewport")
        self._style_themed_surface(viewport)

        content = QWidget()
        content.setObjectName(f"{object_name}_content")
        self._style_themed_surface(content)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        body = QWidget()
        body.setObjectName("settings_page_body")
        body.setProperty("settingsPage", object_name)
        # Fill the settings viewport (including fullscreen). A hard 1080px cap
        # left large empty gutters on wide / maximized windows.
        body.setMinimumWidth(0)
        body.setMaximumWidth(16777215)
        body.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        self._style_themed_surface(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(14)
        # Consume full viewport width; do not pin body to sizeHint via AlignHCenter.
        content_layout.addWidget(body, 1)
        self._settings_page_bodies[object_name] = body

        scroll.setWidget(content)
        return scroll, layout

    def _settings_group(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 18, 14, 14)
        return group, layout

    def _settings_form(self, group: QGroupBox) -> QFormLayout:
        form = QFormLayout(group)
        form.setObjectName("settings_form")
        form.setContentsMargins(14, 18, 14, 14)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        return form

    def _build_settings_workspace_page(self) -> QWidget:
        page, layout = self._settings_page("settings_workspace")
        hint = QLabel(
            "工作区默认未设置，不会自动使用工具目录的上一级。"
            "先「选择工作区…」指定存放 Game_* 的根目录，再扫描、导入或切换项目。"
            "「切换到此项目」会写入当前 game_root 并留在本页；术语表 / 准备流程等到「项目」分区调整。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        self._games_registry_panel = GamesRegistryPanel(
            None,
            workspace_root=self.state.get_workspace_root(),
            current_game_root=self.state.get_game_root(),
            get_doctor_report=self._current_registry_doctor_report,
            on_switch_project=self._on_registry_switch_project,
            on_workspace_changed=self._on_workspace_changed,
        )
        self._games_registry_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self._games_registry_panel, 1)
        return page

    def _build_settings_api_keys_page(self) -> QWidget:
        page, layout = self._settings_page("settings_api_keys")
        api_box, api_layout = self._settings_group("Gemini API Key")

        api_hint = QLabel(
            "此页面只管理 Gemini API 密钥。密钥保存在本地配置文件中，"
            "不会上传或代理。也可通过环境变量配置。"
        )
        api_hint.setWordWrap(True)
        api_hint.setObjectName("config_hint_label")
        api_layout.addWidget(api_hint)

        self.api_status_label = QLabel()
        self.api_status_label.setWordWrap(True)
        self.api_status_label.setObjectName("api_status_label")
        api_layout.addWidget(self.api_status_label)

        api_actions = QHBoxLayout()
        self.api_btn = QPushButton("管理 API Key")
        self.api_btn.setObjectName("api_btn")
        self.api_btn.clicked.connect(self._on_manage_api_keys)
        api_actions.addWidget(self.api_btn)
        api_actions.addStretch()
        api_layout.addLayout(api_actions)

        layout.addWidget(api_box)
        layout.addStretch(1)
        return page

    def _build_settings_project_page(self) -> QWidget:
        page, layout = self._settings_page("settings_project")
        hint = QLabel(
            "配置当前选中项目的 translator_config.json 参数。"
            "切换 work 目录请前往「项目列表」；此处只调整术语表、翻译目录、过滤器和准备流程。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        current_box = QGroupBox("当前项目")
        current_form = self._settings_form(current_box)
        self.settings_project_root_value = QLabel("（尚未选择项目）")
        self.settings_project_root_value.setWordWrap(True)
        self.settings_project_root_value.setObjectName("settings_project_root_value")
        current_form.addRow("游戏 work 目录：", self.settings_project_root_value)
        switch_row = QHBoxLayout()
        switch_row.addStretch(1)
        self.settings_go_workspace_btn = QPushButton("在项目列表切换…")
        self.settings_go_workspace_btn.setObjectName("secondary_btn")
        self.settings_go_workspace_btn.clicked.connect(self._on_go_to_workspace_for_project_switch)
        switch_row.addWidget(self.settings_go_workspace_btn)
        current_form.addRow("", switch_row)
        layout.addWidget(current_box)
        # Page may be lazy-built after a project switch; show the live game_root.
        self._refresh_settings_project_root_display()

        for group_title in ("项目与资源", "准备流程"):
            group = QGroupBox(group_title)
            form = self._settings_form(group)
            for field in [
                item
                for item in ADVANCED_SETTING_FIELDS
                if item.category == group_title and item.key not in _SETTINGS_WORKSPACE_MANAGED_KEYS
            ]:
                widget = self._create_advanced_setting_widget(field)
                self._advanced_setting_widgets[field.key] = widget
                row = self._advanced_setting_row(field, widget)
                form.addRow(f"{field.label}：", row)
            layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _build_settings_context_page(self) -> QWidget:
        page, layout = self._settings_page("settings_context")
        context_box, context_layout = self._settings_group("批量上下文")

        context_hint = QLabel(
            "下列三项按【当前项目】保存（work/project_context_settings.json），"
            "切换游戏不会互相覆盖。启用后先保存设置，再到工作台「上下文库」页预建。"
            "记忆库用已有译文；原文索引只用模板原文；均不修改游戏脚本。"
        )
        context_hint.setWordWrap(True)
        context_hint.setObjectName("config_hint_label")
        context_layout.addWidget(context_hint)

        self.rag_enabled_cb = QCheckBox("启用 RAG 记忆库（批量，当前项目）")
        context_layout.addWidget(self.rag_enabled_cb)

        self.source_index_enabled_cb = QCheckBox("启用原文索引（当前项目）")
        context_layout.addWidget(self.source_index_enabled_cb)

        self.bootstrap_on_build_cb = QCheckBox("开始翻译时自动暖 RAG 库（当前项目）")
        context_layout.addWidget(self.bootstrap_on_build_cb)

        self.context_storage_game_cb = QCheckBox("上下文库保存到游戏目录")
        self.context_storage_game_cb.setToolTip(
            "启用后，默认 RAG / 原文索引 / 剧情图谱路径会使用 work 同级的 translation_context/。"
        )
        context_layout.addWidget(self.context_storage_game_cb)

        layout.addWidget(context_box)

        # P2b / #165: primary toggles live here only (schema widgets = single write source).
        primary_box, primary_layout = self._settings_group("同步 / 剧情记忆主开关")
        primary_hint = QLabel(
            "下列开关与高级页检索参数共用同一配置键；仅在本页编辑主开关，"
            "高级页只保留调参字段。"
        )
        primary_hint.setWordWrap(True)
        primary_hint.setObjectName("config_hint_label")
        primary_layout.addWidget(primary_hint)
        for field in context_primary_setting_fields():
            widget = self._create_advanced_setting_widget(field)
            self._advanced_setting_widgets[field.key] = widget
            row = self._advanced_setting_row(field, widget)
            primary_layout.addWidget(QLabel(f"{field.label}"))
            primary_layout.addWidget(row)
        layout.addWidget(primary_box)
        layout.addStretch(1)
        return page

    def _build_settings_models_page(self) -> QWidget:
        page, layout = self._settings_page("settings_models")
        sync_box = QGroupBox("Gemini 同步翻译")
        sync_layout = self._settings_form(sync_box)

        self.sync_model_combo = NoWheelComboBox()
        self.sync_model_combo.setEditable(False)
        self.sync_model_combo.addItems(list(BUILTIN_GEMINI_TRANSLATION_MODELS))
        sync_layout.addRow("翻译模型：", self.sync_model_combo)

        self.sync_embedding_combo = NoWheelComboBox()
        self.sync_embedding_combo.setEditable(False)
        self.sync_embedding_combo.addItems(list(BUILTIN_GEMINI_EMBEDDING_MODELS))
        sync_layout.addRow("RAG 向量模型：", self.sync_embedding_combo)

        sync_hint = QLabel(
            "此处只配置 Gemini 同步/批量所用模型，从下拉列表选择（不可手输）。"
            "若要增加自定义模型 ID，请到「设置 → 高级 → 模型目录」。"
            "LiteLLM 已移至左侧独立页面。"
        )
        sync_hint.setWordWrap(True)
        sync_hint.setObjectName("config_hint_label")
        sync_layout.addRow(sync_hint)
        layout.addWidget(sync_box)

        batch_box = QGroupBox("批量离线翻译")
        batch_layout = self._settings_form(batch_box)

        self.batch_model_combo = NoWheelComboBox()
        self.batch_model_combo.setEditable(False)
        self.batch_model_combo.addItems(list(BUILTIN_GEMINI_TRANSLATION_MODELS))
        batch_layout.addRow("翻译模型：", self.batch_model_combo)

        self.batch_embedding_combo = NoWheelComboBox()
        self.batch_embedding_combo.setEditable(False)
        self.batch_embedding_combo.addItems(list(BUILTIN_GEMINI_EMBEDDING_MODELS))
        batch_layout.addRow("RAG 向量模型：", self.batch_embedding_combo)

        self.batch_thinking_combo = NoWheelComboBox()
        self.batch_thinking_combo.addItem("（不启用）", "")
        self.batch_thinking_combo.addItem("最小", "minimal")
        self.batch_thinking_combo.addItem("低", "low")
        self.batch_thinking_combo.addItem("中", "medium")
        self.batch_thinking_combo.addItem("高", "high")
        batch_layout.addRow("思考程度：", self.batch_thinking_combo)

        layout.addWidget(batch_box)
        layout.addStretch(1)
        return page

    def _build_settings_extensions_page(self) -> QWidget:
        page, layout = self._settings_page("settings_extensions")
        hint = QLabel(
            "扩展是按需安装的可选能力。是否启用取决于当前 Python 环境中的安装状态，"
            "不会单独保存“已启用”开关。安装在后台进行，不会阻止普通翻译任务。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        card, card_layout = self._settings_group("关系分析器")
        card.setObjectName("relation_analyzer_extension_card")

        purpose = QLabel(
            "从 Ren'Py TL 目录提取人物关系与语义相似度图。"
            "包含 NumPy、Matplotlib、scikit-learn、Pillow 等科学/图像组件；"
            "通过独立 CLI（extract_relations.py）运行，不会把重型库加载进 GUI 进程。"
        )
        purpose.setWordWrap(True)
        purpose.setObjectName("config_hint_label")
        card_layout.addWidget(purpose)

        components = QLabel(
            "组件：numpy · matplotlib · scikit-learn · pillow（及 scipy 传递依赖）"
        )
        components.setWordWrap(True)
        components.setObjectName("config_hint_label")
        card_layout.addWidget(components)

        self.relation_analyzer_status_label = QLabel()
        self.relation_analyzer_status_label.setObjectName("relation_analyzer_status_label")
        self.relation_analyzer_status_label.setWordWrap(True)
        card_layout.addWidget(self.relation_analyzer_status_label)

        self.relation_analyzer_failure_label = QLabel()
        self.relation_analyzer_failure_label.setObjectName("relation_analyzer_failure_label")
        self.relation_analyzer_failure_label.setWordWrap(True)
        self.relation_analyzer_failure_label.setVisible(False)
        card_layout.addWidget(self.relation_analyzer_failure_label)

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        self.relation_analyzer_install_btn = QPushButton("安装并启用")
        self.relation_analyzer_install_btn.setObjectName("relation_analyzer_install_btn")
        self.relation_analyzer_install_btn.clicked.connect(
            self._on_install_relation_analyzer
        )
        action_layout.addWidget(self.relation_analyzer_install_btn)
        self.relation_analyzer_docs_btn = QPushButton("使用说明")
        self.relation_analyzer_docs_btn.setObjectName("secondary_btn")
        self.relation_analyzer_docs_btn.clicked.connect(
            self._on_open_relation_analyzer_docs
        )
        action_layout.addWidget(self.relation_analyzer_docs_btn)
        action_layout.addStretch(1)
        card_layout.addWidget(action_row)

        self.relation_analyzer_install_progress = QProgressBar()
        self.relation_analyzer_install_progress.setObjectName(
            "relation_analyzer_install_progress"
        )
        self.relation_analyzer_install_progress.setTextVisible(True)
        self.relation_analyzer_install_progress.setFormat("正在后台安装关系分析器…")
        self.relation_analyzer_install_progress.setVisible(False)
        card_layout.addWidget(self.relation_analyzer_install_progress)

        layout.addWidget(card)
        layout.addStretch(1)

        self._ensure_relation_analyzer_install_controller()
        self._refresh_relation_analyzer_extension_ui()
        return page

    def _ensure_relation_analyzer_install_controller(self) -> OptionalFeatureInstallController:
        controller = getattr(self, "_relation_analyzer_install", None)
        if controller is not None:
            return controller
        controller = build_relation_analyzer_controller(self)
        controller.output_received.connect(self._on_optional_feature_install_output)
        controller.state_changed.connect(self._on_relation_analyzer_status_changed)
        controller.finished.connect(self._on_relation_analyzer_install_finished)
        self._relation_analyzer_install = controller
        return controller

    def _on_optional_feature_install_output(self, text: str) -> None:
        if text.startswith("==="):
            self._show_workbench_log_drawer()
        self._append_log(text)

    def _on_relation_analyzer_status_changed(
        self,
        _feature_id: str,
        status: FeatureStatus,
    ) -> None:
        self._apply_relation_analyzer_status(status)

    def _on_relation_analyzer_install_finished(
        self,
        feature_id: str,
        succeeded: bool,
        message: str,
    ) -> None:
        if succeeded:
            self._optional_feature_last_failed.discard(feature_id)
            self.statusBar().showMessage(message, 5000)
        else:
            self._optional_feature_last_failed.add(feature_id)
            QMessageBox.warning(self, "关系分析器安装失败", message + "\n请查看工作台日志中的 pip 输出。")
        self._refresh_relation_analyzer_extension_ui()

    def _refresh_relation_analyzer_extension_ui(self) -> None:
        if not hasattr(self, "relation_analyzer_status_label"):
            return
        controller = self._ensure_relation_analyzer_install_controller()
        status = controller.current_status()
        if controller.feature.feature_id in self._optional_feature_last_failed:
            # Re-probe with failed overlay when not actively installing.
            if status.state != FeatureInstallState.INSTALLING:
                from optional_feature import probe_feature

                status = probe_feature(
                    controller.feature,
                    installing=False,
                    last_failed=True,
                )
        self._apply_relation_analyzer_status(status)

    def _apply_relation_analyzer_status(self, status: FeatureStatus) -> None:
        label = getattr(self, "relation_analyzer_status_label", None)
        button = getattr(self, "relation_analyzer_install_btn", None)
        progress = getattr(self, "relation_analyzer_install_progress", None)
        failure = getattr(self, "relation_analyzer_failure_label", None)
        if label is None or button is None:
            return

        versions = ""
        if status.installed_versions:
            versions = "；已装：" + "、".join(
                f"{name} {version}"
                for name, version in sorted(status.installed_versions.items())
            )
        label.setText(f"状态：{status.message}{versions}")

        installing = status.state == FeatureInstallState.INSTALLING
        button.setText(status.action_label)
        button.setEnabled(action_enabled_for_status(status))
        if progress is not None:
            progress.setVisible(installing)
            if installing:
                progress.setRange(0, 0)
                progress.setFormat("正在后台安装关系分析器…")

        if failure is not None:
            show_failure = status.state == FeatureInstallState.FAILED
            failure.setVisible(show_failure)
            if show_failure:
                failure.setText("最近一次安装失败。请查看工作台日志中的 pip 输出后重试。")

    def _on_install_relation_analyzer(self) -> None:
        controller = self._ensure_relation_analyzer_install_controller()
        if controller.is_running():
            return
        started, message = controller.start_install()
        if not started:
            QMessageBox.information(self, "无法开始安装", message)
            return
        self._refresh_relation_analyzer_extension_ui()

    def _on_open_relation_analyzer_docs(self) -> None:
        controller = self._ensure_relation_analyzer_install_controller()
        docs = controller.repo_root / controller.feature.docs_relative_path
        if docs.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(docs)))
            return
        QMessageBox.information(
            self,
            "使用说明",
            "请参阅 docs/relation_analysis.md 与 relation_analyzer/README.md。",
        )

    def _build_settings_litellm_page(self) -> QWidget:
        page, layout = self._settings_page("settings_litellm")

        backend_box = QGroupBox("LiteLLM 同步替代后端")
        backend_layout = self._settings_form(backend_box)

        self.sync_backend_combo = NoWheelComboBox()
        self.sync_backend_combo.addItem("Gemini 同步（推荐）", "gemini")
        self.sync_backend_combo.addItem("启用 LiteLLM 同步替代", "litellm")
        self.sync_backend_combo.currentIndexChanged.connect(self._on_sync_backend_changed)
        backend_layout.addRow("同步执行后端：", self.sync_backend_combo)

        self.litellm_provider_combo = NoWheelComboBox()
        for provider, label in SUPPORTED_PROVIDERS:
            self.litellm_provider_combo.addItem(label, provider)
        backend_layout.addRow("Provider：", self.litellm_provider_combo)

        self.litellm_model_combo = NoWheelComboBox()
        self._configure_editable_model_combo(self.litellm_model_combo)
        self.litellm_model_combo.addItems(DEFAULT_MODELS["openai"])
        self.litellm_model_combo.currentTextChanged.connect(self._on_litellm_model_changed)
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)
        model_layout.addWidget(self.litellm_model_combo, 1)
        self.litellm_refresh_models_btn = QPushButton("联网更新列表")
        self.litellm_refresh_models_btn.setObjectName("secondary_btn")
        self.litellm_refresh_models_btn.clicked.connect(self._on_refresh_litellm_models)
        model_layout.addWidget(self.litellm_refresh_models_btn)
        backend_layout.addRow("LiteLLM 模型：", model_row)
        self.litellm_catalog_status_label = QLabel("目录来源：内置兜底；尚未联网更新。")
        self.litellm_catalog_status_label.setWordWrap(True)
        self.litellm_catalog_status_label.setObjectName("config_hint_label")
        backend_layout.addRow(self.litellm_catalog_status_label)

        self.sync_backend_hint = QLabel()
        self.sync_backend_hint.setWordWrap(True)
        self.sync_backend_hint.setObjectName("config_hint_label")
        backend_layout.addRow(self.sync_backend_hint)

        version_row = QWidget()
        version_layout = QHBoxLayout(version_row)
        version_layout.setContentsMargins(0, 0, 0, 0)
        version_layout.setSpacing(8)
        self.litellm_version_label = QLabel()
        self.litellm_version_label.setWordWrap(True)
        self.litellm_version_label.setMinimumWidth(0)
        self.litellm_version_label.setObjectName("config_hint_label")
        version_layout.addWidget(self.litellm_version_label, 1)
        self.litellm_check_version_btn = QPushButton("检查更新")
        self.litellm_check_version_btn.setObjectName("secondary_btn")
        self.litellm_check_version_btn.clicked.connect(self._on_check_litellm_version)
        version_layout.addWidget(self.litellm_check_version_btn)
        self.install_litellm_btn = QPushButton("安装 LiteLLM")
        self.install_litellm_btn.setObjectName("secondary_btn")
        self.install_litellm_btn.clicked.connect(self._on_install_litellm)
        self.install_litellm_btn.setVisible(False)
        version_layout.addWidget(self.install_litellm_btn)
        backend_layout.addRow("LiteLLM 版本：", version_row)
        self._refresh_litellm_version_label()

        self.litellm_install_progress = QProgressBar()
        self.litellm_install_progress.setObjectName("litellm_install_progress")
        self.litellm_install_progress.setTextVisible(True)
        self.litellm_install_progress.setFormat("正在后台安装 LiteLLM…")
        self.litellm_install_progress.setVisible(False)
        backend_layout.addRow(self.litellm_install_progress)
        layout.addWidget(backend_box)

        credentials_box, credentials_layout = self._settings_group("Provider 凭据")
        credentials_hint = QLabel(
            "密钥与 Gemini 配置完全分离，并保存到操作系统凭据管理器；"
            "不会写入 translator_config.json。也可继续使用 LiteLLM 约定的环境变量。"
        )
        credentials_hint.setWordWrap(True)
        credentials_hint.setObjectName("config_hint_label")
        credentials_layout.addWidget(credentials_hint)
        self.litellm_provider_label = QLabel()
        self.litellm_provider_label.setObjectName("litellm_provider_label")
        credentials_layout.addWidget(self.litellm_provider_label)
        self.litellm_api_key_edit = QLineEdit()
        self.litellm_api_key_edit.setObjectName("litellm_api_key_edit")
        self.litellm_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.litellm_api_key_edit.setPlaceholderText("输入新密钥（已保存的密钥不会显示）")
        credentials_layout.addWidget(self.litellm_api_key_edit)
        credential_actions = QWidget()
        credential_actions_layout = QHBoxLayout(credential_actions)
        credential_actions_layout.setContentsMargins(0, 0, 0, 0)
        credential_actions_layout.setSpacing(8)
        self.litellm_save_key_btn = QPushButton("保存密钥")
        self.litellm_save_key_btn.clicked.connect(self._on_save_litellm_key)
        credential_actions_layout.addWidget(self.litellm_save_key_btn)
        self.litellm_delete_key_btn = QPushButton("删除已保存密钥")
        self.litellm_delete_key_btn.setObjectName("secondary_btn")
        self.litellm_delete_key_btn.clicked.connect(self._on_delete_litellm_key)
        credential_actions_layout.addWidget(self.litellm_delete_key_btn)
        credential_actions_layout.addStretch(1)
        credentials_layout.addWidget(credential_actions)
        self.litellm_credential_status_label = QLabel()
        self.litellm_credential_status_label.setWordWrap(True)
        self.litellm_credential_status_label.setObjectName("api_status_label")
        credentials_layout.addWidget(self.litellm_credential_status_label)
        self.litellm_test_connection_btn = QPushButton("测试连接")
        self.litellm_test_connection_btn.clicked.connect(self._on_test_litellm_connection)
        credentials_layout.addWidget(self.litellm_test_connection_btn)
        self.litellm_connection_status_label = QLabel("尚未测试连接。")
        self.litellm_connection_status_label.setWordWrap(True)
        self.litellm_connection_status_label.setObjectName("config_hint_label")
        credentials_layout.addWidget(self.litellm_connection_status_label)
        layout.addWidget(credentials_box)

        self.litellm_provider_combo.currentIndexChanged.connect(
            self._on_litellm_provider_changed
        )
        self._refresh_litellm_credential_status()
        layout.addStretch(1)
        return page

    def _build_settings_appearance_page(self) -> QWidget:
        page, layout = self._settings_page("settings_appearance")
        appearance_box = QGroupBox("外观")
        appearance_layout = self._settings_form(appearance_box)
        self.theme_combo = NoWheelComboBox()
        self.theme_combo.addItem("跟随系统", THEME_SYSTEM)
        self.theme_combo.addItem("浅色", "light")
        self.theme_combo.addItem("深色", "dark")
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        appearance_layout.addRow("主题：", self.theme_combo)
        hint = QLabel("切换主题会立即预览；点击保存设置后才会写入 translator_config.json。")
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        appearance_layout.addRow("", hint)
        layout.addWidget(appearance_box)

        fonts_box = QGroupBox("推荐字体")
        fonts_layout = self._settings_form(fonts_box)
        self.font_install_status_label = QLabel()
        self.font_install_status_label.setWordWrap(True)
        self.font_install_status_label.setObjectName("config_hint_label")
        fonts_layout.addRow("安装状态：", self.font_install_status_label)

        font_actions = QWidget()
        font_actions_layout = QHBoxLayout(font_actions)
        font_actions_layout.setContentsMargins(0, 0, 0, 0)
        font_actions_layout.setSpacing(8)
        self.download_fonts_btn = QPushButton("下载推荐字体")
        self.download_fonts_btn.setObjectName("secondary_btn")
        self.download_fonts_btn.setToolTip(
            "从华为和霞鹜文楷官方来源下载固定版本字体，并执行 SHA-256 校验。"
        )
        self.download_fonts_btn.clicked.connect(self._on_download_recommended_fonts)
        font_actions_layout.addWidget(self.download_fonts_btn)
        font_actions_layout.addStretch(1)
        fonts_layout.addRow("", font_actions)

        self.font_install_progress = QProgressBar()
        self.font_install_progress.setObjectName("font_install_progress")
        self.font_install_progress.setRange(0, 0)
        self.font_install_progress.setFormat("正在后台下载并校验推荐字体…")
        self.font_install_progress.setVisible(False)
        fonts_layout.addRow(self.font_install_progress)
        layout.addWidget(fonts_box)
        self._refresh_font_install_status()

        layout.addStretch(1)
        return page

    def _build_settings_shortcuts_page(self) -> QWidget:
        """Read-only catalog of global GUI keyboard shortcuts."""
        page, layout = self._settings_page("settings_shortcuts")
        intro = QLabel(
            "以下为图形工作台当前生效的全局快捷键。"
            "任务类快捷键会在对应按钮禁用时同步关闭；"
            "导航类快捷键在任务运行中会遵循锁定规则（不可切换到其它任务页）。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("config_hint_label")
        layout.addWidget(intro)

        for group_title, rows in self._shortcut_catalog():
            box = QGroupBox(group_title)
            form = self._settings_form(box)
            for key, meaning in rows:
                value = QLabel(meaning)
                value.setWordWrap(True)
                value.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                form.addRow(f"{key}：", value)
            layout.addWidget(box)

        layout.addStretch(1)
        return page

    def _build_settings_advanced_page(self) -> QWidget:
        page, layout = self._settings_page("settings_advanced")
        hint = QLabel(
            "高级设置会直接影响请求大小、上下文注入和本地上下文路径。"
            "无效字段会在本页标出，并阻止保存。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        skipped_categories = {
            "项目与资源",
            "准备流程",
            CONTEXT_PRIMARY_SETTING_CATEGORY,
        }
        for group_title, fields in grouped_advanced_fields(
            include_context_primary=False,
        ):
            if group_title in skipped_categories:
                continue
            group = QGroupBox(group_title)
            form = self._settings_form(group)
            for field in fields:
                if field.key in CONTEXT_PRIMARY_SETTING_KEYS:
                    continue
                widget = self._create_advanced_setting_widget(field)
                self._advanced_setting_widgets[field.key] = widget
                row = self._advanced_setting_row(field, widget)
                form.addRow(f"{field.label}：", row)
            if form.rowCount() == 0:
                group.deleteLater()
                continue
            layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _advanced_setting_row(self, field: SettingField, widget: QWidget) -> QWidget:
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        if field.key == "prepare_renpy_sdk_dir" and isinstance(widget, QLineEdit):
            row_layout.addWidget(self._wrap_renpy_sdk_path_widget(widget))
        else:
            row_layout.addWidget(widget)

        desc = QLabel(field.description)
        desc.setWordWrap(True)
        desc.setObjectName("settings_description_label")
        row_layout.addWidget(desc)

        error = QLabel()
        error.setWordWrap(True)
        error.setObjectName("settings_error_label")
        self._advanced_setting_error_labels[field.key] = error
        row_layout.addWidget(error)
        return row

    def _wrap_renpy_sdk_path_widget(self, line_edit: QLineEdit) -> QWidget:
        """Path field + browse/find actions for prepare.renpy_sdk_dir."""
        host = QWidget()
        host.setObjectName("prepare_renpy_sdk_path_row")
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(line_edit, 1)

        browse_btn = QPushButton("浏览…")
        browse_btn.setObjectName("secondary_btn")
        browse_btn.setToolTip("手动选择包含 renpy.py 的 Ren'Py SDK 目录。")
        browse_btn.clicked.connect(lambda: self._on_browse_renpy_sdk_dir(line_edit))
        layout.addWidget(browse_btn)

        find_btn = QPushButton("查找 SDK")
        find_btn.setObjectName("secondary_btn")
        find_btn.setToolTip(
            "仅在点击后才会扫描：当前项目、已选工作区与工具附近的 renpy-*-sdk / renpy.py。"
            "平时加载配置与 prepare 不会自动搜其它目录。找到后填入（多结果时可选）。"
        )
        find_btn.clicked.connect(lambda: self._on_find_renpy_sdk_dir(line_edit))
        layout.addWidget(find_btn)
        self._prepare_renpy_sdk_find_btn = find_btn
        self._prepare_renpy_sdk_browse_btn = browse_btn
        return host

    def _on_browse_renpy_sdk_dir(self, line_edit: QLineEdit) -> None:
        current = line_edit.text().strip()
        start_dir = current or str(
            self.state.get_workspace_root()
            or self.state.get_game_root()
            or self.state.get_tool_root()
        )
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择 Ren'Py SDK 目录（应包含 renpy.py）",
            start_dir,
        )
        if not directory:
            return
        from translator_runtime import is_renpy_sdk_dir

        path_text = canonical_abs_path(directory)
        if not is_renpy_sdk_dir(path_text):
            reply = QMessageBox.question(
                self,
                "目录可能不是 Ren'Py SDK",
                f"所选目录未发现 renpy.py：\n{path_text}\n\n仍要填入吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        line_edit.setText(path_text)
        self.statusBar().showMessage(f"已填入 Ren'Py SDK：{path_text}", 5000)

    def _on_find_renpy_sdk_dir(self, line_edit: QLineEdit) -> None:
        from translator_runtime import discover_renpy_sdk_candidates

        game_root = self.state.get_game_root()
        workspace_root = self.state.get_workspace_root()
        tool_root = self.state.get_tool_root()
        # Explicit GUI roots only (no runtime BASE_DIR/ROOT_DIR mix-in).
        # tool_root / game_root still use intentional parent hops inside
        # renpy_sdk_search_roots; include_runtime_defaults=False skips globals.
        candidates = discover_renpy_sdk_candidates(
            game_root=str(game_root) if game_root is not None else None,
            tool_root=str(tool_root),
            workspace_root=str(workspace_root) if workspace_root is not None else None,
            include_runtime_defaults=False,
        )
        if not candidates:
            roots_hint = []
            if game_root is not None:
                roots_hint.append(f"项目：{game_root}（及其上两级）")
            if workspace_root is not None:
                roots_hint.append(f"工作区：{workspace_root}")
            roots_hint.append(f"工具：{tool_root}（及其上一级）")
            QMessageBox.information(
                self,
                "未找到 Ren'Py SDK",
                "在附近目录未找到包含 renpy.py 的 Ren'Py SDK。\n\n"
                "已搜索：\n- "
                + "\n- ".join(roots_hint)
                + "\n\n可安装 SDK 后重试，或点「浏览…」手动选择。",
            )
            return

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            labels = [canonical_abs_path(path) for path in candidates]
            chosen_label, ok = QInputDialog.getItem(
                self,
                "选择 Ren'Py SDK",
                f"找到 {len(labels)} 个候选（已按版本优先排序）：",
                labels,
                0,
                False,
            )
            if not ok or not chosen_label:
                return
            chosen = chosen_label

        line_edit.setText(canonical_abs_path(chosen))
        self.statusBar().showMessage(f"已填入查找到的 Ren'Py SDK：{chosen}", 6000)
        self._append_log(f"查找 Ren'Py SDK：已选择 {chosen}")

    def _configure_editable_model_combo(self, combo: NoWheelComboBox) -> None:
        """Editable model field that still has a clear dropdown for picking list items.

        On Windows/stylesheets, ``setEditable(True)`` can hide the native combo
        arrow; mirror LiteLLM and install an explicit popup action.
        """
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        add_editable_combo_popup_action(combo)

    def _create_advanced_setting_widget(self, field: SettingField) -> QWidget:
        if field.kind == "bool":
            widget = QCheckBox()
            if field.key == "model_rotation_enabled":
                widget.toggled.connect(self._on_model_rotation_enabled_toggled)
        elif field.kind == "int":
            widget = QSpinBox()
            widget.setAccelerated(True)
            minimum = int(field.minimum if field.minimum is not None else 0)
            maximum = int(field.maximum if field.maximum is not None else 9999999)
            widget.setRange(minimum, maximum)
        elif field.kind == "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(3)
            widget.setSingleStep(0.01 if field.maximum == 1.0 else 0.1)
            minimum = float(field.minimum if field.minimum is not None else -999999.0)
            maximum = float(field.maximum if field.maximum is not None else 999999.0)
            widget.setRange(minimum, maximum)
        elif field.kind == "gemini_model_list":
            widget = self._create_gemini_model_checklist()
        elif field.kind in {"text", "list", "json"}:
            widget = QTextEdit()
            widget.setAcceptRichText(False)
            widget.setMinimumHeight(72 if field.kind != "text" else 96)
            widget.setPlaceholderText(self._advanced_setting_placeholder(field))
        else:
            widget = QLineEdit()
            widget.setClearButtonEnabled(True)
            if field.allow_empty:
                widget.setPlaceholderText("留空使用默认路径" if "路径" in field.label else "可留空")
        widget.setToolTip(field.description)
        return widget

    def _create_gemini_model_checklist(self) -> QListWidget:
        """Multi-select checklist limited to known Gemini translation models."""
        widget = QListWidget()
        widget.setObjectName("model_rotation_models_list")
        widget.setMinimumHeight(160)
        widget.setMaximumHeight(240)
        widget.setAlternatingRowColors(True)
        self._refresh_gemini_model_checklist(widget)
        return widget

    def _refresh_gemini_model_checklist(
        self,
        widget: QListWidget | None = None,
        *,
        selected: object | None = None,
        config: dict | None = None,
    ) -> None:
        """Rebuild checklist rows from the current Gemini catalog."""
        if widget is None:
            widgets = getattr(self, "_advanced_setting_widgets", {})
            candidate = widgets.get("model_rotation_models")
            widget = candidate if isinstance(candidate, QListWidget) else None
        if widget is None:
            return
        if selected is None:
            selected = self._gemini_model_checklist_values(widget)
        if config is None:
            try:
                loaded = self.state.load_translator_config()
                config = loaded if isinstance(loaded, dict) else {}
            except Exception:
                config = {}
        models = allowed_gemini_rotation_models(config)
        previous_block = widget.blockSignals(True)
        try:
            widget.clear()
            selected_set = {
                str(item).strip()
                for item in (selected if isinstance(selected, (list, tuple, set)) else [])
                if str(item).strip()
            }
            for name in models:
                item = QListWidgetItem(name)
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                )
                item.setCheckState(
                    Qt.CheckState.Checked
                    if name in selected_set
                    else Qt.CheckState.Unchecked
                )
                widget.addItem(item)
        finally:
            widget.blockSignals(previous_block)

    def _gemini_model_checklist_values(self, widget: QListWidget) -> list[str]:
        selected: list[str] = []
        for index in range(widget.count()):
            item = widget.item(index)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                text = item.text().strip()
                if text:
                    selected.append(text)
        return selected

    def _set_gemini_model_checklist_values(
        self,
        widget: QListWidget,
        values: object,
    ) -> None:
        selected = {
            str(item).strip()
            for item in (values if isinstance(values, (list, tuple, set)) else [])
            if str(item).strip()
        }
        # Preserve order from the checklist; only toggle known rows.
        for index in range(widget.count()):
            item = widget.item(index)
            if item is None:
                continue
            item.setCheckState(
                Qt.CheckState.Checked
                if item.text().strip() in selected
                else Qt.CheckState.Unchecked
            )

    def _on_model_rotation_enabled_toggled(self, checked: bool) -> None:
        widgets = getattr(self, "_advanced_setting_widgets", {})
        checklist = widgets.get("model_rotation_models")
        if checklist is not None:
            checklist.setEnabled(bool(checked))

    def _advanced_setting_placeholder(self, field: SettingField) -> str:
        if field.kind == "list":
            return "每行一个值，或填写 JSON 数组"
        if field.kind == "json":
            return "留空使用默认；可填写字符串预设或 JSON"
        if field.kind == "text":
            return "可留空"
        return ""

    def _build_log_tab(self) -> None:
        tab = QWidget()
        tab.setObjectName("diagnostics_tab")
        self._diagnostics_tab = tab
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(10)

        diag_hint = QLabel(
            "查看当前任务的上下文、复现命令、任务记录与原始命令输出。"
        )
        self.diagnostics_hint_label = diag_hint
        diag_hint.setWordWrap(True)
        diag_hint.setObjectName("config_hint_label")
        diag_hint.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(diag_hint)

        self.diagnostics_action_panel = FlowButtonBar(spacing=8, row_spacing=8)
        self.diagnostics_action_panel.setObjectName("diagnostics_action_panel")
        diagnostics_action_panel = self.diagnostics_action_panel

        self.refresh_diagnostics_btn = QPushButton("刷新上下文")
        self.refresh_diagnostics_btn.setObjectName("secondary_btn")
        self.refresh_diagnostics_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.refresh_diagnostics_btn.clicked.connect(
            lambda: self._refresh_diagnostics_context(force=True)
        )
        diagnostics_action_panel.add_widget(
            self.refresh_diagnostics_btn,
            min_width=108,
        )

        self.compare_variants_btn = QPushButton("翻译 A/B 对比")
        self.compare_variants_btn.setObjectName("secondary_btn")
        self.compare_variants_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.compare_variants_btn.setToolTip(
            "用同一批 manifest chunk 并排比较多个配置变体的同步译文，不会写回游戏文件。"
        )
        self.compare_variants_btn.clicked.connect(self._on_run_compare_variants)
        self.compare_variants_btn.setEnabled(False)
        diagnostics_action_panel.add_widget(
            self.compare_variants_btn,
            min_width=120,
        )

        # P2b: glossary merge primary entry is workbench · 关键词; keep attribute for
        # enable helpers but do not place a diagnostics toolbar button.
        self.keyword_merge_btn = QPushButton("合并到 glossary")
        self.keyword_merge_btn.setObjectName("secondary_btn")
        self.keyword_merge_btn.setVisible(False)
        self.keyword_merge_btn.setEnabled(False)
        self.keyword_merge_btn.clicked.connect(self._on_open_keyword_merge)

        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.setObjectName("secondary_btn")
        self.clear_log_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.clear_log_btn.clicked.connect(self._on_clear_log)
        diagnostics_action_panel.add_widget(self.clear_log_btn, min_width=108)
        diagnostics_action_panel.finish_setup()
        layout.addWidget(diagnostics_action_panel)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("diagnostics_splitter")
        splitter.setChildrenCollapsible(False)
        self.diagnostics_splitter = splitter

        self.diagnostics_inner_tabs = NoWheelTabWidget()
        self.diagnostics_inner_tabs.setObjectName("diagnostics_inner_tabs")
        self.diagnostics_inner_tabs.tabBar().setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )

        context_tab = QWidget()
        context_tab.setObjectName("diagnostics_context_page")
        self._style_themed_surface(context_tab)
        context_outer = QVBoxLayout(context_tab)
        context_outer.setContentsMargins(0, 0, 0, 0)
        context_scroll = QScrollArea()
        context_scroll.setObjectName("diagnostics_context_scroll")
        self._style_themed_surface(context_scroll)
        context_scroll.setWidgetResizable(True)
        context_scroll.setFrameShape(QFrame.Shape.NoFrame)
        context_scroll.viewport().setObjectName("diagnostics_context_viewport")
        self._style_themed_surface(context_scroll.viewport())
        context_content = QWidget()
        context_content.setObjectName("diagnostics_context_content")
        self._style_themed_surface(context_content)
        context_layout = QVBoxLayout(context_content)
        context_layout.setContentsMargins(12, 12, 12, 12)
        context_layout.setSpacing(10)

        self.diagnostics_status_label = StatusBadge("diagnostics_status_label")
        context_layout.addWidget(self.diagnostics_status_label)
        self.diagnostics_message_label = QLabel()
        self.diagnostics_message_label.setWordWrap(True)
        self.diagnostics_message_label.setObjectName("summary_body_label")
        context_layout.addWidget(self.diagnostics_message_label)

        facts_frame = QFrame()
        facts_frame.setObjectName("diagnostics_facts_frame")
        facts_layout = QVBoxLayout(facts_frame)
        facts_layout.setContentsMargins(10, 10, 10, 10)
        self.diagnostics_facts_label = QLabel()
        self.diagnostics_facts_label.setWordWrap(True)
        self.diagnostics_facts_label.setObjectName("diagnostics_facts_label")
        facts_layout.addWidget(self.diagnostics_facts_label)
        context_layout.addWidget(facts_frame)

        paths_header = QLabel("报告与数据文件")
        paths_header.setObjectName("diagnostics_section_label")
        context_layout.addWidget(paths_header)
        self.diagnostics_paths_host = QWidget()
        self.diagnostics_paths_layout = QVBoxLayout(self.diagnostics_paths_host)
        self.diagnostics_paths_layout.setContentsMargins(0, 0, 0, 0)
        self.diagnostics_paths_layout.setSpacing(6)
        context_layout.addWidget(self.diagnostics_paths_host)
        context_layout.addStretch()
        context_scroll.setWidget(context_content)
        context_outer.addWidget(context_scroll)
        self.diagnostics_inner_tabs.addTab(context_tab, "任务上下文")

        commands_tab = QWidget()
        commands_tab.setObjectName("diagnostics_commands_page")
        self._style_themed_surface(commands_tab)
        commands_outer = QVBoxLayout(commands_tab)
        commands_outer.setContentsMargins(0, 0, 0, 0)
        commands_scroll = QScrollArea()
        commands_scroll.setObjectName("diagnostics_commands_scroll")
        self._style_themed_surface(commands_scroll)
        commands_scroll.setWidgetResizable(True)
        commands_scroll.setFrameShape(QFrame.Shape.NoFrame)
        commands_scroll.viewport().setObjectName("diagnostics_commands_viewport")
        self._style_themed_surface(commands_scroll.viewport())
        commands_content = QWidget()
        commands_content.setObjectName("diagnostics_commands_content")
        self._style_themed_surface(commands_content)
        commands_layout = QVBoxLayout(commands_content)
        commands_layout.setContentsMargins(12, 12, 12, 12)
        commands_layout.setSpacing(8)
        commands_hint = QLabel("复制后在终端运行；命令使用当前解释器与脚本路径。")
        commands_hint.setWordWrap(True)
        commands_hint.setObjectName("config_hint_label")
        commands_layout.addWidget(commands_hint)
        self.diagnostics_commands_host = QWidget()
        self.diagnostics_commands_layout = QVBoxLayout(self.diagnostics_commands_host)
        self.diagnostics_commands_layout.setContentsMargins(0, 0, 0, 0)
        self.diagnostics_commands_layout.setSpacing(8)
        commands_layout.addWidget(self.diagnostics_commands_host)
        commands_layout.addStretch()
        commands_scroll.setWidget(commands_content)
        commands_outer.addWidget(commands_scroll)
        self.diagnostics_inner_tabs.addTab(commands_tab, "命令参考")

        manifest_tab = QWidget()
        manifest_tab.setObjectName("diagnostics_manifest_page")
        self._style_themed_surface(manifest_tab)
        manifest_layout = QVBoxLayout(manifest_tab)
        manifest_layout.setContentsMargins(12, 12, 12, 12)
        manifest_layout.setSpacing(8)
        manifest_hint = QLabel("只读预览；大段条目明细已省略。")
        manifest_hint.setWordWrap(True)
        manifest_hint.setObjectName("config_hint_label")
        manifest_layout.addWidget(manifest_hint)
        self.diagnostics_manifest_preview = QTextEdit()
        self.diagnostics_manifest_preview.setReadOnly(True)
        self.diagnostics_manifest_preview.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.diagnostics_manifest_preview.setObjectName("diagnostics_manifest_preview")
        manifest_layout.addWidget(self.diagnostics_manifest_preview, 1)
        self.diagnostics_inner_tabs.addTab(manifest_tab, "任务记录")

        splitter.addWidget(self.diagnostics_inner_tabs)

        log_panel = QWidget()
        log_panel.setObjectName("diagnostics_log_panel")
        log_panel_layout = QVBoxLayout(log_panel)
        log_panel_layout.setContentsMargins(0, 4, 0, 0)
        log_panel_layout.setSpacing(6)
        log_title = QLabel("原始命令输出")
        log_title.setObjectName("diagnostics_section_label")
        log_panel_layout.addWidget(log_title)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log_view.setObjectName("log_view")
        self.log_view.setMinimumHeight(120)
        self._log_highlighter = LogHighlighter(
            self.log_view.document(),
            dark=self._effective_theme_is_dark(),
        )
        log_panel_layout.addWidget(self.log_view, 1)
        splitter.addWidget(log_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([_DIAGNOSTICS_IDLE_CONTEXT_PX, _DIAGNOSTICS_IDLE_LOG_PX])

        layout.addWidget(splitter, 1)
        self.tab_widget.addTab(tab, "诊断与工具")

    def _scroll_log_views_to_end(self) -> None:
        view = getattr(self, "log_view", None)
        if view is None or not hasattr(view, "verticalScrollBar"):
            return
        scrollbar = view.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def _focus_workbench_main_tab(self, *, special_route: str | None = None) -> None:
        """Switch the top-level shell to Workbench so prep progress is visible."""
        workbench = getattr(self, "_workbench_tab", None)
        if workbench is None or not hasattr(self, "tab_widget"):
            return
        if self.tab_widget.currentWidget() is not workbench:
            self.tab_widget.setCurrentWidget(workbench)
        if self.tab_widget.currentWidget() is workbench:
            self._set_shell_special_route(special_route)
            if hasattr(self, "workbench_stack"):
                self.workbench_stack.setVisible(special_route is None)
            self._sync_shell_nav_selection()

    def _show_workbench_log_drawer(self) -> None:
        """Compatibility hook: logs now live only on the diagnostics page."""
        self._scroll_log_views_to_end()

    def _expand_diagnostics_log(self, *, switch_tab: bool = True) -> None:
        """Enlarge the diagnostics-page log splitter; optionally switch to that tab.

        Used by diagnostics toolbar tools (probe / split / A/B) and 「在诊断中打开」.
        """
        if switch_tab and self._diagnostics_tab is not None:
            self.tab_widget.setCurrentWidget(self._diagnostics_tab)
        if not hasattr(self, "diagnostics_splitter"):
            return
        total = max(sum(self.diagnostics_splitter.sizes()), 1)
        target_context_size = int(total * _DIAGNOSTICS_RUNNING_CONTEXT_RATIO)
        start_context_size = self.diagnostics_splitter.sizes()[0]

        if hasattr(self, "_splitter_anim") and self._splitter_anim.state() == self._splitter_anim.State.Running:
            self._splitter_anim.stop()

        from PySide6.QtCore import QVariantAnimation, QEasingCurve
        anim = QVariantAnimation(self)
        anim.setDuration(300)
        anim.setStartValue(start_context_size)
        anim.setEndValue(target_context_size)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        def update_sizes(val):
            self.diagnostics_splitter.setSizes([int(val), total - int(val)])

        anim.valueChanged.connect(update_sizes)
        anim.start()
        self._splitter_anim = anim
        self._scroll_log_views_to_end()

    def _reveal_log_for_active_context(self) -> None:
        """On runner errors, reveal the only remaining full log surface."""
        self._expand_diagnostics_log(switch_tab=True)

    def _focus_log_tab(self) -> None:
        """Deprecated alias retained for callers that still request the log surface."""
        self._expand_diagnostics_log(switch_tab=True)

    def _focus_workbench_status_tab(self, index: int) -> None:
        if not (0 <= index < self.workbench_status_tabs.count()):
            return
        project_route = self._current_shell_route() == _SHELL_ROUTE_PROJECT_PREPARE
        if project_route:
            index = _BATCH_STAGE_PREPARE
        elif index == _BATCH_STAGE_PREPARE or (
            index == _BATCH_STAGE_RESULT
            and not self._work_mode_has_writeback_surface()
        ):
            index = _BATCH_STAGE_EXECUTE
        # Avoid double chrome sync: currentChanged already calls _sync_workbench_status_chrome.
        if self.workbench_status_tabs.currentIndex() == index:
            self._sync_workbench_status_chrome(stage_index=index)
            return
        self.workbench_status_tabs.setCurrentIndex(index)

    def _on_workbench_status_tab_changed(self, index: int) -> None:
        self._sync_workbench_status_chrome(stage_index=index)

    def _batch_stage_mode_active(self) -> bool:
        return self._current_work_mode() == WorkMode.BATCH_TRANSLATION

    def _current_batch_stage_index(self) -> int:
        if not hasattr(self, "workbench_status_tabs"):
            return _BATCH_STAGE_EXECUTE
        index = self.workbench_status_tabs.currentIndex()
        if index < 0:
            return _BATCH_STAGE_EXECUTE
        return max(_BATCH_STAGE_PREPARE, min(_BATCH_STAGE_RESULT, index))

    def _sync_workbench_status_chrome(
        self,
        *,
        stage_index: int | None = None,
        refresh_readiness: bool = True,
    ) -> None:
        """Sync tab-dependent chrome (batch advanced tools, result reflow)."""
        if stage_index is None:
            stage_index = self._current_batch_stage_index()
        stage_index = max(_BATCH_STAGE_PREPARE, min(_BATCH_STAGE_RESULT, stage_index))
        self._sync_batch_advanced_tools_chrome(
            stage_index=stage_index,
            refresh_readiness=refresh_readiness,
        )
        # Result tab hosts writeback flow bars that may have reflowed while off-tab.
        if stage_index == _BATCH_STAGE_RESULT:
            self._reflow_button_bars()

    # Back-compat alias for older call sites / tests.
    def _sync_batch_stage_chrome(
        self,
        *,
        stage_index: int | None = None,
        refresh_readiness: bool = True,
    ) -> None:
        self._sync_workbench_status_chrome(
            stage_index=stage_index,
            refresh_readiness=refresh_readiness,
        )

    def _sync_batch_advanced_tools_chrome(
        self,
        *,
        stage_index: int | None = None,
        refresh_readiness: bool = True,
    ) -> None:
        """Show probe/split whenever batch translation is active (under main actions).

        Stays on the workbench action column (not diagnostics): batch-only tools that
        prepare packages before / around a run, always reachable without tab switching.

        When ``refresh_readiness`` is False, only visibility/layout is updated —
        probe/split enablement (which may walk manifest history) is deferred.
        """
        if not hasattr(self, "batch_advanced_frame"):
            return
        del stage_index  # kept for call-site compat; visibility is mode-based
        is_batch = self._batch_stage_mode_active()
        show = is_batch
        self.batch_advanced_frame.setVisible(show)
        if show and hasattr(self, "batch_advanced_bar"):
            reflow = getattr(self.batch_advanced_bar, "reflow", None)
            if callable(reflow):
                reflow(force=True)
        if show and refresh_readiness:
            running = (
                bool(getattr(self, "_task_running", False))
                or (hasattr(self, "kill_btn") and self.kill_btn.isEnabled())
            )
            self._update_probe_btn_enabled(running=running)
            self._update_split_btn_enabled(running=running)

    def _same_manifest_path(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        return canonical_abs_path(left).lower() == canonical_abs_path(right).lower()

    def _split_entries_unchanged(
        self,
        entries: list[SplitManifestEntry],
    ) -> bool:
        previous = self._split_status_entries
        if len(previous) != len(entries):
            return False
        for old_entry, new_entry in zip(previous, entries, strict=True):
            if (
                old_entry.manifest_path != new_entry.manifest_path
                or old_entry.part_label != new_entry.part_label
                or old_entry.status_kind != new_entry.status_kind
                or old_entry.status_label != new_entry.status_label
                or old_entry.item_count != new_entry.item_count
                or old_entry.chunk_count != new_entry.chunk_count
                or old_entry.job_name != new_entry.job_name
                or old_entry.job_state != new_entry.job_state
                or old_entry.selectable != new_entry.selectable
                or old_entry.needs_submit != new_entry.needs_submit
            ):
                return False
        return True

    def _split_entries_for_manifest(
        self,
        manifest_path: str,
        manifest: dict[str, object] | None = None,
    ) -> list[SplitManifestEntry]:
        if work_mode_spec(self._current_work_mode()).mode != WorkMode.BATCH_TRANSLATION:
            return []
        try:
            loaded = manifest if manifest is not None else self.state.load_manifest_file(manifest_path)
        except ValueError:
            return []
        source_manifest_path = manifest_path
        source_manifest = loaded
        retry_parent = loaded.get("retry_of_manifest") if isinstance(loaded, dict) else None
        if isinstance(retry_parent, str) and retry_parent.strip():
            source_manifest_path = retry_parent.strip()
            try:
                source_manifest = self.state.load_manifest_file(source_manifest_path)
            except ValueError:
                source_manifest = None
        entries = load_split_manifest_entries(source_manifest_path, source_manifest)
        return entries if len(entries) > 1 else []

    def _load_latest_split_entries(self) -> tuple[str, list[SplitManifestEntry]]:
        spec = work_mode_spec(self._current_work_mode())
        if spec.mode != WorkMode.BATCH_TRANSLATION:
            return "", []
        game_root = self.state.get_game_root()
        if game_root is None:
            return "", []
        latest_manifest = self.state.get_latest_manifest_path_for_mode(game_root, spec.mode)
        if latest_manifest is None:
            return "", []
        try:
            manifest = self.state.load_manifest_file(latest_manifest)
        except ValueError:
            return "", []
        return str(latest_manifest), self._split_entries_for_manifest(str(latest_manifest), manifest)

    def _render_split_status_entries(
        self,
        entries: list[SplitManifestEntry],
        *,
        selected_manifest_path: str = "",
    ) -> None:
        self._split_status_entries = list(entries)
        self._split_status_selected_manifest_path = selected_manifest_path
        if not hasattr(self, "split_status_table"):
            return
        if not entries:
            self.split_status_table.setRowCount(0)
            self.split_status_table.setVisible(False)
            self.split_status_title.setVisible(False)
            self._update_split_submit_btn(entries)
            return

        profile = self._configure_split_status_table_columns()
        group_count = max(1, profile["groups"])
        rows = (len(entries) + group_count - 1) // group_count
        delegate = getattr(self, "_split_status_action_delegate", None)
        if delegate is not None:
            delegate.clear_hover_state()
            delegate.clear_pressed_state()
        self.split_status_table.setRowCount(rows)
        for row in range(rows):
            for column in range(self.split_status_table.columnCount()):
                self._set_split_table_item(row, column, "")

        for index, entry in enumerate(entries):
            group_index = index // rows if rows else 0
            row = index % rows if rows else 0
            base_column = group_index * 6
            self._render_split_status_entry(
                row,
                base_column,
                entry,
                profile,
                selected_manifest_path=selected_manifest_path,
            )

        self.split_status_table.resizeRowsToContents()
        for row in range(self.split_status_table.rowCount()):
            self.split_status_table.setRowHeight(row, max(self.split_status_table.rowHeight(row), 44))
        self._configure_split_status_table_columns()
        self.split_status_title.setVisible(True)
        self.split_status_table.setVisible(True)
        self._update_split_submit_btn(entries)

    def _effective_theme_is_dark(self) -> bool:
        system_is_dark = system_prefers_dark(self._qt_app) if self._qt_app is not None else None
        return (
            resolve_effective_theme(self._theme_preference, system_is_dark=system_is_dark)
            == THEME_DARK
        )

    def _refresh_action_icons(self) -> None:
        """Apply the restrained Tabler action-icon set for the active theme."""
        dark = self._effective_theme_is_dark()
        resources_dir = self._resources_dir

        specs: list[tuple[object, str, str]] = [
            (getattr(self, "header_log_btn", None), "file-text", "default"),
            (getattr(self, "global_switch_project_btn", None), "folder-open", "default"),
            (getattr(self, "global_browse_project_btn", None), "folder-open", "default"),
            (
                getattr(self, "doctor_btn", None),
                *(
                    ("player-stop", "danger")
                    if self._is_doctor_active() or self._is_generate_template_active()
                    else ("stethoscope", "default")
                ),
            ),
            (
                getattr(self, "bootstrap_work_btn", None),
                *(
                    ("player-stop", "danger")
                    if self._is_bootstrap_work_active()
                    else ("folder-cog", "default")
                ),
            ),
            (getattr(self, "translate_btn", None), "language", "on_accent"),
            (getattr(self, "resume_btn", None), "player-play", "default"),
            (getattr(self, "kill_btn", None), "player-stop", "danger"),
            (getattr(self, "apply_btn", None), "file-import", "success"),
            (getattr(self, "apply_revision_btn", None), "file-import", "success"),
            (getattr(self, "recheck_btn", None), "refresh", "default"),
            (getattr(self, "reload_config_btn", None), "refresh", "default"),
            (getattr(self, "save_config_btn", None), "device-floppy", "on_accent"),
        ]

        batch_page = getattr(self, "batch_translation_page", None)
        if batch_page is not None:
            specs.extend(
                (
                    (batch_page.buttons.get("start"), "language", "on_accent"),
                    (batch_page.buttons.get("resume"), "player-play", "default"),
                    (batch_page.buttons.get("stop"), "player-stop", "danger"),
                )
            )

        sync_page = getattr(self, "sync_translation_page", None)
        if sync_page is not None:
            specs.extend(
                (
                    (sync_page.start_btn, "language", "on_accent"),
                    (sync_page.stop_btn, "player-stop", "danger"),
                )
            )

        keywords_page = getattr(self, "keywords_page", None)
        if keywords_page is not None:
            specs.extend(
                (
                    (keywords_page.start_btn, "language", "on_accent"),
                    (keywords_page.resume_btn, "player-play", "default"),
                    (keywords_page.stop_btn, "player-stop", "danger"),
                    (keywords_page.merge_btn, "file-import", "default"),
                )
            )

        revision_page = getattr(self, "revision_page", None)
        if revision_page is not None:
            specs.extend(
                (
                    (revision_page.start_btn, "file-text", "on_accent"),
                    (revision_page.resume_btn, "player-play", "default"),
                    (revision_page.stop_btn, "player-stop", "danger"),
                    (revision_page.writeback_btn, "file-import", "default"),
                )
            )

        context_page = getattr(self, "context_library_page", None)
        if context_page is not None:
            specs.extend(
                (
                    (context_page.bootstrap_rag_btn, "folder-cog", "on_accent"),
                    (context_page.bootstrap_source_index_btn, "folder-cog", "on_accent"),
                    (context_page.open_settings_btn, "folder-cog", "default"),
                    (context_page.stop_btn, "player-stop", "danger"),
                )
            )

        for button, name, role in specs:
            if isinstance(button, QPushButton):
                set_tabler_button_icon(
                    button,
                    resources_dir,
                    name,
                    dark=dark,
                    role=role,
                )

    def _refresh_split_status_table_after_theme_change(self) -> None:
        self._configure_split_status_table_hover_palette()
        if not self._split_status_entries:
            return
        self._update_split_status_selection_ui(self._split_status_selected_manifest_path)

    def _configure_split_status_table_hover_palette(self) -> None:
        if not hasattr(self, "split_status_table"):
            return
        dark = self._effective_theme_is_dark()
        table_palette = self.split_status_table.palette()
        if dark:
            table_palette.setColor(
                QPalette.ColorGroup.All,
                QPalette.ColorRole.Highlight,
                QColor(148, 163, 184, 31),
            )
        else:
            table_palette.setColor(
                QPalette.ColorGroup.All,
                QPalette.ColorRole.Highlight,
                QColor(148, 163, 184, 26),
            )
        self.split_status_table.setPalette(table_palette)

    def _split_status_table_profile(self) -> dict[str, int]:
        table = getattr(self, "split_status_table", None)
        width = 0
        if table is not None:
            width = table.viewport().width() or table.width()
        if width >= 1500:
            return {
                "groups": 2,
                "part_width": 132,
                "status_width": 150,
                "item_width": 72,
                "chunk_width": 60,
                "action_width": 132,
                "job_chars": 42,
            }
        if width >= 980:
            return {
                "groups": 1,
                "part_width": 130,
                "status_width": 145,
                "item_width": 72,
                "chunk_width": 64,
                "action_width": 124,
                "job_chars": 64,
            }
        return {
            "groups": 1,
            "part_width": 112,
            "status_width": 118,
            "item_width": 62,
            "chunk_width": 56,
            "action_width": 108,
            "job_chars": 32,
        }

    def _configure_split_status_table_columns(self) -> dict[str, int]:
        profile = self._split_status_table_profile()
        if not hasattr(self, "split_status_table"):
            return profile
        group_count = max(1, profile["groups"])
        self._split_status_table_group_count = group_count
        self._split_status_table_profile_key = (group_count, profile["job_chars"])
        headers = ["\u5305", "\u72b6\u6001", "\u9879", "\u5757", "\u4e91\u7aef\u4efb\u52a1", "\u64cd\u4f5c"] * group_count
        if self.split_status_table.columnCount() != len(headers):
            self.split_status_table.setColumnCount(len(headers))
        self.split_status_table.setHorizontalHeaderLabels(headers)
        header = self.split_status_table.horizontalHeader()
        for group_index in range(group_count):
            base = group_index * 6
            header.setSectionResizeMode(base + 0, QHeaderView.ResizeMode.Fixed)
            header.setSectionResizeMode(base + 1, QHeaderView.ResizeMode.Fixed)
            header.setSectionResizeMode(base + 2, QHeaderView.ResizeMode.Fixed)
            header.setSectionResizeMode(base + 3, QHeaderView.ResizeMode.Fixed)
            header.setSectionResizeMode(base + 4, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(base + 5, QHeaderView.ResizeMode.Fixed)
            self.split_status_table.setColumnWidth(base + 0, profile["part_width"])
            self.split_status_table.setColumnWidth(base + 1, profile["status_width"])
            self.split_status_table.setColumnWidth(base + 2, profile["item_width"])
            self.split_status_table.setColumnWidth(base + 3, profile["chunk_width"])
            self.split_status_table.setColumnWidth(base + 5, profile["action_width"])
        self._ensure_split_status_action_delegates()
        return profile

    def _ensure_split_status_action_delegates(self) -> None:
        if not hasattr(self, "split_status_table"):
            return
        delegate = getattr(self, "_split_status_action_delegate", None)
        if delegate is None:
            return
        pass

    def _sync_split_status_table_columns(self) -> None:
        if not hasattr(self, "split_status_table"):
            return
        previous_profile_key = getattr(self, "_split_status_table_profile_key", None)
        profile = self._configure_split_status_table_columns()
        current_profile_key = (max(1, profile["groups"]), profile["job_chars"])
        if (
            current_profile_key != previous_profile_key
            and self.split_status_table.isVisible()
            and self._split_status_entries
        ):
            if (
                previous_profile_key is not None
                and current_profile_key[0] == previous_profile_key[0]
            ):
                self._update_split_status_job_column_texts(profile)
            else:
                entries = list(self._split_status_entries)
                selected_manifest_path = self._split_status_selected_manifest_path
                QTimer.singleShot(
                    0,
                    lambda entries=entries,
                    selected_manifest_path=selected_manifest_path,
                    profile_key=current_profile_key: self._deferred_render_split_status_entries(
                        entries,
                        selected_manifest_path=selected_manifest_path,
                        expected_profile_key=profile_key,
                    ),
                )

    def _deferred_render_split_status_entries(
        self,
        entries: list[SplitManifestEntry],
        *,
        selected_manifest_path: str,
        expected_profile_key: tuple[int, int],
    ) -> None:
        if not hasattr(self, "split_status_table") or not self.split_status_table.isVisible():
            return
        if not self._split_status_entries:
            return
        if list(self._split_status_entries) != entries:
            return
        if self._split_status_selected_manifest_path != selected_manifest_path:
            return
        profile = self._split_status_table_profile()
        current_profile_key = (max(1, profile["groups"]), profile["job_chars"])
        if current_profile_key != expected_profile_key:
            return
        self._render_split_status_entries(
            entries,
            selected_manifest_path=selected_manifest_path,
        )

    def _update_split_status_selection_ui(self, selected_manifest_path: str) -> None:
        if not hasattr(self, "split_status_table") or not self._split_status_entries:
            return
        rows = self.split_status_table.rowCount()
        if rows <= 0:
            return
        for index, entry in enumerate(self._split_status_entries):
            group_index = index // rows if rows else 0
            row = index % rows if rows else 0
            base_column = group_index * 6
            if base_column + 5 >= self.split_status_table.columnCount():
                continue
            is_current = self._same_manifest_path(entry.manifest_path, selected_manifest_path)
            part_item = self.split_status_table.item(row, base_column + 0)
            if part_item is not None:
                current_suffix = "\uff08\u5f53\u524d\uff09" if is_current else ""
                part_item.setText(f"{entry.part_label}{current_suffix}")
                font = part_item.font()
                font.setBold(is_current)
                part_item.setFont(font)
            action_item = self.split_status_table.item(row, base_column + 5)
            if action_item is None:
                continue
            show_action_button = entry.selectable and not is_current
            action_payload = split_action_item_payload(
                selectable=show_action_button,
                manifest_path=entry.manifest_path,
                part_label=entry.part_label,
            )
            if action_payload is not None:
                action_item.setData(SPLIT_ACTION_DATA_ROLE, action_payload)
                action_item.setToolTip(f"\u5207\u6362\u5230 {entry.part_label}")
            if show_action_button:
                btn = QPushButton("选择")
                btn.setObjectName("split_select_btn")
                btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                btn.setToolTip(f"切换到 {entry.part_label}")
                btn.clicked.connect(lambda checked=False, path=entry.manifest_path: self._select_split_manifest(path))
                self.split_status_table.setCellWidget(row, base_column + 5, btn)
            else:
                self.split_status_table.removeCellWidget(row, base_column + 5)
            
            self._apply_split_table_row_style(row, entry, base_column=base_column, is_current=is_current)

    def _update_split_status_job_column_texts(self, profile: dict[str, int]) -> None:
        if not hasattr(self, "split_status_table"):
            return
        rows = self.split_status_table.rowCount()
        if rows <= 0:
            return
        for index, entry in enumerate(self._split_status_entries):
            group_index = index // rows
            row = index % rows
            base_column = group_index * 6
            if base_column + 4 >= self.split_status_table.columnCount():
                continue
            self._set_split_table_item(
                row,
                base_column + 4,
                self._split_job_text(entry, profile["job_chars"]),
                tooltip=entry.job_name or entry.manifest_path,
            )

    def _render_split_status_entry(
        self,
        row: int,
        base_column: int,
        entry: SplitManifestEntry,
        profile: dict[str, int],
        *,
        selected_manifest_path: str,
    ) -> None:
        current_suffix = "\uff08\u5f53\u524d\uff09" if self._same_manifest_path(entry.manifest_path, selected_manifest_path) else ""
        is_current = bool(current_suffix)
        self._set_split_table_item(
            row,
            base_column + 0,
            f"{entry.part_label}{current_suffix}",
            tooltip=entry.manifest_path,
        )
        self._set_split_table_item(row, base_column + 1, entry.status_label, tooltip=entry.job_state or entry.status_label)
        self._set_split_table_item(row, base_column + 2, "" if entry.item_count is None else str(entry.item_count))
        self._set_split_table_item(row, base_column + 3, "" if entry.chunk_count is None else str(entry.chunk_count))
        self._set_split_table_item(
            row,
            base_column + 4,
            self._split_job_text(entry, profile["job_chars"]),
            tooltip=entry.job_name or entry.manifest_path,
        )
        
        show_action_button = entry.selectable and not is_current
        
        # Always set an empty QTableWidgetItem to allow background styling on the cell
        action_item = QTableWidgetItem("")
        action_item.setFlags(action_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.split_status_table.setItem(row, base_column + 5, action_item)
        self._apply_split_table_row_style(row, entry, base_column=base_column, is_current=is_current)
        
        if show_action_button:
            btn = QPushButton("选择")
            btn.setObjectName("split_select_btn")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setToolTip(f"切换到 {entry.part_label}")
            btn.clicked.connect(lambda checked=False, path=entry.manifest_path: self._select_split_manifest(path))
            self.split_status_table.setCellWidget(row, base_column + 5, btn)
        else:
            self.split_status_table.removeCellWidget(row, base_column + 5)

    def _split_job_text(self, entry: SplitManifestEntry, max_chars: int) -> str:
        text = entry.job_name if entry.job_name else entry.display_name
        if not text or max_chars <= 0 or len(text) <= max_chars:
            return text
        if text.startswith("batches/") and max_chars > 16:
            suffix_limit = max_chars - len("batches/") - 3
            return f"batches/{text.split('/', 1)[1][:suffix_limit]}..."
        return f"{text[:max(1, max_chars - 3)]}..."

    def _set_split_table_item(
        self,
        row: int,
        column: int,
        text: str,
        *,
        tooltip: str = "",
    ) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        alignment = Qt.AlignmentFlag.AlignVCenter
        if column % 6 in {2, 3, 5}:
            alignment |= Qt.AlignmentFlag.AlignHCenter
        else:
            alignment |= Qt.AlignmentFlag.AlignLeft
        item.setTextAlignment(alignment)
        if tooltip:
            item.setToolTip(tooltip)
        self.split_status_table.setItem(row, column, item)

    def _apply_split_table_row_style(
        self,
        row: int,
        entry: SplitManifestEntry,
        *,
        base_column: int = 0,
        is_current: bool,
    ) -> None:
        dark = self._effective_theme_is_dark()
        bg_color, text_color, status_color = self._split_status_row_colors(entry.status_kind)
        
        # Highlight the currently active/selected split manifest row with neutral colors.
        if is_current:
            bg_color = "#27272a" if dark else "#e4e4e7"
            text_color = "#ffffff" if dark else "#0f172a"

        background = QBrush(QColor(bg_color)) if bg_color != "transparent" else None
        foreground = QBrush(QColor(text_color))
        status_foreground = QBrush(QColor(status_color))
        
        for column in range(base_column, min(base_column + 6, self.split_status_table.columnCount())):
            item = self.split_status_table.item(row, column)
            if item is None:
                continue
            if background is not None:
                item.setBackground(background)
            else:
                item.setData(Qt.ItemDataRole.BackgroundRole, None)
            item.setForeground(status_foreground if column == base_column + 1 else foreground)
            font = item.font()
            font.setBold(column == base_column + 1 or (is_current and column == base_column))
            item.setFont(font)

    def _split_status_row_colors(self, status_kind: str) -> tuple[str, str, str]:
        dark = self._effective_theme_is_dark()
        if dark:
            colors = {
                "applied": ("transparent", "#cbd5e1", "#34d399"),
                "checked_safe": ("transparent", "#cbd5e1", "#4ade80"),
                "checked_warn": ("transparent", "#cbd5e1", "#fbbf24"),
                "checked_block": ("transparent", "#cbd5e1", "#f87171"),
                "failed": ("transparent", "#cbd5e1", "#f87171"),
                "downloaded": ("transparent", "#cbd5e1", "#60a5fa"),
                "running": ("transparent", "#cbd5e1", "#fbbf24"),
                "submitted": ("transparent", "#cbd5e1", "#93c5fd"),
                "succeeded": ("transparent", "#cbd5e1", "#60a5fa"),
                "unsubmitted": ("transparent", "#94a3b8", "#94a3b8"),
            }
            return colors.get(status_kind, ("transparent", "#cbd5e1", "#cbd5e1"))
        colors = {
            "applied": ("transparent", "#334155", "#059669"),
            "checked_safe": ("transparent", "#334155", "#15803d"),
            "checked_warn": ("transparent", "#334155", "#b45309"),
            "checked_block": ("transparent", "#334155", "#dc2626"),
            "failed": ("transparent", "#334155", "#dc2626"),
            "downloaded": ("transparent", "#334155", "#2563eb"),
            "running": ("transparent", "#334155", "#d97706"),
            "submitted": ("transparent", "#334155", "#2563eb"),
            "succeeded": ("transparent", "#334155", "#0284c7"),
            "unsubmitted": ("transparent", "#64748b", "#64748b"),
        }
        return colors.get(status_kind, ("transparent", "#334155", "#334155"))

    def _short_job_name(self, job_name: str) -> str:
        if not job_name:
            return ""
        if job_name.startswith("batches/"):
            suffix = job_name.split("/", 1)[1]
            return f"batches/{suffix[:10]}..." if len(suffix) > 13 else job_name
        return f"{job_name[:18]}..." if len(job_name) > 21 else job_name

    def _refresh_split_status_ui(
        self,
        *,
        manifest_path: str | None = None,
        manifest: dict[str, object] | None = None,
    ) -> list[SplitManifestEntry]:
        if manifest_path:
            entries = self._split_entries_for_manifest(manifest_path, manifest)
            selected_manifest_path = manifest_path
        else:
            selected_manifest_path, entries = self._load_latest_split_entries()
        self._render_split_status_entries(
            entries,
            selected_manifest_path=selected_manifest_path or "",
        )
        return entries

    def _update_split_submit_btn(
        self,
        entries: list[SplitManifestEntry] | None = None,
        *,
        running: bool | None = None,
    ) -> None:
        if not hasattr(self, "split_submit_btn"):
            return
        if work_mode_spec(self._current_work_mode()).mode != WorkMode.BATCH_TRANSLATION:
            self.split_submit_btn.setVisible(False)
            self.split_submit_btn.setEnabled(False)
            return
        current_entries = self._split_status_entries if entries is None else entries
        has_split_group = len(current_entries) > 1
        needs_submit = any(entry.needs_submit for entry in current_entries)
        if running is None:
            running = self.kill_btn.isEnabled()
        self.split_submit_btn.setVisible(has_split_group and needs_submit)
        self.split_submit_btn.setText("提交剩余包")
        self.split_submit_btn.setEnabled(has_split_group and needs_submit and not running)
        if hasattr(self, "action_panel"):
            self.action_panel.reflow(force=True)

    def _refresh_writeback_for_manifest_path(self, manifest_path: str) -> None:
        spec = work_mode_spec(self._current_work_mode())
        if not spec.supports_translation_writeback:
            self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
            return
        try:
            manifest = self.state.load_manifest_file(manifest_path, lite=True)
        except ValueError:
            self._set_writeback_summary(idle_writeback_summary())
            return
        summary = summarize_manifest_writeback(manifest)
        if summary is None:
            self._set_writeback_summary(idle_writeback_summary())
            return
        self._set_writeback_summary(summary)

    def _select_split_manifest(self, manifest_path: str) -> None:
        self._invalidate_manifest_caches(manifest_path)
        try:
            self.state.remember_latest_manifest_path(manifest_path)
        except ValueError as exc:
            QMessageBox.warning(self, "无法选择拆分包", str(exc))
            return
        self._workflow = None
        self._workflow_step_output_lines = []
        self._split_status_selected_manifest_path = manifest_path
        if self._split_status_entries:
            self._update_split_status_selection_ui(manifest_path)
        else:
            self._refresh_split_status_ui(manifest_path=manifest_path)
        self._refresh_writeback_for_manifest_path(manifest_path)
        try:
            selected_manifest = self.state.load_manifest_file(manifest_path, lite=True)
        except ValueError:
            selected_manifest = None
        if selected_manifest is not None:
            self._refresh_workflow_from_latest_manifest(
                latest_manifest=manifest_path,
                manifest=selected_manifest,
                split_entries=self._split_status_entries or None,
                force_expand=True,
            )
        writeback_summary = self._current_writeback_summary()
        if writeback_summary.status not in {"idle", "running", "stale"}:
            self._focus_workbench_status_tab(2)
        else:
            self._focus_workbench_status_tab(1)
        self.statusBar().showMessage("已选择拆分包；可继续下载、检查或写回。", 5000)

    def _writeback_issues_ready(self, summary: WritebackSummary) -> bool:
        if self._uses_revision_writeback():
            return False
        return summary.status == "warn" and bool(summary.manifest_path)

    def _load_writeback_manifest(self) -> dict[str, object] | None:
        if not self._writeback_manifest_path:
            return None
        try:
            return self.state.load_manifest_file(self._writeback_manifest_path)
        except ValueError:
            return None

    def _remediation_ready(
        self,
        summary: WritebackSummary,
        *,
        manifest: dict[str, object] | None = None,
    ) -> bool:
        if not self._writeback_issues_ready(summary):
            return False
        loaded = manifest if manifest is not None else self._load_writeback_manifest()
        if loaded is None:
            return False
        return retry_followup_allowed(
            loaded,
            parent_manifest_path=summary.manifest_path,
            confirmed_parent_paths=self._retry_followup_confirmed,
        )

    def _retry_button_mode(
        self,
        summary: WritebackSummary,
        *,
        manifest: dict[str, object] | None = None,
    ) -> str:
        if not self._writeback_issues_ready(summary):
            return "hidden"
        loaded = manifest if manifest is not None else self._load_writeback_manifest()
        if loaded is None:
            return "hidden"
        if existing_retry_manifest_path(loaded):
            return "view"
        eligibility = assess_retry_eligibility(
            loaded,
            manifest_path=summary.manifest_path,
        )
        return "build" if eligibility.eligible else "hidden"

    def _apply_failure_report_ready(
        self,
        summary: WritebackSummary,
        *,
        manifest: dict[str, object] | None,
    ) -> bool:
        if summary.status == "failed" and summary.manifest_path:
            return True
        if manifest is None or not summary.manifest_path:
            return False
        return apply_failure_report_available(
            manifest,
            manifest_path=summary.manifest_path,
        )

    def _open_apply_failure_report(self) -> None:
        manifest_path = self._writeback_manifest_path
        if not manifest_path:
            QMessageBox.warning(self, "无法查看写回失败报告", "当前没有可用的任务记录。")
            return
        try:
            manifest = self.state.load_manifest_file(manifest_path)
        except ValueError as exc:
            QMessageBox.warning(self, "无法查看写回失败报告", str(exc))
            return

        report = build_apply_failure_report(manifest, manifest_path=manifest_path)
        dialog = ApplyFailureDialog(self, report=report)
        dialog.exec()
        self.statusBar().showMessage("已查看写回失败报告。", 3000)

    def _set_writeback_issues_expanded(self, expanded: bool) -> None:
        self._writeback_issues_expanded = bool(expanded)
        if hasattr(self, "writeback_issues_panel"):
            self.writeback_issues_panel.setVisible(expanded)
            reflow = getattr(self.writeback_issues_panel, "reflow", None)
            if callable(reflow) and expanded:
                reflow(force=True)
        if hasattr(self, "writeback_issues_toggle_btn"):
            self.writeback_issues_toggle_btn.setText(
                "问题处理 ▾" if expanded else "问题处理 ▸"
            )

    def _toggle_writeback_issues_panel(self) -> None:
        self._set_writeback_issues_expanded(not getattr(self, "_writeback_issues_expanded", False))

    def _sync_action_frame_min_height(self) -> None:
        """Ensure action_frame / actions column are tall enough for stacked rows."""
        panel = getattr(self, "action_panel", None)
        frame = getattr(self, "_action_frame", None)
        if panel is None or frame is None:
            return
        margins = 20
        layout = frame.layout()
        if layout is not None:
            m = layout.contentsMargins()
            margins = int(m.top() + m.bottom())
        frame.setMinimumHeight(max(0, int(panel.minimumHeight()) + margins))
        frame.updateGeometry()

        # Advanced tools strip also reports min height after FlowButtonBar reflow.
        advanced = getattr(self, "batch_advanced_frame", None)
        if advanced is not None and advanced.isVisible():
            adv_layout = advanced.layout()
            adv_margins = 20
            if adv_layout is not None:
                m = adv_layout.contentsMargins()
                adv_margins = int(m.top() + m.bottom())
            bar = getattr(self, "batch_advanced_bar", None)
            bar_h = int(bar.minimumHeight()) if bar is not None else 38
            title_h = 22
            advanced.setMinimumHeight(max(60, bar_h + adv_margins + title_h))
            advanced.updateGeometry()
        elif advanced is not None:
            advanced.setMinimumHeight(0)

        column = getattr(self, "_workbench_actions_column", None)
        if column is not None:
            col_layout = column.layout()
            if isinstance(col_layout, QLayout):
                col_layout.invalidate()
                col_layout.activate()
            # Explicit column min height = sum of visible children mins + spacing.
            col_h = 0
            if isinstance(col_layout, QLayout):
                spacing = col_layout.spacing()
                visible_kids = 0
                for i in range(col_layout.count()):
                    item = col_layout.itemAt(i)
                    child = item.widget() if item is not None else None
                    if child is None or not child.isVisible():
                        continue
                    col_h += max(child.minimumHeight(), child.sizeHint().height())
                    visible_kids += 1
                if visible_kids > 1:
                    col_h += spacing * (visible_kids - 1)
            if col_h > 0:
                column.setMinimumHeight(col_h)
            column.updateGeometry()
            parent = column.parentWidget()
            if parent is not None:
                layout = parent.layout()
                if isinstance(layout, QLayout):
                    layout.invalidate()
                    layout.activate()

    def _reflow_button_bars(self) -> None:
        """Re-pack all flow/responsive button strips after visibility or size changes."""
        for name in (
            "action_panel",
            "diagnostics_action_panel",
            "global_project_actions",
            "writeback_primary_bar",
            "writeback_issues_panel",
            "batch_advanced_bar",
        ):
            panel = getattr(self, name, None)
            reflow = getattr(panel, "reflow", None) if panel is not None else None
            if callable(reflow):
                reflow(force=True)
        self._sync_action_frame_min_height()
        # Deferred second pass: parent widths settle after min-height changes.
        # Use the 2-arg form: helper tests construct MainWindow via __new__ without
        # QObject init, and the 3-arg context overload requires a live C++ base.
        QTimer.singleShot(0, self._reflow_button_bars_deferred)

    def _reflow_button_bars_deferred(self) -> None:
        for name in (
            "action_panel",
            "writeback_primary_bar",
            "writeback_issues_panel",
            "batch_advanced_bar",
            "global_project_actions",
        ):
            panel = getattr(self, name, None)
            reflow = getattr(panel, "reflow", None) if panel is not None else None
            if callable(reflow):
                reflow(force=True)
        self._sync_action_frame_min_height()

    def _sync_writeback_issues_panel_visibility(
        self,
        summary: WritebackSummary,
        *,
        force_expand: bool = False,
    ) -> None:
        """Show badge / auto-expand when recovery tools are relevant."""
        if not hasattr(self, "writeback_issues_toggle_btn"):
            return
        status = str(getattr(summary, "status", "") or "")
        needs_attention = status in {"warn", "failed", "block", "unknown"} or bool(
            force_expand
        )
        if hasattr(self, "writeback_issues_badge"):
            if needs_attention and status == "warn":
                self.writeback_issues_badge.setText("有待处理问题")
                self.writeback_issues_badge.setVisible(True)
            elif needs_attention and status in {"failed", "block", "unknown"}:
                self.writeback_issues_badge.setText("需处理")
                self.writeback_issues_badge.setVisible(True)
            else:
                self.writeback_issues_badge.setText("")
                self.writeback_issues_badge.setVisible(False)
        if needs_attention:
            self._set_writeback_issues_expanded(True)
        # Keep toggle visible for batch translation modes only; callers hide when N/A.
        self.writeback_issues_toggle_btn.setVisible(True)

    def _update_writeback_action_buttons(
        self,
        summary: WritebackSummary,
        *,
        running: bool,
    ) -> None:
        spec = work_mode_spec(self._current_work_mode())
        uses_revision_writeback = self._uses_revision_writeback(spec.mode)
        uses_keyword_merge = spec.mode in {
            WorkMode.KEYWORD_EXTRACTION,
            WorkMode.SYNC_KEYWORD_EXTRACTION,
        }
        has_writeback_actions = (
            spec.supports_translation_writeback
            or uses_revision_writeback
            or uses_keyword_merge
        )
        action_buttons = (
            "apply_btn",
            "apply_revision_btn",
            "recheck_btn",
            "check_issues_btn",
            "retry_btn",
            "retry_followup_btn",
            "repair_btn",
            "apply_failure_btn",
            "remediation_btn",
            "keyword_merge_writeback_btn",
        )
        if not has_writeback_actions:
            for button_name in action_buttons:
                if hasattr(self, button_name):
                    button = getattr(self, button_name)
                    button.setVisible(False)
                    button.setEnabled(False)
            if hasattr(self, "writeback_issues_toggle_btn"):
                self.writeback_issues_toggle_btn.setVisible(False)
            if hasattr(self, "writeback_issues_panel"):
                self.writeback_issues_panel.setVisible(False)
            if hasattr(self, "writeback_issues_badge"):
                self.writeback_issues_badge.setVisible(False)
            return

        needs_translation_manifest = (
            spec.supports_translation_writeback or uses_revision_writeback
        )
        keyword_only = uses_keyword_merge and not needs_translation_manifest
        if keyword_only:
            translation_buttons = (
                "apply_btn",
                "apply_revision_btn",
                "recheck_btn",
                "check_issues_btn",
                "retry_btn",
                "retry_followup_btn",
                "repair_btn",
                "apply_failure_btn",
                "remediation_btn",
            )
            for button_name in translation_buttons:
                if hasattr(self, button_name):
                    button = getattr(self, button_name)
                    button.setVisible(False)
                    button.setEnabled(False)

            manifest = None
            state = getattr(self, "state", None)
            if summary.manifest_path and state is not None:
                try:
                    manifest = state.load_manifest_file(summary.manifest_path)
                except ValueError:
                    manifest = None

            candidates_path = self._resolve_keyword_merge_candidates_path(
                manifest_path=summary.manifest_path,
                manifest=manifest,
            )

            glossary_path = ""
            if state is not None:
                try:
                    glossary_path = self._resolve_keyword_merge_glossary_path()
                except Exception:
                    glossary_path = ""

            keyword_merge_ready_flag, _message = keyword_merge_ready(
                candidates_path=candidates_path,
                glossary_path=glossary_path,
            )
            if hasattr(self, "keyword_merge_writeback_btn"):
                self.keyword_merge_writeback_btn.setVisible(
                    workbench_nav_for_work_mode(self._current_work_mode())
                    != WorkbenchNavItem.KEYWORDS
                )
                self.keyword_merge_writeback_btn.setEnabled(
                    not running and keyword_merge_ready_flag
                )
            if hasattr(self, "writeback_issues_toggle_btn"):
                self.writeback_issues_toggle_btn.setVisible(False)
            if hasattr(self, "writeback_issues_panel"):
                self.writeback_issues_panel.setVisible(False)
            if hasattr(self, "writeback_issues_badge"):
                self.writeback_issues_badge.setVisible(False)
            return

        issues_ready = self._writeback_issues_ready(summary)
        manifest = (
            self._load_writeback_manifest()
            if summary.manifest_path and needs_translation_manifest
            else None
        )

        recheck_ready = recheck_writeback_ready(
            summary,
            supports_translation_writeback=spec.supports_translation_writeback,
        )
        if hasattr(self, "recheck_btn"):
            self.recheck_btn.setVisible(spec.supports_translation_writeback)
            self.recheck_btn.setEnabled(not running and recheck_ready)

        if hasattr(self, "check_issues_btn"):
            self.check_issues_btn.setVisible(needs_translation_manifest)
            self.check_issues_btn.setEnabled(
                not running and needs_translation_manifest and issues_ready
            )

        apply_failure_ready = self._apply_failure_report_ready(
            summary,
            manifest=manifest,
        )
        if hasattr(self, "apply_failure_btn"):
            self.apply_failure_btn.setVisible(apply_failure_ready)
            self.apply_failure_btn.setEnabled(not running and apply_failure_ready)

        if hasattr(self, "retry_btn"):
            mode = (
                self._retry_button_mode(summary, manifest=manifest)
                if issues_ready
                else "hidden"
            )
            if mode == "hidden":
                self.retry_btn.setVisible(False)
                self.retry_btn.setEnabled(False)
            else:
                self.retry_btn.setVisible(True)
                self.retry_btn.setText(
                    "查看补译包" if mode == "view" else "生成补译包"
                )
                self.retry_btn.setEnabled(not running)

        retry_manifest = None
        retry_manifest_path = ""
        if manifest is not None:
            retry_manifest_path = existing_retry_manifest_path(manifest) or ""
            if retry_manifest_path:
                try:
                    retry_manifest = self.state.load_manifest_file(retry_manifest_path)
                except ValueError:
                    retry_manifest = None
        followup_ready = retry_followup_workflow_ready(
            summary,
            parent_manifest=manifest,
            retry_manifest=retry_manifest,
            retry_manifest_path=retry_manifest_path,
            parent_manifest_path=summary.manifest_path,
            confirmed_parent_paths=self._retry_followup_confirmed,
            supports_translation_writeback=spec.supports_translation_writeback,
        )
        if hasattr(self, "retry_followup_btn"):
            self.retry_followup_btn.setVisible(spec.supports_translation_writeback)
            if followup_ready and retry_manifest is not None and retry_manifest_path:
                label, tooltip = describe_retry_followup_button(
                    retry_manifest_path,
                    retry_manifest,
                    summary.manifest_path,
                )
                self.retry_followup_btn.setText(label)
                self.retry_followup_btn.setToolTip(tooltip)
            else:
                self.retry_followup_btn.setText("继续补译")
                self.retry_followup_btn.setToolTip("")
            self.retry_followup_btn.setEnabled(not running and followup_ready)

        repair_ready = (
            repair_action_ready(manifest, manifest_path=summary.manifest_path)
            if issues_ready and manifest is not None
            else False
        )
        if hasattr(self, "repair_btn"):
            self.repair_btn.setVisible(spec.supports_translation_writeback and repair_ready)
            self.repair_btn.setEnabled(not running and repair_ready)

        if hasattr(self, "remediation_btn"):
            remediation_ready = (
                self._remediation_ready(summary, manifest=manifest)
                if needs_translation_manifest and issues_ready
                else False
            )
            self.remediation_btn.setVisible(needs_translation_manifest)
            self.remediation_btn.setEnabled(not running and remediation_ready)

        if hasattr(self, "apply_btn"):
            self.apply_btn.setVisible(spec.supports_translation_writeback)
        if hasattr(self, "apply_revision_btn"):
            self.apply_revision_btn.setVisible(
                uses_revision_writeback
                and workbench_nav_for_work_mode(self._current_work_mode())
                != WorkbenchNavItem.REVISION
            )
        keyword_merge_ready_flag = False
        if uses_keyword_merge:
            candidates_path = self._resolve_keyword_merge_candidates_path(
                manifest_path=summary.manifest_path,
                manifest=manifest,
            )
            glossary_path = self._resolve_keyword_merge_glossary_path()
            keyword_merge_ready_flag, _message = keyword_merge_ready(
                candidates_path=candidates_path,
                glossary_path=glossary_path,
            )
        if hasattr(self, "keyword_merge_writeback_btn"):
            self.keyword_merge_writeback_btn.setVisible(
                uses_keyword_merge
                and workbench_nav_for_work_mode(self._current_work_mode())
                != WorkbenchNavItem.KEYWORDS
            )
            self.keyword_merge_writeback_btn.setEnabled(
                not running and keyword_merge_ready_flag
            )

        if hasattr(self, "writeback_issues_toggle_btn"):
            show_issues_chrome = bool(spec.supports_translation_writeback)
            self.writeback_issues_toggle_btn.setVisible(show_issues_chrome)
            if show_issues_chrome:
                attention = bool(
                    str(getattr(summary, "status", "") or "")
                    in {"warn", "failed", "block", "unknown"}
                    or issues_ready
                    or apply_failure_ready
                    or repair_ready
                    or followup_ready
                )
                self._sync_writeback_issues_panel_visibility(
                    summary,
                    force_expand=attention,
                )
                self.writeback_issues_panel.setVisible(self._writeback_issues_expanded)
            else:
                self.writeback_issues_panel.setVisible(False)
                if hasattr(self, "writeback_issues_badge"):
                    self.writeback_issues_badge.setVisible(False)
        if hasattr(self, "writeback_primary_bar"):
            self.writeback_primary_bar.reflow(force=True)
        if hasattr(self, "writeback_issues_panel"):
            reflow = getattr(self.writeback_issues_panel, "reflow", None)
            if callable(reflow) and self.writeback_issues_panel.isVisible():
                reflow(force=True)

    def _show_retry_preview(
        self,
        retry_manifest_path: str,
        *,
        start_followup_on_confirm: bool = False,
    ) -> str:
        try:
            retry_manifest = self.state.load_manifest_file(retry_manifest_path)
        except ValueError as exc:
            QMessageBox.warning(self, "无法预览补译包", str(exc))
            return "error"

        report = summarize_retry_manifest(
            retry_manifest,
            manifest_path=retry_manifest_path,
        )
        dialog = RetryPreviewDialog(self, report=report)
        dialog.exec()
        if not dialog.confirmed:
            return "cancelled"

        parent_path = report.parent_manifest_path or self._writeback_manifest_path
        if parent_path:
            self._retry_followup_confirmed.add(parent_path)
        summary = self._current_writeback_summary()
        self._update_writeback_action_buttons(
            summary,
            running=self.kill_btn.isEnabled(),
        )
        if start_followup_on_confirm:
            self._start_retry_followup_workflow()
        else:
            self.statusBar().showMessage("已确认补译包范围。", 3000)
        return "confirmed"

    def _start_retry_followup_workflow(self, *, retry_manifest_path: str = "") -> bool:
        parent_path = self._writeback_manifest_path
        if not parent_path:
            QMessageBox.information(self, "无法继续补译", "当前没有可用的父任务记录。")
            return False

        try:
            parent_manifest = self.state.load_manifest_file(parent_path)
        except ValueError as exc:
            QMessageBox.warning(self, "无法继续补译", str(exc))
            return False

        if parent_path not in self._retry_followup_confirmed:
            resolved_retry_path = retry_manifest_path or existing_retry_manifest_path(parent_manifest)
            if not resolved_retry_path:
                QMessageBox.information(self, "无法继续补译", "请先生成并预览补译包。")
                return False
            preview_result = self._show_retry_preview(resolved_retry_path)
            if preview_result != "confirmed":
                return False
            try:
                parent_manifest = self.state.load_manifest_file(parent_path)
            except ValueError as exc:
                QMessageBox.warning(self, "无法继续补译", str(exc))
                return False

        resolved_retry_path = retry_manifest_path or existing_retry_manifest_path(parent_manifest)
        if not resolved_retry_path:
            QMessageBox.information(self, "无法继续补译", "未找到补译任务记录。")
            return False

        try:
            retry_manifest = self.state.load_manifest_file(resolved_retry_path)
        except ValueError as exc:
            QMessageBox.warning(self, "无法继续补译", str(exc))
            return False

        workflow = create_retry_followup_workflow(
            resolved_retry_path,
            retry_manifest,
            parent_path,
            submit_max_cost=self._submit_max_cost_from_config(),
        )
        next_step = workflow.current_step()
        if next_step is None:
            QMessageBox.information(
                self,
                "无法继续补译",
                "补译流程暂无后续步骤；可查看诊断与工具任务上下文确认状态。",
            )
            return False

        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._begin_translation_workflow(
            workflow,
            log_heading=f"正在补译：gemini_translate_batch.py {' '.join(next_step.args)}",
        )
        return True

    def _on_retry_followup_action(self) -> None:
        if not work_mode_spec(self._current_work_mode()).supports_translation_writeback:
            QMessageBox.information(
                self,
                "当前模式不支持",
                "「继续补译」仅适用于批量翻译。",
            )
            return
        self._start_retry_followup_workflow()

    def _on_retry_action(self) -> None:
        summary = self._current_writeback_summary()
        mode = self._retry_button_mode(summary)
        if mode == "view":
            manifest = self._load_writeback_manifest()
            if manifest is None:
                return
            retry_path = existing_retry_manifest_path(manifest)
            if retry_path:
                self._show_retry_preview(retry_path)
            return

        if mode != "build" or not self._writeback_manifest_path:
            return

        manifest = self._load_writeback_manifest()
        if manifest is None:
            QMessageBox.warning(self, "无法生成补译包", "无法读取当前任务记录。")
            return

        eligibility = assess_retry_eligibility(
            manifest,
            manifest_path=self._writeback_manifest_path,
        )
        reply = QMessageBox.question(
            self,
            "确认生成补译包",
            "\n".join(
                [
                    eligibility.message,
                    "",
                    "将生成本地补译包。",
                    "界面不会自动提交云端任务；生成后会先让您预览范围。",
                ]
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._show_workbench_log_drawer()
        self._active_command = "build_retry"
        self._build_retry_output_lines = []
        self._set_task_running(True)
        self._append_log(
            "\n=== 正在生成补译包："
            f"gemini_translate_batch.py {' '.join(build_retry_cli_args(self._writeback_manifest_path))} ===\n"
        )
        self.runner.run(
            self.state.get_batch_script_path(),
            build_retry_cli_args(self._writeback_manifest_path),
        )

    def _open_check_issues(self) -> None:
        manifest_path = self._writeback_manifest_path
        if not manifest_path:
            QMessageBox.warning(self, "无法查看问题清单", "当前没有可用的任务记录。")
            return
        try:
            manifest = self.state.load_manifest_file(manifest_path)
        except ValueError as exc:
            QMessageBox.warning(self, "无法查看问题清单", str(exc))
            return

        report = build_check_issues_report(manifest, manifest_path=manifest_path)
        dialog = CheckIssuesDialog(
            self,
            report=report,
            show_repair_action=repair_action_ready(manifest, manifest_path=manifest_path),
        )
        dialog.repair_requested.connect(self._on_run_repair)
        dialog.exec()
        self.statusBar().showMessage("已查看检查问题清单。", 3000)

    def _open_remediation_commands(self) -> None:
        self._refresh_diagnostics_context()
        if self._diagnostics_tab is not None:
            self.tab_widget.setCurrentWidget(self._diagnostics_tab)
        if hasattr(self, "diagnostics_inner_tabs"):
            self.diagnostics_inner_tabs.setCurrentIndex(1)
        self.statusBar().showMessage("已打开补救命令参考。", 3000)

    def _set_details_label(self, label: QLabel, findings: list[str]) -> None:
        # Writeback details stay collapsed-away for now; doctor uses its own toggle.
        del findings
        label.setText("")
        label.setVisible(False)

    def _sync_doctor_details_toggle_chrome(self) -> None:
        """Keep disclosure label width-stable; only the marker character changes."""
        # Same character width for ▸/▾ avoids horizontal drift when expanding.
        marker = "▾" if self._doctor_details_expanded else "▸"
        self.doctor_details_toggle.setText(f"{marker} 更多详情")

    def _on_doctor_details_clicked(self) -> None:
        self._doctor_details_expanded = not self._doctor_details_expanded
        has_text = bool(self.doctor_details_label.text().strip())
        self.doctor_details_label.setVisible(self._doctor_details_expanded and has_text)
        self._sync_doctor_details_toggle_chrome()

    def _set_doctor_detail_facts(self, detail_facts: list[str] | None) -> None:
        lines = [line for line in (detail_facts or []) if str(line).strip()]
        if not lines:
            self._doctor_details_expanded = False
            self.doctor_details_toggle.setVisible(False)
            self.doctor_details_label.setText("")
            self.doctor_details_label.setVisible(False)
            self._sync_doctor_details_toggle_chrome()
            return
        self.doctor_details_label.setText("\n".join(lines))
        self.doctor_details_toggle.setVisible(True)
        self.doctor_details_label.setVisible(self._doctor_details_expanded)
        self._sync_doctor_details_toggle_chrome()

    def _clear_layout(self, layout: QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            child_layout = item.layout()
            if child_layout is not None:
                self._clear_layout(child_layout)
                child_layout.deleteLater()

    def _copy_to_clipboard(self, text: str, *, kind: str = "command") -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        if kind == "path":
            message = "路径已复制到剪贴板。"
        else:
            message = "命令已复制到剪贴板。"
        self.statusBar().showMessage(message, 3000)

    def _on_clear_log(self) -> None:
        self._clear_log_view()
        self.statusBar().showMessage("运行日志已清空。", 3000)

    def _resolve_diagnostics_manifest_path(self) -> str | None:
        if self._workflow is not None and self._workflow.manifest_path:
            return self._workflow.manifest_path
        spec = work_mode_spec(self._current_work_mode())
        if spec.manifest_mode is None:
            return None
        if self._writeback_manifest_path:
            return self._writeback_manifest_path
        game_root = self.state.get_game_root()
        latest_manifest = (
            self.state.get_latest_manifest_path_for_mode(game_root, spec.mode)
            if game_root is not None
            else None
        )
        return str(latest_manifest) if latest_manifest is not None else None

    def _load_diagnostics_manifest(self, manifest_path: str | None) -> dict[str, object] | None:
        if not manifest_path:
            return None
        try:
            return self.state.load_manifest_file(manifest_path)
        except ValueError:
            return None

    def _current_diagnostics_manifest(self) -> tuple[str, dict[str, object] | None]:
        spec = work_mode_spec(self._current_work_mode())
        if spec.manifest_mode is None:
            return "", None

        game_root = self.state.get_game_root()
        latest_manifest = None
        if game_root is not None:
            latest_manifest = self.state.get_latest_manifest_path_for_mode(
                game_root,
                spec.mode,
            )
        if self._workflow is not None and self._workflow.manifest_path:
            manifest_path = self._workflow.manifest_path
        elif self._writeback_manifest_path:
            manifest_path = self._writeback_manifest_path
        else:
            manifest_path = str(latest_manifest) if latest_manifest is not None else ""
        manifest = self._load_diagnostics_manifest(manifest_path or None)
        return manifest_path, manifest

    def _update_probe_btn_enabled(
        self,
        *,
        running: bool | None = None,
        manifest_path: str | None = None,
        manifest: dict[str, object] | None = None,
    ) -> None:
        if not hasattr(self, "probe_btn"):
            return
        if running is None:
            running = self.kill_btn.isEnabled()
        if manifest_path is None:
            manifest_path, manifest = self._current_diagnostics_manifest()
        ready, _message = translation_probe_ready(manifest_path, manifest)
        self.probe_btn.setEnabled(not running and ready)

    def _resolve_keyword_merge_glossary_path(self) -> str:
        import keyword_glossary_merge as merge_mod

        config = self.state.load_translator_config()
        game_root = str(self.state.get_game_root() or "")
        tool_root = str(self.state.get_tool_root())
        return merge_mod.resolve_glossary_path_from_config(
            config,
            game_root=game_root,
            tool_root=tool_root,
        )

    def _validated_cached_keyword_merge_candidates_path(self) -> str:
        cached_candidates = getattr(self, "_keyword_merge_candidates_path", "")
        if not cached_candidates or not os.path.isfile(cached_candidates):
            if cached_candidates:
                self._keyword_merge_candidates_path = ""
            return ""

        state = getattr(self, "state", None)
        if state is None:
            return cached_candidates

        game_root = state.get_game_root()
        if game_root is not None:
            try:
                root = Path(canonical_abs_path(str(game_root)))
                candidate = Path(canonical_abs_path(cached_candidates))
                if candidate == root or root in candidate.parents:
                    return cached_candidates
            except (OSError, ValueError):
                pass

        keyword_manifest_path, keyword_manifest = self._latest_keyword_extraction_manifest()
        expected = keyword_merge_candidates_path_from_manifest(
            keyword_manifest_path,
            keyword_manifest,
        )
        if expected:
            try:
                if (
                    canonical_abs_path(cached_candidates).lower()
                    == canonical_abs_path(expected).lower()
                ):
                    return cached_candidates
            except (OSError, ValueError):
                pass

        self._keyword_merge_candidates_path = ""
        return ""

    def _latest_keyword_extraction_manifest(
        self,
    ) -> tuple[str, dict[str, object] | None]:
        state = getattr(self, "state", None)
        if state is None:
            return "", None
        game_root = state.get_game_root()
        if game_root is None:
            return "", None
        latest_manifest = state.get_latest_manifest_path_for_mode(
            game_root,
            WorkMode.KEYWORD_EXTRACTION,
        )
        if latest_manifest is None:
            return "", None
        manifest_path = str(latest_manifest)
        return manifest_path, self._load_diagnostics_manifest(manifest_path)

    def _resolve_keyword_merge_candidates_path(
        self,
        *,
        manifest_path: str = "",
        manifest: dict[str, object] | None = None,
    ) -> str:
        cached_candidates = self._validated_cached_keyword_merge_candidates_path()
        if cached_candidates:
            return cached_candidates
        if manifest is None and manifest_path:
            state = getattr(self, "state", None)
            if state is not None:
                try:
                    manifest = state.load_manifest_file(manifest_path)
                except ValueError:
                    manifest = None
        if not manifest_path:
            manifest_path, manifest = self._current_diagnostics_manifest()
        elif manifest is None and getattr(self, "state", None) is None:
            manifest_path = ""
        resolved = keyword_merge_candidates_path_from_manifest(manifest_path, manifest)
        if resolved:
            return resolved
        writeback_manifest_path = getattr(self, "_writeback_manifest_path", "")
        if writeback_manifest_path:
            writeback_manifest = None
            state = getattr(self, "state", None)
            if state is not None:
                try:
                    writeback_manifest = state.load_manifest_file(writeback_manifest_path)
                except ValueError:
                    writeback_manifest = None
            resolved = keyword_merge_candidates_path_from_manifest(
                writeback_manifest_path,
                writeback_manifest,
            )
            if resolved:
                return resolved
        keyword_manifest_path, keyword_manifest = self._latest_keyword_extraction_manifest()
        return keyword_merge_candidates_path_from_manifest(
            keyword_manifest_path,
            keyword_manifest,
        )

    def _update_keyword_merge_btn_enabled(self, *, running: bool | None = None) -> None:
        if not hasattr(self, "keyword_merge_btn"):
            return
        if running is None:
            running = self.kill_btn.isEnabled()
        candidates_path = self._resolve_keyword_merge_candidates_path()
        glossary_path = self._resolve_keyword_merge_glossary_path()
        ready, message = keyword_merge_ready(
            candidates_path=candidates_path,
            glossary_path=glossary_path,
        )
        self.keyword_merge_btn.setEnabled(not running and ready)
        primary_hint = "主入口在工作台「关键词 / 术语」结果区。"
        if ready:
            self.keyword_merge_btn.setToolTip(
                f"{primary_hint} 勾选审核关键词候选并写入 glossary.json；不会修改 .rpy 脚本。",
            )
        elif message:
            self.keyword_merge_btn.setToolTip(
                f"{primary_hint} {message} 也可点击后手动选择候选 JSONL 文件。",
            )
        else:
            self.keyword_merge_btn.setToolTip(
                f"{primary_hint} 勾选审核关键词候选并写入 glossary.json；也可手动选择候选 JSONL。",
            )

    def _on_open_keyword_merge(self) -> None:
        candidates_path = self._resolve_keyword_merge_candidates_path()
        if not candidates_path:
            picked, _filter = QFileDialog.getOpenFileName(
                self,
                "选择关键词候选 JSONL",
                str(self.state.get_game_root() or self.state.get_tool_root()),
                "JSON Lines (*.jsonl);;All Files (*)",
            )
            candidates_path = picked.strip()
        glossary_path = self._resolve_keyword_merge_glossary_path()
        ready, message = keyword_merge_ready(
            candidates_path=candidates_path,
            glossary_path=glossary_path,
        )
        if not ready:
            QMessageBox.information(self, "无法合并关键词", message)
            return
        try:
            rows, candidates, resolved_glossary_path, _macro_path = load_keyword_merge_context(
                candidates_path=candidates_path,
                config=self.state.load_translator_config(),
                game_root=str(self.state.get_game_root() or ""),
                tool_root=str(self.state.get_tool_root()),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "无法读取候选", str(exc))
            return
        if not rows:
            QMessageBox.information(self, "没有可合并候选", "候选文件中没有可写入 glossary 的条目。")
            return

        dialog = KeywordMergeDialog(
            self,
            rows=rows,
            candidates_path=candidates_path,
            glossary_path=resolved_glossary_path,
            candidates=candidates,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = dialog.result
        if result is None:
            return

        merge_summary = summarize_keyword_merge_result(result.summary)
        self.statusBar().showMessage(merge_summary["message"], 8000)
        if result.summary.wrote_glossary:
            self._keyword_merge_candidates_path = candidates_path
        manifest_path, manifest = self._current_diagnostics_manifest()
        base_context = build_diagnostics_context(
            latest_manifest_path=manifest_path or None,
            manifest=manifest,
            batch_script_path=str(self.state.get_batch_script_path()),
            logs_dir=str(self.state.get_logs_dir()),
            python_exe=sys.executable,
            submit_max_cost=self._submit_max_cost_from_config(),
        )
        merged_context = DiagnosticsContext(
            status=str(merge_summary["status"]),
            heading=str(merge_summary["heading"]),
            message=str(merge_summary["message"]),
            facts=[*merge_summary["facts"], *base_context.facts],
            paths=base_context.paths,
            commands=base_context.commands,
            manifest_json_preview=base_context.manifest_json_preview,
        )
        self._set_diagnostics_context(merged_context)
        self._update_keyword_merge_btn_enabled()

    def _update_compare_variants_btn_enabled(
        self,
        *,
        running: bool | None = None,
        manifest_path: str | None = None,
        manifest: dict[str, object] | None = None,
    ) -> None:
        if not hasattr(self, "compare_variants_btn"):
            return
        if running is None:
            running = self.kill_btn.isEnabled()
        if manifest_path is None:
            manifest_path, manifest = self._current_diagnostics_manifest()
        ready, message = translation_ab_experiment_ready(manifest_path, manifest)
        self.compare_variants_btn.setEnabled(not running and ready)
        if ready:
            self.compare_variants_btn.setToolTip(
                "用同一批 manifest chunk 并排比较多个配置变体的同步译文，不会写回游戏文件。",
            )
        elif message:
            self.compare_variants_btn.setToolTip(message)

    def _update_split_btn_enabled(
        self,
        *,
        running: bool | None = None,
        manifest_path: str | None = None,
        manifest: dict[str, object] | None = None,
    ) -> None:
        if not hasattr(self, "split_btn"):
            return
        if running is None:
            running = self.kill_btn.isEnabled()
        if manifest_path is None:
            manifest_path, manifest = self._current_diagnostics_manifest()
        ready, _message = translation_split_ready(manifest_path, manifest)
        self.split_btn.setEnabled(not running and ready)

    def _refresh_manifest_derived_ui(
        self,
        *,
        refresh_writeback: bool = False,
        refresh_diagnostics: bool = False,
    ) -> None:
        spec = work_mode_spec(self._current_work_mode())
        game_root = self.state.get_game_root()
        latest_manifest = None
        manifest = None

        needs_manifest = (
            (spec.supports_resume and not self.kill_btn.isEnabled())
            or refresh_writeback
            or refresh_diagnostics
        )
        if needs_manifest and game_root is not None and spec.manifest_mode is not None:
            latest_manifest, manifest = self.state.load_latest_resume_manifest_for_mode(
                game_root,
                spec.mode,
            )

        self._refresh_workflow_from_latest_manifest(
            latest_manifest=latest_manifest,
            manifest=manifest,
        )
        if refresh_writeback:
            self._refresh_writeback_from_latest_manifest(
                latest_manifest=latest_manifest,
                manifest=manifest,
            )
        if refresh_diagnostics:
            self._refresh_diagnostics_context(
                latest_manifest_path=latest_manifest,
                manifest=manifest,
            )

    def _diagnostics_manifest_mtime_ns(self, manifest_path: str | None) -> int:
        if not manifest_path:
            return -1
        try:
            return Path(manifest_path).stat().st_mtime_ns
        except OSError:
            return -1

    def _refresh_diagnostics_context(
        self,
        *,
        latest_manifest_path: Path | str | None = None,
        manifest: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        spec = work_mode_spec(self._current_work_mode())
        running = self.kill_btn.isEnabled()
        if spec.mode == WorkMode.SYNC_TRANSLATION:
            input_key = (
                "sync",
                str(self.state.get_sync_script_path()),
                sys.executable,
                running,
            )
            if not force and self._diagnostics_refresh_input_key == input_key:
                return
            context = sync_diagnostics_context(
                sync_script_path=str(self.state.get_sync_script_path()),
                python_exe=sys.executable,
            )
            self._set_diagnostics_context(context)
            self._diagnostics_refresh_input_key = input_key
            self._update_probe_btn_enabled(running=running, manifest_path="", manifest=None)
            self._update_compare_variants_btn_enabled(
                running=running,
                manifest_path="",
                manifest=None,
            )
            self._update_keyword_merge_btn_enabled(running=running)
            self._update_split_btn_enabled(running=running, manifest_path="", manifest=None)
            return

        uses_batch_manifest = spec.manifest_mode is not None
        game_root = self.state.get_game_root()
        latest_manifest = latest_manifest_path
        if latest_manifest is None and uses_batch_manifest and game_root is not None:
            latest_manifest = self.state.get_latest_manifest_path_for_mode(
                game_root,
                spec.mode,
            )
        if self._workflow is not None and self._workflow.manifest_path:
            manifest_path = self._workflow.manifest_path
        elif self._writeback_manifest_path:
            manifest_path = self._writeback_manifest_path
        else:
            manifest_path = str(latest_manifest) if latest_manifest is not None else None
        shared_path = str(manifest_path or "")
        submit_max_cost = self._submit_max_cost_from_config()
        input_key = (
            "batch",
            spec.mode.value,
            str(game_root or ""),
            str(latest_manifest or ""),
            shared_path,
            self._diagnostics_manifest_mtime_ns(shared_path or None),
            str(self.state.get_batch_script_path()),
            str(self.state.get_logs_dir()),
            submit_max_cost,
            running,
        )
        if not force and self._diagnostics_refresh_input_key == input_key:
            # Inputs unchanged since last tab visit — skip disk/widget rebuild.
            return

        if manifest is None or (
            manifest_path
            and str(manifest.get("_manifest_path", "")) != str(manifest_path)
        ):
            manifest = self._load_diagnostics_manifest(manifest_path)
        context = build_diagnostics_context(
            latest_manifest_path=str(latest_manifest) if latest_manifest is not None else None,
            manifest=manifest,
            batch_script_path=str(self.state.get_batch_script_path()),
            logs_dir=str(self.state.get_logs_dir()),
            python_exe=sys.executable,
            submit_max_cost=submit_max_cost,
        )
        self._set_diagnostics_context(context)
        self._diagnostics_refresh_input_key = input_key
        # Share one loaded manifest across button readiness checks (tab switch hot path).
        self._update_probe_btn_enabled(
            running=running,
            manifest_path=shared_path,
            manifest=manifest,
        )
        self._update_compare_variants_btn_enabled(
            running=running,
            manifest_path=shared_path,
            manifest=manifest,
        )
        self._update_keyword_merge_btn_enabled(running=running)
        self._update_split_btn_enabled(
            running=running,
            manifest_path=shared_path,
            manifest=manifest,
        )

    def _set_diagnostics_context(self, context: DiagnosticsContext) -> None:
        fingerprint = (
            context.status,
            context.heading,
            context.message,
            tuple(context.facts),
            tuple((entry.label, entry.path) for entry in context.paths),
            tuple((command.label, command.command) for command in context.commands),
            context.manifest_json_preview,
        )
        if self._diagnostics_context_fingerprint == fingerprint:
            return
        self._diagnostics_context_fingerprint = fingerprint

        self.diagnostics_status_label.set_status(context.status, context.heading)
        self.diagnostics_message_label.setText(context.message)
        self.diagnostics_facts_label.setText("\n".join(context.facts))

        self._clear_layout(self.diagnostics_paths_layout)
        if context.paths:
            for entry in context.paths:
                row_host = QWidget()
                row_layout = QHBoxLayout(row_host)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)
                row_layout.addWidget(QLabel(f"{entry.label}："))
                path_edit = QLineEdit(entry.path)
                path_edit.setReadOnly(True)
                path_edit.setObjectName("diagnostics_path_edit")
                row_layout.addWidget(path_edit, 1)
                copy_btn = QPushButton("复制")
                copy_btn.setObjectName("secondary_btn")
                copy_btn.clicked.connect(
                    lambda _checked=False, text=entry.path: self._copy_to_clipboard(
                        text,
                        kind="path",
                    )
                )
                row_layout.addWidget(copy_btn)
                self.diagnostics_paths_layout.addWidget(row_host)
        else:
            placeholder = QLabel("暂无已生成的报告或数据文件。")
            placeholder.setWordWrap(True)
            placeholder.setObjectName("config_hint_label")
            self.diagnostics_paths_layout.addWidget(placeholder)

        self._clear_layout(self.diagnostics_commands_layout)
        if context.commands:
            for command in context.commands:
                row_host = QWidget()
                row_layout = QHBoxLayout(row_host)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)
                row_layout.addWidget(QLabel(f"{command.label}："))
                command_edit = QLineEdit(command.command)
                command_edit.setReadOnly(True)
                command_edit.setObjectName("diagnostics_command_edit")
                row_layout.addWidget(command_edit, 1)
                copy_btn = QPushButton("复制")
                copy_btn.setObjectName("secondary_btn")
                copy_btn.clicked.connect(
                    lambda _checked=False, text=command.command: self._copy_to_clipboard(text)
                )
                row_layout.addWidget(copy_btn)
                self.diagnostics_commands_layout.addWidget(row_host)
        else:
            placeholder = QLabel("开始翻译任务后，这里会出现可复制的手动命令。")
            placeholder.setWordWrap(True)
            placeholder.setObjectName("config_hint_label")
            self.diagnostics_commands_layout.addWidget(placeholder)

        self.diagnostics_manifest_preview.setPlainText(context.manifest_json_preview)

    def _current_work_mode(self) -> WorkMode:
        return normalize_work_mode(self._work_mode)

    def _selected_sync_backend(self) -> str:
        combo = getattr(self, "sync_backend_combo", None)
        if combo is None:
            return "gemini"
        value = combo.currentData()
        return value if value in {"gemini", "litellm"} else "gemini"

    def _litellm_model_text(self) -> str:
        combo = getattr(self, "litellm_model_combo", None)
        model = combo.currentText().strip() if combo is not None else ""
        if not model or "/" in model:
            return model
        provider_combo = getattr(self, "litellm_provider_combo", None)
        provider = (
            str(provider_combo.currentData() or "").strip().lower()
            if provider_combo is not None
            else ""
        )
        return f"{provider}/{model}" if provider else model

    def _current_litellm_provider(self) -> str:
        model_provider = provider_from_model(self._litellm_model_text())
        if model_provider:
            return model_provider
        combo = getattr(self, "litellm_provider_combo", None)
        if combo is not None:
            provider = str(combo.currentData() or "").strip().lower()
            if provider:
                return provider
        return provider_from_model(self._litellm_model_text())

    def _set_litellm_models(
        self,
        provider: str,
        models: tuple[str, ...],
        *,
        preserve_current: bool = False,
    ) -> None:
        combo = getattr(self, "litellm_model_combo", None)
        if combo is None:
            return
        current = combo.currentText().strip()
        selected = current if preserve_current else ""
        values = models or DEFAULT_MODELS.get(provider, ())
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(list(values))
        if selected:
            self._set_combo_value(combo, selected)
        elif values:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)
        self._on_litellm_model_changed(combo.currentText())

    def _refresh_litellm_credential_status(self) -> None:
        provider_label = getattr(self, "litellm_provider_label", None)
        status_label = getattr(self, "litellm_credential_status_label", None)
        if provider_label is None or status_label is None:
            return
        status = provider_credential_status(self._litellm_model_text(), os.environ)
        provider_label.setText(
            f"当前 provider：{status.provider}" if status.provider else "当前 provider：尚未识别"
        )
        saved_message = ""
        if status.provider and status.provider != "ollama":
            try:
                saved_message = (
                    "系统凭据管理器中已保存密钥。"
                    if load_provider_api_key(status.provider)
                    else "系统凭据管理器中尚未保存密钥。"
                )
            except ProviderCredentialStoreError as exc:
                saved_message = str(exc)
        status_label.setText(" ".join(part for part in (saved_message, status.message) if part))

    def _on_litellm_model_changed(self, _text: str) -> None:
        provider = provider_from_model(self._litellm_model_text())
        provider_combo = getattr(self, "litellm_provider_combo", None)
        if provider and provider_combo is not None and not getattr(self, "_updating_litellm_provider", False):
            index = provider_combo.findData(provider)
            if index >= 0 and index != provider_combo.currentIndex():
                provider_combo.blockSignals(True)
                provider_combo.setCurrentIndex(index)
                provider_combo.blockSignals(False)
        self._refresh_litellm_credential_status()

    def _on_litellm_provider_changed(self, _index: int) -> None:
        if self._updating_litellm_provider:
            return
        provider_combo = getattr(self, "litellm_provider_combo", None)
        provider = (
            str(provider_combo.currentData() or "").strip().lower()
            if provider_combo is not None
            else ""
        )
        self._updating_litellm_provider = True
        try:
            self._set_litellm_models(
                provider,
                self._litellm_catalog_models.get(provider, DEFAULT_MODELS.get(provider, ())),
            )
        finally:
            self._updating_litellm_provider = False
        self._refresh_litellm_credential_status()

    def _refresh_litellm_version_label(self) -> None:
        label = getattr(self, "litellm_version_label", None)
        if label is None:
            return
        installed = installed_litellm_version()
        latest = str(getattr(self, "_litellm_latest_version", "") or "")
        compatible = str(
            getattr(self, "_litellm_latest_compatible_version", "") or ""
        )
        requires_python = str(
            getattr(self, "_litellm_latest_requires_python", "") or ""
        )
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if not installed:
            label.setText("尚未安装；可检查 PyPI 最新稳定版。")
        elif not latest:
            label.setText(f"本机 {installed}；尚未检查 PyPI。")
        elif compatible and version_key(compatible) < version_key(latest):
            requirement = f"（要求 Python {requires_python}）" if requires_python else ""
            state = (
                f"建议更新到 {compatible}。"
                if version_key(installed) < version_key(compatible)
                else "已是当前 Python 可用最新版。"
            )
            label.setText(
                f"本机 {installed}；PyPI 最新稳定版 {latest}{requirement}不支持当前 "
                f"Python {python_version}；\n兼容最新版 {compatible}，{state}"
            )
        elif compatible and version_key(installed) < version_key(compatible):
            label.setText(f"本机 {installed}；最新兼容稳定版 {compatible}，建议更新。")
        elif compatible:
            label.setText(f"本机 {installed}；已是最新兼容稳定版。")
        elif version_key(installed) < version_key(latest):
            label.setText(f"本机 {installed}；最新稳定版 {latest}，建议更新。")
        else:
            label.setText(f"本机 {installed}；已是最新稳定版。")

    def _on_check_litellm_version(self) -> None:
        if getattr(self, "_litellm_version_worker", None) is not None:
            return
        button = getattr(self, "litellm_check_version_btn", None)
        if button is not None:
            button.setEnabled(False)
            button.setText("正在检查…")
        worker = LiteLLMVersionWorker(self)
        worker.completed.connect(self._on_litellm_version_checked)
        self._litellm_version_worker = worker
        worker.start()

    def _on_litellm_version_checked(
        self,
        installed: str,
        latest: str,
        compatible: str,
        requires_python: str,
        error: object,
    ) -> None:
        worker = getattr(self, "_litellm_version_worker", None)
        self._litellm_version_worker = None
        if worker is not None:
            worker.deleteLater()
        button = getattr(self, "litellm_check_version_btn", None)
        if button is not None:
            button.setText("检查更新")
        self._litellm_latest_version = latest
        self._litellm_latest_compatible_version = compatible
        self._litellm_latest_requires_python = requires_python
        self._refresh_litellm_version_label()
        if error:
            label = getattr(self, "litellm_version_label", None)
            if label is not None:
                current = f"本机 {installed}" if installed else "尚未安装"
                label.setText(f"{current}；检查更新失败，请稍后重试。")
        self._on_sync_backend_changed(-1)

    def _on_refresh_litellm_models(self) -> None:
        if self._litellm_catalog_worker is not None:
            return
        provider = self._current_litellm_provider()
        if not provider:
            return
        button = getattr(self, "litellm_refresh_models_btn", None)
        if button is not None:
            button.setEnabled(False)
            button.setText("正在加载…")
        api_key = ""
        if provider != "ollama":
            try:
                api_key = load_provider_api_key(provider)
            except ProviderCredentialStoreError:
                api_key = ""
        worker = LiteLLMModelCatalogWorker(provider, api_key=api_key, parent=self)
        worker.completed.connect(
            lambda models, source, error, selected=provider: self._on_litellm_models_loaded(
                selected, models, error, source
            )
        )
        self._litellm_catalog_worker = worker
        worker.start()

    def _on_litellm_models_loaded(
        self, provider: str, models: object, error: object, source: str = ""
    ) -> None:
        worker = self._litellm_catalog_worker
        self._litellm_catalog_worker = None
        if worker is not None:
            worker.deleteLater()
        button = getattr(self, "litellm_refresh_models_btn", None)
        if button is not None:
            button.setText("联网更新列表")
        self._on_sync_backend_changed(-1)
        if error and not models:
            QMessageBox.warning(self, "模型列表加载失败", str(error))
            return
        values = tuple(str(model) for model in models)
        self._litellm_catalog_models[provider] = values
        self._litellm_catalog_source = source
        source_label = getattr(self, "litellm_catalog_status_label", None)
        if source_label is not None:
            source_label.setText(catalog_source_label(str(source or "")))
        if self._current_litellm_provider() == provider:
            self._set_litellm_models(provider, values, preserve_current=True)
        message = f"已加载 {len(values)} 个 {provider} 模型。"
        if error:
            message = f"{message} {error}"
        self.statusBar().showMessage(message, 8000)

    def _on_save_litellm_key(self) -> None:
        provider = self._current_litellm_provider()
        api_key = self.litellm_api_key_edit.text().strip()
        try:
            store_provider_api_key(provider, api_key)
        except (ValueError, ProviderCredentialStoreError) as exc:
            QMessageBox.warning(self, "无法保存密钥", str(exc))
            return
        self.litellm_api_key_edit.clear()
        self._refresh_litellm_credential_status()
        self.statusBar().showMessage(f"已安全保存 {provider} 密钥。", 5000)

    def _on_delete_litellm_key(self) -> None:
        provider = self._current_litellm_provider()
        try:
            deleted = delete_provider_api_key(provider)
        except ProviderCredentialStoreError as exc:
            QMessageBox.warning(self, "无法删除密钥", str(exc))
            return
        self.litellm_api_key_edit.clear()
        self._refresh_litellm_credential_status()
        message = "已删除保存的密钥。" if deleted else "没有找到已保存的密钥。"
        self.statusBar().showMessage(message, 5000)

    def _on_test_litellm_connection(self) -> None:
        if self._litellm_connection_worker is not None:
            return
        model = self._litellm_model_text()
        if not model:
            QMessageBox.information(self, "缺少模型", "请先选择或填写模型。")
            return
        api_key = self.litellm_api_key_edit.text().strip()
        self.litellm_test_connection_btn.setEnabled(False)
        self.litellm_test_connection_btn.setText("正在测试…")
        self.litellm_connection_status_label.setText("正在后台发起最小请求…")
        worker = LiteLLMConnectionTestWorker(model, api_key, self)
        worker.completed.connect(self._on_litellm_connection_tested)
        self._litellm_connection_worker = worker
        worker.start()

    def _on_litellm_connection_tested(self, success: bool, message: str) -> None:
        worker = self._litellm_connection_worker
        self._litellm_connection_worker = None
        if worker is not None:
            worker.deleteLater()
        self.litellm_test_connection_btn.setText("测试连接")
        self.litellm_connection_status_label.setText(message)
        self._on_sync_backend_changed(-1)
        if not success:
            self.statusBar().showMessage("LiteLLM 连接测试失败。", 5000)

    def _saved_sync_backend(self) -> str:
        try:
            config = self.state.load_translator_config()
        except Exception:
            return "gemini"
        sync = config.get("sync") if isinstance(config, dict) else None
        if not isinstance(sync, dict):
            return "gemini"
        value = str(sync.get("backend") or "gemini").strip().lower()
        return value if value in {"gemini", "litellm"} else "gemini"

    def _on_sync_backend_changed(self, _index: int) -> None:
        backend = self._selected_sync_backend()
        hint = getattr(self, "sync_backend_hint", None)
        if hint is None:
            return
        install_btn = getattr(self, "install_litellm_btn", None)
        install_progress = getattr(self, "litellm_install_progress", None)
        installing = self._litellm_install_running()
        installed_version = installed_litellm_version()
        installed = bool(installed_version) and importlib.util.find_spec("litellm") is not None
        if backend == "litellm":
            installed_version = installed_litellm_version()
            installed = bool(installed_version) and importlib.util.find_spec("litellm") is not None
            keyring_installed = importlib.util.find_spec("keyring") is not None
            state = "正在后台安装" if installing else ("已安装" if installed else "尚未安装")
            credential_state = "可用" if keyring_installed else "尚未安装"
            hint.setText(
                "同步替代模式；不使用 Gemini API Key，也没有远程 Batch 恢复。"
                f"LiteLLM：{state}；安全凭据支持：{credential_state}。"
            )
            if install_btn is not None:
                latest = str(getattr(self, "_litellm_latest_version", "") or "")
                compatible = str(
                    getattr(self, "_litellm_latest_compatible_version", "") or ""
                )
                target = compatible if latest else ""
                up_to_date = bool(
                    installed
                    and target
                    and version_key(installed_version) >= version_key(target)
                )
                compatibility_limited = bool(
                    latest and compatible and version_key(compatible) < version_key(latest)
                )
                no_compatible_release = bool(latest and not compatible)
                install_btn.setVisible(True)
                install_btn.setEnabled(
                    not installing
                    and not no_compatible_release
                    and not (up_to_date and keyring_installed)
                )
                if installing:
                    install_btn.setText("正在更新…" if installed else "正在安装…")
                elif not installed:
                    install_btn.setText("安装 LiteLLM")
                elif no_compatible_release:
                    install_btn.setText("当前 Python 无兼容版本")
                elif up_to_date and keyring_installed:
                    install_btn.setText(
                        "当前 Python 可用最新版"
                        if compatibility_limited
                        else "已是最新版"
                    )
                else:
                    install_btn.setText("更新 LiteLLM")
        else:
            hint.setText(
                "推荐路径仍为 Gemini；同步配置位于「模型」与「密钥」页，批量离线翻译仍使用 Gemini Batch。"
            )
            if install_btn is not None:
                install_btn.setVisible(False)
        if install_progress is not None:
            install_progress.setVisible(installing)
            if installing:
                # pip does not expose a trustworthy total; busy mode gives honest
                # visual feedback without inventing a completion percentage.
                install_progress.setRange(0, 0)
                install_progress.setFormat("正在后台更新 LiteLLM…" if installed else "正在后台安装 LiteLLM…")

        model_combo = getattr(self, "litellm_model_combo", None)
        if model_combo is not None:
            model_combo.setEnabled(backend == "litellm" and not installing)
        gemini_sync_model_combo = getattr(self, "sync_model_combo", None)
        if gemini_sync_model_combo is not None:
            gemini_sync_model_combo.setEnabled(backend == "gemini")
            gemini_sync_model_combo.setToolTip(
                "当前同步后端为 LiteLLM；切回 Gemini 后可选择此模型。"
                if backend == "litellm"
                else ""
            )
        for name in (
            "litellm_provider_combo",
            "litellm_refresh_models_btn",
            "litellm_api_key_edit",
            "litellm_save_key_btn",
            "litellm_delete_key_btn",
            "litellm_test_connection_btn",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(backend == "litellm" and not installing)
        version_button = getattr(self, "litellm_check_version_btn", None)
        if version_button is not None:
            checking = getattr(self, "_litellm_version_worker", None) is not None
            version_button.setEnabled(not installing and not checking)
        # Skip while loading config into widgets: _load_config_to_ui decides
        # whether to re-gate after the load (and cold start defers that work).
        if hasattr(self, "translate_btn") and not getattr(
            self, "_loading_config_to_ui", False
        ):
            self._set_task_running(bool(getattr(self, "_task_running", False)))
        self._refresh_litellm_credential_status()
        self._refresh_litellm_install_action_gating()

    def _refresh_litellm_install_action_gating(self) -> None:
        """Disable only LiteLLM-backed task actions while installation is active."""
        translate_btn = getattr(self, "translate_btn", None)
        if translate_btn is None:
            return
        mode = self._current_work_mode()
        if self._litellm_install_blocks_mode(mode):
            translate_btn.setEnabled(False)
            self._sync_sync_translation_page_controls()
            self._sync_keywords_page_controls()
            self._sync_revision_page_controls()

    def _litellm_install_blocks_mode(self, mode: WorkMode) -> bool:
        return (
            self._litellm_install_running()
            and self._saved_sync_backend() == "litellm"
            and mode in self._sync_work_modes_requiring_api_key()
        )

    def _ensure_litellm_install_controller(self) -> OptionalFeatureInstallController:
        controller = getattr(self, "_litellm_install", None)
        if controller is not None:
            return controller
        controller = build_litellm_controller(self)
        controller.output_received.connect(self._on_optional_feature_install_output)
        controller.state_changed.connect(self._on_litellm_install_state_changed)
        controller.finished.connect(self._on_litellm_install_finished)
        self._litellm_install = controller
        return controller

    def _litellm_install_running(self) -> bool:
        controller = getattr(self, "_litellm_install", None)
        if controller is not None and controller.is_running():
            return True
        # Lightweight unit tests may force busy state without a QProcess.
        return bool(getattr(self, "_litellm_install_active", False))

    def _on_install_litellm(self) -> None:
        controller = self._ensure_litellm_install_controller()
        if controller.is_running():
            return
        started, message = controller.start_install()
        if not started:
            QMessageBox.information(self, "无法安装 LiteLLM", message)
            return
        self._on_sync_backend_changed(-1)

    def _on_litellm_install_state_changed(
        self,
        _feature_id: str,
        _status: FeatureStatus,
    ) -> None:
        self._on_sync_backend_changed(-1)

    def _on_litellm_install_finished(
        self,
        feature_id: str,
        succeeded: bool,
        message: str,
    ) -> None:
        if feature_id != "litellm":
            return
        if succeeded:
            self.statusBar().showMessage(message, 5000)
        else:
            QMessageBox.warning(
                self,
                "LiteLLM 安装失败",
                f"{message}\n请查看工作台日志中的 pip 输出。",
            )
        self._refresh_litellm_version_label()
        self._on_sync_backend_changed(-1)
        if succeeded:
            self._on_check_litellm_version()
            if self._selected_sync_backend() == "litellm":
                self._on_refresh_litellm_models()

    def _sync_work_modes_requiring_api_key(self) -> frozenset[WorkMode]:
        return frozenset(
            {
                WorkMode.SYNC_TRANSLATION,
                WorkMode.SYNC_KEYWORD_EXTRACTION,
                WorkMode.SYNC_REVISION,
            }
        )

    def _revision_writeback_modes(self) -> frozenset[WorkMode]:
        return frozenset({WorkMode.REVISION, WorkMode.SYNC_REVISION})

    def _uses_revision_writeback(self, mode: WorkMode | None = None) -> bool:
        return work_mode_spec(mode or self._current_work_mode()).mode in (
            self._revision_writeback_modes()
        )

    def _capture_mode_session(self) -> WorkbenchModeSession:
        # Align with product default (进度 tab) when tabs are unavailable.
        stage_index = _BATCH_STAGE_EXECUTE
        if hasattr(self, "workbench_status_tabs"):
            stage_index = self._current_batch_stage_index()
        if getattr(self, "_shell_special_route", None) is not None:
            saved_index = getattr(self, "_shell_task_status_index", None)
            if saved_index is not None:
                stage_index = int(saved_index)
        wf_status = ""
        wf_heading = ""
        if hasattr(self, "workflow_status_label"):
            raw = self.workflow_status_label.property("status")
            wf_status = str(raw) if raw is not None else ""
            # StatusBadge text includes an icon prefix; keep raw heading if tracked.
            wf_heading = str(getattr(self, "_workflow_heading_text", "") or "")
            if not wf_heading:
                wf_heading = self.workflow_status_label.text()
        wf_message = ""
        if hasattr(self, "workflow_message_label"):
            wf_message = self.workflow_message_label.text()
        wf_facts: list[str] = []
        if hasattr(self, "workflow_facts_label"):
            facts_text = self.workflow_facts_label.text()
            wf_facts = [line for line in facts_text.splitlines() if line.strip()]
        return WorkbenchModeSession(
            workflow=getattr(self, "_workflow", None),
            workflow_step_output_lines=list(
                getattr(self, "_workflow_step_output_lines", []) or []
            ),
            writeback_manifest_path=str(
                getattr(self, "_writeback_manifest_path", "") or ""
            ),
            completed_manifest_snapshot=getattr(
                self, "_completed_manifest_snapshot", None
            ),
            viewing_completed_manifest=bool(
                getattr(self, "_viewing_completed_manifest", False)
            ),
            keyword_merge_candidates_path=str(
                getattr(self, "_keyword_merge_candidates_path", "") or ""
            ),
            stage_index=stage_index,
            workflow_status=wf_status,
            workflow_heading=wf_heading,
            workflow_message=wf_message,
            workflow_facts=wf_facts,
            writeback_summary=getattr(self, "_writeback_summary", None),
        )

    def _restore_mode_session(self, session: WorkbenchModeSession) -> None:
        self._workflow = session.workflow
        self._workflow_step_output_lines = list(session.workflow_step_output_lines)
        self._writeback_manifest_path = session.writeback_manifest_path
        self._completed_manifest_snapshot = session.completed_manifest_snapshot
        self._viewing_completed_manifest = session.viewing_completed_manifest
        self._keyword_merge_candidates_path = session.keyword_merge_candidates_path
        # Keep 0 (准备) as a real stage — do not use `or` falsy fallback.
        self._pending_restore_stage_index = int(session.stage_index)
        self._pending_restore_workflow_ui = {
            "status": session.workflow_status,
            "heading": session.workflow_heading,
            "message": session.workflow_message,
            "facts": list(session.workflow_facts),
        }
        self._pending_restore_writeback_summary = session.writeback_summary

    def _apply_session_workflow_ui(self, payload: dict[str, object] | None) -> bool:
        """Re-paint progress labels from a frozen session; return True if applied."""
        if not payload:
            return False
        status = str(payload.get("status") or "").strip()
        heading = str(payload.get("heading") or "").strip()
        message = str(payload.get("message") or "")
        facts_raw = payload.get("facts") or []
        facts = [str(x) for x in facts_raw] if isinstance(facts_raw, list) else []
        if not status and not heading and not message:
            return False
        # Skip pure idle placeholders so mode switch can still show mode-specific idle.
        if status in {"", "idle", "stale"} and not facts and not message.strip():
            return False
        self._set_workflow_summary(status or "idle", heading or "任务状态", message, facts)
        return True

    def _clear_all_mode_sessions(self) -> None:
        """Clear per-mode sessions; safe on partially constructed MainWindow test stubs."""
        sessions = getattr(self, "_mode_sessions", None)
        if isinstance(sessions, dict):
            sessions.clear()
            return
        try:
            self._mode_sessions = {}
        except Exception:
            # QObject stubs from unit tests may reject new attributes.
            pass

    def _reset_last_mode_by_nav(self) -> None:
        """Restore each navigation entry's default mode after a project switch."""
        self._last_mode_by_nav = {
            item: default_work_mode_for_nav(item) for item in WORKBENCH_NAV_ORDER
        }

    def _sync_task_selectors_from_work_mode(self) -> None:
        """Sync the visible navigation and page-local mode selectors."""
        mode = self._current_work_mode()
        nav_item = workbench_nav_for_work_mode(mode)
        self._workbench_nav_item = nav_item

        if hasattr(self, "workbench_nav"):
            blocked = self.workbench_nav.blockSignals(True)
            for row in range(self.workbench_nav.count()):
                item = self.workbench_nav.item(row)
                if item is not None and item.data(Qt.ItemDataRole.UserRole) == nav_item.value:
                    self.workbench_nav.setCurrentRow(row)
                    break
            self.workbench_nav.blockSignals(blocked)

        sessions = getattr(self, "_mode_sessions", None)
        session = sessions.get(mode) if isinstance(sessions, dict) else None
        self._workbench_coordinator.activate(
            mode,
            session or WorkbenchModeSession(),
            running=self._task_page_running_chrome(),
        )
        if nav_item == WorkbenchNavItem.BATCH_TRANSLATION:
            self._sync_batch_translation_page_controls()
        elif nav_item == WorkbenchNavItem.SYNC_TRANSLATION:
            self._sync_sync_translation_page_controls()
        elif nav_item == WorkbenchNavItem.KEYWORDS:
            self._sync_keywords_page_controls()
        elif nav_item == WorkbenchNavItem.REVISION:
            self._sync_revision_page_controls()
        self._sync_shell_nav_selection()

    def _on_workbench_nav_changed(self, row: int) -> None:
        if row < 0:
            return
        if self._context_switching_locked():
            self._sync_task_selectors_from_work_mode()
            return
        item = self.workbench_nav.item(row)
        if item is None:
            return
        nav_item = WorkbenchNavItem(str(item.data(Qt.ItemDataRole.UserRole)))
        if nav_item == self._workbench_nav_item and workbench_nav_for_work_mode(
            self._work_mode
        ) == nav_item:
            return
        target_mode = self._last_mode_by_nav.get(
            nav_item,
            default_work_mode_for_nav(nav_item),
        )
        self._set_work_mode(target_mode, refresh_manifest_writeback=True)

    def _on_keywords_page_mode_selected(self, mode: WorkMode) -> None:
        """Apply the page-local batch/sync selector through the coordinator."""
        if self._context_switching_locked():
            self._sync_task_selectors_from_work_mode()
            return
        if mode in {WorkMode.KEYWORD_EXTRACTION, WorkMode.SYNC_KEYWORD_EXTRACTION}:
            self._set_work_mode(mode, refresh_manifest_writeback=True)

    def _on_revision_page_mode_selected(self, mode: WorkMode) -> None:
        """Apply the page-local revision batch/sync selector through the coordinator."""
        if self._context_switching_locked():
            self._sync_task_selectors_from_work_mode()
            return
        if mode in {WorkMode.REVISION, WorkMode.SYNC_REVISION}:
            self._set_work_mode(mode, refresh_manifest_writeback=True)

    def _set_work_mode(
        self,
        mode: WorkMode,
        *,
        refresh_manifest_writeback: bool,
        reset_session: bool = False,
    ) -> None:
        """Activate *mode*, preserving per-mode sessions unless reset_session=True."""
        mode = normalize_work_mode(mode)
        self._set_shell_special_route(None)
        if hasattr(self, "workbench_stack"):
            self.workbench_stack.show()
        previous = getattr(self, "_work_mode", None)
        if not hasattr(self, "_mode_sessions"):
            self._mode_sessions = {}

        if previous is not None and previous != mode and not reset_session:
            self._mode_sessions[previous] = self._capture_mode_session()

        self._work_mode = mode
        self._workbench_nav_item = workbench_nav_for_work_mode(mode)
        last_mode_by_nav = getattr(self, "_last_mode_by_nav", None)
        if not isinstance(last_mode_by_nav, dict):
            self._reset_last_mode_by_nav()
            last_mode_by_nav = self._last_mode_by_nav
        last_mode_by_nav[self._workbench_nav_item] = mode
        self._pending_restore_workflow_ui = None
        self._pending_restore_writeback_summary = None

        if reset_session or mode not in self._mode_sessions:
            self._mode_sessions[mode] = WorkbenchModeSession()
            self._workflow = None
            self._workflow_step_output_lines = []
            self._writeback_manifest_path = ""
            if hasattr(self, "_clear_completed_manifest_snapshot"):
                self._clear_completed_manifest_snapshot()
            else:
                self._completed_manifest_snapshot = None
                self._viewing_completed_manifest = False
            self._keyword_merge_candidates_path = ""
        else:
            self._restore_mode_session(self._mode_sessions[mode])

        self._apply_work_mode_ui(refresh_manifest_writeback=refresh_manifest_writeback)

    def _update_resume_btn_text(self) -> None:
        if not hasattr(self, "resume_btn") or not hasattr(self, "workflow_status_label"):
            return
        spec = work_mode_spec(self._current_work_mode())
        is_waiting = False
        if self._workflow is not None:
            step = self._workflow.current_step()
            is_waiting = step is not None and step.key == "status"

        status = self.workflow_status_label.property("status")
        if status == "waiting" or is_waiting:
            self.resume_btn.setText("查询云端状态")
        else:
            self.resume_btn.setText(spec.resume_button_label or "继续任务")

    def _resume_button_is_query_mode(self) -> bool:
        return hasattr(self, "resume_btn") and self.resume_btn.text() == "查询云端状态"

    def _latest_resume_manifest_path(self, game_root, spec_or_mode):
        """Choose the cloud task behind an explicit status query when possible."""
        mode = getattr(spec_or_mode, "mode", spec_or_mode)
        if self._resume_button_is_query_mode():
            get_submitted = getattr(
                self.state,
                "get_latest_submitted_manifest_path_for_mode",
                None,
            )
            if callable(get_submitted):
                submitted = get_submitted(game_root, mode)
                if submitted is not None:
                    return submitted
        return self.state.get_latest_manifest_path_for_mode(game_root, mode)
    def _resume_task_available(self) -> tuple[bool, str]:
        """Whether 「继续」 can load a resumable task for the current mode (P3 / #166)."""
        spec = work_mode_spec(self._current_work_mode())
        if not spec.implemented or not spec.supports_resume:
            return False, "当前模式不支持继续任务。"
        if self.kill_btn.isEnabled() or bool(getattr(self, "_task_running", False)):
            return False, "任务运行中。"
        # Mid-workflow waiting: keep enabled for status query.
        if self._workflow is not None and self._workflow.current_step() is not None:
            return True, ""
        game_root = self.state.get_game_root() if hasattr(self, "state") else None
        if game_root is None:
            return False, "请先选择游戏的 work 目录。"
        latest = self._latest_resume_manifest_path(game_root, spec)
        if latest is None:
            return False, f"未找到可继续的{spec.label}任务；请先开始一个任务。"
        try:
            self.state.load_resume_manifest(latest, work_mode=spec.mode)
        except ValueError as exc:
            return False, str(exc)
        return True, ""

    def _update_resume_btn_enabled(
        self,
        *,
        running: bool | None = None,
        resume_available: tuple[bool, str] | None = None,
    ) -> None:
        if not hasattr(self, "resume_btn"):
            return
        if running is None:
            running = bool(getattr(self, "_task_running", False)) or (
                hasattr(self, "kill_btn") and self.kill_btn.isEnabled()
            )
        spec = work_mode_spec(self._current_work_mode())
        if not spec.supports_resume or not spec.implemented:
            self.resume_btn.setEnabled(False)
            self.resume_btn.setToolTip("")
            return
        if running:
            self.resume_btn.setEnabled(False)
            self.resume_btn.setToolTip("任务运行中，请等待结束后再继续。")
            return
        if resume_available is None:
            available, reason = self._resume_task_available()
        else:
            available, reason = resume_available
        self.resume_btn.setEnabled(available)
        if available:
            self.resume_btn.setToolTip("从最近任务记录继续未完成的步骤，或查询云端状态。")
        else:
            self.resume_btn.setToolTip(reason or "当前没有可继续的任务。")

    def _sync_workbench_empty_states(
        self,
        *,
        resume_available: tuple[bool, str] | None = None,
    ) -> None:
        """Show/hide EmptyState widgets on prepare/execute pages (P3 / #166)."""
        doctor_done = bool(getattr(self, "_doctor_check_completed", False))
        has_project = bool(
            hasattr(self, "state") and self.state.get_game_root() is not None
        )
        running = bool(getattr(self, "_task_running", False)) or (
            hasattr(self, "kill_btn") and self.kill_btn.isEnabled()
        )
        has_workflow = self._workflow is not None or bool(
            getattr(self, "_writeback_manifest_path", "")
        )
        if running:
            resume_ok = False
        elif resume_available is not None:
            resume_ok = bool(resume_available[0])
        else:
            resume_ok, _ = self._resume_task_available()

        if hasattr(self, "doctor_empty_state"):
            show_doctor_empty = not doctor_done and not running
            stack = getattr(self, "doctor_page_stack", None)
            if stack is not None and hasattr(self, "doctor_empty_state"):
                # Mutual exclusion via stack — no VBox dual-stretch overlap.
                if show_doctor_empty:
                    stack.setCurrentWidget(self.doctor_empty_state)
                else:
                    # Summary page is index 0 (status + scroll).
                    stack.setCurrentIndex(0)
            else:
                # Fallback if stack is unavailable (older layout).
                self.doctor_empty_state.setVisible(show_doctor_empty)
                for attr in ("doctor_status_label",):
                    widget = getattr(self, attr, None)
                    if widget is not None:
                        widget.setVisible(not show_doctor_empty)
                doctor_scroll = getattr(self, "doctor_summary_scroll", None)
                if doctor_scroll is not None:
                    doctor_scroll.setVisible(not show_doctor_empty)

        if hasattr(self, "workflow_empty_state"):
            show_wf_empty = (
                not running
                and not has_workflow
                and not resume_ok
                and not bool(getattr(self, "_viewing_completed_manifest", False))
            )
            # Hide when workflow labels already show non-idle content.
            status = ""
            if hasattr(self, "workflow_status_label"):
                raw = self.workflow_status_label.property("status")
                status = str(raw or "")
            if status and status not in {"idle", "stale", ""}:
                show_wf_empty = False
            if has_project and status in {"idle", "stale"} and not resume_ok and not has_workflow:
                show_wf_empty = True
            if not has_project:
                show_wf_empty = True
            self.workflow_empty_state.setVisible(show_wf_empty)
            # Empty CTA shares the progress column with summary chrome. Hide the
            # chrome while empty so the action button is not height-crushed.
            for attr in (
                "workflow_status_label",
                "workflow_message_label",
                "workflow_facts_label",
            ):
                widget = getattr(self, attr, None)
                if widget is not None:
                    widget.setVisible(not show_wf_empty)
            if show_wf_empty:
                for attr in (
                    "workflow_progress_bar",
                    "view_last_completed_btn",
                    "hide_completed_view_btn",
                    "split_status_title",
                    "split_status_table",
                ):
                    widget = getattr(self, attr, None)
                    if widget is not None:
                        widget.setVisible(False)
                self._ensure_workflow_empty_cta_visible()
            else:
                if hasattr(self, "_apply_workflow_progress_ui"):
                    self._apply_workflow_progress_ui()
                if hasattr(self, "_update_completed_manifest_entry_ui"):
                    self._update_completed_manifest_entry_ui()

    def _ensure_workflow_empty_cta_visible(self) -> None:
        """Scroll outer/inner workbench scroll areas so the empty CTA is on-screen."""
        empty = getattr(self, "workflow_empty_state", None)
        if empty is None or not empty.isVisible():
            return
        target = getattr(empty, "_action_btn", None) or empty

        def _scroll_ancestors(widget) -> None:
            parent = widget.parentWidget()
            while parent is not None:
                if isinstance(parent, QScrollArea):
                    parent.ensureWidgetVisible(target, 16, 24)
                parent = parent.parentWidget()

        # Defer until after the current layout pass finishes.
        QTimer.singleShot(0, lambda: _scroll_ancestors(empty))

    def _restore_diagnostics_splitter_idle(self) -> None:
        """Return diagnostics splitter toward idle context:log balance (P3 / #166)."""
        if not hasattr(self, "diagnostics_splitter"):
            return
        if hasattr(self, "_splitter_anim") and self._splitter_anim.state() == self._splitter_anim.State.Running:
            self._splitter_anim.stop()
        total = max(sum(self.diagnostics_splitter.sizes()), 1)
        # Prefer fixed idle pixels when total is large enough; else 70/30 split.
        if total >= (_DIAGNOSTICS_IDLE_CONTEXT_PX + _DIAGNOSTICS_IDLE_LOG_PX):
            context = _DIAGNOSTICS_IDLE_CONTEXT_PX
            log = total - context
        else:
            context = int(total * 0.70)
            log = max(1, total - context)
        self.diagnostics_splitter.setSizes([context, log])

    def _should_generate_template_only(self) -> bool:
        spec = work_mode_spec(self._current_work_mode())
        if spec.mode not in {WorkMode.BATCH_TRANSLATION, WorkMode.SYNC_TRANSLATION}:
            return False
        return self._doctor_summary_mode == "can_generate_template"

    def _translate_button_label(self) -> str:
        spec = work_mode_spec(self._current_work_mode())
        if self._should_generate_template_only():
            return "生成翻译模板"
        return spec.start_button_label

    def _translation_requires_doctor_check(self, mode: WorkMode) -> bool:
        return mode in {WorkMode.BATCH_TRANSLATION, WorkMode.SYNC_TRANSLATION}

    def _doctor_allows_translate_action(self) -> bool:
        if not self._doctor_check_completed:
            return False
        if self._should_generate_template_only():
            return True
        return self._doctor_summary_status in {"ready", "warning"}

    def _translate_button_enabled(
        self,
        *,
        spec,
        bootstrap_ready: bool,
        running: bool,
    ) -> bool:
        if self._litellm_install_blocks_mode(spec.mode):
            return False
        if running or not spec.implemented or not bootstrap_ready:
            return False
        if (
            spec.mode == WorkMode.KEYWORD_EXTRACTION
            and self.workflow_status_label.property("status") == "waiting"
        ):
            return False
        if self._translation_requires_doctor_check(spec.mode):
            return self._doctor_allows_translate_action()
        return True

    def _update_translate_button_label(self) -> None:
        self.translate_btn.setText(self._translate_button_label())
        self._sync_sync_translation_page_controls(label_only=True)

    def _on_batch_translation_page_action(self, action: str) -> None:
        """Route a P5 page-local action to the existing batch coordinator."""
        callbacks = {
            "start": self._on_start_translation,
            "resume": self._on_resume_translation,
            "stop": self._on_kill,
            "split_submit": self._on_submit_remaining_split_packages,

            "probe": self._on_run_probe,
            "split": self._on_run_split,
        }
        callback = callbacks.get(action)
        if callback is not None:
            callback()

    def _batch_translation_action_state(
        self,
        *,
        running: bool | None = None,
    ) -> BatchActionState:
        """Compose the single render state consumed by the real batch page."""
        if running is None:
            running = self._task_page_running_chrome()

        def state(source_name: str, *, visible: bool = True) -> tuple[bool, bool, str]:
            source = getattr(self, source_name)
            return (visible and not source.isHidden(), source.isEnabled(), source.text())

        stop_label = (
            self._task_stop_button_label()
            if running
            else (self.kill_btn.text() if hasattr(self, "kill_btn") else "停止")
        )
        return BatchActionState(
            running=running,
            controls={
                "start": (True, self.translate_btn.isEnabled(), self.translate_btn.text()),
                "resume": (True, self.resume_btn.isEnabled(), self.resume_btn.text()),
                "stop": (True, self.kill_btn.isEnabled(), stop_label),
                "split_submit": state("split_submit_btn"),
                "probe": (True, self.probe_btn.isEnabled(), self.probe_btn.text()),
                "split": (True, self.split_btn.isEnabled(), self.split_btn.text()),
            },
        )
    def _sync_batch_translation_page_controls(
        self,
        *,
        running: bool | None = None,
    ) -> None:
        """Render coordinator readiness on the real batch task page."""
        page = getattr(self, "batch_translation_page", None)
        if page is None or self._current_work_mode() != WorkMode.BATCH_TRANSLATION:
            return
        page.set_action_state(
            self._batch_translation_action_state(running=running)
        )
        self._refresh_active_workbench_page(work_mode_spec(WorkMode.BATCH_TRANSLATION))

    def _sync_keywords_page_controls(self, *, running: bool | None = None) -> None:
        """Mirror coordinator-owned keyword actions onto the real page."""
        page = getattr(self, "keywords_page", None)
        if page is None:
            return
        mode = self._current_work_mode()
        if mode not in {WorkMode.KEYWORD_EXTRACTION, WorkMode.SYNC_KEYWORD_EXTRACTION}:
            return
        if running is None:
            running = self._task_page_running_chrome()
        candidates_path = self._resolve_keyword_merge_candidates_path()
        glossary_path = self._resolve_keyword_merge_glossary_path()
        merge_ready, merge_message = keyword_merge_ready(
            candidates_path=candidates_path,
            glossary_path=glossary_path,
        )
        if merge_ready:
            result_hint = "关键词候选已就绪；可审核并合并到 glossary.json。"
        elif merge_message:
            result_hint = f"关键词结果：{merge_message}"
        else:
            result_hint = "提取完成后，可在此合并审核通过的术语候选。"
        page.set_task_running(running)
        if hasattr(page, "stop_btn"):
            page.stop_btn.setText(
                self._task_stop_button_label() if running else "停止"
            )
        page.set_controls(
            start_enabled=self.translate_btn.isEnabled(),
            resume_enabled=self.resume_btn.isEnabled(),
            resume_visible=work_mode_spec(mode).supports_resume,
            resume_label=self.resume_btn.text(),
            merge_enabled=merge_ready,
            merge_message=result_hint,
        )
        if workbench_nav_for_work_mode(mode) == WorkbenchNavItem.KEYWORDS:
            self._refresh_active_workbench_page(work_mode_spec(mode))

    def _sync_revision_page_controls(self, *, running: bool | None = None) -> None:
        """Mirror coordinator-owned revision actions onto the real page."""
        page = getattr(self, "revision_page", None)
        if page is None:
            return
        mode = self._current_work_mode()
        if mode not in {WorkMode.REVISION, WorkMode.SYNC_REVISION}:
            return
        if running is None:
            running = self._task_page_running_chrome()
        summary = self._current_writeback_summary()
        if summary.can_apply:
            result_message = summary.message or "订正预览已通过，可安全写回。"
        elif summary.message:
            result_message = f"订正结果：{summary.message}"
        else:
            result_message = "生成预览后，可在此确认订正结果并安全写回。"
        page.set_task_running(running)
        if hasattr(page, "stop_btn"):
            page.stop_btn.setText(
                self._task_stop_button_label() if running else "停止"
            )
        page.set_controls(
            start_enabled=self.translate_btn.isEnabled(),
            resume_enabled=self.resume_btn.isEnabled(),
            resume_visible=work_mode_spec(mode).supports_resume,
            resume_label=self.resume_btn.text(),
            writeback_enabled=summary.can_apply,
            result_message=result_message,
        )
        if workbench_nav_for_work_mode(mode) == WorkbenchNavItem.REVISION:
            self._refresh_active_workbench_page(work_mode_spec(mode))

    def _sync_sync_translation_page_controls(
        self,
        *,
        running: bool | None = None,
        label_only: bool = False,
    ) -> None:
        """Keep page-local Start/Stop aligned with the legacy translate/kill buttons."""
        sync_page = getattr(self, "sync_translation_page", None)
        if sync_page is None:
            return
        sync_page.set_start_label(self._translate_button_label())
        if label_only:
            return
        if running is None:
            running = self._task_page_running_chrome()
        sync_page.set_task_running(running)
        if hasattr(sync_page, "stop_btn"):
            sync_page.stop_btn.setText(
                self._task_stop_button_label() if running else "停止"
            )
        sync_page.set_start_enabled(self.translate_btn.isEnabled())

    def _apply_work_mode_ui(
        self,
        *,
        refresh_manifest_writeback: bool = False,
        refresh_diagnostics: bool = False,
    ) -> None:
        spec = work_mode_spec(self._current_work_mode())
        self._update_timeline_steps(spec.mode)
        self.timeline.setVisible(False)
        self._sync_task_selectors_from_work_mode()
        self.translate_group_label.setText(spec.task_group_label)
        self._update_translate_button_label()
        if spec.resume_button_label:
            self.resume_btn.setText(spec.resume_button_label)
        self._update_resume_btn_text()
        self.resume_btn.setVisible(spec.supports_resume)
        # Task pages own progress and only expose writeback when the mode supports it.
        self.workbench_status_tabs.setTabText(0, "环境检查")
        self.workbench_status_tabs.setTabText(1, spec.progress_tab_label)
        self.workbench_status_tabs.setTabText(2, spec.writeback_tab_label)
        self._sync_workbench_status_surface()
        if spec.implemented:
            if spec.is_bootstrap and not self._bootstrap_task_ready(spec):
                hint = bootstrap_disabled_message(spec.bootstrap_kind)
            else:
                hint = spec.idle_workflow_message
        else:
            hint = spec.not_implemented_message
        self.work_mode_hint_label.setText(hint)
        self._refresh_active_workbench_page(spec)

        # Prefer session-bound writeback path when restoring a page.
        session_manifest = str(self._writeback_manifest_path or "").strip() or None
        pending_workflow_ui = getattr(self, "_pending_restore_workflow_ui", None)
        pending_writeback = getattr(self, "_pending_restore_writeback_summary", None)
        self._pending_restore_workflow_ui = None
        self._pending_restore_writeback_summary = None
        restored_workflow_ui = False
        restored_writeback_ui = False

        def _try_restore_workflow_snapshot() -> bool:
            return self._apply_session_workflow_ui(pending_workflow_ui)

        def _try_restore_writeback_snapshot() -> bool:
            if pending_writeback is None:
                return False
            self._set_writeback_summary(pending_writeback)
            return True

        if refresh_manifest_writeback or refresh_diagnostics:
            if session_manifest:
                try:
                    session_data = self.state.load_resume_manifest(
                        session_manifest,
                        work_mode=spec.mode,
                    )
                except ValueError:
                    session_data = None
                if session_data is not None:
                    self._refresh_workflow_from_latest_manifest(
                        latest_manifest=session_manifest,
                        manifest=session_data,
                    )
                    if refresh_manifest_writeback:
                        self._refresh_writeback_from_latest_manifest(
                            latest_manifest=session_manifest,
                            manifest=session_data,
                        )
                else:
                    # Path remembered but unloadable — keep UI snapshot if any.
                    restored_workflow_ui = _try_restore_workflow_snapshot()
                    if not restored_workflow_ui:
                        self._refresh_workflow_from_latest_manifest()
                    if refresh_manifest_writeback:
                        restored_writeback_ui = _try_restore_writeback_snapshot()
                        if not restored_writeback_ui:
                            self._refresh_writeback_from_latest_manifest()
                if refresh_diagnostics:
                    self._refresh_diagnostics_context(force=True)
            else:
                self._refresh_manifest_derived_ui(
                    refresh_writeback=refresh_manifest_writeback,
                    refresh_diagnostics=refresh_diagnostics,
                )
                # Non-resume modes have no disk manifest; re-apply progress snapshot.
                if not restored_workflow_ui:
                    restored_workflow_ui = _try_restore_workflow_snapshot()
                if refresh_manifest_writeback and not restored_writeback_ui:
                    restored_writeback_ui = _try_restore_writeback_snapshot()
        else:
            if session_manifest and spec.supports_resume:
                self._refresh_workflow_from_latest_manifest(latest_manifest=session_manifest)
            else:
                restored_workflow_ui = _try_restore_workflow_snapshot()
                if not restored_workflow_ui:
                    self._refresh_workflow_from_latest_manifest()
            # Soft switch (no disk refresh): restore frozen writeback UI when present.
            if pending_writeback is not None:
                _try_restore_writeback_snapshot()

        # Restore status-tab index when returning to a page.
        pending_stage = getattr(self, "_pending_restore_stage_index", None)
        self._pending_restore_stage_index = None
        if (
            pending_stage is not None
            and hasattr(self, "workbench_status_tabs")
            and 0 <= int(pending_stage) < self.workbench_status_tabs.count()
        ):
            blocked = self.workbench_status_tabs.blockSignals(True)
            self.workbench_status_tabs.setCurrentIndex(int(pending_stage))
            self.workbench_status_tabs.blockSignals(blocked)
        self._sync_workbench_status_surface()
        self._sync_workbench_status_chrome()

        # Keep session bag aligned after UI refresh mutates active fields.
        self._mode_sessions[spec.mode] = self._capture_mode_session()
        # Global job gate (disables start/resume) vs page-owned stop chrome.
        running = bool(getattr(self, "_task_running", False)) or (
            hasattr(self, "kill_btn") and self.kill_btn.isEnabled()
        )
        page_running = self._task_page_running_chrome()
        bootstrap_ready = self._bootstrap_task_ready(spec)
        self.translate_btn.setEnabled(
            self._translate_button_enabled(
                spec=spec,
                bootstrap_ready=bootstrap_ready,
                running=running,
            )
        )
        if spec.mode == WorkMode.SYNC_TRANSLATION:
            self._sync_sync_translation_page_controls(running=page_running)
        resume_available = (
            (False, "任务运行中。") if running else self._resume_task_available()
        )
        self._update_resume_btn_enabled(
            running=running,
            resume_available=resume_available,
        )
        self._sync_batch_translation_page_controls(running=page_running)
        self._sync_keywords_page_controls(running=page_running)
        self._sync_revision_page_controls(running=page_running)
        self._update_split_submit_btn(running=running)
        self._sync_task_shortcuts()
        # Re-apply page chrome after enable flags so context cards stay correct.
        self._refresh_active_workbench_page(spec)
        self._sync_workbench_empty_states(resume_available=resume_available)

    def _update_timeline_steps(self, mode: WorkMode) -> None:
        from .work_modes import WorkMode
        if mode == WorkMode.BATCH_TRANSLATION:
            steps = [
                ("build", "准备"),
                ("submit", "提交"),
                ("status", "云端执行"),
                ("download", "下载结果"),
                ("check", "安全校验"),
            ]
        elif mode == WorkMode.KEYWORD_EXTRACTION:
            steps = [
                ("build-keywords", "准备"),
                ("submit", "提交"),
                ("status", "云端执行"),
                ("download", "下载结果"),
                ("export-keywords", "导出报告"),
            ]
        elif mode == WorkMode.REVISION:
            steps = [
                ("build-revisions", "准备"),
                ("submit", "提交"),
                ("status", "云端执行"),
                ("download", "下载结果"),
                ("preview-revisions", "校验预览"),
            ]
        elif mode in {
            WorkMode.SYNC_TRANSLATION,
            WorkMode.SYNC_KEYWORD_EXTRACTION,
            WorkMode.SYNC_REVISION,
            WorkMode.BOOTSTRAP_RAG,
            WorkMode.BOOTSTRAP_SOURCE_INDEX,
        }:
            steps = [
                ("run", "同步运行"),
            ]
        else:
            steps = []
        self.timeline.set_steps(steps)
        self.timeline.set_current_step(None, "idle")

    def _sync_timeline_from_workflow_status(
        self,
        status: str,
        workflow: Any | None = None,
        *,
        step_key: str | None = None,
    ) -> None:
        if not hasattr(self, "timeline"):
            return
        if not self.timeline.steps:
            self.timeline.set_current_step(None, "idle")
            self.timeline.setVisible(False)
            return
        if step_key is None and workflow is not None:
            current_step = workflow.current_step()
            step_key = current_step.key if current_step else None
        if status == "idle":
            self.timeline.set_current_step(None, "idle")
            self.timeline.setVisible(False)
            return
        if status == "done":
            self.timeline.set_current_step(None, "done")
        else:
            self.timeline.set_current_step(step_key, status)
        self.timeline.setVisible(True)

    def _invalidate_manifest_caches(self, manifest_path: str | Path | None = None) -> None:
        state = getattr(self, "state", None)
        if state is None:
            return
        invalidate_history = getattr(state, "invalidate_manifest_history_cache", None)
        invalidate_file = getattr(state, "invalidate_manifest_file_cache", None)
        if callable(invalidate_history):
            invalidate_history()
        if callable(invalidate_file):
            invalidate_file(manifest_path)

    def _clear_completed_manifest_snapshot(self) -> None:
        self._completed_manifest_snapshot = None
        self._viewing_completed_manifest = False
        self._update_completed_manifest_entry_ui()

    def _update_completed_manifest_entry_ui(self) -> None:
        if not hasattr(self, "view_last_completed_btn"):
            return
        running = self.kill_btn.isEnabled()
        if self._viewing_completed_manifest:
            self.view_last_completed_btn.setVisible(False)
            self.hide_completed_view_btn.setVisible(not running)
            return
        has_snapshot = self._completed_manifest_snapshot is not None
        self.view_last_completed_btn.setVisible(has_snapshot and not running)
        self.hide_completed_view_btn.setVisible(False)

    def _apply_manifest_workflow_display(
        self,
        display: ManifestWorkflowDisplay,
        split_entries: list[SplitManifestEntry],
    ) -> None:
        self._set_workflow_summary(
            display.status,
            display.heading,
            display.message,
            list(display.facts),
        )
        self._sync_timeline_from_workflow_status(
            display.status,
            display.workflow,
            step_key=display.timeline_step_key,
        )
        if self._split_entries_unchanged(split_entries):
            self._split_status_selected_manifest_path = display.selected_manifest_path
            self._update_split_status_selection_ui(display.selected_manifest_path)
        else:
            self._render_split_status_entries(
                split_entries,
                selected_manifest_path=display.selected_manifest_path,
            )

    def _show_completed_manifest_snapshot(self) -> None:
        snapshot = self._completed_manifest_snapshot
        if not isinstance(snapshot, dict):
            return
        display = snapshot.get("display")
        split_entries = snapshot.get("split_entries")
        if not isinstance(display, ManifestWorkflowDisplay):
            return
        entries = split_entries if isinstance(split_entries, list) else []
        self._viewing_completed_manifest = True
        self._apply_manifest_workflow_display(display, entries)
        self._update_completed_manifest_entry_ui()

    def _on_view_last_completed_task(self) -> None:
        self._show_completed_manifest_snapshot()
        self._focus_workbench_status_tab(1)

    def _on_hide_completed_manifest_view(self) -> None:
        self._viewing_completed_manifest = False
        self._refresh_workflow_idle_summary()

    def _refresh_workflow_idle_summary(self) -> None:
        if self.kill_btn.isEnabled():
            return
        if hasattr(self, "timeline"):
            self.timeline.set_current_step(None, "idle")
            self.timeline.setVisible(False)
        self._render_split_status_entries([])
        spec = work_mode_spec(self._current_work_mode())
        facts: list[str] = []
        if self._completed_manifest_snapshot and not self._viewing_completed_manifest:
            manifest_path = self._completed_manifest_snapshot.get("manifest_path")
            if isinstance(manifest_path, str) and manifest_path.strip():
                facts.append(completed_manifest_entry_fact(spec, manifest_path))
        self._set_workflow_summary(
            "idle",
            spec.idle_workflow_heading,
            spec.idle_workflow_message,
            facts,
        )
        self._update_completed_manifest_entry_ui()

    def _refresh_workflow_from_latest_manifest(
        self,
        *,
        latest_manifest: Path | str | None = None,
        manifest: dict[str, object] | None = None,
        split_entries: list[SplitManifestEntry] | None = None,
        force_expand: bool = False,
    ) -> None:
        if self.kill_btn.isEnabled():
            return
        spec = work_mode_spec(self._current_work_mode())
        if not spec.implemented or not spec.supports_resume:
            self._refresh_workflow_idle_summary()
            return

        game_root = self.state.get_game_root()
        if game_root is None:
            self._refresh_workflow_idle_summary()
            return

        if latest_manifest is None:
            latest_manifest = self.state.get_latest_manifest_path_for_mode(
                game_root,
                spec.mode,
            )
        if latest_manifest is None:
            self._refresh_workflow_idle_summary()
            return

        if manifest is None:
            try:
                manifest = self.state.load_resume_manifest(
                    latest_manifest,
                    work_mode=spec.mode,
                )
            except ValueError:
                self._refresh_workflow_idle_summary()
                return

        if self._viewing_completed_manifest and not force_expand:
            snapshot = self._completed_manifest_snapshot
            if isinstance(snapshot, dict) and snapshot.get("manifest_path") == str(latest_manifest):
                self._show_completed_manifest_snapshot()
                return

        if split_entries is None:
            split_entries = self._split_entries_for_manifest(str(latest_manifest), manifest)
        split_fact_lines = summarize_split_entries(split_entries)
        display = build_manifest_workflow_display(
            spec,
            str(latest_manifest),
            manifest,
            extra_facts=split_fact_lines,
        )

        if display.archive_when_idle and not force_expand:
            self._completed_manifest_snapshot = {
                "manifest_path": str(latest_manifest),
                "display": display,
                "split_entries": list(split_entries),
            }
            self._viewing_completed_manifest = False
            self._refresh_workflow_idle_summary()
            return

        self._completed_manifest_snapshot = None
        self._viewing_completed_manifest = False
        self._apply_manifest_workflow_display(display, split_entries)
        self._update_completed_manifest_entry_ui()

    def _clear_bootstrap_progress_ui(self) -> None:
        self._bootstrap_progress = None
        self._bootstrap_progress_tracker = None
        self._workflow_progress = None
        self._workflow_progress_base_facts = []
        if hasattr(self, "_bootstrap_progress_eta_timer"):
            self._bootstrap_progress_eta_timer.stop()
        if hasattr(self, "workflow_progress_bar"):
            self.workflow_progress_bar.setVisible(False)

    def _clear_workflow_progress_ui(self) -> None:
        self._workflow_progress = None
        self._workflow_progress_base_facts = []
        if hasattr(self, "workflow_progress_bar"):
            self.workflow_progress_bar.setVisible(False)

    def _apply_workflow_progress_ui(self) -> None:
        if not hasattr(self, "workflow_progress_bar"):
            return

        state = self._workflow_progress
        if state is None or not state.visible:
            self.workflow_progress_bar.setVisible(False)
            return

        if state.indeterminate or state.total <= 0:
            self.workflow_progress_bar.setRange(0, 0)
        else:
            total = max(state.total, 1)
            current = min(max(state.current, 0), total)
            self.workflow_progress_bar.setRange(0, total)
            self.workflow_progress_bar.setValue(current)
        self.workflow_progress_bar.setFormat(state.label or "正在处理…")
        self.workflow_progress_bar.setVisible(True)

        if hasattr(self, "workflow_facts_label"):
            facts = list(self._workflow_progress_base_facts)
            for fact in state.facts:
                if fact and fact not in facts:
                    facts.append(fact)
            self.workflow_facts_label.setText("\n".join(facts))

    def _workflow_progress_kind_for_step(self, step: Any) -> str:
        spec = work_mode_spec(self._current_work_mode())
        if spec.mode == WorkMode.BATCH_TRANSLATION and step.key == "build":
            return "source_index_build"
        if spec.mode == WorkMode.SYNC_TRANSLATION and step.key in {"preview", "apply"}:
            return "sync_translation"
        if spec.mode in {WorkMode.SYNC_KEYWORD_EXTRACTION, WorkMode.SYNC_REVISION}:
            return "sync_requests"
        return ""

    def _on_bootstrap_progress_eta_tick(self) -> None:
        if (
            self._active_command == "bootstrap_source_index"
            and self._bootstrap_progress is not None
        ):
            self._apply_bootstrap_progress_ui()

    def _apply_bootstrap_progress_ui(self) -> None:
        if not hasattr(self, "workflow_progress_bar"):
            return

        state = self._bootstrap_progress
        if self._active_command == "bootstrap_rag" and self.kill_btn.isEnabled():
            workflow_state = getattr(self, "_workflow_progress", None)
            if workflow_state is not None and workflow_state.visible:
                self._apply_workflow_progress_ui()
                return
            self.workflow_progress_bar.setRange(0, 0)
            self.workflow_progress_bar.setFormat("正在处理…")
            self.workflow_progress_bar.setVisible(True)
            return

        if state is None or state.kind != "source_index":
            self.workflow_progress_bar.setVisible(False)
            return

        tracker = self._bootstrap_progress_tracker
        if tracker is not None:
            tracker.observe(state)

        if state.total_segments <= 0 and state.stored_segments <= 0:
            self.workflow_progress_bar.setRange(0, 0)
            self.workflow_progress_bar.setFormat("正在扫描原文…")
            self.workflow_progress_bar.setVisible(True)
        else:
            total = max(state.total_segments, 1)
            stored = min(max(state.stored_segments, 0), total)
            remaining_seconds = (
                tracker.estimate_remaining_seconds(state)
                if tracker is not None
                else None
            )
            self.workflow_progress_bar.setRange(0, total)
            self.workflow_progress_bar.setValue(stored)
            self.workflow_progress_bar.setFormat(
                format_bootstrap_progress_bar_label(state, remaining_seconds)
            )
            self.workflow_progress_bar.setVisible(True)

        facts = format_bootstrap_progress_facts(state)
        if facts:
            self.workflow_facts_label.setText("\n".join(facts))

    def _set_workflow_from_bootstrap_summary(self, summary: BootstrapSummary) -> None:
        if summary.status == "running":
            self._sync_timeline_from_workflow_status("running", step_key="run")
        elif summary.status in {"ready", "warning"}:
            self._sync_timeline_from_workflow_status("done", step_key="run")
            self._clear_bootstrap_progress_ui()
        elif summary.status == "failed":
            self._sync_timeline_from_workflow_status("failed", step_key="run")
            self._clear_bootstrap_progress_ui()
        elif summary.status in {"idle", "stale"}:
            self._sync_timeline_from_workflow_status("idle", step_key="run")
            self._clear_bootstrap_progress_ui()
        if hasattr(self, "timeline"):
            self.timeline.setVisible(False)
        self._set_workflow_summary(
            summary.status,
            summary.heading,
            summary.message,
            summary.facts,
        )

    def _refresh_writeback_from_latest_manifest(
        self,
        *,
        latest_manifest: Path | str | None = None,
        manifest: dict[str, object] | None = None,
    ) -> None:
        spec = work_mode_spec(self._current_work_mode())
        game_root = self.state.get_game_root()
        if spec.mode == WorkMode.REVISION:
            if latest_manifest is None and game_root is not None:
                latest_manifest = self.state.get_latest_manifest_path_for_mode(
                    game_root,
                    spec.mode,
                )
            if latest_manifest is None:
                self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
                return
            if manifest is None:
                try:
                    manifest = self.state.load_resume_manifest(
                        latest_manifest,
                        work_mode=spec.mode,
                    )
                except ValueError:
                    self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
                    return
            summary = summarize_revision_writeback_from_manifest(manifest)
            if summary is None:
                self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
                return
            self._set_writeback_summary(summary)
            return
        if spec.mode == WorkMode.KEYWORD_EXTRACTION:
            if latest_manifest is None and game_root is not None:
                latest_manifest = self.state.get_latest_manifest_path_for_mode(
                    game_root,
                    spec.mode,
                )
            if latest_manifest is None:
                self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
                return
            if manifest is None:
                try:
                    manifest = self.state.load_resume_manifest(
                        latest_manifest,
                        work_mode=spec.mode,
                    )
                except ValueError:
                    self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
                    return
            summary = summarize_keyword_result_from_manifest(manifest)
            if summary is None:
                self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
                return
            self._set_writeback_summary(summary)
            return

        if not spec.supports_translation_writeback:
            self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
            return

        if latest_manifest is None and game_root is not None:
            latest_manifest = self.state.get_latest_manifest_path_for_mode(
                game_root,
                spec.mode,
            )
        if latest_manifest is None:
            self._set_writeback_summary(idle_writeback_summary())
            return
        if manifest is None:
            try:
                manifest = self.state.load_resume_manifest(
                    latest_manifest,
                    work_mode=spec.mode,
                )
            except ValueError:
                self._set_writeback_summary(idle_writeback_summary())
                return

        summary = summarize_manifest_writeback(manifest)
        if summary is None:
            self._set_writeback_summary(idle_writeback_summary())
            return
        self._set_writeback_summary(summary)
        if not getattr(self, "_split_status_entries", []):
            self._refresh_split_status_ui(
                manifest_path=str(latest_manifest),
                manifest=manifest,
            )

    # --- UI actions ---

    def _on_global_switch_project(self) -> None:
        """Open settings workspace — same registry switch path, no third state machine."""
        if self._task_running:
            return
        self._on_go_to_workspace_for_project_switch()
        if self.state.get_workspace_root() is None:
            self.statusBar().showMessage(
                "请先选择工作区目录，再从项目列表切换项目。",
                6000,
            )
        else:
            self.statusBar().showMessage(
                "请在项目列表中选择项目并「切换到此项目」。",
                5000,
            )

    def _on_select_project(self):
        if self._task_running:
            return
        # Same dirty-settings leave-guard as registry switch (global bar contract).
        if not self._confirm_unsaved_config_before_registry_switch():
            return
        start_dir = str(self.state.get_game_root() or Path.home())
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择游戏目录（项目根目录或 work 目录；项目根目录下存在 work/ 时会自动切换）",
            start_dir,
        )
        if directory:
            self._switch_game_root(directory)

    def _invalidate_doctor_worker(self) -> None:
        worker = self._doctor_worker
        if worker is None:
            return
        try:
            worker.completed.disconnect(self._on_doctor_completed)
        except (RuntimeError, TypeError):
            pass
        if worker.isRunning():
            worker.requestInterruption()
            # Subprocess isolation needs a moment to terminate the child; keep
            # this short so cancel stays snappy if the worker is already done.
            worker.wait(1500)
        self._doctor_worker = None
        if self._active_command == "doctor":
            self._active_command = ""
            self._set_task_running(False)

    def _switch_game_root(self, directory: str) -> bool:
        if self._is_doctor_running():
            self._invalidate_doctor_worker()
        if self.runner.is_running():
            self.runner.kill()
            self._set_task_running(False)
        self._invalidate_doctor_worker()
        try:
            effective_root, adjusted = self.state.set_game_root(directory)
            if adjusted:
                self._show_game_root_redirect_notice(
                    Path(directory),
                    effective_root,
                )
            else:
                self._clear_game_root_redirect_notice()
        except ValueError as exc:
            QMessageBox.warning(self, "无法更新配置", str(exc))
            self._append_log(f"更新 translator_config.json 失败：{exc}")
            return False

        self._refresh_project_label()
        self._load_config_to_ui()
        self._active_command = ""
        self._doctor_output_lines = []
        # Project sessions are reset below, so their status-tab snapshot must
        # also start from the product default instead of leaking the old root.
        if getattr(self, "_shell_special_route", None) is not None:
            self._shell_task_status_index = _BATCH_STAGE_EXECUTE
        else:
            self._shell_task_status_index = None
            if hasattr(self, "workbench_status_tabs"):
                self._focus_workbench_status_tab(_BATCH_STAGE_EXECUTE)
        self._clear_all_mode_sessions()
        previous_mode = getattr(self, "_work_mode", WorkMode.BATCH_TRANSLATION)
        self._workbench_nav_item = workbench_nav_for_work_mode(previous_mode)
        self._reset_last_mode_by_nav()
        self._work_mode = default_work_mode_for_nav(self._workbench_nav_item)
        context_page = getattr(self, "context_library_page", None)
        if context_page is not None:
            context_page.reset_project()
        sync_page = getattr(self, "sync_translation_page", None)
        if sync_page is not None:
            sync_page.reset_project()
        keywords_page = getattr(self, "keywords_page", None)
        if keywords_page is not None:
            keywords_page.reset_project()
        revision_page = getattr(self, "revision_page", None)
        if revision_page is not None:
            revision_page.reset_project()
        self._workflow = None
        self._workflow_step_output_lines = []
        self._clear_completed_manifest_snapshot()
        self._doctor_check_completed = False
        self._doctor_summary_status = ""
        self._last_doctor_report = None
        self._last_doctor_report_game_root = ""
        self._set_doctor_summary(stale_summary())
        spec = work_mode_spec(self._current_work_mode())
        if spec.is_bootstrap:
            self._set_workflow_from_bootstrap_summary(stale_bootstrap_summary())
        else:
            self._set_workflow_summary(
                "stale",
                "项目已切换",
                "任务状态已清空；请先针对新项目重新检查。",
            )
        self._writeback_manifest_path = ""
        self._keyword_merge_candidates_path = ""
        sessions = getattr(self, "_mode_sessions", None)
        if not isinstance(sessions, dict):
            try:
                self._mode_sessions = {}
                sessions = self._mode_sessions
            except Exception:
                sessions = None
        if isinstance(sessions, dict):
            work_mode = getattr(self, "_work_mode", WorkMode.BATCH_TRANSLATION)
            sessions[work_mode] = self._capture_mode_session()
        if spec.supports_translation_writeback:
            self._set_writeback_summary(stale_writeback_summary())
        else:
            self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
        self._apply_work_mode_ui(refresh_manifest_writeback=False)
        self._refresh_diagnostics_context()
        self._invalidate_manifest_caches()
        self._append_log(f"项目目录已设置为：{directory}")
        return True

    def _handle_post_apply_registry_sync(self) -> None:
        manifest_path = self._writeback_manifest_path
        if not manifest_path:
            return
        workspace_root = self.state.get_workspace_root()
        if workspace_root is None:
            return
        result = handle_post_apply_registry_update(
            self,
            workspace_root=workspace_root,
            game_root=self.state.get_game_root(),
            manifest_path=manifest_path,
        )
        if result.message:
            self._append_log(result.message)

    def _on_workspace_changed(self, workspace_root: Path) -> None:
        """Persist explicit workspace selection from the project list panel."""
        panel = self.__dict__.get("_games_registry_panel")
        try:
            self.state.set_workspace_root(workspace_root)
        except ValueError as exc:
            # Revert panel UI if it already showed the unpersisted path.
            if panel is not None and hasattr(panel, "set_workspace_root"):
                panel.set_workspace_root(self.state.get_workspace_root())
            QMessageBox.warning(self, "无法更新工作区", str(exc))
            self._append_log(f"更新 workspace_root 失败：{exc}")
            return
        if panel is not None and hasattr(panel, "set_workspace_root"):
            panel.set_workspace_root(self.state.get_workspace_root())
        self._append_log(f"工作区已设置为：{workspace_root}")
        self.statusBar().showMessage(f"工作区：{workspace_root}", 5000)

    def _current_registry_doctor_report(self) -> dict | None:
        current_game_root = self.state.get_game_root()
        if (
            self._last_doctor_report is not None
            and current_game_root
            and self._last_doctor_report_game_root == current_game_root
        ):
            return self._last_doctor_report
        return None

    def _on_registry_switch_project(self, target: str) -> bool:
        if not self._confirm_unsaved_config_before_registry_switch():
            return False
        if not self._switch_game_root(target):
            return False
        panel = getattr(self, "_games_registry_panel", None)
        if panel is not None:
            # Highlight against the effective work root, not the Game_* folder.
            panel.set_current_game_root(self.state.get_game_root())
        # Stay on 项目列表; user can open 项目 when they need per-project knobs.
        self._show_settings_status("已切换项目", 5000)
        return True

    def _focus_settings_section(self, key: str) -> None:
        if hasattr(self, "_config_tab") and hasattr(self, "tab_widget"):
            settings_index = self.tab_widget.indexOf(self._config_tab)
            if settings_index >= 0:
                self.tab_widget.setCurrentIndex(settings_index)
        self._ensure_settings_page(key)
        row = getattr(self, "_settings_nav_rows", {}).get(key)
        nav = getattr(self, "settings_nav", None)
        if nav is not None and row is not None:
            nav.setCurrentRow(row)
            stack = getattr(self, "settings_stack", None)
            if stack is not None:
                stack.setCurrentIndex(row)
        self._sync_shell_nav_selection()

    def _on_go_to_workspace_for_project_switch(self) -> None:
        self._focus_settings_section("workspace")

    def _is_config_tab_active(self) -> bool:
        tab_widget = getattr(self, "tab_widget", None)
        config_tab = getattr(self, "_config_tab", None)
        if tab_widget is None or config_tab is None:
            return False
        return tab_widget.currentWidget() is config_tab

    def _activate_workspace_registry_section(self) -> None:
        # Materialize workspace page if the user navigated there.
        self._ensure_settings_page("workspace")
        panel = self.__dict__.get("_games_registry_panel")
        if panel is None:
            return
        panel.set_current_game_root(self.state.get_game_root())
        panel.activate_section()

    def _on_settings_nav_row_changed(self, row: int) -> None:
        if row < 0:
            return
        key = None
        for page_key, index in getattr(self, "_settings_nav_rows", {}).items():
            if index == row:
                key = page_key
                break
        if key is not None:
            self._ensure_settings_page(key)
        self.settings_stack.setCurrentIndex(row)
        self._sync_settings_action_bar_enabled(task_running=self._task_running, nav_row=row)
        if self._settings_nav_rows.get("workspace") == row and self._is_config_tab_active():
            self._activate_workspace_registry_section()
        self._sync_shell_nav_selection()

    def _sync_settings_action_bar_enabled(
        self,
        *,
        task_running: bool,
        nav_row: int | None = None,
    ) -> None:
        if nav_row is None:
            nav = getattr(self, "settings_nav", None)
            current_row = nav.currentRow() if nav is not None else -1
        else:
            current_row = nav_row
        on_workspace = self._settings_nav_rows.get("workspace") == current_row
        allow_config_save = not task_running and not on_workspace
        if hasattr(self, "save_config_btn"):
            self.save_config_btn.setEnabled(allow_config_save)
        if hasattr(self, "restore_defaults_btn"):
            self.restore_defaults_btn.setEnabled(allow_config_save)
        if hasattr(self, "reload_config_btn"):
            self.reload_config_btn.setEnabled(not task_running)

    def _refresh_api_status(self) -> None:
        # Do not materialize the keys page just to refresh a hidden label.
        label = self._settings_widget("api_status_label") if hasattr(self, "_settings_widget") else self.__dict__.get("api_status_label")
        if label is None:
            return
        # Load keys once — get_api_key_status reuses the same list.
        file_keys = self.state.load_api_keys()
        count, source = self.state.get_api_key_status(file_keys=file_keys)

        if count == 0:
            message = "当前状态：尚未配置有效 API Key。"
        elif source == "environment":
            message = f"当前状态：已通过环境变量配置 {count} 个有效 Key（只读）。"
            if file_keys:
                message += f" 本地文件另保存了 {len(file_keys)} 条 Key 记录。"
        else:
            masked = "、".join(mask_api_key(key) for key in file_keys if key.strip())
            message = f"当前状态：已保存 {len(file_keys)} 个 Key（有效 {count} 个）。"
            if masked:
                message += f"\n{masked}"

        label.setText(message)

    def _current_theme_preference_from_ui(self) -> str:
        # Do not materialize Settings · 外观 just to read a theme preference.
        combo = self._settings_widget("theme_combo")
        if combo is not None:
            try:
                preference = combo.currentData()
            except Exception:
                preference = None
            if isinstance(preference, str):
                return normalize_theme_preference(preference)
        return normalize_theme_preference(getattr(self, "_theme_preference", DEFAULT_THEME_PREFERENCE))

    def _advanced_settings_values_from_ui(self) -> dict[str, object]:
        widgets = getattr(self, "_advanced_setting_widgets", {})
        if not widgets:
            return {}
        values: dict[str, object] = {}
        for field in ADVANCED_SETTING_FIELDS:
            if field.key in _SETTINGS_WORKSPACE_MANAGED_KEYS:
                continue
            widget = widgets.get(field.key)
            if widget is None:
                continue
            if field.kind == "bool":
                values[field.key] = widget.isChecked()
            elif field.kind in {"int", "float"}:
                values[field.key] = widget.value()
            elif field.kind == "gemini_model_list" and isinstance(widget, QListWidget):
                values[field.key] = self._gemini_model_checklist_values(widget)
            elif hasattr(widget, "toPlainText"):
                values[field.key] = widget.toPlainText().strip()
            else:
                values[field.key] = widget.text().strip()
        return values

    def _load_advanced_settings_to_ui(self, values: dict[str, object]) -> None:
        widgets = getattr(self, "_advanced_setting_widgets", {})
        if not widgets:
            return
        for field in ADVANCED_SETTING_FIELDS:
            if field.key in _SETTINGS_WORKSPACE_MANAGED_KEYS:
                continue
            widget = widgets.get(field.key)
            if widget is None:
                continue
            value = values.get(field.key, field.default)
            if field.kind == "bool":
                widget.setChecked(bool(value))
            elif field.kind == "int":
                widget.setValue(int(value))
            elif field.kind == "float":
                widget.setValue(float(value))
            elif field.kind == "gemini_model_list" and isinstance(widget, QListWidget):
                try:
                    config = self.state.load_translator_config()
                except Exception:
                    config = {}
                self._refresh_gemini_model_checklist(
                    widget,
                    selected=value,
                    config=config if isinstance(config, dict) else {},
                )
            elif hasattr(widget, "setPlainText"):
                widget.setPlainText(self._format_advanced_setting_text(field, value))
            else:
                widget.setText(str(value))
        # Keep rotation pool checklist enabled only when model rotation is on.
        enabled_widget = widgets.get("model_rotation_enabled")
        if isinstance(enabled_widget, QCheckBox):
            self._on_model_rotation_enabled_toggled(enabled_widget.isChecked())

    def _format_advanced_setting_text(self, field: SettingField, value: object) -> str:
        if field.kind in {"list", "gemini_model_list"}:
            if isinstance(value, (list, tuple, set)):
                return "\n".join(str(item) for item in value)
            return str(value or "")
        if field.kind == "json":
            if value in (None, ""):
                return ""
            if isinstance(value, str):
                return value
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except TypeError:
                return str(value)
        return str(value or "")


    def _refresh_font_install_status(self) -> None:
        label = getattr(self, "font_install_status_label", None)
        button = getattr(self, "download_fonts_btn", None)
        if label is None:
            return
        installed = optional_fonts_installed()
        if installed:
            label.setText(
                f"已安装到 {user_fonts_dir()}。GUI 将使用 HarmonyOS Sans SC 和 "
                "LXGW WenKai Mono GB。"
            )
        else:
            label.setText(
                "尚未完整安装推荐字体；当前使用系统字体回退，不影响 GUI 功能。"
            )
        if button is not None and getattr(self, "_font_install_worker", None) is None:
            button.setEnabled(True)
            button.setText("重新下载推荐字体" if installed else "下载推荐字体")

    def _on_download_recommended_fonts(self) -> None:
        if getattr(self, "_font_install_worker", None) is not None:
            return
        reply = QMessageBox.question(
            self,
            "下载推荐字体",
            (
                "将从华为官方资源下载 HarmonyOS Sans，并从霞鹜文楷官方 Release "
                "下载 LXGW WenKai Mono GB。\n\n"
                "预计下载约 80 MB；下载后会校验 SHA-256，并保存到当前用户缓存目录。"
                "是否继续？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        button = getattr(self, "download_fonts_btn", None)
        if button is not None:
            button.setEnabled(False)
            button.setText("正在下载…")
        label = getattr(self, "font_install_status_label", None)
        if label is not None:
            label.setText("正在从字体发布者的官方来源下载并校验，请稍候…")
        progress = getattr(self, "font_install_progress", None)
        if progress is not None:
            progress.setVisible(True)

        worker = FontInstallWorker(self)
        worker.completed.connect(self._on_recommended_fonts_downloaded)
        self._font_install_worker = worker
        worker.start()

    def _on_recommended_fonts_downloaded(self, result: FontInstallResult) -> None:
        worker = getattr(self, "_font_install_worker", None)
        self._font_install_worker = None
        if worker is not None:
            worker.deleteLater()
        progress = getattr(self, "font_install_progress", None)
        if progress is not None:
            progress.setVisible(False)
        self._refresh_font_install_status()
        if result.ok:
            self._show_settings_status("推荐字体下载完成；重启 GUI 后生效。", 8000)
            QMessageBox.information(
                self,
                "字体安装完成",
                "HarmonyOS Sans SC 和 LXGW WenKai Mono GB 已安装。\n"
                "请重启 GUI 以加载推荐字体。",
            )
            return
        label = getattr(self, "font_install_status_label", None)
        if label is not None:
            label.setText(
                "字体下载失败，GUI 将继续使用系统字体回退。\n"
                f"错误：{result.error}"
            )
        self._show_settings_status("推荐字体下载失败；已继续使用系统字体。", 8000)

    def _show_advanced_setting_errors(self, errors: dict[str, str]) -> None:
        labels = getattr(self, "_advanced_setting_error_labels", {})
        for key, label in labels.items():
            label.setText(errors.get(key, ""))
        if errors:
            self._focus_advanced_setting(next(iter(errors)))

    def _clear_advanced_setting_errors(self) -> None:
        self._show_advanced_setting_errors({})

    def _focus_advanced_setting(self, key: str) -> None:
        if key in _SETTINGS_WORKSPACE_MANAGED_KEYS:
            self._focus_settings_section("workspace")
            widget = getattr(self, "_advanced_setting_widgets", {}).get(key)
            if widget is not None:
                widget.setFocus()
            return
        field = ADVANCED_SETTING_FIELD_BY_KEY.get(key)
        if key in CONTEXT_PRIMARY_SETTING_KEYS or (
            field is not None and field.category == CONTEXT_PRIMARY_SETTING_CATEGORY
        ):
            page_key = "context"
        elif field is not None and field.category in {"项目与资源", "准备流程"}:
            page_key = "project"
        else:
            page_key = "advanced"
        self._focus_settings_section(page_key)
        widget = getattr(self, "_advanced_setting_widgets", {}).get(key)
        if widget is not None:
            widget.setFocus()

    def _show_settings_status(self, message: str, timeout: int = 4000) -> None:
        try:
            self.statusBar().showMessage(message, timeout)
        except Exception:
            return

    def _current_config_ui_snapshot(self) -> dict[str, object]:
        thinking_val = self.batch_thinking_combo.currentData()
        thinking_level = thinking_val if isinstance(thinking_val, str) else ""
        snapshot: dict[str, object] = {
            "rag_enabled": self.rag_enabled_cb.isChecked(),
            "source_index_enabled": self.source_index_enabled_cb.isChecked(),
            "bootstrap_on_build": self.bootstrap_on_build_cb.isChecked(),
            "context_storage_location": "game" if self.context_storage_game_cb.isChecked() else "tool",
            "sync_backend": self._selected_sync_backend(),
            "sync_model": self.sync_model_combo.currentText().strip(),
            "litellm_model": self._litellm_model_text(),
            "batch_model": self.batch_model_combo.currentText().strip(),
            "sync_embedding_model": self.sync_embedding_combo.currentText().strip(),
            "batch_embedding_model": self.batch_embedding_combo.currentText().strip(),
            "batch_thinking_level": thinking_level,
            "theme": self._current_theme_preference_from_ui(),
        }
        snapshot.update(self._advanced_settings_values_from_ui())
        return snapshot

    def _config_tab_has_unsaved_changes(self) -> bool:
        if getattr(self, "_loading_config_to_ui", False):
            return False
        # Cold start leaves an empty snapshot until settings pages materialize
        # and load config into widgets — empty means "nothing editable yet".
        if not getattr(self, "_config_ui_saved_snapshot", None):
            return False
        return self._current_config_ui_snapshot() != self._config_ui_saved_snapshot

    def _confirm_unsaved_config_before_workflow(self) -> bool:
        if not self._config_tab_has_unsaved_changes():
            return True

        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("设置尚未保存")
        message.setText("设置页有未保存的更改。")
        message.setInformativeText(
            "当前任务会读取已保存的 translator_config.json；"
            "未保存的更改不会生效。"
        )
        save_btn = message.addButton("保存并继续", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = message.addButton("不保存继续", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = message.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        message.setDefaultButton(save_btn)
        message.exec()
        clicked = message.clickedButton()

        if clicked is save_btn:
            return self._on_save_config()
        if clicked is discard_btn:
            return True
        return False

    def _confirm_unsaved_config_before_registry_switch(self) -> bool:
        if not self._config_tab_has_unsaved_changes():
            return True

        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("设置尚未保存")
        message.setText("设置页有未保存的更改。")
        message.setInformativeText(
            "切换工作区项目会重新加载设置，未保存的更改将丢失。"
        )
        save_btn = message.addButton("保存并切换", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = message.addButton("不保存切换", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = message.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        message.setDefaultButton(save_btn)
        message.exec()
        clicked = message.clickedButton()

        if clicked is save_btn:
            return self._on_save_config()
        if clicked is discard_btn:
            return True
        return False

    def _confirm_leave_config_tab(self, previous_index: int) -> bool:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("设置尚未保存")
        message.setText("设置页有未保存的更改。")
        message.setInformativeText("离开前可以保存设置，或留在设置页继续检查。")
        save_btn = message.addButton("保存并离开", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = message.addButton("不保存离开", QMessageBox.ButtonRole.DestructiveRole)
        stay_btn = message.addButton("留在设置页", QMessageBox.ButtonRole.RejectRole)
        message.setDefaultButton(save_btn)
        message.exec()
        clicked = message.clickedButton()

        if clicked is save_btn:
            if self._on_save_config():
                return True
            clicked = stay_btn
        elif clicked is discard_btn:
            if _SETTINGS_CONFIG_PAGE_KEYS & getattr(self, "_settings_pages_built", set()):
                self._load_config_to_ui()
            return True

        self._handling_config_tab_leave = True
        try:
            self.tab_widget.setCurrentIndex(previous_index)
        finally:
            self._handling_config_tab_leave = False
        return False

    def _on_tab_changed(self, index: int) -> None:
        if self._handling_config_tab_leave:
            self._last_main_tab_index = index
            return

        previous_index = self._last_main_tab_index
        previous_widget = self.tab_widget.widget(previous_index) if previous_index >= 0 else None
        current_widget = self.tab_widget.widget(index)
        if (
            previous_widget is self._config_tab
            and current_widget is not self._config_tab
            and self._config_tab_has_unsaved_changes()
        ):
            if not self._confirm_leave_config_tab(previous_index):
                self._sync_shell_nav_selection()
                return

        # Commit the new tab index immediately, then defer heavy enter work so Qt
        # can paint the switched tab first (settings / diagnostics enter path).
        self._last_main_tab_index = index
        self._sync_shell_nav_selection()
        self._schedule_main_tab_enter_effects(current_widget)

    def _schedule_main_tab_enter_effects(self, widget: QWidget | None) -> None:
        self._main_tab_enter_generation = int(getattr(self, "_main_tab_enter_generation", 0)) + 1
        generation = self._main_tab_enter_generation
        # Defer only for a fully constructed window with a live Qt event loop.
        # Partial MainWindow.__new__ tests (and headless runs) execute synchronously.
        can_defer = False
        try:
            can_defer = (
                QApplication.instance() is not None
                and self.thread() is not None
            )
        except RuntimeError:
            can_defer = False
        if not can_defer:
            self._apply_main_tab_enter_effects(widget, generation)
            return
        QTimer.singleShot(
            0,
            lambda w=widget, g=generation: self._apply_main_tab_enter_effects(w, g),
        )

    def _apply_main_tab_enter_effects(
        self,
        widget: QWidget | None,
        generation: int,
    ) -> None:
        if generation != getattr(self, "_main_tab_enter_generation", 0):
            return
        if self.tab_widget.widget(self._last_main_tab_index) is not widget:
            return

        if widget is self._config_tab:
            nav = getattr(self, "settings_nav", None)
            row = nav.currentRow() if nav is not None else 0
            key = None
            for page_key, index in getattr(self, "_settings_nav_rows", {}).items():
                if index == row:
                    key = page_key
                    break
            if key is not None:
                self._ensure_settings_page(key)
            self._refresh_api_status()
            if (
                nav is not None
                and self._settings_nav_rows.get("workspace") == nav.currentRow()
            ):
                self._activate_workspace_registry_section()
            return

        if widget is getattr(self, "_diagnostics_tab", None):
            self._refresh_diagnostics_context()


    def _on_reload_config(self) -> None:
        self._ensure_settings_pages_for_config()
        self._load_config_to_ui()
        self._refresh_api_status()
        self._show_settings_status("已重新加载已保存设置。")

    def _on_restore_recommended_config(self) -> None:
        self._ensure_settings_pages_for_config()
        values = BASIC_RECOMMENDED_VALUES
        self.rag_enabled_cb.setChecked(bool(values["rag_enabled"]))
        self.source_index_enabled_cb.setChecked(bool(values["source_index_enabled"]))
        self.bootstrap_on_build_cb.setChecked(bool(values["bootstrap_on_build"]))
        self.context_storage_game_cb.setChecked(values["context_storage_location"] == "game")
        backend_combo = getattr(self, "sync_backend_combo", None)
        backend_idx = backend_combo.findData("gemini") if backend_combo is not None else -1
        if backend_combo is not None:
            backend_combo.setCurrentIndex(backend_idx)
        self._set_combo_value(self.sync_model_combo, values["sync_model"])
        litellm_combo = getattr(self, "litellm_model_combo", None)
        if litellm_combo is not None:
            self._set_combo_value(litellm_combo, "")
        self._on_sync_backend_changed(backend_idx)
        self._set_combo_value(self.batch_model_combo, values["batch_model"])
        self._set_combo_value(self.sync_embedding_combo, values["sync_embedding_model"])
        self._set_combo_value(self.batch_embedding_combo, values["batch_embedding_model"])
        self._set_batch_thinking_value(str(values["batch_thinking_level"]))
        self._set_theme_combo_value(str(values["theme"]))
        self._set_theme_preference(str(values["theme"]), persist=False)
        advanced_defaults = recommended_advanced_settings()
        state = getattr(self, "state", None)
        get_game_root = getattr(state, "get_game_root", None)
        current_game_root = get_game_root() if callable(get_game_root) else None
        if current_game_root:
            advanced_defaults["game_root"] = str(current_game_root)
        self._load_advanced_settings_to_ui(advanced_defaults)
        self._clear_advanced_setting_errors()
        self._batch_thinking_user_changed = True
        self._show_settings_status("已恢复推荐值，保存后生效。")


    def _on_manage_api_keys(self):
        env_count, env_source = self.state.get_api_key_status()
        env_key_count = env_count if env_source == "environment" else 0

        dialog = ApiKeyDialog(
            self,
            keys=self.state.load_api_keys(),
            env_key_count=env_key_count,
        )
        if dialog.exec() != ApiKeyDialog.DialogCode.Accepted:
            return

        new_keys = dialog.result_keys()
        try:
            self.state.save_api_keys(new_keys)
        except ValueError as exc:
            QMessageBox.warning(self, "无法更新 API Key", str(exc))
            self._append_log(f"更新 api_keys.json 失败：{exc}")
            return

        self._refresh_api_status()
        self._append_log(f"API Key 已保存（当前数量：{len(new_keys)}）。")

    def _format_game_root_redirect_notice(self, original: Path, effective: Path) -> str:
        return (
            f"提示：检测到 work 目录，已从 {original} 自动切换到 {effective}。"
        )

    def _show_game_root_redirect_notice(self, original: Path, effective: Path) -> None:
        notice = self._format_game_root_redirect_notice(original, effective)
        self.project_redirect_label.setText(notice)
        self.project_redirect_label.setVisible(True)
        self.statusBar().showMessage(
            f"已自动切换到 work 目录：{effective}",
            8000,
        )

    def _clear_game_root_redirect_notice(self) -> None:
        self.project_redirect_label.setText("")
        self.project_redirect_label.setVisible(False)

    def _show_pending_game_root_redirect_notice(self) -> None:
        original = self.state.take_game_root_redirect_from()
        effective = self.state.get_game_root()
        if original is None or effective is None:
            self._clear_game_root_redirect_notice()
            return
        self._show_game_root_redirect_notice(original, effective)

    def _refresh_settings_project_root_display(self) -> None:
        # Never lazy-materialize the project settings page just to update a label.
        label = self.__dict__.get("settings_project_root_value")
        if label is None:
            return
        root = self.state.get_game_root()
        if root:
            label.setText(str(root))
            label.setToolTip(str(root))
        else:
            label.setText("（尚未选择项目，请前往「项目列表」切换）")
            label.setToolTip("")

    def _refresh_project_label(self):
        root = self.state.get_game_root()
        path_text = str(root) if root else "（尚未选择项目）"
        if hasattr(self, "global_project_path_edit"):
            self.global_project_path_edit.setText(path_text)
            self.global_project_path_edit.setToolTip(path_text)
        elif hasattr(self, "project_path_edit"):
            self.project_path_edit.setText(path_text)
        if not root:
            self._clear_game_root_redirect_notice()
        self._refresh_settings_project_root_display()

    def _is_doctor_running(self) -> bool:
        return self._doctor_worker is not None and self._doctor_worker.isRunning()

    def _on_doctor_button_clicked(self) -> None:
        """Project-bar CTA: start 环境检查, or cancel doctor/template jobs."""
        if self._is_doctor_active() or self._is_generate_template_active():
            self._on_kill()
            return
        self._on_run_doctor()

    def _on_bootstrap_button_clicked(self) -> None:
        """Project-bar CTA: start 准备工作目录, or cancel while it is running."""
        if self._is_bootstrap_work_active():
            self._on_kill()
            return
        self._on_bootstrap_work()

    def _snapshot_runtime_config_for_job(self):
        """Build a frozen RuntimeConfig for an in-process background job.

        Reloads translator settings under the runtime lock so the snapshot
        matches the current project on disk, then returns an independent copy.
        Returns None if runtime cannot be imported (worker will reload itself).
        """
        try:
            import translator_runtime as legacy
        except Exception:
            return None
        try:
            with legacy.locked_runtime_state():
                legacy.load_translator_settings()
                return legacy.snapshot_runtime_config()
        except SystemExit:
            # Hard config errors surface again inside the worker with user-facing text.
            return None
        except Exception:
            return None

    def _on_run_doctor(self):
        if not self._confirm_unsaved_config_before_workflow():
            return
        if not self.state.get_game_root():
            QMessageBox.information(
                self, "请先选择项目",
                "请先选择游戏的 work 目录。\n"
                "环境检查会读取本地配置中的项目路径。"
            )
            return
        if self._is_doctor_running():
            return

        # Global bar can launch from Settings/Diagnostics; show status on Workbench.
        self._focus_workbench_main_tab(special_route=_SHELL_ROUTE_PROJECT_PREPARE)
        self._clear_log_view()
        self._active_command = "doctor"
        self._doctor_output_lines = []
        self._focus_workbench_status_tab(0)
        self._show_workbench_log_drawer()
        self._set_doctor_summary(running_summary())
        self._append_log("=== 正在运行环境检查（collect_doctor_report）===\n")
        self._set_task_running(True)

        # Snapshot current process runtime for this job so a later project switch
        # (or disk reload) cannot mutate globals mid-doctor. The worker restores
        # prior globals after the check (issue #216 phase 2).
        doctor_config = self._snapshot_runtime_config_for_job()
        self._doctor_worker = DoctorWorker(config=doctor_config, parent=self)
        self._doctor_worker.completed.connect(self._on_doctor_completed)
        self._doctor_worker.start()

    def _on_doctor_completed(self, result: DoctorWorkerResult) -> None:
        worker = self.sender()
        if worker is self._doctor_worker:
            self._doctor_worker = None
        self._active_command = ""
        self._set_task_running(False)

        log_text = result.log_text.strip()
        if log_text:
            self._doctor_output_lines = log_text.splitlines()
            self._append_log(log_text)
        elif result.error:
            self._doctor_output_lines = [result.error]
            self._append_log(result.error)

        self._append_log("\n[环境检查已结束]")

        api_key_count, api_key_source = self.state.get_api_key_status()
        if result.ok and result.report is not None:
            self._last_doctor_report = result.report
            self._last_doctor_report_game_root = self.state.get_game_root() or ""
            summary = summarize_doctor_report(
                result.report,
                exit_code=0,
                api_key_count=api_key_count,
                api_key_source=api_key_source,
            )
            workspace_root = self.state.get_workspace_root()
            compare = None
            if workspace_root is not None:
                compare = compare_registry_with_doctor_report(
                    workspace_root,
                    game_root=self.state.get_game_root(),
                    report=result.report,
                )
            if compare is not None:
                self._append_log(compare.log_line)
                summary = DoctorSummary(
                    status=summary.status,
                    heading=summary.heading,
                    message=summary.message,
                    facts=[*summary.facts, compare.message],
                    findings=summary.findings,
                    mode=summary.mode,
                    detail_facts=list(summary.detail_facts or []),
                )
            self._doctor_check_completed = True
            status_message = "项目检查完成。"
            if compare is not None and compare.matched is False:
                status_message = "项目检查完成。总表记录需刷新。"
            elif compare is not None and compare.matched is None:
                status_message = "项目检查完成。当前项目未在总表中登记。"
            self.statusBar().showMessage(status_message, 8000)
        else:
            summary = summarize_doctor_output(
                result.error or log_text,
                exit_code=-1,
                api_key_count=api_key_count,
                api_key_source=api_key_source,
            )
            self._doctor_check_completed = False
            self._last_doctor_report = None
            self._last_doctor_report_game_root = ""
            self.statusBar().showMessage("项目检查失败。", 6000)

        self._set_doctor_summary(summary)

    def _on_bootstrap_work(self):
        if not self._confirm_unsaved_config_before_workflow():
            return
        if not self.state.get_game_root():
            QMessageBox.information(
                self,
                "请先选择项目",
                "请先选择游戏目录（项目根目录或 work 目录均可）。\n"
                "准备工作目录会在存在 original/game 时，把内容复制到 work/game。",
            )
            return

        # Global bar can launch from Settings/Diagnostics; show status on Workbench.
        self._focus_workbench_main_tab(special_route=_SHELL_ROUTE_PROJECT_PREPARE)
        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._active_command = "bootstrap_work"
        self._work_bootstrap_output_lines = []
        self._workflow_progress = create_workflow_progress_state("work_bootstrap")
        self._workflow_progress_base_facts = []
        # 项目与环境只有环境检查页；准备进度折叠进同一项目状态摘要。
        self._focus_workbench_status_tab(0)
        doctor_summary = work_bootstrap_to_doctor_summary(running_work_bootstrap_summary())
        self._set_doctor_summary(doctor_summary)
        self._set_workflow_summary(
            "running",
            doctor_summary.heading,
            doctor_summary.message,
            doctor_summary.facts,
        )
        self._apply_workflow_progress_ui()
        self._append_log("=== 正在运行：gemini_translate_batch.py bootstrap-work ===\n")
        self._set_task_running(True)
        self.runner.run(self.state.get_batch_script_path(), ["bootstrap-work"])

    def _on_generate_template(self):
        if not self._confirm_unsaved_config_before_workflow():
            return
        if not self.state.get_game_root():
            QMessageBox.information(
                self,
                "请先选择项目",
                "请先选择游戏的 work 目录。",
            )
            return

        # Status + cancel live on 项目与环境 (doctor summary surface).
        self._focus_workbench_main_tab(special_route=_SHELL_ROUTE_PROJECT_PREPARE)
        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._active_command = "generate_template"
        self._template_generation_output_lines = []
        self._focus_workbench_status_tab(0)
        running_summary = running_template_generation_summary()
        doctor_summary = template_generation_to_doctor_summary(running_summary)
        self._set_doctor_summary(doctor_summary)
        self._set_workflow_summary(
            "running",
            running_summary.heading,
            running_summary.message,
            running_summary.facts,
        )
        self._append_log("=== 正在运行：gemini_translate_batch.py generate-template ===\n")
        self._set_task_running(True)
        self.runner.run(self.state.get_batch_script_path(), ["generate-template"])

    def _game_root_str_for_flags(self) -> str | None:
        """Resolve current game_root for project-scoped context flags (defensive)."""
        state = getattr(self, "state", None)
        get_game_root = getattr(state, "get_game_root", None)
        game_root = get_game_root() if callable(get_game_root) else None
        return str(game_root) if game_root else None

    def _saved_batch_context_flags(self) -> dict[str, bool]:
        return read_batch_context_flags(
            self.state.load_translator_config(),
            game_root=self._game_root_str_for_flags(),
        )

    def _submit_max_cost_from_config(self) -> float | None:
        load_config = getattr(self.state, "load_translator_config", None)
        if not callable(load_config):
            return None
        return resolve_submit_max_cost(load_config())

    def _bootstrap_task_ready(self, spec) -> bool:
        if not spec.is_bootstrap:
            return True
        flags = self._saved_batch_context_flags()
        if spec.bootstrap_kind == "rag":
            return flags["rag_enabled"]
        if spec.bootstrap_kind == "source_index":
            return flags["source_index_enabled"]
        return True

    def _start_bootstrap_task(self, kind: str) -> bool:
        if bool(getattr(self, "_task_running", False)):
            return False
        # FakeRunner stubs in unit tests may omit is_running(); treat as idle then.
        runner = getattr(self, "runner", None)
        is_running = getattr(runner, "is_running", None) if runner is not None else None
        if callable(is_running) and bool(is_running()):
            return False
        if not self._confirm_unsaved_config_before_workflow():
            return False
        if not self.state.get_game_root():
            QMessageBox.information(self, "请先选择项目", "请先选择游戏的 work 目录。")
            return False

        flags = self._saved_batch_context_flags()
        if kind == "rag":
            if not flags["rag_enabled"]:
                QMessageBox.information(
                    self,
                    "记忆库未启用",
                    bootstrap_disabled_message("rag"),
                )
                return False
            command = "bootstrap_rag"
            args = ["bootstrap-rag", "--skip-prepare"]
            log_heading = "gemini_translate_batch.py bootstrap-rag --skip-prepare"
            running_summary = running_bootstrap_summary("rag")
        else:
            if not flags["source_index_enabled"]:
                QMessageBox.information(
                    self,
                    "原文索引未启用",
                    bootstrap_disabled_message("source_index"),
                )
                return False
            command = "bootstrap_source_index"
            args = ["bootstrap-source-index", "--skip-prepare"]
            log_heading = "gemini_translate_batch.py bootstrap-source-index --skip-prepare"
            running_summary = running_bootstrap_summary("source_index")

        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._active_command = command
        self._bootstrap_output_lines = []
        self._bootstrap_progress = create_bootstrap_progress_state(kind)
        self._bootstrap_progress_tracker = (
            create_bootstrap_progress_tracker()
            if kind == "source_index"
            else None
        )
        self._workflow_progress = (
            create_workflow_progress_state("rag_bootstrap")
            if kind == "rag"
            else None
        )
        self._workflow_progress_base_facts = list(running_summary.facts)
        if kind == "source_index" and hasattr(self, "_bootstrap_progress_eta_timer"):
            self._bootstrap_progress_eta_timer.start()
        elif hasattr(self, "_bootstrap_progress_eta_timer"):
            self._bootstrap_progress_eta_timer.stop()
        self._focus_workbench_status_tab(1)
        self._set_workflow_from_bootstrap_summary(running_summary)
        self._append_log(f"=== 正在运行：{log_heading} ===\n")
        self._set_task_running(True)
        self._apply_bootstrap_progress_ui()
        self.runner.run(self.state.get_batch_script_path(), args)
        return True

    def _on_start_translation(self):
        spec = work_mode_spec(self._current_work_mode())
        if not spec.implemented:
            QMessageBox.information(self, "功能开发中", spec.not_implemented_message)
            return
        if spec.is_bootstrap:
            self._start_bootstrap_task(spec.bootstrap_kind)
            return
        if self._should_generate_template_only():
            self._on_generate_template()
            return
        if not self.state.get_game_root():
            QMessageBox.information(
                self,
                "请先选择项目",
                "请先选择游戏的 work 目录。",
            )
            return
        if self._translation_requires_doctor_check(spec.mode) and not self._doctor_allows_translate_action():
            if not self._doctor_check_completed:
                detail = "批量翻译与同步翻译需要先完成环境检查，确认项目状态后再开始。"
            else:
                detail = (
                    "当前环境检查未通过或仍有阻塞项，无法开始翻译。"
                    "请查看环境检查摘要，处理问题后重新运行检查。"
                )
            QMessageBox.information(self, "请先运行环境检查", detail)
            return

        if not self._confirm_unsaved_config_before_workflow():
            return

        if self._litellm_install_blocks_mode(spec.mode):
            QMessageBox.information(
                self,
                "LiteLLM 正在安装",
                "请等待 LiteLLM 后台安装完成后再启动同步任务。其他功能仍可继续使用。",
            )
            return

        if spec.mode in self._sync_work_modes_requiring_api_key():
            sync_backend = self._saved_sync_backend()
            if sync_backend == "litellm":
                if importlib.util.find_spec("litellm") is None:
                    QMessageBox.information(
                        self,
                        "尚未安装 LiteLLM",
                        "当前同步后端是 LiteLLM。请先运行：\n"
                        "pip install -r requirements-litellm.txt\n\n"
                        "供应商密钥可在设置 · LiteLLM 中保存，也可使用环境变量。",
                    )
                    return
            else:
                api_key_count, _ = self.state.get_api_key_status()
                if api_key_count == 0:
                    QMessageBox.information(
                        self,
                        "请先配置 API Key",
                        "Gemini 同步模式需要 API 密钥；请在设置页管理 API Key 或设置环境变量。",
                    )
                    return

        workflow = create_workflow(
            spec.mode,
            submit_max_cost=self._submit_max_cost_from_config(),
        )
        if workflow is None:
            QMessageBox.information(self, "无法开始任务", spec.not_implemented_message)
            return

        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._clear_completed_manifest_snapshot()
        self._writeback_manifest_path = ""
        if spec.mode == WorkMode.SYNC_TRANSLATION:
            self.sync_translation_page.clear_preview()
        if spec.supports_translation_writeback:
            self._set_writeback_summary(stale_writeback_summary())
        else:
            self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
        self._begin_translation_workflow(
            workflow,
            log_heading=f"正在运行：{spec.label}",
            status_tab=1,
        )

    def _begin_translation_workflow(
        self,
        workflow,
        *,
        log_heading: str,
        status_tab: int = 1,
    ) -> None:
        self._workflow = workflow
        self._active_command = "translation_workflow"
        self._workflow_step_output_lines = []
        self._focus_workbench_status_tab(status_tab)
        self._append_log(f"=== {log_heading} ===\n")
        self._set_task_running(True)
        self._run_workflow_current_step()

    def _on_resume_translation(self):
        if not self._confirm_unsaved_config_before_workflow():
            return
        spec = work_mode_spec(self._current_work_mode())
        if not spec.implemented or not spec.supports_resume:
            QMessageBox.information(self, "功能开发中", spec.not_implemented_message)
            return
        if not self.state.get_game_root():
            QMessageBox.information(
                self,
                "请先选择项目",
                "请先选择游戏的 work 目录。",
            )
            return

        game_root = self.state.get_game_root()
        latest_manifest = self._latest_resume_manifest_path(game_root, spec)
        if latest_manifest is None:
            QMessageBox.information(
                self,
                "没有可继续的任务",
                f"未找到最近任务记录；请先开始一个{spec.label}任务。",
            )
            return

        try:
            manifest = self.state.load_resume_manifest(
                latest_manifest,
                work_mode=spec.mode,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "无法继续最新任务", str(exc))
            return

        only_query = self._resume_button_is_query_mode()
        split_entries = self._split_entries_for_manifest(str(latest_manifest), manifest)
        if only_query and spec.mode == WorkMode.BATCH_TRANSLATION and len(split_entries) > 1:
            split_workflow = SplitBatchQueueWorkflow.refresh_status(
                split_entries,
                anchor_manifest_path=str(latest_manifest),
            )
            if split_workflow.current_step() is not None:
                self._clear_log_view()
                self._show_workbench_log_drawer()
                self._workflow = split_workflow
                self._refresh_diagnostics_context()
                self._active_command = "translation_workflow"
                self._workflow_step_output_lines = []
                self._focus_workbench_status_tab(1)
                self._append_log("=== 正在刷新全部拆分包状态 ===\n")
                self._set_task_running(True)
                self._run_workflow_current_step()
                return

        workflow = resume_workflow(
            spec.mode,
            str(latest_manifest),
            manifest,
            submit_max_cost=self._submit_max_cost_from_config(),
        )
        if workflow is None:
            QMessageBox.information(self, "无法继续任务", spec.not_implemented_message)
            return
        if workflow.current_step() is None:
            self._refresh_diagnostics_context()
            self._refresh_writeback_from_latest_manifest()
            self._refresh_workflow_from_latest_manifest(
                latest_manifest=latest_manifest,
                manifest=manifest,
            )
            writeback_summary = self._current_writeback_summary()
            if writeback_summary.status not in {"idle", "running", "stale"}:
                self._focus_workbench_status_tab(2)
            self.statusBar().showMessage(f"最新{spec.label}任务已完成。", 6000)
            return

        only_query = self._resume_button_is_query_mode()
        workflow.only_query = only_query

        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._workflow = workflow
        self._refresh_diagnostics_context()
        self._active_command = "translation_workflow"
        self._workflow_step_output_lines = []
        self._focus_workbench_status_tab(1)
        self._append_log(f"=== 正在继续最新 {spec.label} 任务 ===\n")
        self._set_task_running(True)
        self._run_workflow_current_step()

    def _on_submit_remaining_split_packages(self) -> None:
        spec = work_mode_spec(self._current_work_mode())
        if spec.mode != WorkMode.BATCH_TRANSLATION:
            return
        latest_manifest, entries = self._load_latest_split_entries()
        pending_count = sum(1 for entry in entries if entry.needs_submit)
        if not latest_manifest or not entries or pending_count <= 0:
            QMessageBox.information(self, "没有待提交拆分包", "当前拆分组没有尚未提交的包。")
            self._refresh_split_status_ui()
            return

        reply = QMessageBox.question(
            self,
            "确认批量提交",
            (
                f"将按顺序提交 {pending_count} 个尚未提交的拆分包。\n"
                "已提交、处理中、已完成或已写回的包不会重复提交。"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        workflow = SplitBatchQueueWorkflow.submit_remaining(
            entries,
            anchor_manifest_path=latest_manifest,
            submit_max_cost=self._submit_max_cost_from_config(),
        )
        if workflow.current_step() is None:
            QMessageBox.information(self, "没有待提交拆分包", "当前拆分组没有尚未提交的包。")
            return

        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._workflow = workflow
        self._refresh_diagnostics_context()
        self._active_command = "translation_workflow"
        self._workflow_step_output_lines = []
        self._focus_workbench_status_tab(1)
        self._append_log(f"=== 正在批量提交剩余拆分包（{pending_count} 个） ===\n")
        self._set_task_running(True)
        self._run_workflow_current_step()

    def _on_kill(self):
        if self._is_doctor_running():
            self._invalidate_doctor_worker()
            self._append_log("\n[环境检查已取消]\n")
            self._doctor_check_completed = False
            self._doctor_summary_status = ""
            self._last_doctor_report = None
            self._last_doctor_report_game_root = ""
            self._set_doctor_summary(cancelled_summary())
            self.statusBar().showMessage("环境检查已取消。", 6000)
            return
        self.runner.kill()

    def _prompt_probe_options(self) -> dict[str, int | None] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("试跑样本请求")
        layout = QVBoxLayout(dialog)

        hint = QLabel(
            "将对当前翻译包执行少量同步请求，不会提交批量任务，也不会修改项目文件。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        limit_spin = QSpinBox()
        limit_spin.setRange(1, 50)
        limit_spin.setValue(3)
        form.addRow("样本条数 (--limit)", limit_spin)

        offset_spin = QSpinBox()
        offset_spin.setRange(0, 1_000_000)
        offset_spin.setValue(0)
        form.addRow("起始偏移 (--offset)", offset_spin)

        api_key_spin = QSpinBox()
        api_key_spin.setRange(-1, 99)
        api_key_spin.setValue(-1)
        api_key_spin.setSpecialValueText("默认")
        form.addRow("API Key 索引 (--api-key-index)", api_key_spin)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(cancel_btn)
        run_btn = QPushButton("开始试跑")
        run_btn.setObjectName("primary_btn")
        run_btn.clicked.connect(dialog.accept)
        buttons.addWidget(run_btn)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        api_key_index = None if api_key_spin.value() < 0 else api_key_spin.value()
        return {
            "limit": limit_spin.value(),
            "offset": offset_spin.value(),
            "api_key_index": api_key_index,
        }

    def _prompt_compare_variants_options(self) -> dict[str, object] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("翻译 A/B 对比")
        layout = QVBoxLayout(dialog)

        hint = QLabel(
            "将对当前翻译包采样若干 chunk，用 baseline（当前 translator_config）"
            "与所选配置变体并排生成译文报告；不会写回 .rpy 或 glossary.json。"
            "每个对比项只能选一种：不参与、强制开启或强制关闭。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        variants_group = QGroupBox("对比变体")
        variants_layout = QVBoxLayout(variants_group)
        baseline_check = QCheckBox("baseline（当前配置，始终包含）")
        baseline_check.setChecked(True)
        baseline_check.setEnabled(False)
        variants_layout.addWidget(baseline_check)

        current_states = read_ab_dimension_enabled_states(self.state.load_translator_config())
        dimension_button_groups: dict[str, QButtonGroup] = {}
        for dimension in AB_VARIANT_DIMENSIONS:
            dimension_id = str(dimension["id"])
            label = str(dimension["label"])
            current_enabled = current_states.get(dimension_id, False)
            current_text = "当前：开启" if current_enabled else "当前：关闭"

            row = QHBoxLayout()
            row.addWidget(QLabel(f"{label}（{current_text}）"))

            button_group = QButtonGroup(dialog)
            skip_radio = QRadioButton("不参与")
            on_radio = QRadioButton("强制开启")
            off_radio = QRadioButton("强制关闭")
            skip_radio.setChecked(True)
            button_group.addButton(skip_radio, 0)
            button_group.addButton(on_radio, 1)
            button_group.addButton(off_radio, 2)
            row.addWidget(skip_radio)
            row.addWidget(on_radio)
            row.addWidget(off_radio)
            row.addStretch(1)
            variants_layout.addLayout(row)
            dimension_button_groups[dimension_id] = button_group

        form.addRow(variants_group)

        limit_spin = QSpinBox()
        limit_spin.setRange(1, 50)
        limit_spin.setValue(3)
        form.addRow("采样块数 (--limit)", limit_spin)

        offset_spin = QSpinBox()
        offset_spin.setRange(0, 1_000_000)
        offset_spin.setValue(0)
        form.addRow("起始偏移 (--offset)", offset_spin)

        output_dir_edit = QLineEdit()
        output_dir_edit.setPlaceholderText("留空则写入 logs/experiments/")
        form.addRow("输出目录 (--output-dir)", output_dir_edit)

        dry_run_check = QCheckBox("仅试跑（不调用翻译 API）")
        dry_run_check.setChecked(True)
        form.addRow("", dry_run_check)

        api_key_spin = QSpinBox()
        api_key_spin.setRange(-1, 99)
        api_key_spin.setValue(-1)
        api_key_spin.setSpecialValueText("默认")
        form.addRow("API Key 索引 (--api-key-index)", api_key_spin)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(cancel_btn)
        run_btn = QPushButton("开始对比")
        run_btn.setObjectName("primary_btn")
        run_btn.clicked.connect(dialog.accept)
        buttons.addWidget(run_btn)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        dimension_choices: dict[str, str] = {}
        for dimension_id, button_group in dimension_button_groups.items():
            checked_id = button_group.checkedId()
            if checked_id == 1:
                dimension_choices[dimension_id] = AB_VARIANT_CHOICE_FORCE_ON
            elif checked_id == 2:
                dimension_choices[dimension_id] = AB_VARIANT_CHOICE_FORCE_OFF
            else:
                dimension_choices[dimension_id] = AB_VARIANT_CHOICE_SKIP
        variants = build_variants_from_gui_selection(dimension_choices)
        valid, validation_message = validate_ab_experiment_variants(variants)
        if not valid:
            QMessageBox.warning(dialog, "对比项不足", validation_message)
            return None

        api_key_index = None if api_key_spin.value() < 0 else api_key_spin.value()
        return {
            "variants": variants,
            "limit": limit_spin.value(),
            "offset": offset_spin.value(),
            "output_dir": output_dir_edit.text().strip(),
            "dry_run": dry_run_check.isChecked(),
            "api_key_index": api_key_index,
        }

    def _prompt_split_options(self) -> dict[str, int | str] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("拆分翻译包")
        layout = QVBoxLayout(dialog)

        hint = QLabel(
            "将把当前翻译包拆成多个子包。拆分后 RAG 记忆库为静态快照，"
            "各子包需分别 submit，不会自动提交。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        max_chunks_spin = QSpinBox()
        max_chunks_spin.setRange(1, 10_000)
        max_chunks_spin.setValue(600)
        form.addRow("每包最大块数 (--max-chunks)", max_chunks_spin)

        max_items_spin = QSpinBox()
        max_items_spin.setRange(0, 1_000_000)
        max_items_spin.setValue(0)
        max_items_spin.setSpecialValueText("不限制")
        form.addRow("每包最大条目 (--max-items)", max_items_spin)

        prefix_edit = QLineEdit()
        prefix_edit.setPlaceholderText("留空则沿用源包显示名")
        form.addRow("显示名前缀 (--display-name-prefix)", prefix_edit)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(cancel_btn)
        run_btn = QPushButton("开始拆分")
        run_btn.setObjectName("primary_btn")
        run_btn.clicked.connect(dialog.accept)
        buttons.addWidget(run_btn)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        return {
            "max_chunks": max_chunks_spin.value(),
            "max_items": max_items_spin.value(),
            "display_name_prefix": prefix_edit.text().strip(),
        }

    def _on_run_probe(self) -> None:
        manifest_path, manifest = self._current_diagnostics_manifest()
        ready, message = translation_probe_ready(manifest_path, manifest)
        if not ready:
            QMessageBox.information(self, "无法试跑样本请求", message)
            return

        options = self._prompt_probe_options()
        if options is None:
            return

        self._clear_log_view()
        # Workbench entry (P2a): keep user on batch · 执行 with drawer logs.
        self._show_workbench_log_drawer()
        self._active_command = "probe"
        self._probe_output_lines = []
        base_context = build_diagnostics_context(
            latest_manifest_path=manifest_path,
            manifest=manifest,
            batch_script_path=str(self.state.get_batch_script_path()),
            logs_dir=str(self.state.get_logs_dir()),
            python_exe=sys.executable,
            submit_max_cost=self._submit_max_cost_from_config(),
        )
        self._set_diagnostics_context(
            probe_summary_to_diagnostics_context(
                running_probe_summary(manifest_path=manifest_path),
                base_context,
            )
        )
        args = build_probe_cli_args(
            manifest_path,
            limit=int(options["limit"]),
            offset=int(options["offset"]),
            api_key_index=options["api_key_index"],
        )
        self._append_log(
            f"=== 正在试跑样本请求：gemini_translate_batch.py {' '.join(args)} ===\n"
        )
        self._set_task_running(True)
        self.runner.run(self.state.get_batch_script_path(), args)

    def _cleanup_compare_variants_temp_file(self) -> None:
        temp_file = self._compare_variants_temp_file
        self._compare_variants_temp_file = ""
        if not temp_file:
            return
        Path(temp_file).unlink(missing_ok=True)

    def _on_run_compare_variants(self) -> None:
        manifest_path, manifest = self._current_diagnostics_manifest()
        ready, message = translation_ab_experiment_ready(manifest_path, manifest)
        if not ready:
            QMessageBox.information(self, "无法运行翻译 A/B 对比", message)
            return

        options = self._prompt_compare_variants_options()
        if options is None:
            return

        self._clear_log_view()
        self._expand_diagnostics_log()
        self._active_command = "compare_variants"
        self._compare_variants_output_lines = []
        variants = list(options["variants"])
        variant_names = format_variant_names(variants)
        self._compare_variants_names = variant_names
        variants_file = write_variants_to_temp_file(variants)
        self._compare_variants_temp_file = variants_file
        dry_run = bool(options["dry_run"])
        base_context = build_diagnostics_context(
            latest_manifest_path=manifest_path,
            manifest=manifest,
            batch_script_path=str(self.state.get_batch_script_path()),
            logs_dir=str(self.state.get_logs_dir()),
            python_exe=sys.executable,
            submit_max_cost=self._submit_max_cost_from_config(),
        )
        self._set_diagnostics_context(
            ab_experiment_summary_to_diagnostics_context(
                running_ab_experiment_summary(
                    manifest_path=manifest_path,
                    variant_names=variant_names,
                    dry_run=dry_run,
                ),
                base_context,
            )
        )
        args = build_compare_variants_cli_args(
            manifest_path,
            variants_file,
            limit=int(options["limit"]),
            offset=int(options["offset"]),
            output_dir=str(options["output_dir"]),
            dry_run=dry_run,
            api_key_index=options["api_key_index"],  # type: ignore[arg-type]
        )
        self._append_log(
            f"=== 正在运行翻译 A/B 对比：gemini_translate_batch.py {' '.join(args)} ===\n"
        )
        self._set_task_running(True)
        self.runner.run(self.state.get_batch_script_path(), args)

    def _on_run_split(self) -> None:
        manifest_path, manifest = self._current_diagnostics_manifest()
        ready, message = translation_split_ready(manifest_path, manifest)
        if not ready:
            QMessageBox.information(self, "无法拆分翻译包", message)
            return

        options = self._prompt_split_options()
        if options is None:
            return

        self._clear_log_view()
        # Workbench entry (P2a): keep user on batch · 执行 with drawer logs.
        self._show_workbench_log_drawer()
        self._active_command = "split"
        self._split_output_lines = []
        base_context = build_diagnostics_context(
            latest_manifest_path=manifest_path,
            manifest=manifest,
            batch_script_path=str(self.state.get_batch_script_path()),
            logs_dir=str(self.state.get_logs_dir()),
            python_exe=sys.executable,
            submit_max_cost=self._submit_max_cost_from_config(),
        )
        self._set_diagnostics_context(
            split_summary_to_diagnostics_context(
                running_split_summary(manifest_path=manifest_path),
                base_context,
            )
        )
        args = build_split_cli_args(
            manifest_path,
            max_chunks=int(options["max_chunks"]),
            max_items=int(options["max_items"]),
            display_name_prefix=str(options["display_name_prefix"]),
        )
        self._append_log(
            f"=== 正在拆分翻译包：gemini_translate_batch.py {' '.join(args)} ===\n"
        )
        self._set_task_running(True)
        self.runner.run(self.state.get_batch_script_path(), args)

    def _repair_search_roots(self) -> list[str]:
        roots: list[str] = []
        game_root = self.state.get_game_root()
        if game_root is not None:
            roots.append(str(game_root))
        return roots

    def _discover_repair_report_candidates(
        self,
        manifest: dict[str, object],
        *,
        manifest_path: str,
    ) -> list[RepairReportCandidate]:
        return discover_repair_report_candidates(
            manifest,
            manifest_path=manifest_path,
            search_roots=self._repair_search_roots(),
        )

    def _prompt_repair_report_path(
        self,
        candidates: list[RepairReportCandidate],
    ) -> str:
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0].path

        dialog = QDialog(self)
        dialog.setWindowTitle("选择修补报告")
        layout = QVBoxLayout(dialog)
        hint = QLabel("找到多个可用于同步修补的报告，请选择其一：")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        combo = QComboBox()
        for candidate in candidates:
            combo.addItem(f"{candidate.label} — {candidate.path}", candidate.path)
        layout.addWidget(combo)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(cancel_btn)
        ok_btn = QPushButton("继续")
        ok_btn.setObjectName("primary_btn")
        ok_btn.clicked.connect(dialog.accept)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return ""
        return str(combo.currentData() or "")

    def _prompt_repair_options(self) -> dict[str, int | None] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("同步修补")
        layout = QVBoxLayout(dialog)

        hint = QLabel(
            "将按 JSONL 报告同步修补剩余条目。"
            "该操作会直接修改翻译文件，不会提交批量任务。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        batch_size_spin = QSpinBox()
        batch_size_spin.setRange(1, 20)
        batch_size_spin.setValue(2)
        form.addRow("每批条目 (--batch-size)", batch_size_spin)

        limit_spin = QSpinBox()
        limit_spin.setRange(0, 1_000_000)
        limit_spin.setValue(0)
        limit_spin.setSpecialValueText("不限制")
        form.addRow("最大条目 (--limit)", limit_spin)

        offset_spin = QSpinBox()
        offset_spin.setRange(0, 1_000_000)
        offset_spin.setValue(0)
        form.addRow("起始偏移 (--offset)", offset_spin)

        context_before_spin = QSpinBox()
        context_before_spin.setRange(0, 20)
        context_before_spin.setValue(2)
        form.addRow("上文条数 (--context-before)", context_before_spin)

        context_after_spin = QSpinBox()
        context_after_spin.setRange(0, 20)
        context_after_spin.setValue(2)
        form.addRow("下文条数 (--context-after)", context_after_spin)

        api_key_spin = QSpinBox()
        api_key_spin.setRange(-1, 99)
        api_key_spin.setValue(-1)
        api_key_spin.setSpecialValueText("默认")
        form.addRow("API Key 索引 (--api-key-index)", api_key_spin)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(cancel_btn)
        run_btn = QPushButton("开始修补")
        run_btn.setObjectName("primary_btn")
        run_btn.clicked.connect(dialog.accept)
        buttons.addWidget(run_btn)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        api_key_index = None if api_key_spin.value() < 0 else api_key_spin.value()
        return {
            "batch_size": batch_size_spin.value(),
            "limit": limit_spin.value(),
            "offset": offset_spin.value(),
            "context_before": context_before_spin.value(),
            "context_after": context_after_spin.value(),
            "api_key_index": api_key_index,
        }

    def _on_run_repair(self) -> None:
        spec = work_mode_spec(self._current_work_mode())
        if not spec.supports_translation_writeback:
            QMessageBox.information(
                self,
                "当前模式不支持",
                "「同步修补」仅适用于批量翻译。",
            )
            return

        manifest_path = self._writeback_manifest_path
        if not manifest_path:
            QMessageBox.information(self, "无法同步修补", "没有可用的任务记录。")
            return

        try:
            manifest = self.state.load_manifest_file(manifest_path)
        except ValueError as exc:
            QMessageBox.warning(self, "无法同步修补", str(exc))
            return

        eligibility = assess_repair_eligibility(manifest, manifest_path=manifest_path)
        if not eligibility.eligible:
            QMessageBox.information(self, eligibility.heading, eligibility.message)
            return

        candidates = self._discover_repair_report_candidates(
            manifest,
            manifest_path=manifest_path,
        )
        report_path = self._prompt_repair_report_path(candidates)
        if not report_path:
            if not candidates:
                QMessageBox.information(
                    self,
                    "没有可用的修补报告",
                    "未在任务包或项目目录找到 remaining_need_translate_*.jsonl、"
                    "failures.jsonl，也无法从检查报告提取 repair 类条目。"
                    "请先生成或准备 JSONL 报告后再试。",
                )
            return

        options = self._prompt_repair_options()
        if options is None:
            return

        confirm_lines = [
            "即将执行同步修补，并直接修改翻译文件。",
            "修补前请确认已在副本或备份上验证。",
            "",
            eligibility.message,
            f"修补报告：{report_path}",
        ]
        reply = QMessageBox.question(
            self,
            "确认同步修补",
            "\n".join(confirm_lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._active_command = "repair"
        self._repair_output_lines = []
        base_context = build_diagnostics_context(
            latest_manifest_path=manifest_path,
            manifest=manifest,
            batch_script_path=str(self.state.get_batch_script_path()),
            logs_dir=str(self.state.get_logs_dir()),
            python_exe=sys.executable,
            submit_max_cost=self._submit_max_cost_from_config(),
        )
        self._set_diagnostics_context(
            repair_summary_to_diagnostics_context(
                running_repair_summary(
                    report_path=report_path,
                    manifest_path=manifest_path,
                ),
                base_context,
            )
        )
        args = build_repair_cli_args(
            report_path,
            limit=int(options["limit"]),
            offset=int(options["offset"]),
            batch_size=int(options["batch_size"]),
            context_before=int(options["context_before"]),
            context_after=int(options["context_after"]),
            api_key_index=options["api_key_index"],
        )
        self._append_log(
            f"=== 正在同步修补：gemini_translate_batch.py {' '.join(args)} ===\n"
        )
        self._set_task_running(True)
        self.runner.run(self.state.get_batch_script_path(), args)

    def _on_recheck_writeback(self) -> None:
        spec = work_mode_spec(self._current_work_mode())
        if not spec.supports_translation_writeback:
            QMessageBox.information(
                self,
                "当前模式不支持",
                "「重新检查」仅适用于批量翻译。",
            )
            return
        if not self._writeback_manifest_path:
            QMessageBox.information(self, "无法重新检查", "没有可检查的任务记录。")
            return

        summary = self._current_writeback_summary()
        if not recheck_writeback_ready(
            summary,
            supports_translation_writeback=spec.supports_translation_writeback,
        ):
            QMessageBox.information(
                self,
                "无法重新检查",
                "请先完成翻译并下载结果，再重新检查。",
            )
            return

        manifest_path = self._writeback_manifest_path
        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._active_command = "recheck"
        self._recheck_output_lines = []
        self._focus_workbench_status_tab(2)
        self._set_writeback_summary(
            running_writeback_summary(
                manifest_path=manifest_path,
                heading="正在重新检查",
                message="正在校验翻译结果是否可以写回；完成后这里会更新写回摘要。",
            )
        )
        args = build_recheck_cli_args(manifest_path)
        self._append_log(
            f"=== 正在重新检查：gemini_translate_batch.py {' '.join(args)} ===\n"
        )
        self._set_task_running(True)
        self.runner.run(self.state.get_batch_script_path(), args)

    def _on_apply_writeback(self):
        if not work_mode_spec(self._current_work_mode()).supports_translation_writeback:
            QMessageBox.information(
                self,
                "当前模式不支持",
                "「写回翻译」仅适用于批量翻译。",
            )
            return
        if not self._writeback_manifest_path:
            QMessageBox.information(self, "无法写回", "没有可写回的任务；请先完成结果检查。")
            return

        summary = self._current_writeback_summary()
        if not summary.can_apply:
            QMessageBox.information(
                self,
                "当前不能写回",
                summary.message or "只有检查结果为可写回时才允许写回。",
            )
            return

        confirm_lines = [
            "即将把翻译写回游戏脚本。",
            "写回前请确认已在副本或备份上验证。",
            "",
            *summary.facts,
        ]

        reply = QMessageBox.question(
            self,
            "确认写回翻译",
            "\n".join(confirm_lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        manifest_path = self._writeback_manifest_path
        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._active_command = "apply"
        self._apply_output_lines = []
        self._focus_workbench_status_tab(2)
        self._set_writeback_summary(running_writeback_summary(manifest_path=manifest_path))
        self._append_log(
            f"=== 正在写回：gemini_translate_batch.py apply {manifest_path} ===\n"
        )
        self._set_task_running(True)
        self.runner.run(
            self.state.get_batch_script_path(),
            ["apply", manifest_path],
        )

    def _on_apply_sync_translation(self) -> None:
        if self._current_work_mode() != WorkMode.SYNC_TRANSLATION:
            QMessageBox.information(self, "当前模式不支持", "请先切换到同步翻译。")
            return
        manifest_path = self.sync_translation_page.preview_manifest_path()
        if not manifest_path:
            QMessageBox.information(self, "无法写回", "请先生成包含变更的同步翻译预览。")
            return

        reply = QMessageBox.question(
            self,
            "确认写回同步翻译",
            "即将重新校验项目、源文件和预览制品，然后修改项目脚本。\n\n"
            f"预览清单：{manifest_path}\n\n"
            "如果脚本在预览后发生变化，写回会自动拒绝。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._clear_log_view()
        self._show_workbench_log_drawer()
        workflow = SyncTranslationWorkflow.apply_existing(manifest_path)
        self._begin_translation_workflow(
            workflow,
            log_heading="正在写回同步翻译预览",
            status_tab=1,
        )

    def _on_apply_revision(self) -> None:
        spec = work_mode_spec(self._current_work_mode())
        if not self._uses_revision_writeback(spec.mode):
            QMessageBox.information(
                self,
                "当前模式不支持",
                "「写回订正」仅适用于订正相关模式。",
            )
            return

        summary = self._current_writeback_summary()
        if not summary.can_apply:
            QMessageBox.information(
                self,
                "当前不能写回订正",
                summary.message or "请先完成订正预览并确认有可写回项。",
            )
            return

        confirm_lines = [
            "即将把订正写回游戏脚本。",
            "订正会修改现有译文行；写回前请确认已在副本或备份上验证。",
            "",
            *summary.facts,
        ]
        reply = QMessageBox.question(
            self,
            "确认写回订正",
            "\n".join(confirm_lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        manifest_path = summary.manifest_path or self._writeback_manifest_path
        if not manifest_path:
            QMessageBox.information(self, "无法写回订正", "没有可写回的订正任务记录。")
            return

        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._active_command = "apply_revision"
        self._apply_revision_output_lines = []
        self._focus_workbench_status_tab(2)
        self._set_writeback_summary(
            running_writeback_summary(
                manifest_path=manifest_path,
                heading="正在写回订正",
                message="正在写回订正；完成后这里会显示写回摘要。",
            )
        )
        self._set_task_running(True)

        command_label = f"gemini_translate_batch.py apply-revisions {manifest_path}"
        args = ["apply-revisions", manifest_path]

        self._append_log(f"=== 正在写回订正：{command_label} ===\n")
        self.runner.run(self.state.get_batch_script_path(), args)

    def _update_revision_writeback_from_preview(
        self,
        output: str,
        exit_code: int,
        manifest_path: str,
    ) -> None:
        resolved_manifest_path = manifest_path or self._manifest_path_from_sync_revision_output(
            output
        )
        already_applied = False
        if resolved_manifest_path:
            try:
                loaded = self.state.load_manifest_file(resolved_manifest_path)
                already_applied = bool(loaded.get("revision_applied_at"))
            except ValueError:
                pass
        summary = summarize_revision_writeback_from_preview_output(
            output,
            exit_code,
            manifest_path=resolved_manifest_path,
            already_applied=already_applied,
        )
        self._set_writeback_summary(summary)
        self._refresh_diagnostics_context()
        if summary.status not in {"idle", "running", "stale"}:
            self._focus_workbench_status_tab(2)

    @staticmethod
    def _manifest_path_from_sync_revision_output(output: str) -> str:
        import re
        from pathlib import Path

        match = re.search(r"^Sync revision run:\s*(.+?)\s*$", output, re.MULTILINE)
        if not match:
            return ""
        return str(Path(match.group(1).strip()) / "manifest.json")

    # --- Runner callbacks ---

    def _on_cli_line_ready(self, text: str):
        if self._active_command == "doctor":
            self._doctor_output_lines.append(text)
        elif self._active_command == "translation_workflow":
            self._workflow_step_output_lines.append(text)
            self._workflow_progress = update_workflow_progress_from_line(
                text,
                self._workflow_progress,
            )
            self._schedule_progress_ui_flush()
        elif self._active_command == "apply":
            self._apply_output_lines.append(text)
        elif self._active_command == "recheck":
            self._recheck_output_lines.append(text)
        elif self._active_command == "probe":
            self._probe_output_lines.append(text)
        elif self._active_command == "compare_variants":
            self._compare_variants_output_lines.append(text)
        elif self._active_command == "split":
            self._split_output_lines.append(text)
        elif self._active_command == "repair":
            self._repair_output_lines.append(text)
        elif self._active_command == "apply_revision":
            self._apply_revision_output_lines.append(text)
        elif self._active_command == "build_retry":
            self._build_retry_output_lines.append(text)
        elif self._active_command in {"bootstrap_rag", "bootstrap_source_index"}:
            self._bootstrap_output_lines.append(text)
            if (
                self._active_command == "bootstrap_source_index"
                and self._bootstrap_progress is not None
            ):
                self._bootstrap_progress = update_bootstrap_progress_from_line(
                    text,
                    self._bootstrap_progress,
                )
                self._schedule_progress_ui_flush()
            elif self._active_command == "bootstrap_rag":
                self._workflow_progress = update_workflow_progress_from_line(
                    text,
                    self._workflow_progress,
                )
                self._schedule_progress_ui_flush()
        elif self._active_command == "bootstrap_work":
            self._work_bootstrap_output_lines.append(text)
            self._workflow_progress = update_workflow_progress_from_line(
                text,
                self._workflow_progress,
            )
            self._schedule_progress_ui_flush()
        elif self._active_command == "generate_template":
            self._template_generation_output_lines.append(text)
        self._append_log(text)

    def _schedule_progress_ui_flush(self) -> None:
        self._workflow_progress_dirty = True
        timer = getattr(self, "_progress_flush_timer", None)
        if timer is None:
            self._flush_throttled_progress_ui()
            return
        if not timer.isActive():
            timer.start()

    def _flush_throttled_progress_ui(self) -> None:
        if not self._workflow_progress_dirty:
            return
        self._workflow_progress_dirty = False
        if self._active_command == "bootstrap_source_index":
            self._apply_bootstrap_progress_ui()
        elif self._active_command in {"bootstrap_rag", "bootstrap_work", "translation_workflow"}:
            if self._active_command == "bootstrap_rag":
                self._apply_bootstrap_progress_ui()
            else:
                self._apply_workflow_progress_ui()

    def _clear_log_view(self) -> None:
        pending = getattr(self, "_pending_log_lines", None)
        if pending is not None:
            pending.clear()
        timer = getattr(self, "_log_flush_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        if hasattr(self, "log_view"):
            self.log_view.clear()

    def _append_log(self, text: str) -> None:
        line = text.rstrip("\n")
        if not line:
            return
        pending = getattr(self, "_pending_log_lines", None)
        timer = getattr(self, "_log_flush_timer", None)
        if pending is None or timer is None:
            if hasattr(self, "log_view"):
                self.log_view.append(line)
                self._scroll_log_views_to_end()
            return
        pending.append(line)
        if not timer.isActive():
            timer.start()

    def _flush_pending_log_lines(self) -> None:
        if not self._pending_log_lines or not hasattr(self, "log_view"):
            self._pending_log_lines = []
            return
        lines = self._pending_log_lines
        self._pending_log_lines = []
        self.log_view.append("\n".join(lines))
        self._scroll_log_views_to_end()

    def _set_task_running(self, running: bool):
        was_running = bool(getattr(self, "_task_running", False))
        self._task_running = running
        # Update the stop action before recomputing resume availability. The
        # availability check treats an enabled stop action as an active task;
        # leaving the old running state here would keep resume/status disabled
        # until another UI refresh (or an application restart).
        self.kill_btn.setEnabled(running)
        if hasattr(self, "kill_btn"):
            self.kill_btn.setText(self._task_stop_button_label() if running else "停止")
        spec = work_mode_spec(self._current_work_mode())
        project_switch_enabled = not running
        if hasattr(self, "select_btn"):
            self.select_btn.setEnabled(project_switch_enabled)
        if hasattr(self, "global_browse_project_btn"):
            self.global_browse_project_btn.setEnabled(project_switch_enabled)
        if hasattr(self, "global_switch_project_btn"):
            self.global_switch_project_btn.setEnabled(project_switch_enabled)
        # Use __dict__ so task-running gates never force-build 设置 · 项目列表.
        panel = self.__dict__.get("_games_registry_panel")
        if panel is not None:
            # Browse/filter stay usable; only switch/mutate paths are gated.
            set_gate = getattr(panel, "set_host_task_running", None)
            if callable(set_gate):
                set_gate(running)
            else:
                panel.setEnabled(not running)
        go_workspace = self._settings_widget("settings_go_workspace_btn")
        if go_workspace is not None:
            # Settings entry stays openable; workspace switch is gated in-panel.
            go_workspace.setEnabled(True)
        # Doctor does not freeze shell/context browsing; keep nav usable.
        if hasattr(self, "workbench_nav"):
            self.workbench_nav.setEnabled(not self._context_switching_locked())
        self._set_shell_nav_task_lock(running)
        self._sync_doctor_prep_button_chrome()
        self._sync_bootstrap_prep_button_chrome()
        api_btn = self._settings_widget("api_btn")
        if api_btn is not None:
            api_btn.setEnabled(not running)
        bootstrap_ready = self._bootstrap_task_ready(spec)
        self.translate_btn.setEnabled(
            self._translate_button_enabled(
                spec=spec,
                bootstrap_ready=bootstrap_ready,
                running=running,
            )
        )
        resume_available = (
            (False, "任务运行中。") if running else self._resume_task_available()
        )
        self._update_resume_btn_enabled(
            running=running,
            resume_available=resume_available,
        )
        self._update_split_submit_btn(running=running)
        self._sync_settings_action_bar_enabled(task_running=running)
        writeback_summary = self._current_writeback_summary()
        self.apply_btn.setEnabled(
            not running
            and writeback_summary.can_apply
            and not self._uses_revision_writeback()
        )
        if hasattr(self, "apply_revision_btn"):
            self.apply_revision_btn.setEnabled(
                not running
                and self._uses_revision_writeback()
                and writeback_summary.can_apply
            )
        self._update_writeback_action_buttons(writeback_summary, running=running)
        self._update_probe_btn_enabled(running=running)
        self._update_compare_variants_btn_enabled(running=running)
        self._update_keyword_merge_btn_enabled(running=running)
        self._update_split_btn_enabled(running=running)
        # Task pages only show stop chrome for page-owned jobs, not doctor.
        page_running = self._task_page_running_chrome()
        self._sync_sync_translation_page_controls(running=page_running)
        self._sync_batch_translation_page_controls(running=page_running)
        self._sync_keywords_page_controls(running=page_running)
        self._sync_revision_page_controls(running=page_running)
        # Context-library prebuild CTAs stay on-page while nav is locked; gate them here.
        if hasattr(self, "context_library_panel"):
            self._refresh_context_library_panel(running=page_running)
        self._sync_task_shortcuts()
        self._reflow_button_bars()
        self._sync_workbench_empty_states(resume_available=resume_available)
        self._refresh_shell_status()
        # After a workbench/diagnostics task finishes, ease splitter back to idle.
        if was_running and not running:
            self._restore_diagnostics_splitter_idle()

    def _sync_task_shortcuts(self) -> None:
        """Keep task shortcuts aligned with the corresponding action buttons."""
        if hasattr(self, "_doctor_shortcut"):
            self._doctor_shortcut.setEnabled(self.doctor_btn.isEnabled())
        if hasattr(self, "_translate_shortcut"):
            self._translate_shortcut.setEnabled(self.translate_btn.isEnabled())
        if hasattr(self, "_kill_shortcut"):
            self._kill_shortcut.setEnabled(self.kill_btn.isEnabled())
        if hasattr(self, "_save_config_shortcut"):
            self._save_config_shortcut.setEnabled(self.save_config_btn.isEnabled())

    def _set_workflow_summary(
        self,
        status: str,
        heading: str,
        message: str,
        facts: list[str] | None = None,
    ):
        self._workflow_heading_text = heading
        self.workflow_status_label.set_status(status, heading)
        self.workflow_message_label.setText(message)
        self.workflow_facts_label.setText("\n".join(facts or []))
        self._update_resume_btn_text()
        resume_available = self._resume_task_available()
        self._update_resume_btn_enabled(resume_available=resume_available)
        self._sync_batch_translation_page_controls()
        self._sync_keywords_page_controls()
        self._sync_revision_page_controls()
        self._sync_workbench_empty_states(resume_available=resume_available)
        self._sync_layout_sizes()

    def _set_workflow_update(self, update: WorkflowUpdate):
        override_step_key = update.timeline_step_key
        if self._workflow is not None:
            current_step = self._workflow.current_step()
            step_key = override_step_key or (current_step.key if current_step else self.timeline.current_step_key)
            self.timeline.set_current_step(step_key, update.status)
        else:
            self.timeline.set_current_step(override_step_key or self.timeline.current_step_key, update.status)
        self._set_workflow_summary(
            update.status,
            update.heading,
            update.message,
            update.facts,
        )

    def _run_workflow_current_step(self):
        if self._workflow is not None:
            step = self._workflow.current_step()
            if step is not None:
                self.timeline.setVisible(True)
                self.timeline.set_current_step(step.key, "running")
        if self._workflow is None:
            self._set_task_running(False)
            return

        step = self._workflow.current_step()
        if step is None:
            self._set_task_running(False)
            self._active_command = ""
            self._workflow = None
            self._clear_workflow_progress_ui()
            return

        facts = []
        if self._workflow.manifest_path:
            facts.append(format_manifest_path_fact(self._workflow.manifest_path))
        self._workflow_progress_base_facts = list(facts)
        progress_kind = self._workflow_progress_kind_for_step(step)
        self._workflow_progress = (
            create_workflow_progress_state(progress_kind)
            if progress_kind
            else None
        )
        self._workflow_step_output_lines = []
        self._set_workflow_summary("running", step.heading, step.message, facts)
        self._apply_workflow_progress_ui()
        if step.script_basename == "gemini_translate.py":
            script_path = self.state.get_sync_script_path()
        else:
            script_path = self.state.get_batch_script_path()
        args_text = " ".join(step.args)
        command_label = f"{step.script_basename} {args_text}".strip()
        self._append_log(f"\n=== {step.heading}：{command_label} ===\n")
        self.runner.run(script_path, step.args)

    def _current_writeback_summary(self) -> WritebackSummary:
        return getattr(self, "_writeback_summary", idle_writeback_summary())

    def _set_writeback_summary(self, summary: WritebackSummary) -> None:
        self._writeback_summary = summary
        self._writeback_manifest_path = summary.manifest_path
        self.writeback_status_label.set_status(summary.status, summary.heading)
        self.writeback_message_label.setText(summary.message)
        self.writeback_facts_label.setText("\n".join(summary.facts))
        self._set_details_label(self.writeback_details_label, summary.findings)
        self._update_writeback_action_buttons(
            summary,
            running=self.kill_btn.isEnabled(),
        )
        self._sync_batch_translation_page_controls()
        self._sync_keywords_page_controls()
        self._sync_revision_page_controls()
        if not self.kill_btn.isEnabled():
            self.apply_btn.setEnabled(
                summary.can_apply and not self._uses_revision_writeback()
            )
            if hasattr(self, "apply_revision_btn"):
                self.apply_revision_btn.setEnabled(
                    self._uses_revision_writeback() and summary.can_apply
                )
        self._refresh_shell_status()
        self._sync_layout_sizes()

    def _update_writeback_from_check(
        self,
        output: str,
        exit_code: int,
        manifest_path: str,
    ) -> None:
        already_applied = False
        if manifest_path:
            try:
                manifest = self.state.load_manifest_file(manifest_path)
                already_applied = bool(manifest.get("applied_at"))
            except ValueError:
                pass
        summary = summarize_check_output(
            output,
            exit_code,
            manifest_path=manifest_path,
            already_applied=already_applied,
        )
        self._set_writeback_summary(summary)
        self._refresh_diagnostics_context()
        if summary.status not in {"idle", "running", "stale"}:
            self._focus_workbench_status_tab(2)

    def _set_doctor_summary(
        self,
        summary: DoctorSummary,
        *,
        resume_available: tuple[bool, str] | None = None,
    ) -> None:
        self._doctor_summary_mode = summary.mode
        self._doctor_summary_status = summary.status
        self.doctor_status_label.set_status(summary.status, summary.heading)
        self.doctor_message_label.setText(summary.message)
        self.doctor_facts_label.setText("\n".join(summary.facts))
        self._set_doctor_detail_facts(summary.detail_facts)
        self._update_translate_button_label()
        spec = work_mode_spec(self._current_work_mode())
        running = self.kill_btn.isEnabled()
        bootstrap_ready = self._bootstrap_task_ready(spec)
        self.translate_btn.setEnabled(
            self._translate_button_enabled(
                spec=spec,
                bootstrap_ready=bootstrap_ready,
                running=running,
            )
        )
        self._sync_sync_translation_page_controls(running=running)
        self._sync_batch_translation_page_controls(running=running)
        self._sync_keywords_page_controls(running=running)
        self._sync_revision_page_controls(running=running)
        self._sync_task_shortcuts()
        # When resume_available is provided (including cold-start False), skip
        # _resume_task_available's manifest history walk.
        self._sync_workbench_empty_states(resume_available=resume_available)
        self._sync_layout_sizes()

    def _on_runner_error(self, message: str):
        self._append_log(message)
        if self._active_command == "doctor":
            self._doctor_output_lines.append(message)
        elif self._active_command == "translation_workflow":
            self._workflow_step_output_lines.append(message)
        elif self._active_command == "apply":
            self._apply_output_lines.append(message)
        elif self._active_command == "recheck":
            self._recheck_output_lines.append(message)
        elif self._active_command == "probe":
            self._probe_output_lines.append(message)
        elif self._active_command == "compare_variants":
            self._compare_variants_output_lines.append(message)
            self._cleanup_compare_variants_temp_file()
        elif self._active_command == "split":
            self._split_output_lines.append(message)
        elif self._active_command == "repair":
            self._repair_output_lines.append(message)
        elif self._active_command == "build_retry":
            self._build_retry_output_lines.append(message)
        elif self._active_command in {"bootstrap_rag", "bootstrap_source_index"}:
            self._bootstrap_output_lines.append(message)
        elif self._active_command == "bootstrap_work":
            self._work_bootstrap_output_lines.append(message)
        elif self._active_command == "generate_template":
            self._template_generation_output_lines.append(message)
        self._reveal_log_for_active_context()
        self.statusBar().showMessage("任务运行失败，请查看运行日志。", 6000)

    def _on_finished(self, exit_code: int):
        self._append_log(f"\n[进程已结束，退出码：{exit_code}]")
        if self._active_command == "translation_workflow":
            self._on_workflow_step_finished(exit_code)
            return

        if self._active_command == "apply":
            self._invalidate_manifest_caches(self._writeback_manifest_path or None)
            summary = summarize_apply_output(
                "\n".join(self._apply_output_lines),
                exit_code,
                manifest_path=self._writeback_manifest_path,
            )
            self._set_writeback_summary(summary)
            self._refresh_diagnostics_context()
            if exit_code == 0:
                self._refresh_workflow_from_latest_manifest()
                self._handle_post_apply_registry_sync()
            self._active_command = ""
            self._set_task_running(False)
            if exit_code == 0:
                self.statusBar().showMessage("翻译写回完成。", 6000)
            else:
                self.statusBar().showMessage("翻译写回失败，请查看诊断日志。", 8000)
            return

        if self._active_command == "recheck":
            manifest_path = self._writeback_manifest_path
            self._invalidate_manifest_caches(manifest_path or None)
            self._update_writeback_from_check(
                "\n".join(self._recheck_output_lines),
                exit_code,
                manifest_path,
            )
            self._active_command = ""
            self._set_task_running(False)
            if exit_code == 0:
                self._refresh_workflow_from_latest_manifest()
            if exit_code != 0:
                self.statusBar().showMessage("重新检查失败，请查看诊断日志。", 8000)
            else:
                summary = self._current_writeback_summary()
                if summary.status == "safe":
                    self.statusBar().showMessage("重新检查完成，当前可写回。", 6000)
                elif summary.status == "warn":
                    self.statusBar().showMessage("重新检查完成，仍需处理问题。", 6000)
                elif summary.status == "block":
                    self.statusBar().showMessage("重新检查完成，当前禁止写回。", 6000)
                else:
                    self.statusBar().showMessage("重新检查完成。", 6000)
            return

        if self._active_command == "probe":
            manifest_path, manifest = self._current_diagnostics_manifest()
            probe_summary = summarize_probe_output(
                "\n".join(self._probe_output_lines),
                exit_code,
                manifest_path=manifest_path,
            )
            base_context = build_diagnostics_context(
                latest_manifest_path=manifest_path or None,
                manifest=manifest,
                batch_script_path=str(self.state.get_batch_script_path()),
                logs_dir=str(self.state.get_logs_dir()),
                python_exe=sys.executable,
                submit_max_cost=self._submit_max_cost_from_config(),
            )
            self._set_diagnostics_context(
                probe_summary_to_diagnostics_context(probe_summary, base_context)
            )
            self._active_command = ""
            self._set_task_running(False)
            if probe_summary.status == "ok":
                self.statusBar().showMessage("样本试跑通过。", 6000)
            elif probe_summary.status == "warn":
                self.statusBar().showMessage("样本试跑完成，但需关注部分结果。", 6000)
            elif probe_summary.status == "failed":
                self.statusBar().showMessage("样本试跑失败，请查看诊断日志。", 8000)
            else:
                self.statusBar().showMessage("样本试跑已结束。", 6000)
            return

        if self._active_command == "compare_variants":
            manifest_path, manifest = self._current_diagnostics_manifest()
            ab_summary = summarize_compare_variants_output(
                "\n".join(self._compare_variants_output_lines),
                exit_code,
                manifest_path=manifest_path,
                variant_names=self._compare_variants_names,
            )
            base_context = build_diagnostics_context(
                latest_manifest_path=manifest_path or None,
                manifest=manifest,
                batch_script_path=str(self.state.get_batch_script_path()),
                logs_dir=str(self.state.get_logs_dir()),
                python_exe=sys.executable,
                submit_max_cost=self._submit_max_cost_from_config(),
            )
            self._set_diagnostics_context(
                ab_experiment_summary_to_diagnostics_context(ab_summary, base_context)
            )
            self._active_command = ""
            self._set_task_running(False)
            if ab_summary.status == "ok":
                if ab_summary.dry_run:
                    self.statusBar().showMessage("翻译 A/B 试跑完成。", 6000)
                else:
                    self.statusBar().showMessage("翻译 A/B 对比完成。", 6000)
            elif ab_summary.status == "failed":
                self.statusBar().showMessage("翻译 A/B 对比失败，请查看诊断日志。", 8000)
            elif ab_summary.status == "warn":
                self.statusBar().showMessage("翻译 A/B 对比完成，但需关注部分结果。", 6000)
            else:
                self.statusBar().showMessage("翻译 A/B 对比已结束。", 6000)
            self._cleanup_compare_variants_temp_file()
            return

        if self._active_command == "split":
            manifest_path, _manifest = self._current_diagnostics_manifest()
            split_summary = summarize_split_output(
                "\n".join(self._split_output_lines),
                exit_code,
                manifest_path=manifest_path,
            )
            latest_manifest_path = split_summary.latest_manifest_path or manifest_path
            context_manifest_path = (
                split_summary.source_manifest_path or manifest_path
            )
            context_manifest = self._load_diagnostics_manifest(
                context_manifest_path or None
            )
            base_context = build_diagnostics_context(
                latest_manifest_path=latest_manifest_path or None,
                manifest=context_manifest,
                batch_script_path=str(self.state.get_batch_script_path()),
                logs_dir=str(self.state.get_logs_dir()),
                python_exe=sys.executable,
                submit_max_cost=self._submit_max_cost_from_config(),
            )
            self._set_diagnostics_context(
                split_summary_to_diagnostics_context(split_summary, base_context)
            )
            self._active_command = ""
            self._set_task_running(False)
            if split_summary.status == "ok":
                latest_manifest = self._load_diagnostics_manifest(
                    latest_manifest_path or None
                )
                self._refresh_workflow_from_latest_manifest(
                    latest_manifest=latest_manifest_path or None,
                    manifest=latest_manifest,
                )
                self._refresh_split_status_ui(
                    manifest_path=context_manifest_path or manifest_path,
                    manifest=context_manifest,
                )
                self.statusBar().showMessage(
                    f"拆分完成，已生成 {len(split_summary.child_manifest_paths or [])} 个子包。",
                    8000,
                )
            elif split_summary.status == "unchanged":
                self.statusBar().showMessage("当前翻译包无需拆分。", 6000)
            elif split_summary.status == "failed":
                self.statusBar().showMessage("拆分翻译包失败，请查看诊断日志。", 8000)
            else:
                self.statusBar().showMessage("拆分翻译包已结束。", 6000)
            return

        if self._active_command == "repair":
            manifest_path = self._writeback_manifest_path
            manifest = self._load_diagnostics_manifest(manifest_path or None)
            output_text = "\n".join(self._repair_output_lines)
            parsed_output = parse_repair_output(output_text)
            report_path = parsed_output.get("report_path")
            parsed = summarize_repair_output(
                output_text,
                exit_code,
                report_path=report_path if isinstance(report_path, str) else "",
                manifest_path=manifest_path or "",
            )
            base_context = build_diagnostics_context(
                latest_manifest_path=manifest_path or None,
                manifest=manifest,
                batch_script_path=str(self.state.get_batch_script_path()),
                logs_dir=str(self.state.get_logs_dir()),
                python_exe=sys.executable,
                submit_max_cost=self._submit_max_cost_from_config(),
            )
            self._set_diagnostics_context(
                repair_summary_to_diagnostics_context(parsed, base_context)
            )
            self._active_command = ""
            self._set_task_running(False)
            if parsed.status == "ok":
                self.statusBar().showMessage(
                    "同步修补完成；建议点击「重新检查」。",
                    8000,
                )
                QMessageBox.information(
                    self,
                    "同步修补完成",
                    "修补已写入翻译文件。请在「批量翻译 · 结果」的「问题处理」中"
                    "点击「重新检查」更新检查结果，"
                    "显示「可写回」后再决定是否写入项目。",
                )
            elif parsed.status == "warn":
                self.statusBar().showMessage(
                    "同步修补部分完成；请查看摘要并重新检查。",
                    8000,
                )
            elif parsed.status == "failed":
                self.statusBar().showMessage("同步修补失败，请查看诊断日志。", 8000)
            else:
                self.statusBar().showMessage("同步修补已结束。", 6000)
            return

        if self._active_command == "apply_revision":
            summary = summarize_revision_apply_output(
                "\n".join(self._apply_revision_output_lines),
                exit_code,
                manifest_path=self._writeback_manifest_path,
            )
            self._set_writeback_summary(summary)
            self._refresh_diagnostics_context()
            self._active_command = ""
            self._set_task_running(False)
            if exit_code == 0:
                self.statusBar().showMessage("订正写回完成。", 6000)
            else:
                self.statusBar().showMessage("订正写回失败，请查看诊断日志。", 8000)
            return

        if self._active_command == "build_retry":
            result = parse_build_retry_output(
                "\n".join(self._build_retry_output_lines),
                exit_code,
            )
            self._active_command = ""
            self._set_task_running(False)
            self._refresh_diagnostics_context()
            if result.status == "ok":
                parent_manifest = self._load_writeback_manifest()
                retry_path = ""
                if parent_manifest is not None:
                    retry_path = existing_retry_manifest_path(parent_manifest)
                if not retry_path:
                    retry_path = result.retry_manifest_path
                preview_result = "cancelled"
                if retry_path:
                    preview_result = self._show_retry_preview(
                        retry_path,
                        start_followup_on_confirm=True,
                    )
                if preview_result == "cancelled":
                    self.statusBar().showMessage("补译包已生成，请先确认预览范围。", 6000)
            else:
                QMessageBox.warning(self, result.heading, result.message)
                self.statusBar().showMessage(result.heading, 6000)
            return

        if self._active_command == "bootstrap_rag":
            summary = summarize_rag_bootstrap_output(
                "\n".join(self._bootstrap_output_lines),
                exit_code,
            )
            self._set_workflow_from_bootstrap_summary(summary)
            self._active_command = ""
            self._set_task_running(False)
            if exit_code == 0 and summary.status == "ready":
                self.statusBar().showMessage("预建记忆库完成。", 6000)
            elif exit_code == 0:
                self.statusBar().showMessage("预建记忆库已结束，请查看摘要。", 6000)
            else:
                self.statusBar().showMessage("预建记忆库失败，请查看诊断日志。", 8000)
            return

        if self._active_command == "bootstrap_source_index":
            summary = summarize_source_index_bootstrap_output(
                "\n".join(self._bootstrap_output_lines),
                exit_code,
            )
            self._set_workflow_from_bootstrap_summary(summary)
            self._active_command = ""
            self._set_task_running(False)
            if exit_code == 0 and summary.status == "ready":
                self.statusBar().showMessage("原文索引预建完成。", 6000)
            elif exit_code == 0:
                self.statusBar().showMessage("原文索引预建已结束，请查看摘要。", 6000)
            else:
                self.statusBar().showMessage("原文索引预建失败，请查看诊断日志。", 8000)
            return

        if self._active_command == "bootstrap_work":
            summary = summarize_work_bootstrap_output(
                "\n".join(self._work_bootstrap_output_lines),
                exit_code,
            )
            game_root_update_failed = False
            if summary.status == "ready" and summary.work_dir:
                if not self._switch_game_root(summary.work_dir):
                    game_root_update_failed = True
                    summary = with_game_root_persist_warning(summary)
            self._set_doctor_summary(work_bootstrap_to_doctor_summary(summary))
            self._set_workflow_summary(
                summary.status,
                summary.heading,
                summary.message,
                summary.facts,
            )
            self._clear_workflow_progress_ui()
            self._active_command = ""
            self._set_task_running(False)
            if game_root_update_failed:
                self.statusBar().showMessage(
                    "工作目录已复制，但项目路径未保存，请查看诊断日志。",
                    8000,
                )
            elif exit_code == 0 and summary.status == "ready":
                self.statusBar().showMessage("工作目录已准备完成。", 6000)
            elif exit_code == 0:
                self.statusBar().showMessage("准备工作目录已结束，请查看摘要。", 6000)
            else:
                self.statusBar().showMessage("准备工作目录失败，请查看诊断日志。", 8000)
            return

        if self._active_command == "generate_template":
            summary = summarize_template_generation_output(
                "\n".join(self._template_generation_output_lines),
                exit_code,
            )
            self._set_doctor_summary(template_generation_to_doctor_summary(summary))
            self._set_workflow_summary(
                summary.status,
                summary.heading,
                summary.message,
                summary.facts,
            )
            self._active_command = ""
            self._set_task_running(False)
            if exit_code == 0 and summary.status == "ready":
                self.statusBar().showMessage(
                    "翻译模板已生成，可以开始翻译。",
                    6000,
                )
            elif exit_code == 0:
                self.statusBar().showMessage("翻译模板生成已结束，请查看摘要。", 6000)
            else:
                self.statusBar().showMessage("翻译模板生成失败，请查看诊断日志。", 8000)
            return

        self._set_task_running(False)

    def _copy_keyword_report_dir_to_game_parent(self, report_dir: Path | str | None) -> None:
        if not report_dir:
            return
        game_root = self.state.get_game_root()
        if not game_root:
            return

        target_dir = game_root.parent / "extracted_keywords"
        source_dir = Path(report_dir)

        files_to_copy = [
            "keyword_candidates.jsonl",
            "keyword_candidates.md",
            "keyword_chunk_summaries.jsonl",
            "keyword_chunk_summaries.md",
        ]

        copied_files = []
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            for filename in files_to_copy:
                src = source_dir / filename
                if src.exists():
                    dest = target_dir / filename
                    shutil.copy2(src, dest)
                    copied_files.append(filename)
            if copied_files:
                self._append_log(
                    f"\n[系统] 已将关键词提取报告复制一份至：{target_dir}\n"
                    f"复制的文件：{', '.join(copied_files)}\n"
                )
        except Exception as e:
            self._append_log(f"\n[系统警告] 复制关键词提取报告到游戏上级目录失败：{e}\n")

    def _copy_keyword_reports_to_game_parent(self, manifest_path: str) -> None:
        if not manifest_path:
            return
        self._copy_keyword_report_dir_to_game_parent(Path(manifest_path).parent)

    def _copy_sync_keyword_reports_to_game_parent(self, output: str) -> None:
        match = re.search(r"^Sync keyword run:\s*(.+?)\s*$", output, re.MULTILINE)
        if not match:
            return
        self._copy_keyword_report_dir_to_game_parent(match.group(1).strip())

    def _on_workflow_step_finished(self, exit_code: int):
        if self._workflow is None:
            self._active_command = ""
            self._set_task_running(False)
            return

        step_output = "\n".join(self._workflow_step_output_lines)
        manifest_path = self._workflow.manifest_path
        self._invalidate_manifest_caches(manifest_path or None)

        current_step = self._workflow.current_step()
        step_key = current_step.key if current_step else ""

        restore_latest_manifest_path = getattr(self._workflow, "restore_latest_manifest_path", "")
        update = self._workflow.complete_current_step(exit_code, step_output)

        is_sync_translation_workflow = isinstance(self._workflow, SyncTranslationWorkflow)
        if is_sync_translation_workflow and step_key == "preview" and exit_code == 0:
            preview_count_match = re.search(r"^Preview files:\s*(\d+)\s*$", step_output, re.MULTILINE)
            preview_count = int(preview_count_match.group(1)) if preview_count_match else 0
            self.sync_translation_page.set_preview_ready(
                self._workflow.manifest_path if preview_count > 0 else ""
            )
        elif is_sync_translation_workflow and step_key == "apply" and update.status == "done":
            self.sync_translation_page.clear_preview()

        if restore_latest_manifest_path and not update.should_continue:
            try:
                self.state.remember_latest_manifest_path(restore_latest_manifest_path)
            except ValueError as exc:
                self._append_log(f"[GUI] 无法恢复最新任务指针：{exc}")

        keyword_export_completed = step_key == "export-keywords" and exit_code == 0
        if keyword_export_completed:
            self._copy_keyword_reports_to_game_parent(manifest_path)
            exported_manifest = None
            state = getattr(self, "state", None)
            if state is not None:
                try:
                    exported_manifest = state.load_manifest_file(manifest_path)
                except ValueError:
                    exported_manifest = None
            exported_candidates = keyword_merge_candidates_path_from_manifest(
                manifest_path,
                exported_manifest,
            )
            if exported_candidates:
                self._keyword_merge_candidates_path = exported_candidates
        sync_keyword_completed = step_key == "sync-keywords" and exit_code == 0
        if sync_keyword_completed:
            self._copy_sync_keyword_reports_to_game_parent(step_output)
            sync_candidates = keyword_merge_candidates_path_from_sync_output(step_output)
            if sync_candidates:
                self._keyword_merge_candidates_path = sync_candidates

        retry_parent = getattr(self._workflow, "retry_parent_manifest_path", "")
        if "Safety status:" in step_output and (
            step_key == "check-parent" or not retry_parent
        ):
            self._update_writeback_from_check(step_output, exit_code, manifest_path)
        if self._uses_revision_writeback() and (
            "Preview JSONL:" in step_output or "Preview Markdown:" in step_output
        ):
            self._update_revision_writeback_from_preview(
                step_output,
                exit_code,
                manifest_path,
            )
        archive_completed = not update.should_continue and update.status == "done"
        if not archive_completed:
            self._set_workflow_update(update)
        self._clear_workflow_progress_ui()
        self._workflow_step_output_lines = []
        self._refresh_diagnostics_context()
        if restore_latest_manifest_path:
            self._refresh_split_status_ui(manifest_path=restore_latest_manifest_path)

        if update.should_continue:
            self.statusBar().showMessage(update.heading, 3000)
            QTimer.singleShot(0, self._run_workflow_current_step)
            return

        self._active_command = ""
        self._workflow = None
        self._set_task_running(False)
        finish_spec = work_mode_spec(self._current_work_mode())
        if (
            not finish_spec.supports_translation_writeback
            and finish_spec.mode not in self._revision_writeback_modes()
        ):
            if finish_spec.mode == WorkMode.KEYWORD_EXTRACTION and keyword_export_completed:
                self._refresh_writeback_from_latest_manifest()
            else:
                self._set_writeback_summary(
                    idle_writeback_summary_for_work_mode(finish_spec.mode)
                )
        if update.status == "failed":
            self.statusBar().showMessage("翻译任务失败，请查看诊断日志。", 8000)
        elif update.status == "waiting":
            self.statusBar().showMessage("批量任务仍在处理，可稍后继续最新任务。", 8000)
        elif archive_completed:
            self._viewing_completed_manifest = False
            self._refresh_workflow_from_latest_manifest()
            self.statusBar().showMessage(update.heading, 6000)
        else:
            self.statusBar().showMessage("翻译任务流程完成。", 6000)

    # --- Config loading/saving helpers ---

    def _config_section(self, config: dict[str, Any], key: str) -> dict[str, Any]:
        section = config.get(key)
        return section if isinstance(section, dict) else {}

    def _ensure_config_section(self, config: dict[str, Any], key: str) -> dict[str, Any]:
        section = config.get(key)
        if not isinstance(section, dict):
            section = {}
            config[key] = section
        return section

    def _config_string(self, value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _supports_batch_thinking(self, model_name: Any) -> bool:
        return self._config_string(model_name).startswith("gemini-3")

    def _sync_models_for_save(
        self,
        existing_models: Any,
        selected_model: str,
    ) -> list[str] | None:
        existing: list[str] = []
        if isinstance(existing_models, list):
            for model in existing_models:
                cleaned = self._config_string(model)
                if cleaned and cleaned not in existing:
                    existing.append(cleaned)
        else:
            cleaned = self._config_string(existing_models)
            if cleaned:
                existing.append(cleaned)

        if not selected_model:
            return existing or None

        return [
            selected_model,
            *[model for model in existing if model != selected_model],
        ]

    def _batch_thinking_value_for_load(
        self,
        batch_config: dict[str, Any],
        batch_model: Any,
    ) -> str:
        if "thinking_level" in batch_config:
            return self._config_string(batch_config.get("thinking_level", ""))
        return "minimal" if self._supports_batch_thinking(batch_model) else ""

    def _batch_thinking_value_for_model_change(
        self,
        batch_model: Any,
        current_thinking_level: Any,
        config_has_key: bool,
        user_changed: bool,
    ) -> str | None:
        if (
            self._supports_batch_thinking(batch_model)
            and not self._config_string(current_thinking_level)
            and not config_has_key
            and not user_changed
        ):
            return "minimal"
        return None

    def _should_save_batch_thinking_level(
        self,
        batch_config: dict[str, Any],
        batch_model: str,
        thinking_level: str,
        user_changed: bool,
    ) -> bool:
        return (
            bool(thinking_level)
            or (self._supports_batch_thinking(batch_model) and user_changed)
            or "thinking_level" in batch_config
        )

    def _combo_item_texts(self, combo: NoWheelComboBox | None) -> list[str]:
        if combo is None:
            return []
        items: list[str] = []
        count = getattr(combo, "count", None)
        item_text = getattr(combo, "itemText", None)
        if callable(count) and callable(item_text):
            for index in range(int(count())):
                text = self._config_string(item_text(index))
                if text and text not in items:
                    items.append(text)
        current = self._config_string(combo.currentText())
        if current and current not in items:
            items.append(current)
        return items

    def _repopulate_model_combo(
        self,
        combo: NoWheelComboBox | None,
        models: list[str],
        selected: str,
    ) -> None:
        if combo is None:
            return
        selected = self._config_string(selected)
        block_signals = getattr(combo, "blockSignals", None)
        previous_block = block_signals(True) if callable(block_signals) else False
        try:
            clear = getattr(combo, "clear", None)
            if callable(clear):
                clear()
            add_items = getattr(combo, "addItems", None)
            if models and callable(add_items):
                add_items(models)
            if selected:
                self._set_combo_value(combo, selected)
            else:
                count = getattr(combo, "count", None)
                if callable(count) and int(count()) > 0:
                    combo.setCurrentIndex(0)
                else:
                    combo.setCurrentIndex(-1)
        finally:
            if callable(block_signals):
                block_signals(previous_block)

    def _set_combo_value(self, combo: NoWheelComboBox, value: Any):
        value = self._config_string(value)
        if not value:
            combo.setCurrentIndex(-1)
            return
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.addItem(value)
            count = getattr(combo, "count", None)
            if callable(count):
                combo.setCurrentIndex(int(count()) - 1)
        is_editable = getattr(combo, "isEditable", None)
        line_edit = getattr(combo, "lineEdit", None)
        if callable(is_editable) and is_editable() and callable(line_edit):
            editor = line_edit()
            if editor is not None and hasattr(editor, "setText"):
                editor.setText(value)

    def _set_theme_combo_value(self, value: str) -> None:
        # Use __dict__ lookup so _load_config_to_ui / project switch never force
        # construction of every config settings page via theme_combo getattr.
        combo = self._settings_widget("theme_combo")
        if combo is None:
            return
        theme = normalize_theme_preference(value)
        idx = combo.findData(theme)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _set_batch_thinking_value(self, value: str):
        idx = self.batch_thinking_combo.findData(value)
        self._updating_batch_thinking_combo = True
        try:
            if idx >= 0:
                self.batch_thinking_combo.setCurrentIndex(idx)
            elif value:
                self.batch_thinking_combo.addItem(f"{value} (自定义)", value)
                self.batch_thinking_combo.setCurrentIndex(self.batch_thinking_combo.count() - 1)
            else:
                self.batch_thinking_combo.setCurrentIndex(0)
        finally:
            self._updating_batch_thinking_combo = False

    def _load_theme_to_ui(self, config: dict[str, Any]) -> None:
        theme = read_gui_theme_from_config(config)
        self._theme_preference = theme
        self._loading_theme_to_ui = True
        try:
            self._set_theme_combo_value(theme)
        finally:
            self._loading_theme_to_ui = False
        self._apply_theme()

    def _apply_theme(self) -> None:
        qt_app = getattr(self, "_qt_app", None)
        if qt_app is None:
            return
        try:
            resources_dir = getattr(self, "_resources_dir", Path(__file__).resolve().parent / "resources")
            apply_theme(qt_app, resources_dir, self._theme_preference)
            self._refresh_action_icons()
            if hasattr(self, "_log_highlighter"):
                self._log_highlighter.update_theme(self._effective_theme_is_dark())
            QTimer.singleShot(0, self._refresh_split_status_table_after_theme_change)
        except OSError as exc:
            self.statusBar().showMessage("主题样式加载失败，已保留当前样式。", 6000)
            self._append_log(f"加载主题样式失败：{exc}")

    def _set_theme_preference(self, preference: str, *, persist: bool) -> None:
        self._theme_preference = normalize_theme_preference(preference)
        self._apply_theme()
        if not persist:
            return
        try:
            config = self.state.load_translator_config()
            write_gui_theme_to_config(config, self._theme_preference)
            self.state.save_translator_config(config)
            self._append_log(f"外观主题已保存：{self._theme_preference}。")
        except Exception as exc:
            QMessageBox.warning(self, "保存主题失败", str(exc))
            self._append_log(f"保存主题失败：{exc}")

    def _on_theme_changed(self, _index: int) -> None:
        if self._loading_config_to_ui or self._loading_theme_to_ui:
            return
        preference = self.theme_combo.currentData()
        if not isinstance(preference, str):
            return
        self._set_theme_preference(preference, persist=False)

    def _on_system_color_scheme_changed(self, _scheme: object) -> None:
        if self._theme_preference != THEME_SYSTEM:
            return
        self._apply_theme()

    def _settings_widget(self, name: str) -> Any:
        """Return a settings widget only if its page is already built.

        Does not trigger lazy materialization (unlike normal attribute access).
        """
        return self.__dict__.get(name)

    def _load_config_to_ui(self, *, refresh_task_gates: bool = True) -> None:
        """Push translator_config into settings widgets that already exist.

        When ``refresh_task_gates`` is False (cold start), skip probe/resume
        readiness walks; ``_deferred_startup_refresh`` applies them after show.
        Missing lazy settings widgets are skipped — call
        ``_ensure_settings_pages_for_config()`` first for a full UI sync.
        """
        self._loading_config_to_ui = True
        try:
            config = self.state.load_translator_config()
            self._load_theme_to_ui(config)
            # Nothing else to fill until config-bearing pages exist.
            # Use __dict__ so we do not trigger lazy materialization here.
            if (
                self._settings_widget("rag_enabled_cb") is None
                and self._settings_widget("batch_model_combo") is None
            ):
                return
            sync_config = self._config_section(config, "sync")
            batch_config = self._config_section(config, "batch")
            sync_rag_config = self._config_section(sync_config, "rag")
            batch_rag_config = self._config_section(batch_config, "rag")
            context_flags = read_batch_context_flags(
                config,
                game_root=self._game_root_str_for_flags(),
            )
            rag_cb = self._settings_widget("rag_enabled_cb")
            if rag_cb is not None:
                rag_cb.setChecked(context_flags["rag_enabled"])
            source_cb = self._settings_widget("source_index_enabled_cb")
            if source_cb is not None:
                source_cb.setChecked(context_flags["source_index_enabled"])
            bootstrap_cb = self._settings_widget("bootstrap_on_build_cb")
            if bootstrap_cb is not None:
                bootstrap_cb.setChecked(context_flags["bootstrap_on_build"])
            storage_config = self._config_section(config, "context_storage")
            storage_location = normalize_context_storage_location(
                storage_config.get("location", config.get("context_storage_location", ""))
            )
            storage_cb = self._settings_widget("context_storage_game_cb")
            if storage_cb is not None:
                storage_cb.setChecked(storage_location == "game")
            self._batch_thinking_config_has_key = "thinking_level" in batch_config

            sync_backend = self._config_string(sync_config.get("backend", "gemini")).lower()
            if sync_backend not in {"gemini", "litellm"}:
                sync_backend = "gemini"
            backend_combo = self._settings_widget("sync_backend_combo")
            backend_idx = backend_combo.findData(sync_backend) if backend_combo is not None else -1
            if backend_combo is not None:
                backend_combo.setCurrentIndex(backend_idx)

            backend_models = read_sync_backend_models(
                sync_config,
                sync_backend,
                str(BASIC_RECOMMENDED_VALUES["sync_model"]),
            )
            batch_val = self._config_string(batch_config.get("model", "")) or str(
                BASIC_RECOMMENDED_VALUES["batch_model"]
            )
            sync_emb_val = self._config_string(sync_rag_config.get("embedding_model", "")) or str(
                BASIC_RECOMMENDED_VALUES["sync_embedding_model"]
            )
            batch_emb_val = self._config_string(batch_rag_config.get("embedding_model", "")) or str(
                BASIC_RECOMMENDED_VALUES["batch_embedding_model"]
            )
            translation_models = resolve_gemini_translation_models(
                config,
                extra_selected=[backend_models.gemini_model, batch_val],
            )
            embedding_models = resolve_gemini_embedding_models(
                config,
                extra_selected=[sync_emb_val, batch_emb_val],
            )
            sync_model = self._settings_widget("sync_model_combo")
            self._repopulate_model_combo(
                sync_model,
                translation_models,
                backend_models.gemini_model,
            )
            litellm_combo = self._settings_widget("litellm_model_combo")
            if litellm_combo is not None:
                self._set_combo_value(litellm_combo, backend_models.litellm_model)
            if backend_combo is not None:
                self._on_sync_backend_changed(backend_idx)

            batch_model = self._settings_widget("batch_model_combo")
            self._repopulate_model_combo(batch_model, translation_models, batch_val)

            sync_emb = self._settings_widget("sync_embedding_combo")
            self._repopulate_model_combo(sync_emb, embedding_models, sync_emb_val)

            batch_emb = self._settings_widget("batch_embedding_combo")
            self._repopulate_model_combo(batch_emb, embedding_models, batch_emb_val)

            if batch_model is not None:
                self._on_batch_model_changed(batch_val)
                thinking_val = self._batch_thinking_value_for_load(batch_config, batch_val)
                self._set_batch_thinking_value(thinking_val)
            advanced_values = read_advanced_settings(config)
            get_game_root = getattr(self.state, "get_game_root", None)
            current_game_root = get_game_root() if callable(get_game_root) else None
            if current_game_root and not self._config_string(advanced_values.get("game_root")):
                advanced_values["game_root"] = str(current_game_root)
            if self.__dict__.get("_advanced_setting_widgets"):
                self._load_advanced_settings_to_ui(advanced_values)
                self._clear_advanced_setting_errors()
        finally:
            self._batch_thinking_user_changed = False
            self._loading_config_to_ui = False
        if (
            self._settings_widget("rag_enabled_cb") is not None
            or self._settings_widget("batch_model_combo") is not None
        ):
            self._config_ui_saved_snapshot = self._current_config_ui_snapshot()
        if refresh_task_gates and "translate_btn" in self.__dict__:
            self._set_task_running(bool(getattr(self, "_task_running", False)))

    def _on_batch_model_changed(self, text: str):
        is_thinking_supported = self._supports_batch_thinking(text)
        self.batch_thinking_combo.setEnabled(is_thinking_supported)
        if not is_thinking_supported:
            self._set_batch_thinking_value("")
            return

        default_value = self._batch_thinking_value_for_model_change(
            text,
            self.batch_thinking_combo.currentData(),
            self._batch_thinking_config_has_key,
            self._batch_thinking_user_changed,
        )
        if default_value is not None and not self._loading_config_to_ui:
            self._set_batch_thinking_value(default_value)

    def _on_batch_thinking_changed(self, _index: int):
        if not self._loading_config_to_ui and not self._updating_batch_thinking_combo:
            self._batch_thinking_user_changed = True

    def _on_save_config(self) -> bool:
        if not self.state.get_game_root():
            QMessageBox.information(self, "未选择项目", "请先选择游戏的 work 目录。")
            return False
        self._ensure_settings_pages_for_config()

        try:
            config = self.state.load_translator_config()
            original_config = copy.deepcopy(config)
            sync_config = self._ensure_config_section(config, "sync")
            batch_config = self._ensure_config_section(config, "batch")
            sync_rag_config = self._ensure_config_section(sync_config, "rag")
            batch_rag_config = self._ensure_config_section(batch_config, "rag")
            batch_source_index_config = self._ensure_config_section(batch_config, "source_index")
            context_storage_config = self._ensure_config_section(config, "context_storage")

            context_storage_config["location"] = "game" if self.context_storage_game_cb.isChecked() else "tool"
            context_storage_config["game_dir_name"] = (
                self._config_string(
                    context_storage_config.get(
                        "game_dir_name",
                        context_storage_config.get(
                            "directory_name",
                            context_storage_config.get("directory", "translation_context"),
                        ),
                    )
                )
                or "translation_context"
            )
            # RAG / source-index / bootstrap_on_build are per-project only.
            # Do not write them into the global translator_config.json.
            project_context_flags = {
                "rag_enabled": self.rag_enabled_cb.isChecked(),
                "source_index_enabled": self.source_index_enabled_cb.isChecked(),
                "bootstrap_on_build": self.bootstrap_on_build_cb.isChecked(),
            }

            sync_backend = self._selected_sync_backend()
            litellm_model = self._litellm_model_text()
            if sync_backend == "litellm" and not litellm_model:
                self._focus_settings_section("litellm")
                QMessageBox.information(
                    self,
                    "请配置 LiteLLM 模型",
                    "启用 LiteLLM 前，请填写带 provider 前缀的模型名称。",
                )
                return False
            sync_model = write_sync_backend_models(
                sync_config,
                sync_backend,
                self.sync_model_combo.currentText(),
                litellm_model,
            )
            if "models" in sync_config:
                sync_models = self._sync_models_for_save(sync_config.get("models"), sync_model)
                if sync_models:
                    sync_config["models"] = sync_models
                else:
                    sync_config.pop("models", None)
            batch_model = self.batch_model_combo.currentText().strip()
            batch_config["model"] = batch_model
            sync_embedding_model = self.sync_embedding_combo.currentText().strip()
            batch_embedding_model = self.batch_embedding_combo.currentText().strip()
            sync_rag_config["embedding_model"] = sync_embedding_model
            batch_rag_config["embedding_model"] = batch_embedding_model
            thinking_val = self.batch_thinking_combo.currentData()
            thinking_level = thinking_val if isinstance(thinking_val, str) else ""
            if self._should_save_batch_thinking_level(
                batch_config,
                batch_model,
                thinking_level,
                self._batch_thinking_user_changed,
            ):
                batch_config["thinking_level"] = thinking_level

            write_gui_theme_to_config(config, self._current_theme_preference_from_ui())

            advanced_values = self._advanced_settings_values_from_ui()
            if advanced_values:
                complete_advanced_values = read_advanced_settings(config)
                complete_advanced_values.update(advanced_values)
                current_game_root = self.state.get_game_root()
                if current_game_root is not None:
                    complete_advanced_values["game_root"] = str(current_game_root)
                errors = validate_advanced_settings(complete_advanced_values)
                if errors:
                    self._show_advanced_setting_errors(errors)
                    self._show_settings_status("高级设置有无效字段，未保存。", 6000)
                    return False
                self._clear_advanced_setting_errors()
                apply_advanced_settings(config, complete_advanced_values)
                # Persist only non-builtin catalog extensions; drop empty keys.
                write_model_catalog_extras(
                    config,
                    translation_models=list(
                        complete_advanced_values.get("catalog_gemini_models") or []
                    ),
                    embedding_models=list(
                        complete_advanced_values.get("catalog_gemini_embedding_models") or []
                    ),
                )

            # Validate every field before either settings file is written. This
            # avoids persisting project flags when the UI reports "未保存".
            from project_context_settings import save_project_context_settings

            # Write the global file first. If the project write then fails,
            # restore the original global config so the two files commit together.
            self.state.save_translator_config(config)
            try:
                project_settings_path = save_project_context_settings(
                    self.state.get_game_root(),
                    project_context_flags,
                )
            except Exception:
                try:
                    self.state.save_translator_config(original_config)
                except Exception as rollback_exc:
                    self._append_log(
                        f"全局设置回滚失败，请检查 translator_config.json：{rollback_exc}"
                    )
                raise
            self._append_log(
                f"当前项目上下文开关已保存：{project_settings_path}"
            )
            if not self._sync_state_game_root_from_settings(config.get("game_root")):
                self._show_settings_status("设置已保存，但同步项目目录到工作台失败。", 6000)
                self._append_log("设置已保存，但同步项目目录到工作台失败。")
                self._config_ui_saved_snapshot = self._current_config_ui_snapshot()
                return True
            if work_mode_spec(self._current_work_mode()).is_bootstrap:
                self._apply_work_mode_ui(refresh_manifest_writeback=False)
            self._append_log(
                "设置已成功保存（全局项 → translator_config.json；"
                "RAG/原文索引 → 当前项目 project_context_settings.json）。"
            )
            # Refresh select-only model dropdowns / rotation checklist from catalog.
            try:
                self._load_config_to_ui(refresh_task_gates=False)
            except Exception as refresh_exc:
                self._append_log(f"保存后刷新模型列表失败：{refresh_exc}")
            try:
                ToastNotification.show_toast(self, "设置已成功保存")
            except Exception as exc:
                self._append_log(f"提示通知显示失败：{exc}")
                self.statusBar().showMessage("设置已成功保存", 3000)
            self._config_ui_saved_snapshot = self._current_config_ui_snapshot()
            return True
        except Exception as exc:
            QMessageBox.warning(self, "保存设置失败", str(exc))
            self._append_log(f"保存设置失败：{exc}")
            return False


    def _sync_state_game_root_from_settings(self, value: object) -> bool:
        if not isinstance(value, str) or not value.strip():
            return True
        try:
            new_root, _adjusted = self.state.normalize_game_root(value)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "同步项目目录失败", str(exc))
            self._append_log(f"同步设置页 game_root 到当前窗口状态失败: {exc}")
            return False

        current_root = self.state.get_game_root()
        if current_root is not None:
            try:
                current_normalized, _ = self.state.normalize_game_root(current_root)
                if canonical_abs_path(str(current_normalized)).lower() == canonical_abs_path(str(new_root)).lower():
                    self._refresh_project_label()
                    return True
            except (TypeError, ValueError):
                pass

        return self._switch_game_root(value.strip())


def run_app(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    app = QApplication(argv)
    app.setApplicationName("Ren'Py Translation Lab")
    app.setApplicationVersion(__version__)
    resources_dir = Path(__file__).resolve().parent / "resources"
    project_state = ProjectState()
    try:
        bootstrap_config = project_state.load_translator_config()
        theme_preference = read_gui_theme_from_config(bootstrap_config)
    except Exception as exc:
        print(f"警告：无法读取主题配置，将使用系统跟随：{exc}")
        theme_preference = DEFAULT_THEME_PREFERENCE

    # Build the widget tree *before* applying the app stylesheet. Under a large
    # QSS, every QStackedWidget/QBoxLayout.addWidget pays style-polish cost and
    # dominated cold startup (~0.5s). One setStyleSheet after construction is
    # cheaper and matches the first painted theme.
    win = MainWindow(
        qt_app=app,
        resources_dir=resources_dir,
        project_state=project_state,
    )
    win._theme_preference = normalize_theme_preference(theme_preference)
    try:
        apply_theme(app, resources_dir, theme_preference)
    except OSError as exc:
        print(f"警告：无法加载 GUI 样式表：{exc}")

    app.styleHints().colorSchemeChanged.connect(win._on_system_color_scheme_changed)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
