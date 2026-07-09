"""Main GUI application for the optional workbench.

This is the first version shell (per #42):
- Pure PySide6 with tabbed layout (workbench / config / diagnostics)
- Delegates everything to the existing CLI via QProcess
- Workbench tab: project selection, doctor + translation workflow status
"""
from __future__ import annotations

import os
import sys
import re
import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QKeySequence, QPalette, QShortcut
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
    QStackedWidget,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QRadioButton,
    QButtonGroup,
)

from .path_utils import canonical_abs_path, normalize_context_storage_location
from .responsive_layout import FlowButtonBar, ResponsiveActionPanel
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
from .games_registry_actions import handle_post_apply_registry_update
from .games_registry_panel import GamesRegistryPanel
from .games_registry_doctor_compare import compare_registry_with_doctor_report
from .games_registry_view import resolve_workspace_root
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
from .project_state import ProjectState
from .theme import apply_theme, system_prefers_dark
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
    BASIC_RECOMMENDED_VALUES,
    SettingField,
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
    TASK_CATEGORY_ORDER,
    TaskCategory,
    WORKBENCH_NAV_ORDER,
    WorkMode,
    WorkbenchNavItem,
    bootstrap_disabled_message,
    default_work_mode_for_category,
    default_work_mode_for_nav,
    normalize_task_category,
    normalize_work_mode,
    task_category_for_work_mode,
    task_category_spec,
    work_mode_hint_texts,
    work_mode_spec,
    work_mode_submode_label,
    work_modes_for_category,
    workbench_nav_for_work_mode,
    workbench_nav_spec,
)
from .workbench_session import WorkbenchModeSession
from .batch_workflow_support import resolve_submit_max_cost
from .workflow_factory import create_workflow, resume_workflow
from .workflow_progress import (
    WorkflowProgressState,
    create_workflow_progress_state,
    update_workflow_progress_from_line,
)
from .widget_helpers import NoWheelComboBox, NoWheelTabWidget
from .wizard_timeline import WizardTimeline
from .log_highlighter import LogHighlighter
from .status_icons import StatusBadge
from .toast_widget import ToastNotification

# Diagnostics splitter: idle favors task context; running tasks expand the log.
_DIAGNOSTICS_IDLE_CONTEXT_PX = 420
_DIAGNOSTICS_IDLE_LOG_PX = 180
_DIAGNOSTICS_RUNNING_CONTEXT_RATIO = 0.32

# Workbench bottom log drawer (P0a / issue #158).
_WORKBENCH_LOG_DRAWER_EXPANDED_HEIGHT = 180
_WORKBENCH_LOG_DRAWER_HEADER_HEIGHT = 40

# Batch translation stages (P1b / issue #161) — maps to workbench_status_tabs indices.
_BATCH_STAGE_PREPARE = 0
_BATCH_STAGE_EXECUTE = 1
_BATCH_STAGE_RESULT = 2
_BATCH_STAGE_LABELS = ("准备", "执行", "结果")
_BATCH_STAGE_HINTS = (
    "环境检查与项目准备",
    "打包、提交、下载与进度",
    "安全检查与写回",
)

_LOG_FLUSH_INTERVAL_MS = 80
_LAYOUT_SYNC_DEBOUNCE_MS = 32
_UI_PROGRESS_FLUSH_INTERVAL_MS = 100


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
        self.resize(960, 760)

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
        self._last_main_tab_index = 0
        self._handling_config_tab_leave = False
        self._task_running = False
        self._games_registry_panel: GamesRegistryPanel | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        header = QLabel("Ren'Py Translation Lab · 图形工作台")
        header.setObjectName("header_label")
        root_layout.addWidget(header)

        root_layout.addWidget(self._build_global_project_bar())

        self.tab_widget = NoWheelTabWidget()
        self.tab_widget.setObjectName("main_tabs")
        root_layout.addWidget(self.tab_widget, 1)

        self._build_workbench_tab()
        self._build_config_tab()
        self._build_log_tab()

        self.batch_model_combo.currentTextChanged.connect(self._on_batch_model_changed)
        self.batch_thinking_combo.currentIndexChanged.connect(self._on_batch_thinking_changed)
        self._last_main_tab_index = self.tab_widget.currentIndex()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Connect runner
        self.runner.line_ready.connect(self._on_cli_line_ready)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_runner_error)

        self._refresh_project_label()
        self._show_pending_game_root_redirect_notice()
        self._refresh_api_status()
        self._load_config_to_ui()
        self._set_doctor_summary(idle_summary())
        QTimer.singleShot(0, self._deferred_startup_refresh)
        QTimer.singleShot(0, self._sync_work_mode_hint_height)

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Status
        self.statusBar().showMessage(
            "图形界面是可选组件；核心命令行不受影响。"
        )

    def _setup_shortcuts(self) -> None:
        """Bind global keyboard shortcuts and update button tooltips."""
        self._doctor_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        self._doctor_shortcut.activated.connect(self._on_run_doctor)

        self._translate_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        self._translate_shortcut.activated.connect(self._on_start_translation)

        self._kill_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self._kill_shortcut.activated.connect(self._on_kill)

        clear_log_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        clear_log_shortcut.activated.connect(self._on_clear_log)

        # Config save — only active when config tab is shown
        self._save_config_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self._save_config_shortcut.activated.connect(self._shortcut_save_config)

        # Tab switching
        for i in range(min(3, self.tab_widget.count())):
            sc = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda idx=i: self.tab_widget.setCurrentIndex(idx))

        # Button tooltips with shortcut hints
        if hasattr(self, "doctor_btn"):
            self.doctor_btn.setToolTip("环境检查 (Ctrl+D)")
        if hasattr(self, "translate_btn"):
            self.translate_btn.setToolTip("开始翻译 (Ctrl+T)")
        if hasattr(self, "kill_btn"):
            self.kill_btn.setToolTip("停止任务 (Ctrl+K)")
        if hasattr(self, "clear_log_btn"):
            self.clear_log_btn.setToolTip("清空日志 (Ctrl+L)")
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

    def _build_global_project_bar(self) -> QFrame:
        """Always-visible project path + switch entries (GUI IA P0b / #159)."""
        bar = QFrame()
        bar.setObjectName("global_project_bar")
        self.global_project_bar = bar
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(6)

        path_row = QHBoxLayout()
        path_row.setSpacing(10)
        title = QLabel("项目：")
        title.setObjectName("global_project_bar_label")
        path_row.addWidget(title)

        self.global_project_path_edit = QLineEdit("尚未选择项目")
        self.global_project_path_edit.setReadOnly(True)
        self.global_project_path_edit.setObjectName("global_project_path_edit")
        path_row.addWidget(self.global_project_path_edit, 1)
        # Keep legacy objectName for mono-font QSS / tests that still look up project_path_edit.
        self.project_path_edit = self.global_project_path_edit
        outer.addLayout(path_row)

        # Buttons wrap under the path on narrow windows instead of colliding.
        self.global_project_actions = FlowButtonBar(spacing=8)
        self.global_project_actions.setObjectName("global_project_actions")
        self.global_switch_project_btn = QPushButton("切换项目")
        self.global_switch_project_btn.setObjectName("secondary_btn")
        self.global_switch_project_btn.setToolTip(
            "打开设置 → 工作区，从项目总表选择并切换当前 game_root。"
        )
        self.global_switch_project_btn.clicked.connect(self._on_global_switch_project)
        self.global_project_actions.add_widget(self.global_switch_project_btn, min_width=88)

        self.global_browse_project_btn = QPushButton("指定本地目录…")
        self.global_browse_project_btn.setObjectName("secondary_btn")
        self.global_browse_project_btn.setToolTip(
            "通过文件夹对话框指定本地路径（可与总表无关）；会立即写入 game_root。"
        )
        self.global_browse_project_btn.clicked.connect(self._on_select_project)
        self.global_project_actions.add_widget(self.global_browse_project_btn, min_width=120)
        # Alias for existing enable/disable paths that still reference select_btn.
        self.select_btn = self.global_browse_project_btn
        self.global_project_actions.finish_setup()
        outer.addWidget(self.global_project_actions)

        return bar

    def _build_workbench_tab(self) -> None:
        tab = QWidget()
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

        right = QWidget()
        right.setObjectName("workbench_content")
        layout = QVBoxLayout(right)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(14)
        outer.addWidget(right, 1)

        # Stack keeps page identity for each nav item (shared chrome below).
        # Keep pages empty/minimal so they do not compete with action buttons for space.
        self.workbench_stack = QStackedWidget()
        self.workbench_stack.setObjectName("workbench_stack")
        self._workbench_stack_pages: dict[WorkbenchNavItem, QWidget] = {}
        for nav_item in WORKBENCH_NAV_ORDER:
            page = QWidget()
            page.setObjectName(f"workbench_page_{nav_item.value}")
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(0)
            self._workbench_stack_pages[nav_item] = page
            self.workbench_stack.addWidget(page)
        self.workbench_stack.setFixedHeight(0)
        self.workbench_stack.setVisible(False)
        layout.addWidget(self.workbench_stack)

        # Project path lives on the global bar; keep redirect notice only on workbench.
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

        # Submode (batch|sync / rag|index) only for multi-mode nav items.
        submode_row = QHBoxLayout()
        submode_row.setSpacing(10)
        self.work_submode_label = QLabel("模式：")
        submode_row.addWidget(self.work_submode_label)
        self.work_submode_combo = NoWheelComboBox()
        self.work_submode_combo.setObjectName("work_submode_combo")
        self.work_submode_combo.currentIndexChanged.connect(self._on_work_submode_changed)
        submode_row.addWidget(self.work_submode_combo, 1)
        mode_outer.addLayout(submode_row)
        self._work_submode_row_widgets = (self.work_submode_label, self.work_submode_combo)

        # Legacy combos kept for residual helpers/tests — not placed in the layout
        # so they cannot reserve vertical space or collide with action buttons.
        self.task_category_combo = NoWheelComboBox(mode_frame)
        self.task_category_combo.setObjectName("task_category_combo")
        self.task_category_combo.hide()
        self.work_task_combo = NoWheelComboBox(mode_frame)
        self.work_task_combo.setObjectName("work_task_combo")
        self.work_task_combo.hide()

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
            "警告：同步翻译可能直接修改项目文件，请先备份或在副本上试跑。"
        )
        self.sync_mode_warning.setObjectName("sync_mode_warning")
        self.sync_mode_warning.setWordWrap(True)
        self.sync_mode_warning.setVisible(False)
        mode_outer.addWidget(self.sync_mode_warning)
        layout.addWidget(mode_frame)

        # Context library dual status cards (P1c / #162 · §3.2.5).
        layout.addWidget(self._build_context_library_panel())

        action_frame = QFrame()
        action_frame.setObjectName("action_frame")
        self._action_frame = action_frame
        action_outer = QVBoxLayout(action_frame)
        action_outer.setContentsMargins(12, 10, 12, 10)
        action_outer.setSpacing(8)

        # Left nav narrows the content column; stack action rows before buttons clip.
        self.action_panel = ResponsiveActionPanel(compact_width=640)
        self.translate_group_label = self.action_panel.translate_label
        self.doctor_btn = self.action_panel.add_prep_button(QPushButton("环境检查"))
        self.doctor_btn.setObjectName("secondary_btn")
        self.doctor_btn.clicked.connect(self._on_run_doctor)
        self.bootstrap_work_btn = self.action_panel.add_prep_button(QPushButton("准备工作目录"))
        self.bootstrap_work_btn.setObjectName("secondary_btn")
        self.bootstrap_work_btn.clicked.connect(self._on_bootstrap_work)
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
        layout.addWidget(action_frame)

        self.timeline = WizardTimeline()
        self.timeline.setObjectName("workbench_timeline")
        self.timeline.setVisible(False)
        layout.addWidget(self.timeline)

        layout.addWidget(self._build_batch_stage_bar())

        self.workbench_status_tabs = NoWheelTabWidget()
        self.workbench_status_tabs.setObjectName("workbench_status_tabs")
        self.workbench_status_tabs.currentChanged.connect(self._on_workbench_status_tab_changed)

        doctor_tab = QWidget()
        doctor_tab.setObjectName("workbench_doctor_page")
        self._style_themed_surface(doctor_tab)
        doctor_layout = QVBoxLayout(doctor_tab)
        doctor_layout.setContentsMargins(12, 12, 12, 12)
        doctor_layout.setSpacing(6)
        self.doctor_status_label = StatusBadge("doctor_status_label")
        doctor_layout.addWidget(self.doctor_status_label)
        doctor_scroll = QScrollArea()
        doctor_scroll.setObjectName("doctor_summary_scroll")
        self._style_themed_surface(doctor_scroll)
        doctor_scroll.setWidgetResizable(True)
        doctor_scroll.setFrameShape(QFrame.Shape.NoFrame)
        doctor_viewport = doctor_scroll.viewport()
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
        self.doctor_details_label = QLabel()
        self.doctor_details_label.setWordWrap(True)
        self.doctor_details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.doctor_details_label.setObjectName("config_hint_label")
        self.doctor_details_label.setVisible(False)
        doctor_content_layout.addWidget(self.doctor_details_label)
        doctor_content_layout.addStretch()
        doctor_scroll.setWidget(doctor_content)
        doctor_layout.addWidget(doctor_scroll, 1)
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
        writeback_layout.setSpacing(6)
        self.writeback_status_label = StatusBadge("writeback_status_label")
        writeback_layout.addWidget(self.writeback_status_label)
        writeback_scroll = QScrollArea()
        writeback_scroll.setObjectName("writeback_summary_scroll")
        self._style_themed_surface(writeback_scroll)
        writeback_scroll.setWidgetResizable(True)
        writeback_scroll.setFrameShape(QFrame.Shape.NoFrame)
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

        self.writeback_issues_panel = FlowButtonBar(spacing=8, row_spacing=6)
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

        self.workbench_status_tabs.setCurrentIndex(_BATCH_STAGE_EXECUTE)
        layout.addWidget(self.workbench_status_tabs, 1)

        layout.addWidget(self._build_workbench_log_drawer())

        self.tab_widget.addTab(tab, "工作台")
        self._sync_batch_stage_chrome()

    def _build_context_library_panel(self) -> QFrame:
        """RAG / source-index status cards for the 上下文库 nav page (P1c / #162)."""
        frame = QFrame()
        frame.setObjectName("context_library_panel")
        self.context_library_panel = frame
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(10)

        title = QLabel("上下文库状态")
        title.setObjectName("diagnostics_section_label")
        outer.addWidget(title)

        rag_row = QHBoxLayout()
        rag_row.setSpacing(8)
        self.context_rag_status_label = QLabel("记忆库：—")
        self.context_rag_status_label.setObjectName("summary_body_label")
        self.context_rag_status_label.setWordWrap(True)
        rag_row.addWidget(self.context_rag_status_label, 1)
        self.context_bootstrap_rag_btn = QPushButton("预建记忆库")
        self.context_bootstrap_rag_btn.setObjectName("secondary_btn")
        self.context_bootstrap_rag_btn.clicked.connect(
            lambda: self._on_context_bootstrap_clicked("rag")
        )
        rag_row.addWidget(self.context_bootstrap_rag_btn, 0)
        outer.addLayout(rag_row)

        idx_row = QHBoxLayout()
        idx_row.setSpacing(8)
        self.context_source_index_status_label = QLabel("原文索引：—")
        self.context_source_index_status_label.setObjectName("summary_body_label")
        self.context_source_index_status_label.setWordWrap(True)
        idx_row.addWidget(self.context_source_index_status_label, 1)
        self.context_bootstrap_source_index_btn = QPushButton("预建原文索引")
        self.context_bootstrap_source_index_btn.setObjectName("secondary_btn")
        self.context_bootstrap_source_index_btn.clicked.connect(
            lambda: self._on_context_bootstrap_clicked("source_index")
        )
        idx_row.addWidget(self.context_bootstrap_source_index_btn, 0)
        outer.addLayout(idx_row)

        self.context_open_settings_btn = QPushButton("打开设置 · 上下文")
        self.context_open_settings_btn.setObjectName("secondary_btn")
        self.context_open_settings_btn.setToolTip(
            "开关须在设置中保存后才能预建；打开设置的「上下文」分区。"
        )
        self.context_open_settings_btn.clicked.connect(self._on_open_context_settings)
        outer.addWidget(self.context_open_settings_btn, 0, Qt.AlignmentFlag.AlignLeft)

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
        if not hasattr(self, "context_library_panel"):
            return
        flags = self._saved_batch_context_flags()
        rag_on = bool(flags.get("rag_enabled"))
        idx_on = bool(flags.get("source_index_enabled"))
        game_root = self.state.get_game_root() if hasattr(self, "state") else None
        root_hint = str(game_root) if game_root else "未选择项目"
        if hasattr(self, "context_rag_status_label"):
            self.context_rag_status_label.setText(
                f"记忆库：{'已启用' if rag_on else '未启用'} · 项目 {root_hint}"
                + ("" if rag_on else " · 请先在设置 · 上下文开启并保存")
            )
        if hasattr(self, "context_source_index_status_label"):
            self.context_source_index_status_label.setText(
                f"原文索引：{'已启用' if idx_on else '未启用'} · 项目 {root_hint}"
                + ("" if idx_on else " · 请先在设置 · 上下文开启并保存")
            )
        if running is None:
            running = bool(getattr(self, "_task_running", False)) or (
                hasattr(self, "kill_btn") and self.kill_btn.isEnabled()
            )
        if hasattr(self, "context_bootstrap_rag_btn"):
            self.context_bootstrap_rag_btn.setEnabled(not running and rag_on)
        if hasattr(self, "context_bootstrap_source_index_btn"):
            self.context_bootstrap_source_index_btn.setEnabled(not running and idx_on)

    def _apply_task_page_chrome(self, spec) -> None:
        """Show/hide per-nav chrome: sync warn, context cards, prep buttons (P1c)."""
        mode = spec.mode
        nav = workbench_nav_for_work_mode(mode)
        is_sync = mode == WorkMode.SYNC_TRANSLATION
        is_context = nav == WorkbenchNavItem.CONTEXT
        is_translation = mode in {
            WorkMode.BATCH_TRANSLATION,
            WorkMode.SYNC_TRANSLATION,
        }

        if hasattr(self, "sync_mode_warning"):
            self.sync_mode_warning.setVisible(is_sync)
        if hasattr(self, "context_library_panel"):
            self.context_library_panel.setVisible(is_context)
            if is_context:
                self._refresh_context_library_panel()

        if hasattr(self, "doctor_btn"):
            self.doctor_btn.setVisible(is_translation)
        if hasattr(self, "bootstrap_work_btn"):
            self.bootstrap_work_btn.setVisible(is_translation)

        if hasattr(self, "translate_btn"):
            self.translate_btn.setVisible(not is_context)
        if hasattr(self, "resume_btn") and is_context:
            self.resume_btn.setVisible(False)
        if hasattr(self, "action_panel"):
            reflow = getattr(self.action_panel, "reflow", None)
            if callable(reflow):
                reflow(force=True)

    def _build_batch_stage_bar(self) -> QFrame:
        """Step strip for batch translation: 准备 → 执行 → 结果 (P1b / #161)."""
        frame = QFrame()
        frame.setObjectName("batch_stage_bar")
        self.batch_stage_bar = frame
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._batch_stage_buttons: list[QPushButton] = []
        self._batch_stage_button_group = QButtonGroup(frame)
        self._batch_stage_button_group.setExclusive(True)
        for index, label in enumerate(_BATCH_STAGE_LABELS):
            btn = QPushButton(f"{index + 1}  {label}")
            btn.setObjectName("batch_stage_btn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(
                lambda _checked=False, idx=index: self._on_batch_stage_clicked(idx)
            )
            self._batch_stage_button_group.addButton(btn, index)
            row.addWidget(btn)
            self._batch_stage_buttons.append(btn)
            if index < len(_BATCH_STAGE_LABELS) - 1:
                sep = QLabel("→")
                sep.setObjectName("batch_stage_sep")
                row.addWidget(sep)
        row.addStretch(1)
        outer.addLayout(row)

        # Hint on its own row so stage buttons are never squeezed into each other.
        self.batch_stage_hint = QLabel("")
        self.batch_stage_hint.setObjectName("config_hint_label")
        self.batch_stage_hint.setWordWrap(True)
        outer.addWidget(self.batch_stage_hint)
        frame.setVisible(False)
        return frame

    def _build_workbench_log_drawer(self) -> QFrame:
        """Bottom collapsible log drawer for the workbench (shares log document with diagnostics)."""
        drawer = QFrame()
        drawer.setObjectName("workbench_log_drawer")
        self.workbench_log_drawer = drawer
        drawer_layout = QVBoxLayout(drawer)
        drawer_layout.setContentsMargins(10, 6, 10, 8)
        drawer_layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.workbench_log_drawer_title = QLabel("运行日志")
        self.workbench_log_drawer_title.setObjectName("workbench_log_drawer_title")
        header.addWidget(self.workbench_log_drawer_title)
        header.addStretch()

        self.workbench_log_open_diagnostics_btn = QPushButton("在诊断中打开")
        self.workbench_log_open_diagnostics_btn.setObjectName("secondary_btn")
        self.workbench_log_open_diagnostics_btn.setToolTip(
            "打开诊断页查看任务上下文、命令参考与完整日志布局。"
        )
        self.workbench_log_open_diagnostics_btn.clicked.connect(
            self._on_open_diagnostics_from_log_drawer
        )
        header.addWidget(self.workbench_log_open_diagnostics_btn)

        self.workbench_log_clear_btn = QPushButton("清空")
        self.workbench_log_clear_btn.setObjectName("secondary_btn")
        self.workbench_log_clear_btn.clicked.connect(self._on_clear_log)
        header.addWidget(self.workbench_log_clear_btn)

        self.workbench_log_toggle_btn = QPushButton("展开")
        self.workbench_log_toggle_btn.setObjectName("secondary_btn")
        self.workbench_log_toggle_btn.clicked.connect(self._toggle_workbench_log_drawer)
        header.addWidget(self.workbench_log_toggle_btn)
        drawer_layout.addLayout(header)

        self.workbench_log_body = QWidget()
        self.workbench_log_body.setObjectName("workbench_log_body")
        body_layout = QVBoxLayout(self.workbench_log_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        self.workbench_log_view = QTextEdit()
        self.workbench_log_view.setReadOnly(True)
        self.workbench_log_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.workbench_log_view.setObjectName("workbench_log_view")
        self.workbench_log_view.setMinimumHeight(100)
        body_layout.addWidget(self.workbench_log_view, 1)
        drawer_layout.addWidget(self.workbench_log_body, 1)

        self._workbench_log_drawer_expanded = False
        self._set_workbench_log_drawer_expanded(False)
        return drawer

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

        outer_layout = QHBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.settings_nav = QListWidget()
        self.settings_nav.setObjectName("settings_nav")
        self.settings_nav.setFixedWidth(150)
        self.settings_nav.setSpacing(2)
        self.settings_nav.setFrameShape(QFrame.Shape.NoFrame)
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

        pages = [
            ("workspace", "工作区", self._build_settings_workspace_page()),
            ("project", "项目", self._build_settings_project_page()),
            ("api_keys", "密钥", self._build_settings_api_keys_page()),
            ("models", "模型", self._build_settings_models_page()),
            ("context", "上下文", self._build_settings_context_page()),
            ("appearance", "外观", self._build_settings_appearance_page()),
            ("advanced", "高级", self._build_settings_advanced_page()),
        ]
        for index, (key, label, page) in enumerate(pages):
            self._settings_nav_rows[key] = index
            self.settings_nav.addItem(label)
            self.settings_stack.addWidget(page)
        self.settings_nav.currentRowChanged.connect(self._on_settings_nav_row_changed)

        action_bar = QFrame()
        action_bar.setObjectName("settings_action_bar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(12, 10, 12, 10)
        action_layout.setSpacing(8)
        action_layout.addStretch()
        self.reload_config_btn = QPushButton("重新加载")
        self.reload_config_btn.setObjectName("secondary_btn")
        self.reload_config_btn.clicked.connect(self._on_reload_config)
        action_layout.addWidget(self.reload_config_btn)
        self.restore_defaults_btn = QPushButton("恢复推荐值")
        self.restore_defaults_btn.setObjectName("secondary_btn")
        self.restore_defaults_btn.clicked.connect(self._on_restore_recommended_config)
        action_layout.addWidget(self.restore_defaults_btn)
        self.save_config_btn = QPushButton("保存设置")
        self.save_config_btn.setObjectName("save_config_btn")
        self.save_config_btn.clicked.connect(self._on_save_config)
        action_layout.addWidget(self.save_config_btn)
        right_layout.addWidget(action_bar)

        self.settings_nav.setCurrentRow(0)

        self.tab_widget.addTab(tab, "设置")

    def _settings_page(self, object_name: str) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setObjectName(f"{object_name}_scroll")
        self._style_themed_surface(scroll)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        viewport = scroll.viewport()
        viewport.setObjectName(f"{object_name}_viewport")
        self._style_themed_surface(viewport)

        content = QWidget()
        content.setObjectName(f"{object_name}_content")
        self._style_themed_surface(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        scroll.setWidget(content)
        return scroll, layout

    def _settings_group(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 16, 12, 12)
        return group, layout

    def _build_settings_workspace_page(self) -> QWidget:
        page, layout = self._settings_page("settings_workspace")
        hint = QLabel(
            "浏览和切换 Ren'Py 工作区内的游戏项目。"
            "当前 work 目录只能在这里切换；切换后会自动打开「项目」分区以调整术语表、准备流程等参数。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        self._games_registry_panel = GamesRegistryPanel(
            None,
            workspace_root=resolve_workspace_root(self.state.get_tool_root()),
            current_game_root=self.state.get_game_root(),
            get_doctor_report=self._current_registry_doctor_report,
            on_switch_project=self._on_registry_switch_project,
        )
        layout.addWidget(self._games_registry_panel, 1)
        return page

    def _build_settings_api_keys_page(self) -> QWidget:
        page, layout = self._settings_page("settings_api_keys")
        api_box, api_layout = self._settings_group("API Key")

        api_hint = QLabel(
            "翻译任务需要 Gemini API 密钥。密钥保存在本地配置文件中，"
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
            "切换 work 目录请前往「工作区」；此处只调整术语表、翻译目录、过滤器和准备流程。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        current_box = QGroupBox("当前项目")
        current_form = QFormLayout(current_box)
        current_form.setSpacing(8)
        current_form.setContentsMargins(12, 16, 12, 12)
        self.settings_project_root_value = QLabel("（尚未选择项目）")
        self.settings_project_root_value.setWordWrap(True)
        self.settings_project_root_value.setObjectName("settings_project_root_value")
        current_form.addRow("游戏 work 目录：", self.settings_project_root_value)
        switch_row = QHBoxLayout()
        switch_row.addStretch(1)
        self.settings_go_workspace_btn = QPushButton("在工作区切换…")
        self.settings_go_workspace_btn.setObjectName("secondary_btn")
        self.settings_go_workspace_btn.clicked.connect(self._on_go_to_workspace_for_project_switch)
        switch_row.addWidget(self.settings_go_workspace_btn)
        current_form.addRow("", switch_row)
        layout.addWidget(current_box)

        for group_title in ("项目与资源", "准备流程"):
            group = QGroupBox(group_title)
            form = QFormLayout(group)
            form.setSpacing(8)
            form.setContentsMargins(12, 16, 12, 12)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
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
            "启用后先保存设置，再到工作台「分析与准备」下运行预建子任务。"
            "记忆库使用已有译文；原文索引只使用翻译模板里的原文；均不修改游戏脚本。"
        )
        context_hint.setWordWrap(True)
        context_hint.setObjectName("config_hint_label")
        context_layout.addWidget(context_hint)

        self.rag_enabled_cb = QCheckBox("启用 RAG 记忆库")
        context_layout.addWidget(self.rag_enabled_cb)

        self.source_index_enabled_cb = QCheckBox("启用原文索引")
        context_layout.addWidget(self.source_index_enabled_cb)

        self.bootstrap_on_build_cb = QCheckBox("开始翻译时自动暖 RAG 库")
        context_layout.addWidget(self.bootstrap_on_build_cb)

        self.context_storage_game_cb = QCheckBox("上下文库保存到游戏目录")
        self.context_storage_game_cb.setToolTip(
            "启用后，默认 RAG / 原文索引 / 剧情图谱路径会使用 work 同级的 translation_context/。"
        )
        context_layout.addWidget(self.context_storage_game_cb)

        layout.addWidget(context_box)
        layout.addStretch(1)
        return page

    def _build_settings_models_page(self) -> QWidget:
        page, layout = self._settings_page("settings_models")
        config_row = QHBoxLayout()
        config_row.setSpacing(16)

        sync_box = QGroupBox("同步翻译")
        sync_layout = QFormLayout(sync_box)
        sync_layout.setSpacing(8)
        sync_layout.setContentsMargins(12, 16, 12, 12)

        self.sync_model_combo = NoWheelComboBox()
        self.sync_model_combo.addItems([
            "gemini-3.5-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ])
        sync_layout.addRow("翻译模型：", self.sync_model_combo)

        self.sync_embedding_combo = NoWheelComboBox()
        self.sync_embedding_combo.addItems([
            "gemini-embedding-2",
            "gemini-embedding-001",
        ])
        sync_layout.addRow("RAG 向量模型：", self.sync_embedding_combo)

        config_row.addWidget(sync_box, 1)

        batch_box = QGroupBox("批量离线翻译")
        batch_layout = QFormLayout(batch_box)
        batch_layout.setSpacing(8)
        batch_layout.setContentsMargins(12, 16, 12, 12)

        self.batch_model_combo = NoWheelComboBox()
        self.batch_model_combo.addItems([
            "gemini-3.5-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ])
        batch_layout.addRow("翻译模型：", self.batch_model_combo)

        self.batch_embedding_combo = NoWheelComboBox()
        self.batch_embedding_combo.addItems([
            "gemini-embedding-2",
            "gemini-embedding-001",
        ])
        batch_layout.addRow("RAG 向量模型：", self.batch_embedding_combo)

        self.batch_thinking_combo = NoWheelComboBox()
        self.batch_thinking_combo.addItem("（不启用）", "")
        self.batch_thinking_combo.addItem("最小", "minimal")
        self.batch_thinking_combo.addItem("低", "low")
        self.batch_thinking_combo.addItem("中", "medium")
        self.batch_thinking_combo.addItem("高", "high")
        batch_layout.addRow("思考程度：", self.batch_thinking_combo)

        config_row.addWidget(batch_box, 1)
        layout.addLayout(config_row)
        layout.addStretch(1)
        return page

    def _build_settings_appearance_page(self) -> QWidget:
        page, layout = self._settings_page("settings_appearance")
        appearance_box = QGroupBox("外观")
        appearance_layout = QFormLayout(appearance_box)
        appearance_layout.setSpacing(8)
        appearance_layout.setContentsMargins(12, 16, 12, 12)
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

        skipped_categories = {"项目与资源", "准备流程"}
        for group_title, fields in grouped_advanced_fields():
            if group_title in skipped_categories:
                continue
            group = QGroupBox(group_title)
            form = QFormLayout(group)
            form.setSpacing(8)
            form.setContentsMargins(12, 16, 12, 12)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for field in fields:
                widget = self._create_advanced_setting_widget(field)
                self._advanced_setting_widgets[field.key] = widget
                row = self._advanced_setting_row(field, widget)
                form.addRow(f"{field.label}：", row)
            layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _advanced_setting_row(self, field: SettingField, widget: QWidget) -> QWidget:
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
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

    def _create_advanced_setting_widget(self, field: SettingField) -> QWidget:
        if field.kind == "bool":
            widget = QCheckBox()
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
            "上方可查看任务上下文、命令参考与任务记录；下方显示原始命令输出。"
            "工作台任务运行时默认留在工作台并展开底部日志抽屉；"
            "从本页启动的试跑 / 拆分 / A/B 会放大下方日志区域。"
        )
        diag_hint.setWordWrap(True)
        diag_hint.setObjectName("config_hint_label")
        diag_hint.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(diag_hint)

        action_frame = QFrame()
        action_frame.setObjectName("action_frame")
        action_outer = QVBoxLayout(action_frame)
        action_outer.setContentsMargins(12, 10, 12, 10)
        action_outer.setSpacing(8)

        self.diagnostics_action_panel = ResponsiveActionPanel(
            prep_label="上下文",
            translate_label="工具",
            compact_width=720,
        )
        diagnostics_action_panel = self.diagnostics_action_panel
        self.refresh_diagnostics_btn = diagnostics_action_panel.add_prep_button(
            QPushButton("刷新上下文")
        )
        self.refresh_diagnostics_btn.setObjectName("secondary_btn")
        self.refresh_diagnostics_btn.clicked.connect(
            lambda: self._refresh_diagnostics_context(force=True)
        )

        self.probe_btn = diagnostics_action_panel.add_translate_button(
            QPushButton("试跑样本请求")
        )
        self.probe_btn.setObjectName("secondary_btn")
        self.probe_btn.setToolTip(
            "对当前翻译包执行少量同步请求，提交批量任务前验证 API 与请求格式。"
        )
        self.probe_btn.clicked.connect(self._on_run_probe)
        self.probe_btn.setEnabled(False)

        self.compare_variants_btn = diagnostics_action_panel.add_translate_button(
            QPushButton("翻译 A/B 对比")
        )
        self.compare_variants_btn.setObjectName("secondary_btn")
        self.compare_variants_btn.setToolTip(
            "用同一批 manifest chunk 并排比较多个配置变体的同步译文，不会写回游戏文件。"
        )
        self.compare_variants_btn.clicked.connect(self._on_run_compare_variants)
        self.compare_variants_btn.setEnabled(False)

        self.keyword_merge_btn = diagnostics_action_panel.add_translate_button(
            QPushButton("合并到 glossary")
        )
        self.keyword_merge_btn.setObjectName("secondary_btn")
        self.keyword_merge_btn.setToolTip(
            "快捷入口：主路径在工作台 · 结果/写回区的「合并到 glossary」。"
            "勾选审核关键词候选并写入 glossary.json；不会修改 .rpy 脚本。"
        )
        self.keyword_merge_btn.clicked.connect(self._on_open_keyword_merge)
        self.keyword_merge_btn.setEnabled(False)

        self.split_btn = diagnostics_action_panel.add_translate_button(
            QPushButton("拆分翻译包")
        )
        self.split_btn.setObjectName("secondary_btn")
        self.split_btn.setToolTip(
            "将过大的翻译包拆成多个子包；拆分后需分别提交，RAG 为静态快照。"
        )
        self.split_btn.clicked.connect(self._on_run_split)
        self.split_btn.setEnabled(False)

        self.clear_log_btn = diagnostics_action_panel.add_translate_trailing(
            QPushButton("清空日志")
        )
        self.clear_log_btn.setObjectName("secondary_btn")
        self.clear_log_btn.clicked.connect(self._on_clear_log)
        diagnostics_action_panel.finish_setup()
        action_outer.addWidget(diagnostics_action_panel)
        layout.addWidget(action_frame)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("diagnostics_splitter")
        splitter.setChildrenCollapsible(False)
        self.diagnostics_splitter = splitter

        self.diagnostics_inner_tabs = NoWheelTabWidget()
        self.diagnostics_inner_tabs.setObjectName("diagnostics_inner_tabs")

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
        context_content = QWidget()
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
        commands_content = QWidget()
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
        # Share one document with the workbench drawer so append/clear stay in sync.
        if hasattr(self, "workbench_log_view"):
            self.workbench_log_view.setDocument(self.log_view.document())
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
        self.tab_widget.addTab(tab, "诊断日志")

    def _set_workbench_log_drawer_expanded(self, expanded: bool) -> None:
        self._workbench_log_drawer_expanded = bool(expanded)
        if not hasattr(self, "workbench_log_body"):
            return
        self.workbench_log_body.setVisible(expanded)
        if hasattr(self, "workbench_log_toggle_btn"):
            self.workbench_log_toggle_btn.setText("折叠" if expanded else "展开")
        if hasattr(self, "workbench_log_drawer"):
            if expanded:
                self.workbench_log_drawer.setMinimumHeight(
                    _WORKBENCH_LOG_DRAWER_HEADER_HEIGHT + _WORKBENCH_LOG_DRAWER_EXPANDED_HEIGHT
                )
                self.workbench_log_drawer.setMaximumHeight(16777215)
                self.workbench_log_drawer.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred,
                )
            else:
                self.workbench_log_drawer.setMinimumHeight(0)
                self.workbench_log_drawer.setMaximumHeight(
                    _WORKBENCH_LOG_DRAWER_HEADER_HEIGHT + 16
                )
                self.workbench_log_drawer.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )

    def _toggle_workbench_log_drawer(self) -> None:
        self._set_workbench_log_drawer_expanded(not self._workbench_log_drawer_expanded)

    def _scroll_log_views_to_end(self) -> None:
        for attr in ("log_view", "workbench_log_view"):
            view = getattr(self, attr, None)
            if view is None or not hasattr(view, "verticalScrollBar"):
                continue
            scrollbar = view.verticalScrollBar()
            if scrollbar is not None:
                scrollbar.setValue(scrollbar.maximum())

    def _show_workbench_log_drawer(self) -> None:
        """Reveal CLI output on the workbench without switching the main tab.

        Used by workbench-started tasks (translate, writeback, repair, bootstrap, …).
        """
        self._set_workbench_log_drawer_expanded(True)
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
        """On runner errors: expand drawer, or enlarge diagnostics log if already there."""
        on_diagnostics = (
            getattr(self, "_diagnostics_tab", None) is not None
            and self.tab_widget.currentWidget() is self._diagnostics_tab
        )
        if on_diagnostics:
            self._expand_diagnostics_log(switch_tab=False)
        else:
            self._show_workbench_log_drawer()

    def _on_open_diagnostics_from_log_drawer(self) -> None:
        self._expand_diagnostics_log(switch_tab=True)
        self.statusBar().showMessage("已打开诊断日志页。", 3000)

    def _focus_log_tab(self) -> None:
        """Deprecated dual-purpose helper; prefer workbench drawer or diagnostics expand.

        Kept as a thin alias to the workbench drawer so accidental leftover calls no longer
        force a main-tab switch. New code must call the explicit APIs.
        """
        self._show_workbench_log_drawer()

    def _focus_workbench_status_tab(self, index: int) -> None:
        if not (0 <= index < self.workbench_status_tabs.count()):
            return
        # Avoid double chrome sync: currentChanged already calls _sync_batch_stage_chrome.
        if self.workbench_status_tabs.currentIndex() == index:
            self._sync_batch_stage_chrome(stage_index=index)
            return
        self.workbench_status_tabs.setCurrentIndex(index)

    def _on_workbench_status_tab_changed(self, index: int) -> None:
        self._sync_batch_stage_chrome(stage_index=index)

    def _on_batch_stage_clicked(self, index: int) -> None:
        self._focus_workbench_status_tab(index)

    def _batch_stage_mode_active(self) -> bool:
        return self._current_work_mode() == WorkMode.BATCH_TRANSLATION

    def _current_batch_stage_index(self) -> int:
        if not hasattr(self, "workbench_status_tabs"):
            return _BATCH_STAGE_EXECUTE
        index = self.workbench_status_tabs.currentIndex()
        if index < 0:
            return _BATCH_STAGE_EXECUTE
        return max(_BATCH_STAGE_PREPARE, min(_BATCH_STAGE_RESULT, index))

    def _sync_batch_stage_chrome(self, *, stage_index: int | None = None) -> None:
        """Show/hide stage strip; in batch mode hide duplicate status tab bar."""
        is_batch = self._batch_stage_mode_active()
        if hasattr(self, "batch_stage_bar"):
            self.batch_stage_bar.setVisible(is_batch)
        if hasattr(self, "workbench_status_tabs"):
            tab_bar = self.workbench_status_tabs.tabBar()
            if tab_bar is not None:
                # Stage strip replaces the flat three-tab chrome in batch mode.
                tab_bar.setVisible(not is_batch)
        if not is_batch or not hasattr(self, "_batch_stage_buttons"):
            return
        if stage_index is None:
            stage_index = self._current_batch_stage_index()
        stage_index = max(_BATCH_STAGE_PREPARE, min(_BATCH_STAGE_RESULT, stage_index))
        for i, btn in enumerate(self._batch_stage_buttons):
            active = i == stage_index
            btn.setChecked(active)
            btn.setProperty("active", "true" if active else "false")
            style = btn.style()
            if style is not None:
                style.unpolish(btn)
                style.polish(btn)
        if hasattr(self, "batch_stage_hint"):
            self.batch_stage_hint.setText(_BATCH_STAGE_HINTS[stage_index])
        # Result stage hosts writeback flow bars that may have reflowed while off-stage.
        if stage_index == _BATCH_STAGE_RESULT:
            self._reflow_button_bars()

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

    def _reflow_button_bars(self) -> None:
        """Re-pack all flow/responsive button strips after visibility or size changes."""
        for name in (
            "action_panel",
            "diagnostics_action_panel",
            "global_project_actions",
            "writeback_primary_bar",
            "writeback_issues_panel",
        ):
            panel = getattr(self, name, None)
            reflow = getattr(panel, "reflow", None) if panel is not None else None
            if callable(reflow):
                reflow(force=True)

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
                self.keyword_merge_writeback_btn.setVisible(True)
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
            self.apply_revision_btn.setVisible(uses_revision_writeback)

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
            self.keyword_merge_writeback_btn.setVisible(uses_keyword_merge)
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
                "补译流程暂无后续步骤；可查看诊断页任务上下文确认状态。",
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
        del findings
        label.setText("")
        label.setVisible(False)

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
        primary_hint = "主入口在工作台写回/结果区。"
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

    def _current_task_category(self) -> TaskCategory:
        return task_category_for_work_mode(self._current_work_mode())

    def _set_combo_value_by_data(self, combo: NoWheelComboBox, data: str) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == data:
                blocked = combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(blocked)
                return

    def _capture_mode_session(self) -> WorkbenchModeSession:
        # Align with product default (执行) when tabs are unavailable.
        stage_index = _BATCH_STAGE_EXECUTE
        if hasattr(self, "workbench_status_tabs"):
            stage_index = self._current_batch_stage_index()
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

    def _rebuild_work_task_combo(
        self,
        category: TaskCategory,
        *,
        selected_mode: WorkMode | None = None,
    ) -> None:
        """Legacy helper: keep hidden work_task_combo in sync for residual callers."""
        if not hasattr(self, "work_task_combo"):
            return
        blocked = self.work_task_combo.blockSignals(True)
        self.work_task_combo.clear()
        modes = work_modes_for_category(category)
        selected = selected_mode if selected_mode in modes else default_work_mode_for_category(category)
        for mode in modes:
            self.work_task_combo.addItem(work_mode_spec(mode).label, mode.value)
        self._set_combo_value_by_data(self.work_task_combo, selected.value)
        self.work_task_combo.blockSignals(blocked)

    def _sync_task_selectors_from_work_mode(self) -> None:
        """Sync left nav + submode (+ hidden legacy combos) to the active WorkMode."""
        mode = self._current_work_mode()
        spec = work_mode_spec(mode)
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

        if hasattr(self, "workbench_stack"):
            page = self._workbench_stack_pages.get(nav_item)
            if page is not None:
                self.workbench_stack.setCurrentWidget(page)

        nav_spec = workbench_nav_spec(nav_item)
        show_sub = nav_spec.show_submode
        for widget in getattr(self, "_work_submode_row_widgets", ()):
            widget.setVisible(show_sub)
        if hasattr(self, "work_submode_combo"):
            blocked = self.work_submode_combo.blockSignals(True)
            self.work_submode_combo.clear()
            for sub_mode in nav_spec.work_modes:
                self.work_submode_combo.addItem(
                    work_mode_submode_label(sub_mode),
                    sub_mode.value,
                )
            self._set_combo_value_by_data(self.work_submode_combo, mode.value)
            self.work_submode_combo.blockSignals(blocked)

        if hasattr(self, "task_category_combo"):
            blocked_category = self.task_category_combo.blockSignals(True)
            # Ensure category items exist (hidden legacy combo may be empty).
            if self.task_category_combo.count() == 0:
                for category in TASK_CATEGORY_ORDER:
                    self.task_category_combo.addItem(
                        task_category_spec(category).label,
                        category.value,
                    )
            self._set_combo_value_by_data(self.task_category_combo, spec.category.value)
            self.task_category_combo.blockSignals(blocked_category)
        self._rebuild_work_task_combo(spec.category, selected_mode=spec.mode)

    def _on_workbench_nav_changed(self, row: int) -> None:
        if row < 0:
            return
        if self.kill_btn.isEnabled() or self._task_running:
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
        # Prefer last session mode within this nav if present.
        target_mode = default_work_mode_for_nav(nav_item)
        for mode in workbench_nav_spec(nav_item).work_modes:
            session = self._mode_sessions.get(mode)
            if session is not None and not session.is_empty():
                target_mode = mode
                break
            if mode == self._work_mode:
                target_mode = mode
                break
        self._set_work_mode(target_mode, refresh_manifest_writeback=True)

    def _on_work_submode_changed(self) -> None:
        if self.kill_btn.isEnabled() or self._task_running:
            self._sync_task_selectors_from_work_mode()
            return
        mode = normalize_work_mode(self.work_submode_combo.currentData())
        if mode == self._work_mode:
            return
        self._set_work_mode(mode, refresh_manifest_writeback=True)

    def _on_task_category_changed(self) -> None:
        # Legacy path (hidden combo); keep behavior for residual callers.
        if self.kill_btn.isEnabled():
            self._sync_task_selectors_from_work_mode()
            return
        category = normalize_task_category(self.task_category_combo.currentData())
        if category == self._current_task_category():
            return
        self._set_work_mode(
            default_work_mode_for_category(category),
            refresh_manifest_writeback=True,
        )

    def _on_work_task_changed(self) -> None:
        if self.kill_btn.isEnabled():
            self._sync_task_selectors_from_work_mode()
            return
        mode = normalize_work_mode(self.work_task_combo.currentData())
        if mode == self._work_mode:
            return
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
        previous = getattr(self, "_work_mode", None)
        if not hasattr(self, "_mode_sessions"):
            self._mode_sessions = {}

        if previous is not None and previous != mode and not reset_session:
            self._mode_sessions[previous] = self._capture_mode_session()

        self._work_mode = mode
        self._workbench_nav_item = workbench_nav_for_work_mode(mode)
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
        if running or not spec.implemented or not bootstrap_ready:
            return False
        if self._translation_requires_doctor_check(spec.mode):
            return self._doctor_allows_translate_action()
        return True

    def _update_translate_button_label(self) -> None:
        self.translate_btn.setText(self._translate_button_label())

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
        if spec.mode == WorkMode.BATCH_TRANSLATION:
            self.workbench_status_tabs.setTabText(0, _BATCH_STAGE_LABELS[0])
            self.workbench_status_tabs.setTabText(1, _BATCH_STAGE_LABELS[1])
            self.workbench_status_tabs.setTabText(2, _BATCH_STAGE_LABELS[2])
        else:
            self.workbench_status_tabs.setTabText(0, "环境检查")
            self.workbench_status_tabs.setTabText(1, spec.progress_tab_label)
            self.workbench_status_tabs.setTabText(2, spec.writeback_tab_label)
        if spec.implemented:
            if spec.is_bootstrap and not self._bootstrap_task_ready(spec):
                hint = bootstrap_disabled_message(spec.bootstrap_kind)
            else:
                hint = spec.idle_workflow_message
        else:
            hint = spec.not_implemented_message
        self.work_mode_hint_label.setText(hint)
        self._apply_task_page_chrome(spec)

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

        # Restore stage tab when returning to a page (batch: 准备/执行/结果).
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
        self._sync_batch_stage_chrome()

        # Keep session bag aligned after UI refresh mutates active fields.
        self._mode_sessions[spec.mode] = self._capture_mode_session()
        running = self.kill_btn.isEnabled()
        bootstrap_ready = self._bootstrap_task_ready(spec)
        self.translate_btn.setEnabled(
            self._translate_button_enabled(
                spec=spec,
                bootstrap_ready=bootstrap_ready,
                running=running,
            )
        )
        self.resume_btn.setEnabled(spec.implemented and spec.supports_resume and not running)
        self._update_split_submit_btn(running=running)
        self._sync_task_shortcuts()
        # Re-apply page chrome after enable flags so context cards stay correct.
        self._apply_task_page_chrome(spec)

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
        if spec.mode == WorkMode.SYNC_TRANSLATION and step.key == "run":
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
        self.statusBar().showMessage("请在工作区列表中选择项目并「切换到此项目」。", 5000)

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
            worker.wait(100)
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
        self._clear_all_mode_sessions()
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
        result = handle_post_apply_registry_update(
            self,
            workspace_root=resolve_workspace_root(self.state.get_tool_root()),
            game_root=self.state.get_game_root(),
            manifest_path=manifest_path,
        )
        if result.message:
            self._append_log(result.message)

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
            panel.set_current_game_root(self.state.get_game_root())
        self._show_settings_status("已切换工作区项目", 5000)
        self._focus_settings_section("project")
        return True

    def _focus_settings_section(self, key: str) -> None:
        if hasattr(self, "_config_tab") and hasattr(self, "tab_widget"):
            settings_index = self.tab_widget.indexOf(self._config_tab)
            if settings_index >= 0:
                self.tab_widget.setCurrentIndex(settings_index)
        row = getattr(self, "_settings_nav_rows", {}).get(key)
        nav = getattr(self, "settings_nav", None)
        if nav is not None and row is not None:
            nav.setCurrentRow(row)

    def _on_go_to_workspace_for_project_switch(self) -> None:
        self._focus_settings_section("workspace")

    def _is_config_tab_active(self) -> bool:
        tab_widget = getattr(self, "tab_widget", None)
        config_tab = getattr(self, "_config_tab", None)
        if tab_widget is None or config_tab is None:
            return False
        return tab_widget.currentWidget() is config_tab

    def _activate_workspace_registry_section(self) -> None:
        panel = getattr(self, "_games_registry_panel", None)
        if panel is None:
            return
        panel.set_current_game_root(self.state.get_game_root())
        panel.activate_section()

    def _on_settings_nav_row_changed(self, row: int) -> None:
        if row < 0:
            return
        self.settings_stack.setCurrentIndex(row)
        self._sync_settings_action_bar_enabled(task_running=self._task_running, nav_row=row)
        if self._settings_nav_rows.get("workspace") == row and self._is_config_tab_active():
            self._activate_workspace_registry_section()

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

        self.api_status_label.setText(message)

    def _current_theme_preference_from_ui(self) -> str:
        combo = getattr(self, "theme_combo", None)
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
            elif hasattr(widget, "setPlainText"):
                widget.setPlainText(self._format_advanced_setting_text(field, value))
            else:
                widget.setText(str(value))

    def _format_advanced_setting_text(self, field: SettingField, value: object) -> str:
        if field.kind == "list":
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
        page_key = (
            "project"
            if field is not None and field.category in {"项目与资源", "准备流程"}
            else "advanced"
        )
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
            "sync_model": self.sync_model_combo.currentText().strip(),
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
                return

        # Commit the new tab index immediately, then defer heavy enter work so Qt
        # can paint the switched tab first (settings / diagnostics enter path).
        self._last_main_tab_index = index
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
            self._refresh_api_status()
            nav = getattr(self, "settings_nav", None)
            if (
                nav is not None
                and self._settings_nav_rows.get("workspace") == nav.currentRow()
            ):
                self._activate_workspace_registry_section()
            return

        if widget is getattr(self, "_diagnostics_tab", None):
            self._refresh_diagnostics_context()


    def _on_reload_config(self) -> None:
        self._load_config_to_ui()
        self._refresh_api_status()
        self._show_settings_status("已重新加载已保存设置。")

    def _on_restore_recommended_config(self) -> None:
        values = BASIC_RECOMMENDED_VALUES
        self.rag_enabled_cb.setChecked(bool(values["rag_enabled"]))
        self.source_index_enabled_cb.setChecked(bool(values["source_index_enabled"]))
        self.bootstrap_on_build_cb.setChecked(bool(values["bootstrap_on_build"]))
        self.context_storage_game_cb.setChecked(values["context_storage_location"] == "game")
        self._set_combo_value(self.sync_model_combo, values["sync_model"])
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
        label = getattr(self, "settings_project_root_value", None)
        if label is None:
            return
        root = self.state.get_game_root()
        if root:
            label.setText(str(root))
        else:
            label.setText("（尚未选择项目，请前往「工作区」切换）")

    def _refresh_project_label(self):
        root = self.state.get_game_root()
        path_text = str(root) if root else "（尚未选择项目）"
        if hasattr(self, "global_project_path_edit"):
            self.global_project_path_edit.setText(path_text)
        elif hasattr(self, "project_path_edit"):
            self.project_path_edit.setText(path_text)
        if not root:
            self._clear_game_root_redirect_notice()
        self._refresh_settings_project_root_display()

    def _is_doctor_running(self) -> bool:
        return self._doctor_worker is not None and self._doctor_worker.isRunning()

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

        self._clear_log_view()
        self._active_command = "doctor"
        self._doctor_output_lines = []
        self._focus_workbench_status_tab(0)
        self._set_doctor_summary(running_summary())
        self._append_log("=== 正在运行环境检查（collect_doctor_report）===\n")
        self._set_task_running(True)

        self._doctor_worker = DoctorWorker(parent=self)
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
            compare = compare_registry_with_doctor_report(
                resolve_workspace_root(self.state.get_tool_root()),
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
                )
            self._doctor_check_completed = True
            status_message = "项目检查完成。"
            if compare is not None and compare.matched is False:
                status_message = "项目检查完成；与工作区总表 layout 不一致。"
            elif compare is not None and compare.matched is None:
                status_message = "项目检查完成；当前项目未登记在总表。"
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

        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._active_command = "bootstrap_work"
        self._work_bootstrap_output_lines = []
        self._workflow_progress = create_workflow_progress_state("work_bootstrap")
        self._workflow_progress_base_facts = []
        self._focus_workbench_status_tab(1)
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

        self._clear_log_view()
        self._show_workbench_log_drawer()
        self._active_command = "generate_template"
        self._template_generation_output_lines = []
        self._focus_workbench_status_tab(1)
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

    def _saved_batch_context_flags(self) -> dict[str, bool]:
        return read_batch_context_flags(self.state.load_translator_config())

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
        if bool(getattr(self, "_task_running", False)) or (
            hasattr(self, "runner") and self.runner.is_running()
        ):
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

        if spec.mode in self._sync_work_modes_requiring_api_key():
            api_key_count, _ = self.state.get_api_key_status()
            if api_key_count == 0:
                QMessageBox.information(
                    self,
                    "请先配置 API Key",
                    "同步模式需要 Gemini API 密钥；请在设置页管理 API Key 或设置环境变量。",
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
        latest_manifest = self.state.get_latest_manifest_path_for_mode(game_root, spec.mode)
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
        self._expand_diagnostics_log()
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
        self._expand_diagnostics_log()
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
        self._task_running = running
        spec = work_mode_spec(self._current_work_mode())
        project_switch_enabled = not running
        if hasattr(self, "select_btn"):
            self.select_btn.setEnabled(project_switch_enabled)
        if hasattr(self, "global_browse_project_btn"):
            self.global_browse_project_btn.setEnabled(project_switch_enabled)
        if hasattr(self, "global_switch_project_btn"):
            self.global_switch_project_btn.setEnabled(project_switch_enabled)
        panel = getattr(self, "_games_registry_panel", None)
        if panel is not None:
            panel.setEnabled(not running)
        if hasattr(self, "settings_go_workspace_btn"):
            self.settings_go_workspace_btn.setEnabled(not running)
        if hasattr(self, "task_category_combo"):
            self.task_category_combo.setEnabled(not running)
        if hasattr(self, "work_task_combo"):
            self.work_task_combo.setEnabled(not running)
        if hasattr(self, "workbench_nav"):
            self.workbench_nav.setEnabled(not running)
        if hasattr(self, "work_submode_combo"):
            self.work_submode_combo.setEnabled(not running)
        self.doctor_btn.setEnabled(not running)
        self.bootstrap_work_btn.setEnabled(not running)
        self.api_btn.setEnabled(not running)
        bootstrap_ready = self._bootstrap_task_ready(spec)
        self.translate_btn.setEnabled(
            self._translate_button_enabled(
                spec=spec,
                bootstrap_ready=bootstrap_ready,
                running=running,
            )
        )
        self.resume_btn.setEnabled(spec.implemented and spec.supports_resume and not running)
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
        self.kill_btn.setEnabled(running)
        # Context-library prebuild CTAs stay on-page while nav is locked; gate them here.
        if hasattr(self, "context_library_panel"):
            self._refresh_context_library_panel(running=running)
        self._sync_task_shortcuts()
        self._reflow_button_bars()

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
        if not self.kill_btn.isEnabled():
            self.apply_btn.setEnabled(
                summary.can_apply and not self._uses_revision_writeback()
            )
            if hasattr(self, "apply_revision_btn"):
                self.apply_revision_btn.setEnabled(
                    self._uses_revision_writeback() and summary.can_apply
                )
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

    def _set_doctor_summary(self, summary: DoctorSummary):
        self._doctor_summary_mode = summary.mode
        self._doctor_summary_status = summary.status
        self.doctor_status_label.set_status(summary.status, summary.heading)
        self.doctor_message_label.setText(summary.message)
        self.doctor_facts_label.setText("\n".join(summary.facts))
        self._set_details_label(self.doctor_details_label, [])
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
        self._sync_task_shortcuts()
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
                    "修补已写入翻译文件。请点击写回页的「重新检查」更新检查结果，"
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
            combo.setCurrentIndex(combo.count() - 1)

    def _set_theme_combo_value(self, value: str) -> None:
        combo = getattr(self, "theme_combo", None)
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

    def _load_config_to_ui(self):
        self._loading_config_to_ui = True
        try:
            config = self.state.load_translator_config()
            self._load_theme_to_ui(config)
            sync_config = self._config_section(config, "sync")
            batch_config = self._config_section(config, "batch")
            sync_rag_config = self._config_section(sync_config, "rag")
            batch_rag_config = self._config_section(batch_config, "rag")
            context_flags = read_batch_context_flags(config)
            self.rag_enabled_cb.setChecked(context_flags["rag_enabled"])
            self.source_index_enabled_cb.setChecked(context_flags["source_index_enabled"])
            self.bootstrap_on_build_cb.setChecked(context_flags["bootstrap_on_build"])
            storage_config = self._config_section(config, "context_storage")
            storage_location = normalize_context_storage_location(
                storage_config.get("location", config.get("context_storage_location", ""))
            )
            self.context_storage_game_cb.setChecked(storage_location == "game")
            self._batch_thinking_config_has_key = "thinking_level" in batch_config

            sync_models = sync_config.get("models")
            sync_val = ""
            if isinstance(sync_models, list):
                for model in sync_models:
                    sync_val = self._config_string(model)
                    if sync_val:
                        break
            elif isinstance(sync_models, str):
                sync_val = self._config_string(sync_models)
            if not sync_val:
                sync_val = self._config_string(sync_config.get("model", ""))
            if not sync_val:
                sync_val = str(BASIC_RECOMMENDED_VALUES["sync_model"])
            self._set_combo_value(self.sync_model_combo, sync_val)

            batch_val = self._config_string(batch_config.get("model", "")) or str(
                BASIC_RECOMMENDED_VALUES["batch_model"]
            )
            self._set_combo_value(self.batch_model_combo, batch_val)

            sync_emb_val = self._config_string(sync_rag_config.get("embedding_model", "")) or str(
                BASIC_RECOMMENDED_VALUES["sync_embedding_model"]
            )
            self._set_combo_value(self.sync_embedding_combo, sync_emb_val)

            batch_emb_val = self._config_string(batch_rag_config.get("embedding_model", "")) or str(
                BASIC_RECOMMENDED_VALUES["batch_embedding_model"]
            )
            self._set_combo_value(self.batch_embedding_combo, batch_emb_val)

            self._on_batch_model_changed(batch_val)
            thinking_val = self._batch_thinking_value_for_load(batch_config, batch_val)
            self._set_batch_thinking_value(thinking_val)
            advanced_values = read_advanced_settings(config)
            get_game_root = getattr(self.state, "get_game_root", None)
            current_game_root = get_game_root() if callable(get_game_root) else None
            if current_game_root and not self._config_string(advanced_values.get("game_root")):
                advanced_values["game_root"] = str(current_game_root)
            self._load_advanced_settings_to_ui(advanced_values)
            self._clear_advanced_setting_errors()
        finally:
            self._batch_thinking_user_changed = False
            self._loading_config_to_ui = False
        self._config_ui_saved_snapshot = self._current_config_ui_snapshot()

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

        try:
            config = self.state.load_translator_config()
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
            batch_rag_config["enabled"] = self.rag_enabled_cb.isChecked()
            batch_rag_config["bootstrap_on_build"] = self.bootstrap_on_build_cb.isChecked()
            batch_source_index_config["enabled"] = self.source_index_enabled_cb.isChecked()

            sync_model = self.sync_model_combo.currentText().strip()
            sync_config["model"] = sync_model
            if "models" in sync_config:
                sync_models = self._sync_models_for_save(sync_config.get("models"), sync_model)
                if sync_models:
                    sync_config["models"] = sync_models
                else:
                    sync_config.pop("models", None)
            batch_model = self.batch_model_combo.currentText().strip()
            batch_config["model"] = batch_model
            sync_rag_config["embedding_model"] = self.sync_embedding_combo.currentText().strip()
            batch_rag_config["embedding_model"] = self.batch_embedding_combo.currentText().strip()
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

            self.state.save_translator_config(config)
            if not self._sync_state_game_root_from_settings(config.get("game_root")):
                self._show_settings_status("设置已保存，但同步项目目录到工作台失败。", 6000)
            if work_mode_spec(self._current_work_mode()).is_bootstrap:
                self._apply_work_mode_ui(refresh_manifest_writeback=False)
            self._append_log("设置已成功保存至 translator_config.json。")
            try:
                ToastNotification.show_toast(self, "✓ 设置已成功保存")
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
    resources_dir = Path(__file__).resolve().parent / "resources"
    project_state = ProjectState()
    try:
        bootstrap_config = project_state.load_translator_config()
        theme_preference = read_gui_theme_from_config(bootstrap_config)
    except Exception as exc:
        print(f"警告：无法读取主题配置，将使用系统跟随：{exc}")
        theme_preference = DEFAULT_THEME_PREFERENCE
    try:
        apply_theme(app, resources_dir, theme_preference)
    except OSError as exc:
        print(f"警告：无法加载 GUI 样式表：{exc}")

    win = MainWindow(
        qt_app=app,
        resources_dir=resources_dir,
        project_state=project_state,
    )
    app.styleHints().colorSchemeChanged.connect(win._on_system_color_scheme_changed)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
