"""Main GUI application for the optional workbench.

This is the first version shell (per #42):
- Pure PySide6 with tabbed layout (workbench / config / diagnostics)
- Delegates everything to the existing CLI via QProcess
- Workbench tab: project selection, doctor + translation workflow status
"""
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
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
)

from .path_utils import canonical_abs_path, normalize_context_storage_location
from .responsive_layout import ResponsiveActionPanel
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
    idle_writeback_summary,
    idle_writeback_summary_for_work_mode,
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
from .cli_runner import CliRunner
from .doctor_report import (
    DoctorSummary,
    idle_summary,
    running_summary,
    stale_summary,
    summarize_doctor_output,
)
from .manifest_resume_summary import (
    ManifestWorkflowDisplay,
    build_manifest_workflow_display,
    completed_manifest_entry_fact,
)
from .work_bootstrap_report import (
    running_work_bootstrap_summary,
    summarize_work_bootstrap_output,
    with_game_root_persist_warning,
    work_bootstrap_to_doctor_summary,
)
from .project_state import ProjectState
from .theme import apply_theme
from .theme_helpers import (
    DEFAULT_THEME_PREFERENCE,
    THEME_SYSTEM,
    normalize_theme_preference,
    read_gui_theme_from_config,
    write_gui_theme_to_config,
)
from .translation_workflow import WorkflowUpdate
from .user_copy import (
    format_job_fact,
    format_job_state_fact,
    format_manifest_path_fact,
)
from .work_modes import (
    TASK_CATEGORY_ORDER,
    TaskCategory,
    WorkMode,
    bootstrap_disabled_message,
    default_work_mode_for_category,
    normalize_task_category,
    normalize_work_mode,
    task_category_for_work_mode,
    task_category_spec,
    work_mode_hint_texts,
    work_mode_spec,
    work_modes_for_category,
)
from .workflow_factory import create_workflow, resume_workflow
from .workflow_progress import (
    WorkflowProgressState,
    create_workflow_progress_state,
    update_workflow_progress_from_line,
)
from .widget_helpers import NoWheelComboBox, NoWheelTabWidget
from .wizard_timeline import WizardTimeline

# Diagnostics splitter: idle favors task context; running tasks expand the log.
_DIAGNOSTICS_IDLE_CONTEXT_PX = 420
_DIAGNOSTICS_IDLE_LOG_PX = 180
_DIAGNOSTICS_RUNNING_CONTEXT_RATIO = 0.32


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
        self._workflow_step_output_lines: list[str] = []
        self._apply_output_lines: list[str] = []
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
        self._build_retry_output_lines: list[str] = []
        self._retry_followup_confirmed: set[str] = set()
        self._writeback_manifest_path = ""
        self._config_ui_saved_snapshot: dict[str, object] = {}
        self._last_main_tab_index = 0
        self._handling_config_tab_leave = False

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        header = QLabel("Ren'Py Translation Lab · 图形工作台")
        header.setObjectName("header_label")
        root_layout.addWidget(header)

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

        # Status
        self.statusBar().showMessage(
            "图形界面是可选组件；核心命令行不受影响。"
        )

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

    def _build_workbench_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(14)

        project_frame = QFrame()
        project_frame.setObjectName("project_frame")
        proj_outer = QVBoxLayout(project_frame)
        proj_outer.setContentsMargins(12, 10, 12, 10)
        proj_outer.setSpacing(6)
        proj_layout = QHBoxLayout()
        proj_layout.setSpacing(10)

        proj_layout.addWidget(QLabel("当前工作目录："))

        self.project_path_edit = QLineEdit("尚未选择项目")
        self.project_path_edit.setReadOnly(True)
        self.project_path_edit.setObjectName("project_path_edit")
        proj_layout.addWidget(self.project_path_edit, 1)

        self.select_btn = QPushButton("选择游戏目录...")
        self.select_btn.clicked.connect(self._on_select_project)
        proj_layout.addWidget(self.select_btn)
        proj_outer.addLayout(proj_layout)

        self.project_redirect_label = QLabel()
        self.project_redirect_label.setWordWrap(True)
        self.project_redirect_label.setObjectName("config_hint_label")
        self.project_redirect_label.setVisible(False)
        proj_outer.addWidget(self.project_redirect_label)

        layout.addWidget(project_frame)

        mode_frame = QFrame()
        mode_frame.setObjectName("mode_frame")
        self._mode_frame = mode_frame
        mode_outer = QVBoxLayout(mode_frame)
        mode_outer.setContentsMargins(12, 8, 12, 8)
        mode_outer.setSpacing(6)

        selectors_row = QHBoxLayout()
        selectors_row.setSpacing(10)
        selectors_row.addWidget(QLabel("任务类型："))
        self.task_category_combo = NoWheelComboBox()
        self.task_category_combo.setObjectName("task_category_combo")
        for category in TASK_CATEGORY_ORDER:
            self.task_category_combo.addItem(
                task_category_spec(category).label,
                category.value,
            )
        self.task_category_combo.currentIndexChanged.connect(self._on_task_category_changed)
        selectors_row.addWidget(self.task_category_combo)

        selectors_row.addWidget(QLabel("子任务："))
        self.work_task_combo = NoWheelComboBox()
        self.work_task_combo.setObjectName("work_task_combo")
        self.work_task_combo.currentIndexChanged.connect(self._on_work_task_changed)
        selectors_row.addWidget(self.work_task_combo, 1)
        mode_outer.addLayout(selectors_row)

        self.work_mode_hint_label = QLabel()
        self.work_mode_hint_label.setWordWrap(True)
        self.work_mode_hint_label.setObjectName("work_mode_hint_label")
        self.work_mode_hint_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.work_mode_hint_label.installEventFilter(self)
        mode_outer.addWidget(self.work_mode_hint_label)
        layout.addWidget(mode_frame)

        action_frame = QFrame()
        action_frame.setObjectName("action_frame")
        action_outer = QVBoxLayout(action_frame)
        action_outer.setContentsMargins(12, 10, 12, 10)
        action_outer.setSpacing(8)

        self.action_panel = ResponsiveActionPanel()
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

        self.workbench_status_tabs = NoWheelTabWidget()
        self.workbench_status_tabs.setObjectName("workbench_status_tabs")

        doctor_tab = QWidget()
        self._style_themed_surface(doctor_tab)
        doctor_layout = QVBoxLayout(doctor_tab)
        doctor_layout.setContentsMargins(12, 12, 12, 12)
        doctor_layout.setSpacing(6)
        self.doctor_status_label = QLabel()
        self.doctor_status_label.setObjectName("doctor_status_label")
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
        self.workflow_status_label = QLabel()
        self.workflow_status_label.setObjectName("workflow_status_label")
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
        self._style_themed_surface(writeback_tab)
        writeback_layout = QVBoxLayout(writeback_tab)
        writeback_layout.setContentsMargins(12, 12, 12, 12)
        writeback_layout.setSpacing(6)
        self.writeback_status_label = QLabel()
        self.writeback_status_label.setObjectName("writeback_status_label")
        writeback_layout.addWidget(self.writeback_status_label)
        writeback_scroll = QScrollArea()
        writeback_scroll.setWidgetResizable(True)
        writeback_scroll.setFrameShape(QFrame.Shape.NoFrame)
        writeback_content = QWidget()
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
        writeback_actions = QHBoxLayout()
        self.apply_btn = QPushButton("写回翻译")
        self.apply_btn.setObjectName("apply_btn")
        self.apply_btn.clicked.connect(self._on_apply_writeback)
        self.apply_btn.setEnabled(False)
        writeback_actions.addWidget(self.apply_btn)
        self.apply_revision_btn = QPushButton("写回订正")
        self.apply_revision_btn.setObjectName("apply_revision_btn")
        self.apply_revision_btn.clicked.connect(self._on_apply_revision)
        self.apply_revision_btn.setEnabled(False)
        self.apply_revision_btn.setVisible(False)
        writeback_actions.addWidget(self.apply_revision_btn)
        self.check_issues_btn = QPushButton("查看问题清单")
        self.check_issues_btn.setObjectName("secondary_btn")
        self.check_issues_btn.clicked.connect(self._open_check_issues)
        self.check_issues_btn.setEnabled(False)
        writeback_actions.addWidget(self.check_issues_btn)
        self.retry_btn = QPushButton("生成补译包")
        self.retry_btn.setObjectName("secondary_btn")
        self.retry_btn.clicked.connect(self._on_retry_action)
        self.retry_btn.setEnabled(False)
        self.retry_btn.setVisible(False)
        writeback_actions.addWidget(self.retry_btn)
        self.apply_failure_btn = QPushButton("查看写回失败报告")
        self.apply_failure_btn.setObjectName("secondary_btn")
        self.apply_failure_btn.clicked.connect(self._open_apply_failure_report)
        self.apply_failure_btn.setEnabled(False)
        self.apply_failure_btn.setVisible(False)
        writeback_actions.addWidget(self.apply_failure_btn)
        self.remediation_btn = QPushButton("补救命令")
        self.remediation_btn.setObjectName("secondary_btn")
        self.remediation_btn.clicked.connect(self._open_remediation_commands)
        self.remediation_btn.setEnabled(False)
        writeback_actions.addWidget(self.remediation_btn)
        writeback_actions.addStretch()
        writeback_layout.addLayout(writeback_actions)
        writeback_layout.addStretch()
        self.workbench_status_tabs.addTab(writeback_tab, "写回")

        self.workbench_status_tabs.setCurrentIndex(1)
        layout.addWidget(self.workbench_status_tabs, 1)

        self.tab_widget.addTab(tab, "工作台")

    def _style_themed_surface(self, widget: QWidget) -> None:
        widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def _build_config_tab(self) -> None:
        tab = QWidget()
        tab.setObjectName("config_tab")
        self._style_themed_surface(tab)
        self._config_tab = tab
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("config_scroll")
        self._style_themed_surface(scroll)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        viewport = scroll.viewport()
        viewport.setObjectName("config_scroll_viewport")
        self._style_themed_surface(viewport)

        content = QWidget()
        content.setObjectName("config_scroll_content")
        self._style_themed_surface(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(14)

        api_box = QGroupBox("API Key")
        api_layout = QVBoxLayout(api_box)
        api_layout.setSpacing(10)
        api_layout.setContentsMargins(12, 16, 12, 12)

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

        context_box = QGroupBox("批量上下文")
        context_layout = QVBoxLayout(context_box)
        context_layout.setSpacing(8)
        context_layout.setContentsMargins(12, 16, 12, 12)

        context_hint = QLabel(
            "启用后先保存配置，再到工作台「分析与准备」下运行预建子任务。"
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
        layout.addLayout(config_row, 1)

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
        layout.addWidget(appearance_box)

        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self.save_config_btn = QPushButton("保存参数配置")
        self.save_config_btn.setObjectName("save_config_btn")
        self.save_config_btn.clicked.connect(self._on_save_config)
        save_layout.addWidget(self.save_config_btn)
        layout.addLayout(save_layout)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        self.tab_widget.addTab(tab, "配置")

    def _build_log_tab(self) -> None:
        tab = QWidget()
        tab.setObjectName("diagnostics_tab")
        self._diagnostics_tab = tab
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(10)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        diag_hint = QLabel(
            "上方可查看任务上下文、命令参考与任务记录；下方显示原始命令输出。"
            "任务运行时会自动切到此页并放大日志区域。"
        )
        diag_hint.setWordWrap(True)
        diag_hint.setObjectName("config_hint_label")
        toolbar.addWidget(diag_hint, 1)

        self.refresh_diagnostics_btn = QPushButton("刷新上下文")
        self.refresh_diagnostics_btn.setObjectName("secondary_btn")
        self.refresh_diagnostics_btn.clicked.connect(self._refresh_diagnostics_context)
        toolbar.addWidget(self.refresh_diagnostics_btn)

        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.setObjectName("secondary_btn")
        self.clear_log_btn.clicked.connect(self._on_clear_log)
        toolbar.addWidget(self.clear_log_btn)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("diagnostics_splitter")
        splitter.setChildrenCollapsible(False)
        self.diagnostics_splitter = splitter

        self.diagnostics_inner_tabs = NoWheelTabWidget()
        self.diagnostics_inner_tabs.setObjectName("diagnostics_inner_tabs")

        context_tab = QWidget()
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

        self.diagnostics_status_label = QLabel()
        self.diagnostics_status_label.setObjectName("diagnostics_status_label")
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
        log_panel_layout.addWidget(self.log_view, 1)
        splitter.addWidget(log_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([_DIAGNOSTICS_IDLE_CONTEXT_PX, _DIAGNOSTICS_IDLE_LOG_PX])

        layout.addWidget(splitter, 1)
        self.tab_widget.addTab(tab, "诊断日志")

    def _focus_log_tab(self) -> None:
        if self._diagnostics_tab is not None:
            self.tab_widget.setCurrentWidget(self._diagnostics_tab)
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

    def _focus_workbench_status_tab(self, index: int) -> None:
        if 0 <= index < self.workbench_status_tabs.count():
            self.workbench_status_tabs.setCurrentIndex(index)

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
        for column in range(self.split_status_table.columnCount()):
            if is_split_action_column(column):
                self.split_status_table.setItemDelegateForColumn(column, delegate)

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
            else:
                action_item.setData(SPLIT_ACTION_DATA_ROLE, None)
                action_item.setToolTip("")
            model_index = self.split_status_table.model().index(row, base_column + 5)
            self.split_status_table.viewport().update(
                self.split_status_table.visualRect(model_index)
            )

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
        action_payload = split_action_item_payload(
            selectable=show_action_button,
            manifest_path=entry.manifest_path,
            part_label=entry.part_label,
        )
        action_item = QTableWidgetItem("")
        action_item.setFlags(action_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        action_item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        if action_payload is not None:
            action_item.setData(SPLIT_ACTION_DATA_ROLE, action_payload)
            action_item.setToolTip(f"\u5207\u6362\u5230 {entry.part_label}")
        self.split_status_table.setItem(row, base_column + 5, action_item)
        self._apply_split_table_row_style(row, entry, base_column=base_column, is_current=is_current)

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
        bg_color, text_color, status_color = self._split_status_row_colors(entry.status_kind)
        background = QBrush(QColor(bg_color))
        foreground = QBrush(QColor(text_color))
        status_foreground = QBrush(QColor(status_color))
        for column in range(base_column, min(base_column + 6, self.split_status_table.columnCount())):
            item = self.split_status_table.item(row, column)
            if item is None:
                continue
            item.setBackground(background)
            item.setForeground(status_foreground if column == base_column + 1 else foreground)
            font = item.font()
            font.setBold(column == base_column + 1 or (is_current and column == base_column))
            item.setFont(font)

    def _split_status_row_colors(self, status_kind: str) -> tuple[str, str, str]:
        dark = self.palette().window().color().lightness() < 128
        if dark:
            colors = {
                "applied": ("#052e2b", "#d1fae5", "#34d399"),
                "checked_safe": ("#12351f", "#dcfce7", "#4ade80"),
                "checked_warn": ("#422006", "#fef3c7", "#fbbf24"),
                "checked_block": ("#450a0a", "#fee2e2", "#f87171"),
                "failed": ("#450a0a", "#fee2e2", "#f87171"),
                "downloaded": ("#172554", "#dbeafe", "#60a5fa"),
                "running": ("#3b2505", "#fef3c7", "#f59e0b"),
                "submitted": ("#1e293b", "#cbd5e1", "#93c5fd"),
                "succeeded": ("#14345b", "#dbeafe", "#60a5fa"),
                "unsubmitted": ("#111827", "#94a3b8", "#94a3b8"),
            }
            return colors.get(status_kind, ("#0f172a", "#cbd5e1", "#cbd5e1"))
        colors = {
            "applied": ("#d1fae5", "#064e3b", "#047857"),
            "checked_safe": ("#dcfce7", "#14532d", "#15803d"),
            "checked_warn": ("#fef3c7", "#78350f", "#b45309"),
            "checked_block": ("#fee2e2", "#7f1d1d", "#dc2626"),
            "failed": ("#fee2e2", "#7f1d1d", "#dc2626"),
            "downloaded": ("#dbeafe", "#1e3a8a", "#2563eb"),
            "running": ("#fef3c7", "#78350f", "#d97706"),
            "submitted": ("#e2e8f0", "#334155", "#2563eb"),
            "succeeded": ("#e0f2fe", "#075985", "#0284c7"),
            "unsubmitted": ("#f8fafc", "#64748b", "#64748b"),
        }
        return colors.get(status_kind, ("#f8fafc", "#334155", "#334155"))

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

    def _update_writeback_action_buttons(
        self,
        summary: WritebackSummary,
        *,
        running: bool,
    ) -> None:
        spec = work_mode_spec(self._current_work_mode())
        uses_revision_writeback = self._uses_revision_writeback(spec.mode)
        has_writeback_actions = spec.supports_translation_writeback or uses_revision_writeback
        action_buttons = (
            "apply_btn",
            "apply_revision_btn",
            "check_issues_btn",
            "retry_btn",
            "apply_failure_btn",
            "remediation_btn",
        )
        if not has_writeback_actions:
            for button_name in action_buttons:
                if hasattr(self, button_name):
                    button = getattr(self, button_name)
                    button.setVisible(False)
                    button.setEnabled(False)
            return

        issues_ready = self._writeback_issues_ready(summary)
        manifest = self._load_writeback_manifest() if summary.manifest_path else None

        if hasattr(self, "check_issues_btn"):
            self.check_issues_btn.setVisible(True)
            self.check_issues_btn.setEnabled(not running and issues_ready)

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

        if hasattr(self, "remediation_btn"):
            remediation_ready = (
                self._remediation_ready(summary, manifest=manifest)
                if issues_ready
                else False
            )
            self.remediation_btn.setVisible(True)
            self.remediation_btn.setEnabled(not running and remediation_ready)

        if hasattr(self, "apply_btn"):
            self.apply_btn.setVisible(spec.supports_translation_writeback)
        if hasattr(self, "apply_revision_btn"):
            self.apply_revision_btn.setVisible(uses_revision_writeback)

    def _show_retry_preview(
        self,
        retry_manifest_path: str,
        *,
        open_remediation_on_confirm: bool = False,
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
        if open_remediation_on_confirm:
            self._open_remediation_commands()
        else:
            self.statusBar().showMessage("已确认补译包范围。", 3000)
        return "confirmed"

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
        dialog = CheckIssuesDialog(self, report=report)
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
        self.statusBar().showMessage("诊断日志已清空。", 3000)

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

    def _refresh_diagnostics_context(
        self,
        *,
        latest_manifest_path: Path | str | None = None,
        manifest: dict[str, object] | None = None,
    ) -> None:
        spec = work_mode_spec(self._current_work_mode())
        if spec.mode == WorkMode.SYNC_TRANSLATION:
            context = sync_diagnostics_context(
                sync_script_path=str(self.state.get_sync_script_path()),
                python_exe=sys.executable,
            )
            self._set_diagnostics_context(context)
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
        )
        self._set_diagnostics_context(context)

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

        self.diagnostics_status_label.setText(context.heading)
        self.diagnostics_status_label.setProperty("status", context.status)
        self.diagnostics_status_label.style().unpolish(self.diagnostics_status_label)
        self.diagnostics_status_label.style().polish(self.diagnostics_status_label)
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

    def _rebuild_work_task_combo(
        self,
        category: TaskCategory,
        *,
        selected_mode: WorkMode | None = None,
    ) -> None:
        blocked = self.work_task_combo.blockSignals(True)
        self.work_task_combo.clear()
        modes = work_modes_for_category(category)
        selected = selected_mode if selected_mode in modes else default_work_mode_for_category(category)
        for mode in modes:
            self.work_task_combo.addItem(work_mode_spec(mode).label, mode.value)
        self._set_combo_value_by_data(self.work_task_combo, selected.value)
        self.work_task_combo.blockSignals(blocked)

    def _sync_task_selectors_from_work_mode(self) -> None:
        spec = work_mode_spec(self._current_work_mode())
        blocked_category = self.task_category_combo.blockSignals(True)
        self._set_combo_value_by_data(self.task_category_combo, spec.category.value)
        self.task_category_combo.blockSignals(blocked_category)
        self._rebuild_work_task_combo(spec.category, selected_mode=spec.mode)

    def _on_task_category_changed(self) -> None:
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

    def _set_work_mode(self, mode: WorkMode, *, refresh_manifest_writeback: bool) -> None:
        self._work_mode = mode
        self._workflow = None
        self._workflow_step_output_lines = []
        self._writeback_manifest_path = ""
        self._clear_completed_manifest_snapshot()
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
        self.translate_btn.setText(spec.start_button_label)
        if spec.resume_button_label:
            self.resume_btn.setText(spec.resume_button_label)
        self._update_resume_btn_text()
        self.resume_btn.setVisible(spec.supports_resume)
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
        if refresh_manifest_writeback or refresh_diagnostics:
            self._refresh_manifest_derived_ui(
                refresh_writeback=refresh_manifest_writeback,
                refresh_diagnostics=refresh_diagnostics,
            )
        else:
            self._refresh_workflow_from_latest_manifest()
        running = self.kill_btn.isEnabled()
        bootstrap_ready = self._bootstrap_task_ready(spec)
        self.translate_btn.setEnabled(spec.implemented and bootstrap_ready and not running)
        self.resume_btn.setEnabled(spec.implemented and spec.supports_resume and not running)
        self._update_split_submit_btn(running=running)

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

    def _on_select_project(self):
        start_dir = str(self.state.get_game_root() or Path.home())
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择游戏目录（项目根目录或 work 目录；项目根目录下存在 work/ 时会自动切换）",
            start_dir,
        )
        if directory:
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
                return
            self._refresh_project_label()
            self._load_config_to_ui()
            self._active_command = ""
            self._doctor_output_lines = []
            self._workflow = None
            self._workflow_step_output_lines = []
            self._clear_completed_manifest_snapshot()
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
            if spec.supports_translation_writeback:
                self._set_writeback_summary(stale_writeback_summary())
            else:
                self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
            self._apply_work_mode_ui(refresh_manifest_writeback=False)
            self._refresh_diagnostics_context()
            self._append_log(f"项目目录已设置为：{directory}")

    def _refresh_api_status(self) -> None:
        count, source = self.state.get_api_key_status()
        file_keys = self.state.load_api_keys()

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

    def _current_config_ui_snapshot(self) -> dict[str, object]:
        thinking_val = self.batch_thinking_combo.currentData()
        thinking_level = thinking_val if isinstance(thinking_val, str) else ""
        return {
            "rag_enabled": self.rag_enabled_cb.isChecked(),
            "source_index_enabled": self.source_index_enabled_cb.isChecked(),
            "bootstrap_on_build": self.bootstrap_on_build_cb.isChecked(),
            "context_storage_location": "game" if self.context_storage_game_cb.isChecked() else "tool",
            "sync_model": self.sync_model_combo.currentText().strip(),
            "batch_model": self.batch_model_combo.currentText().strip(),
            "sync_embedding_model": self.sync_embedding_combo.currentText().strip(),
            "batch_embedding_model": self.batch_embedding_combo.currentText().strip(),
            "batch_thinking_level": thinking_level,
        }

    def _config_tab_has_unsaved_changes(self) -> bool:
        if self._loading_config_to_ui:
            return False
        if not self._config_ui_saved_snapshot:
            return False
        return self._current_config_ui_snapshot() != self._config_ui_saved_snapshot

    def _confirm_leave_config_tab(self, previous_index: int) -> bool:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("配置尚未保存")
        message.setText("配置页有未保存的更改。")
        message.setInformativeText("离开前可以保存配置，或留在配置页继续检查。")
        save_btn = message.addButton("保存并离开", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = message.addButton("不保存离开", QMessageBox.ButtonRole.DestructiveRole)
        stay_btn = message.addButton("留在配置页", QMessageBox.ButtonRole.RejectRole)
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

        if current_widget is self._config_tab:
            self._refresh_api_status()
        if current_widget is getattr(self, "_diagnostics_tab", None):
            self._refresh_diagnostics_context()
        self._last_main_tab_index = index


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

    def _refresh_project_label(self):
        root = self.state.get_game_root()
        if root:
            self.project_path_edit.setText(str(root))
        else:
            self.project_path_edit.setText("（尚未选择项目）")
            self._clear_game_root_redirect_notice()

    def _on_run_doctor(self):
        if not self.state.get_game_root():
            QMessageBox.information(
                self, "请先选择项目",
                "请先选择游戏的 work 目录。\n"
                "环境检查会读取本地配置中的项目路径。"
            )
            return

        self._clear_log_view()
        self._active_command = "doctor"
        self._doctor_output_lines = []
        self._focus_workbench_status_tab(0)
        self._set_doctor_summary(running_summary())
        self._append_log("=== 正在运行：gemini_translate_batch.py doctor ===\n")
        self._set_task_running(True)

        script = self.state.get_batch_script_path()
        # Run with no extra args — it will pick up translator_config.json
        self.runner.run(script, ["doctor"])

    def _on_bootstrap_work(self):
        if not self.state.get_game_root():
            QMessageBox.information(
                self,
                "请先选择项目",
                "请先选择游戏目录（项目根目录或 work 目录均可）。\n"
                "准备工作目录会在存在 original/game 时，把内容复制到 work/game。",
            )
            return

        self._clear_log_view()
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

    def _saved_batch_context_flags(self) -> dict[str, bool]:
        return read_batch_context_flags(self.state.load_translator_config())

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
        self._focus_log_tab()
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
        if not self.state.get_game_root():
            QMessageBox.information(
                self,
                "请先选择项目",
                "请先选择游戏的 work 目录。",
            )
            return

        if spec.mode in self._sync_work_modes_requiring_api_key():
            api_key_count, _ = self.state.get_api_key_status()
            if api_key_count == 0:
                QMessageBox.information(
                    self,
                    "请先配置 API Key",
                    "同步模式需要 Gemini API 密钥；请在配置页管理 API Key 或设置环境变量。",
                )
                return

        workflow = create_workflow(spec.mode)
        if workflow is None:
            QMessageBox.information(self, "无法开始任务", spec.not_implemented_message)
            return

        self._clear_log_view()
        self._focus_log_tab()
        self._clear_completed_manifest_snapshot()
        self._writeback_manifest_path = ""
        if spec.supports_translation_writeback:
            self._set_writeback_summary(stale_writeback_summary())
        else:
            self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
        self._workflow = workflow
        self._active_command = "translation_workflow"
        self._workflow_step_output_lines = []
        self._focus_workbench_status_tab(1)
        self._append_log(f"=== 正在运行：{spec.label} ===\n")
        self._set_task_running(True)
        self._run_workflow_current_step()

    def _on_resume_translation(self):
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
                self._focus_log_tab()
                self._workflow = split_workflow
                self._refresh_diagnostics_context()
                self._active_command = "translation_workflow"
                self._workflow_step_output_lines = []
                self._focus_workbench_status_tab(1)
                self._append_log("=== 正在刷新全部拆分包状态 ===\n")
                self._set_task_running(True)
                self._run_workflow_current_step()
                return

        workflow = resume_workflow(spec.mode, str(latest_manifest), manifest)
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
        self._focus_log_tab()
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
        )
        if workflow.current_step() is None:
            QMessageBox.information(self, "没有待提交拆分包", "当前拆分组没有尚未提交的包。")
            return

        self._clear_log_view()
        self._focus_log_tab()
        self._workflow = workflow
        self._refresh_diagnostics_context()
        self._active_command = "translation_workflow"
        self._workflow_step_output_lines = []
        self._focus_workbench_status_tab(1)
        self._append_log(f"=== 正在批量提交剩余拆分包（{pending_count} 个） ===\n")
        self._set_task_running(True)
        self._run_workflow_current_step()

    def _on_kill(self):
        self.runner.kill()

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
        self._focus_log_tab()
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
        self._focus_log_tab()
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
                scrollbar = self.log_view.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
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
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_task_running(self, running: bool):
        spec = work_mode_spec(self._current_work_mode())
        self.select_btn.setEnabled(not running)
        self.task_category_combo.setEnabled(not running)
        self.work_task_combo.setEnabled(not running)
        self.doctor_btn.setEnabled(not running)
        self.bootstrap_work_btn.setEnabled(not running)
        self.api_btn.setEnabled(not running)
        bootstrap_ready = self._bootstrap_task_ready(spec)
        self.translate_btn.setEnabled(spec.implemented and bootstrap_ready and not running)
        self.resume_btn.setEnabled(spec.implemented and spec.supports_resume and not running)
        self._update_split_submit_btn(running=running)
        self.save_config_btn.setEnabled(not running)
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
        self.kill_btn.setEnabled(running)

    def _set_workflow_summary(
        self,
        status: str,
        heading: str,
        message: str,
        facts: list[str] | None = None,
    ):
        self.workflow_status_label.setText(heading)
        self.workflow_status_label.setProperty("status", status)
        self.workflow_status_label.style().unpolish(self.workflow_status_label)
        self.workflow_status_label.style().polish(self.workflow_status_label)
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
        self.writeback_status_label.setText(summary.heading)
        self.writeback_status_label.setProperty("status", summary.status)
        self.writeback_status_label.style().unpolish(self.writeback_status_label)
        self.writeback_status_label.style().polish(self.writeback_status_label)
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
        self.doctor_status_label.setText(summary.heading)
        self.doctor_status_label.setProperty("status", summary.status)
        self.doctor_status_label.style().unpolish(self.doctor_status_label)
        self.doctor_status_label.style().polish(self.doctor_status_label)
        self.doctor_message_label.setText(summary.message)
        self.doctor_facts_label.setText("\n".join(summary.facts))
        self._set_details_label(self.doctor_details_label, [])
        self._sync_layout_sizes()

    def _on_runner_error(self, message: str):
        self._append_log(message)
        if self._active_command == "doctor":
            self._doctor_output_lines.append(message)
        elif self._active_command == "translation_workflow":
            self._workflow_step_output_lines.append(message)
        elif self._active_command == "apply":
            self._apply_output_lines.append(message)
        elif self._active_command == "build_retry":
            self._build_retry_output_lines.append(message)
        elif self._active_command in {"bootstrap_rag", "bootstrap_source_index"}:
            self._bootstrap_output_lines.append(message)
        elif self._active_command == "bootstrap_work":
            self._work_bootstrap_output_lines.append(message)
        self._focus_log_tab()
        self.statusBar().showMessage("任务运行失败，请查看诊断日志。", 6000)

    def _on_finished(self, exit_code: int):
        self._append_log(f"\n[进程已结束，退出码：{exit_code}]")
        if self._active_command == "doctor":
            api_key_count, api_key_source = self.state.get_api_key_status()
            summary = summarize_doctor_output(
                "\n".join(self._doctor_output_lines),
                exit_code,
                api_key_count=api_key_count,
                api_key_source=api_key_source,
            )
            self._set_doctor_summary(summary)
            self._active_command = ""
            self._set_task_running(False)
            if exit_code == 0:
                self.statusBar().showMessage("项目检查完成。", 6000)
            else:
                self.statusBar().showMessage(f"项目检查失败（退出码：{exit_code}）", 6000)
            return

        if self._active_command == "translation_workflow":
            self._on_workflow_step_finished(exit_code)
            return

        if self._active_command == "apply":
            summary = summarize_apply_output(
                "\n".join(self._apply_output_lines),
                exit_code,
                manifest_path=self._writeback_manifest_path,
            )
            self._set_writeback_summary(summary)
            self._refresh_diagnostics_context()
            if exit_code == 0:
                self._refresh_workflow_from_latest_manifest()
            self._active_command = ""
            self._set_task_running(False)
            if exit_code == 0:
                self.statusBar().showMessage("翻译写回完成。", 6000)
            else:
                self.statusBar().showMessage("翻译写回失败，请查看诊断日志。", 8000)
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
                        open_remediation_on_confirm=True,
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
                try:
                    self.state.set_game_root(summary.work_dir)
                    self._refresh_project_label()
                except ValueError as exc:
                    self._append_log(f"未能保存项目路径：{exc}")
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
        sync_keyword_completed = step_key == "sync-keywords" and exit_code == 0
        if sync_keyword_completed:
            self._copy_sync_keyword_reports_to_game_parent(step_output)

        if "Safety status:" in step_output:
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
            idx = self.theme_combo.findData(theme)
            if idx >= 0:
                self.theme_combo.setCurrentIndex(idx)
        finally:
            self._loading_theme_to_ui = False

    def _apply_theme(self) -> None:
        if self._qt_app is None:
            return
        try:
            apply_theme(self._qt_app, self._resources_dir, self._theme_preference)
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
        self._set_theme_preference(preference, persist=True)

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

            # Populate sync model
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
                sync_val = sync_config.get("model", "")
            self._set_combo_value(self.sync_model_combo, sync_val)

            # Populate batch model
            batch_val = batch_config.get("model", "")
            self._set_combo_value(self.batch_model_combo, batch_val)

            # Populate sync embedding
            sync_emb_val = sync_rag_config.get("embedding_model", "")
            self._set_combo_value(self.sync_embedding_combo, sync_emb_val)

            # Populate batch embedding
            batch_emb_val = batch_rag_config.get("embedding_model", "")
            self._set_combo_value(self.batch_embedding_combo, batch_emb_val)

            self._on_batch_model_changed(batch_val)
            # Populate thinking level. Missing config keeps the CLI's supported-model
            # default visible; choosing "not enabled" then saves an explicit empty value.
            thinking_val = self._batch_thinking_value_for_load(batch_config, batch_val)
            self._set_batch_thinking_value(thinking_val)
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

            self.state.save_translator_config(config)
            if work_mode_spec(self._current_work_mode()).is_bootstrap:
                self._apply_work_mode_ui(refresh_manifest_writeback=False)
            self._append_log("配置已成功保存至 translator_config.json。")
            self.statusBar().showMessage("配置已成功保存", 3000)
            self._config_ui_saved_snapshot = self._current_config_ui_snapshot()
            return True
        except Exception as exc:
            QMessageBox.warning(self, "保存配置失败", str(exc))
            self._append_log(f"保存配置失败：{exc}")
            return False


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
