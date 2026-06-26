"""Main GUI application for the optional workbench.

This is the first version shell (per #42):
- Pure PySide6 with tabbed layout (workbench / config / diagnostics)
- Delegates everything to the existing CLI via QProcess
- Workbench tab: project selection, doctor + translation workflow status
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
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
)

from .api_key_dialog import ApiKeyDialog
from .api_key_helpers import mask_api_key
from .bootstrap_report import (
    BootstrapSummary,
    read_batch_context_flags,
    running_bootstrap_summary,
    stale_bootstrap_summary,
    summarize_rag_bootstrap_output,
    summarize_source_index_bootstrap_output,
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
)
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
from .user_copy import format_manifest_path_fact
from .work_modes import (
    TASK_CATEGORY_ORDER,
    TaskCategory,
    WorkMode,
    default_work_mode_for_category,
    normalize_task_category,
    normalize_work_mode,
    task_category_for_work_mode,
    task_category_spec,
    work_mode_spec,
    work_modes_for_category,
)
from .workflow_factory import create_workflow, resume_workflow
from .widget_helpers import NoWheelComboBox, NoWheelTabWidget

# Diagnostics splitter: idle favors task context; running tasks expand the log.
_DIAGNOSTICS_IDLE_CONTEXT_PX = 420
_DIAGNOSTICS_IDLE_LOG_PX = 180
_DIAGNOSTICS_RUNNING_CONTEXT_RATIO = 0.32


class MainWindow(QMainWindow):
    def __init__(self, *, qt_app: QApplication | None = None, resources_dir: Path | None = None):
        super().__init__()
        self.setWindowTitle("Ren'Py Translation Lab - 图形工作台（实验版）")
        self.resize(1100, 720)

        self.state = ProjectState()
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
        self._work_mode = WorkMode.BATCH_TRANSLATION
        self._workflow_step_output_lines: list[str] = []
        self._apply_output_lines: list[str] = []
        self._bootstrap_output_lines: list[str] = []
        self._work_bootstrap_output_lines: list[str] = []
        self._build_retry_output_lines: list[str] = []
        self._retry_followup_confirmed: set[str] = set()
        self._writeback_manifest_path = ""

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
        self._apply_work_mode_ui(refresh_manifest_writeback=True)
        self._refresh_diagnostics_context()

        # Status
        self.statusBar().showMessage(
            "图形界面是可选组件；核心命令行不受影响。"
        )

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

        proj_layout.addWidget(QLabel("当前 work 目录："))

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
        mode_outer = QHBoxLayout(mode_frame)
        mode_outer.setContentsMargins(12, 8, 12, 8)
        mode_outer.setSpacing(10)
        mode_outer.addWidget(QLabel("任务类型："))
        self.task_category_combo = NoWheelComboBox()
        self.task_category_combo.setObjectName("task_category_combo")
        for category in TASK_CATEGORY_ORDER:
            self.task_category_combo.addItem(
                task_category_spec(category).label,
                category.value,
            )
        self.task_category_combo.currentIndexChanged.connect(self._on_task_category_changed)
        mode_outer.addWidget(self.task_category_combo)

        mode_outer.addWidget(QLabel("子任务："))
        self.work_task_combo = NoWheelComboBox()
        self.work_task_combo.setObjectName("work_task_combo")
        self.work_task_combo.currentIndexChanged.connect(self._on_work_task_changed)
        mode_outer.addWidget(self.work_task_combo, 1)
        self.work_mode_hint_label = QLabel()
        self.work_mode_hint_label.setWordWrap(True)
        self.work_mode_hint_label.setObjectName("config_hint_label")
        mode_outer.addWidget(self.work_mode_hint_label, 2)
        layout.addWidget(mode_frame)

        action_frame = QFrame()
        action_frame.setObjectName("action_frame")
        action_outer = QHBoxLayout(action_frame)
        action_outer.setContentsMargins(12, 10, 12, 10)
        action_outer.setSpacing(12)

        prep_group = QHBoxLayout()
        prep_group.setSpacing(8)
        prep_label = QLabel("项目准备")
        prep_label.setObjectName("action_group_label")
        prep_group.addWidget(prep_label)
        self.doctor_btn = QPushButton("环境检查")
        self.doctor_btn.setObjectName("secondary_btn")
        self.doctor_btn.clicked.connect(self._on_run_doctor)
        prep_group.addWidget(self.doctor_btn)
        self.bootstrap_work_btn = QPushButton("准备工作目录")
        self.bootstrap_work_btn.setObjectName("secondary_btn")
        self.bootstrap_work_btn.clicked.connect(self._on_bootstrap_work)
        prep_group.addWidget(self.bootstrap_work_btn)
        action_outer.addLayout(prep_group)

        action_separator = QFrame()
        action_separator.setFrameShape(QFrame.Shape.VLine)
        action_separator.setObjectName("action_separator")
        action_outer.addWidget(action_separator)

        translate_group = QHBoxLayout()
        translate_group.setSpacing(8)
        self.translate_group_label = QLabel("翻译任务")
        self.translate_group_label.setObjectName("action_group_label")
        translate_group.addWidget(self.translate_group_label)
        self.translate_btn = QPushButton("开始翻译")
        self.translate_btn.setObjectName("translate_btn")
        self.translate_btn.clicked.connect(self._on_start_translation)
        translate_group.addWidget(self.translate_btn)
        self.resume_btn = QPushButton("继续翻译")
        self.resume_btn.setObjectName("secondary_btn")
        self.resume_btn.clicked.connect(self._on_resume_translation)
        translate_group.addWidget(self.resume_btn)
        action_outer.addLayout(translate_group)

        action_outer.addStretch()

        self.kill_btn = QPushButton("停止")
        self.kill_btn.setObjectName("kill_btn")
        self.kill_btn.clicked.connect(self._on_kill)
        self.kill_btn.setEnabled(False)
        action_outer.addWidget(self.kill_btn)
        layout.addWidget(action_frame)

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
        self.doctor_message_label = QLabel()
        self.doctor_message_label.setWordWrap(True)
        self.doctor_message_label.setObjectName("summary_body_label")
        doctor_layout.addWidget(self.doctor_message_label)
        self.doctor_facts_label = QLabel()
        self.doctor_facts_label.setWordWrap(True)
        self.doctor_facts_label.setObjectName("doctor_facts_label")
        doctor_layout.addWidget(self.doctor_facts_label)
        self.doctor_details_label = QLabel()
        self.doctor_details_label.setWordWrap(True)
        self.doctor_details_label.setObjectName("config_hint_label")
        self.doctor_details_label.setVisible(False)
        doctor_layout.addWidget(self.doctor_details_label)
        doctor_layout.addStretch()
        self.workbench_status_tabs.addTab(doctor_tab, "环境检查")

        workflow_tab = QWidget()
        self._style_themed_surface(workflow_tab)
        workflow_layout = QVBoxLayout(workflow_tab)
        workflow_layout.setContentsMargins(12, 12, 12, 12)
        workflow_layout.setSpacing(6)
        self.workflow_status_label = QLabel()
        self.workflow_status_label.setObjectName("workflow_status_label")
        workflow_layout.addWidget(self.workflow_status_label)
        self.workflow_message_label = QLabel()
        self.workflow_message_label.setWordWrap(True)
        self.workflow_message_label.setObjectName("summary_body_label")
        workflow_layout.addWidget(self.workflow_message_label)
        self.workflow_facts_label = QLabel()
        self.workflow_facts_label.setWordWrap(True)
        self.workflow_facts_label.setObjectName("workflow_facts_label")
        workflow_layout.addWidget(self.workflow_facts_label)
        workflow_layout.addStretch()
        self.workbench_status_tabs.addTab(workflow_tab, "翻译进度")

        writeback_tab = QWidget()
        self._style_themed_surface(writeback_tab)
        writeback_layout = QVBoxLayout(writeback_tab)
        writeback_layout.setContentsMargins(12, 12, 12, 12)
        writeback_layout.setSpacing(6)
        self.writeback_status_label = QLabel()
        self.writeback_status_label.setObjectName("writeback_status_label")
        writeback_layout.addWidget(self.writeback_status_label)
        self.writeback_message_label = QLabel()
        self.writeback_message_label.setWordWrap(True)
        self.writeback_message_label.setObjectName("summary_body_label")
        writeback_layout.addWidget(self.writeback_message_label)
        self.writeback_facts_label = QLabel()
        self.writeback_facts_label.setWordWrap(True)
        self.writeback_facts_label.setObjectName("writeback_facts_label")
        writeback_layout.addWidget(self.writeback_facts_label)
        self.writeback_details_label = QLabel()
        self.writeback_details_label.setWordWrap(True)
        self.writeback_details_label.setObjectName("config_hint_label")
        self.writeback_details_label.setVisible(False)
        writeback_layout.addWidget(self.writeback_details_label)
        writeback_actions = QHBoxLayout()
        self.apply_btn = QPushButton("写回翻译")
        self.apply_btn.setObjectName("apply_btn")
        self.apply_btn.clicked.connect(self._on_apply_writeback)
        self.apply_btn.setEnabled(False)
        writeback_actions.addWidget(self.apply_btn)
        self.check_issues_btn = QPushButton("查看问题清单")
        self.check_issues_btn.setObjectName("secondary_btn")
        self.check_issues_btn.clicked.connect(self._open_check_issues)
        self.check_issues_btn.setEnabled(False)
        writeback_actions.addWidget(self.check_issues_btn)
        self.retry_btn = QPushButton("生成 retry 包")
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
            "RAG 使用已有译文；原文索引只使用 TL 模板原文；均不修改 .rpy 文件。"
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
            "上方可查看任务上下文、命令参考与任务清单；下方显示原始命令输出。"
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
        self.diagnostics_inner_tabs.addTab(manifest_tab, "任务清单")

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
        context_size = int(total * _DIAGNOSTICS_RUNNING_CONTEXT_RATIO)
        self.diagnostics_splitter.setSizes([context_size, total - context_size])

    def _focus_workbench_status_tab(self, index: int) -> None:
        if 0 <= index < self.workbench_status_tabs.count():
            self.workbench_status_tabs.setCurrentIndex(index)

    def _writeback_issues_ready(self, summary: WritebackSummary) -> bool:
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
            QMessageBox.warning(self, "无法查看写回失败报告", "当前没有可用的任务清单。")
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
        issues_ready = self._writeback_issues_ready(summary)
        manifest = self._load_writeback_manifest() if summary.manifest_path else None

        if hasattr(self, "check_issues_btn"):
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
                    "查看 retry 包" if mode == "view" else "生成 retry 包"
                )
                self.retry_btn.setEnabled(not running)

        if hasattr(self, "remediation_btn"):
            remediation_ready = (
                self._remediation_ready(summary, manifest=manifest)
                if issues_ready
                else False
            )
            self.remediation_btn.setEnabled(not running and remediation_ready)

    def _show_retry_preview(
        self,
        retry_manifest_path: str,
        *,
        open_remediation_on_confirm: bool = False,
    ) -> None:
        try:
            retry_manifest = self.state.load_manifest_file(retry_manifest_path)
        except ValueError as exc:
            QMessageBox.warning(self, "无法预览 retry 包", str(exc))
            return

        report = summarize_retry_manifest(
            retry_manifest,
            manifest_path=retry_manifest_path,
        )
        dialog = RetryPreviewDialog(self, report=report)
        dialog.exec()
        if not dialog.confirmed:
            return

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
            self.statusBar().showMessage("已确认 retry 包范围。", 3000)

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
            QMessageBox.warning(self, "无法生成 retry 包", "无法读取当前任务清单。")
            return

        eligibility = assess_retry_eligibility(
            manifest,
            manifest_path=self._writeback_manifest_path,
        )
        reply = QMessageBox.question(
            self,
            "确认生成 retry 包",
            "\n".join(
                [
                    eligibility.message,
                    "",
                    "将运行 build-retry 生成本地 retry 包。",
                    "GUI 不会自动提交云端任务；生成后会先预览范围。",
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
            "\n=== 正在生成 retry 包："
            f"gemini_translate_batch.py {' '.join(build_retry_cli_args(self._writeback_manifest_path))} ===\n"
        )
        self.runner.run(
            self.state.get_batch_script_path(),
            build_retry_cli_args(self._writeback_manifest_path),
        )

    def _open_check_issues(self) -> None:
        manifest_path = self._writeback_manifest_path
        if not manifest_path:
            QMessageBox.warning(self, "无法查看问题清单", "当前没有可用的任务清单。")
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
        self.log_view.clear()
        self.statusBar().showMessage("诊断日志已清空。", 3000)

    def _resolve_diagnostics_manifest_path(self) -> str | None:
        if self._workflow is not None and self._workflow.manifest_path:
            return self._workflow.manifest_path
        if self._writeback_manifest_path:
            return self._writeback_manifest_path
        latest_manifest = self.state.get_latest_manifest_path()
        return str(latest_manifest) if latest_manifest is not None else None

    def _load_diagnostics_manifest(self, manifest_path: str | None) -> dict[str, object] | None:
        if not manifest_path:
            return None
        try:
            return self.state.load_manifest_file(manifest_path)
        except ValueError:
            return None

    def _refresh_diagnostics_context(self) -> None:
        latest_manifest = self.state.get_latest_manifest_path()
        manifest_path = self._resolve_diagnostics_manifest_path()
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
        self._apply_work_mode_ui(refresh_manifest_writeback=refresh_manifest_writeback)

    def _apply_work_mode_ui(self, *, refresh_manifest_writeback: bool = False) -> None:
        spec = work_mode_spec(self._current_work_mode())
        self._sync_task_selectors_from_work_mode()
        self.translate_group_label.setText(spec.task_group_label)
        self.translate_btn.setText(spec.start_button_label)
        if spec.resume_button_label:
            self.resume_btn.setText(spec.resume_button_label)
        self.resume_btn.setVisible(spec.supports_resume)
        self.workbench_status_tabs.setTabText(1, spec.progress_tab_label)
        self.workbench_status_tabs.setTabText(2, spec.writeback_tab_label)
        if spec.implemented:
            if spec.is_bootstrap and not self._bootstrap_task_ready(spec):
                hint = self._bootstrap_disabled_message(spec.bootstrap_kind)
            else:
                hint = spec.idle_workflow_message
        else:
            hint = spec.not_implemented_message
        self.work_mode_hint_label.setText(hint)
        self._refresh_workflow_idle_summary()
        if refresh_manifest_writeback:
            self._refresh_writeback_from_latest_manifest()
        running = self.kill_btn.isEnabled()
        bootstrap_ready = self._bootstrap_task_ready(spec)
        self.translate_btn.setEnabled(spec.implemented and bootstrap_ready and not running)
        self.resume_btn.setEnabled(spec.implemented and spec.supports_resume and not running)

    def _refresh_workflow_idle_summary(self) -> None:
        if self.kill_btn.isEnabled():
            return
        spec = work_mode_spec(self._current_work_mode())
        self._set_workflow_summary(
            "idle",
            spec.idle_workflow_heading,
            spec.idle_workflow_message,
        )

    def _set_workflow_from_bootstrap_summary(self, summary: BootstrapSummary) -> None:
        self._set_workflow_summary(
            summary.status,
            summary.heading,
            summary.message,
            summary.facts,
        )

    def _refresh_writeback_from_latest_manifest(self) -> None:
        spec = work_mode_spec(self._current_work_mode())
        if not spec.supports_translation_writeback:
            self._set_writeback_summary(idle_writeback_summary_for_work_mode(spec.mode))
            return

        latest_manifest = self.state.get_latest_manifest_path()
        if latest_manifest is None:
            self._set_writeback_summary(idle_writeback_summary())
            return
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

    def _on_tab_changed(self, index: int) -> None:
        if self.tab_widget.widget(index) is self._config_tab:
            self._refresh_api_status()
        if self.tab_widget.widget(index) is getattr(self, "_diagnostics_tab", None):
            self._refresh_diagnostics_context()

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

        self.log_view.clear()
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

        self.log_view.clear()
        self._active_command = "bootstrap_work"
        self._work_bootstrap_output_lines = []
        self._focus_workbench_status_tab(0)
        doctor_summary = work_bootstrap_to_doctor_summary(running_work_bootstrap_summary())
        self._set_doctor_summary(doctor_summary)
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

    def _bootstrap_disabled_message(self, kind: str) -> str:
        if kind == "rag":
            return "请先在配置页勾选「启用 RAG 记忆库」，并点击「保存参数配置」。"
        return "请先在配置页勾选「启用原文索引」，并点击「保存参数配置」。"

    def _start_bootstrap_task(self, kind: str) -> bool:
        if not self.state.get_game_root():
            QMessageBox.information(self, "请先选择项目", "请先选择游戏的 work 目录。")
            return False

        flags = self._saved_batch_context_flags()
        if kind == "rag":
            if not flags["rag_enabled"]:
                QMessageBox.information(
                    self,
                    "RAG 未启用",
                    self._bootstrap_disabled_message("rag"),
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
                    self._bootstrap_disabled_message("source_index"),
                )
                return False
            command = "bootstrap_source_index"
            args = ["bootstrap-source-index", "--skip-prepare"]
            log_heading = "gemini_translate_batch.py bootstrap-source-index --skip-prepare"
            running_summary = running_bootstrap_summary("source_index")

        self.log_view.clear()
        self._focus_log_tab()
        self._active_command = command
        self._bootstrap_output_lines = []
        self._focus_workbench_status_tab(1)
        self._set_workflow_from_bootstrap_summary(running_summary)
        self._append_log(f"=== 正在运行：{log_heading} ===\n")
        self._set_task_running(True)
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

        if spec.mode == WorkMode.SYNC_TRANSLATION:
            api_key_count, _ = self.state.get_api_key_status()
            if api_key_count == 0:
                QMessageBox.information(
                    self,
                    "请先配置 API Key",
                    "同步翻译需要 Gemini API 密钥；请在配置页管理 API Key 或设置环境变量。",
                )
                return

        workflow = create_workflow(spec.mode)
        if workflow is None:
            QMessageBox.information(self, "无法开始任务", spec.not_implemented_message)
            return

        self.log_view.clear()
        self._focus_log_tab()
        self._writeback_manifest_path = ""
        self._set_writeback_summary(stale_writeback_summary())
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

        latest_manifest = self.state.get_latest_manifest_path()
        if latest_manifest is None:
            QMessageBox.information(
                self,
                "没有可继续的任务",
                f"未找到最近任务清单；请先开始一个{spec.label}任务。",
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

        workflow = resume_workflow(spec.mode, str(latest_manifest), manifest)
        if workflow is None:
            QMessageBox.information(self, "无法继续任务", spec.not_implemented_message)
            return

        self.log_view.clear()
        self._focus_log_tab()
        self._workflow = workflow
        self._refresh_diagnostics_context()
        self._active_command = "translation_workflow"
        self._workflow_step_output_lines = []
        self._focus_workbench_status_tab(1)
        self._append_log(f"=== 正在继续最新 {spec.label} 任务 ===\n")
        self._set_task_running(True)
        self._run_workflow_current_step()

    def _on_kill(self):
        self.runner.kill()

    def _on_apply_writeback(self):
        if not work_mode_spec(self._current_work_mode()).supports_translation_writeback:
            QMessageBox.information(
                self,
                "当前模式不支持",
                "普通「写回翻译」仅适用于 Batch 翻译模式。",
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
            "即将把翻译写回项目 .rpy 文件。",
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

        self.log_view.clear()
        self._focus_log_tab()
        self._active_command = "apply"
        self._apply_output_lines = []
        self._focus_workbench_status_tab(2)
        self._set_writeback_summary(running_writeback_summary())
        self._append_log(
            f"=== 正在写回：gemini_translate_batch.py apply {self._writeback_manifest_path} ===\n"
        )
        self._set_task_running(True)
        self.runner.run(
            self.state.get_batch_script_path(),
            ["apply", self._writeback_manifest_path],
        )

    # --- Runner callbacks ---

    def _on_cli_line_ready(self, text: str):
        if self._active_command == "doctor":
            self._doctor_output_lines.append(text)
        elif self._active_command == "translation_workflow":
            self._workflow_step_output_lines.append(text)
        elif self._active_command == "apply":
            self._apply_output_lines.append(text)
        elif self._active_command == "build_retry":
            self._build_retry_output_lines.append(text)
        elif self._active_command in {"bootstrap_rag", "bootstrap_source_index"}:
            self._bootstrap_output_lines.append(text)
        elif self._active_command == "bootstrap_work":
            self._work_bootstrap_output_lines.append(text)
        self._append_log(text)

    def _append_log(self, text: str):
        self.log_view.append(text.rstrip("\n"))
        # scroll to bottom
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
        self.save_config_btn.setEnabled(not running)
        writeback_summary = self._current_writeback_summary()
        self.apply_btn.setEnabled(not running and writeback_summary.can_apply)
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

    def _set_workflow_update(self, update: WorkflowUpdate):
        self._set_workflow_summary(
            update.status,
            update.heading,
            update.message,
            update.facts,
        )

    def _run_workflow_current_step(self):
        if self._workflow is None:
            self._set_task_running(False)
            return

        step = self._workflow.current_step()
        if step is None:
            self._set_task_running(False)
            self._active_command = ""
            self._workflow = None
            return

        facts = []
        if self._workflow.manifest_path:
            facts.append(format_manifest_path_fact(self._workflow.manifest_path))
        self._workflow_step_output_lines = []
        self._set_workflow_summary("running", step.heading, step.message, facts)
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
            self.apply_btn.setEnabled(summary.can_apply)

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
            self._active_command = ""
            self._set_task_running(False)
            if exit_code == 0:
                self.statusBar().showMessage("翻译写回完成。", 6000)
            else:
                self.statusBar().showMessage("翻译写回失败，请查看诊断日志。", 8000)
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
                if retry_path:
                    self._show_retry_preview(
                        retry_path,
                        open_remediation_on_confirm=True,
                    )
                self.statusBar().showMessage("retry 包已生成，请先确认预览范围。", 6000)
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
                self.statusBar().showMessage("RAG 预建完成。", 6000)
            elif exit_code == 0:
                self.statusBar().showMessage("RAG 预建已结束，请查看摘要。", 6000)
            else:
                self.statusBar().showMessage("RAG 预建失败，请查看诊断日志。", 8000)
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
                    self._append_log(f"更新 translator_config.json 失败：{exc}")
                    game_root_update_failed = True
                    summary = with_game_root_persist_warning(summary)
            self._set_doctor_summary(work_bootstrap_to_doctor_summary(summary))
            self._active_command = ""
            self._set_task_running(False)
            if game_root_update_failed:
                self.statusBar().showMessage(
                    "工作目录已复制，但更新 game_root 失败，请查看诊断日志。",
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

    def _on_workflow_step_finished(self, exit_code: int):
        if self._workflow is None:
            self._active_command = ""
            self._set_task_running(False)
            return

        step_output = "\n".join(self._workflow_step_output_lines)
        manifest_path = self._workflow.manifest_path
        update = self._workflow.complete_current_step(exit_code, step_output)
        if "Safety status:" in step_output:
            self._update_writeback_from_check(step_output, exit_code, manifest_path)
        self._set_workflow_update(update)
        self._workflow_step_output_lines = []
        self._refresh_diagnostics_context()

        if update.should_continue:
            self.statusBar().showMessage(update.heading, 3000)
            QTimer.singleShot(0, self._run_workflow_current_step)
            return

        self._active_command = ""
        self._workflow = None
        self._set_task_running(False)
        if update.status == "failed":
            self.statusBar().showMessage("翻译任务失败，请查看诊断日志。", 8000)
        elif update.status == "waiting":
            self.statusBar().showMessage("批量任务仍在处理，可稍后继续最新任务。", 8000)
        elif update.status == "done":
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

    def _on_save_config(self):
        if not self.state.get_game_root():
            QMessageBox.information(self, "未选择项目", "请先选择游戏的 work 目录。")
            return

        try:
            config = self.state.load_translator_config()
            sync_config = self._ensure_config_section(config, "sync")
            batch_config = self._ensure_config_section(config, "batch")
            sync_rag_config = self._ensure_config_section(sync_config, "rag")
            batch_rag_config = self._ensure_config_section(batch_config, "rag")
            batch_source_index_config = self._ensure_config_section(batch_config, "source_index")

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
        except Exception as exc:
            QMessageBox.warning(self, "保存配置失败", str(exc))
            self._append_log(f"保存配置失败：{exc}")


def run_app(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    app = QApplication(argv)
    resources_dir = Path(__file__).resolve().parent / "resources"
    bootstrap_state = ProjectState()
    try:
        bootstrap_config = bootstrap_state.load_translator_config()
        theme_preference = read_gui_theme_from_config(bootstrap_config)
    except Exception as exc:
        print(f"警告：无法读取主题配置，将使用系统跟随：{exc}")
        theme_preference = DEFAULT_THEME_PREFERENCE
    try:
        apply_theme(app, resources_dir, theme_preference)
    except OSError as exc:
        print(f"警告：无法加载 GUI 样式表：{exc}")

    win = MainWindow(qt_app=app, resources_dir=resources_dir)
    app.styleHints().colorSchemeChanged.connect(win._on_system_color_scheme_changed)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
