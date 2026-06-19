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
    QLayout,
)

from .api_key_dialog import ApiKeyDialog
from .api_key_helpers import mask_api_key
from .bootstrap_report import (
    BootstrapSummary,
    idle_bootstrap_summary,
    read_batch_context_flags,
    running_bootstrap_summary,
    stale_bootstrap_summary,
    summarize_rag_bootstrap_output,
    summarize_source_index_bootstrap_output,
)
from .check_report import (
    WritebackSummary,
    idle_writeback_summary,
    running_writeback_summary,
    stale_writeback_summary,
    summarize_apply_output,
    summarize_check_output,
    summarize_manifest_writeback,
)
from .diagnostics_context import (
    DiagnosticsContext,
    build_diagnostics_context,
)
from .cli_runner import CliRunner
from .doctor_report import (
    DoctorSummary,
    idle_summary,
    running_summary,
    stale_summary,
    summarize_doctor_output,
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
from .translation_workflow import TranslationWorkflow, WorkflowUpdate
from .widget_helpers import NoWheelComboBox, NoWheelTabWidget


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
        self._workflow: TranslationWorkflow | None = None
        self._workflow_step_output_lines: list[str] = []
        self._apply_output_lines: list[str] = []
        self._bootstrap_output_lines: list[str] = []
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
        self._refresh_api_status()
        self._load_config_to_ui()
        self._set_doctor_summary(idle_summary())
        self._set_workflow_summary(
            "idle",
            "尚未开始翻译任务",
            "完成环境检查后，可以开始基础 Batch 翻译流程。",
        )
        self._refresh_writeback_from_latest_manifest()
        self._set_bootstrap_summary(idle_bootstrap_summary())
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
        proj_layout = QHBoxLayout(project_frame)
        proj_layout.setContentsMargins(12, 10, 12, 10)
        proj_layout.setSpacing(10)

        proj_layout.addWidget(QLabel("当前游戏 work 目录："))

        self.project_path_edit = QLineEdit("尚未选择项目")
        self.project_path_edit.setReadOnly(True)
        self.project_path_edit.setObjectName("project_path_edit")
        proj_layout.addWidget(self.project_path_edit, 1)

        self.select_btn = QPushButton("选择游戏目录...")
        self.select_btn.clicked.connect(self._on_select_project)
        proj_layout.addWidget(self.select_btn)

        layout.addWidget(project_frame)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.doctor_btn = QPushButton("环境检查")
        self.doctor_btn.setObjectName("doctor_btn")
        self.doctor_btn.clicked.connect(self._on_run_doctor)
        action_row.addWidget(self.doctor_btn)

        self.translate_btn = QPushButton("开始翻译")
        self.translate_btn.setObjectName("translate_btn")
        self.translate_btn.clicked.connect(self._on_start_translation)
        action_row.addWidget(self.translate_btn)

        self.resume_btn = QPushButton("继续任务")
        self.resume_btn.setObjectName("secondary_btn")
        self.resume_btn.clicked.connect(self._on_resume_translation)
        action_row.addWidget(self.resume_btn)

        action_row.addStretch()

        self.kill_btn = QPushButton("停止")
        self.kill_btn.setObjectName("kill_btn")
        self.kill_btn.clicked.connect(self._on_kill)
        self.kill_btn.setEnabled(False)
        action_row.addWidget(self.kill_btn)
        layout.addLayout(action_row)

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
            "翻译任务需要 Gemini API Key。Key 保存在本地 api_keys.json，"
            "不会上传或代理。也可通过环境变量 GEMINI_API_KEY 配置。"
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

        context_box = QGroupBox("Batch 上下文")
        context_layout = QVBoxLayout(context_box)
        context_layout.setSpacing(8)
        context_layout.setContentsMargins(12, 16, 12, 12)

        context_hint = QLabel(
            "启用后先保存配置，再运行下方预建按钮。"
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

        bootstrap_actions = QHBoxLayout()
        bootstrap_actions.setSpacing(10)
        self.bootstrap_rag_btn = QPushButton("预建 RAG 库")
        self.bootstrap_rag_btn.setObjectName("secondary_btn")
        self.bootstrap_rag_btn.clicked.connect(self._on_bootstrap_rag)
        bootstrap_actions.addWidget(self.bootstrap_rag_btn)

        self.bootstrap_source_index_btn = QPushButton("预建原文索引")
        self.bootstrap_source_index_btn.setObjectName("secondary_btn")
        self.bootstrap_source_index_btn.clicked.connect(self._on_bootstrap_source_index)
        bootstrap_actions.addWidget(self.bootstrap_source_index_btn)
        bootstrap_actions.addStretch()
        context_layout.addLayout(bootstrap_actions)

        self.bootstrap_status_label = QLabel()
        self.bootstrap_status_label.setObjectName("bootstrap_status_label")
        context_layout.addWidget(self.bootstrap_status_label)

        self.bootstrap_message_label = QLabel()
        self.bootstrap_message_label.setWordWrap(True)
        context_layout.addWidget(self.bootstrap_message_label)

        self.bootstrap_facts_label = QLabel()
        self.bootstrap_facts_label.setWordWrap(True)
        self.bootstrap_facts_label.setObjectName("bootstrap_facts_label")
        context_layout.addWidget(self.bootstrap_facts_label)

        self.bootstrap_details_label = QLabel()
        self.bootstrap_details_label.setWordWrap(True)
        self.bootstrap_details_label.setObjectName("config_hint_label")
        self.bootstrap_details_label.setVisible(False)
        context_layout.addWidget(self.bootstrap_details_label)

        layout.addWidget(context_box)

        config_row = QHBoxLayout()
        config_row.setSpacing(16)

        sync_box = QGroupBox("同步翻译 (Sync API)")
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

        batch_box = QGroupBox("批量离线 (Batch API)")
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
        self.batch_thinking_combo.addItem("最小 (minimal)", "minimal")
        self.batch_thinking_combo.addItem("低 (low)", "low")
        self.batch_thinking_combo.addItem("中 (medium)", "medium")
        self.batch_thinking_combo.addItem("高 (high)", "high")
        batch_layout.addRow("Batch 思考程度：", self.batch_thinking_combo)

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

        diag_hint = QLabel(
            "上半部分展示任务上下文、报告路径、可复制 CLI 命令和 manifest 预览；"
            "下方为原始终端输出。翻译、预建库与写回运行时会自动切换到此页。"
        )
        diag_hint.setWordWrap(True)
        diag_hint.setObjectName("config_hint_label")
        layout.addWidget(diag_hint)

        diag_scroll = QScrollArea()
        diag_scroll.setObjectName("diagnostics_scroll")
        self._style_themed_surface(diag_scroll)
        diag_scroll.setWidgetResizable(True)
        diag_scroll.setFrameShape(QFrame.Shape.NoFrame)
        diag_viewport = diag_scroll.viewport()
        diag_viewport.setObjectName("diagnostics_scroll_viewport")
        self._style_themed_surface(diag_viewport)

        diag_content = QWidget()
        diag_content.setObjectName("diagnostics_scroll_content")
        self._style_themed_surface(diag_content)
        diag_layout = QVBoxLayout(diag_content)
        diag_layout.setContentsMargins(0, 0, 0, 0)
        diag_layout.setSpacing(12)

        context_box = QGroupBox("任务上下文")
        context_layout = QVBoxLayout(context_box)
        context_layout.setSpacing(6)
        context_layout.setContentsMargins(12, 16, 12, 12)
        self.diagnostics_status_label = QLabel()
        self.diagnostics_status_label.setObjectName("diagnostics_status_label")
        context_layout.addWidget(self.diagnostics_status_label)
        self.diagnostics_message_label = QLabel()
        self.diagnostics_message_label.setWordWrap(True)
        context_layout.addWidget(self.diagnostics_message_label)
        self.diagnostics_facts_label = QLabel()
        self.diagnostics_facts_label.setWordWrap(True)
        self.diagnostics_facts_label.setObjectName("diagnostics_facts_label")
        context_layout.addWidget(self.diagnostics_facts_label)
        diag_layout.addWidget(context_box)

        reports_box = QGroupBox("报告与数据文件")
        reports_layout = QVBoxLayout(reports_box)
        reports_layout.setSpacing(6)
        reports_layout.setContentsMargins(12, 16, 12, 12)
        self.diagnostics_paths_label = QLabel()
        self.diagnostics_paths_label.setWordWrap(True)
        self.diagnostics_paths_label.setObjectName("diagnostics_paths_label")
        reports_layout.addWidget(self.diagnostics_paths_label)
        diag_layout.addWidget(reports_box)

        commands_box = QGroupBox("手动 CLI 命令")
        commands_layout = QVBoxLayout(commands_box)
        commands_layout.setSpacing(8)
        commands_layout.setContentsMargins(12, 16, 12, 12)
        self.diagnostics_commands_host = QWidget()
        self.diagnostics_commands_layout = QVBoxLayout(self.diagnostics_commands_host)
        self.diagnostics_commands_layout.setContentsMargins(0, 0, 0, 0)
        self.diagnostics_commands_layout.setSpacing(8)
        commands_layout.addWidget(self.diagnostics_commands_host)
        diag_layout.addWidget(commands_box)

        manifest_box = QGroupBox("Manifest 预览")
        manifest_layout = QVBoxLayout(manifest_box)
        manifest_layout.setContentsMargins(12, 16, 12, 12)
        self.diagnostics_manifest_preview = QTextEdit()
        self.diagnostics_manifest_preview.setReadOnly(True)
        self.diagnostics_manifest_preview.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.diagnostics_manifest_preview.setObjectName("diagnostics_manifest_preview")
        self.diagnostics_manifest_preview.setMaximumHeight(220)
        manifest_layout.addWidget(self.diagnostics_manifest_preview)
        diag_layout.addWidget(manifest_box)

        diag_scroll.setWidget(diag_content)
        layout.addWidget(diag_scroll, 2)

        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("原始 CLI 输出"))
        log_header.addStretch()
        self.refresh_diagnostics_btn = QPushButton("刷新上下文")
        self.refresh_diagnostics_btn.setObjectName("secondary_btn")
        self.refresh_diagnostics_btn.clicked.connect(self._refresh_diagnostics_context)
        log_header.addWidget(self.refresh_diagnostics_btn)
        layout.addLayout(log_header)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log_view.setObjectName("log_view")
        layout.addWidget(self.log_view, 3)

        self.tab_widget.addTab(tab, "诊断日志")

    def _focus_log_tab(self) -> None:
        self.tab_widget.setCurrentWidget(self.log_view.parentWidget())

    def _focus_workbench_status_tab(self, index: int) -> None:
        if 0 <= index < self.workbench_status_tabs.count():
            self.workbench_status_tabs.setCurrentIndex(index)

    def _set_details_label(self, label: QLabel, findings: list[str]) -> None:
        if findings:
            label.setText("\n".join(f"- {item}" for item in findings))
            label.setVisible(True)
        else:
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

    def _copy_to_clipboard(self, text: str) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self.statusBar().showMessage("命令已复制到剪贴板。", 3000)

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

        if context.paths:
            self.diagnostics_paths_label.setText(
                "\n".join(f"{entry.label}：{entry.path}" for entry in context.paths)
            )
        else:
            self.diagnostics_paths_label.setText("暂无已生成的报告或数据文件。")

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
            placeholder = QLabel("开始翻译任务后，这里会出现可复制的手动 CLI 命令。")
            placeholder.setWordWrap(True)
            placeholder.setObjectName("config_hint_label")
            self.diagnostics_commands_layout.addWidget(placeholder)

        self.diagnostics_manifest_preview.setPlainText(context.manifest_json_preview)

    # --- UI actions ---

    def _on_select_project(self):
        start_dir = str(self.state.get_game_root() or Path.home())
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择游戏的 work 目录（通常包含 game/tl/schinese）",
            start_dir,
        )
        if directory:
            try:
                self.state.set_game_root(directory)
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
            self._set_workflow_summary(
                "stale",
                "项目已切换",
                "翻译任务状态已清空；请先针对新项目重新检查。",
            )
            self._writeback_manifest_path = ""
            self._set_writeback_summary(stale_writeback_summary())
            self._set_bootstrap_summary(stale_bootstrap_summary())
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

    def _refresh_project_label(self):
        root = self.state.get_game_root()
        if root:
            self.project_path_edit.setText(str(root))
        else:
            self.project_path_edit.setText("（尚未选择项目）")

    def _on_run_doctor(self):
        if not self.state.get_game_root():
            QMessageBox.information(
                self, "请先选择项目",
                "请先选择游戏的 work 目录。\n"
                "项目检查（doctor）会读取 translator_config.json 中的 game_root。"
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

    def _saved_batch_context_flags(self) -> dict[str, bool]:
        return read_batch_context_flags(self.state.load_translator_config())

    def _on_bootstrap_rag(self) -> None:
        if not self.state.get_game_root():
            QMessageBox.information(self, "请先选择项目", "请先选择游戏的 work 目录。")
            return
        if not self._saved_batch_context_flags()["rag_enabled"]:
            QMessageBox.information(
                self,
                "RAG 未启用",
                "请先启用 Batch RAG 记忆库，并点击「保存参数配置」。",
            )
            return

        self.log_view.clear()
        self._focus_log_tab()
        self._active_command = "bootstrap_rag"
        self._bootstrap_output_lines = []
        self._set_bootstrap_summary(running_bootstrap_summary("rag"))
        self._append_log("=== 正在运行：gemini_translate_batch.py bootstrap-rag --skip-prepare ===\n")
        self._set_task_running(True)
        self.runner.run(self.state.get_batch_script_path(), ["bootstrap-rag", "--skip-prepare"])

    def _on_bootstrap_source_index(self) -> None:
        if not self.state.get_game_root():
            QMessageBox.information(self, "请先选择项目", "请先选择游戏的 work 目录。")
            return
        if not self._saved_batch_context_flags()["source_index_enabled"]:
            QMessageBox.information(
                self,
                "原文索引未启用",
                "请先启用 Batch 原文索引，并点击「保存参数配置」。",
            )
            return

        self.log_view.clear()
        self._focus_log_tab()
        self._active_command = "bootstrap_source_index"
        self._bootstrap_output_lines = []
        self._set_bootstrap_summary(running_bootstrap_summary("source_index"))
        self._append_log(
            "=== 正在运行：gemini_translate_batch.py bootstrap-source-index --skip-prepare ===\n"
        )
        self._set_task_running(True)
        self.runner.run(
            self.state.get_batch_script_path(),
            ["bootstrap-source-index", "--skip-prepare"],
        )

    def _on_start_translation(self):
        if not self.state.get_game_root():
            QMessageBox.information(
                self,
                "请先选择项目",
                "请先选择游戏的 work 目录。",
            )
            return

        self.log_view.clear()
        self._focus_log_tab()
        self._writeback_manifest_path = ""
        self._set_writeback_summary(stale_writeback_summary())
        self._workflow = TranslationWorkflow.start_new()
        self._active_command = "translation_workflow"
        self._workflow_step_output_lines = []
        self._focus_workbench_status_tab(1)
        self._append_log("=== 正在运行：基础 Batch 翻译流程 ===\n")
        self._set_task_running(True)
        self._run_workflow_current_step()

    def _on_resume_translation(self):
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
                "未找到 latest manifest；请先开始一个翻译任务。",
            )
            return

        try:
            manifest = self.state.load_resume_manifest(latest_manifest)
        except ValueError as exc:
            QMessageBox.warning(self, "无法继续最新任务", str(exc))
            return

        self.log_view.clear()
        self._focus_log_tab()
        self._workflow = TranslationWorkflow.resume_manifest(str(latest_manifest), manifest)
        self._refresh_diagnostics_context()
        self._active_command = "translation_workflow"
        self._workflow_step_output_lines = []
        self._focus_workbench_status_tab(1)
        self._append_log("=== 正在继续最新 Batch 翻译任务 ===\n")
        self._set_task_running(True)
        self._run_workflow_current_step()

    def _on_kill(self):
        self.runner.kill()

    def _on_apply_writeback(self):
        if not self._writeback_manifest_path:
            QMessageBox.information(self, "无法写回", "没有可写回的 manifest；请先完成 check。")
            return

        summary = self._current_writeback_summary()
        if not summary.can_apply:
            QMessageBox.information(
                self,
                "当前不能写回",
                summary.message or "只有 safe 的 check 结果才允许写回。",
            )
            return

        confirm_lines = [
            "即将把翻译写回项目 .rpy 文件。",
            "写回前请确认已在副本或备份上验证。",
            "",
            *summary.facts,
        ]
        if summary.findings:
            confirm_lines.extend(["", "待处理问题：", *[f"- {item}" for item in summary.findings]])

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
        elif self._active_command in {"bootstrap_rag", "bootstrap_source_index"}:
            self._bootstrap_output_lines.append(text)
        self._append_log(text)

    def _append_log(self, text: str):
        self.log_view.append(text.rstrip("\n"))
        # scroll to bottom
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_task_running(self, running: bool):
        self.select_btn.setEnabled(not running)
        self.doctor_btn.setEnabled(not running)
        self.api_btn.setEnabled(not running)
        self.translate_btn.setEnabled(not running)
        self.resume_btn.setEnabled(not running)
        self.save_config_btn.setEnabled(not running)
        self.apply_btn.setEnabled(not running and self._current_writeback_summary().can_apply)
        self.bootstrap_rag_btn.setEnabled(not running)
        self.bootstrap_source_index_btn.setEnabled(not running)
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
            facts.append(f"Manifest：{self._workflow.manifest_path}")
        self._workflow_step_output_lines = []
        self._set_workflow_summary("running", step.heading, step.message, facts)
        self._append_log(f"\n=== {step.heading}：gemini_translate_batch.py {' '.join(step.args)} ===\n")
        self.runner.run(self.state.get_batch_script_path(), step.args)

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
        if not self.kill_btn.isEnabled():
            self.apply_btn.setEnabled(summary.can_apply)

    def _refresh_writeback_from_latest_manifest(self) -> None:
        latest_manifest = self.state.get_latest_manifest_path()
        if latest_manifest is None:
            self._set_writeback_summary(idle_writeback_summary())
            return
        try:
            manifest = self.state.load_resume_manifest(latest_manifest)
        except ValueError:
            self._set_writeback_summary(idle_writeback_summary())
            return

        summary = summarize_manifest_writeback(manifest)
        if summary is None:
            self._set_writeback_summary(idle_writeback_summary())
            return
        self._set_writeback_summary(summary)

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

    def _set_bootstrap_summary(self, summary: BootstrapSummary) -> None:
        self.bootstrap_status_label.setText(summary.heading)
        self.bootstrap_status_label.setProperty("status", summary.status)
        self.bootstrap_status_label.style().unpolish(self.bootstrap_status_label)
        self.bootstrap_status_label.style().polish(self.bootstrap_status_label)
        self.bootstrap_message_label.setText(summary.message)
        self.bootstrap_facts_label.setText("\n".join(summary.facts))
        self._set_details_label(self.bootstrap_details_label, summary.findings)

    def _set_doctor_summary(self, summary: DoctorSummary):
        self.doctor_status_label.setText(summary.heading)
        self.doctor_status_label.setProperty("status", summary.status)
        self.doctor_status_label.style().unpolish(self.doctor_status_label)
        self.doctor_status_label.style().polish(self.doctor_status_label)
        self.doctor_message_label.setText(summary.message)
        self.doctor_facts_label.setText("\n".join(summary.facts))
        self._set_details_label(self.doctor_details_label, summary.findings)

    def _on_runner_error(self, message: str):
        self._append_log(message)
        if self._active_command == "doctor":
            self._doctor_output_lines.append(message)
        elif self._active_command == "translation_workflow":
            self._workflow_step_output_lines.append(message)
        elif self._active_command == "apply":
            self._apply_output_lines.append(message)
        elif self._active_command in {"bootstrap_rag", "bootstrap_source_index"}:
            self._bootstrap_output_lines.append(message)
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

        if self._active_command == "bootstrap_rag":
            summary = summarize_rag_bootstrap_output(
                "\n".join(self._bootstrap_output_lines),
                exit_code,
            )
            self._set_bootstrap_summary(summary)
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
            self._set_bootstrap_summary(summary)
            self._active_command = ""
            self._set_task_running(False)
            if exit_code == 0 and summary.status == "ready":
                self.statusBar().showMessage("原文索引预建完成。", 6000)
            elif exit_code == 0:
                self.statusBar().showMessage("原文索引预建已结束，请查看摘要。", 6000)
            else:
                self.statusBar().showMessage("原文索引预建失败，请查看诊断日志。", 8000)
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
            self.statusBar().showMessage("Batch 任务仍在处理，可稍后继续最新任务。", 8000)
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
