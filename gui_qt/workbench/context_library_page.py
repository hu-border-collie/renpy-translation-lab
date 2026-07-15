"""Persistent context-library page used by the workbench stack (#176 P1)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..empty_state import EmptyStateWidget
from ..work_modes import WorkMode
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions
from .task_controls import TaskPageLayout, TaskStatusActionRow


class ContextLibraryPage(QFrame):
    """Page-local presentation for RAG and source-index preparation."""

    supported_modes = (
        WorkMode.BOOTSTRAP_RAG,
        WorkMode.BOOTSTRAP_SOURCE_INDEX,
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("context_library_panel")
        self._actions = WorkbenchPageActions()
        self._running = False
        self._rag_enabled = False
        self._source_index_enabled = False
        self._active_mode = WorkMode.BOOTSTRAP_RAG

        outer = QVBoxLayout(self)
        # No page title — left nav already names the task (same as sync page).
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("context_library_page_stack")
        outer.addWidget(self.page_stack)

        self.empty_state = EmptyStateWidget(
            "",
            "尚未启用上下文库",
            "请先在设置 · 上下文启用记忆库或原文索引并保存，然后回到这里预建。",
            action_text="打开设置 · 上下文",
        )
        self.empty_state.setObjectName("context_library_empty_state")
        self.empty_state.action_clicked.connect(self._trigger_open_settings)
        self.page_stack.addWidget(self.empty_state)

        self.status_page = QWidget()
        self.status_page.setObjectName("context_library_status_page")
        self.status_layout = TaskPageLayout(self.status_page)

        self.bootstrap_rag_btn = QPushButton("预建记忆库")
        self.bootstrap_rag_btn.setObjectName("context_bootstrap_rag_btn")
        self.bootstrap_rag_btn.clicked.connect(lambda: self._trigger_prebuild("rag"))
        self.rag_status_row = TaskStatusActionRow(
            "记忆库",
            self.bootstrap_rag_btn,
            parent=self.status_page,
        )
        self.rag_status_label = self.rag_status_row.status_label
        self.status_layout.root.addWidget(self.rag_status_row)

        self.bootstrap_source_index_btn = QPushButton("预建原文索引")
        self.bootstrap_source_index_btn.setObjectName("context_bootstrap_source_index_btn")
        self.bootstrap_source_index_btn.clicked.connect(
            lambda: self._trigger_prebuild("source_index")
        )
        self.source_index_status_row = TaskStatusActionRow(
            "原文索引",
            self.bootstrap_source_index_btn,
            parent=self.status_page,
        )
        self.source_index_status_label = self.source_index_status_row.status_label
        self.status_layout.root.addWidget(self.source_index_status_row)

        self.context_actions = self.status_layout.add_section(
            "上下文任务",
            role="context_library",
            secondary=True,
        )
        self.open_settings_btn = QPushButton("打开设置 · 上下文")
        self.open_settings_btn.setObjectName("secondary_btn")
        self.open_settings_btn.setToolTip(
            "开关须在设置中保存后才能预建；打开设置的「上下文」分区。"
        )
        self.open_settings_btn.clicked.connect(self._trigger_open_settings)
        self.context_actions.add_action(self.open_settings_btn, min_width=140)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("context_library_stop_btn")
        self.stop_btn.setToolTip("停止当前预建任务")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._trigger_stop)
        self.context_actions.add_action(self.stop_btn, min_width=80)
        self.context_actions.finish_setup()

        self.status_layout.root.addStretch(1)
        self.page_stack.addWidget(self.status_page)
        self.page_stack.setCurrentWidget(self.empty_state)

    def set_action_callbacks(self, actions: WorkbenchPageActions) -> None:
        self._actions = actions

    def activate(self, mode: WorkMode, session: WorkbenchModeSession) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported context-library mode: {mode.value}")
        self._active_mode = mode
        # Session remains owned by MainWindow; the page only renders it when P1
        # eventually gains page-local bootstrap progress presentation.
        del session

    def set_context_status(
        self,
        *,
        rag_enabled: bool,
        source_index_enabled: bool,
        game_root: str,
    ) -> None:
        self._rag_enabled = rag_enabled
        self._source_index_enabled = source_index_enabled
        root_hint = game_root or "未选择项目"
        self.rag_status_row.set_status(
            f"{'已启用' if rag_enabled else '未启用'} · 项目 {root_hint}"
            + ("" if rag_enabled else " · 请先在设置 · 上下文开启并保存")
        )
        self.source_index_status_row.set_status(
            f"{'已启用' if source_index_enabled else '未启用'} · 项目 {root_hint}"
            + ("" if source_index_enabled else " · 请先在设置 · 上下文开启并保存")
        )
        self.page_stack.setCurrentWidget(
            self.status_page if rag_enabled or source_index_enabled else self.empty_state
        )
        self._refresh_action_states()

    def set_task_running(self, running: bool) -> None:
        self._running = running
        self._refresh_action_states()

    def reset_project(self) -> None:
        self.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="",
        )

    def _refresh_action_states(self) -> None:
        self.bootstrap_rag_btn.setEnabled(not self._running and self._rag_enabled)
        self.bootstrap_source_index_btn.setEnabled(
            not self._running and self._source_index_enabled
        )
        self.open_settings_btn.setEnabled(not self._running)
        self.stop_btn.setEnabled(self._running)
        self.context_actions.reflow()

    def _trigger_prebuild(self, kind: str) -> None:
        if self._running or self._actions.prebuild is None:
            return
        self._actions.prebuild(kind)

    def _trigger_open_settings(self) -> None:
        if not self._running and self._actions.open_settings is not None:
            self._actions.open_settings()

    def _trigger_stop(self) -> None:
        if self._running and self._actions.stop is not None:
            self._actions.stop()
