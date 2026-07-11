"""Persistent synchronous-translation page for the workbench stack (#176 P2)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..status_icons import StatusBadge
from ..work_modes import WorkMode
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions


class SyncTranslationPage(QFrame):
    """Page-local controls and summary for direct synchronous translation."""

    supported_modes = (WorkMode.SYNC_TRANSLATION,)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sync_translation_page")
        self._actions = WorkbenchPageActions()
        self._running = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        title = QLabel("同步翻译")
        title.setObjectName("diagnostics_section_label")
        outer.addWidget(title)

        self.risk_warning = QLabel(
            "警告：同步翻译可能直接修改项目文件，请先备份或在副本上试跑。"
        )
        self.risk_warning.setObjectName("sync_mode_warning")
        self.risk_warning.setWordWrap(True)
        outer.addWidget(self.risk_warning)

        actions = QFrame()
        actions.setObjectName("action_frame")
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(10, 8, 10, 8)
        action_layout.setSpacing(8)
        self.start_btn = QPushButton("开始同步翻译")
        self.start_btn.setObjectName("translate_btn")
        self.start_btn.clicked.connect(self._trigger_start)
        self.start_btn.setEnabled(False)
        action_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("kill_btn")
        self.stop_btn.clicked.connect(self._trigger_stop)
        self.stop_btn.setEnabled(False)
        action_layout.addWidget(self.stop_btn)
        action_layout.addStretch()
        outer.addWidget(actions)

        summary = QFrame()
        summary.setObjectName("workbench_status_card")
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(12, 12, 12, 12)
        summary_layout.setSpacing(6)
        self.status_label = StatusBadge("sync_translation_status_label")
        summary_layout.addWidget(self.status_label)
        self.heading_label = QLabel("尚未开始同步翻译")
        self.heading_label.setObjectName("action_group_label")
        self.heading_label.setWordWrap(True)
        summary_layout.addWidget(self.heading_label)
        self.message_label = QLabel(
            "完成环境检查后即可开始；同步翻译会直接写入项目文件。"
        )
        self.message_label.setObjectName("summary_body_label")
        self.message_label.setWordWrap(True)
        summary_layout.addWidget(self.message_label)
        self.facts_label = QLabel()
        self.facts_label.setObjectName("workflow_facts_label")
        self.facts_label.setWordWrap(True)
        summary_layout.addWidget(self.facts_label)
        outer.addWidget(summary)
        outer.addStretch()

    def set_action_callbacks(self, actions: WorkbenchPageActions) -> None:
        self._actions = actions

    def activate(self, mode: WorkMode, session: WorkbenchModeSession) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported sync-translation mode: {mode.value}")
        self.render_summary(
            session.workflow_status or "idle",
            session.workflow_heading or "尚未开始同步翻译",
            session.workflow_message or "完成环境检查后即可开始；同步翻译会直接写入项目文件。",
            session.workflow_facts,
        )

    def set_task_running(self, running: bool) -> None:
        self._running = running
        self.stop_btn.setEnabled(running)
        if running:
            self.start_btn.setEnabled(False)

    def set_start_enabled(self, enabled: bool) -> None:
        self.start_btn.setEnabled(enabled and not self._running)

    def reset_project(self) -> None:
        self.render_summary(
            "stale",
            "项目已切换",
            "任务状态已清空；请先针对新项目重新检查。",
            [],
        )

    def render_summary(
        self,
        status: str,
        heading: str,
        message: str,
        facts: list[str],
    ) -> None:
        self.status_label.set_status(status, heading)
        self.heading_label.setText(heading)
        self.message_label.setText(message)
        self.facts_label.setText("\n".join(facts))

    def _trigger_start(self) -> None:
        if not self._running and self._actions.start is not None:
            self._actions.start()

    def _trigger_stop(self) -> None:
        if self._running and self._actions.stop is not None:
            self._actions.stop()
