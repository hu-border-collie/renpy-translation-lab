"""Persistent synchronous-translation page for the workbench stack (#176 P2)."""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..work_modes import WorkMode
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions


class SyncTranslationPage(QFrame):
    """Page-local risk notice and start/stop for direct synchronous translation.

    Live progress, doctor status, and writeback stay on the shared workbench
    status card so the page does not duplicate that surface.
    """

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
        self.risk_warning.setObjectName("sync_translation_risk_warning")
        self.risk_warning.setWordWrap(True)
        outer.addWidget(self.risk_warning)

        actions = QFrame()
        actions.setObjectName("action_frame")
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(10, 8, 10, 8)
        action_layout.setSpacing(8)
        self.start_btn = QPushButton("开始同步翻译")
        self.start_btn.setObjectName("sync_translation_start_btn")
        self.start_btn.clicked.connect(self._trigger_start)
        self.start_btn.setEnabled(False)
        action_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("sync_translation_stop_btn")
        self.stop_btn.clicked.connect(self._trigger_stop)
        self.stop_btn.setEnabled(False)
        action_layout.addWidget(self.stop_btn)
        action_layout.addStretch()
        outer.addWidget(actions)
        outer.addStretch()

    def set_action_callbacks(self, actions: WorkbenchPageActions) -> None:
        self._actions = actions

    def activate(self, mode: WorkMode, session: WorkbenchModeSession) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported sync-translation mode: {mode.value}")
        # Progress / doctor / writeback live on the shared status card.
        del session

    def set_task_running(self, running: bool) -> None:
        self._running = running
        self.stop_btn.setEnabled(running)
        if running:
            self.start_btn.setEnabled(False)

    def set_start_enabled(self, enabled: bool) -> None:
        self.start_btn.setEnabled(enabled and not self._running)

    def set_start_label(self, text: str) -> None:
        self.start_btn.setText(text)

    def reset_project(self) -> None:
        """Page-local chrome has no project-bound state beyond button enablement."""
        self.set_task_running(False)
        self.set_start_enabled(False)

    def _trigger_start(self) -> None:
        if not self._running and self._actions.start is not None:
            self._actions.start()

    def _trigger_stop(self) -> None:
        if self._running and self._actions.stop is not None:
            self._actions.stop()
