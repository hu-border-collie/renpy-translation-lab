"""Persistent synchronous-translation page for the workbench stack (#176 P2)."""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QPushButton, QSizePolicy, QWidget

from ..work_modes import WorkMode
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions
from .task_controls import TaskPageLayout


class SyncTranslationPage(QFrame):
    """Compact risk notice and task-local start/stop controls."""

    supported_modes = (WorkMode.SYNC_TRANSLATION,)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sync_translation_page")
        # Height-for-content only — keep the task page compact.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._actions = WorkbenchPageActions()
        self._running = False

        self.task_layout = TaskPageLayout(self)

        self.risk_warning = self.task_layout.add_notice(
            "警告：可能直接修改项目文件，请先备份或在副本上试跑。"
        )

        self.actions = self.task_layout.add_section(
            "翻译任务",
            role="sync_translation",
        )
        self.start_btn = QPushButton("开始同步翻译")
        self.start_btn.setObjectName("sync_translation_start_btn")
        self.start_btn.clicked.connect(self._trigger_start)
        self.start_btn.setEnabled(False)
        self.actions.add_action(self.start_btn, min_width=120)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("sync_translation_stop_btn")
        self.stop_btn.clicked.connect(self._trigger_stop)
        self.stop_btn.setEnabled(False)
        self.actions.add_action(self.stop_btn, min_width=80)
        self.actions.finish_setup()

    def preferred_height(self, width: int) -> int:
        """Return the word-wrap-aware content height for the current width."""
        return self.task_layout.preferred_height(width)

    def sizeHint(self):  # noqa: N802
        """Keep stack sizing honest — QStackedWidget takes max of all pages."""
        return self.minimumSizeHint()

    def minimumSizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize

        hint = super().minimumSizeHint()
        return QSize(max(hint.width(), 200), hint.height())

    def set_action_callbacks(self, actions: WorkbenchPageActions) -> None:
        self._actions = actions

    def activate(self, mode: WorkMode, session: WorkbenchModeSession) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported sync-translation mode: {mode.value}")
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
        self.set_task_running(False)
        self.set_start_enabled(False)

    def _trigger_start(self) -> None:
        if not self._running and self._actions.start is not None:
            self._actions.start()

    def _trigger_stop(self) -> None:
        if self._running and self._actions.stop is not None:
            self._actions.stop()
