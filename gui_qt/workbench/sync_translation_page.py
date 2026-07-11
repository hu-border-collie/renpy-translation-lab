"""Persistent synchronous-translation page for the workbench stack (#176 P2)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..work_modes import WorkMode
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions


class SyncTranslationPage(QFrame):
    """Compact risk notice + start/stop; shared status card holds doctor/progress."""

    supported_modes = (WorkMode.SYNC_TRANSLATION,)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sync_translation_page")
        # Height-for-content only — never eat space meant for the status card.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._actions = WorkbenchPageActions()
        self._running = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.risk_warning = QLabel(
            "警告：可能直接修改项目文件，请先备份或在副本上试跑。"
        )
        self.risk_warning.setObjectName("sync_translation_risk_warning")
        self.risk_warning.setWordWrap(True)
        self.risk_warning.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        outer.addWidget(self.risk_warning)

        # Compact glass action card (same look as legacy action_frame, fixed height).
        actions = QFrame()
        actions.setObjectName("action_frame")
        actions.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(12, 8, 12, 8)
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

    def sizeHint(self):  # noqa: N802
        """Keep stack sizing honest — QStackedWidget takes max of all pages."""
        return self.minimumSizeHint()

    def minimumSizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize

        # warning + spacing + compact action card (padding + button)
        w = super().minimumSizeHint().width()
        btn_h = max(self.start_btn.sizeHint().height(), 32)
        action_h = btn_h + 16  # 8px vertical padding × 2
        h = self.risk_warning.sizeHint().height() + 6 + action_h
        return QSize(max(w, 200), h)

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
