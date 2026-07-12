"""Persistent revision page for the workbench stack (#176 P4)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..work_modes import WorkMode, work_mode_submode_label
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions


class RevisionPage(QFrame):
    """Page-local controls for batch and synchronous revision workflows."""

    supported_modes = (WorkMode.REVISION, WorkMode.SYNC_REVISION)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("revision_page")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._actions = WorkbenchPageActions()
        self._running = False
        self._active_mode = WorkMode.REVISION

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(QLabel("订正模式："))
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("revision_mode_combo")
        for mode in self.supported_modes:
            self.mode_combo.addItem(work_mode_submode_label(mode), mode.value)
        self.mode_combo.currentIndexChanged.connect(self._trigger_mode_change)
        mode_row.addWidget(self.mode_combo, 1)
        outer.addLayout(mode_row)

        actions = QFrame()
        actions.setObjectName("action_frame")
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(12, 8, 12, 8)
        action_layout.setSpacing(8)
        self.start_btn = QPushButton("生成订正预览")
        self.start_btn.setObjectName("revision_start_btn")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._trigger_start)
        action_layout.addWidget(self.start_btn)
        self.resume_btn = QPushButton("继续订正")
        self.resume_btn.setObjectName("revision_resume_btn")
        self.resume_btn.setEnabled(False)
        self.resume_btn.clicked.connect(self._trigger_resume)
        action_layout.addWidget(self.resume_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("revision_stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._trigger_stop)
        action_layout.addWidget(self.stop_btn)
        self.writeback_btn = QPushButton("写回订正")
        self.writeback_btn.setObjectName("revision_writeback_btn")
        self.writeback_btn.setEnabled(False)
        self.writeback_btn.setToolTip("仅在订正预览通过后写回；不会使用翻译写回入口。")
        self.writeback_btn.clicked.connect(self._trigger_writeback)
        action_layout.addWidget(self.writeback_btn)
        action_layout.addStretch()
        outer.addWidget(actions)

        self.result_hint = QLabel("生成预览后，可在此确认订正结果并安全写回。")
        self.result_hint.setObjectName("config_hint_label")
        self.result_hint.setWordWrap(True)
        outer.addWidget(self.result_hint)

    def preferred_height(self, width: int) -> int:
        """Return the layout's word-wrap-aware height for the current page width."""
        layout = self.layout()
        if layout is None:
            return self.sizeHint().height()
        return max(self.minimumSizeHint().height(), layout.heightForWidth(max(width, 260)))

    def set_action_callbacks(self, actions: WorkbenchPageActions) -> None:
        self._actions = actions

    def activate(self, mode: WorkMode, session: WorkbenchModeSession) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported revision mode: {mode.value}")
        self._active_mode = mode
        index = self.mode_combo.findData(mode.value)
        if index >= 0:
            blocked = self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(blocked)
        del session

    def set_task_running(self, running: bool) -> None:
        self._running = running
        self.mode_combo.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if running:
            self.start_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.writeback_btn.setEnabled(False)

    def set_controls(
        self,
        *,
        start_enabled: bool,
        resume_enabled: bool,
        resume_visible: bool,
        resume_label: str,
        writeback_enabled: bool,
        result_message: str,
    ) -> None:
        self.start_btn.setEnabled(start_enabled and not self._running)
        self.resume_btn.setVisible(resume_visible)
        self.resume_btn.setText(resume_label)
        self.resume_btn.setEnabled(resume_enabled and not self._running)
        self.writeback_btn.setEnabled(writeback_enabled and not self._running)
        self.result_hint.setText(result_message)
        self.updateGeometry()

    def reset_project(self) -> None:
        self.set_task_running(False)
        self.set_controls(
            start_enabled=False,
            resume_enabled=False,
            resume_visible=self._active_mode == WorkMode.REVISION,
            resume_label="继续订正",
            writeback_enabled=False,
            result_message="项目已切换；请先完成环境检查并重新生成订正预览。",
        )

    def _trigger_mode_change(self) -> None:
        mode = WorkMode(str(self.mode_combo.currentData()))
        if (
            not self._running
            and mode != self._active_mode
            and self._actions.select_mode is not None
        ):
            self._actions.select_mode(mode)

    def _trigger_start(self) -> None:
        if not self._running and self._actions.start is not None:
            self._actions.start()

    def _trigger_resume(self) -> None:
        if not self._running and self._actions.resume is not None:
            self._actions.resume()

    def _trigger_stop(self) -> None:
        if self._running and self._actions.stop is not None:
            self._actions.stop()

    def _trigger_writeback(self) -> None:
        if (
            not self._running
            and self.writeback_btn.isEnabled()
            and self._actions.writeback is not None
        ):
            self._actions.writeback()
