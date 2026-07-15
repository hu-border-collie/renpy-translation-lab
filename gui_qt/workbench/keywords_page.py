"""Persistent keywords/terminology page for the workbench stack (#176 P3)."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFrame, QPushButton, QSizePolicy, QWidget

from ..work_modes import WorkMode, work_mode_submode_label
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions
from .task_controls import TaskPageLayout


class KeywordsPage(QFrame):
    """Page-local controls for batch and synchronous keyword extraction."""

    supported_modes = (
        WorkMode.KEYWORD_EXTRACTION,
        WorkMode.SYNC_KEYWORD_EXTRACTION,
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("keywords_page")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._actions = WorkbenchPageActions()
        self._running = False
        self._active_mode = WorkMode.KEYWORD_EXTRACTION

        self.task_layout = TaskPageLayout(self)

        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("keywords_mode_combo")
        for mode in self.supported_modes:
            self.mode_combo.addItem(work_mode_submode_label(mode), mode.value)
        self.mode_combo.currentIndexChanged.connect(self._trigger_mode_change)
        self.mode_label = self.task_layout.add_mode_selector(
            "关键词 / 术语模式：",
            self.mode_combo,
        )

        self.actions = self.task_layout.add_section(
            "提取任务",
            role="keywords",
        )
        self.start_btn = QPushButton("提取关键词")
        self.start_btn.setObjectName("keywords_start_btn")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._trigger_start)
        self.actions.add_action(self.start_btn, min_width=108)

        self.resume_btn = QPushButton("继续提取")
        self.resume_btn.setObjectName("keywords_resume_btn")
        self.resume_btn.setEnabled(False)
        self.resume_btn.clicked.connect(self._trigger_resume)
        self.actions.add_action(self.resume_btn, min_width=108)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("keywords_stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._trigger_stop)
        self.actions.add_action(self.stop_btn, min_width=80)

        self.merge_btn = QPushButton("合并到 glossary")
        self.merge_btn.setObjectName("keywords_merge_btn")
        self.merge_btn.setEnabled(False)
        self.merge_btn.setToolTip("提取完成后，审核候选并写入 glossary.json；不会修改 .rpy 脚本。")
        self.merge_btn.clicked.connect(self._trigger_merge)
        self.actions.add_action(self.merge_btn, min_width=130)
        self.actions.finish_setup()

        self.result_hint = self.task_layout.add_result_hint(
            "提取完成后，可在此合并审核通过的术语候选。"
        )

    def preferred_height(self, width: int) -> int:
        """Return the layout's word-wrap-aware height for the current page width."""
        return self.task_layout.preferred_height(width)

    def set_action_callbacks(self, actions: WorkbenchPageActions) -> None:
        self._actions = actions

    def activate(self, mode: WorkMode, session: WorkbenchModeSession) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported keywords mode: {mode.value}")
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
            self.merge_btn.setEnabled(False)

    def set_controls(
        self,
        *,
        start_enabled: bool,
        resume_enabled: bool,
        resume_visible: bool,
        resume_label: str,
        merge_enabled: bool,
        merge_message: str,
    ) -> None:
        self.start_btn.setEnabled(start_enabled and not self._running)
        self.resume_btn.setVisible(resume_visible)
        self.resume_btn.setText(resume_label)
        self.resume_btn.setEnabled(resume_enabled and not self._running)
        self.merge_btn.setEnabled(merge_enabled and not self._running)
        self.result_hint.setText(merge_message)
        self.task_layout.reflow()
        self.updateGeometry()

    def reset_project(self) -> None:
        self.set_task_running(False)
        self.set_controls(
            start_enabled=False,
            resume_enabled=False,
            resume_visible=self._active_mode == WorkMode.KEYWORD_EXTRACTION,
            resume_label="继续提取",
            merge_enabled=False,
            merge_message="项目已切换；请先完成环境检查并重新提取关键词。",
        )

    def _trigger_mode_change(self) -> None:
        mode = WorkMode(str(self.mode_combo.currentData()))
        if not self._running and mode != self._active_mode and self._actions.select_mode is not None:
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

    def _trigger_merge(self) -> None:
        if not self._running and self.merge_btn.isEnabled() and self._actions.writeback is not None:
            self._actions.writeback()
