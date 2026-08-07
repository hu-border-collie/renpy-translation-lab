"""Persistent keywords/terminology page for the workbench stack (#176 P3)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..empty_state import EmptyStateWidget
from ..user_copy import TASK_PROJECT_GATE_COPY
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("keywords_page_stack")
        outer.addWidget(self.page_stack)

        # Project gate (#298/#316): until project prep passes, the only
        # dominant action is 去环境检查.
        self.empty_state = EmptyStateWidget(
            "",
            TASK_PROJECT_GATE_COPY["title"],
            TASK_PROJECT_GATE_COPY["keywords"],
            action_text=TASK_PROJECT_GATE_COPY["action"],
            action_style="primary",
        )
        self.empty_state.setObjectName("keywords_empty_state")
        self.empty_state.action_clicked.connect(self._trigger_open_doctor)
        self.page_stack.addWidget(self.empty_state)

        self.content_page = QWidget()
        self.content_page.setObjectName("keywords_content")
        self.task_layout = TaskPageLayout(self.content_page)

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

        self.status_section = self.task_layout.add_status_section(
            TASK_PROJECT_GATE_COPY["status_section_title"]
        )
        self.page_stack.addWidget(self.content_page)
        self.page_stack.setCurrentWidget(self.empty_state)

    def preferred_height(self, width: int) -> int:
        """Return the layout's word-wrap-aware height for the current page width."""
        return self.task_layout.preferred_height(width)

    def set_action_callbacks(self, actions: WorkbenchPageActions) -> None:
        self._actions = actions

    def set_project_ready(self, ready: bool) -> None:
        """Show the environment-check gate until project prep is done."""
        self._project_ready = bool(ready)
        self.page_stack.setCurrentWidget(
            self.content_page if self._project_ready else self.empty_state
        )

    def set_workflow_status(
        self,
        status: str,
        heading: str,
        message: str,
        facts: list[str] | None = None,
    ) -> None:
        """Render workflow progress inside the page (#298)."""
        self.status_section.set_status(status, heading, message, facts)

    def set_workflow_progress(self, state: object | None) -> None:
        """Render an optional progress bar inside the page."""
        self.status_section.set_progress(state)

    def set_workflow_facts(self, facts: list[str]) -> None:
        """Replace the workflow facts shown in the page status section."""
        self.status_section.facts_label.setText("\n".join(facts))

    def set_writeback_status(self, summary: object | None) -> None:
        """Render the writeback/merge result inside the page (#298)."""
        if summary is None:
            return
        self.status_section.set_status(
            getattr(summary, "status", "idle"),
            getattr(summary, "heading", ""),
            getattr(summary, "message", ""),
            list(getattr(summary, "facts", []) or []),
        )

    def workflow_status_snapshot(self) -> tuple[str, str, str, list[str]]:
        """Return (status, heading, message, facts) for session freeze."""
        badge = self.status_section.status_badge
        status = str(badge.property("status") or "")
        heading = badge.text()
        message = self.status_section.message_label.text()
        facts = [
            line
            for line in self.status_section.facts_label.text().splitlines()
            if line.strip()
        ]
        return status, heading, message, facts

    def _trigger_open_doctor(self) -> None:
        if self._actions.action is not None:
            self._actions.action("open_doctor")

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
