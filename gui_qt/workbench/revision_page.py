"""Persistent revision page for the workbench stack (#176 P4)."""
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


class RevisionPage(QFrame):
    """Page-local controls for batch and synchronous revision workflows."""

    supported_modes = (WorkMode.REVISION, WorkMode.SYNC_REVISION, WorkMode.FINAL_REVIEW)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("revision_page")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._actions = WorkbenchPageActions()
        self._running = False
        self._active_mode = WorkMode.REVISION

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("revision_page_stack")
        outer.addWidget(self.page_stack)

        # Project gate (#298/#316): until project prep passes, the only
        # dominant action is 去环境检查.
        self.empty_state = EmptyStateWidget(
            "",
            TASK_PROJECT_GATE_COPY["title"],
            TASK_PROJECT_GATE_COPY["revision_body"],
            action_text=TASK_PROJECT_GATE_COPY["action"],
            action_style="primary",
        )
        self.empty_state.setObjectName("revision_empty_state")
        self.empty_state.action_clicked.connect(self._trigger_open_doctor)
        self.page_stack.addWidget(self.empty_state)

        self.content_page = QWidget()
        self.content_page.setObjectName("revision_content")
        self.task_layout = TaskPageLayout(self.content_page)

        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("revision_mode_combo")
        for mode in self.supported_modes:
            self.mode_combo.addItem(work_mode_submode_label(mode), mode.value)
        self.mode_combo.currentIndexChanged.connect(self._trigger_mode_change)
        self.mode_label = self.task_layout.add_mode_selector(
            "订正模式：",
            self.mode_combo,
        )

        self.actions = self.task_layout.add_section(
            "订正任务",
            role="revision",
        )
        self.start_btn = QPushButton("生成订正预览")
        self.start_btn.setObjectName("revision_start_btn")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._trigger_start)
        self.actions.add_action(self.start_btn, min_width=120)

        self.resume_btn = QPushButton("继续订正")
        self.resume_btn.setObjectName("revision_resume_btn")
        self.resume_btn.setEnabled(False)
        self.resume_btn.clicked.connect(self._trigger_resume)
        self.actions.add_action(self.resume_btn, min_width=108)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("revision_stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._trigger_stop)
        self.actions.add_action(self.stop_btn, min_width=80)

        self.writeback_btn = QPushButton("写回订正")
        self.writeback_btn.setObjectName("revision_writeback_btn")
        self.writeback_btn.setEnabled(False)
        self.writeback_btn.setToolTip("仅在订正预览通过后写回；不会使用翻译写回入口。")
        self.writeback_btn.clicked.connect(self._trigger_writeback)
        self.actions.add_action(self.writeback_btn, min_width=108)
        self.review_findings_btn = QPushButton("选择问题并生成预览")
        self.review_findings_btn.setObjectName("final_review_select_btn")
        self.review_findings_btn.setEnabled(False)
        self.review_findings_btn.setVisible(False)
        self.review_findings_btn.setToolTip("只转换人工勾选的问题；仍需经过订正预览后才能写回。")
        self.review_findings_btn.clicked.connect(self._trigger_review_findings)
        self.actions.add_action(self.review_findings_btn, min_width=160)
        self.actions.finish_setup()

        self.result_hint = self.task_layout.add_result_hint(
            "生成预览后，可在此确认订正结果并安全写回。"
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
        """Show the environment-check gate until project prep is done.

        Existing task results keep the content view visible: the gate only
        controls operation availability, never hides finished results (#298).
        """
        self._project_ready = bool(ready)
        if self._project_ready:
            self.page_stack.setCurrentWidget(self.content_page)
            return
        status, _heading, _message, _facts = self.workflow_status_snapshot()
        has_result = bool(status and status not in {"idle", "stale"})
        self.page_stack.setCurrentWidget(
            self.content_page if has_result else self.empty_state
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
        """Render the writeback result inside the page (#298)."""
        if summary is None:
            return
        self.status_section.set_status(
            getattr(summary, "status", "idle"),
            getattr(summary, "heading", ""),
            getattr(summary, "message", ""),
            list(getattr(summary, "facts", []) or []),
        )
        self.status_section.set_details(getattr(summary, "findings", None))


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
            raise ValueError(f"Unsupported revision mode: {mode.value}")
        self._active_mode = mode
        index = self.mode_combo.findData(mode.value)
        if index >= 0:
            blocked = self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(blocked)
        final_review = mode == WorkMode.FINAL_REVIEW
        self.actions.title_label.setText("最终审校任务" if final_review else "订正任务")
        self.start_btn.setText("开始最终审校" if final_review else "生成订正预览")
        self.writeback_btn.setText("写回所选订正" if final_review else "写回订正")
        self.review_findings_btn.setVisible(final_review)
        self.result_hint.setText(
            "审查完成后，选择需要处理的问题并生成订正预览。"
            if final_review
            else "生成预览后，可在此确认订正结果并安全写回。"
        )
        del session

    def set_task_running(self, running: bool) -> None:
        self._running = running
        self.mode_combo.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if running:
            self.start_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.writeback_btn.setEnabled(False)
            self.review_findings_btn.setEnabled(False)

    def set_controls(
        self,
        *,
        start_enabled: bool,
        resume_enabled: bool,
        resume_visible: bool,
        resume_label: str,
        writeback_enabled: bool,
        result_message: str,
        findings_enabled: bool = False,
    ) -> None:
        self.start_btn.setEnabled(start_enabled and not self._running)
        self.resume_btn.setVisible(resume_visible)
        self.resume_btn.setText(resume_label)
        self.resume_btn.setEnabled(resume_enabled and not self._running)
        self.writeback_btn.setEnabled(writeback_enabled and not self._running)
        self.review_findings_btn.setEnabled(findings_enabled and not self._running)
        self.result_hint.setText(result_message)
        self.task_layout.reflow()
        self.updateGeometry()

    def reset_project(self) -> None:
        self.set_task_running(False)
        self.status_section.set_status("", "", "", [])
        self.status_section.set_progress(None)
        final_review = self._active_mode == WorkMode.FINAL_REVIEW
        self.set_controls(
            start_enabled=False,
            resume_enabled=False,
            resume_visible=self._active_mode in (WorkMode.REVISION, WorkMode.FINAL_REVIEW),
            resume_label="继续审查" if final_review else "继续订正",
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

    def _trigger_review_findings(self) -> None:
        if (
            not self._running
            and self.review_findings_btn.isEnabled()
            and self._actions.action is not None
        ):
            self._actions.action("select_final_review_findings")

    def _trigger_writeback(self) -> None:
        if (
            not self._running
            and self.writeback_btn.isEnabled()
            and self._actions.writeback is not None
        ):
            self._actions.writeback()
