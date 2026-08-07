"""Persistent synchronous-translation page for the workbench stack (#176 P2)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..empty_state import EmptyStateWidget
from ..user_copy import TASK_PROJECT_GATE_COPY
from ..work_modes import WorkMode, work_mode_spec
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
        self._preview_manifest_path = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("sync_translation_page_stack")
        outer.addWidget(self.page_stack)

        # Project gate (#298/#316): until a project is selected and the
        # environment check passes, the only dominant action is 去环境检查.
        self.empty_state = EmptyStateWidget(
            "",
            TASK_PROJECT_GATE_COPY["title"],
            f"选择项目并运行环境检查后，才能{work_mode_spec(self.supported_modes[0]).start_button_label}。",
            action_text=TASK_PROJECT_GATE_COPY["action"],
            action_style="primary",
        )
        self.empty_state.setObjectName("sync_translation_empty_state")
        self.empty_state.action_clicked.connect(self._trigger_open_doctor)
        self.page_stack.addWidget(self.empty_state)

        self.content_page = QWidget()
        self.content_page.setObjectName("sync_translation_content")
        self.task_layout = TaskPageLayout(self.content_page)

        self.risk_warning = self.task_layout.add_notice(
            "默认只生成差异预览，不会修改项目脚本；确认预览后才可写回。",
            tone="info",
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
        self.apply_btn = QPushButton("确认并写回预览")
        self.apply_btn.setObjectName("sync_translation_apply_btn")
        self.apply_btn.clicked.connect(self._trigger_writeback)
        self.apply_btn.setEnabled(False)
        self.actions.add_action(self.apply_btn, min_width=128)
        self.actions.finish_setup()

        self.preview_status = QLabel("尚未生成同步翻译预览。")
        self.preview_status.setObjectName("task_status_detail")
        self.preview_status.setWordWrap(True)
        self.task_layout.root.addWidget(self.preview_status)

        self.status_section = self.task_layout.add_status_section(
            TASK_PROJECT_GATE_COPY["status_section_title"]
        )
        self.page_stack.addWidget(self.content_page)
        self.page_stack.setCurrentWidget(self.empty_state)

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
        """Render the writeback result inside the page (sync has no result tab)."""
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
            raise ValueError(f"Unsupported sync-translation mode: {mode.value}")
        del session

    def set_task_running(self, running: bool) -> None:
        self._running = running
        self.stop_btn.setEnabled(running)
        if running:
            self.start_btn.setEnabled(False)
        self.apply_btn.setEnabled(bool(self._preview_manifest_path) and not running)

    def set_start_enabled(self, enabled: bool) -> None:
        self.start_btn.setEnabled(enabled and not self._running)

    def set_start_label(self, text: str) -> None:
        self.start_btn.setText(text)

    def reset_project(self) -> None:
        self.set_task_running(False)
        self.set_start_enabled(False)
        self.clear_preview()
        self.status_section.set_status("", "", "", [])
        self.status_section.set_progress(None)

    def set_preview_ready(self, manifest_path: str) -> None:
        self._preview_manifest_path = str(manifest_path or "").strip()
        self.apply_btn.setEnabled(bool(self._preview_manifest_path) and not self._running)
        if self._preview_manifest_path:
            self.preview_status.setText(f"预览已生成：{self._preview_manifest_path}")
        else:
            self.preview_status.setText("预览中没有可写回的变更。")

    def clear_preview(self) -> None:
        self._preview_manifest_path = ""
        self.apply_btn.setEnabled(False)
        self.preview_status.setText("尚未生成同步翻译预览。")

    def preview_manifest_path(self) -> str:
        return self._preview_manifest_path

    def _trigger_start(self) -> None:
        if not self._running and self._actions.start is not None:
            self._actions.start()

    def _trigger_stop(self) -> None:
        if self._running and self._actions.stop is not None:
            self._actions.stop()

    def _trigger_writeback(self) -> None:
        if not self._running and self._preview_manifest_path and self._actions.writeback is not None:
            self._actions.writeback()
