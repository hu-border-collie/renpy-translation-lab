"""Persistent revision page for the workbench stack (#176 P4)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..empty_state import EmptyStateWidget
from ..revision_corpus_report import RevisionCorpusExportResult
from ..user_copy import (
    REVISION_CORPUS_COPY,
    REVISION_PROPOSAL_COPY,
    TASK_PROJECT_GATE_COPY,
)
from ..work_modes import WorkMode, work_mode_submode_label
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions
from .task_controls import TaskPageLayout, task_status_has_result


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

        self.export_corpus_btn = QPushButton(REVISION_CORPUS_COPY["action"])
        self.export_corpus_btn.setObjectName("revision_export_corpus_btn")
        self.export_corpus_btn.setEnabled(False)
        self.export_corpus_btn.setToolTip(REVISION_CORPUS_COPY["tooltip"])
        self.export_corpus_btn.clicked.connect(self._trigger_export_corpus)
        self.actions.add_action(self.export_corpus_btn, min_width=128)

        self.import_proposals_btn = QPushButton(REVISION_PROPOSAL_COPY["action"])
        self.import_proposals_btn.setObjectName("revision_import_proposals_btn")
        self.import_proposals_btn.setEnabled(False)
        self.import_proposals_btn.setToolTip(REVISION_PROPOSAL_COPY["tooltip"])
        self.import_proposals_btn.clicked.connect(self._trigger_import_proposals)
        self.actions.add_action(self.import_proposals_btn, min_width=128)

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

        self.corpus_result = QFrame(self.content_page)
        self.corpus_result.setObjectName("revision_corpus_result")
        self.corpus_result.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        corpus_layout = QGridLayout(self.corpus_result)
        corpus_layout.setContentsMargins(12, 10, 12, 10)
        corpus_layout.setHorizontalSpacing(10)
        corpus_layout.setVerticalSpacing(5)
        self.corpus_result_title = QLabel(REVISION_CORPUS_COPY["result_title"])
        self.corpus_result_title.setObjectName("revision_corpus_result_title")
        corpus_layout.addWidget(self.corpus_result_title, 0, 0, 1, 2)

        self.corpus_result_summary = QLabel("")
        self.corpus_result_summary.setObjectName("revision_corpus_result_summary")
        self.corpus_result_summary.setWordWrap(True)
        corpus_layout.addWidget(self.corpus_result_summary, 1, 0, 1, 2)

        self.corpus_result_created_at = QLabel("")
        self.corpus_result_created_at.setObjectName(
            "revision_corpus_result_created_at"
        )
        self.corpus_result_created_at.setWordWrap(True)
        corpus_layout.addWidget(self.corpus_result_created_at, 2, 0, 1, 2)

        self.corpus_result_paths = QLabel("")
        self.corpus_result_paths.setObjectName("revision_corpus_result_paths")
        self.corpus_result_paths.setWordWrap(True)
        self.corpus_result_paths.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        corpus_layout.addWidget(self.corpus_result_paths, 3, 0, 1, 2)

        corpus_actions = QFrame(self.corpus_result)
        corpus_actions_layout = QGridLayout(corpus_actions)
        corpus_actions_layout.setContentsMargins(0, 2, 0, 0)
        corpus_actions_layout.setHorizontalSpacing(8)
        self.corpus_open_output_btn = QPushButton(
            REVISION_CORPUS_COPY["open_output_dir"]
        )
        self.corpus_open_output_btn.setObjectName("revision_corpus_open_output_btn")
        self.corpus_open_output_btn.setEnabled(False)
        self.corpus_open_output_btn.clicked.connect(self._trigger_open_corpus_output)
        corpus_actions_layout.addWidget(self.corpus_open_output_btn, 0, 0)
        self.corpus_copy_paths_btn = QPushButton(REVISION_CORPUS_COPY["copy_paths"])
        self.corpus_copy_paths_btn.setObjectName("revision_corpus_copy_paths_btn")
        self.corpus_copy_paths_btn.setEnabled(False)
        self.corpus_copy_paths_btn.clicked.connect(self._trigger_copy_corpus_paths)
        corpus_actions_layout.addWidget(self.corpus_copy_paths_btn, 0, 1)
        corpus_actions_layout.setColumnStretch(2, 1)
        corpus_layout.addWidget(corpus_actions, 4, 0, 1, 2)
        self.corpus_result.setVisible(False)
        self.task_layout.root.addWidget(self.corpus_result)

        self.proposal_result = QFrame(self.content_page)
        self.proposal_result.setObjectName("revision_proposal_result")
        self.proposal_result.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        proposal_layout = QGridLayout(self.proposal_result)
        proposal_layout.setContentsMargins(12, 8, 12, 8)
        proposal_layout.setHorizontalSpacing(10)
        proposal_layout.setVerticalSpacing(4)
        self.proposal_result_title = QLabel(REVISION_PROPOSAL_COPY["result_title"])
        self.proposal_result_title.setObjectName("revision_proposal_result_title")
        proposal_layout.addWidget(self.proposal_result_title, 0, 0, 1, 2)
        self.proposal_result_summary = QLabel("")
        self.proposal_result_summary.setObjectName("revision_proposal_result_summary")
        self.proposal_result_summary.setWordWrap(True)
        proposal_layout.addWidget(self.proposal_result_summary, 1, 0, 1, 2)
        self.proposal_result_session = QLabel("")
        self.proposal_result_session.setObjectName("revision_proposal_result_session")
        self.proposal_result_session.setWordWrap(True)
        self.proposal_result_session.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        proposal_layout.addWidget(self.proposal_result_session, 2, 0, 1, 2)
        self.select_proposals_btn = QPushButton(REVISION_PROPOSAL_COPY["select_action"])
        self.select_proposals_btn.setObjectName("revision_select_proposals_btn")
        self.select_proposals_btn.setEnabled(False)
        self.select_proposals_btn.clicked.connect(self._trigger_select_proposals)
        proposal_layout.addWidget(self.select_proposals_btn, 3, 0)
        proposal_layout.setColumnStretch(1, 1)
        self.proposal_result.setVisible(False)
        self.task_layout.root.addWidget(self.proposal_result)

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
        has_result = task_status_has_result(status)
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
        self.status_section.set_facts(facts)

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

    def set_corpus_export_result(
        self,
        result: RevisionCorpusExportResult | None,
    ) -> None:
        """Render the structured corpus artifact result independently of writeback."""
        self._corpus_export_result = result
        if result is None:
            self.corpus_result.setVisible(False)
            self.corpus_result_summary.setText("")
            self.corpus_result_created_at.setText("")
            self.corpus_result_paths.setText("")
            self.corpus_open_output_btn.setEnabled(False)
            self.corpus_copy_paths_btn.setEnabled(False)
            self.task_layout.reflow()
            self.updateGeometry()
            return

        self.corpus_result.setVisible(True)
        self.corpus_result_summary.setText(
            f"条目数：{result.item_count} · 文件数：{result.file_count}"
        )
        self.corpus_result_created_at.setText(
            f"生成时间：{result.created_at or '未读取（请检查 manifest）'}"
        )
        self.corpus_result_paths.setText(
            "\n".join(
                (
                    f"JSONL：{result.jsonl_path}",
                    f"Markdown：{result.markdown_path}",
                    f"manifest：{result.manifest_path}",
                )
            )
        )
        self.corpus_open_output_btn.setEnabled(bool(result.output_dir))
        self.corpus_copy_paths_btn.setEnabled(result.has_paths)
        self.task_layout.reflow()
        self.updateGeometry()

    def corpus_export_result(self) -> RevisionCorpusExportResult | None:
        """Return the last result for the coordinator's session snapshot."""
        return getattr(self, "_corpus_export_result", None)

    def set_proposal_stage_result(self, result: dict[str, object] | None) -> None:
        """Render the structured staged-selection summary without parsing stdout."""
        self._proposal_stage_result = result
        if result is None:
            self.proposal_result.setVisible(False)
            self.proposal_result_summary.setText("")
            self.proposal_result_session.setText("")
            self.select_proposals_btn.setEnabled(False)
            self.task_layout.reflow()
            self.updateGeometry()
            return

        self.proposal_result.setVisible(True)
        self.proposal_result_summary.setText(
            " · ".join(
                (
                    f"候选 {int(result.get('candidate_count') or 0)}",
                    f"有效 {int(result.get('selectable_count') or 0)}",
                    f"未选择 {int(result.get('unselected_count') or 0)}",
                    f"无需修改 {int(result.get('no_op_count') or 0)}",
                    f"无效 {int(result.get('invalid_count') or 0)}",
                    f"过期 {int(result.get('stale_count') or 0)}",
                    f"冲突 {int(result.get('conflict_count') or 0)}",
                )
            )
        )
        paths = result.get("paths") if isinstance(result, dict) else {}
        stage_path = paths.get("staged_selection") if isinstance(paths, dict) else ""
        self.proposal_result_session.setText(f"候选会话：{stage_path or '未记录'}")
        self.select_proposals_btn.setEnabled(
            str(result.get("session_status") or "") == "ready"
            and int(result.get("selectable_count") or 0) > 0
        )
        self.task_layout.reflow()
        self.updateGeometry()

    def proposal_stage_result(self) -> dict[str, object] | None:
        """Return the staged candidate summary for the coordinator/session."""
        return getattr(self, "_proposal_stage_result", None)

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
        self.import_proposals_btn.setVisible(mode == WorkMode.REVISION)
        self.result_hint.setText(
            "审查完成后，选择需要处理的问题并生成订正预览。"
            if final_review
            else "生成预览后，可在此确认订正结果并安全写回。"
        )
        self.export_corpus_btn.setVisible(mode == WorkMode.REVISION)
        self.set_corpus_export_result(
            getattr(session, "revision_corpus_export_result", None)
        )
        self.set_proposal_stage_result(
            getattr(session, "revision_proposal_stage_result", None)
            if mode == WorkMode.REVISION
            else None
        )

    def set_task_running(self, running: bool) -> None:
        self._running = running
        self.mode_combo.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if running:
            self.start_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.writeback_btn.setEnabled(False)
            self.review_findings_btn.setEnabled(False)
            self.export_corpus_btn.setEnabled(False)
            self.import_proposals_btn.setEnabled(False)
            self.select_proposals_btn.setEnabled(False)

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
        export_enabled: bool = False,
        export_tooltip: str = "",
        selection_enabled: bool = False,
    ) -> None:
        self.start_btn.setEnabled(start_enabled and not self._running)
        self.resume_btn.setVisible(resume_visible)
        self.resume_btn.setText(resume_label)
        self.resume_btn.setEnabled(resume_enabled and not self._running)
        self.writeback_btn.setEnabled(writeback_enabled and not self._running)
        self.review_findings_btn.setEnabled(findings_enabled and not self._running)
        self.export_corpus_btn.setEnabled(
            export_enabled
            and self._active_mode == WorkMode.REVISION
            and not self._running
        )
        self.export_corpus_btn.setToolTip(
            export_tooltip or REVISION_CORPUS_COPY["tooltip"]
        )
        self.import_proposals_btn.setEnabled(start_enabled and not self._running)
        stage_result = self.proposal_stage_result()
        self.select_proposals_btn.setEnabled(
            selection_enabled
            and self._active_mode == WorkMode.REVISION
            and not self._running
            and isinstance(stage_result, dict)
            and str(stage_result.get("session_status") or "") == "ready"
            and int(stage_result.get("selectable_count") or 0) > 0
        )
        self.result_hint.setText(result_message)
        self.task_layout.reflow()
        self.updateGeometry()

    def reset_project(self) -> None:
        self.set_task_running(False)
        self.status_section.set_status("", "", "", [])
        self.status_section.set_progress(None)
        self.set_corpus_export_result(None)
        self.set_proposal_stage_result(None)
        final_review = self._active_mode == WorkMode.FINAL_REVIEW
        self.set_controls(
            start_enabled=False,
            resume_enabled=False,
            resume_visible=self._active_mode in (WorkMode.REVISION, WorkMode.FINAL_REVIEW),
            resume_label="继续审查" if final_review else "继续订正",
            writeback_enabled=False,
            result_message="项目已切换；请先完成环境检查并重新生成订正预览。",
            export_enabled=False,
            export_tooltip=REVISION_CORPUS_COPY["gate_no_project"],
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

    def _trigger_import_proposals(self) -> None:
        if (
            not self._running
            and self.import_proposals_btn.isEnabled()
            and self._actions.action is not None
        ):
            self._actions.action("import_revision_proposals")

    def _trigger_export_corpus(self) -> None:
        if (
            not self._running
            and self.export_corpus_btn.isEnabled()
            and self._actions.action is not None
        ):
            self._actions.action("export_revision_corpus")

    def _trigger_select_proposals(self) -> None:
        if (
            not self._running
            and self.select_proposals_btn.isEnabled()
            and self._actions.action is not None
        ):
            self._actions.action("select_revision_proposals")

    def _trigger_open_corpus_output(self) -> None:
        if (
            self.corpus_open_output_btn.isEnabled()
            and self._actions.action is not None
        ):
            self._actions.action("open_revision_corpus_output")

    def _trigger_copy_corpus_paths(self) -> None:
        if (
            self.corpus_copy_paths_btn.isEnabled()
            and self._actions.action is not None
        ):
            self._actions.action("copy_revision_corpus_paths")

    def _trigger_writeback(self) -> None:
        if (
            not self._running
            and self.writeback_btn.isEnabled()
            and self._actions.writeback is not None
        ):
            self._actions.writeback()
