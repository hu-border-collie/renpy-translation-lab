"""Persistent context-library page used by the workbench stack (#176 P1)."""
from __future__ import annotations

from typing import Any, Mapping

from PySide6.QtWidgets import (
    QFrame,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..empty_state import EmptyStateWidget
from ..user_copy import CONTEXT_LIBRARY_COPY, PROJECT_ANALYSIS_COPY
from ..work_modes import WorkMode
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions
from .task_controls import TaskPageLayout, TaskStatusActionRow


class ContextLibraryPage(QFrame):
    """Page-local presentation for RAG, source-index, and project-analysis status."""

    supported_modes = (
        WorkMode.BOOTSTRAP_RAG,
        WorkMode.BOOTSTRAP_SOURCE_INDEX,
        WorkMode.PROJECT_ANALYSIS,
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("context_library_panel")
        self._actions = WorkbenchPageActions()
        self._running = False
        self._rag_enabled = False
        self._source_index_enabled = False
        self._project_analysis_present = False
        self._project_analysis_enabled = False
        self._project_analysis_inject_enabled = False
        self._project_analysis_status: dict[str, Any] = {}
        self._project_analysis_primary_action = "project_analysis_build_structure"
        self._has_project = False
        self._active_mode = WorkMode.BOOTSTRAP_RAG

        outer = QVBoxLayout(self)
        # No page title — left nav already names the task (same as sync page).
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("context_library_page_stack")
        outer.addWidget(self.page_stack)

        self.empty_state = EmptyStateWidget(
            "",
            CONTEXT_LIBRARY_COPY["empty_title"],
            CONTEXT_LIBRARY_COPY["empty_body"],
            action_text="打开设置 · 上下文",
        )
        self.empty_state.setObjectName("context_library_empty_state")
        self.empty_state.action_clicked.connect(self._trigger_open_settings)
        self.page_stack.addWidget(self.empty_state)

        self.status_page = QWidget()
        self.status_page.setObjectName("context_library_status_page")
        self.status_layout = TaskPageLayout(self.status_page)

        self.bootstrap_rag_btn = QPushButton("预建记忆库")
        self.bootstrap_rag_btn.setObjectName("context_bootstrap_rag_btn")
        self.bootstrap_rag_btn.clicked.connect(lambda: self._trigger_prebuild("rag"))
        self.rag_status_row = TaskStatusActionRow(
            "记忆库",
            self.bootstrap_rag_btn,
            parent=self.status_page,
        )
        self.rag_status_label = self.rag_status_row.status_label
        self.status_layout.root.addWidget(self.rag_status_row)

        self.bootstrap_source_index_btn = QPushButton("预建原文索引")
        self.bootstrap_source_index_btn.setObjectName("context_bootstrap_source_index_btn")
        self.bootstrap_source_index_btn.clicked.connect(
            lambda: self._trigger_prebuild("source_index")
        )
        self.source_index_status_row = TaskStatusActionRow(
            "原文索引",
            self.bootstrap_source_index_btn,
            parent=self.status_page,
        )
        self.source_index_status_label = self.source_index_status_row.status_label
        self.status_layout.root.addWidget(self.source_index_status_row)

        # Project analysis (#254): generate / publish / unpublish via coordinator CLI.
        self.project_analysis_generate_btn = QPushButton(PROJECT_ANALYSIS_COPY["start"])
        self.project_analysis_generate_btn.setObjectName(
            "context_project_analysis_primary_btn"
        )
        self.project_analysis_generate_btn.setToolTip(
            "自动衔接可用的剧情概要、结构构建与摘要生成；"
            "生成后进入审查，不会自动用于翻译。"
        )
        self.project_analysis_generate_btn.clicked.connect(
            self._trigger_project_analysis_primary
        )
        self.project_analysis_status_row = TaskStatusActionRow(
            "项目分析",
            self.project_analysis_generate_btn,
            parent=self.status_page,
        )
        self.project_analysis_status_label = self.project_analysis_status_row.status_label
        # Keep alias used by older tests that looked for a readonly badge.
        self.project_analysis_readonly_label = self.project_analysis_generate_btn
        self.status_layout.root.addWidget(self.project_analysis_status_row)

        self.context_actions = self.status_layout.add_section(
            "上下文任务",
            role="context_library",
            secondary=True,
        )
        self.project_analysis_review_btn = QPushButton("审查对照")
        self.project_analysis_review_btn.setObjectName(
            "context_project_analysis_review_btn"
        )
        self.project_analysis_review_btn.setToolTip(
            "查看完整差异、摘要全文、证据来源与实际注入预览。"
        )
        self.project_analysis_review_btn.clicked.connect(
            lambda: self._trigger_action("project_analysis_review")
        )
        self.context_actions.add_action(self.project_analysis_review_btn, min_width=100)

        self.project_analysis_publish_btn = QPushButton("发布 brief")
        self.project_analysis_publish_btn.setObjectName(
            "context_project_analysis_publish_btn"
        )
        self.project_analysis_publish_btn.setToolTip(
            "将 project_brief.draft.md 发布为 published（可注入）。"
            "会校验当前脚本 fingerprint；不匹配时拒绝发布。"
        )
        self.project_analysis_publish_btn.clicked.connect(
            lambda: self._trigger_action("project_analysis_publish")
        )
        self.context_actions.add_action(self.project_analysis_publish_btn, min_width=100)

        self.project_analysis_unpublish_btn = QPushButton("撤销发布")
        self.project_analysis_unpublish_btn.setObjectName(
            "context_project_analysis_unpublish_btn"
        )
        self.project_analysis_unpublish_btn.setToolTip("经确认后撤销 published 副本并停止注入。")
        self.project_analysis_unpublish_btn.clicked.connect(
            lambda: self._trigger_action("project_analysis_unpublish")
        )
        self.context_actions.add_action(self.project_analysis_unpublish_btn, min_width=100)

        self.project_analysis_build_btn = QPushButton("构建结构")
        self.project_analysis_build_btn.setObjectName(
            "context_project_analysis_build_btn"
        )
        self.project_analysis_build_btn.setToolTip(
            "重新解析 label / jump / route，并生成新的待审查摘要。"
        )
        self.project_analysis_build_btn.clicked.connect(
            lambda: self._trigger_action("project_analysis_build_structure")
        )
        self.context_actions.add_action(self.project_analysis_build_btn, min_width=100)

        self.open_settings_btn = QPushButton("打开设置 · 上下文")
        self.open_settings_btn.setObjectName("secondary_btn")
        self.open_settings_btn.setToolTip(
            "开关须在设置中保存后才能预建；打开设置的「上下文」分区。"
        )
        self.open_settings_btn.clicked.connect(self._trigger_open_settings)
        self.context_actions.add_action(self.open_settings_btn, min_width=140)

        self.stop_btn = QPushButton("停止预建")
        self.stop_btn.setObjectName("context_library_stop_btn")
        self.stop_btn.setToolTip("停止当前预建任务")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._trigger_stop)
        self.context_actions.add_action(self.stop_btn, min_width=100)
        self.context_actions.finish_setup()

        self.status_layout.root.addStretch(1)
        self.page_stack.addWidget(self.status_page)
        self.page_stack.setCurrentWidget(self.empty_state)


        self._refresh_action_states()
    def set_action_callbacks(self, actions: WorkbenchPageActions) -> None:
        self._actions = actions

    def activate(self, mode: WorkMode, session: WorkbenchModeSession) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported context-library mode: {mode.value}")
        self._active_mode = mode
        # Session remains owned by MainWindow; the page only renders it when P1
        # eventually gains page-local bootstrap progress presentation.
        del session

    def set_context_status(
        self,
        *,
        rag_enabled: bool,
        source_index_enabled: bool,
        game_root: str,
        project_analysis_status: Mapping[str, Any] | None = None,
        project_analysis_label: str = "",
        project_analysis_enabled: bool = False,
        project_analysis_inject_enabled: bool = False,
    ) -> None:
        """Render context availability and the Project Analysis lifecycle.

        ``rag_enabled`` and ``source_index_enabled`` control their respective
        context rows. ``project_analysis_enabled`` gates lifecycle actions,
        while ``project_analysis_inject_enabled`` controls translation
        eligibility. ``project_analysis_status`` supplies artifact lifecycle
        state, and ``project_analysis_label`` optionally overrides its rendered
        label. A non-empty ``game_root`` indicates that project-scoped actions
        may run.
        """
        self._rag_enabled = rag_enabled
        self._source_index_enabled = source_index_enabled
        root_hint = game_root or "未选择项目"
        self._project_analysis_enabled = project_analysis_enabled
        self._project_analysis_inject_enabled = project_analysis_inject_enabled
        self._project_analysis_status = dict(project_analysis_status or {})
        self._has_project = bool(game_root)
        self.rag_status_row.set_status(
            f"{'已启用' if rag_enabled else '未启用'} · 项目 {root_hint}"
            + ("" if rag_enabled else " · 请先在设置 · 上下文开启并保存")
        )
        self.source_index_status_row.set_status(
            f"{'已启用' if source_index_enabled else '未启用'} · 项目 {root_hint}"
            + ("" if source_index_enabled else " · 请先在设置 · 上下文开启并保存")
        )
        if project_analysis_label:
            analysis_text = project_analysis_label
        elif project_analysis_status is not None:
            from project_analysis import format_status_label

            analysis_text = format_status_label(project_analysis_status)
        else:
            analysis_text = "未检测"
        overall = ""
        if not project_analysis_enabled:
            analysis_text += " · 功能未启用"
        elif project_analysis_inject_enabled and bool(
            (project_analysis_status or {}).get("injectable")
        ):
            analysis_text += " · 当前会用于翻译"
        elif project_analysis_inject_enabled:
            analysis_text += " · 尚不可用于翻译"
        else:
            analysis_text += " · 用于翻译：关闭"
        self.project_analysis_status_row.set_status(
            f"{analysis_text} · 项目 {root_hint}"
        )
        if project_analysis_status is not None:
            overall = str(project_analysis_status.get("overall_status") or "")
            self._project_analysis_present = bool(
                project_analysis_status.get("store_exists")
            ) or overall not in {"", "missing"}
        else:
            self._project_analysis_present = bool(project_analysis_label) and project_analysis_label not in {
                "未检测",
                "未生成",
            }
        show_status = (
            rag_enabled
            or source_index_enabled
            or project_analysis_enabled
            or self._project_analysis_present
        )
        self.page_stack.setCurrentWidget(
            self.status_page if show_status else self.empty_state
        )
        self._refresh_action_states()

    def set_task_running(self, running: bool, operation: str = "") -> None:
        self._running = running
        if not running:
            self.stop_btn.setText("停止")
        elif operation == "project_analysis_build_structure":
            self.stop_btn.setText("停止构建")
        elif operation == "project_analysis_generate":
            self.stop_btn.setText("停止生成")
        elif operation == "project_analysis_workflow":
            self.stop_btn.setText("停止分析")
        elif operation in {"bootstrap_rag", "bootstrap_source_index"}:
            self.stop_btn.setText("停止预建")
        else:
            self.stop_btn.setText("停止任务")
        self._refresh_action_states()

    def reset_project(self) -> None:
        self.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="",
            project_analysis_status=None,
            project_analysis_label="未检测",
        )

    def _refresh_action_states(self) -> None:
        self.bootstrap_rag_btn.setEnabled(not self._running and self._rag_enabled)
        self.bootstrap_source_index_btn.setEnabled(
            not self._running and self._source_index_enabled
        )
        status = self._project_analysis_status
        overall = str(status.get("overall_status") or "missing")
        has_draft = bool(status.get("brief_draft_present")) or overall in {
            "draft",
            "review_required",
        }
        has_published = bool(status.get("brief_published_present"))
        if "structure_present" in status:
            has_structure = bool(status.get("structure_present"))
        else:
            # Compatibility for older/mocked status payloads.
            has_structure = bool(
                has_draft
                or has_published
                or status.get("label_count")
                or status.get("route_count")
            )

        can_manage = (
            not self._running
            and self._has_project
            and self._project_analysis_enabled
        )

        if overall == "missing":
            self._project_analysis_primary_action = (
                "project_analysis_build_structure"
            )
            primary_text = PROJECT_ANALYSIS_COPY["start"]
            primary_tip = "导入可用剧情概要（可跳过），再构建结构并生成待审查摘要。"
        elif overall in {"failed", "stale"} or not has_structure:
            self._project_analysis_primary_action = (
                "project_analysis_build_structure"
            )
            primary_text = PROJECT_ANALYSIS_COPY["rebuild"]
            primary_tip = (
                "上次构建失败，请重新构建并查看运行日志。"
                if overall == "failed"
                else "项目结构不可用或已过期，将重新构建并更新项目摘要。"
            )
        else:
            self._project_analysis_primary_action = "project_analysis_generate"
            primary_text = (
                PROJECT_ANALYSIS_COPY["refresh"]
                if overall == "published"
                else PROJECT_ANALYSIS_COPY["generate"]
            )
            primary_tip = "基于现有结构生成或更新项目摘要；生成后仍需审查并启用。"

        self.project_analysis_generate_btn.setText(primary_text)
        self.project_analysis_generate_btn.setToolTip(
            primary_tip
            if can_manage
            else "请先选择项目，并在设置 · 上下文启用项目剧情分析后保存。"
        )
        self.project_analysis_generate_btn.setEnabled(can_manage)
        self.project_analysis_review_btn.setText(PROJECT_ANALYSIS_COPY["review"])
        self.project_analysis_review_btn.setEnabled(
            can_manage and (has_structure or has_draft or has_published)
        )
        self.project_analysis_publish_btn.setText(PROJECT_ANALYSIS_COPY["publish"])
        self.project_analysis_publish_btn.setEnabled(
            can_manage and has_draft and overall in {"draft", "review_required"}
        )
        self.project_analysis_unpublish_btn.setText(
            PROJECT_ANALYSIS_COPY["unpublish"]
        )
        self.project_analysis_unpublish_btn.setEnabled(
            can_manage and has_published
        )
        self.project_analysis_build_btn.setText("重新分析")
        self.project_analysis_build_btn.setEnabled(can_manage and has_structure)
        self.open_settings_btn.setEnabled(not self._running)
        self.stop_btn.setEnabled(self._running)
        self.context_actions.reflow()

    def _trigger_project_analysis_primary(self) -> None:
        self._trigger_action(self._project_analysis_primary_action)

    def _trigger_prebuild(self, kind: str) -> None:
        if self._running or self._actions.prebuild is None:
            return
        self._actions.prebuild(kind)

    def _trigger_action(self, name: str) -> None:
        if self._running or self._actions.action is None:
            return
        self._actions.action(name)

    def _trigger_open_settings(self) -> None:
        if not self._running and self._actions.open_settings is not None:
            self._actions.open_settings()

    def _trigger_stop(self) -> None:
        if self._running and self._actions.stop is not None:
            self._actions.stop()
