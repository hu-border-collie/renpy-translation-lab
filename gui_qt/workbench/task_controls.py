"""Shared task-page layout primitives for workbench workflows."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..status_icons import StatusBadge

from ..responsive_layout import FlowButtonBar


TASK_PAGE_MIN_WIDTH = 260
_NON_RESULT_TASK_STATUSES = frozenset(
    {"", "idle", "stale", "running", "waiting"}
)


def task_status_has_result(status: str) -> bool:
    """Return whether a page snapshot is a result worth keeping behind its gate."""
    return str(status or "").strip() not in _NON_RESULT_TASK_STATUSES


class TaskControlSection(QFrame):
    """Titled, responsive action section shared by task pages."""

    def __init__(
        self,
        title: str,
        *,
        role: str,
        secondary: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("task_control_section")
        self.setProperty("taskRole", role)
        self.setProperty("sectionLevel", "secondary" if secondary else "primary")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("task_control_section_title")
        layout.addWidget(self.title_label)

        self.action_bar = FlowButtonBar(spacing=8, row_spacing=8)
        self.action_bar.setObjectName(f"{role}_actions")
        self.action_bar.setProperty("taskRole", role)
        layout.addWidget(self.action_bar)

    def add_action(self, widget: QWidget, *, min_width: int = 88) -> QWidget:
        return self.action_bar.add_widget(widget, min_width=min_width)

    def finish_setup(self) -> None:
        self.action_bar.finish_setup()

    def reflow(self) -> None:
        self.action_bar.reflow(force=True)


class TaskStatusSection(QFrame):
    """Page-local workflow / result status area for task pages (#298).

    Renders the same badge + message + progress + facts vocabulary the shared
    status card used, but inside the owning task page so every workbench route
    shows only its own status chrome.
    """

    def __init__(
        self,
        title: str = "任务状态",
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("task_status_section")
        self.setProperty("taskRole", "status")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("task_control_section_title")
        layout.addWidget(self.title_label)

        self.status_badge = StatusBadge("task_status_badge")
        layout.addWidget(self.status_badge)

        self.message_label = QLabel("")
        self.message_label.setObjectName("task_status_message")
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("task_status_progress")
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.facts_label = QLabel("")
        self.facts_label.setObjectName("task_status_facts")
        self.facts_label.setWordWrap(True)
        self.facts_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.facts_label)

        self.details_label = QLabel("")
        self.details_label.setObjectName("task_status_details")
        self.details_label.setWordWrap(True)
        self.details_label.setTextFormat(Qt.TextFormat.PlainText)
        self.details_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.details_label.setVisible(False)
        layout.addWidget(self.details_label)

    def set_status(
        self,
        status: str,
        heading: str,
        message: str,
        facts: list[str] | None = None,
    ) -> None:
        """Render one workflow / writeback snapshot inside the page."""
        self.status_badge.set_status(status, heading)
        self.message_label.setText(message)
        self.facts_label.setText("\n".join(facts or []))
        # A new snapshot supersedes previous progress/details; callers that
        # need them re-apply via set_progress/set_details.
        self.progress_bar.setVisible(False)
        self.details_label.setText("")
        self.details_label.setVisible(False)
        self.updateGeometry()

    def reflow(self) -> None:
        """Compatibility with TaskPageLayout.sections bookkeeping."""
        self.updateGeometry()

    def set_details(self, lines: list[str] | None) -> None:
        """Render optional issue/notice lines; hidden when empty."""
        cleaned = [str(x).strip() for x in (lines or []) if str(x).strip()]
        if not cleaned:
            self.details_label.setText("")
            self.details_label.setVisible(False)
            return
        self.details_label.setText("\n".join(cleaned))
        self.details_label.setVisible(True)
        self.updateGeometry()

    def set_progress(self, state: object | None) -> None:
        """Render an optional progress state; ``None`` hides the bar."""
        visible = bool(state is not None and getattr(state, "visible", False))
        if not visible:
            self.progress_bar.setVisible(False)
            self.updateGeometry()
            return
        indeterminate = bool(getattr(state, "indeterminate", False))
        total = int(getattr(state, "total", 0) or 0)
        if indeterminate or total <= 0:
            self.progress_bar.setRange(0, 0)
        else:
            current = min(max(int(getattr(state, "current", 0) or 0), 0), total)
            self.progress_bar.setRange(0, max(total, 1))
            self.progress_bar.setValue(current)
        self.progress_bar.setFormat(getattr(state, "label", None) or "正在处理…")
        self.progress_bar.setVisible(True)
        self.updateGeometry()


class TaskStatusActionRow(QFrame):
    """Compact resource status with its directly related action kept nearby."""

    def __init__(
        self,
        title: str,
        action: QWidget,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("task_status_action_row")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QGridLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("task_status_title")
        self.title_label.setMinimumWidth(72)
        layout.addWidget(
            self.title_label,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self.status_label = QLabel("—")
        self.status_label.setObjectName("task_status_detail")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(self.status_label, 0, 1)
        layout.setColumnStretch(1, 1)

        action.setMinimumWidth(116)
        action.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.action = action
        layout.addWidget(
            action,
            0,
            2,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.updateGeometry()


class TaskPageLayout:
    """Shared vertical anatomy for task notices and action sections."""

    def __init__(self, page: QWidget, *, spacing: int = 8) -> None:
        self.page = page
        self.sections: list[TaskControlSection | TaskStatusSection] = []
        self.root = QVBoxLayout(page)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.root.setSpacing(spacing)
        self.root.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

    def add_notice(self, text: str, *, tone: str = "warning") -> QLabel:
        notice = QLabel(text)
        notice.setObjectName("task_page_notice")
        notice.setProperty("tone", tone)
        notice.setWordWrap(True)
        notice.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.root.addWidget(notice)
        return notice

    def add_mode_selector(self, label_text: str, combo: QComboBox) -> QLabel:
        row = QFrame(self.page)
        row.setObjectName("task_mode_row")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        label = QLabel(label_text)
        label.setObjectName("task_mode_label")
        row_layout.addWidget(label)

        combo.setMinimumWidth(160)
        combo.setMaximumWidth(240)
        combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row_layout.addWidget(combo)
        row_layout.addStretch(1)
        self.root.addWidget(row)
        return label

    def add_status_section(
        self,
        title: str = "任务状态",
    ) -> TaskStatusSection:
        """Add the page-local status area below the action sections."""
        section = TaskStatusSection(title, parent=self.page)
        self.sections.append(section)
        self.root.addWidget(section)
        return section

    def add_result_hint(self, text: str) -> QLabel:
        hint = QLabel(text)
        hint.setObjectName("task_result_hint")
        hint.setWordWrap(True)
        hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.root.addWidget(hint)
        return hint

    def add_section(
        self,
        title: str,
        *,
        role: str,
        secondary: bool = False,
    ) -> TaskControlSection:
        section = TaskControlSection(
            title,
            role=role,
            secondary=secondary,
            parent=self.page,
        )
        self.sections.append(section)
        self.root.addWidget(section)
        return section

    def reflow(self) -> None:
        for section in self.sections:
            section.reflow()

    def preferred_height(self, width: int) -> int:
        self.reflow()
        content_width = max(width, TASK_PAGE_MIN_WIDTH)
        return max(
            self.page.minimumSizeHint().height(),
            self.root.heightForWidth(content_width),
        )
