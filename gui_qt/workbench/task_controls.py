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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..responsive_layout import FlowButtonBar


TASK_PAGE_MIN_WIDTH = 260


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
        self.sections: list[TaskControlSection] = []
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
