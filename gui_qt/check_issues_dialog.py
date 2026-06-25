"""Dialog for displaying structured check=warn issue lists."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .check_failures_report import CheckIssuesReport, format_category_overview


class CheckIssuesDialog(QDialog):
    """Modal dialog that presents a structured check=warn issue report."""

    def __init__(self, parent: QWidget | None, *, report: CheckIssuesReport):
        super().__init__(parent)
        self.setObjectName("check_issues_dialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("检查问题清单")
        self.setModal(True)
        self.resize(720, 560)
        self._report = report

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        heading = QLabel(report.heading)
        heading.setObjectName("writeback_status_label")
        layout.addWidget(heading)

        message = QLabel(report.message)
        message.setWordWrap(True)
        message.setObjectName("summary_body_label")
        layout.addWidget(message)

        if report.facts:
            facts = QLabel("\n".join(report.facts))
            facts.setWordWrap(True)
            facts.setObjectName("writeback_facts_label")
            layout.addWidget(facts)

        overview_lines = format_category_overview(report.category_counts)
        if overview_lines:
            overview = QLabel("【处理建议分类】\n" + "\n".join(overview_lines))
            overview.setWordWrap(True)
            overview.setObjectName("config_hint_label")
            layout.addWidget(overview)

        layout.addWidget(QLabel("问题详情："))

        self.detail_view = QPlainTextEdit()
        self.detail_view.setObjectName("check_issues_detail_view")
        self.detail_view.setReadOnly(True)
        self.detail_view.setPlainText("\n".join(report.detail_lines))
        self.detail_view.setMinimumHeight(280)
        layout.addWidget(self.detail_view)

        actions = QHBoxLayout()
        if report.report_path:
            copy_path_btn = QPushButton("复制报告路径")
            copy_path_btn.setObjectName("secondary_btn")
            copy_path_btn.clicked.connect(self._copy_report_path)
            actions.addWidget(copy_path_btn)
        actions.addStretch()
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setText("关闭")
        layout.addWidget(buttons)

    def _copy_report_path(self) -> None:
        from PySide6.QtGui import QGuiApplication

        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._report.report_path)