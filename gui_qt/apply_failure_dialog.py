"""Dialog for displaying structured apply failure diagnostics."""
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

from .apply_failure_report import ApplyFailureReportView


class ApplyFailureDialog(QDialog):
    """Modal dialog that presents an apply failure diagnostic report."""

    def __init__(self, parent: QWidget | None, *, report: ApplyFailureReportView):
        super().__init__(parent)
        self.setObjectName("apply_failure_dialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("写回失败报告")
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

        layout.addWidget(QLabel("诊断详情："))

        self.detail_view = QPlainTextEdit()
        self.detail_view.setObjectName("apply_failure_detail_view")
        self.detail_view.setReadOnly(True)
        self.detail_view.setPlainText("\n".join(report.detail_lines))
        self.detail_view.setMinimumHeight(280)
        layout.addWidget(self.detail_view)

        actions = QHBoxLayout()
        if report.report_path:
            copy_report_btn = QPushButton("复制报告路径")
            copy_report_btn.setObjectName("secondary_btn")
            copy_report_btn.clicked.connect(self._copy_report_path)
            actions.addWidget(copy_report_btn)
        if report.failures_path:
            copy_failures_btn = QPushButton("复制失败明细路径")
            copy_failures_btn.setObjectName("secondary_btn")
            copy_failures_btn.clicked.connect(self._copy_failures_path)
            actions.addWidget(copy_failures_btn)
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
        self._copy_text(self._report.report_path)

    def _copy_failures_path(self) -> None:
        self._copy_text(self._report.failures_path)

    def _copy_text(self, text: str) -> None:
        from PySide6.QtGui import QGuiApplication

        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)