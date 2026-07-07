"""Dialog for previewing a generated retry package before follow-up steps."""
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

from .retry_report import RetryPreviewReport


class RetryPreviewDialog(QDialog):
    """Modal dialog that previews retry package scope and asks for confirmation."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        report: RetryPreviewReport,
        confirm_label: str = "确认并继续补译",
    ):
        super().__init__(parent)
        self.setObjectName("retry_preview_dialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("补译包预览")
        self.setModal(True)
        self.resize(720, 560)
        self._report = report
        self._confirmed = False

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

        layout.addWidget(QLabel("预览详情："))

        self.detail_view = QPlainTextEdit()
        self.detail_view.setObjectName("retry_preview_detail_view")
        self.detail_view.setReadOnly(True)
        self.detail_view.setPlainText("\n".join(report.detail_lines))
        self.detail_view.setMinimumHeight(280)
        layout.addWidget(self.detail_view)

        actions = QHBoxLayout()
        if report.retry_manifest_path:
            copy_manifest_btn = QPushButton("复制任务记录路径")
            copy_manifest_btn.setObjectName("secondary_btn")
            copy_manifest_btn.clicked.connect(self._copy_retry_manifest_path)
            actions.addWidget(copy_manifest_btn)
        if report.package_dir:
            copy_package_btn = QPushButton("复制补译包路径")
            copy_package_btn.setObjectName("secondary_btn")
            copy_package_btn.clicked.connect(self._copy_package_dir)
            actions.addWidget(copy_package_btn)
        actions.addStretch()
        layout.addLayout(actions)

        buttons = QDialogButtonBox()
        confirm_btn = buttons.addButton(confirm_label, QDialogButtonBox.ButtonRole.AcceptRole)
        close_btn = buttons.addButton("关闭", QDialogButtonBox.ButtonRole.RejectRole)
        if confirm_btn is not None:
            confirm_btn.setObjectName("apply_btn")
        if close_btn is not None:
            close_btn.setObjectName("secondary_btn")
        buttons.accepted.connect(self._on_confirm)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    def _on_confirm(self) -> None:
        self._confirmed = True
        self.accept()

    def _copy_retry_manifest_path(self) -> None:
        self._copy_text(self._report.retry_manifest_path)

    def _copy_package_dir(self) -> None:
        self._copy_text(self._report.package_dir)

    def _copy_text(self, text: str) -> None:
        from PySide6.QtGui import QGuiApplication

        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)