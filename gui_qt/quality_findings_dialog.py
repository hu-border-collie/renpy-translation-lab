"""Dialog for filtering and reviewing quality_findings.jsonl alarms."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from .quality_findings_report import (
    QualityFindingsReport,
    filter_quality_items,
    reason_label,
    severity_label,
    _format_item_line,
)


class QualityFindingsDialog(QDialog):
    """Modal dialog that displays quality alarms with rule/file/severity filters."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        report: QualityFindingsReport,
    ):
        super().__init__(parent)
        self.setObjectName("quality_findings_dialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("译文质量报警")
        self.setModal(True)
        self.resize(760, 620)
        self._report = report
        self._items = list(report.items)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

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

        filters = QHBoxLayout()
        filters.addWidget(QLabel("规则："))
        self.rule_filter = QComboBox()
        self.rule_filter.addItem("全部规则", "")
        reason_codes = sorted({item.reason_code for item in self._items})
        for reason_code in reason_codes:
            self.rule_filter.addItem(
                f"{reason_label(reason_code)} ({reason_code})",
                reason_code,
            )
        filters.addWidget(self.rule_filter, 1)

        filters.addWidget(QLabel("文件："))
        self.file_filter = QLineEdit()
        self.file_filter.setPlaceholderText("输入文件名片段")
        self.file_filter.setClearButtonEnabled(True)
        filters.addWidget(self.file_filter, 1)

        filters.addWidget(QLabel("最低严重程度："))
        self.severity_filter = QComboBox()
        self.severity_filter.addItem("全部", "")
        for severity in ("info", "low", "medium", "high"):
            self.severity_filter.addItem(severity_label(severity), severity)
        filters.addWidget(self.severity_filter)
        layout.addLayout(filters)

        self.detail_view = QPlainTextEdit()
        self.detail_view.setObjectName("quality_findings_detail_view")
        self.detail_view.setReadOnly(True)
        self.detail_view.setMinimumHeight(300)
        layout.addWidget(self.detail_view)

        self.rule_filter.currentIndexChanged.connect(self._refresh)
        self.file_filter.textChanged.connect(self._refresh)
        self.severity_filter.currentIndexChanged.connect(self._refresh)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setText("关闭")
        layout.addWidget(buttons)

        self._refresh()

    def _refresh(self) -> None:
        selected_reason = str(self.rule_filter.currentData() or "")
        selected_file = self.file_filter.text().strip()
        selected_severity = str(self.severity_filter.currentData() or "")
        items = filter_quality_items(
            self._items,
            reason_code=selected_reason,
            file_text=selected_file,
            min_severity=selected_severity,
        )
        lines = [f"显示 {len(items)} / {len(self._items)} 条。"]
        if items:
            lines.append("")
            lines.extend(_format_item_line(item) for item in items)
        self.detail_view.setPlainText("\n".join(lines))
