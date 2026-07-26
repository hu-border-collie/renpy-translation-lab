"""Human selection dialog for final-review findings."""
from __future__ import annotations

from typing import Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QLabel, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)


class FinalReviewFindingsDialog(QDialog):
    def __init__(self, findings: Sequence[Mapping[str, object]], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("选择需要订正的问题")
        self.resize(1100, 620)
        layout = QVBoxLayout(self)
        hint = QLabel("最终审校默认只报告。勾选的问题会先生成订正预览；此操作不会修改 .rpy 文件。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["选择", "严重度", "类型", "文件", "当前译文", "建议译文", "原因"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        self._finding_ids: list[str] = []
        for finding in findings:
            if not str(finding.get("suggested_revision") or "").strip():
                continue
            if str(finding.get("revision_state") or "") == "applied":
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            finding_id = str(finding.get("finding_id") or "")
            self._finding_ids.append(finding_id)
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Checked if finding.get("selection_state") == "selected" else Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, check)
            values = (
                finding.get("severity"), finding.get("finding_type"), finding.get("file_rel_path"),
                finding.get("current_translation"), finding.get("suggested_revision"), finding.get("reason"),
            )
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value or ""))
                item.setToolTip(str(value or ""))
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("生成订正预览")
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_selection(self) -> None:
        if self.selected_finding_ids():
            self.accept()

    def selected_finding_ids(self) -> list[str]:
        return [finding_id for row, finding_id in enumerate(self._finding_ids)
                if self.table.item(row, 0).checkState() == Qt.CheckState.Checked]
