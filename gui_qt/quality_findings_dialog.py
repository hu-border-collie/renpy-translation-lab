"""Dialog for filtering and reviewing quality_findings.jsonl alarms."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import translation_quality
from quality_report_export import (
    DEFAULT_REPORT_FILENAME,
    QualityReportExportError,
    export_quality_report,
)

from .quality_findings_report import (
    QualityFindingsReport,
    _format_item_line,
    acknowledged_finding_ids_from_manifest,
    filter_quality_items,
    persist_quality_acknowledgement,
    reason_label,
    severity_label,
)
from .user_copy import (
    QUALITY_REPORT_EXPORT_LABEL,
    QUALITY_REPORT_EXPORT_SUCCESS,
    QUALITY_REPORT_EXPORT_TITLE,
)


class QualityFindingsDialog(QDialog):
    """Modal dialog that displays quality alarms with acknowledgement actions."""

    quality_acknowledged = Signal()

    def __init__(
        self,
        parent: QWidget | None,
        *,
        report: QualityFindingsReport,
        manifest: dict[str, object] | None = None,
        manifest_path: str = "",
    ):
        super().__init__(parent)
        self.setObjectName("quality_findings_dialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("译文质量报警")
        self.setModal(True)
        self.resize(800, 660)
        self._report = report
        self._items = list(report.items)
        self._manifest = dict(manifest or {})
        self._manifest_path = manifest_path
        self._acknowledged_ids = acknowledged_finding_ids_from_manifest(
            self._manifest
        )
        self._visible_items: list[object] = []

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
        layout.addLayout(filters)

        filters2 = QHBoxLayout()
        filters2.addWidget(QLabel("最低严重程度："))
        self.severity_filter = QComboBox()
        self.severity_filter.addItem("全部", "")
        for severity in ("info", "low", "medium", "high"):
            self.severity_filter.addItem(severity_label(severity), severity)
        filters2.addWidget(self.severity_filter)

        filters2.addWidget(QLabel("确认状态："))
        self.state_filter = QComboBox()
        self.state_filter.setObjectName("quality_state_filter")
        self.state_filter.addItem("全部状态", "")
        self.state_filter.addItem("未确认", "unacknowledged")
        self.state_filter.addItem("已确认", "acknowledged")
        filters2.addWidget(self.state_filter)
        layout.addLayout(filters2)

        self.count_label = QLabel("")
        self.count_label.setObjectName("quality_findings_count_label")
        layout.addWidget(self.count_label)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("quality_findings_list")
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setMinimumHeight(260)
        layout.addWidget(self.list_widget)

        self.rule_filter.currentIndexChanged.connect(self._refresh)
        self.file_filter.textChanged.connect(self._refresh)
        self.severity_filter.currentIndexChanged.connect(self._refresh)
        self.state_filter.currentIndexChanged.connect(self._refresh)
        self.list_widget.itemSelectionChanged.connect(self._update_action_buttons)

        actions = QHBoxLayout()
        self.ack_selected_btn = QPushButton("确认所选")
        self.ack_selected_btn.setObjectName("quality_ack_selected_btn")
        self.ack_selected_btn.setToolTip(
            "仅确认当前列表里选中的 warning 报警；blocker 不能被确认。"
        )
        self.ack_all_btn = QPushButton("确认全部")
        self.ack_all_btn.setObjectName("quality_ack_all_btn")
        self.ack_all_btn.setToolTip(
            "确认当前筛选结果中的全部 warning 报警；blocker 不能被确认。"
        )
        self.export_html_btn = QPushButton(QUALITY_REPORT_EXPORT_LABEL)
        self.export_html_btn.setObjectName("quality_export_html_btn")
        self.export_html_btn.setToolTip(
            "生成可离线打开、筛选和打印的单文件报告；不会修改译文或 manifest。"
        )
        actions.addWidget(self.ack_selected_btn)
        actions.addWidget(self.ack_all_btn)
        actions.addStretch(1)
        actions.addWidget(self.export_html_btn)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setText("关闭")
        layout.addWidget(buttons)

        self.ack_selected_btn.clicked.connect(
            lambda: self._apply_acknowledgement(selected_only=True)
        )
        self.ack_all_btn.clicked.connect(
            lambda: self._apply_acknowledgement(selected_only=False)
        )
        self.export_html_btn.clicked.connect(self._export_html_report)
        self.export_html_btn.setEnabled(bool(self._manifest_path))

        self._refresh()

    def _export_html_report(self) -> None:
        if not self._manifest_path:
            QMessageBox.warning(self, "无法导出", "当前任务没有可用的 manifest 路径。")
            return
        base_dir = (
            str(Path(self._report.report_path).parent)
            if self._report.report_path
            else str(Path(self._manifest_path).parent)
        )
        suggested = str(Path(base_dir) / DEFAULT_REPORT_FILENAME)
        output_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            QUALITY_REPORT_EXPORT_TITLE,
            suggested,
            "HTML 报告 (*.html)",
        )
        if not output_path:
            return
        try:
            import gemini_translate_batch as batch

            result = export_quality_report(
                self._manifest,
                manifest_path=self._manifest_path,
                output_path=output_path,
                protected_paths=batch.collect_manifest_protected_paths(
                    self._manifest_path
                ),
            )
        except (OSError, QualityReportExportError) as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QMessageBox.information(
            self,
            QUALITY_REPORT_EXPORT_TITLE,
            f"{QUALITY_REPORT_EXPORT_SUCCESS}\n{result['output_path']}",
        )

    def _refresh(self) -> None:
        selected_reason = str(self.rule_filter.currentData() or "")
        selected_file = self.file_filter.text().strip()
        selected_severity = str(self.severity_filter.currentData() or "")
        selected_state = str(self.state_filter.currentData() or "")
        items = filter_quality_items(
            self._items,
            reason_code=selected_reason,
            file_text=selected_file,
            min_severity=selected_severity,
        )
        if selected_state == "unacknowledged":
            items = [
                item for item in items
                if item.finding_id not in self._acknowledged_ids
            ]
        elif selected_state == "acknowledged":
            items = [
                item for item in items
                if item.finding_id in self._acknowledged_ids
            ]
        self._visible_items = items
        visible_acknowledged = sum(
            1 for item in items if item.finding_id in self._acknowledged_ids
        )
        self.count_label.setText(
            f"显示 {len(items)} / {len(self._items)} 条；"
            f"其中已确认 {visible_acknowledged} 条，"
            f"未确认 {len(items) - visible_acknowledged} 条。"
        )
        self._render_items(items)
        self._update_action_buttons()

    def _render_items(self, items: list[object]) -> None:
        current_selected = {
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.list_widget.selectedItems()
        }
        self.list_widget.clear()
        for item in items:
            acknowledged = item.finding_id in self._acknowledged_ids
            list_item = QListWidgetItem(
                _format_item_line(item, acknowledged=acknowledged)
            )
            list_item.setData(Qt.ItemDataRole.UserRole, item.finding_id)
            if acknowledged:
                list_item.setForeground(QBrush(QColor("gray")))
            self.list_widget.addItem(list_item)
            if item.finding_id in current_selected:
                list_item.setSelected(True)

    def _update_action_buttons(self) -> None:
        has_selection = bool(self.list_widget.selectedItems())
        self.ack_selected_btn.setEnabled(has_selection)
        self.ack_all_btn.setEnabled(len(self._visible_items) > 0)

    def _warning_target_ids(self, items: list[object]) -> list[str]:
        return [
            item.finding_id
            for item in items
            if item.finding_id
            and item.disposition == translation_quality.DISPOSITION_WARNING
            and item.finding_id not in self._acknowledged_ids
        ]

    def _apply_acknowledgement(self, *, selected_only: bool) -> None:
        if selected_only:
            selected_ids = {
                item.data(Qt.ItemDataRole.UserRole)
                for item in self.list_widget.selectedItems()
            }
            target_items = [
                item for item in self._visible_items if item.finding_id in selected_ids
            ]
        else:
            target_items = list(self._visible_items)

        target_ids = self._warning_target_ids(target_items)
        if not target_ids:
            QMessageBox.information(
                self,
                "无需确认",
                "当前范围内没有可确认的未确认 warning 报警。\n"
                "blocker 报警不能通过确认来解除写回阻断。",
            )
            return

        filter_active = bool(
            str(self.rule_filter.currentData() or "")
            or self.file_filter.text().strip()
            or str(self.severity_filter.currentData() or "")
            or str(self.state_filter.currentData() or "")
        )
        scope_note = (
            "当前筛选条件生效，确认范围仅限于筛选结果中的报警。\n"
            if filter_active
            else ""
        )
        answer = QMessageBox.question(
            self,
            "确认质量报警",
            f"将确认 {len(target_ids)} 条未确认 warning 报警。\n"
            f"{scope_note}"
            "确认后这些报警仍会保留在质量检查报告中；"
            "译文变化或重新检查后，不再匹配的确认会自动失效。\n\n"
            "确认不会解除 blocker 或结构检查的写回阻断。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        if not self._manifest_path:
            QMessageBox.warning(
                self,
                "无法确认",
                "当前任务没有可用的 manifest 路径。",
            )
            return
        try:
            applied = persist_quality_acknowledgement(
                self._manifest_path,
                finding_ids=target_ids,
                unack=False,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "确认失败", str(exc))
            return

        self._acknowledged_ids = {
            str(value).strip()
            for value in applied.get("acknowledged_finding_ids") or []
            if str(value).strip()
        }
        updated_manifest = applied.get("manifest")
        if isinstance(updated_manifest, dict):
            self._manifest = dict(updated_manifest)
        self._refresh()
        self.quality_acknowledged.emit()
