"""GUI candidate filtering and explicit selection for staged proposals."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import revision_selection

from .user_copy import REVISION_PROPOSAL_COPY


_STATUS_LABELS = {
    revision_selection.STATUS_VALID: "有效",
    revision_selection.STATUS_NO_OP: "无需修改",
    revision_selection.STATUS_INVALID: "无效",
    revision_selection.STATUS_STALE: "过期",
    revision_selection.STATUS_CONFLICT: "冲突",
}


class RevisionProposalSelectionDialog(QDialog):
    """Display a staged candidate set and return an explicit selection request."""

    def __init__(
        self,
        stage: Mapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.stage = dict(stage)
        self._selection_cache = {
            str(candidate.get("identity_v2") or "")
            for candidate in self.stage.get("candidates") or []
            if candidate.get("selected") and candidate.get("selectable")
        }
        self.setWindowTitle(REVISION_PROPOSAL_COPY["selection_dialog_title"])
        self.resize(940, 600)
        self.setMinimumSize(760, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(7)

        hint = QLabel(
            REVISION_PROPOSAL_COPY["selection_hint"]
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("revision_selection_summary")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        filters = QGridLayout()
        filters.setHorizontalSpacing(6)
        filters.setVerticalSpacing(5)
        self.reason_combo = self._filter_combo("原因")
        self.file_combo = self._filter_combo("文件")
        self.status_combo = self._filter_combo("状态")
        filters.addWidget(QLabel("原因："), 0, 0)
        filters.addWidget(self.reason_combo, 0, 1)
        filters.addWidget(QLabel("文件："), 0, 2)
        filters.addWidget(self.file_combo, 0, 3)
        filters.addWidget(QLabel("状态："), 0, 4)
        filters.addWidget(self.status_combo, 0, 5)
        self.valid_only_cb = QCheckBox(REVISION_PROPOSAL_COPY["selection_valid_only"])
        self.valid_only_cb.setObjectName("revision_selection_valid_only")
        filters.addWidget(self.valid_only_cb, 1, 0, 1, 2)
        self.select_all_btn = QPushButton(REVISION_PROPOSAL_COPY["selection_select_all"])
        self.select_all_btn.setObjectName("revision_selection_select_all_btn")
        filters.addWidget(self.select_all_btn, 1, 2)
        self.clear_btn = QPushButton(REVISION_PROPOSAL_COPY["selection_clear"])
        self.clear_btn.setObjectName("revision_selection_clear_btn")
        filters.addWidget(self.clear_btn, 1, 3)
        filters.setColumnStretch(1, 1)
        filters.setColumnStretch(3, 1)
        filters.setColumnStretch(5, 1)
        layout.addLayout(filters)

        self.table = QTableWidget(0, 8)
        self.table.setObjectName("revision_selection_table")
        self.table.setHorizontalHeaderLabels(
            ["选择", "状态", "文件", "原因", "原文", "当前译文", "建议译文", "identity"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(270)
        self.table.setColumnWidth(0, 45)
        self.table.setColumnWidth(1, 68)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnWidth(3, 120)
        self.table.setColumnWidth(4, 190)
        self.table.setColumnWidth(5, 190)
        self.table.setColumnWidth(6, 190)
        self.table.setColumnWidth(7, 150)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setText(REVISION_PROPOSAL_COPY["selection_confirm"])
        self._ok_button.setObjectName("revision_selection_confirm_btn")
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.reason_combo.currentIndexChanged.connect(self._refresh_table)
        self.file_combo.currentIndexChanged.connect(self._refresh_table)
        self.status_combo.currentIndexChanged.connect(self._refresh_table)
        self.valid_only_cb.toggled.connect(self._refresh_table)
        self.select_all_btn.clicked.connect(self._select_all_valid)
        self.clear_btn.clicked.connect(self._clear_selection)
        self.table.itemChanged.connect(self._sync_accept_enabled)
        self._populate_filters()
        self._refresh_table()

    def _filter_combo(self, object_name: str) -> QComboBox:
        combo = QComboBox()
        combo.setObjectName(f"revision_selection_{object_name}_combo")
        combo.addItem("全部", "")
        combo.setMinimumWidth(110)
        return combo

    def _populate_filters(self) -> None:
        candidates = list(self.stage.get("candidates") or [])
        for combo, key in (
            (self.reason_combo, "reason"),
            (self.file_combo, "file_rel_path"),
            (self.status_combo, "status"),
        ):
            values = sorted(
                {
                    str(candidate.get(key) or "").strip()
                    for candidate in candidates
                    if str(candidate.get(key) or "").strip()
                }
            )
            for value in values:
                label = _STATUS_LABELS.get(value, value) if key == "status" else value
                combo.addItem(label, value)

    def _current_filter(self, combo: QComboBox) -> str:
        return str(combo.currentData() or "").strip()

    def _filtered_candidates(self) -> list[dict[str, Any]]:
        return revision_selection.filter_candidates(
            self.stage,
            reason=self._current_filter(self.reason_combo),
            file_rel_path=self._current_filter(self.file_combo),
            status=self._current_filter(self.status_combo),
            valid_only=self.valid_only_cb.isChecked(),
        )

    def _refresh_table(self, *_args: object) -> None:
        candidates = self._filtered_candidates()
        selected = self.selected_identity_v2()
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(0)
            for candidate in candidates:
                row = self.table.rowCount()
                self.table.insertRow(row)
                selectable = bool(candidate.get("selectable"))
                check = QTableWidgetItem()
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
                check.setFlags(flags if selectable else Qt.ItemFlag.ItemIsEnabled)
                check.setCheckState(
                    Qt.CheckState.Checked
                    if str(candidate.get("identity_v2") or "") in selected
                    else Qt.CheckState.Unchecked
                )
                check.setToolTip(
                    "有效候选，可选择。"
                    if selectable
                    else "无效、过期、冲突或无需修改，不能选择。"
                )
                self.table.setItem(row, 0, check)
                status = str(candidate.get("status") or "")
                values = (
                    _STATUS_LABELS.get(status, status),
                    candidate.get("file_rel_path"),
                    candidate.get("reason"),
                    candidate.get("source"),
                    candidate.get("current_translation"),
                    candidate.get("proposed_translation"),
                    candidate.get("identity_v2"),
                )
                for column, value in enumerate(values, start=1):
                    text = str(value or "")
                    item = QTableWidgetItem(text)
                    item.setToolTip(text)
                    self.table.setItem(row, column, item)
        finally:
            self.table.blockSignals(False)
        self._sync_accept_enabled()

    def _select_all_valid(self) -> None:
        visible_identities = {
            str(candidate.get("identity_v2") or "")
            for candidate in self._filtered_candidates()
            if candidate.get("selectable")
        }
        self._selection_cache = {
            identity
            for identity in self.selected_identity_v2()
            if identity not in visible_identities
        }
        self._selection_cache.update(visible_identities)
        self.table.blockSignals(True)
        try:
            for row in range(self.table.rowCount()):
                check = self.table.item(row, 0)
                if check is not None and check.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    check.setCheckState(Qt.CheckState.Checked)
        finally:
            self.table.blockSignals(False)
        self._sync_accept_enabled()

    def _clear_selection(self) -> None:
        self._selection_cache = set()
        self.table.blockSignals(True)
        try:
            for row in range(self.table.rowCount()):
                check = self.table.item(row, 0)
                if check is not None and check.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    check.setCheckState(Qt.CheckState.Unchecked)
        finally:
            self.table.blockSignals(False)
        self._sync_accept_enabled()

    def selected_identity_v2(self) -> list[str]:
        identities: list[str] = []
        for row in range(self.table.rowCount()):
            check = self.table.item(row, 0)
            identity_item = self.table.item(row, 7)
            if (
                check is not None
                and identity_item is not None
                and check.checkState() == Qt.CheckState.Checked
            ):
                identities.append(identity_item.text())
        # A filtered table only owns visible checkboxes; preserve checked rows
        # from the stage when a filter is changed by keeping a small selection
        # cache in the dialog.
        cached = set(getattr(self, "_selection_cache", set()))
        cached.update(identities)
        visible = {
            self.table.item(row, 7).text()
            for row in range(self.table.rowCount())
            if self.table.item(row, 7) is not None
        }
        cached.difference_update(
            identity for identity in visible if identity not in identities
        )
        self._selection_cache = cached
        return [
            str(candidate.get("identity_v2") or "")
            for candidate in self.stage.get("candidates") or []
            if str(candidate.get("identity_v2") or "") in cached
        ]

    def _sync_accept_enabled(self, *_args: object) -> None:
        selected_count = len(self.selected_identity_v2())
        summary = self.stage.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        self.summary_label.setText(
            " · ".join(
                (
                    f"总计 {int(summary.get('total_count') or 0)}",
                    f"有效 {int(summary.get('selectable_count') or 0)}",
                    f"未选择 {int(summary.get('unselected_count') or 0)}",
                    f"无需修改 {int(summary.get('no_op_count') or 0)}",
                    f"无效 {int(summary.get('invalid_count') or 0)}",
                    f"过期 {int(summary.get('stale_count') or 0)}",
                    f"冲突 {int(summary.get('conflict_count') or 0)}",
                    f"当前选择 {selected_count}",
                )
            )
        )
        self._ok_button.setEnabled(bool(selected_count))

    def _accept_selection(self) -> None:
        if self.selected_identity_v2():
            self.accept()

    def selection_request(self) -> dict[str, Any]:
        return revision_selection.make_selection_request(
            self.stage,
            self.selected_identity_v2(),
        )
