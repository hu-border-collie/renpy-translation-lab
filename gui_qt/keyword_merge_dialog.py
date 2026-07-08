"""Dialog for reviewing and merging keyword candidates into glossary.json."""
from __future__ import annotations

from dataclasses import dataclass

import keyword_glossary_merge as merge_mod
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .keyword_merge_report import format_merge_preview_text


@dataclass(frozen=True)
class KeywordMergeDialogResult:
    summary: merge_mod.MergeSummary
    dry_run: bool


class KeywordMergeDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        *,
        rows: list[merge_mod.CandidateMergeRow],
        candidates_path: str,
        glossary_path: str,
        candidates: list[dict],
        min_confidence: float = 0.0,
    ):
        super().__init__(parent)
        self.setWindowTitle("合并关键词到 glossary")
        self.setModal(True)
        self.resize(960, 640)
        self._rows = rows
        self._candidates = candidates
        self._candidates_path = candidates_path
        self._glossary_path = glossary_path
        self._min_confidence = min_confidence
        self._result: KeywordMergeDialogResult | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        hint = QLabel(
            "请勾选要写入 glossary 的候选。默认不勾选疑似 Ren'Py 启动器/UI 噪音项，"
            "以及与 macro_setting 或现有 glossary 冲突的条目会以红色提示。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        paths = QLabel(
            f"候选文件：{candidates_path}\n术语表：{glossary_path}"
        )
        paths.setWordWrap(True)
        paths.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(paths)

        selection_row = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self._select_all)
        selection_row.addWidget(select_all_btn)
        select_none_btn = QPushButton("全不选")
        select_none_btn.clicked.connect(self._select_none)
        selection_row.addWidget(select_none_btn)
        invert_btn = QPushButton("反选")
        invert_btn.clicked.connect(self._invert_selection)
        selection_row.addWidget(invert_btn)
        selection_row.addStretch(1)
        self.overwrite_check = QCheckBox("覆盖已有 glossary 冲突项")
        selection_row.addWidget(self.overwrite_check)
        self.overwrite_check.toggled.connect(self._refresh_preview)
        layout.addLayout(selection_row)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "写入",
                "source",
                "suggested_target",
                "category",
                "confidence",
                "分区",
                "提示",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self._populate_table()
        layout.addWidget(self.table, 1)

        layout.addWidget(QLabel("预览摘要："))
        self.preview_view = QPlainTextEdit()
        self.preview_view.setReadOnly(True)
        self.preview_view.setMaximumHeight(120)
        layout.addWidget(self.preview_view)

        buttons = QDialogButtonBox()
        preview_btn = buttons.addButton("预览写入", QDialogButtonBox.ButtonRole.ActionRole)
        write_btn = buttons.addButton("写入 glossary", QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_btn = buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        if preview_btn is not None:
            preview_btn.clicked.connect(self._on_preview)
        if write_btn is not None:
            write_btn.clicked.connect(self._on_write)
        if cancel_btn is not None:
            cancel_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

        self.table.itemChanged.connect(self._on_item_changed)
        self._refresh_preview()

    @property
    def result(self) -> KeywordMergeDialogResult | None:
        return self._result

    def _populate_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._rows))
        warn_color = QBrush(QColor("#b45309"))
        for table_row, row in enumerate(self._rows):
            check_item = QTableWidgetItem()
            check_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable,
            )
            check_item.setCheckState(
                Qt.CheckState.Checked if row.default_checked else Qt.CheckState.Unchecked,
            )
            check_item.setData(Qt.ItemDataRole.UserRole, row.index)
            self.table.setItem(table_row, 0, check_item)

            source = row.action.source or str(row.candidate.get("source") or "").strip()
            target = row.action.target or str(row.candidate.get("suggested_target") or "").strip()
            category = str(row.candidate.get("category") or "other").strip()
            try:
                confidence = float(row.candidate.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            section = row.action.section or "-"
            warning_text = "；".join(row.warnings)

            for column, value in enumerate(
                (source, target, category, f"{confidence:.2f}", section, warning_text),
                start=1,
            ):
                item = QTableWidgetItem(value)
                if row.warnings:
                    item.setForeground(warn_color)
                self.table.setItem(table_row, column, item)
        self.table.blockSignals(False)

    def _selected_indices(self) -> set[int]:
        selected: set[int] = set()
        for table_row in range(self.table.rowCount()):
            item = self.table.item(table_row, 0)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                index = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(index, int):
                    selected.add(index)
        return selected

    def _refresh_preview(self) -> None:
        counts = merge_mod.preview_selected_merge_actions(
            self._rows,
            self._selected_indices(),
            overwrite=self.overwrite_check.isChecked(),
        )
        self.preview_view.setPlainText(
            format_merge_preview_text(
                counts,
                overwrite=self.overwrite_check.isChecked(),
            )
        )

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self._refresh_preview()

    def _select_all(self) -> None:
        self._set_all_checks(Qt.CheckState.Checked)

    def _select_none(self) -> None:
        self._set_all_checks(Qt.CheckState.Unchecked)

    def _invert_selection(self) -> None:
        for table_row in range(self.table.rowCount()):
            item = self.table.item(table_row, 0)
            if item is None:
                continue
            item.setCheckState(
                Qt.CheckState.Unchecked
                if item.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )

    def _set_all_checks(self, state: Qt.CheckState) -> None:
        self.table.blockSignals(True)
        for table_row in range(self.table.rowCount()):
            item = self.table.item(table_row, 0)
            if item is not None:
                item.setCheckState(state)
        self.table.blockSignals(False)
        self._refresh_preview()

    def _run_merge(self, *, dry_run: bool) -> merge_mod.MergeSummary:
        return merge_mod.merge_selected_candidates(
            self._candidates,
            self._selected_indices(),
            self._glossary_path,
            candidates_path=self._candidates_path,
            dry_run=dry_run,
            overwrite=self.overwrite_check.isChecked(),
            min_confidence=self._min_confidence,
        )

    def _on_preview(self) -> None:
        if not self._selected_indices():
            QMessageBox.information(self, "未选择候选", "请至少勾选一条候选再预览。")
            return
        summary = self._run_merge(dry_run=True)
        self.preview_view.setPlainText("\n".join(summary.preview_lines[:40]))
        self._result = KeywordMergeDialogResult(summary=summary, dry_run=True)

    def _on_write(self) -> None:
        if not self._selected_indices():
            QMessageBox.information(self, "未选择候选", "请至少勾选一条候选再写入。")
            return
        counts = merge_mod.preview_selected_merge_actions(
            self._rows,
            self._selected_indices(),
            overwrite=self.overwrite_check.isChecked(),
        )
        if counts.get("accept", 0) + counts.get("overwrite", 0) <= 0:
            QMessageBox.warning(
                self,
                "没有可写入项",
                "当前勾选项均会被跳过（重复、冲突或未启用覆盖）。请调整勾选或启用覆盖。",
            )
            return
        confirm = QMessageBox.question(
            self,
            "确认写入 glossary",
            (
                f"将把 {counts.get('accept', 0)} 条新增、"
                f"{counts.get('overwrite', 0)} 条覆盖写入：\n{self._glossary_path}\n\n"
                "写入前会自动备份 glossary。是否继续？"
            ),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        summary = self._run_merge(dry_run=False)
        self._result = KeywordMergeDialogResult(summary=summary, dry_run=False)
        self.accept()