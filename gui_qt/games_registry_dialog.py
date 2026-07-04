"""Modal dialog for browsing and switching workspace projects."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .games_registry_view import (
    REGISTRY_TABLE_COLUMNS,
    RegistryRow,
    format_registry_status_message,
    load_registry_rows,
    resolve_registry_path,
    row_matches_game_root,
)


class GamesRegistryDialog(QDialog):
    """Browse all workspace projects and switch the active game root."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        workspace_root: Path,
        current_game_root: Path | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("games_registry_dialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("工作区项目总览")
        self.setModal(True)
        self.resize(920, 560)

        self._workspace_root = workspace_root
        self._rows: list[RegistryRow] = []
        self._selected_project_root = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("工作区项目总览")
        title.setObjectName("diagnostics_section_label")
        header.addWidget(title)
        header.addStretch(1)
        reload_btn = QPushButton("重新加载")
        reload_btn.setObjectName("secondary_btn")
        reload_btn.clicked.connect(self._reload_table)
        header.addWidget(reload_btn)
        self._switch_btn = QPushButton("切换到此项目")
        self._switch_btn.setObjectName("secondary_btn")
        self._switch_btn.clicked.connect(self._switch_to_selected)
        self._switch_btn.setEnabled(False)
        header.addWidget(self._switch_btn)
        layout.addLayout(header)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setObjectName("config_hint_label")
        layout.addWidget(self._status_label)

        self._table = QTableWidget(0, len(REGISTRY_TABLE_COLUMNS))
        self._table.setObjectName("games_registry_table")
        self._table.setHorizontalHeaderLabels(list(REGISTRY_TABLE_COLUMNS))
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setVisible(False)
        table_header = self._table.horizontalHeader()
        table_header.setStretchLastSection(True)
        table_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.cellDoubleClicked.connect(self._on_row_activated)
        layout.addWidget(self._table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setText("关闭")
        layout.addWidget(buttons)

        self._current_game_root = current_game_root
        self._reload_table()

    def selected_project_root(self) -> str:
        return self._selected_project_root

    def _reload_table(self) -> None:
        registry_path = resolve_registry_path(self._workspace_root)
        rows, summary = load_registry_rows(
            workspace_root=self._workspace_root,
            registry_path=registry_path,
        )
        self._rows = rows
        self._table.setRowCount(len(rows))

        selected_row = -1
        for row_index, row in enumerate(rows):
            values = (
                row.name,
                row.path,
                row.version,
                row.play_status,
                row.translation_status,
            )
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if row.tooltip:
                    item.setToolTip(row.tooltip)
                if not row.in_renpy_pipeline:
                    item.setForeground(QBrush(QColor("#64748b")))
                self._table.setItem(row_index, column_index, item)

            if selected_row < 0 and row_matches_game_root(row, self._current_game_root):
                selected_row = row_index

        if rows:
            status_message = format_registry_status_message(len(rows), summary)
        else:
            status_message = summary or format_registry_status_message(0, "")
        self._status_label.setText(status_message)

        self._table.clearSelection()
        if selected_row >= 0:
            self._table.selectRow(selected_row)
        self._on_selection_changed()

    def _selected_row(self) -> RegistryRow | None:
        selected = self._table.selectionModel().selectedRows()
        if not selected:
            return None
        row_index = selected[0].row()
        if row_index < 0 or row_index >= len(self._rows):
            return None
        return self._rows[row_index]

    def _on_selection_changed(self) -> None:
        row = self._selected_row()
        self._switch_btn.setEnabled(row is not None and bool(row.work_dir))

    def _on_row_activated(self, row_index: int, _column: int) -> None:
        if row_index < 0 or row_index >= len(self._rows):
            return
        self._table.selectRow(row_index)
        self._switch_to_selected()

    def _switch_to_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if not row.work_dir:
            return
        self._selected_project_root = str(self._workspace_root / row.path)
        self.accept()