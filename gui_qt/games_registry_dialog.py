"""Modal dialog for browsing and switching workspace projects."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from games_registry import PLAY_STATUSES, REFRESH_MODE_DEEP, REFRESH_MODE_LITE, TRANSLATION_STATUSES

from .games_registry_actions import (
    RegistryActionResult,
    discover_registry_projects,
    import_registry_from_games_md,
    render_registry_games_md,
    save_registry_project_fields,
)
from .games_registry_worker import RegistryRefreshWorker
from .widget_helpers import NoWheelComboBox
from .games_registry_view import (
    REGISTRY_TABLE_COLUMNS,
    RegistryRow,
    count_undiscovered_projects,
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
        self.resize(980, 680)

        self._workspace_root = workspace_root
        self._rows: list[RegistryRow] = []
        self._selected_project_root = ""
        self._refresh_worker: RegistryRefreshWorker | None = None
        self._edit_loading = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("工作区项目总览")
        title.setObjectName("diagnostics_section_label")
        header.addWidget(title)
        header.addStretch(1)

        self._import_md_btn = QPushButton("从 GAMES.md 导入")
        self._import_md_btn.setObjectName("secondary_btn")
        self._import_md_btn.clicked.connect(self._import_from_games_md)
        header.addWidget(self._import_md_btn)

        self._discover_btn = QPushButton("扫描新项目")
        self._discover_btn.setObjectName("secondary_btn")
        self._discover_btn.setToolTip("扫描工作区中的 Game_* 目录，并把未登记的项目加入总表")
        self._discover_btn.clicked.connect(self._discover_new_projects)
        header.addWidget(self._discover_btn)

        self._sync_md_btn = QPushButton("同步 GAMES.md")
        self._sync_md_btn.setObjectName("secondary_btn")
        self._sync_md_btn.clicked.connect(self._sync_games_md)
        header.addWidget(self._sync_md_btn)

        self._refresh_current_btn = QPushButton("刷新当前")
        self._refresh_current_btn.setObjectName("secondary_btn")
        self._refresh_current_btn.clicked.connect(self._refresh_current_project)
        header.addWidget(self._refresh_current_btn)

        self._refresh_all_btn = QPushButton("刷新全部")
        self._refresh_all_btn.setObjectName("secondary_btn")
        self._refresh_all_btn.clicked.connect(self._refresh_all_projects)
        header.addWidget(self._refresh_all_btn)

        mode_label = QLabel("扫描模式")
        mode_label.setObjectName("config_hint_label")
        header.addWidget(mode_label)

        self._refresh_mode_combo = NoWheelComboBox()
        self._refresh_mode_combo.setObjectName("games_registry_refresh_mode_combo")
        self._refresh_mode_combo.addItem("快速", REFRESH_MODE_LITE)
        self._refresh_mode_combo.addItem("深度", REFRESH_MODE_DEEP)
        self._refresh_mode_combo.setToolTip("快速：只扫磁盘与翻译文件；深度：额外运行 doctor（较慢）")
        header.addWidget(self._refresh_mode_combo)

        self._stop_refresh_btn = QPushButton("停止")
        self._stop_refresh_btn.setObjectName("kill_btn")
        self._stop_refresh_btn.setEnabled(False)
        self._stop_refresh_btn.clicked.connect(self._on_stop_refresh)
        header.addWidget(self._stop_refresh_btn)

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

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("games_registry_progress")
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("准备刷新…")
        layout.addWidget(self._progress_bar)

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
        for column_index in range(2, len(REGISTRY_TABLE_COLUMNS)):
            table_header.setSectionResizeMode(column_index, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.cellDoubleClicked.connect(self._on_row_activated)
        layout.addWidget(self._table, 1)

        self._edit_group = QGroupBox("项目详情")
        self._edit_group.setObjectName("games_registry_edit_group")
        edit_layout = QFormLayout(self._edit_group)
        edit_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._play_status_combo = NoWheelComboBox()
        self._play_status_combo.addItems(sorted(PLAY_STATUSES))
        edit_layout.addRow("游玩状态", self._play_status_combo)

        self._translation_status_combo = NoWheelComboBox()
        self._translation_status_combo.addItems(sorted(TRANSLATION_STATUSES))
        edit_layout.addRow("翻译状态", self._translation_status_combo)

        self._notes_edit = QPlainTextEdit()
        self._notes_edit.setPlaceholderText("备注 / 下一步")
        self._notes_edit.setFixedHeight(72)
        edit_layout.addRow("备注", self._notes_edit)

        edit_actions = QHBoxLayout()
        edit_actions.addStretch(1)
        self._save_fields_btn = QPushButton("保存修改")
        self._save_fields_btn.setObjectName("secondary_btn")
        self._save_fields_btn.clicked.connect(self._save_selected_project_fields)
        self._save_fields_btn.setEnabled(False)
        edit_actions.addWidget(self._save_fields_btn)
        edit_layout.addRow("", edit_actions)
        layout.addWidget(self._edit_group)

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
        preserved_selection = self._selected_row()
        preserved_id = preserved_selection.project_id if preserved_selection else ""

        for row_index, row in enumerate(rows):
            values = (
                row.name,
                row.path,
                row.version,
                row.layout_status,
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

            if preserved_id and row.project_id == preserved_id:
                selected_row = row_index
            elif selected_row < 0 and row_matches_game_root(row, self._current_game_root):
                selected_row = row_index

        if rows:
            status_message = format_registry_status_message(len(rows), summary)
        else:
            status_message = summary or format_registry_status_message(0, "")
        undiscovered = count_undiscovered_projects(
            workspace_root=self._workspace_root,
            registry_path=registry_path,
        )
        if undiscovered > 0:
            status_message = f"{status_message} 发现 {undiscovered} 个未登记的 Game_* 目录，可点击「扫描新项目」。"
        if not self._progress_bar.isVisible():
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

    def _is_refresh_running(self) -> bool:
        return self._refresh_worker is not None and self._refresh_worker.isRunning()

    def _populate_edit_panel(self, row: RegistryRow | None) -> None:
        self._edit_loading = True
        try:
            if row is None:
                self._play_status_combo.setCurrentText("待确认")
                self._translation_status_combo.setCurrentText("待确认")
                self._notes_edit.clear()
                return

            play_status = row.play_status if row.play_status in PLAY_STATUSES else "待确认"
            translation_status = (
                row.translation_status
                if row.translation_status in TRANSLATION_STATUSES
                else "待确认"
            )
            self._play_status_combo.setCurrentText(play_status)
            self._translation_status_combo.setCurrentText(translation_status)
            self._notes_edit.setPlainText(row.notes)
        finally:
            self._edit_loading = False

    def _on_selection_changed(self) -> None:
        row = self._selected_row()
        can_use_row = row is not None and bool(row.project_id)
        self._populate_edit_panel(row)
        if not self._is_refresh_running():
            self._switch_btn.setEnabled(row is not None and bool(row.work_dir))
            self._refresh_current_btn.setEnabled(can_use_row)
            self._save_fields_btn.setEnabled(can_use_row)

    def _selected_refresh_mode(self) -> str:
        mode = self._refresh_mode_combo.currentData()
        if mode in {REFRESH_MODE_LITE, REFRESH_MODE_DEEP}:
            return str(mode)
        return REFRESH_MODE_LITE

    def _set_maintenance_busy(self, busy: bool) -> None:
        for widget in (
            self._import_md_btn,
            self._discover_btn,
            self._sync_md_btn,
            self._save_fields_btn,
            self._play_status_combo,
            self._translation_status_combo,
            self._notes_edit,
        ):
            widget.setEnabled(not busy)

    def _set_refresh_busy(self, busy: bool) -> None:
        for widget in (
            self._refresh_current_btn,
            self._refresh_all_btn,
            self._refresh_mode_combo,
            self._switch_btn,
        ):
            widget.setEnabled(not busy)
        self._set_maintenance_busy(busy)
        self._stop_refresh_btn.setEnabled(busy)
        self._progress_bar.setVisible(busy)
        if busy:
            self._progress_bar.setValue(0)
        else:
            self._progress_bar.setFormat("准备刷新…")
            self._on_selection_changed()

    def _import_from_games_md(self) -> None:
        if self._is_refresh_running():
            return
        registry_path = resolve_registry_path(self._workspace_root)
        merge = registry_path.is_file()
        if merge:
            reply = QMessageBox.question(
                self,
                "从 GAMES.md 导入",
                "已存在 games_registry.json。\n选择「是」按路径合并更新；选择「否」将覆盖现有总表。",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            merge = reply == QMessageBox.StandardButton.Yes

        result = import_registry_from_games_md(self._workspace_root, merge=merge)
        self._handle_action_result(result, title="导入失败")

    def _discover_new_projects(self) -> None:
        if self._is_refresh_running():
            return
        result = discover_registry_projects(
            self._workspace_root,
            refresh_new=True,
            mode=self._selected_refresh_mode(),
        )
        self._handle_action_result(result, title="扫描失败")

    def _sync_games_md(self) -> None:
        if self._is_refresh_running():
            return
        result = render_registry_games_md(self._workspace_root)
        self._handle_action_result(result, title="同步失败")

    def _save_selected_project_fields(self) -> None:
        if self._is_refresh_running():
            return
        row = self._selected_row()
        if row is None or not row.project_id:
            return
        result = save_registry_project_fields(
            self._workspace_root,
            project_id=row.project_id,
            play_status=self._play_status_combo.currentText(),
            translation_status=self._translation_status_combo.currentText(),
            notes=self._notes_edit.toPlainText(),
        )
        self._handle_action_result(result, title="保存失败")

    def _handle_action_result(self, result: RegistryActionResult, *, title: str) -> None:
        self._reload_table()
        self._status_label.setText(result.message)
        if not result.ok:
            QMessageBox.warning(self, title, result.message)

    def _on_refresh_progress(self, current: int, total: int, name: str) -> None:
        self._progress_bar.setMaximum(max(total, 1))
        self._progress_bar.setValue(current)
        self._progress_bar.setFormat(f"{current}/{total} — {name}")
        self._status_label.setText(f"正在刷新：{name}（{current}/{total}）")

    def _on_stop_refresh(self) -> None:
        if self._refresh_worker is None or not self._refresh_worker.isRunning():
            return
        self._refresh_worker.request_stop()
        self._stop_refresh_btn.setEnabled(False)
        self._status_label.setText("正在停止…（等待当前项目完成）")

    def _refresh_current_project(self) -> None:
        row = self._selected_row()
        if row is None or not row.project_id:
            QMessageBox.information(self, "请选择项目", "请先在表格里选中一个项目。")
            return
        self._run_refresh(project_id=row.project_id, mode=self._selected_refresh_mode())

    def _refresh_all_projects(self) -> None:
        mode = self._selected_refresh_mode()
        if mode == REFRESH_MODE_DEEP:
            reply = QMessageBox.question(
                self,
                "深度刷新全部",
                f"将对全部 {len(self._rows)} 个项目运行完整 doctor，可能需要数分钟。\n确定继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._run_refresh(refresh_everything=True, mode=mode)

    def _run_refresh(
        self,
        *,
        project_id: str | None = None,
        refresh_everything: bool = False,
        mode: str = REFRESH_MODE_LITE,
    ) -> None:
        if self._is_refresh_running():
            return

        self._refresh_worker = RegistryRefreshWorker(
            workspace_root=self._workspace_root,
            project_id=project_id,
            refresh_everything=refresh_everything,
            mode=mode,
            parent=self,
        )
        self._refresh_worker.progress.connect(self._on_refresh_progress)
        self._refresh_worker.completed.connect(self._on_refresh_completed)
        self._set_refresh_busy(True)
        self._refresh_worker.start()

    def _on_refresh_completed(self, result: RegistryActionResult) -> None:
        worker = self.sender()
        if worker is self._refresh_worker:
            self._refresh_worker = None
        self._set_refresh_busy(False)

        self._reload_table()
        self._status_label.setText(result.message)

        if result.cancelled:
            return
        if not result.ok:
            QMessageBox.warning(self, "刷新失败", result.message)

    def closeEvent(self, event) -> None:
        if self._is_refresh_running() and self._refresh_worker is not None:
            self._refresh_worker.request_stop()
            self._status_label.setText("正在停止…请稍候再关闭")
            if not self._refresh_worker.wait(5000):
                event.ignore()
                return
        super().closeEvent(event)

    def _on_row_activated(self, row_index: int, _column: int) -> None:
        if self._is_refresh_running():
            return
        if row_index < 0 or row_index >= len(self._rows):
            return
        self._table.selectRow(row_index)
        self._switch_to_selected()

    def _switch_to_selected(self) -> None:
        if self._is_refresh_running():
            return
        row = self._selected_row()
        if row is None:
            return
        if not row.work_dir:
            return
        self._selected_project_root = str(self._workspace_root / row.path)
        self.accept()