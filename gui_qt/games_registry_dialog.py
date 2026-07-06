"""Modal dialog for browsing and switching workspace projects."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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
    delete_registry_project,
    discover_registry_projects,
    import_registry_from_games_md,
    prompt_render_games_md_after_refresh,
    render_registry_games_md,
    save_registry_dialog_preference,
    save_registry_project_fields,
)
from .games_registry_doctor_compare import (
    compare_registry_with_doctor_report,
    format_registry_compare_hint,
)
from .games_registry_worker import RegistryRefreshWorker
from .widget_helpers import NoWheelComboBox
from .games_registry_view import (
    REGISTRY_PREF_AUTO_DISCOVER,
    REGISTRY_SORT_NAME_ASC,
    REGISTRY_SORT_OPTIONS,
    REGISTRY_TABLE_COLUMNS,
    RegistryRow,
    count_undiscovered_projects,
    filter_and_sort_registry_rows,
    format_registry_status_message,
    load_registry_preferences,
    load_registry_rows,
    registry_engine_filter_options,
    registry_translation_filter_options,
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
        current_doctor_report: dict | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("games_registry_dialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("工作区项目总览")
        self.setModal(True)
        self.resize(1040, 760)

        self._workspace_root = workspace_root
        self._all_rows: list[RegistryRow] = []
        self._filtered_rows: list[RegistryRow] = []
        self._registry_summary = ""
        self._missing_message = ""
        self._selected_project_root = ""
        self._refresh_worker: RegistryRefreshWorker | None = None
        self._edit_loading = False
        self._preserved_project_id = ""
        self._current_game_root = current_game_root
        self._current_doctor_report = current_doctor_report

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
        self._import_md_btn.setToolTip(
            "把 GAMES.md 表格合并进 games_registry.json（按路径匹配）。"
            "JSON 仍是日常真源；导入后建议再点「同步 GAMES.md」。"
        )
        self._import_md_btn.clicked.connect(self._import_from_games_md)
        header.addWidget(self._import_md_btn)

        self._discover_btn = QPushButton("扫描新项目")
        self._discover_btn.setObjectName("secondary_btn")
        self._discover_btn.setToolTip("扫描工作区中的 Game_* 目录，并把未登记的项目加入总表")
        self._discover_btn.clicked.connect(self._discover_new_projects)
        header.addWidget(self._discover_btn)

        self._sync_md_btn = QPushButton("同步 GAMES.md")
        self._sync_md_btn.setObjectName("secondary_btn")
        self._sync_md_btn.setToolTip("用 games_registry.json 重新生成 GAMES.md 表格（覆盖表格区）")
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

        options_row = QHBoxLayout()
        self._auto_discover_checkbox = QCheckBox("打开时自动扫描新项目")
        self._auto_discover_checkbox.toggled.connect(self._on_auto_discover_toggled)
        options_row.addWidget(self._auto_discover_checkbox)
        options_row.addStretch(1)
        layout.addLayout(options_row)

        filter_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("搜索项目、路径、备注、状态…")
        self._search_edit.textChanged.connect(self._apply_filters)
        filter_row.addWidget(self._search_edit, 2)

        self._engine_filter_combo = NoWheelComboBox()
        for value, label in registry_engine_filter_options():
            self._engine_filter_combo.addItem(label, value)
        self._engine_filter_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self._engine_filter_combo)

        self._translation_filter_combo = NoWheelComboBox()
        for label in registry_translation_filter_options():
            self._translation_filter_combo.addItem(label, label)
        self._translation_filter_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self._translation_filter_combo)

        sort_label = QLabel("排序")
        sort_label.setObjectName("config_hint_label")
        filter_row.addWidget(sort_label)
        self._sort_combo = NoWheelComboBox()
        for value, label in REGISTRY_SORT_OPTIONS:
            self._sort_combo.addItem(label, value)
        self._sort_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self._sort_combo)
        layout.addLayout(filter_row)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setObjectName("config_hint_label")
        layout.addWidget(self._status_label)

        self._sync_hint_label = QLabel(
            "同步策略：日常以 games_registry.json 为准；「同步 GAMES.md」会覆盖 Markdown 表格。"
            "若手改了 GAMES.md，用「从 GAMES.md 导入」按路径合并回 JSON。"
        )
        self._sync_hint_label.setWordWrap(True)
        self._sync_hint_label.setObjectName("config_hint_label")
        layout.addWidget(self._sync_hint_label)

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

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("显示名称")
        edit_layout.addRow("项目名称", self._name_edit)

        self._layout_status_label = QLabel("—")
        edit_layout.addRow("目录状态", self._layout_status_label)

        self._doctor_mode_label = QLabel("—")
        edit_layout.addRow("Doctor 模式", self._doctor_mode_label)

        self._last_refresh_label = QLabel("—")
        edit_layout.addRow("最近刷新", self._last_refresh_label)

        self._doctor_check_layout_label = QLabel("—")
        edit_layout.addRow("环境检查 layout", self._doctor_check_layout_label)

        self._doctor_check_mode_label = QLabel("—")
        edit_layout.addRow("环境检查 mode", self._doctor_check_mode_label)

        self._registry_compare_label = QLabel("—")
        self._registry_compare_label.setWordWrap(True)
        edit_layout.addRow("总表对比", self._registry_compare_label)

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
        self._delete_project_btn = QPushButton("删除项目")
        self._delete_project_btn.setObjectName("kill_btn")
        self._delete_project_btn.clicked.connect(self._delete_selected_project)
        self._delete_project_btn.setEnabled(False)
        edit_actions.addWidget(self._delete_project_btn)
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

        prefs = load_registry_preferences(workspace_root=self._workspace_root)
        self._auto_discover_checkbox.setChecked(bool(prefs.get(REGISTRY_PREF_AUTO_DISCOVER, False)))
        self._reload_table_from_disk()
        self._maybe_auto_discover_on_open()

    def selected_project_root(self) -> str:
        return self._selected_project_root

    def _selected_sort_key(self) -> str:
        value = self._sort_combo.currentData()
        if isinstance(value, str) and value:
            return value
        return REGISTRY_SORT_NAME_ASC

    def _reload_table_from_disk(self) -> None:
        registry_path = resolve_registry_path(self._workspace_root)
        rows, summary = load_registry_rows(
            workspace_root=self._workspace_root,
            registry_path=registry_path,
        )
        self._all_rows = rows
        if rows:
            self._registry_summary = summary
            self._missing_message = ""
        else:
            self._registry_summary = ""
            self._missing_message = summary
        self._apply_filters()

    def _apply_filters(self) -> None:
        engine_filter = str(self._engine_filter_combo.currentData() or "all")
        translation_filter = str(self._translation_filter_combo.currentData() or "全部")
        self._filtered_rows = filter_and_sort_registry_rows(
            self._all_rows,
            search_text=self._search_edit.text(),
            engine_filter=engine_filter,
            translation_filter=translation_filter,
            sort_key=self._selected_sort_key(),
        )
        self._render_table_rows()

    def _render_table_rows(self) -> None:
        rows = self._filtered_rows
        self._table.setRowCount(len(rows))

        selected_row = -1
        preserved_id = self._preserved_project_id or (
            self._selected_row().project_id if self._selected_row() else ""
        )

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

        self._update_status_label(len(rows))
        self._table.clearSelection()
        if selected_row >= 0:
            self._table.selectRow(selected_row)
        self._on_selection_changed()

    def _update_status_label(self, visible_count: int) -> None:
        if self._progress_bar.isVisible():
            return

        total_count = len(self._all_rows)
        if self._missing_message:
            status_message = self._missing_message
        elif total_count <= 0:
            status_message = format_registry_status_message(0, "")
        elif visible_count < total_count:
            status_message = (
                f"工作区项目总览：显示 {visible_count} / {total_count} 个项目。"
            )
            if self._registry_summary:
                status_message = f"{status_message} {self._registry_summary}"
        else:
            status_message = format_registry_status_message(total_count, self._registry_summary)

        undiscovered = count_undiscovered_projects(
            workspace_root=self._workspace_root,
            registry_path=resolve_registry_path(self._workspace_root),
        )
        if undiscovered > 0:
            status_message = (
                f"{status_message} 发现 {undiscovered} 个未登记的 Game_* 目录，"
                "可点击「扫描新项目」。"
            )
        self._status_label.setText(status_message)

    def _selected_row(self) -> RegistryRow | None:
        selected = self._table.selectionModel().selectedRows()
        if not selected:
            return None
        row_index = selected[0].row()
        if row_index < 0 or row_index >= len(self._filtered_rows):
            return None
        return self._filtered_rows[row_index]

    def _is_refresh_running(self) -> bool:
        return self._refresh_worker is not None and self._refresh_worker.isRunning()

    def _populate_edit_panel(self, row: RegistryRow | None) -> None:
        self._edit_loading = True
        try:
            if row is None:
                self._name_edit.clear()
                self._layout_status_label.setText("—")
                self._doctor_mode_label.setText("—")
                self._last_refresh_label.setText("—")
                self._doctor_check_layout_label.setText("—")
                self._doctor_check_mode_label.setText("—")
                self._registry_compare_label.setText("—")
                self._play_status_combo.setCurrentText("待确认")
                self._translation_status_combo.setCurrentText("待确认")
                self._notes_edit.clear()
                return

            self._name_edit.setText(row.name)
            self._layout_status_label.setText(row.layout_status or "—")
            self._doctor_mode_label.setText(row.doctor_mode or "—")
            self._last_refresh_label.setText(row.last_refresh_at or "—")

            compare = None
            show_doctor = (
                self._current_doctor_report is not None
                and row_matches_game_root(row, self._current_game_root)
            )
            if show_doctor:
                doctor_layout = str(
                    self._current_doctor_report.get("layout_status") or ""
                ).strip()
                doctor_mode = str(self._current_doctor_report.get("mode") or "").strip()
                self._doctor_check_layout_label.setText(doctor_layout or "—")
                self._doctor_check_mode_label.setText(doctor_mode or "—")
                compare = compare_registry_with_doctor_report(
                    self._workspace_root,
                    game_root=self._current_game_root,
                    report=self._current_doctor_report,
                )
            else:
                self._doctor_check_layout_label.setText("—")
                self._doctor_check_mode_label.setText("—")

            self._registry_compare_label.setText(
                format_registry_compare_hint(
                    compare,
                    for_registry_dialog=True,
                )
                if show_doctor
                else (
                    "选中当前工作台项目并运行环境检查后，可在此对比总表记录。"
                    if row_matches_game_root(row, self._current_game_root)
                    else "切换到该项目后运行环境检查，可对比总表 layout / mode。"
                )
            )

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
            self._delete_project_btn.setEnabled(can_use_row)

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
            self._delete_project_btn,
            self._name_edit,
            self._play_status_combo,
            self._translation_status_combo,
            self._notes_edit,
            self._search_edit,
            self._engine_filter_combo,
            self._translation_filter_combo,
            self._sort_combo,
            self._auto_discover_checkbox,
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

    def _on_auto_discover_toggled(self, checked: bool) -> None:
        if self._is_refresh_running():
            return
        save_registry_dialog_preference(
            self._workspace_root,
            key=REGISTRY_PREF_AUTO_DISCOVER,
            value=checked,
        )

    def _maybe_auto_discover_on_open(self) -> None:
        if not self._auto_discover_checkbox.isChecked() or self._is_refresh_running():
            return
        if count_undiscovered_projects(workspace_root=self._workspace_root) <= 0:
            return
        result = discover_registry_projects(
            self._workspace_root,
            refresh_new=True,
            mode=REFRESH_MODE_LITE,
        )
        self._handle_action_result(result, title="自动扫描失败")

    def _import_from_games_md(self) -> None:
        if self._is_refresh_running():
            return
        registry_path = resolve_registry_path(self._workspace_root)
        merge = registry_path.is_file()
        if merge:
            reply = QMessageBox.question(
                self,
                "从 GAMES.md 导入",
                "已存在 games_registry.json。\n\n"
                "• 选择「是」：按路径合并（GAMES.md 中的列覆盖同路径项目；"
                "未出现在 MD 中的 JSON 项目保留）\n"
                "• 选择「否」：用 GAMES.md 完全替换总表\n"
                "• 选择「取消」：放弃\n\n"
                "合并后如需让 Markdown 与 JSON 一致，请再点「同步 GAMES.md」。",
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
            mode=REFRESH_MODE_LITE,
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
            name=self._name_edit.text(),
            play_status=self._play_status_combo.currentText(),
            translation_status=self._translation_status_combo.currentText(),
            notes=self._notes_edit.toPlainText(),
        )
        self._preserved_project_id = row.project_id
        self._handle_action_result(result, title="保存失败")

    def _delete_selected_project(self) -> None:
        if self._is_refresh_running():
            return
        row = self._selected_row()
        if row is None or not row.project_id:
            return
        reply = QMessageBox.warning(
            self,
            "删除项目",
            f"确定从总表移除「{row.name}」？\n\n"
            "这不会删除磁盘上的 Game_* 目录，只是不再在 registry / GAMES.md 中显示。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        result = delete_registry_project(self._workspace_root, project_id=row.project_id)
        self._preserved_project_id = ""
        self._handle_action_result(result, title="删除失败")

    def _handle_action_result(self, result: RegistryActionResult, *, title: str) -> None:
        if result.ok:
            selected = self._selected_row()
            if selected is not None:
                self._preserved_project_id = selected.project_id
        self._reload_table_from_disk()
        if result.message:
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
        self._preserved_project_id = row.project_id
        self._run_refresh(project_id=row.project_id, mode=self._selected_refresh_mode())

    def _refresh_all_projects(self) -> None:
        mode = self._selected_refresh_mode()
        if mode == REFRESH_MODE_DEEP:
            reply = QMessageBox.question(
                self,
                "深度刷新全部",
                f"将对全部 {len(self._all_rows)} 个项目运行完整 doctor，可能需要数分钟。\n确定继续？",
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

        self._reload_table_from_disk()
        message = result.message
        if result.cancelled:
            self._status_label.setText(message)
            return
        if not result.ok:
            QMessageBox.warning(self, "刷新失败", message)
            self._status_label.setText(message)
            return

        render_result = prompt_render_games_md_after_refresh(
            self,
            self._workspace_root,
            message,
        )
        if render_result.message:
            message = f"{message} {render_result.message}".strip()
        self._status_label.setText(message)

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
        if row_index < 0 or row_index >= len(self._filtered_rows):
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