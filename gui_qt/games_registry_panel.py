"""Embeddable panel for browsing and maintaining workspace projects."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionComboBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from games_registry import (
    PLAY_STATUSES,
    REFRESH_MODE_DEEP,
    REFRESH_MODE_LITE,
    TRANSLATION_STATUSES,
    format_doctor_mode_label,
    format_layout_status_label,
    normalize_play_status,
    normalize_translation_status,
)

from .empty_state import EmptyStateWidget
from .game_ingest_dialog import GameIngestDialog
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
from .games_registry_worker import RegistryIngestWorker, RegistryRefreshWorker
from .path_utils import canonical_abs_path
from .responsive_layout import FlowButtonBar
from .widget_helpers import (
    NoWheelComboBox,
    message_box_information,
    message_box_question,
    message_box_warning,
)
from .games_registry_table import (
    REGISTRY_PREF_TABLE_COLUMN_WIDTHS,
    REGISTRY_PREF_TABLE_COLUMN_WIDTHS_LEGACY,
    REGISTRY_TABLE_COLUMN_DEFS,
    REGISTRY_TABLE_PATH_COLUMN,
    clamp_width,
    clamp_width_for_fit,
    column_at,
    column_headers,
    default_width_map,
    interactive_column_indexes,
    migrate_stored_widths,
    min_width_for_column,
    row_cell_values,
    widths_for_persist,
)
from .games_registry_view import (
    REGISTRY_PREF_AUTO_DISCOVER,
    REGISTRY_SORT_NAME_ASC,
    REGISTRY_SORT_OPTIONS,
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

SwitchProjectHandler = Callable[[str], bool]
DoctorReportProvider = Callable[[], dict | None]
WorkspaceChangedHandler = Callable[[Path], None]


def _combo_natural_width(combo: NoWheelComboBox) -> int:
    """Width needed to show the longest item under the current style/font."""
    # Clear prior fixed constraints so sizeHint is honest.
    combo.setMinimumWidth(0)
    combo.setMaximumWidth(16777215)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    metrics = combo.fontMetrics()
    longest = ""
    longest_px = 0
    for index in range(combo.count()):
        text = combo.itemText(index)
        advance = metrics.horizontalAdvance(text)
        if advance > longest_px:
            longest_px = advance
            longest = text
    opt = QStyleOptionComboBox()
    combo.initStyleOption(opt)
    opt.currentText = longest
    styled = combo.style().sizeFromContents(
        QStyle.ContentsType.CT_ComboBox,
        opt,
        QSize(longest_px, metrics.height()),
        combo,
    )
    return max(combo.sizeHint().width(), styled.width())


def _uniform_combo_width(*combos: NoWheelComboBox, minimum: int = 0, pad: int = 8) -> None:
    """Lock a group of combos to one content-fitting width (not full form row).

    Uses style/font sizeFromContents so CJK (e.g. YaHei UI) is not clipped.
    Fixed width keeps the pair equal after first show.
    """
    if not combos:
        return
    width = max(minimum, max(_combo_natural_width(combo) for combo in combos) + pad)
    for combo in combos:
        # Stop per-combo AdjustToContentsOnFirstShow from diverging after paint.
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setFixedWidth(width)


def _set_status_combo_value(combo: NoWheelComboBox, value: str) -> None:
    """Select *value* in combo; add item when registry has a free-form label.

    The table shows the raw registry string (e.g. ``已完成（6.7 增量）``). The
    detail combo used to only allow the closed status set and fell back to
    「待确认」 for annotated values, so table and detail disagreed.
    """
    text = (value or "").strip() or "待确认"
    existing = {combo.itemText(i) for i in range(combo.count())}
    if text not in existing:
        combo.addItem(text)
    combo.setCurrentText(text)


class GamesRegistryPanel(QWidget):
    """Browse workspace projects, maintain registry, and request game_root switches."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        workspace_root: Path | None = None,
        current_game_root: Path | None = None,
        get_doctor_report: DoctorReportProvider | None = None,
        on_switch_project: SwitchProjectHandler | None = None,
        on_workspace_changed: WorkspaceChangedHandler | None = None,
        auto_discover_on_show: bool = True,
    ):
        super().__init__(parent)
        self.setObjectName("games_registry_panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._workspace_root: Path | None = (
            Path(workspace_root) if workspace_root is not None else None
        )
        self._all_rows: list[RegistryRow] = []
        self._filtered_rows: list[RegistryRow] = []
        self._registry_summary = ""
        self._missing_message = ""
        self._selected_project_root = ""
        self._refresh_worker: RegistryRefreshWorker | None = None
        self._ingest_worker: RegistryIngestWorker | None = None
        self._edit_loading = False
        self._preserved_project_id = ""
        self._current_game_root = current_game_root
        self._get_doctor_report = get_doctor_report
        self._on_switch_project = on_switch_project
        self._on_workspace_changed = on_workspace_changed
        self._auto_discover_on_show = auto_discover_on_show
        self._section_visible = False
        self._registry_disk_signature: tuple[str, int] | None = None
        # When the main workbench has a job running, keep browse/filter usable and
        # only gate switch / mutate actions (issue: whole-panel disable was too coarse).
        self._host_task_running = False
        # Explicit UI busy flag: set before the worker starts and cleared on completion.
        self._refresh_ui_busy = False
        # Interactive column widths by id (path is last flex section, not stored).
        self._table_column_widths: dict[str, int] = default_width_map()
        self._applying_table_columns = False
        self._column_width_save_timer = QTimer(self)
        self._column_width_save_timer.setSingleShot(True)
        self._column_width_save_timer.setInterval(250)
        self._column_width_save_timer.timeout.connect(self._persist_table_column_widths)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Title + primary switch stay on one short row; tools wrap below so the
        # panel no longer forces ~1250px min width (settings viewport is ~770–1100).
        title_row = QHBoxLayout()
        title = QLabel("工作区项目")
        title.setObjectName("diagnostics_section_label")
        title_row.addWidget(title)
        title_row.addStretch(1)
        self._choose_workspace_btn = QPushButton("选择工作区…")
        self._choose_workspace_btn.setObjectName("secondary_btn")
        self._choose_workspace_btn.setToolTip(
            "指定存放 Game_* 与 games_registry.json 的工作区根目录。"
            "不会默认使用工具安装目录的上一级。"
        )
        self._choose_workspace_btn.clicked.connect(self._choose_workspace)
        title_row.addWidget(self._choose_workspace_btn)
        self._switch_btn = QPushButton("切换到此项目")
        self._switch_btn.setObjectName("secondary_btn")
        self._switch_btn.clicked.connect(self._switch_to_selected)
        self._switch_btn.setEnabled(False)
        title_row.addWidget(self._switch_btn)
        layout.addLayout(title_row)

        workspace_row = QHBoxLayout()
        workspace_caption = QLabel("工作区")
        workspace_caption.setObjectName("config_hint_label")
        workspace_row.addWidget(workspace_caption)
        self._workspace_path_edit = QLineEdit()
        self._workspace_path_edit.setReadOnly(True)
        self._workspace_path_edit.setObjectName("games_registry_workspace_path")
        self._workspace_path_edit.setPlaceholderText("尚未指定工作区")
        workspace_row.addWidget(self._workspace_path_edit, 1)
        layout.addLayout(workspace_row)

        self._workspace_empty_state = EmptyStateWidget(
            "",
            "尚未指定工作区",
            "工作区默认未设置，也不会使用工具目录的上一级。"
            "请先选择存放 Game_* 项目与 games_registry.json 的文件夹，"
            "然后再扫描、导入或切换项目。",
            action_text="选择工作区…",
            parent=self,
        )
        self._workspace_empty_state.action_clicked.connect(self._choose_workspace)
        layout.addWidget(self._workspace_empty_state, 1)

        self._workspace_body = QWidget()
        self._workspace_body.setObjectName("games_registry_workspace_body")
        body_layout = QVBoxLayout(self._workspace_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)

        self._toolbar = FlowButtonBar(spacing=8, row_spacing=8)
        self._toolbar.setObjectName("games_registry_toolbar")
        self._maintenance_toolbar = FlowButtonBar(spacing=8, row_spacing=8)
        self._maintenance_toolbar.setObjectName("games_registry_maintenance_toolbar")
        self._registry_toolbar = FlowButtonBar(spacing=8, row_spacing=8)
        self._registry_toolbar.setObjectName("games_registry_file_toolbar")
        self._toolbars = (self._toolbar, self._maintenance_toolbar, self._registry_toolbar)

        self._import_md_btn = QPushButton("从 GAMES.md 导入")
        self._import_md_btn.setObjectName("secondary_btn")
        self._import_md_btn.setToolTip(
            "把 GAMES.md 表格合并进 games_registry.json（按路径匹配）。"
            "JSON 仍是日常真源；导入后建议再点「同步 GAMES.md」。"
        )
        self._import_md_btn.clicked.connect(self._import_from_games_md)
        self._registry_toolbar.add_widget(self._import_md_btn, min_width=100)

        self._discover_btn = QPushButton("扫描新项目")
        self._discover_btn.setObjectName("secondary_btn")
        self._discover_btn.setToolTip("扫描工作区中的 Game_* 目录，并把未登记的项目加入总表")
        self._discover_btn.clicked.connect(self._discover_new_projects)
        self._maintenance_toolbar.add_widget(self._discover_btn, min_width=88)

        self._ingest_btn = QPushButton("导入游戏…")
        self._ingest_btn.setObjectName("secondary_btn")
        self._ingest_btn.setToolTip(
            "从游戏目录或 zip 复制整理为 Game_*/original/work/build 并加入总表；"
            "自动预填游戏名称，可改，并实时预览最终 Game_* 目录。不移动源文件，不自动准备 work。"
        )
        self._ingest_btn.clicked.connect(self._ingest_game)
        self._maintenance_toolbar.add_widget(self._ingest_btn, min_width=88)

        self._sync_md_btn = QPushButton("同步 GAMES.md")
        self._sync_md_btn.setObjectName("secondary_btn")
        self._sync_md_btn.setToolTip("用 games_registry.json 重新生成 GAMES.md 表格（覆盖表格区）")
        self._sync_md_btn.clicked.connect(self._sync_games_md)
        self._registry_toolbar.add_widget(self._sync_md_btn, min_width=100)

        self._refresh_current_btn = QPushButton("刷新当前")
        self._refresh_current_btn.setObjectName("secondary_btn")
        self._refresh_current_btn.clicked.connect(self._refresh_current_project)
        self._toolbar.add_widget(self._refresh_current_btn, min_width=80)

        self._refresh_all_btn = QPushButton("刷新全部")
        self._refresh_all_btn.setObjectName("secondary_btn")
        self._refresh_all_btn.clicked.connect(self._refresh_all_projects)
        self._toolbar.add_widget(self._refresh_all_btn, min_width=80)

        mode_host = QWidget()
        mode_host.setObjectName("games_registry_mode_host")
        mode_row = QHBoxLayout(mode_host)
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(6)
        mode_label = QLabel("扫描模式")
        mode_label.setObjectName("config_hint_label")
        mode_row.addWidget(mode_label)
        self._refresh_mode_combo = NoWheelComboBox()
        self._refresh_mode_combo.setObjectName("games_registry_refresh_mode_combo")
        self._refresh_mode_combo.addItem("快速", REFRESH_MODE_LITE)
        self._refresh_mode_combo.addItem("深度", REFRESH_MODE_DEEP)
        self._refresh_mode_combo.setToolTip("快速：只扫磁盘与翻译文件；深度：额外运行 doctor（较慢）")
        self._refresh_mode_combo.setMinimumWidth(72)
        mode_row.addWidget(self._refresh_mode_combo)
        self._toolbar.add_widget(mode_host, min_width=None)

        self._stop_refresh_btn = QPushButton("停止")
        self._stop_refresh_btn.setObjectName("kill_btn")
        self._stop_refresh_btn.setEnabled(False)
        self._stop_refresh_btn.clicked.connect(self._on_stop_refresh)
        self._toolbar.add_widget(self._stop_refresh_btn, min_width=64)

        self._auto_discover_checkbox = QCheckBox("打开分区时自动扫描新项目")
        self._auto_discover_checkbox.setToolTip(
            "默认关闭。勾选后，进入本分区时会扫描工作区内未登记的 Game_* 并写入总表。"
            "仅扫描已选工作区，不会访问工作区以外的路径。"
        )
        self._auto_discover_checkbox.toggled.connect(self._on_auto_discover_toggled)
        self._maintenance_toolbar.add_widget(self._auto_discover_checkbox, min_width=None)

        for toolbar in self._toolbars:
            toolbar.finish_setup()

        for section_text, section_toolbar in (
            ("项目刷新", self._toolbar),
            ("项目发现", self._maintenance_toolbar),
            ("总表维护", self._registry_toolbar),
        ):
            section_label = QLabel(section_text)
            section_label.setObjectName("settings_inline_section_label")
            body_layout.addWidget(section_label)
            body_layout.addWidget(section_toolbar)

        filter_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("搜索项目、路径、备注、状态…")
        self._search_edit.textChanged.connect(self._apply_filters)
        filter_row.addWidget(self._search_edit, 2)

        self._engine_filter_combo = NoWheelComboBox()
        self._engine_filter_combo.setObjectName("games_registry_engine_filter_combo")
        for value, label in registry_engine_filter_options():
            self._engine_filter_combo.addItem(label, value)
        self._engine_filter_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self._engine_filter_combo)

        self._translation_filter_combo = NoWheelComboBox()
        self._translation_filter_combo.setObjectName("games_registry_translation_filter_combo")
        for label in registry_translation_filter_options():
            self._translation_filter_combo.addItem(label, label)
        self._translation_filter_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self._translation_filter_combo)

        sort_label = QLabel("排序")
        sort_label.setObjectName("config_hint_label")
        filter_row.addWidget(sort_label)
        self._sort_combo = NoWheelComboBox()
        self._sort_combo.setObjectName("games_registry_sort_combo")
        for value, label in REGISTRY_SORT_OPTIONS:
            self._sort_combo.addItem(label, value)
        self._sort_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self._sort_combo)

        # Engine / translation / sort share one compact fixed width after show.
        _uniform_combo_width(
            self._engine_filter_combo,
            self._translation_filter_combo,
            self._sort_combo,
        )
        body_layout.addLayout(filter_row)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setObjectName("config_hint_label")
        body_layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("games_registry_progress")
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("准备刷新…")
        body_layout.addWidget(self._progress_bar)

        self._table = QTableWidget(0, len(REGISTRY_TABLE_COLUMN_DEFS))
        self._table.setObjectName("games_registry_table")
        self._table.setHorizontalHeaderLabels(column_headers())
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        # Long cells elide; full text stays on item tooltips (EUI truncation default).
        self._table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.verticalHeader().setVisible(False)
        self._configure_table_header()
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.cellDoubleClicked.connect(self._on_row_activated)
        body_layout.addWidget(self._table, 1)
        self._load_table_column_widths()
        self._apply_table_column_layout()

        self._edit_group = QGroupBox("项目详情")
        self._edit_group.setObjectName("games_registry_edit_group")
        self._edit_group.setVisible(False)
        edit_layout = QFormLayout(self._edit_group)
        edit_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        edit_layout.setHorizontalSpacing(12)
        edit_layout.setVerticalSpacing(8)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("显示名称")
        edit_layout.addRow("项目名称", self._name_edit)

        self._layout_status_label = QLabel("—")
        self._layout_status_label.setWordWrap(False)
        self._layout_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        edit_layout.addRow("目录状态", self._layout_status_label)

        self._doctor_mode_label = QLabel("—")
        self._doctor_mode_label.setWordWrap(False)
        edit_layout.addRow("模板模式", self._doctor_mode_label)

        self._last_refresh_label = QLabel("—")
        self._last_refresh_label.setWordWrap(True)
        edit_layout.addRow("最近刷新", self._last_refresh_label)

        self._doctor_check_layout_label = QLabel("—")
        self._doctor_check_layout_label.setWordWrap(False)
        edit_layout.addRow("环境检查·目录", self._doctor_check_layout_label)

        self._doctor_check_mode_label = QLabel("—")
        self._doctor_check_mode_label.setWordWrap(False)
        edit_layout.addRow("环境检查·模板", self._doctor_check_mode_label)

        self._registry_compare_label = QLabel("—")
        self._registry_compare_label.setWordWrap(True)
        self._registry_compare_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        edit_layout.addRow("总表对比", self._registry_compare_label)

        self._play_status_combo = NoWheelComboBox()
        self._play_status_combo.setObjectName("games_registry_play_status_combo")
        self._play_status_combo.addItems(sorted(PLAY_STATUSES))
        self._translation_status_combo = NoWheelComboBox()
        self._translation_status_combo.setObjectName("games_registry_translation_status_combo")
        self._translation_status_combo.addItems(sorted(TRANSLATION_STATUSES))
        # Compact equal width for both status fields (not full form row).
        _uniform_combo_width(
            self._play_status_combo,
            self._translation_status_combo,
            minimum=96,
        )
        edit_layout.addRow("游玩状态", self._play_status_combo)
        edit_layout.addRow("翻译状态", self._translation_status_combo)

        self._notes_edit = QPlainTextEdit()
        self._notes_edit.setPlaceholderText("备注 / 下一步")
        self._notes_edit.setFixedHeight(88)
        self._notes_edit.setMinimumHeight(72)
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
        body_layout.addWidget(self._edit_group)
        layout.addWidget(self._workspace_body, 1)

        prefs = load_registry_preferences(workspace_root=self._workspace_root)
        self._auto_discover_checkbox.setChecked(bool(prefs.get(REGISTRY_PREF_AUTO_DISCOVER, False)))
        self._apply_workspace_presence_ui()
        if self._workspace_root is not None:
            self._reload_table_from_disk()

    def selected_project_root(self) -> str:
        return self._selected_project_root

    def workspace_root(self) -> Path | None:
        return self._workspace_root

    def set_workspace_root(self, workspace_root: Path | None) -> None:
        """Update the active workspace and refresh registry UI."""
        self._workspace_root = Path(workspace_root) if workspace_root is not None else None
        self._registry_disk_signature = None
        prefs = load_registry_preferences(workspace_root=self._workspace_root)
        self._auto_discover_checkbox.blockSignals(True)
        self._auto_discover_checkbox.setChecked(bool(prefs.get(REGISTRY_PREF_AUTO_DISCOVER, False)))
        self._auto_discover_checkbox.blockSignals(False)
        self._load_table_column_widths()
        self._apply_workspace_presence_ui()
        if self._workspace_root is not None:
            self._reload_table_from_disk(force=True)
        else:
            self._all_rows = []
            self._filtered_rows = []
            self._registry_summary = ""
            self._missing_message = "工作区未设置。"
            self._table.setRowCount(0)
            self._status_label.setText(self._missing_message)
            self._edit_group.setVisible(False)
            self._switch_btn.setEnabled(False)

    def set_current_game_root(self, game_root: Path | None) -> None:
        self._current_game_root = game_root
        self._on_selection_changed()

    def _has_workspace(self) -> bool:
        return self._workspace_root is not None

    def _require_workspace(self) -> Path | None:
        if self._workspace_root is None:
            message_box_information(
                self,
                "工作区未设置",
                "请先点击「选择工作区…」指定存放 Game_* 与 games_registry.json 的目录。",
            )
            return None
        return self._workspace_root

    def _apply_workspace_presence_ui(self) -> None:
        has_workspace = self._has_workspace()
        if has_workspace and self._workspace_root is not None:
            self._workspace_path_edit.setText(canonical_abs_path(str(self._workspace_root)))
            self._choose_workspace_btn.setText("更改工作区…")
        else:
            self._workspace_path_edit.clear()
            self._choose_workspace_btn.setText("选择工作区…")
        self._workspace_empty_state.setVisible(not has_workspace)
        self._workspace_body.setVisible(has_workspace)
        self._switch_btn.setEnabled(False)
        if not has_workspace:
            self._apply_interaction_gate()

    def _choose_workspace(self) -> None:
        if self._host_task_running or self._is_registry_task_running():
            message_box_information(
                self,
                "暂时无法切换工作区",
                "工作台任务或总表刷新进行中，请结束后再选择工作区。",
            )
            return
        start_dir = str(self._workspace_root or Path.home())
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择工作区根目录（存放 Game_* 与 games_registry.json）",
            start_dir,
        )
        if not directory:
            return
        chosen = Path(canonical_abs_path(directory))
        if self._on_workspace_changed is not None:
            # Host persists first and re-syncs this panel (including failure revert).
            self._on_workspace_changed(chosen)
            return
        self.set_workspace_root(chosen)

    def activate_section(self) -> None:
        """Refresh view when the settings nav selects the workspace section."""
        if not self._has_workspace():
            self._apply_workspace_presence_ui()
            return
        self._reload_table_from_disk(force=False)
        if self._auto_discover_on_show:
            self._maybe_auto_discover_on_open()

    def _reflow_toolbars(self, *, force: bool = False) -> None:
        for toolbar in getattr(self, "_toolbars", ()):
            reflow = getattr(toolbar, "reflow", None)
            if callable(reflow):
                reflow(force=force)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._section_visible:
            self._section_visible = True
            self.activate_section()
        self._reflow_toolbars(force=True)
        self._reflow_uniform_combos()
        # Apply stored interactive widths once geometry is real; path flex fills rest.
        self._apply_table_column_layout()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reflow_toolbars()
        # Interactive widths stay put; last-section Stretch absorbs leftover space.

    # --- Table columns (EUI presets: interactive + last flex path) -------------

    def _configure_table_header(self) -> None:
        header = self._table.horizontalHeader()
        # Path is always last → StretchLastSection is Qt-stable (no mid-Stretch fight).
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(48)
        header.setSectionsClickable(True)
        header.setSectionsMovable(False)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._on_table_header_menu)
        header.sectionResized.connect(self._on_table_section_resized)
        header.sectionDoubleClicked.connect(self._on_table_section_double_clicked)
        path_index = REGISTRY_TABLE_PATH_COLUMN
        for index in range(len(REGISTRY_TABLE_COLUMN_DEFS)):
            if index == path_index:
                header.setSectionResizeMode(index, QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(index, QHeaderView.ResizeMode.Interactive)

    def _header_min_width(self, column_index: int) -> int:
        """EUI min: header + longest enum sample fully readable."""
        column = column_at(column_index)
        if column is None:
            return 56
        metrics = self._table.fontMetrics()
        return min_width_for_column(column, metrics.horizontalAdvance)

    def _load_table_column_widths(self) -> None:
        prefs = load_registry_preferences(workspace_root=self._workspace_root)
        raw = prefs.get(REGISTRY_PREF_TABLE_COLUMN_WIDTHS)
        if raw is None:
            raw = prefs.get(REGISTRY_PREF_TABLE_COLUMN_WIDTHS_LEGACY)
        self._table_column_widths = migrate_stored_widths(raw)

    def _persist_table_column_widths(self) -> None:
        if self._applying_table_columns:
            return
        if self._workspace_root is None:
            return
        live: dict[str, int] = {}
        for index in interactive_column_indexes():
            column = column_at(index)
            if column is None:
                continue
            live[column.id] = int(self._table.columnWidth(index))
        self._table_column_widths = widths_for_persist(live)
        save_registry_dialog_preference(
            self._workspace_root,
            key=REGISTRY_PREF_TABLE_COLUMN_WIDTHS,
            value=dict(self._table_column_widths),
        )

    def _apply_table_column_layout(self) -> None:
        """Push stored interactive widths; path Stretch fills remaining space."""
        table = getattr(self, "_table", None)
        if table is None:
            return
        self._applying_table_columns = True
        try:
            for index in interactive_column_indexes():
                column = column_at(index)
                if column is None:
                    continue
                minimum = self._header_min_width(index)
                preferred = int(
                    self._table_column_widths.get(column.id, column.default_width)
                )
                width = clamp_width(column, preferred, min_width=minimum)
                table.setColumnWidth(index, width)
                self._table_column_widths[column.id] = width
        finally:
            self._applying_table_columns = False

    def _on_table_section_resized(self, logical: int, _old: int, new: int) -> None:
        """User drag: follow the mouse; only clamp to EUI min. Path is flex."""
        if self._applying_table_columns:
            return
        column = column_at(logical)
        if column is None or column.flex:
            return
        minimum = self._header_min_width(logical)
        width = clamp_width(column, new, min_width=minimum)
        if width != self._table.columnWidth(logical):
            self._applying_table_columns = True
            try:
                self._table.setColumnWidth(logical, width)
            finally:
                self._applying_table_columns = False
        self._table_column_widths[column.id] = width
        self._column_width_save_timer.start()

    def _on_table_section_double_clicked(self, logical: int) -> None:
        """Double-click: size column to contents (min header/enum, max preset)."""
        column = column_at(logical)
        if column is None or column.flex:
            return
        minimum = self._header_min_width(logical)
        self._applying_table_columns = True
        try:
            self._table.resizeColumnToContents(logical)
            width = clamp_width_for_fit(
                column,
                self._table.columnWidth(logical),
                min_width=minimum,
            )
            self._table.setColumnWidth(logical, width)
        finally:
            self._applying_table_columns = False
        self._table_column_widths[column.id] = width
        self._column_width_save_timer.start()

    def _on_table_header_menu(self, pos) -> None:
        menu = QMenu(self)
        reset_action = QAction("重置列宽", self)
        reset_action.setToolTip("恢复默认固定列宽；路径列继续自动占满剩余空间")
        reset_action.triggered.connect(self._reset_table_column_layout)
        menu.addAction(reset_action)
        fit_action = QAction("按内容调整固定列", self)
        fit_action.setToolTip("按当前单元格内容调整固定列（受列最大宽限制），路径列仍弹性")
        fit_action.triggered.connect(self._fit_all_table_columns)
        menu.addAction(fit_action)
        header = self._table.horizontalHeader()
        menu.exec(header.mapToGlobal(pos))

    def _reset_table_column_layout(self) -> None:
        self._table_column_widths = default_width_map()
        self._apply_table_column_layout()
        self._persist_table_column_widths()

    def _fit_all_table_columns(self) -> None:
        self._applying_table_columns = True
        try:
            for index in interactive_column_indexes():
                column = column_at(index)
                if column is None:
                    continue
                minimum = self._header_min_width(index)
                self._table.resizeColumnToContents(index)
                width = clamp_width_for_fit(
                    column,
                    self._table.columnWidth(index),
                    min_width=minimum,
                )
                self._table.setColumnWidth(index, width)
                self._table_column_widths[column.id] = width
        finally:
            self._applying_table_columns = False
        self._persist_table_column_widths()

    def _reflow_uniform_combos(self) -> None:
        engine = getattr(self, "_engine_filter_combo", None)
        translation_filter = getattr(self, "_translation_filter_combo", None)
        sort_combo = getattr(self, "_sort_combo", None)
        filter_group = [
            combo
            for combo in (engine, translation_filter, sort_combo)
            if combo is not None
        ]
        if len(filter_group) >= 2:
            _uniform_combo_width(*filter_group)
        play = getattr(self, "_play_status_combo", None)
        translation = getattr(self, "_translation_status_combo", None)
        if play is not None and translation is not None:
            _uniform_combo_width(play, translation)

    def hideEvent(self, event) -> None:
        if self._is_refresh_running() and self._refresh_worker is not None:
            self._refresh_worker.request_stop()
            self._refresh_worker.wait(5000)
        if self._is_ingest_running() and self._ingest_worker is not None:
            self._ingest_worker.request_stop()
            self._ingest_worker.wait(5000)
        super().hideEvent(event)

    def _registry_disk_signature_now(self) -> tuple[str, int]:
        if self._workspace_root is None:
            return ("", -1)
        registry_path = resolve_registry_path(self._workspace_root)
        path_text = str(registry_path)
        try:
            mtime_ns = registry_path.stat().st_mtime_ns
        except OSError:
            mtime_ns = -1
        return path_text, mtime_ns

    def _current_doctor_report(self) -> dict | None:
        if self._get_doctor_report is None:
            return None
        report = self._get_doctor_report()
        return report if isinstance(report, dict) else None

    def _selected_sort_key(self) -> str:
        value = self._sort_combo.currentData()
        if isinstance(value, str) and value:
            return value
        return REGISTRY_SORT_NAME_ASC

    def _reload_table_from_disk(self, *, force: bool = True) -> None:
        if self._workspace_root is None:
            self._all_rows = []
            self._filtered_rows = []
            self._missing_message = "工作区未设置。"
            self._table.setRowCount(0)
            self._status_label.setText(self._missing_message)
            return

        signature = self._registry_disk_signature_now()
        if not force and self._registry_disk_signature == signature:
            # Unchanged registry file: keep rows, only re-apply selection highlight.
            self._on_selection_changed()
            return

        registry_path = resolve_registry_path(self._workspace_root)
        rows, summary = load_registry_rows(
            workspace_root=self._workspace_root,
            registry_path=registry_path,
        )
        self._registry_disk_signature = signature
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
            values = row_cell_values(row)
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                # Full cell text in tooltip when elided; keep row scan summary too.
                tips = [part for part in (value, row.tooltip) if part and part.strip()]
                if tips:
                    unique: list[str] = []
                    for part in tips:
                        if part not in unique:
                            unique.append(part)
                    item.setToolTip("\n".join(unique))
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
        # Do not re-clamp column widths on every data refresh (preserves user drag).

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

    def _is_ingest_running(self) -> bool:
        return self._ingest_worker is not None and self._ingest_worker.isRunning()

    def _is_registry_task_running(self) -> bool:
        return self._is_refresh_running() or self._is_ingest_running()

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

            doctor_report = self._current_doctor_report()
            compare = None
            show_doctor = doctor_report is not None and row_matches_game_root(
                row,
                self._current_game_root,
            )
            if show_doctor:
                doctor_layout = str(doctor_report.get("layout_status") or "").strip()
                doctor_mode = str(doctor_report.get("mode") or "").strip()
                self._doctor_check_layout_label.setText(
                    format_layout_status_label(doctor_layout) if doctor_layout else "—"
                )
                self._doctor_check_mode_label.setText(
                    format_doctor_mode_label(doctor_mode) if doctor_mode else "—"
                )
                compare = (
                    compare_registry_with_doctor_report(
                        self._workspace_root,
                        game_root=self._current_game_root,
                        report=doctor_report,
                    )
                    if self._workspace_root is not None
                    else None
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
                    "选中当前工作台项目并完成「环境检查」后，可在此查看总表是否需刷新。"
                    if row_matches_game_root(row, self._current_game_root)
                    else "切换到该项目并完成「环境检查」后，可对比总表记录。"
                )
            )

            # Match table text: keep free-form / annotated statuses (e.g. 已完成（6.7 增量）).
            _set_status_combo_value(
                self._play_status_combo,
                normalize_play_status(row.play_status),
            )
            _set_status_combo_value(
                self._translation_status_combo,
                normalize_translation_status(row.translation_status),
            )
            self._notes_edit.setPlainText(row.notes)
        finally:
            self._edit_loading = False

    def _on_selection_changed(self) -> None:
        row = self._selected_row()
        can_use_row = row is not None and bool(row.project_id)
        self._edit_group.setVisible(row is not None)
        self._populate_edit_panel(row)
        refresh_busy = bool(self._refresh_ui_busy) or self._is_registry_task_running()
        if not refresh_busy and not self._host_task_running:
            self._switch_btn.setEnabled(row is not None and bool(row.work_dir))
            self._refresh_current_btn.setEnabled(can_use_row)
            self._save_fields_btn.setEnabled(can_use_row)
            self._delete_project_btn.setEnabled(can_use_row)
        elif not refresh_busy and self._host_task_running:
            self._switch_btn.setEnabled(False)
            self._refresh_current_btn.setEnabled(False)
            self._save_fields_btn.setEnabled(False)
            self._delete_project_btn.setEnabled(False)
            # Keep detail fields read-only while host task runs.
            for widget in (
                self._name_edit,
                self._play_status_combo,
                self._translation_status_combo,
                self._notes_edit,
            ):
                widget.setEnabled(False)

    def _selected_refresh_mode(self) -> str:
        mode = self._refresh_mode_combo.currentData()
        if mode in {REFRESH_MODE_LITE, REFRESH_MODE_DEEP}:
            return str(mode)
        return REFRESH_MODE_LITE

    def set_host_task_running(self, running: bool) -> None:
        """Allow browse while a workbench job runs; block switch/mutate only."""
        self._host_task_running = bool(running)
        self._apply_interaction_gate()

    def _apply_interaction_gate(self) -> None:
        """Combine local refresh-busy and host workbench-running gates."""
        # Prefer the explicit UI flag so buttons lock as soon as refresh is
        # requested, even before the QThread reports isRunning().
        if not self._has_workspace():
            choose_enabled = not self._host_task_running
            self._choose_workspace_btn.setEnabled(choose_enabled)
            empty = getattr(self, "_workspace_empty_state", None)
            if empty is not None and hasattr(empty, "set_action_enabled"):
                empty.set_action_enabled(choose_enabled)
            return

        refresh_busy = bool(self._refresh_ui_busy) or self._is_registry_task_running()
        host_busy = self._host_task_running
        mutate_blocked = refresh_busy or host_busy

        self._choose_workspace_btn.setEnabled(not mutate_blocked)

        for widget in (
            self._refresh_current_btn,
            self._refresh_all_btn,
            self._refresh_mode_combo,
            self._switch_btn,
            self._import_md_btn,
            self._discover_btn,
            self._ingest_btn,
            self._sync_md_btn,
            self._save_fields_btn,
            self._delete_project_btn,
            self._name_edit,
            self._play_status_combo,
            self._translation_status_combo,
            self._notes_edit,
            self._auto_discover_checkbox,
        ):
            widget.setEnabled(not mutate_blocked)

        # Browse/filter remain available during host jobs; only local refresh
        # keeps the table interactive (scroll/read while refresh runs).
        for widget in (
            self._search_edit,
            self._engine_filter_combo,
            self._translation_filter_combo,
            self._sort_combo,
        ):
            widget.setEnabled(not refresh_busy)
        # Table stays enabled during refresh so users can still scroll the list.
        self._table.setEnabled(True)

        self._stop_refresh_btn.setEnabled(refresh_busy)
        if not refresh_busy:
            self._on_selection_changed()
            if host_busy:
                self._switch_btn.setEnabled(False)
                self._refresh_current_btn.setEnabled(False)
                self._save_fields_btn.setEnabled(False)
                self._delete_project_btn.setEnabled(False)

    def _set_refresh_busy(self, busy: bool) -> None:
        self._refresh_ui_busy = bool(busy)
        self._progress_bar.setVisible(busy)
        if busy:
            self._progress_bar.setValue(0)
        else:
            self._progress_bar.setFormat("准备刷新…")
        self._apply_interaction_gate()

    def _on_auto_discover_toggled(self, checked: bool) -> None:
        if self._is_registry_task_running():
            return
        workspace = self._require_workspace()
        if workspace is None:
            return
        save_registry_dialog_preference(
            workspace,
            key=REGISTRY_PREF_AUTO_DISCOVER,
            value=checked,
        )

    def _maybe_auto_discover_on_open(self) -> None:
        if not self._has_workspace():
            return
        if not self._auto_discover_checkbox.isChecked() or self._is_registry_task_running():
            return
        workspace = self._workspace_root
        if workspace is None:
            return
        if count_undiscovered_projects(workspace_root=workspace) <= 0:
            return
        result = discover_registry_projects(
            workspace,
            refresh_new=True,
            mode=REFRESH_MODE_LITE,
        )
        self._handle_action_result(result, title="自动扫描失败")

    def _import_from_games_md(self) -> None:
        if self._is_registry_task_running():
            return
        workspace = self._require_workspace()
        if workspace is None:
            return
        registry_path = resolve_registry_path(workspace)
        merge = registry_path.is_file()
        if merge:
            reply = message_box_question(
                self,
                "从 GAMES.md 导入",
                "已存在 games_registry.json。\n\n"
                "• 选择「是」：按路径合并（GAMES.md 中的列覆盖同路径项目；"
                "未出现在 MD 中的 JSON 项目保留）\n"
                "• 选择「否」：用 GAMES.md 完全替换总表\n"
                "• 选择「取消」：放弃\n\n"
                "合并后如需让 Markdown 与 JSON 一致，请再点「同步 GAMES.md」。",
                yes_text="是",
                no_text="否",
                cancel_text="取消",
                default="yes",
            )
            if reply == "cancel":
                return
            merge = reply == "yes"

        result = import_registry_from_games_md(workspace, merge=merge)
        self._handle_action_result(result, title="导入失败")

    def _discover_new_projects(self) -> None:
        if self._is_registry_task_running():
            return
        workspace = self._require_workspace()
        if workspace is None:
            return
        result = discover_registry_projects(
            workspace,
            refresh_new=True,
            mode=REFRESH_MODE_LITE,
        )
        self._handle_action_result(result, title="扫描失败")

    def _ingest_game(self) -> None:
        if self._is_registry_task_running():
            return
        workspace = self._require_workspace()
        if workspace is None:
            return
        dialog = GameIngestDialog(
            self,
            workspace_root=workspace,
            start_dir=workspace,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        payload = dialog.result_payload()
        if payload is None:
            return
        dest = (workspace / payload.folder_name).resolve()
        reply = message_box_question(
            self,
            "确认导入",
            "将复制到：\n"
            f"{dest}\n\n"
            f"游戏名称：{payload.game_name}\n"
            f"最终目录：{payload.folder_name}\n\n"
            "源文件会保留，不会自动准备 work。确定继续？",
            yes_text="确定",
            no_text="取消",
            default="yes",
        )
        if reply != "yes":
            return
        self._run_ingest(source=payload.source, game_name=payload.game_name)

    def _run_ingest(self, *, source: Path, game_name: str) -> None:
        if self._is_registry_task_running():
            return
        workspace = self._require_workspace()
        if workspace is None:
            return
        self._set_refresh_busy(True)
        self._progress_bar.setFormat("准备导入…")
        self._status_label.setText(f"正在导入：{game_name}")
        worker = RegistryIngestWorker(
            workspace_root=workspace,
            source=source,
            game_name=game_name,
            mode=REFRESH_MODE_LITE,
            parent=self,
        )
        self._ingest_worker = worker
        worker.progress.connect(self._on_ingest_progress)
        worker.completed.connect(self._on_ingest_finished)
        worker.start()

    def _on_ingest_progress(self, current: int, total: int, name: str) -> None:
        self._progress_bar.setMaximum(max(total, 1))
        self._progress_bar.setValue(current)
        self._progress_bar.setFormat(f"{current}/{total} — {name}")
        self._status_label.setText(f"正在导入：{name}（{current}/{total}）")

    def _on_ingest_finished(self, result: object) -> None:
        worker = self._ingest_worker
        self._ingest_worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_refresh_busy(False)
        if not isinstance(result, RegistryActionResult):
            result = RegistryActionResult(False, "导入失败：未知结果。")
        if result.project_id:
            self._preserved_project_id = result.project_id
        self._handle_action_result(result, title="导入失败")
        if result.ok and self._workspace_root is not None:
            prompt_render_games_md_after_refresh(
                self,
                self._workspace_root,
                result.message or "已将新项目登记到 games_registry.json。",
            )

    def _sync_games_md(self) -> None:
        if self._is_registry_task_running():
            return
        workspace = self._require_workspace()
        if workspace is None:
            return
        result = render_registry_games_md(workspace)
        self._handle_action_result(result, title="同步失败")

    def _save_selected_project_fields(self) -> None:
        if self._is_registry_task_running():
            return
        workspace = self._require_workspace()
        if workspace is None:
            return
        row = self._selected_row()
        if row is None or not row.project_id:
            return
        result = save_registry_project_fields(
            workspace,
            project_id=row.project_id,
            name=self._name_edit.text(),
            play_status=self._play_status_combo.currentText(),
            translation_status=self._translation_status_combo.currentText(),
            notes=self._notes_edit.toPlainText(),
        )
        self._preserved_project_id = row.project_id
        self._handle_action_result(result, title="保存失败")

    def _delete_selected_project(self) -> None:
        if self._is_registry_task_running():
            return
        workspace = self._require_workspace()
        if workspace is None:
            return
        row = self._selected_row()
        if row is None or not row.project_id:
            return
        reply = message_box_question(
            self,
            "删除项目",
            f"确定从总表移除「{row.name}」？\n\n"
            "这不会删除磁盘上的 Game_* 目录，只是不再在 registry / GAMES.md 中显示。",
            yes_text="删除",
            no_text="取消",
            default="no",
            icon=QMessageBox.Icon.Warning,
        )
        if reply != "yes":
            return
        result = delete_registry_project(workspace, project_id=row.project_id)
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
            message_box_warning(self, title, result.message)

    def _on_refresh_progress(self, current: int, total: int, name: str) -> None:
        self._progress_bar.setMaximum(max(total, 1))
        self._progress_bar.setValue(current)
        self._progress_bar.setFormat(f"{current}/{total} — {name}")
        self._status_label.setText(f"正在刷新：{name}（{current}/{total}）")

    def _on_stop_refresh(self) -> None:
        stopped = False
        if self._refresh_worker is not None and self._refresh_worker.isRunning():
            self._refresh_worker.request_stop()
            stopped = True
        if self._ingest_worker is not None and self._ingest_worker.isRunning():
            self._ingest_worker.request_stop()
            stopped = True
        if not stopped:
            return
        self._stop_refresh_btn.setEnabled(False)
        self._status_label.setText("正在停止…（等待当前任务完成）")

    def _refresh_current_project(self) -> None:
        row = self._selected_row()
        if row is None or not row.project_id:
            message_box_information(self, "请选择项目", "请先在表格里选中一个项目。")
            return
        self._preserved_project_id = row.project_id
        self._run_refresh(project_id=row.project_id, mode=self._selected_refresh_mode())

    def _refresh_all_projects(self) -> None:
        mode = self._selected_refresh_mode()
        if mode == REFRESH_MODE_DEEP:
            reply = message_box_question(
                self,
                "深度刷新全部",
                f"将对全部 {len(self._all_rows)} 个项目运行完整 doctor，可能需要数分钟。\n确定继续？",
                yes_text="确定",
                no_text="取消",
                default="no",
            )
            if reply != "yes":
                return
        self._run_refresh(refresh_everything=True, mode=mode)

    def _run_refresh(
        self,
        *,
        project_id: str | None = None,
        refresh_everything: bool = False,
        mode: str = REFRESH_MODE_LITE,
    ) -> None:
        if self._is_registry_task_running():
            return
        workspace = self._require_workspace()
        if workspace is None:
            return

        self._refresh_worker = RegistryRefreshWorker(
            workspace_root=workspace,
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
            message_box_warning(self, "刷新失败", message)
            self._status_label.setText(message)
            return

        if self._workspace_root is None:
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

    def _on_row_activated(self, row_index: int, _column: int) -> None:
        if self._is_registry_task_running() or self._host_task_running:
            return
        if row_index < 0 or row_index >= len(self._filtered_rows):
            return
        self._table.selectRow(row_index)
        self._switch_to_selected()

    def _switch_to_selected(self) -> None:
        if self._is_registry_task_running() or self._host_task_running:
            return
        workspace = self._require_workspace()
        if workspace is None:
            return
        row = self._selected_row()
        if row is None or not row.work_dir:
            return
        # Prefer effective work dir when present; set_game_root still normalizes.
        target = row.work_dir or str(workspace / row.path)
        self._selected_project_root = target
        if self._on_switch_project is None:
            return
        if self._on_switch_project(target):
            # Handler already refreshed current_game_root from state; keep a
            # local fallback for dialog wrappers that only set selected path.
            if self._current_game_root is None:
                self.set_current_game_root(Path(target))