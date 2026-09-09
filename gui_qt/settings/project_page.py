"""Independent Project Settings page (#202 Phase D).

Owns translator_config fields in the 「项目与资源」 and 「准备流程」 groups.
game_root is display-only here; switching projects stays on the workspace page.
Ren'Py SDK browse/find/download stay on the host (dialogs + SdkInstallWorker).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QWidget,
)

from ..settings_schema import ADVANCED_SETTING_FIELDS, SettingField
from .field_widgets import (
    apply_setting_value_to_widget,
    create_basic_setting_widget,
    setting_field_row,
    setting_value_from_widget,
)
from .page_chrome import build_settings_scroll_page, settings_form
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import PROJECT_CONFIG_KEYS, SETTINGS_PAGE_SPEC_OBJECTS

PROJECT_PAGE_KEY = "project"
_PROJECT_SPEC = next(
    spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == PROJECT_PAGE_KEY
)
PROJECT_NAV_LABEL = _PROJECT_SPEC.nav_label
PROJECT_IMMEDIATE_ACTION_IDS = _PROJECT_SPEC.immediate_action_ids

PROJECT_FIELDS: tuple[SettingField, ...] = tuple(
    field for field in ADVANCED_SETTING_FIELDS if field.key in PROJECT_CONFIG_KEYS
)
_PROJECT_CATEGORIES: tuple[str, ...] = ("项目与资源", "准备流程")

PROJECT_WIDGET_ATTRS: tuple[str, ...] = (
    "settings_project_root_value",
    "settings_go_workspace_btn",
    "_prepare_renpy_sdk_browse_btn",
    "_prepare_renpy_sdk_find_btn",
    "_prepare_renpy_sdk_download_btn",
)

SdkPathCallback = Callable[[QLineEdit], None]


class ProjectSettingsPage(QObject):
    """Settings page for the current project's translator_config fields."""

    page_key = PROJECT_PAGE_KEY
    nav_label = PROJECT_NAV_LABEL
    config_keys = PROJECT_CONFIG_KEYS
    immediate_action_ids = PROJECT_IMMEDIATE_ACTION_IDS

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        actions: SettingsPageActions | None = None,
        on_browse_sdk: SdkPathCallback | None = None,
        on_find_sdk: SdkPathCallback | None = None,
        on_download_sdk: SdkPathCallback | None = None,
        is_sdk_install_running: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self._actions = actions or SettingsPageActions()
        self._on_browse_sdk = on_browse_sdk
        self._on_find_sdk = on_find_sdk
        self._on_download_sdk = on_download_sdk
        self._is_sdk_install_running = is_sdk_install_running
        self._loading = False
        self._task_running = False
        self._baseline: dict[str, object] = {}
        self.field_widgets: dict[str, QWidget] = {}
        self.error_labels: dict[str, QLabel] = {}
        self.widget, self.body = self._build_widgets()
        self._baseline = dict(self.collect())

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self._actions = actions

    def attach_widget_aliases(self, host_obj: object) -> None:
        target = getattr(host_obj, "__dict__", None)
        if not isinstance(target, dict):
            return
        for name in PROJECT_WIDGET_ATTRS:
            target[name] = getattr(self, name)
        widgets = getattr(host_obj, "_advanced_setting_widgets", None)
        if isinstance(widgets, dict):
            widgets.update(self.field_widgets)
        errors = getattr(host_obj, "_advanced_setting_error_labels", None)
        if isinstance(errors, dict):
            errors.update(self.error_labels)

    def set_game_root_display(self, root: object) -> None:
        if root:
            text = str(root)
            self.settings_project_root_value.setText(text)
            self.settings_project_root_value.setToolTip(text)
            return
        self.settings_project_root_value.setText(
            "（尚未选择项目，请前往「项目列表」切换）"
        )
        self.settings_project_root_value.setToolTip("")

    def load(self, snapshot: Mapping[str, object]) -> None:
        previous = self._loading
        self._loading = True
        try:
            for field in PROJECT_FIELDS:
                if field.key not in snapshot:
                    continue
                widget = self.field_widgets.get(field.key)
                if widget is None:
                    continue
                apply_setting_value_to_widget(
                    field, widget, snapshot.get(field.key, field.default)
                )
            self._baseline = dict(self.collect())
        finally:
            self._loading = previous

    def collect(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for field in PROJECT_FIELDS:
            widget = self.field_widgets.get(field.key)
            if widget is None:
                continue
            values[field.key] = setting_value_from_widget(field, widget)
        return values

    def validate(self) -> Sequence[SettingsIssue]:
        return []

    def reset(self) -> None:
        if self._baseline:
            self.load(self._baseline)

    def focus_issue(self, issue: SettingsIssue) -> bool:
        widget = self.field_widgets.get(issue.field_key)
        if widget is not None and hasattr(widget, "setFocus"):
            widget.setFocus()
            return True
        return False

    def set_task_running(self, running: bool) -> None:
        self._task_running = bool(running)
        idle = not self._task_running
        for widget in self.field_widgets.values():
            widget.setEnabled(idle)
        sdk_busy = bool(
            self._is_sdk_install_running and self._is_sdk_install_running()
        )
        self._prepare_renpy_sdk_browse_btn.setEnabled(idle)
        self._prepare_renpy_sdk_find_btn.setEnabled(idle)
        self._prepare_renpy_sdk_download_btn.setEnabled(idle or sdk_busy)

    def _on_go_workspace(self) -> None:
        navigate = self._actions.navigate
        if callable(navigate):
            navigate("workspace")

    def _wrap_sdk_path(self, line_edit: QLineEdit) -> QWidget:
        host = QWidget()
        host.setObjectName("prepare_renpy_sdk_path_row")
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(line_edit, 1)

        browse_btn = QPushButton("浏览…")
        browse_btn.setObjectName("secondary_btn")
        browse_btn.setToolTip("手动选择包含 renpy.py 的 Ren'Py SDK 目录。")
        browse_btn.clicked.connect(lambda: self._call_sdk(self._on_browse_sdk, line_edit))
        layout.addWidget(browse_btn)

        find_btn = QPushButton("查找 SDK")
        find_btn.setObjectName("secondary_btn")
        find_btn.setToolTip(
            "仅在点击后才会扫描：当前项目、已选工作区与工具附近的 renpy-*-sdk / renpy.py。"
            "平时加载配置与 prepare 不会自动搜其它目录。找到后填入（多结果时可选）。"
        )
        find_btn.clicked.connect(lambda: self._call_sdk(self._on_find_sdk, line_edit))
        layout.addWidget(find_btn)

        download_btn = QPushButton("下载推荐 SDK…")
        download_btn.setObjectName("secondary_btn")
        download_btn.setToolTip(
            "仅在确认后从官方 renpy.org 下载本工具维护的推荐稳定版 SDK。"
            "不会因打开本页或运行 prepare 自动联网。下载中可再次点击取消。"
        )
        download_btn.clicked.connect(
            lambda: self._call_sdk(self._on_download_sdk, line_edit)
        )
        layout.addWidget(download_btn)

        self._prepare_renpy_sdk_browse_btn = browse_btn
        self._prepare_renpy_sdk_find_btn = find_btn
        self._prepare_renpy_sdk_download_btn = download_btn
        return host

    def _call_sdk(
        self,
        callback: SdkPathCallback | None,
        line_edit: QLineEdit,
    ) -> None:
        if callable(callback):
            callback(line_edit)

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = build_settings_scroll_page("settings_project")
        hint = QLabel(
            "配置当前选中项目的 translator_config.json 参数。"
            "切换 work 目录请前往「项目列表」；此处只调整术语表、翻译目录、过滤器和准备流程。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        current_box = QGroupBox("当前项目")
        current_form = settings_form(current_box)
        self.settings_project_root_value = QLabel("（尚未选择项目）")
        self.settings_project_root_value.setWordWrap(True)
        self.settings_project_root_value.setObjectName("settings_project_root_value")
        current_form.addRow("游戏 work 目录：", self.settings_project_root_value)
        switch_row = QHBoxLayout()
        switch_row.addStretch(1)
        self.settings_go_workspace_btn = QPushButton("在项目列表切换…")
        self.settings_go_workspace_btn.setObjectName("secondary_btn")
        self.settings_go_workspace_btn.clicked.connect(self._on_go_workspace)
        switch_row.addWidget(self.settings_go_workspace_btn)
        current_form.addRow("", switch_row)
        layout.addWidget(current_box)

        fields_by_category: dict[str, list[SettingField]] = {
            category: [] for category in _PROJECT_CATEGORIES
        }
        for field in PROJECT_FIELDS:
            if field.category in fields_by_category:
                fields_by_category[field.category].append(field)

        for category in _PROJECT_CATEGORIES:
            group = QGroupBox(category)
            form = settings_form(group)
            for field in fields_by_category[category]:
                editor = create_basic_setting_widget(field)
                self.field_widgets[field.key] = editor
                error = QLabel()
                self.error_labels[field.key] = error
                row_widget: QWidget = editor
                if field.key == "prepare_renpy_sdk_dir" and isinstance(
                    editor, QLineEdit
                ):
                    row_widget = self._wrap_sdk_path(editor)
                form.addRow(f"{field.label}：", setting_field_row(field, row_widget, error))
            layout.addWidget(group)
        layout.addStretch(1)
        return page, body
