"""Independent Advanced Settings page (#202 Phase D).

Owns ADVANCED_CONFIG_KEYS (schema fields not on project/context/workspace).
Gemini catalog extras and the rotation checklist live here. Persistence still
goes through MainWindow's single save transaction.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QListWidget,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gemini_model_catalog import allowed_gemini_rotation_models, write_model_catalog_extras

from ..settings_schema import (
    CONTEXT_PRIMARY_SETTING_CATEGORY,
    ADVANCED_SETTING_FIELDS,
    SettingField,
    grouped_advanced_fields,
)
from .field_widgets import (
    apply_setting_value_to_widget,
    create_basic_setting_widget,
    setting_field_row,
    setting_value_from_widget,
)
from .gemini_catalog_widgets import (
    create_gemini_catalog_list_editor,
    create_gemini_model_checklist,
    gemini_catalog_list_values,
    gemini_model_checklist_values,
    refresh_gemini_model_checklist,
    set_gemini_catalog_list_values,
    set_gemini_model_checklist_values,
)
from .page_chrome import build_settings_scroll_page, settings_form
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import ADVANCED_CONFIG_KEYS, SETTINGS_PAGE_SPEC_OBJECTS

ADVANCED_PAGE_KEY = "advanced"
_ADVANCED_SPEC = next(
    spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == ADVANCED_PAGE_KEY
)
ADVANCED_NAV_LABEL = _ADVANCED_SPEC.nav_label
ADVANCED_IMMEDIATE_ACTION_IDS = _ADVANCED_SPEC.immediate_action_ids

ADVANCED_FIELDS: tuple[SettingField, ...] = tuple(
    field for field in ADVANCED_SETTING_FIELDS if field.key in ADVANCED_CONFIG_KEYS
)

_SKIPPED_CATEGORIES = frozenset(
    {
        "项目与资源",
        "准备流程",
        CONTEXT_PRIMARY_SETTING_CATEGORY,
    }
)


class AdvancedSettingsPage(QObject):
    """Settings page for remaining translator_config advanced fields."""

    page_key = ADVANCED_PAGE_KEY
    nav_label = ADVANCED_NAV_LABEL
    config_keys = ADVANCED_CONFIG_KEYS
    immediate_action_ids = ADVANCED_IMMEDIATE_ACTION_IDS

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        actions: SettingsPageActions | None = None,
    ) -> None:
        super().__init__(parent)
        self._actions = actions or SettingsPageActions()
        self._loading = False
        self._task_running = False
        self._baseline: dict[str, object] = {}
        self.field_widgets: dict[str, QWidget] = {}
        self.error_labels: dict[str, QLabel] = {}
        self.widget, self.body = self._build_widgets()
        self._sync_rotation_enabled()
        self._baseline = dict(self.collect())

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self._actions = actions

    def attach_widget_aliases(self, host_obj: object) -> None:
        widgets = getattr(host_obj, "_advanced_setting_widgets", None)
        if isinstance(widgets, dict):
            widgets.update(self.field_widgets)
        errors = getattr(host_obj, "_advanced_setting_error_labels", None)
        if isinstance(errors, dict):
            errors.update(self.error_labels)

    def load(self, snapshot: Mapping[str, object], *, restore: bool = False) -> None:
        previous = self._loading
        self._loading = True
        try:
            for field in ADVANCED_FIELDS:
                if field.key not in snapshot or field.kind == "gemini_model_list":
                    continue
                widget = self.field_widgets.get(field.key)
                if widget is None:
                    continue
                self._apply_value(field, widget, snapshot.get(field.key, field.default))
            self._refresh_rotation_checklist(snapshot)
            if "model_rotation_models" in snapshot:
                widget = self.field_widgets.get("model_rotation_models")
                if isinstance(widget, QListWidget):
                    set_gemini_model_checklist_values(
                        widget, snapshot.get("model_rotation_models")
                    )
            self._sync_rotation_enabled()
            if not restore:
                self._baseline = dict(self.collect())
        finally:
            self._loading = previous

    def collect(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for field in ADVANCED_FIELDS:
            widget = self.field_widgets.get(field.key)
            if widget is None:
                continue
            values[field.key] = self._value_from_widget(field, widget)
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
        if idle:
            self._sync_rotation_enabled()

    def _show_status(self, message: str) -> None:
        show_status = self._actions.show_status
        if callable(show_status):
            show_status(message)

    def _create_field_widget(self, field: SettingField) -> QWidget:
        if field.kind == "gemini_model_list":
            return create_gemini_model_checklist()
        if field.kind == "gemini_catalog_list":
            kind = (
                "embedding"
                if field.key == "catalog_gemini_embedding_models"
                else "translation"
            )
            return create_gemini_catalog_list_editor(
                kind=kind,
                on_status=self._show_status,
            )
        widget = create_basic_setting_widget(field)
        if field.key == "model_rotation_enabled" and isinstance(widget, QCheckBox):
            widget.toggled.connect(self._on_model_rotation_enabled_toggled)
        return widget

    def _value_from_widget(self, field: SettingField, widget: QWidget) -> object:
        if field.kind == "gemini_model_list" and isinstance(widget, QListWidget):
            return gemini_model_checklist_values(widget)
        if field.kind == "gemini_catalog_list":
            return gemini_catalog_list_values(widget)
        return setting_value_from_widget(field, widget)

    def _apply_value(
        self,
        field: SettingField,
        widget: QWidget,
        value: object,
    ) -> None:
        if field.kind == "gemini_catalog_list":
            set_gemini_catalog_list_values(widget, value)
            return
        if field.kind == "gemini_model_list" and isinstance(widget, QListWidget):
            set_gemini_model_checklist_values(widget, value)
            return
        apply_setting_value_to_widget(field, widget, value)

    def _rotation_config(self, snapshot: Mapping[str, object] | None = None) -> dict:
        translation = None
        embedding = None
        if snapshot is not None:
            if "catalog_gemini_models" in snapshot:
                translation = snapshot.get("catalog_gemini_models") or []
            if "catalog_gemini_embedding_models" in snapshot:
                embedding = snapshot.get("catalog_gemini_embedding_models") or []
        if translation is None:
            widget = self.field_widgets.get("catalog_gemini_models")
            translation = gemini_catalog_list_values(widget) if widget is not None else []
        if embedding is None:
            widget = self.field_widgets.get("catalog_gemini_embedding_models")
            embedding = gemini_catalog_list_values(widget) if widget is not None else []
        config: dict = {}
        write_model_catalog_extras(
            config,
            translation_models=list(translation or []),
            embedding_models=list(embedding or []),
        )
        return config

    def _refresh_rotation_checklist(
        self,
        snapshot: Mapping[str, object] | None = None,
    ) -> None:
        widget = self.field_widgets.get("model_rotation_models")
        if not isinstance(widget, QListWidget):
            return
        selected = (
            snapshot.get("model_rotation_models")
            if snapshot is not None and "model_rotation_models" in snapshot
            else gemini_model_checklist_values(widget)
        )
        refresh_gemini_model_checklist(
            widget,
            models=allowed_gemini_rotation_models(self._rotation_config(snapshot)),
            selected=selected,
        )

    def _on_model_rotation_enabled_toggled(self, checked: bool) -> None:
        if self._task_running:
            return
        checklist = self.field_widgets.get("model_rotation_models")
        if checklist is not None:
            checklist.setEnabled(bool(checked))

    def _sync_rotation_enabled(self) -> None:
        enabled_widget = self.field_widgets.get("model_rotation_enabled")
        if isinstance(enabled_widget, QCheckBox):
            self._on_model_rotation_enabled_toggled(enabled_widget.isChecked())

    def _build_model_catalog_group(self, fields: Sequence[SettingField]) -> QWidget:
        group = QGroupBox("模型目录")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(14, 18, 14, 14)
        outer.setSpacing(12)

        intro = QLabel(
            "扩展「设置 → 模型」下拉可选的自定义模型。内置模型始终可用，无需在此添加。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("config_hint_label")
        outer.addWidget(intro)

        for field in fields:
            widget = self._create_field_widget(field)
            self.field_widgets[field.key] = widget
            section = QWidget()
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(4)
            title = QLabel(field.label)
            title.setObjectName("settings_description_label")
            section_layout.addWidget(title)
            section_layout.addWidget(widget)
            error = QLabel()
            error.setWordWrap(True)
            error.setObjectName("settings_error_label")
            self.error_labels[field.key] = error
            section_layout.addWidget(error)
            outer.addWidget(section)
        return group

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = build_settings_scroll_page("settings_advanced")
        hint = QLabel(
            "高级设置会直接影响请求大小、上下文注入和本地上下文路径。"
            "无效字段会在本页标出，并阻止保存。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        for group_title, fields in grouped_advanced_fields(
            include_context_primary=False,
        ):
            owned = [field for field in fields if field.key in ADVANCED_CONFIG_KEYS]
            if not owned or group_title in _SKIPPED_CATEGORIES:
                continue
            if group_title == "模型目录":
                layout.addWidget(self._build_model_catalog_group(owned))
                continue
            group = QGroupBox(group_title)
            form = settings_form(group)
            for field in owned:
                widget = self._create_field_widget(field)
                self.field_widgets[field.key] = widget
                error = QLabel()
                self.error_labels[field.key] = error
                form.addRow(
                    f"{field.label}：",
                    setting_field_row(field, widget, error),
                )
            if form.rowCount() == 0:
                group.deleteLater()
                continue
            layout.addWidget(group)
        layout.addStretch(1)
        return page, body
