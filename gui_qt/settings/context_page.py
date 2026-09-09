"""Independent Context Settings page (#202 Phase D).

Owns project-level RAG/index/analysis switches plus context primary
translator_config fields. Persistence still goes through MainWindow's
single save transaction (global config + project_context_settings.json).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
)

from ..settings_schema import (
    PROJECT_ANALYSIS_CONTEXT_SETTING_KEYS,
    SettingField,
    context_primary_setting_fields,
)
from .field_widgets import (
    apply_setting_value_to_widget,
    create_basic_setting_widget,
    setting_field_row,
    setting_value_from_widget,
)
from .page_chrome import build_settings_scroll_page, settings_group
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import CONTEXT_CONFIG_KEYS, SETTINGS_PAGE_SPEC_OBJECTS

CONTEXT_PAGE_KEY = "context"
_CONTEXT_SPEC = next(
    spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == CONTEXT_PAGE_KEY
)
CONTEXT_NAV_LABEL = _CONTEXT_SPEC.nav_label
CONTEXT_IMMEDIATE_ACTION_IDS = _CONTEXT_SPEC.immediate_action_ids

CONTEXT_PRIMARY_FIELDS: tuple[SettingField, ...] = context_primary_setting_fields()

CONTEXT_WIDGET_ATTRS: tuple[str, ...] = (
    "rag_enabled_cb",
    "source_index_enabled_cb",
    "sync_source_index_enabled_cb",
    "bootstrap_on_build_cb",
    "context_storage_game_cb",
    "sync_inject_published_brief_cb",
)

_CHECKBOX_KEYS: tuple[tuple[str, str], ...] = (
    ("rag_enabled", "rag_enabled_cb"),
    ("source_index_enabled", "source_index_enabled_cb"),
    ("bootstrap_on_build", "bootstrap_on_build_cb"),
    ("sync_source_index_enabled", "sync_source_index_enabled_cb"),
    ("sync_project_analysis_inject_enabled", "sync_inject_published_brief_cb"),
)


class ContextSettingsPage(QObject):
    """Settings page for project context switches and primary context fields."""

    page_key = CONTEXT_PAGE_KEY
    nav_label = CONTEXT_NAV_LABEL
    config_keys = CONTEXT_CONFIG_KEYS
    immediate_action_ids = CONTEXT_IMMEDIATE_ACTION_IDS

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        actions: SettingsPageActions | None = None,
        on_open_analysis_advanced: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._actions = actions or SettingsPageActions()
        self._on_open_analysis_advanced = on_open_analysis_advanced
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
        for name in CONTEXT_WIDGET_ATTRS:
            target[name] = getattr(self, name)
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
            for key, attr in _CHECKBOX_KEYS:
                if key not in snapshot:
                    continue
                getattr(self, attr).setChecked(bool(snapshot.get(key)))
            if "context_storage_location" in snapshot:
                self.context_storage_game_cb.setChecked(
                    snapshot.get("context_storage_location") == "game"
                )
            for field in CONTEXT_PRIMARY_FIELDS:
                if field.key not in snapshot:
                    continue
                widget = self.field_widgets.get(field.key)
                if widget is None:
                    continue
                apply_setting_value_to_widget(
                    field, widget, snapshot.get(field.key, field.default)
                )
            if not restore:
                self._baseline = dict(self.collect())
        finally:
            self._loading = previous

    def collect(self) -> dict[str, object]:
        values: dict[str, object] = {
            key: bool(getattr(self, attr).isChecked()) for key, attr in _CHECKBOX_KEYS
        }
        values["context_storage_location"] = (
            "game" if self.context_storage_game_cb.isChecked() else "tool"
        )
        for field in CONTEXT_PRIMARY_FIELDS:
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
        if widget is None:
            attr = dict(_CHECKBOX_KEYS).get(issue.field_key)
            if issue.field_key == "context_storage_location":
                widget = self.context_storage_game_cb
            elif attr:
                widget = getattr(self, attr, None)
        if widget is not None and hasattr(widget, "setFocus"):
            widget.setFocus()
            return True
        return False

    def set_task_running(self, running: bool) -> None:
        self._task_running = bool(running)
        idle = not self._task_running
        for attr in CONTEXT_WIDGET_ATTRS:
            getattr(self, attr).setEnabled(idle)
        for widget in self.field_widgets.values():
            widget.setEnabled(idle)

    def _open_analysis_advanced(self) -> None:
        if callable(self._on_open_analysis_advanced):
            self._on_open_analysis_advanced()
            return
        navigate = self._actions.navigate
        if callable(navigate):
            navigate("advanced")

    def _add_primary_fields(
        self,
        layout,
        fields: Sequence[SettingField],
    ) -> None:
        for field in fields:
            editor = create_basic_setting_widget(field)
            self.field_widgets[field.key] = editor
            error = QLabel()
            self.error_labels[field.key] = error
            layout.addWidget(QLabel(f"{field.label}"))
            layout.addWidget(setting_field_row(field, editor, error))

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = build_settings_scroll_page("settings_context")

        context_box, context_layout = settings_group("批量上下文")
        context_hint = QLabel(
            "下列三项按【当前项目】保存（work/project_context_settings.json），"
            "切换游戏不会互相覆盖。启用后先保存设置，再到工作台「上下文库」页预建。"
            "记忆库用已有译文；原文索引只用模板原文；均不修改游戏脚本。"
        )
        context_hint.setWordWrap(True)
        context_hint.setObjectName("config_hint_label")
        context_layout.addWidget(context_hint)

        self.rag_enabled_cb = QCheckBox("启用 RAG 记忆库（批量，当前项目）")
        context_layout.addWidget(self.rag_enabled_cb)
        self.source_index_enabled_cb = QCheckBox("启用原文索引（批量，当前项目）")
        context_layout.addWidget(self.source_index_enabled_cb)
        self.sync_source_index_enabled_cb = QCheckBox(
            "同步翻译使用原文索引（当前项目）"
        )
        self.sync_source_index_enabled_cb.setToolTip(
            "复用已预建的 Source Index Store。关闭时同步初译保持原行为。"
        )
        context_layout.addWidget(self.sync_source_index_enabled_cb)
        self.bootstrap_on_build_cb = QCheckBox("开始翻译时自动暖 RAG 库（当前项目）")
        context_layout.addWidget(self.bootstrap_on_build_cb)
        self.context_storage_game_cb = QCheckBox("上下文库保存到游戏目录")
        self.context_storage_game_cb.setToolTip(
            "启用后，默认 RAG / 原文索引 / 剧情图谱路径会使用 work 同级的 translation_context/。"
        )
        context_layout.addWidget(self.context_storage_game_cb)
        layout.addWidget(context_box)

        primary_box, primary_layout = settings_group("同步与剧情记忆")
        primary_hint = QLabel(
            "下列开关与高级页检索参数共用同一配置键；仅在本页编辑主开关，"
            "高级页只保留调参字段。"
        )
        primary_hint.setWordWrap(True)
        primary_hint.setObjectName("config_hint_label")
        primary_layout.addWidget(primary_hint)
        self._add_primary_fields(
            primary_layout,
            [
                field
                for field in CONTEXT_PRIMARY_FIELDS
                if field.key not in PROJECT_ANALYSIS_CONTEXT_SETTING_KEYS
            ],
        )
        layout.addWidget(primary_box)

        analysis_box, analysis_layout = settings_group("项目剧情分析")
        analysis_hint = QLabel(
            "先启用分析并保存，再到「上下文库」开始构建。只有人工确认、"
            "仍与当前脚本一致且开启“用于翻译”的项目摘要才会进入批量提示词。"
            "同步翻译需另外开启“同步翻译注入已发布项目摘要”。"
        )
        analysis_hint.setWordWrap(True)
        analysis_hint.setObjectName("config_hint_label")
        analysis_layout.addWidget(analysis_hint)
        self.sync_inject_published_brief_cb = QCheckBox(
            "同步翻译注入已发布项目摘要（当前项目）"
        )
        self.sync_inject_published_brief_cb.setToolTip(
            "仅注入 published 且 fingerprint 匹配的摘要；draft/stale/missing 不会静默注入。"
        )
        analysis_layout.addWidget(self.sync_inject_published_brief_cb)
        self._add_primary_fields(
            analysis_layout,
            [
                field
                for field in CONTEXT_PRIMARY_FIELDS
                if field.key in PROJECT_ANALYSIS_CONTEXT_SETTING_KEYS
            ],
        )
        analysis_advanced_btn = QPushButton("打开项目分析高级参数")
        analysis_advanced_btn.setObjectName("secondary_btn")
        analysis_advanced_btn.clicked.connect(self._open_analysis_advanced)
        self.analysis_advanced_btn = analysis_advanced_btn
        analysis_layout.addWidget(analysis_advanced_btn)
        layout.addWidget(analysis_box)
        layout.addStretch(1)
        return page, body
