"""Independent Models Settings page (#202 Phase D).

The page owns Gemini sync/batch model combos and batch thinking selection.
Catalog extras still come from the host load path; save stays on MainWindow.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QGroupBox, QLabel, QScrollArea, QWidget

from gemini_model_catalog import (
    BUILTIN_GEMINI_EMBEDDING_MODELS,
    BUILTIN_GEMINI_TRANSLATION_MODELS,
)

from ..user_copy import MODEL_ROUTING_RUNTIME_COPY
from ..widget_helpers import NoWheelComboBox
from .page_chrome import build_settings_scroll_page, settings_form
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import SETTINGS_PAGE_SPEC_OBJECTS

MODELS_PAGE_KEY = "models"
_MODELS_SPEC = next(
    spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == MODELS_PAGE_KEY
)
MODELS_NAV_LABEL = _MODELS_SPEC.nav_label
MODELS_CONFIG_KEYS = _MODELS_SPEC.config_keys
MODELS_IMMEDIATE_ACTION_IDS = _MODELS_SPEC.immediate_action_ids

MODELS_WIDGET_ATTRS: tuple[str, ...] = (
    "sync_model_combo",
    "sync_embedding_combo",
    "batch_model_combo",
    "batch_embedding_combo",
    "batch_thinking_combo",
)

MODELS_FORWARDED_ATTRS: frozenset[str] = frozenset(
    {
        "_batch_thinking_config_has_key",
        "_batch_thinking_user_changed",
        "_updating_batch_thinking_combo",
    }
)

_FIELD_WIDGETS = {
    "sync_model": "sync_model_combo",
    "sync_embedding_model": "sync_embedding_combo",
    "batch_model": "batch_model_combo",
    "batch_embedding_model": "batch_embedding_combo",
    "batch_thinking_level": "batch_thinking_combo",
}


def config_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def supports_batch_thinking(model_name: Any) -> bool:
    return config_string(model_name).startswith("gemini-3")


def batch_thinking_value_for_load(
    batch_config: Mapping[str, Any],
    batch_model: Any,
) -> str:
    if "thinking_level" in batch_config:
        return config_string(batch_config.get("thinking_level", ""))
    return "minimal" if supports_batch_thinking(batch_model) else ""


def batch_thinking_value_for_model_change(
    batch_model: Any,
    current_thinking_level: Any,
    config_has_key: bool,
    user_changed: bool,
) -> str | None:
    if (
        supports_batch_thinking(batch_model)
        and not config_string(current_thinking_level)
        and not config_has_key
        and not user_changed
    ):
        return "minimal"
    return None


def should_save_batch_thinking_level(
    batch_config: Mapping[str, Any],
    batch_model: str,
    thinking_level: str,
    user_changed: bool,
) -> bool:
    return (
        bool(thinking_level)
        or (supports_batch_thinking(batch_model) and user_changed)
        or "thinking_level" in batch_config
    )


class ModelsSettingsPage(QObject):
    """Settings page for Gemini sync/batch model selection."""

    page_key = MODELS_PAGE_KEY
    nav_label = MODELS_NAV_LABEL
    config_keys = MODELS_CONFIG_KEYS
    immediate_action_ids = MODELS_IMMEDIATE_ACTION_IDS

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
        self._gemini_sync_allowed = True
        self._baseline: dict[str, object] = {}
        self._batch_thinking_config_has_key = False
        self._batch_thinking_user_changed = False
        self._updating_batch_thinking_combo = False
        self.widget, self.body = self._build_widgets()
        self._baseline = dict(self.collect())

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self._actions = actions

    def attach_widget_aliases(self, host_obj: object) -> None:
        target = getattr(host_obj, "__dict__", None)
        if not isinstance(target, dict):
            return
        for name in MODELS_WIDGET_ATTRS:
            target[name] = getattr(self, name)

    def set_catalog(
        self,
        translation_models: Sequence[str],
        embedding_models: Sequence[str],
    ) -> None:
        translation = [config_string(model) for model in translation_models if config_string(model)]
        embedding = [config_string(model) for model in embedding_models if config_string(model)]
        self._repopulate_model_combo(
            self.sync_model_combo,
            translation,
            config_string(self.sync_model_combo.currentText()),
        )
        self._repopulate_model_combo(
            self.batch_model_combo,
            translation,
            config_string(self.batch_model_combo.currentText()),
        )
        self._repopulate_model_combo(
            self.sync_embedding_combo,
            embedding,
            config_string(self.sync_embedding_combo.currentText()),
        )
        self._repopulate_model_combo(
            self.batch_embedding_combo,
            embedding,
            config_string(self.batch_embedding_combo.currentText()),
        )

    def load(self, snapshot: Mapping[str, object], *, restore: bool = False) -> None:
        previous = self._loading
        self._loading = True
        try:
            if "sync_model" in snapshot:
                self._set_combo_value(
                    self.sync_model_combo, snapshot.get("sync_model", "")
                )
            if "batch_model" in snapshot:
                self._set_combo_value(
                    self.batch_model_combo, snapshot.get("batch_model", "")
                )
            if "sync_embedding_model" in snapshot:
                self._set_combo_value(
                    self.sync_embedding_combo,
                    snapshot.get("sync_embedding_model", ""),
                )
            if "batch_embedding_model" in snapshot:
                self._set_combo_value(
                    self.batch_embedding_combo,
                    snapshot.get("batch_embedding_model", ""),
                )
            if "batch_model" in snapshot:
                self._on_batch_model_changed(
                    config_string(snapshot.get("batch_model", ""))
                )
            if "batch_thinking_level" in snapshot:
                previous_level = config_string(self.batch_thinking_combo.currentData())
                restored_level = config_string(snapshot.get("batch_thinking_level", ""))
                self._set_batch_thinking_value(restored_level)
                if restore and restored_level != previous_level:
                    self._batch_thinking_user_changed = True
            if not restore:
                self._batch_thinking_user_changed = False
                self._baseline = dict(self.collect())
        finally:
            self._loading = previous

    def collect(self) -> dict[str, object]:
        thinking_val = self.batch_thinking_combo.currentData()
        thinking_level = thinking_val if isinstance(thinking_val, str) else ""
        return {
            "sync_model": config_string(self.sync_model_combo.currentText()),
            "sync_embedding_model": config_string(
                self.sync_embedding_combo.currentText()
            ),
            "batch_model": config_string(self.batch_model_combo.currentText()),
            "batch_embedding_model": config_string(
                self.batch_embedding_combo.currentText()
            ),
            "batch_thinking_level": thinking_level,
        }

    def validate(self) -> Sequence[SettingsIssue]:
        return []

    def reset(self) -> None:
        if self._baseline:
            self.load(self._baseline)

    def focus_issue(self, issue: SettingsIssue) -> bool:
        attr = _FIELD_WIDGETS.get(issue.field_key)
        widget = getattr(self, attr, None) if attr else None
        if widget is not None and hasattr(widget, "setFocus"):
            widget.setFocus()
            return True
        return False

    def set_task_running(self, running: bool) -> None:
        self._task_running = bool(running)
        self._apply_combo_enabled_state()

    def set_gemini_sync_allowed(self, allowed: bool, tooltip: str = "") -> None:
        """Host LiteLLM gating: Gemini sync model is unusable while LiteLLM is selected."""
        self._gemini_sync_allowed = bool(allowed)
        self.sync_model_combo.setToolTip(tooltip)
        self._apply_combo_enabled_state()

    def _apply_combo_enabled_state(self) -> None:
        idle = not self._task_running
        self.sync_model_combo.setEnabled(idle and self._gemini_sync_allowed)
        self.sync_embedding_combo.setEnabled(idle)
        self.batch_model_combo.setEnabled(idle)
        self.batch_embedding_combo.setEnabled(idle)
        self.batch_thinking_combo.setEnabled(
            idle and supports_batch_thinking(self.batch_model_combo.currentText())
        )

    def _on_batch_model_changed(self, text: str) -> None:
        supported = supports_batch_thinking(text)
        if not supported:
            self._set_batch_thinking_value("")
        else:
            default_value = batch_thinking_value_for_model_change(
                text,
                self.batch_thinking_combo.currentData(),
                self._batch_thinking_config_has_key,
                self._batch_thinking_user_changed,
            )
            if default_value is not None and not self._loading:
                self._set_batch_thinking_value(default_value)
        self._apply_combo_enabled_state()

    def _on_batch_thinking_changed(self, _index: int) -> None:
        if not self._loading and not self._updating_batch_thinking_combo:
            self._batch_thinking_user_changed = True

    def _set_batch_thinking_value(self, value: str) -> None:
        idx = self.batch_thinking_combo.findData(value)
        self._updating_batch_thinking_combo = True
        try:
            if idx >= 0:
                self.batch_thinking_combo.setCurrentIndex(idx)
            elif value:
                self.batch_thinking_combo.addItem(f"{value} (自定义)", value)
                self.batch_thinking_combo.setCurrentIndex(
                    self.batch_thinking_combo.count() - 1
                )
            else:
                self.batch_thinking_combo.setCurrentIndex(0)
        finally:
            self._updating_batch_thinking_combo = False

    def _repopulate_model_combo(
        self,
        combo: NoWheelComboBox,
        models: Sequence[str],
        selected: str,
    ) -> None:
        selected = config_string(selected)
        previous = combo.blockSignals(True)
        try:
            combo.clear()
            if models:
                combo.addItems(list(models))
            if selected:
                self._set_combo_value(combo, selected)
            elif combo.count() > 0:
                combo.setCurrentIndex(0)
            else:
                combo.setCurrentIndex(-1)
        finally:
            combo.blockSignals(previous)

    def _set_combo_value(self, combo: NoWheelComboBox, value: object) -> None:
        text = config_string(value)
        if not text:
            combo.setCurrentIndex(-1)
            return
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return
        combo.addItem(text)
        combo.setCurrentIndex(combo.count() - 1)

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = build_settings_scroll_page("settings_models")
        routing_hint = QLabel(MODEL_ROUTING_RUNTIME_COPY["settings_hint"])
        routing_hint.setWordWrap(True)
        layout.addWidget(routing_hint)

        sync_box = QGroupBox("Gemini 同步翻译")
        sync_layout = settings_form(sync_box)
        self.sync_model_combo = NoWheelComboBox()
        self.sync_model_combo.setEditable(False)
        self.sync_model_combo.addItems(list(BUILTIN_GEMINI_TRANSLATION_MODELS))
        sync_layout.addRow("翻译模型：", self.sync_model_combo)
        self.sync_embedding_combo = NoWheelComboBox()
        self.sync_embedding_combo.setEditable(False)
        self.sync_embedding_combo.addItems(list(BUILTIN_GEMINI_EMBEDDING_MODELS))
        sync_layout.addRow("RAG 向量模型：", self.sync_embedding_combo)
        sync_hint = QLabel(
            "此处只配置 Gemini 同步/批量所用模型，从下拉列表选择（不可手输）。"
            "若要增加自定义模型 ID，请到「设置 → 高级 → 模型目录」。"
            "LiteLLM 已移至左侧独立页面。"
        )
        sync_hint.setWordWrap(True)
        sync_hint.setObjectName("config_hint_label")
        sync_layout.addRow(sync_hint)
        layout.addWidget(sync_box)

        batch_box = QGroupBox("批量离线翻译")
        batch_layout = settings_form(batch_box)
        self.batch_model_combo = NoWheelComboBox()
        self.batch_model_combo.setEditable(False)
        self.batch_model_combo.addItems(list(BUILTIN_GEMINI_TRANSLATION_MODELS))
        batch_layout.addRow("翻译模型：", self.batch_model_combo)
        self.batch_embedding_combo = NoWheelComboBox()
        self.batch_embedding_combo.setEditable(False)
        self.batch_embedding_combo.addItems(list(BUILTIN_GEMINI_EMBEDDING_MODELS))
        batch_layout.addRow("RAG 向量模型：", self.batch_embedding_combo)
        self.batch_thinking_combo = NoWheelComboBox()
        self.batch_thinking_combo.addItem("（不启用）", "")
        self.batch_thinking_combo.addItem("最小", "minimal")
        self.batch_thinking_combo.addItem("低", "low")
        self.batch_thinking_combo.addItem("中", "medium")
        self.batch_thinking_combo.addItem("高", "high")
        batch_layout.addRow("思考程度：", self.batch_thinking_combo)
        layout.addWidget(batch_box)
        layout.addStretch(1)

        self.batch_model_combo.currentTextChanged.connect(self._on_batch_model_changed)
        self.batch_thinking_combo.currentIndexChanged.connect(
            self._on_batch_thinking_changed
        )
        return page, body
