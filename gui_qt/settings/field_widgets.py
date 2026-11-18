"""Basic SettingField widgets shared by migrated Settings pages (#202 Phase D).

Gemini catalog extras and rotation-checklist widgets live in
``gemini_catalog_widgets.py``. This module covers bool/int/float/str/text/list/json.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import json
from typing import Any

from PySide6.QtCore import QObject, QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui_qt.settings.settings_schema import SettingField, validate_value
from gui_qt.widgets.widget_helpers import NoWheelComboBox

SettingValue = Any

_MAX_INLINE_DESC_CHARS = 38
_INVALID_VALUE_PROP = "_rtl_invalid_raw_value"


def _set_invalid_raw(widget: object, value: object) -> None:
    setter = getattr(widget, "setProperty", None)
    if callable(setter):
        setter(_INVALID_VALUE_PROP, value)


def _clear_invalid_raw(widget: object) -> None:
    _set_invalid_raw(widget, None)


def _on_choice_combo_changed(field: SettingField, combo: QComboBox) -> None:
    _clear_invalid_raw(combo)
    choices = field.choices or ()
    expected = len(choices) + (1 if field.allow_empty and "" not in choices else 0)
    if 0 <= combo.currentIndex() < expected and combo.count() > expected:
        with QSignalBlocker(combo):
            while combo.count() > expected:
                combo.removeItem(combo.count() - 1)


def _populate_choice_combo(
    field: SettingField,
    combo: QComboBox,
    value: object,
    *,
    is_valid: bool,
) -> None:
    choices = field.choices or ()
    combo.clear()
    for choice in choices:
        combo.addItem(choice, choice)
    if field.allow_empty and "" not in choices:
        combo.addItem("", "")
    if is_valid:
        target = value.strip() if isinstance(value, str) else value
        index = combo.findData(target)
        if index < 0 and target is not None:
            index = combo.findText(str(target))
        if index >= 0:
            combo.setCurrentIndex(index)
        _clear_invalid_raw(combo)
        return
    display = "" if value is None else str(value)
    combo.addItem(display, value)
    combo.setCurrentIndex(combo.count() - 1)
    _set_invalid_raw(combo, value)


def update_setting_error_label(
    field: SettingField,
    label: Any,
    value: object,
    *,
    translator_config: Mapping[str, Any] | None = None,
) -> str:
    """Validate *value* for *field* and reflect any error on *label*."""
    error = validate_value(field, value, translator_config=translator_config)
    if label is not None:
        if error and field.kind in {"bool", "int", "float"}:
            label.setText(f"{error}（当前配置值：{value!r}）")
        else:
            label.setText(error)
    return error


def _concise_field_summary(description: str) -> str:
    """Return a concise single-clause summary for inline form descriptions.

    Full multi-sentence explanations stay in the widget and label tooltips so
    settings forms remain scannable without losing technical detail.
    """
    cleaned = " ".join(str(description or "").split())
    if len(cleaned) <= _MAX_INLINE_DESC_CHARS:
        return cleaned
    for sep in ("。", "；"):
        if sep in cleaned:
            first = cleaned.split(sep, 1)[0].strip()
            if first:
                return f"{first}。"
    return cleaned


def setting_placeholder(field: SettingField) -> str:
    if field.kind == "list":
        return "每行一个值，或填写 JSON 数组"
    if field.kind == "json":
        return "留空使用默认；可填写字符串预设或 JSON"
    if field.kind == "text":
        return "可留空"
    return ""


def format_setting_text(field: SettingField, value: object) -> str:
    if field.kind in {"list", "gemini_model_list", "gemini_catalog_list"}:
        if isinstance(value, (list, tuple, set)):
            return "\n".join(str(item) for item in value)
        return str(value or "")
    if field.kind == "json":
        if value in (None, ""):
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)
    return str(value or "")


def create_basic_setting_widget(field: SettingField) -> QWidget:
    if field.kind == "bool":
        checkbox = QCheckBox()
        checkbox.toggled.connect(lambda _checked=False, w=checkbox: _clear_invalid_raw(w))
        widget: QWidget = checkbox
    elif field.kind == "int":
        spin = QSpinBox()
        spin.setAccelerated(True)
        minimum = int(field.minimum if field.minimum is not None else 0)
        maximum = int(field.maximum if field.maximum is not None else 9999999)
        spin.setRange(minimum, maximum)
        spin.valueChanged.connect(lambda _val=0, w=spin: _clear_invalid_raw(w))
        if spin.lineEdit() is not None:
            spin.lineEdit().textEdited.connect(lambda _txt="", w=spin: _clear_invalid_raw(w))
        widget = spin
    elif field.kind == "float":
        spin_f = QDoubleSpinBox()
        spin_f.setDecimals(3)
        spin_f.setSingleStep(0.01 if field.maximum == 1.0 else 0.1)
        minimum_f = float(field.minimum if field.minimum is not None else -999999.0)
        maximum_f = float(field.maximum if field.maximum is not None else 999999.0)
        spin_f.setRange(minimum_f, maximum_f)
        spin_f.valueChanged.connect(lambda _val=0.0, w=spin_f: _clear_invalid_raw(w))
        if spin_f.lineEdit() is not None:
            spin_f.lineEdit().textEdited.connect(lambda _txt="", w=spin_f: _clear_invalid_raw(w))
        widget = spin_f
    elif field.kind in {"text", "list", "json"}:
        text = QTextEdit()
        text.setAcceptRichText(False)
        text.setMinimumHeight(72 if field.kind != "text" else 96)
        text.setPlaceholderText(setting_placeholder(field))
        text.textChanged.connect(lambda w=text: _clear_invalid_raw(w))
        widget = text
    elif field.kind == "str" and field.choices is not None:
        combo = NoWheelComboBox()
        combo.currentIndexChanged.connect(
            lambda _idx=0, f=field, c=combo: _on_choice_combo_changed(f, c)
        )
        widget = combo
    else:
        line = QLineEdit()
        line.setClearButtonEnabled(True)
        if field.allow_empty:
            line.setPlaceholderText(
                "留空使用默认路径" if "路径" in field.label else "可留空"
            )
        line.textChanged.connect(lambda _txt="", w=line: _clear_invalid_raw(w))
        widget = line
    apply_setting_value_to_widget(field, widget, field.default)
    widget.setToolTip(field.description)
    return widget


def setting_value_from_widget(field: SettingField, widget: QWidget) -> SettingValue:
    prop_fn = getattr(widget, "property", None)
    if callable(prop_fn):
        invalid = prop_fn(_INVALID_VALUE_PROP)
        if invalid is not None:
            return invalid
    if field.kind == "bool":
        return bool(getattr(widget, "isChecked")())
    if field.kind in {"int", "float"}:
        return getattr(widget, "value")()
    if isinstance(widget, QComboBox) or callable(getattr(widget, "currentData", None)):
        current_data = getattr(widget, "currentData", None)
        data = current_data() if callable(current_data) else None
        if data is not None:
            return data.strip() if isinstance(data, str) else data
        current_text = getattr(widget, "currentText", None)
        return str(current_text()).strip() if callable(current_text) else ""
    to_plain = getattr(widget, "toPlainText", None)
    if callable(to_plain):
        return str(to_plain()).strip()
    text = getattr(widget, "text", None)
    if callable(text):
        return str(text()).strip()
    return ""


def apply_setting_value_to_widget(
    field: SettingField,
    widget: QWidget,
    value: object,
) -> None:
    blocker = QSignalBlocker(widget) if isinstance(widget, QObject) else nullcontext()
    with blocker:
        is_valid = not validate_value(field, value)
        if isinstance(widget, QComboBox):
            _populate_choice_combo(field, widget, value, is_valid=is_valid)
            return
        if field.kind == "bool":
            _set_invalid_raw(widget, None if is_valid else value)
            widget.setChecked(bool(value) if is_valid else bool(field.default))
            return
        if field.kind == "int":
            _set_invalid_raw(widget, None if is_valid else value)
            widget.setValue(int(value) if is_valid else int(field.default))
            return
        if field.kind == "float":
            _set_invalid_raw(widget, None if is_valid else value)
            widget.setValue(float(value) if is_valid else float(field.default))
            return
        _set_invalid_raw(widget, None if is_valid else value)
        set_plain = getattr(widget, "setPlainText", None)
        if callable(set_plain):
            set_plain(format_setting_text(field, value))
            return
        set_text = getattr(widget, "setText", None)
        if callable(set_text):
            set_text("" if value is None else str(value))


def setting_field_row(
    field: SettingField,
    widget: QWidget,
    error: QLabel,
) -> QWidget:
    row = QWidget()
    layout = QVBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(widget)
    desc = QLabel(_concise_field_summary(field.description))
    desc.setWordWrap(True)
    desc.setObjectName("settings_description_label")
    desc.setToolTip(field.description)
    layout.addWidget(desc)
    error.setWordWrap(True)
    error.setObjectName("settings_error_label")
    layout.addWidget(error)
    return row
