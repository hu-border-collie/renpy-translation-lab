"""Basic SettingField widgets shared by migrated Settings pages (#202 Phase D).

Specialized Gemini catalog/checklist widgets stay on MainWindow until those
pages migrate. This module only covers bool/int/float/str/text/list/json.
"""
from __future__ import annotations

import json
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..settings_schema import SettingField

SettingValue = Any


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
        widget: QWidget = QCheckBox()
    elif field.kind == "int":
        spin = QSpinBox()
        spin.setAccelerated(True)
        minimum = int(field.minimum if field.minimum is not None else 0)
        maximum = int(field.maximum if field.maximum is not None else 9999999)
        spin.setRange(minimum, maximum)
        widget = spin
    elif field.kind == "float":
        spin_f = QDoubleSpinBox()
        spin_f.setDecimals(3)
        spin_f.setSingleStep(0.01 if field.maximum == 1.0 else 0.1)
        minimum_f = float(field.minimum if field.minimum is not None else -999999.0)
        maximum_f = float(field.maximum if field.maximum is not None else 999999.0)
        spin_f.setRange(minimum_f, maximum_f)
        widget = spin_f
    elif field.kind in {"text", "list", "json"}:
        text = QTextEdit()
        text.setAcceptRichText(False)
        text.setMinimumHeight(72 if field.kind != "text" else 96)
        text.setPlaceholderText(setting_placeholder(field))
        widget = text
    else:
        line = QLineEdit()
        line.setClearButtonEnabled(True)
        if field.allow_empty:
            line.setPlaceholderText(
                "留空使用默认路径" if "路径" in field.label else "可留空"
            )
        widget = line
    widget.setToolTip(field.description)
    return widget


def setting_value_from_widget(field: SettingField, widget: QWidget) -> SettingValue:
    if field.kind == "bool":
        return bool(getattr(widget, "isChecked")())
    if field.kind in {"int", "float"}:
        return getattr(widget, "value")()
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
    if field.kind == "bool":
        widget.setChecked(bool(value))
        return
    if field.kind == "int":
        widget.setValue(int(value))
        return
    if field.kind == "float":
        widget.setValue(float(value))
        return
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
    desc = QLabel(field.description)
    desc.setWordWrap(True)
    desc.setObjectName("settings_description_label")
    layout.addWidget(desc)
    error.setWordWrap(True)
    error.setObjectName("settings_error_label")
    layout.addWidget(error)
    return row
