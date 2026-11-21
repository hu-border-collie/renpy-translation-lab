"""Gemini catalog extras and rotation-checklist widgets (#202 Phase D).

These editors used to live on MainWindow. The Advanced Settings page owns
them after migration; host load/collect helpers reuse the same functions.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from providers.gemini_model_catalog import (
    allowed_gemini_rotation_models,
    extras_beyond_builtins,
    normalize_model_names,
)

StatusCallback = Callable[[str], None]
ChangeCallback = Callable[[], None]
_INVALID_VALUE_PROP = "_rtl_invalid_raw_value"
_HAS_INVALID_VALUE_PROP = "_rtl_has_invalid_raw_value"


def _has_invalid_raw(widget: QWidget) -> bool:
    return bool(widget.property(_HAS_INVALID_VALUE_PROP))


def _set_invalid_raw(widget: QWidget, value: object) -> None:
    widget.setProperty(_HAS_INVALID_VALUE_PROP, True)
    widget.setProperty(_INVALID_VALUE_PROP, value)
    clear_btn = getattr(widget, "_catalog_clear_invalid_btn", None)
    if clear_btn is not None:
        clear_btn.setVisible(True)


def _clear_invalid_raw(widget: QWidget) -> None:
    widget.setProperty(_HAS_INVALID_VALUE_PROP, False)
    widget.setProperty(_INVALID_VALUE_PROP, None)
    clear_btn = getattr(widget, "_catalog_clear_invalid_btn", None)
    if clear_btn is not None:
        clear_btn.setVisible(False)


def create_gemini_catalog_list_editor(
    *,
    kind: str,
    on_status: StatusCallback | None = None,
    on_change: ChangeCallback | None = None,
) -> QWidget:
    """Compact custom-model editor: short list + single-line add/remove."""
    host = QWidget()
    host.setObjectName(f"gemini_catalog_editor_{kind}")
    host.setProperty("catalog_kind", kind)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    model_list = QListWidget()
    model_list.setObjectName(f"gemini_catalog_list_{kind}")
    model_list.setAlternatingRowColors(True)
    model_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    model_list.setMaximumHeight(110)
    model_list.setMinimumHeight(72)
    layout.addWidget(model_list)

    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    entry = QLineEdit()
    entry.setObjectName(f"gemini_catalog_entry_{kind}")
    entry.setClearButtonEnabled(True)
    entry.setPlaceholderText(
        "例如 gemini-embedding-custom"
        if kind == "embedding"
        else "例如 gemini-experimental-foo"
    )
    row_layout.addWidget(entry, 1)

    add_btn = QPushButton("添加")
    add_btn.setObjectName("secondary_btn")
    add_btn.setToolTip("将上方输入的模型 ID 加入列表（仅非内置 ID）。")
    row_layout.addWidget(add_btn)

    remove_btn = QPushButton("删除所选")
    remove_btn.setObjectName("secondary_btn")
    remove_btn.setToolTip("删除列表中选中的自定义模型。")
    row_layout.addWidget(remove_btn)

    clear_invalid_btn = QPushButton("清空无效值")
    clear_invalid_btn.setObjectName("secondary_btn")
    clear_invalid_btn.setToolTip("清除当前无效配置并恢复为空列表。")
    clear_invalid_btn.setVisible(False)
    row_layout.addWidget(clear_invalid_btn)
    layout.addWidget(row)

    def _emit_status(message: str) -> None:
        if callable(on_status):
            on_status(message)

    def _notify_changed() -> None:
        if callable(on_change):
            on_change()

    def _add_model() -> None:
        name = entry.text().strip()
        if not name:
            return
        extras = extras_beyond_builtins([name], kind=kind)
        if not extras:
            _emit_status("内置模型无需添加，已在「设置 → 模型」中可选。")
            entry.clear()
            return
        cleaned = extras[0]
        existing = {
            model_list.item(i).text().strip()
            for i in range(model_list.count())
            if model_list.item(i) is not None
        }
        if cleaned in existing:
            _emit_status(f"已在列表中：{cleaned}")
            entry.clear()
            return
        _clear_invalid_raw(host)
        model_list.addItem(cleaned)
        entry.clear()
        model_list.setCurrentRow(model_list.count() - 1)
        _notify_changed()

    def _remove_selected() -> None:
        selected_items = model_list.selectedItems()
        if not selected_items:
            return
        _clear_invalid_raw(host)
        for item in selected_items:
            row_index = model_list.row(item)
            model_list.takeItem(row_index)
        _notify_changed()

    def _clear_invalid() -> None:
        _clear_invalid_raw(host)
        _emit_status("已清空无效模型目录配置。")
        _notify_changed()

    add_btn.clicked.connect(_add_model)
    entry.returnPressed.connect(_add_model)
    remove_btn.clicked.connect(_remove_selected)
    clear_invalid_btn.clicked.connect(_clear_invalid)
    host._catalog_list = model_list  # type: ignore[attr-defined]
    host._catalog_entry = entry  # type: ignore[attr-defined]
    host._catalog_add_btn = add_btn  # type: ignore[attr-defined]
    host._catalog_remove_btn = remove_btn  # type: ignore[attr-defined]
    host._catalog_clear_invalid_btn = clear_invalid_btn  # type: ignore[attr-defined]
    return host


def gemini_catalog_list_values(widget: QWidget) -> object:
    if _has_invalid_raw(widget):
        return widget.property(_INVALID_VALUE_PROP)
    model_list = getattr(widget, "_catalog_list", None)
    if not isinstance(model_list, QListWidget):
        return []
    values: list[str] = []
    for index in range(model_list.count()):
        item = model_list.item(index)
        if item is None:
            continue
        text = item.text().strip()
        if text and text not in values:
            values.append(text)
    return values


def set_gemini_catalog_list_values(widget: QWidget, values: object) -> None:
    model_list = getattr(widget, "_catalog_list", None)
    if not isinstance(model_list, QListWidget):
        return
    kind = str(widget.property("catalog_kind") or "translation")
    raw_items: object = values
    invalid = False
    if isinstance(values, str):
        text = values.strip()
        if text.startswith("["):
            try:
                raw_items = json.loads(text)
            except json.JSONDecodeError:
                invalid = True
        elif text:
            raw_items = [part.strip() for line in text.splitlines() for part in line.split(",")]
        else:
            raw_items = []
    if not invalid and raw_items is not None and not isinstance(raw_items, (list, tuple, set)):
        invalid = True
    if invalid:
        _set_invalid_raw(widget, copy.deepcopy(values))
        cleaned: list[str] = []
    else:
        _clear_invalid_raw(widget)
        cleaned = extras_beyond_builtins(raw_items, kind=kind)
    previous = model_list.blockSignals(True)
    try:
        model_list.clear()
        for name in cleaned:
            model_list.addItem(name)
    finally:
        model_list.blockSignals(previous)


def create_gemini_model_checklist(
    *,
    models: Sequence[str] | None = None,
    selected: object | None = None,
) -> QListWidget:
    """Multi-select checklist limited to known Gemini translation models."""
    widget = QListWidget()
    widget.setObjectName("model_rotation_models_list")
    widget.setMinimumHeight(160)
    widget.setMaximumHeight(240)
    widget.setAlternatingRowColors(True)
    refresh_gemini_model_checklist(
        widget,
        models=list(models)
        if models is not None
        else allowed_gemini_rotation_models(),
        selected=selected,
    )
    widget.itemChanged.connect(lambda _item, w=widget: _clear_invalid_raw(w))
    return widget


def refresh_gemini_model_checklist(
    widget: QListWidget,
    *,
    models: Sequence[str],
    selected: object | None = None,
) -> None:
    """Rebuild checklist rows from the supplied Gemini catalog."""
    previous_block = widget.blockSignals(True)
    try:
        widget.clear()
        selected_set = set(normalize_model_names(selected))
        for name in models:
            item = QListWidgetItem(name)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if name in selected_set
                else Qt.CheckState.Unchecked
            )
            widget.addItem(item)
    finally:
        widget.blockSignals(previous_block)


def gemini_model_checklist_values(widget: QListWidget) -> object:
    if _has_invalid_raw(widget):
        return widget.property(_INVALID_VALUE_PROP)
    selected: list[str] = []
    for index in range(widget.count()):
        item = widget.item(index)
        if item is None:
            continue
        if item.checkState() == Qt.CheckState.Checked:
            text = item.text().strip()
            if text:
                selected.append(text)
    return selected


def set_gemini_model_checklist_values(
    widget: QListWidget,
    values: object,
) -> None:
    allowed_names = {
        widget.item(index).text().strip()
        for index in range(widget.count())
        if widget.item(index) is not None and widget.item(index).text().strip()
    }
    invalid = False
    if values is None:
        selected_items: list[str] = []
    elif not isinstance(values, (str, list, tuple, set)):
        invalid = True
        selected_items = []
    else:
        selected_items = normalize_model_names(values)
        if any(name not in allowed_names for name in selected_items):
            invalid = True
    selected = set(selected_items)
    previous_block = widget.blockSignals(True)
    try:
        for index in range(widget.count()):
            item = widget.item(index)
            if item is None:
                continue
            item.setCheckState(
                Qt.CheckState.Checked
                if item.text().strip() in selected
                else Qt.CheckState.Unchecked
            )
    finally:
        widget.blockSignals(previous_block)
    if invalid:
        _set_invalid_raw(widget, copy.deepcopy(values))
    else:
        _clear_invalid_raw(widget)
