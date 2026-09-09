"""Gemini catalog extras and rotation-checklist widgets (#202 Phase D).

These editors used to live on MainWindow. The Advanced Settings page owns
them after migration; host load/collect helpers reuse the same functions.
"""
from __future__ import annotations

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

from gemini_model_catalog import allowed_gemini_rotation_models, extras_beyond_builtins

StatusCallback = Callable[[str], None]


def create_gemini_catalog_list_editor(
    *,
    kind: str,
    on_status: StatusCallback | None = None,
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
    layout.addWidget(row)

    def _emit_status(message: str) -> None:
        if callable(on_status):
            on_status(message)

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
        model_list.addItem(cleaned)
        entry.clear()
        model_list.setCurrentRow(model_list.count() - 1)

    def _remove_selected() -> None:
        for item in model_list.selectedItems():
            row_index = model_list.row(item)
            model_list.takeItem(row_index)

    add_btn.clicked.connect(_add_model)
    entry.returnPressed.connect(_add_model)
    remove_btn.clicked.connect(_remove_selected)
    host._catalog_list = model_list  # type: ignore[attr-defined]
    host._catalog_entry = entry  # type: ignore[attr-defined]
    return host


def gemini_catalog_list_values(widget: QWidget) -> list[str]:
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
    cleaned = extras_beyond_builtins(values, kind=kind)
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
        selected_set = {
            str(item).strip()
            for item in (selected if isinstance(selected, (list, tuple, set)) else [])
            if str(item).strip()
        }
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


def gemini_model_checklist_values(widget: QListWidget) -> list[str]:
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
    selected = {
        str(item).strip()
        for item in (values if isinstance(values, (list, tuple, set)) else [])
        if str(item).strip()
    }
    for index in range(widget.count()):
        item = widget.item(index)
        if item is None:
            continue
        item.setCheckState(
            Qt.CheckState.Checked
            if item.text().strip() in selected
            else Qt.CheckState.Unchecked
        )
