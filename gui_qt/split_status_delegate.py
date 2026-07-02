"""Painted action buttons for the split-status table via QStyledItemDelegate."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QModelIndex, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QHelpEvent, QPainter
from PySide6.QtWidgets import (
    QPushButton,
    QStyle,
    QStyleOptionButton,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolTip,
    QWidget,
)

from .split_status_table_helpers import (
    SPLIT_ACTION_BUTTON_HEIGHT,
    SPLIT_ACTION_BUTTON_LABEL,
    SPLIT_ACTION_BUTTON_MIN_WIDTH,
    split_action_button_rect,
)

SPLIT_ACTION_DATA_ROLE = Qt.ItemDataRole.UserRole + 42


def read_split_action_payload(index: QModelIndex) -> dict[str, str] | None:
    data = index.data(SPLIT_ACTION_DATA_ROLE)
    if not isinstance(data, dict):
        return None
    manifest_path = data.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        return None
    part_label = data.get("part_label")
    if not isinstance(part_label, str):
        part_label = ""
    return {"manifest_path": manifest_path, "part_label": part_label}


class SplitStatusActionDelegate(QStyledItemDelegate):
    select_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hover_index: QModelIndex | None = None
        self._pressed_index: QModelIndex | None = None
        self._style_button: QPushButton | None = None

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        payload = read_split_action_payload(index)
        if payload is None:
            self._paint_empty_cell(painter, option)
            return

        rect = option.rect
        painter.save()
        painter.fillRect(rect, option.backgroundBrush)

        left, top, width, height = split_action_button_rect(float(rect.width()), float(rect.height()))
        button_rect = QRect(
            int(rect.left() + left),
            int(rect.top() + top),
            int(width),
            int(height),
        )
        style_button = self._ensure_style_button(option)
        btn_option = QStyleOptionButton()
        btn_option.initFrom(style_button)
        btn_option.rect = button_rect
        btn_option.text = SPLIT_ACTION_BUTTON_LABEL
        btn_option.state = QStyle.StateFlag.State_Enabled
        if self._pressed_index == index:
            btn_option.state |= QStyle.StateFlag.State_Sunken
        elif self._hover_index == index:
            btn_option.state |= QStyle.StateFlag.State_MouseOver

        widget = option.widget
        if widget is not None:
            widget.style().drawControl(
                QStyle.ControlElement.CE_PushButton,
                btn_option,
                painter,
                style_button,
            )
        painter.restore()

    def editorEvent(
        self,
        event: QEvent,
        model: Any,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> bool:
        payload = read_split_action_payload(index)
        if payload is None:
            return False

        event_type = event.type()
        if event_type == QEvent.Type.MouseMove:
            self._set_hover_index(index)
            return False
        if event_type == QEvent.Type.MouseButtonPress:
            if hasattr(event, "button") and event.button() != Qt.MouseButton.LeftButton:
                return False
            if self._event_hits_button(event, option):
                self._pressed_index = index
                self._repaint_index(option, index)
                return True
            return False
        if event_type == QEvent.Type.MouseButtonRelease:
            if hasattr(event, "button") and event.button() != Qt.MouseButton.LeftButton:
                return False
            pressed = self._pressed_index
            self._pressed_index = None
            if pressed == index and self._event_hits_button(event, option):
                self.select_requested.emit(payload["manifest_path"])
                self._repaint_index(option, index)
                return True
            self._repaint_index(option, index)
            return False
        if event_type == QEvent.Type.Leave:
            self.clear_hover_state()
            return False
        return False

    def helpEvent(
        self,
        event: QEvent,
        view: Any,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> bool:
        if not isinstance(event, QHelpEvent):
            return False
        payload = read_split_action_payload(index)
        if payload is None:
            return False
        part_label = payload.get("part_label") or payload["manifest_path"]
        QToolTip.showText(event.globalPos(), f"\u5207\u6362\u5230 {part_label}", view)
        return True

    def _ensure_style_button(self, option: QStyleOptionViewItem) -> QPushButton:
        if self._style_button is None:
            parent = option.widget if option.widget is not None else self.parent()
            self._style_button = QPushButton(SPLIT_ACTION_BUTTON_LABEL, parent)
            self._style_button.setObjectName("split_select_btn")
            self._style_button.setFixedSize(
                SPLIT_ACTION_BUTTON_MIN_WIDTH,
                SPLIT_ACTION_BUTTON_HEIGHT,
            )
            self._style_button.hide()
            self._style_button.ensurePolished()
        return self._style_button

    def _event_hits_button(self, event: QEvent, option: QStyleOptionViewItem) -> bool:
        if not hasattr(event, "position"):
            return False
        left, top, width, height = split_action_button_rect(
            float(option.rect.width()),
            float(option.rect.height()),
        )
        button_rect = QRectF(
            option.rect.left() + left,
            option.rect.top() + top,
            width,
            height,
        )
        return button_rect.contains(event.position())

    def _set_hover_index(self, index: QModelIndex) -> None:
        if self._hover_index == index:
            return
        previous = self._hover_index
        self._hover_index = index
        view = self.parent()
        if view is None or not hasattr(view, "viewport"):
            return
        viewport = view.viewport()
        if previous is not None and previous.isValid():
            viewport.update(view.visualRect(previous))
        if index.isValid():
            viewport.update(view.visualRect(index))

    def clear_pressed_state(self) -> None:
        previous = self._pressed_index
        self._pressed_index = None
        if previous is None or not previous.isValid():
            return
        view = self.parent()
        if view is None or not hasattr(view, "viewport"):
            return
        view.viewport().update(view.visualRect(previous))

    def clear_hover_state(self) -> None:
        previous = self._hover_index
        self._hover_index = None
        if previous is None or not previous.isValid():
            return
        view = self.parent()
        if view is None or not hasattr(view, "viewport"):
            return
        view.viewport().update(view.visualRect(previous))

    def _repaint_index(self, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        widget = option.widget
        if widget is None or not hasattr(widget, "viewport"):
            return
        widget.viewport().update(widget.visualRect(index))

    def _paint_empty_cell(self, painter: QPainter, option: QStyleOptionViewItem) -> None:
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            painter.fillRect(option.rect, option.backgroundBrush)