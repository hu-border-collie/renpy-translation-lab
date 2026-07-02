"""Painted action buttons for the split-status table via QStyledItemDelegate."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QModelIndex, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QHelpEvent, QPainter
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem, QToolTip, QWidget

from .split_status_table_helpers import (
    SPLIT_ACTION_BUTTON_LABEL,
    ButtonVisualState,
    split_action_button_colors,
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

        state = self._visual_state(index)
        dark = option.palette.window().color().lightness() < 128
        bg_color, border_color, text_color = split_action_button_colors(dark=dark, state=state)
        rect = option.rect
        painter.save()
        painter.fillRect(rect, option.backgroundBrush.color())

        left, top, width, height = split_action_button_rect(float(rect.width()), float(rect.height()))
        button_rect = QRectF(rect.left() + left, rect.top() + top, width, height)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QColor(border_color))
        painter.setBrush(QColor(bg_color))
        painter.drawRoundedRect(button_rect, 6.0, 6.0)

        font = QFont(option.font)
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        painter.setPen(QColor(text_color))
        painter.drawText(button_rect, int(Qt.AlignmentFlag.AlignCenter), SPLIT_ACTION_BUTTON_LABEL)
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
                if self.parent() is not None and hasattr(self.parent(), "viewport"):
                    self.parent().viewport().update(option.rect)
                return True
            return False
        if event_type == QEvent.Type.MouseButtonRelease:
            if hasattr(event, "button") and event.button() != Qt.MouseButton.LeftButton:
                return False
            pressed = self._pressed_index
            self._pressed_index = None
            if pressed == index and self._event_hits_button(event, option):
                self.select_requested.emit(payload["manifest_path"])
                if self.parent() is not None and hasattr(self.parent(), "viewport"):
                    self.parent().viewport().update(option.rect)
                return True
            if self.parent() is not None and hasattr(self.parent(), "viewport"):
                self.parent().viewport().update(option.rect)
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

    def _visual_state(self, index: QModelIndex) -> ButtonVisualState:
        if self._pressed_index == index:
            return "pressed"
        if self._hover_index == index:
            return "hover"
        return "normal"

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

    def clear_hover_state(self) -> None:
        previous = self._hover_index
        self._hover_index = None
        if previous is None or not previous.isValid():
            return
        view = self.parent()
        if view is None or not hasattr(view, "viewport"):
            return
        view.viewport().update(view.visualRect(previous))

    def _paint_empty_cell(self, painter: QPainter, option: QStyleOptionViewItem) -> None:
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            painter.fillRect(option.rect, option.backgroundBrush)