"""Small Qt widget subclasses for safer desktop UX."""
from __future__ import annotations

from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QComboBox, QLineEdit, QStyle, QTabWidget


class NoWheelComboBox(QComboBox):
    """Ignore mouse-wheel selection changes unless the dropdown list is open."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        popup = self.view()
        if popup is not None and popup.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


def add_editable_combo_popup_action(combo: QComboBox) -> None:
    """Give an editable combo an explicit, visible way to open its item list."""
    line_edit = combo.lineEdit()
    if not combo.isEditable() or line_edit is None:
        raise ValueError("popup actions require an editable QComboBox")
    if line_edit.property("popup_action_installed"):
        return
    action = line_edit.addAction(
        combo.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown),
        QLineEdit.ActionPosition.TrailingPosition,
    )
    action.setObjectName("combo_popup_action")
    action.setToolTip("选择模型")
    action.triggered.connect(combo.showPopup)
    line_edit.setProperty("popup_action_installed", True)
    combo._popup_action = action


class NoWheelTabWidget(QTabWidget):
    """Ignore mouse-wheel tab switching; use clicks to change tabs."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()
        return
