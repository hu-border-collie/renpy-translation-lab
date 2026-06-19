"""Small Qt widget subclasses for safer desktop UX."""
from __future__ import annotations

from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QComboBox, QTabWidget


class NoWheelComboBox(QComboBox):
    """Ignore mouse-wheel selection changes unless the dropdown list is open."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        popup = self.view()
        if popup is not None and popup.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelTabWidget(QTabWidget):
    """Ignore mouse-wheel tab switching; use clicks to change tabs."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()
        return
