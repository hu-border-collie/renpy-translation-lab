"""Small Qt widget subclasses for safer desktop UX."""
from __future__ import annotations

from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QComboBox, QTabWidget


class NoWheelComboBox(QComboBox):
    """Ignore mouse-wheel changes unless the combo already has keyboard focus."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class NoWheelTabWidget(QTabWidget):
    """Ignore mouse-wheel tab switching; use clicks to change tabs."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()