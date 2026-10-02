"""Content sizing for the always-visible Settings category selector."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, QTimer
from PySide6.QtWidgets import QStyle, QStyleOptionComboBox

from ..widget_helpers import NoWheelComboBox


class SettingsCategoryComboBox(NoWheelComboBox):
    """Keep category labels readable using the active font and Qt style.

    Native Windows combo size hints can be smaller than the styled text's
    required width. Recompute that width instead of relying on a cached hint
    or a fixed pixel allowance, including after font/theme changes.
    """

    def event(self, event: QEvent) -> bool:
        result = super().event(event)
        if event.type() in {
            QEvent.Type.FontChange, QEvent.Type.StyleChange, QEvent.Type.Polish,
        }:
            # Query the style after repolishing finishes, outside Qt's native
            # size-hint callbacks. The context cancels callbacks on deletion.
            QTimer.singleShot(0, self, self.refresh_minimum_width)
        return result

    def refresh_minimum_width(self) -> None:
        """Include every category label and the style's frame/drop-down space."""
        if not self.count():
            return
        metrics = self.fontMetrics()
        longest = max(
            (self.itemText(index) for index in range(self.count())),
            key=metrics.horizontalAdvance,
        )
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = longest
        styled = self.style().sizeFromContents(
            QStyle.ContentsType.CT_ComboBox,
            option,
            QSize(metrics.horizontalAdvance(longest), metrics.height()),
            self,
        )
        self.setMinimumWidth(max(super().sizeHint().width(), styled.width()))
