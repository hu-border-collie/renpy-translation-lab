"""Centered empty-state placeholder for tabs that have no data yet."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPalette
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _is_dark_palette(widget: QWidget) -> bool:
    """Return *True* when the widget's palette looks like a dark theme.

    Heuristic: if the window-background lightness is below 50 % the
    palette is considered "dark".
    """
    bg = widget.palette().color(QPalette.ColorRole.Window)
    return bg.lightnessF() < 0.5


class EmptyStateWidget(QWidget):
    """A full-area placeholder shown when a tab has nothing to display.

    Parameters
    ----------
    icon:
        A single Unicode emoji / symbol rendered at 48 px.
    title:
        Short headline (16 px, semi-bold).
    description:
        Explanatory body text (13 px, word-wrapped, max 400 px wide).
    action_text:
        If supplied, a secondary-style ``QPushButton`` is added below the
        description and :pyattr:`action_clicked` is emitted on click.
    parent:
        Optional parent widget.
    """

    action_clicked = Signal()
    """Emitted when the optional action button is clicked."""

    def __init__(
        self,
        icon: str,
        title: str,
        description: str,
        *,
        action_text: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        # ---- outer layout (centers the content block) --------------------
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ---- icon --------------------------------------------------------
        self._icon_label = QLabel(icon)
        icon_font = QFont()
        icon_font.setPixelSize(48)
        self._icon_label.setFont(icon_font)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ---- title -------------------------------------------------------
        self._title_label = QLabel(title)
        title_font = QFont()
        title_font.setPixelSize(16)
        title_font.setWeight(QFont.Weight.DemiBold)
        self._title_label.setFont(title_font)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ---- description -------------------------------------------------
        self._desc_label = QLabel(description)
        desc_font = QFont()
        desc_font.setPixelSize(13)
        self._desc_label.setFont(desc_font)
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._desc_label.setWordWrap(True)
        self._desc_label.setMaximumWidth(400)

        # ---- optional action button --------------------------------------
        self._action_btn: QPushButton | None = None
        if action_text is not None:
            self._action_btn = QPushButton(action_text)
            self._action_btn.setObjectName("secondary_btn")
            self._action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._action_btn.clicked.connect(self.action_clicked)

        # ---- assemble ----------------------------------------------------
        outer.addWidget(self._icon_label)
        outer.addWidget(self._title_label)
        outer.addWidget(self._desc_label, alignment=Qt.AlignmentFlag.AlignCenter)

        if self._action_btn is not None:
            btn_row = QHBoxLayout()
            btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn_row.addWidget(self._action_btn)
            outer.addSpacing(8)
            outer.addLayout(btn_row)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    # ------------------------------------------------------------------
    # Theme-aware colours
    # ------------------------------------------------------------------

    def _apply_theme_colors(self) -> None:
        """Set foreground colours based on the current palette brightness."""
        dark = _is_dark_palette(self)
        icon_color = "#9ca3af" if dark else "#64748b"
        title_color = "#d1d5db" if dark else "#475569"
        desc_color = "#9ca3af" if dark else "#64748b"

        self._icon_label.setStyleSheet(f"color: {icon_color}; background: transparent;")
        self._title_label.setStyleSheet(f"color: {title_color}; background: transparent;")
        self._desc_label.setStyleSheet(f"color: {desc_color}; background: transparent;")

    def showEvent(self, event):  # noqa: N802 – Qt naming convention
        """Re-apply theme colours every time the widget becomes visible."""
        super().showEvent(event)
        self._apply_theme_colors()

    def changeEvent(self, event):  # noqa: N802 – Qt naming convention
        """React to palette / style changes at runtime."""
        super().changeEvent(event)
        if event.type() in (
            event.Type.PaletteChange,
            event.Type.StyleChange,
        ):
            self._apply_theme_colors()
