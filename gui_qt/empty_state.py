"""Centered empty-state placeholder for tabs that have no data yet."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import system_prefers_dark
from .theme_helpers import DEFAULT_THEME_PREFERENCE, THEME_DARK, THEME_LIGHT, resolve_effective_theme
from .theme_tokens import tokens_for_theme


def _effective_theme_name(widget: QWidget) -> str:
    """Resolve the app's effective theme instead of inferring from QPalette."""
    current: QWidget | None = widget
    while current is not None:
        is_dark = getattr(current, "_effective_theme_is_dark", None)
        if callable(is_dark):
            return THEME_DARK if is_dark() else THEME_LIGHT
        preference = getattr(current, "_theme_preference", None)
        if preference is not None:
            qt_app = getattr(current, "_qt_app", None)
            system_is_dark = system_prefers_dark(qt_app) if qt_app is not None else None
            return resolve_effective_theme(preference, system_is_dark=system_is_dark)
        current = current.parentWidget()

    app = QApplication.instance()
    system_is_dark = system_prefers_dark(app) if app is not None else None
    return resolve_effective_theme(DEFAULT_THEME_PREFERENCE, system_is_dark=system_is_dark)


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
        # Compact margins so CTA fits short workbench status tabs.
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ---- icon --------------------------------------------------------
        self._icon_label = QLabel(icon)
        icon_font = QFont()
        icon_font.setPixelSize(32)
        self._icon_label.setFont(icon_font)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )

        # ---- title -------------------------------------------------------
        self._title_label = QLabel(title)
        title_font = QFont()
        title_font.setPixelSize(15)
        title_font.setWeight(QFont.Weight.DemiBold)
        self._title_label.setFont(title_font)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setWordWrap(True)
        self._title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )

        # ---- description -------------------------------------------------
        self._desc_label = QLabel(description)
        desc_font = QFont()
        desc_font.setPixelSize(12)
        self._desc_label.setFont(desc_font)
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._desc_label.setWordWrap(True)
        self._desc_label.setMaximumWidth(420)
        self._desc_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )

        # ---- optional action button --------------------------------------
        self._action_btn: QPushButton | None = None
        if action_text is not None:
            self._action_btn = QPushButton(action_text)
            self._action_btn.setObjectName("secondary_btn")
            self._action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._action_btn.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            # Keep full label visible even when parent is height-constrained.
            hint = self._action_btn.sizeHint()
            self._action_btn.setMinimumSize(hint)
            self._action_btn.clicked.connect(self.action_clicked)

        # ---- assemble: stretch centers content; CTA stays after description ----
        outer.addStretch(1)
        outer.addWidget(self._icon_label)
        outer.addWidget(self._title_label)
        outer.addWidget(self._desc_label, alignment=Qt.AlignmentFlag.AlignCenter)

        if self._action_btn is not None:
            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(0, 0, 0, 0)
            btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn_row.addWidget(self._action_btn)
            outer.addSpacing(2)
            outer.addLayout(btn_row)
        outer.addStretch(1)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    # ------------------------------------------------------------------
    # Theme-aware colours
    # ------------------------------------------------------------------

    def _apply_theme_colors(self) -> None:
        """Set foreground colours from the app's effective theme tokens."""
        tokens = tokens_for_theme(_effective_theme_name(self))
        icon_color = tokens["fg_muted"]
        title_color = tokens["fg_secondary"]
        desc_color = tokens["fg_muted"]

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
