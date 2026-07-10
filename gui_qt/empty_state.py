"""Centered empty-state placeholder for tabs that have no data yet."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPalette, QResizeEvent
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

_DESC_MAX_WIDTH = 360


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
        A single Unicode emoji / symbol rendered large.
    title:
        Short headline (semi-bold).
    description:
        Explanatory body text (word-wrapped, fixed max width so multi-line
        height is stable — avoids QLabel wrap height collapse that paints
        Chinese lines on top of each other).
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

        # Outer layout only centers a solid content block — never stretch the
        # labels themselves, or word-wrapped Chinese collapses to one line.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)

        self._content = QWidget()
        self._content.setObjectName("empty_state_content")
        self._content.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

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
        self._title_label.setFixedWidth(_DESC_MAX_WIDTH)
        self._title_label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Preferred,
        )

        # ---- description -------------------------------------------------
        self._desc_label = QLabel(description)
        desc_font = QFont()
        desc_font.setPixelSize(12)
        self._desc_label.setFont(desc_font)
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._desc_label.setWordWrap(True)
        # Fixed width + heightForWidth so multi-line Chinese never collapses.
        self._desc_label.setFixedWidth(_DESC_MAX_WIDTH)
        self._desc_label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Preferred,
        )
        self._reflow_desc_height()

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
            hint = self._action_btn.sizeHint()
            self._action_btn.setMinimumSize(hint)
            self._action_btn.clicked.connect(self.action_clicked)

        content_layout.addWidget(self._icon_label, 0, Qt.AlignmentFlag.AlignHCenter)
        content_layout.addWidget(self._title_label, 0, Qt.AlignmentFlag.AlignHCenter)
        content_layout.addWidget(self._desc_label, 0, Qt.AlignmentFlag.AlignHCenter)
        if self._action_btn is not None:
            content_layout.addWidget(self._action_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        outer.addStretch(1)
        outer.addWidget(self._content, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        # Opaque fill so any residual sibling chrome never bleeds through.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def _reflow_desc_height(self) -> None:
        """Reserve full multi-line height for the wrapped description."""
        width = max(1, self._desc_label.width() or _DESC_MAX_WIDTH)
        needed = max(
            self._desc_label.fontMetrics().height(),
            self._desc_label.heightForWidth(width),
        )
        if needed > 0:
            self._desc_label.setMinimumHeight(needed)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reflow_desc_height()

    # ------------------------------------------------------------------
    # Theme-aware colours
    # ------------------------------------------------------------------

    def _apply_theme_colors(self) -> None:
        """Set foreground colours from the app's effective theme tokens."""
        if getattr(self, "_applying_theme_colors", False):
            return
        self._applying_theme_colors = True
        try:
            tokens = tokens_for_theme(_effective_theme_name(self))
            icon_color = tokens["fg_muted"]
            title_color = tokens["fg_secondary"]
            desc_color = tokens["fg_muted"]
            bg = tokens.get("bg_surface") or tokens.get("bg_window") or ""

            # Prefer palette fill over setStyleSheet on self — StyleChange would
            # re-enter changeEvent and recurse.
            if bg:
                color = QColor(bg)
                if color.isValid():
                    for widget in (self, self._content):
                        palette = widget.palette()
                        palette.setColor(QPalette.ColorRole.Window, color)
                        widget.setAutoFillBackground(True)
                        widget.setPalette(palette)

            self._icon_label.setStyleSheet(
                f"color: {icon_color}; background: transparent;"
            )
            self._title_label.setStyleSheet(
                f"color: {title_color}; background: transparent;"
            )
            self._desc_label.setStyleSheet(
                f"color: {desc_color}; background: transparent;"
            )
        finally:
            self._applying_theme_colors = False

    def showEvent(self, event):  # noqa: N802 – Qt naming convention
        """Re-apply theme colours every time the widget becomes visible."""
        super().showEvent(event)
        self._apply_theme_colors()
        self._reflow_desc_height()

    def changeEvent(self, event):  # noqa: N802 – Qt naming convention
        """React to palette / style changes at runtime."""
        super().changeEvent(event)
        if event.type() in (
            event.Type.PaletteChange,
            event.Type.StyleChange,
        ):
            self._apply_theme_colors()
