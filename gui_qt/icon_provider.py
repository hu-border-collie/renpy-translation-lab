"""Theme-aware loading for the vendored Tabler action icons."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QAbstractButton

from .theme_tokens import DARK_TOKENS, LIGHT_TOKENS


_ICON_SUBDIR = Path("icons") / "tabler"
_ROLE_COLORS = {
    "default": ("fg_secondary_btn", "fg_button_disabled"),
    "on_accent": ("fg_on_accent", "fg_button_disabled"),
    "danger": ("fg_on_accent", "fg_danger_disabled"),
    "success": ("fg_on_accent", "fg_success_disabled"),
}


def _render_svg(svg: bytes, color: str, *, size: int) -> QPixmap:
    """Render one SVG color variant at high DPI for a crisp Qt icon."""
    colored_svg = svg.replace(b"currentColor", color.encode("ascii"))
    renderer = QSvgRenderer(QByteArray(colored_svg))
    if not renderer.isValid():
        return QPixmap()

    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return pixmap


def tabler_icon(
    resources_dir: Path,
    name: str,
    *,
    dark: bool,
    role: str = "default",
    size: int = 18,
) -> QIcon:
    """Return a theme-colored Tabler icon, or a null icon when unavailable."""
    color_keys = _ROLE_COLORS.get(role)
    if color_keys is None:
        raise ValueError(f"Unknown Tabler icon role: {role}")

    source_path = resources_dir / _ICON_SUBDIR / f"{name}.svg"
    try:
        svg = source_path.read_bytes()
    except OSError:
        return QIcon()

    tokens = DARK_TOKENS if dark else LIGHT_TOKENS
    normal_color, disabled_color = (tokens[key] for key in color_keys)
    icon = QIcon()
    normal = _render_svg(svg, QColor(normal_color).name(), size=size)
    disabled = _render_svg(svg, QColor(disabled_color).name(), size=size)
    if not normal.isNull():
        icon.addPixmap(normal, QIcon.Mode.Normal, QIcon.State.Off)
        icon.addPixmap(normal, QIcon.Mode.Active, QIcon.State.Off)
        icon.addPixmap(normal, QIcon.Mode.Selected, QIcon.State.Off)
        icon.addPixmap(normal, QIcon.Mode.Normal, QIcon.State.On)
    if not disabled.isNull():
        icon.addPixmap(disabled, QIcon.Mode.Disabled, QIcon.State.Off)
        icon.addPixmap(disabled, QIcon.Mode.Disabled, QIcon.State.On)
    return icon


def set_tabler_button_icon(
    button: QAbstractButton,
    resources_dir: Path,
    name: str,
    *,
    dark: bool,
    role: str = "default",
    size: int = 18,
) -> None:
    """Apply one consistently sized Tabler icon to a Qt button."""
    button.setIcon(tabler_icon(resources_dir, name, dark=dark, role=role, size=size))
    button.setIconSize(QSize(size, size))
