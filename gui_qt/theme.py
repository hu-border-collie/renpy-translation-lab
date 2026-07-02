"""Qt integration for applying GUI theme stylesheets."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .font_helpers import GuiFontFamilies, build_font_stylesheet, load_gui_fonts
from .theme_helpers import (
    resolve_effective_theme,
    theme_stylesheet_filename,
)

_STYLESHEET_CACHE: dict[tuple[str, str], str] = {}
_FONT_CACHE: dict[str, GuiFontFamilies] = {}


def clear_theme_caches() -> None:
    """Reset module-level caches; primarily intended for test isolation."""
    _STYLESHEET_CACHE.clear()
    _FONT_CACHE.clear()


def system_prefers_dark(app: QApplication) -> bool | None:
    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return True
    if scheme == Qt.ColorScheme.Light:
        return False
    return None


def load_theme_stylesheet(resources_dir: Path, effective_theme: str) -> str:
    cache_key = (str(resources_dir.resolve()), effective_theme)
    cached = _STYLESHEET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    filename = theme_stylesheet_filename(effective_theme)
    path = resources_dir / filename
    if not path.exists() and effective_theme == "dark":
        path = resources_dir / "app.qss"
    if not path.exists():
        stylesheet = ""
    else:
        stylesheet = path.read_text(encoding="utf-8")
    _STYLESHEET_CACHE[cache_key] = stylesheet
    return stylesheet


def cached_gui_fonts(resources_dir: Path) -> GuiFontFamilies:
    cache_key = str(resources_dir.resolve())
    cached = _FONT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    fonts = load_gui_fonts(resources_dir)
    _FONT_CACHE[cache_key] = fonts
    return fonts


def apply_theme(app: QApplication, resources_dir: Path, preference: str) -> str:
    effective = resolve_effective_theme(
        preference,
        system_is_dark=system_prefers_dark(app),
    )
    stylesheet = load_theme_stylesheet(resources_dir, effective)
    font_stylesheet = build_font_stylesheet(cached_gui_fonts(resources_dir))
    if font_stylesheet:
        stylesheet = f"{font_stylesheet}\n{stylesheet}"
    if stylesheet:
        app.setStyleSheet(stylesheet)
    else:
        app.setStyleSheet("")
    return effective