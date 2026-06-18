"""Qt integration for applying GUI theme stylesheets."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .theme_helpers import (
    resolve_effective_theme,
    theme_stylesheet_filename,
)


def system_prefers_dark(app: QApplication) -> bool | None:
    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return True
    if scheme == Qt.ColorScheme.Light:
        return False
    return None


def load_theme_stylesheet(resources_dir: Path, effective_theme: str) -> str:
    filename = theme_stylesheet_filename(effective_theme)
    path = resources_dir / filename
    if not path.exists() and effective_theme == "dark":
        path = resources_dir / "app.qss"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def apply_theme(app: QApplication, resources_dir: Path, preference: str) -> str:
    effective = resolve_effective_theme(
        preference,
        system_is_dark=system_prefers_dark(app),
    )
    stylesheet = load_theme_stylesheet(resources_dir, effective)
    if stylesheet:
        app.setStyleSheet(stylesheet)
    else:
        app.setStyleSheet("")
    return effective