"""Pure helpers for GUI theme preference resolution (no Qt dependency)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

THEME_SYSTEM = "system"
THEME_LIGHT = "light"
THEME_DARK = "dark"

VALID_THEME_PREFERENCES = frozenset({THEME_SYSTEM, THEME_LIGHT, THEME_DARK})
DEFAULT_THEME_PREFERENCE = THEME_SYSTEM


def normalize_theme_preference(value: Any) -> str:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in VALID_THEME_PREFERENCES:
            return cleaned
    return DEFAULT_THEME_PREFERENCE


def resolve_effective_theme(preference: Any, *, system_is_dark: bool | None = None) -> str:
    normalized = normalize_theme_preference(preference)
    if normalized == THEME_LIGHT:
        return THEME_LIGHT
    if normalized == THEME_DARK:
        return THEME_DARK
    if system_is_dark is True:
        return THEME_DARK
    return THEME_LIGHT


def theme_stylesheet_filename(effective_theme: str) -> str:
    if effective_theme == THEME_DARK:
        return "app_dark.qss"
    return "app.qss"


def read_gui_theme_from_config(config: dict[str, Any]) -> str:
    gui = config.get("gui")
    if not isinstance(gui, dict):
        return DEFAULT_THEME_PREFERENCE
    return normalize_theme_preference(gui.get("theme"))


def write_gui_theme_to_config(config: dict[str, Any], preference: Any) -> None:
    gui = config.get("gui")
    if not isinstance(gui, dict):
        gui = {}
        config["gui"] = gui
    gui["theme"] = normalize_theme_preference(preference)


_STYLESHEET_CACHE: dict[tuple[str, str], str] = {}


def clear_stylesheet_cache() -> None:
    """Reset stylesheet cache; primarily intended for test isolation."""
    _STYLESHEET_CACHE.clear()


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