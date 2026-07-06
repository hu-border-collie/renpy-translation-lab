"""Pure helpers for GUI theme preference resolution (no Qt dependency)."""
from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

THEME_SYSTEM = "system"
THEME_LIGHT = "light"
THEME_DARK = "dark"

VALID_THEME_PREFERENCES = frozenset({THEME_SYSTEM, THEME_LIGHT, THEME_DARK})
DEFAULT_THEME_PREFERENCE = THEME_SYSTEM

_TEMPLATE_FILENAME = "app_template.qss"


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
    """Load and render the themed stylesheet.

    Prefers the token-based template (``app_template.qss`` + ``theme_tokens``).
    Falls back to legacy per-theme files (``app.qss`` / ``app_dark.qss``) if the
    template does not exist, ensuring a safe migration path.
    """
    cache_key = (str(resources_dir.resolve()), effective_theme)
    cached = _STYLESHEET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    template_path = resources_dir / _TEMPLATE_FILENAME
    if template_path.exists():
        stylesheet = _render_template(template_path, effective_theme)
    else:
        stylesheet = _load_legacy_stylesheet(resources_dir, effective_theme)

    _STYLESHEET_CACHE[cache_key] = stylesheet
    return stylesheet


def _render_template(template_path: Path, effective_theme: str) -> str:
    """Read the QSS template and substitute theme tokens."""
    from .theme_tokens import tokens_for_theme

    raw = template_path.read_text(encoding="utf-8")
    tokens = tokens_for_theme(effective_theme)
    return Template(raw).safe_substitute(tokens)


def _load_legacy_stylesheet(resources_dir: Path, effective_theme: str) -> str:
    """Fallback: load a pre-baked per-theme QSS file."""
    filename = theme_stylesheet_filename(effective_theme)
    path = resources_dir / filename
    if not path.exists() and effective_theme == "dark":
        path = resources_dir / "app.qss"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")