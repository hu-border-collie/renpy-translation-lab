"""Optional GUI font loading with system-font fallback."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

MONO_FONT_SELECTORS = (
    "QListWidget#api_key_list",
    "QLineEdit#project_path_edit",
    "QLineEdit#global_project_path_edit",
    "QLabel#diagnostics_facts_label",
    "QLineEdit#diagnostics_command_edit",
    "QLineEdit#diagnostics_path_edit",
    "QTextEdit#diagnostics_manifest_preview",
    "QTextEdit#log_view",
)

UI_FONT_FILENAME = "HarmonyOS_Sans_SC_Regular.ttf"
MONO_FONT_FILENAME = "LXGWWenKaiMonoGB-Regular.ttf"

# Hint / body copy sizes (px). Keep in sync with app.qss and app_dark.qss.
UI_FONT_SIZE_HINT_PX = 15
UI_FONT_SIZE_BODY_PX = 14
UI_FONT_SIZE_SECTION_PX = 14
UI_FONT_SIZE_MONO_BODY_PX = 13


@dataclass(frozen=True)
class GuiFontFamilies:
    ui_family: str = ""
    mono_family: str = ""


def bundled_fonts_dir(resources_dir: Path) -> Path:
    return resources_dir / "fonts"


def user_fonts_dir() -> Path:
    """Return the per-user directory used by ``download_gui_fonts.py``."""
    override = os.environ.get("RENPY_TRANSLATION_LAB_FONT_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        cache_root = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
    elif sys.platform == "darwin":
        cache_root = Path.home() / "Library" / "Caches"
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "renpy-translation-lab" / "fonts"


def optional_font_files() -> tuple[Path, Path]:
    fonts_dir = user_fonts_dir()
    return fonts_dir / UI_FONT_FILENAME, fonts_dir / MONO_FONT_FILENAME


def optional_fonts_installed() -> bool:
    return all(path.is_file() for path in optional_font_files())


def _font_path(resources_dir: Path, filename: str) -> Path:
    for fonts_dir in (user_fonts_dir(), bundled_fonts_dir(resources_dir)):
        candidate = fonts_dir / filename
        if candidate.is_file():
            return candidate
    return user_fonts_dir() / filename


def _qss_family_name(family: str) -> str:
    escaped = family.replace('"', '\\"')
    return f'"{escaped}"'


def _load_font_family(font_path: Path) -> str:
    if not font_path.is_file():
        return ""
    try:
        from PySide6.QtGui import QFontDatabase
    except ImportError:
        return ""
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    if font_id < 0:
        return ""
    families = QFontDatabase.applicationFontFamilies(font_id)
    return families[0] if families else ""


def load_gui_fonts(resources_dir: Path) -> GuiFontFamilies:
    ui_family = _load_font_family(_font_path(resources_dir, UI_FONT_FILENAME))
    mono_family = _load_font_family(_font_path(resources_dir, MONO_FONT_FILENAME))
    return GuiFontFamilies(ui_family=ui_family, mono_family=mono_family)


def build_font_stylesheet(fonts: GuiFontFamilies) -> str:
    blocks: list[str] = []
    if fonts.ui_family:
        ui_name = _qss_family_name(fonts.ui_family)
        blocks.append(
            f"QWidget {{ font-family: {ui_name}, \"Segoe UI\", sans-serif; }}"
        )
    if fonts.mono_family:
        mono_name = _qss_family_name(fonts.mono_family)
        selectors = ", ".join(MONO_FONT_SELECTORS)
        blocks.append(
            f"{selectors} {{ font-family: {mono_name}, \"Cascadia Mono\", \"Consolas\", monospace; }}"
        )
    return "\n".join(blocks)
