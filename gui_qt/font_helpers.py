"""Bundled GUI font loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MONO_FONT_SELECTORS = (
    "QListWidget#api_key_list",
    "QLineEdit#project_path_edit",
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
    fonts_dir = bundled_fonts_dir(resources_dir)
    ui_family = _load_font_family(fonts_dir / UI_FONT_FILENAME)
    mono_family = _load_font_family(fonts_dir / MONO_FONT_FILENAME)
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