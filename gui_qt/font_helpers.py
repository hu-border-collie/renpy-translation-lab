"""Optional workspace font loading for the GUI."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QFontDatabase

MONO_FONT_SELECTORS = (
    "QListWidget#api_key_list",
    "QLineEdit#project_path_edit",
    "QLabel#diagnostics_facts_label",
    "QLineEdit#diagnostics_command_edit",
    "QLineEdit#diagnostics_path_edit",
    "QTextEdit#diagnostics_manifest_preview",
    "QTextEdit#log_view",
)

UI_FONT_RELATIVE = Path("HarmonyOS_Sans") / "HarmonyOS_Sans_SC_Regular.ttf"
MONO_FONT_RELATIVE = Path("LXGW_WenKai") / "LXGWWenKaiMonoGB-Regular.ttf"


@dataclass(frozen=True)
class GuiFontFamilies:
    ui_family: str = ""
    mono_family: str = ""


def resolve_workspace_fonts_dir(tool_root: Path) -> Path | None:
    env_dir = os.environ.get("RENPY_TRANSLATION_LAB_FONTS_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir).expanduser()
        if candidate.is_dir():
            return candidate
    sibling = tool_root.parent / "_fonts"
    if sibling.is_dir():
        return sibling
    return None


def _qss_family_name(family: str) -> str:
    escaped = family.replace('"', '\\"')
    return f'"{escaped}"'


def _load_font_family(font_path: Path) -> str:
    if not font_path.is_file():
        return ""
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    if font_id < 0:
        return ""
    families = QFontDatabase.applicationFontFamilies(font_id)
    return families[0] if families else ""


def load_gui_fonts(tool_root: Path) -> GuiFontFamilies:
    fonts_dir = resolve_workspace_fonts_dir(tool_root)
    if fonts_dir is None:
        return GuiFontFamilies()

    ui_family = _load_font_family(fonts_dir / UI_FONT_RELATIVE)
    mono_family = _load_font_family(fonts_dir / MONO_FONT_RELATIVE)
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