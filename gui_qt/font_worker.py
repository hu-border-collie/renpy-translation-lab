"""Background worker for downloading optional GUI fonts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from scripts.download_gui_fonts import install_fonts


@dataclass(frozen=True)
class FontInstallResult:
    ok: bool
    installed: tuple[Path, ...] = ()
    error: str = ""


def run_font_install(destination: Path | None = None) -> FontInstallResult:
    try:
        installed = install_fonts(destination) if destination is not None else install_fonts_default()
        return FontInstallResult(True, tuple(installed))
    except Exception as exc:
        return FontInstallResult(False, error=str(exc))


def install_fonts_default() -> list[Path]:
    from gui_qt.font_helpers import user_fonts_dir

    return install_fonts(user_fonts_dir())


class FontInstallWorker(QThread):
    completed = Signal(object)

    def run(self) -> None:
        self.completed.emit(run_font_install())
