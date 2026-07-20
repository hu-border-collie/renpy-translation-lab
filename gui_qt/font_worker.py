"""Background worker for downloading optional GUI fonts."""

from __future__ import annotations

import multiprocessing as mp
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


def _font_install_process_entry(destination: str | None, result_queue) -> None:
    try:
        dest = Path(destination) if destination else None
        result_queue.put(run_font_install(dest))
    except BaseException as exc:  # pragma: no cover - process boundary
        result_queue.put(FontInstallResult(False, error=str(exc)))


def run_font_install_in_subprocess(
    destination: Path | None = None,
) -> FontInstallResult:
    """Install fonts in a child process so extraction work does not freeze the GUI GIL."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(1)
    proc = ctx.Process(
        target=_font_install_process_entry,
        args=(str(destination) if destination is not None else None, result_queue),
        daemon=True,
    )
    proc.start()
    try:
        proc.join(timeout=1800)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
            return FontInstallResult(False, error="字体下载超时或被中断。")
        try:
            payload = result_queue.get(timeout=1.0)
        except Exception:
            return FontInstallResult(
                False,
                error=f"字体安装进程异常退出（code={proc.exitcode}）。",
            )
        if isinstance(payload, FontInstallResult):
            return payload
        return FontInstallResult(False, error="字体安装返回了未知结果。")
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)


class FontInstallWorker(QThread):
    completed = Signal(object)

    def __init__(self, parent=None, *, isolate_process: bool = True) -> None:
        super().__init__(parent)
        self._isolate_process = isolate_process

    def run(self) -> None:
        if self._isolate_process:
            try:
                result = run_font_install_in_subprocess()
            except Exception:
                result = run_font_install()
        else:
            result = run_font_install()
        self.completed.emit(result)
