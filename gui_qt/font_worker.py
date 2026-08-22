"""Background worker for downloading optional GUI fonts."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from scripts.download_gui_fonts import install_fonts


# Whole-subprocess budget; the GUI offers cancellation long before this fires.
FONT_INSTALL_TOTAL_TIMEOUT_S = 1800.0
# Poll granularity between child-process joins so cancel stays responsive.
FONT_INSTALL_JOIN_POLL_S = 0.2


@dataclass(frozen=True)
class FontInstallResult:
    ok: bool
    installed: tuple[Path, ...] = ()
    error: str = ""
    cancelled: bool = False


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


def _stop_child_process(proc) -> None:
    proc.terminate()
    proc.join(timeout=2)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=1)


def run_font_install_in_subprocess(
    destination: Path | None = None,
    *,
    should_cancel: Callable[[], bool] | None = None,
    poll_interval_s: float = FONT_INSTALL_JOIN_POLL_S,
    total_timeout_s: float = FONT_INSTALL_TOTAL_TIMEOUT_S,
) -> FontInstallResult:
    """Install fonts in a child process so extraction work does not freeze the GUI GIL.

    ``should_cancel`` is polled between short joins: a cancel requested on the
    GUI thread terminates the child promptly instead of blocking for the whole
    budget.  The child is a daemon process, so a hard interpreter exit also
    takes it down.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(1)
    proc = ctx.Process(
        target=_font_install_process_entry,
        args=(str(destination) if destination is not None else None, result_queue),
        daemon=True,
    )
    proc.start()
    try:
        deadline = time.monotonic() + max(0.0, float(total_timeout_s))
        poll_s = max(0.01, float(poll_interval_s))
        while True:
            proc.join(timeout=poll_s)
            if not proc.is_alive():
                break
            if should_cancel is not None and should_cancel():
                _stop_child_process(proc)
                return FontInstallResult(
                    False,
                    error="字体下载已取消。",
                    cancelled=True,
                )
            if time.monotonic() >= deadline:
                _stop_child_process(proc)
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
    """Download recommended fonts off the GUI thread with cooperative cancel.

    ``completed`` always fires exactly once, including for cancelled and
    timed-out installs, so callers keep ownership until the real ``finished``.
    The non-isolated fallback path cannot be interrupted mid-install; a cancel
    requested during it only relabels the failure, never the success.
    """

    completed = Signal(object)

    def __init__(self, parent=None, *, isolate_process: bool = True) -> None:
        super().__init__(parent)
        self._isolate_process = isolate_process
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Ask the install to stop; the child process is terminated promptly."""
        self._cancel_requested = True
        self.requestInterruption()

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested or self.isInterruptionRequested())

    def run(self) -> None:
        if self.is_cancel_requested():
            self.completed.emit(
                FontInstallResult(False, error="字体下载已取消。", cancelled=True)
            )
            return
        if self._isolate_process:
            try:
                result = run_font_install_in_subprocess(
                    should_cancel=self.is_cancel_requested
                )
            except Exception:
                result = run_font_install()
        else:
            result = run_font_install()
        if not result.ok and self.is_cancel_requested():
            result = replace(result, cancelled=True)
        self.completed.emit(result)
