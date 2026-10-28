"""Background status collection for the context-library page."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import multiprocessing as mp
from queue import Empty
import time
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal

from gui_qt.runtime.operation_identity import context_library_config_digest
from gui_qt.user_copy import CONTEXT_LIBRARY_COPY


@dataclass(frozen=True)
class ContextLibraryStatusResult:
    """Readonly context-library status produced outside the GUI thread."""

    base_dir: str
    live_fingerprint: str
    status: dict[str, Any] | None
    label: str
    context_flags: dict[str, bool] = field(default_factory=dict)
    error: str = ""
    config_digest: str = ""


def collect_context_library_status(
    base_dir: str,
    config: dict[str, Any] | None = None,
    *, config_file: str = 'translator_config.json',
) -> ContextLibraryStatusResult:
    """Read project context flags and analysis artifacts outside the GUI thread."""
    context_flags: dict[str, bool] = {}
    config_digest = ''
    try:
        from engine_adapters.project_config import resolve_project_configuration
        from gui_qt.reports.bootstrap_report import read_batch_context_flags

        try:
            config = resolve_project_configuration(config or {}, config_file)
        finally:
            # Error results need the same effective flags as the live window.
            context_flags = read_batch_context_flags(
                config or {},
                game_root=base_dir or None,
            )
        if (config or {}).get('engine') == 'rpgmaker_mv':
            from execution.mv_preprocessing import context_status
            config_digest = context_library_config_digest(config, context_flags=context_flags, config_file=config_file)
            preparation = context_status(config, config_file)
            return ContextLibraryStatusResult(base_dir, '',
                {'overall_status': 'unsupported', 'mv_preparation': preparation},
                'MV 项目分析当前未支持', context_flags=context_flags,
                config_digest=config_digest)
        import gemini_translate_batch as batch_mod
        from quality_review.project_analysis import (
            collect_project_analysis_status,
            format_status_label,
        )

        live_fingerprint = str(
            batch_mod.compute_current_project_analysis_fingerprint(base_dir or None) or ""
        )
        status = collect_project_analysis_status(
            base_dir=base_dir or None,
            expected_source_fingerprint=live_fingerprint,
        )
        return ContextLibraryStatusResult(
            base_dir=base_dir,
            live_fingerprint=live_fingerprint,
            status=status,
            label=format_status_label(status),
            context_flags=context_flags,
            config_digest=config_digest or context_library_config_digest(
                config,
                context_flags=context_flags,
                config_file=config_file,
            ),
        )
    except Exception as exc:
        return ContextLibraryStatusResult(
            base_dir=base_dir,
            live_fingerprint="",
            status=None,
            label=f"读取失败 · {exc}",
            context_flags=context_flags,
            error=str(exc),
            config_digest=config_digest or context_library_config_digest(
                config,
                context_flags=context_flags,
                config_file=config_file,
            ),
        )


def _status_failure(base_dir, config, config_file, error) -> ContextLibraryStatusResult:
    from gui_qt.reports.bootstrap_report import read_batch_context_flags

    flags = read_batch_context_flags(config or {}, game_root=base_dir or None)
    return ContextLibraryStatusResult(
        base_dir, "", None, f"读取失败 · {error}", context_flags=flags, error=error,
        config_digest=context_library_config_digest(
            config, context_flags=flags, config_file=config_file),
    )


def _context_status_process_entry(base_dir, config, config_file, result_queue) -> None:
    """Collect a read-only status in a spawn-safe, independent Python heap."""
    try:
        result = collect_context_library_status(base_dir, config, config_file=config_file)
    except BaseException as exc:  # defensive process boundary, including SystemExit
        result = _status_failure(base_dir, config, config_file, str(exc))
    result_queue.put(result)


def run_context_library_status_in_subprocess(
    base_dir: str,
    config: dict[str, Any] | None = None,
    *,
    config_file: str = 'translator_config.json',
    should_cancel=None,
    poll_s: float = .05,
    timeout_s: float = 120,
) -> ContextLibraryStatusResult | None:
    """Isolate full material scans and their GC pauses from the Qt process.

    Drain the result while the child is alive, including reports larger than a
    pipe buffer. Cancellation, timeout, startup failure and crashes all reclaim
    the owned process/queue. A failed spawn never falls back to a GUI-heap scan.
    The host still validates the returned project/material identity before use.
    """
    if should_cancel is not None and should_cancel():
        return None
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue(1)
    proc = ctx.Process(
        target=_context_status_process_entry,
        args=(base_dir, config, config_file, result_queue),
        daemon=True,
    )
    started = False
    try:
        try:
            proc.start()
            started = True
        except Exception:
            return _status_failure(base_dir, config, config_file,
                                   CONTEXT_LIBRARY_COPY['status_process_start_failed'])
        deadline = time.monotonic() + timeout_s
        while True:
            if should_cancel is not None and should_cancel():
                return None
            if time.monotonic() >= deadline:
                return _status_failure(base_dir, config, config_file,
                                       CONTEXT_LIBRARY_COPY['status_timeout'])
            try:
                result = result_queue.get(timeout=poll_s)
            except (EOFError, OSError):
                return _status_failure(
                    base_dir, config, config_file,
                    CONTEXT_LIBRARY_COPY['status_process_failed'].format(code=proc.exitcode))
            except Empty:
                if proc.is_alive():
                    continue
                return _status_failure(
                    base_dir, config, config_file,
                    CONTEXT_LIBRARY_COPY['status_process_failed'].format(code=proc.exitcode))
            if isinstance(result, ContextLibraryStatusResult):
                return result
            return _status_failure(
                base_dir, config, config_file,
                CONTEXT_LIBRARY_COPY['status_process_failed'].format(code=proc.exitcode))
    finally:
        if started:
            # A result is already fully drained. Give the read-only collector a
            # short grace period, then avoid waiting for large-heap finalization.
            proc.join(timeout=poll_s)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
        result_queue.close()
        result_queue.join_thread()
        proc.close()


class ContextLibraryStatusSignals(QObject):
    """Signals owned by a thread-pool job."""

    completed = Signal(object)


class ContextLibraryStatusJob(QRunnable):
    """Wait for an isolated status scan from Qt's pool without sharing its heap."""

    def __init__(
        self,
        base_dir: str,
        config: dict[str, Any] | None = None,
        *, config_file: str = 'translator_config.json',
    ) -> None:
        super().__init__()
        self.base_dir = str(base_dir or "")
        self.config = copy.deepcopy(config or {})
        self.config_file = config_file
        self.signals = ContextLibraryStatusSignals()
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Cancel queued collection or stop its running read-only child process."""
        self._cancel_requested = True

    def run(self) -> None:
        if self._cancel_requested:
            self.signals.completed.emit(None)
            return
        try:
            result = run_context_library_status_in_subprocess(
                self.base_dir, self.config, config_file=self.config_file,
                should_cancel=lambda: self._cancel_requested)
        except Exception:
            result = _status_failure(
                self.base_dir, self.config, self.config_file,
                CONTEXT_LIBRARY_COPY['status_process_start_failed'])
        self.signals.completed.emit(result)
