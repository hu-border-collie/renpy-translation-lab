"""Background status collection for the context-library page."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal

from .operation_identity import context_library_config_digest


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
) -> ContextLibraryStatusResult:
    """Read project context flags and analysis artifacts outside the GUI thread."""
    context_flags: dict[str, bool] = {}
    try:
        from .bootstrap_report import read_batch_context_flags

        context_flags = read_batch_context_flags(
            config or {},
            game_root=base_dir or None,
        )
        import gemini_translate_batch as batch_mod
        from project_analysis import (
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
            config_digest=context_library_config_digest(
                config,
                context_flags=context_flags,
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
            config_digest=context_library_config_digest(
                config,
                context_flags=context_flags,
            ),
        )


class ContextLibraryStatusSignals(QObject):
    """Signals owned by a thread-pool job."""

    completed = Signal(object)


class ContextLibraryStatusJob(QRunnable):
    """Collect context status in Qt's shared pool without blocking shutdown."""

    def __init__(
        self,
        base_dir: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.base_dir = str(base_dir or "")
        self.config = dict(config or {})
        self.signals = ContextLibraryStatusSignals()
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Cancel a queued job; an already-running scan finishes cooperatively."""
        self._cancel_requested = True

    def run(self) -> None:
        if self._cancel_requested:
            self.signals.completed.emit(None)
            return
        result = collect_context_library_status(self.base_dir, self.config)
        self.signals.completed.emit(result)
