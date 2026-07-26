"""Background status collection for the context-library page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal


@dataclass(frozen=True)
class ContextLibraryStatusResult:
    """Readonly Project Analysis status produced outside the GUI thread."""

    base_dir: str
    live_fingerprint: str
    status: dict[str, Any] | None
    label: str
    error: str = ""


def collect_context_library_status(base_dir: str) -> ContextLibraryStatusResult:
    """Scan scripts and analysis artifacts without touching Qt widgets."""
    try:
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
        )
    except Exception as exc:
        return ContextLibraryStatusResult(
            base_dir=base_dir,
            live_fingerprint="",
            status=None,
            label=f"读取失败 · {exc}",
            error=str(exc),
        )


class ContextLibraryStatusSignals(QObject):
    """Signals owned by a thread-pool job."""

    completed = Signal(object)


class ContextLibraryStatusJob(QRunnable):
    """Collect context status in Qt's shared pool without blocking shutdown."""

    def __init__(self, base_dir: str) -> None:
        super().__init__()
        self.base_dir = str(base_dir or "")
        self.signals = ContextLibraryStatusSignals()

    def run(self) -> None:
        self.signals.completed.emit(collect_context_library_status(self.base_dir))
