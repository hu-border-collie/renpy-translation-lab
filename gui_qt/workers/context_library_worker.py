"""Background status collection for the context-library page."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal

from gui_qt.runtime.operation_identity import context_library_config_digest


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
        from gui_qt.reports.bootstrap_report import read_batch_context_flags

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


class ContextLibraryStatusSignals(QObject):
    """Signals owned by a thread-pool job."""

    completed = Signal(object)


class ContextLibraryStatusJob(QRunnable):
    """Collect context status in Qt's shared pool without blocking shutdown."""

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
        """Cancel a queued job; an already-running scan finishes cooperatively."""
        self._cancel_requested = True

    def run(self) -> None:
        if self._cancel_requested:
            self.signals.completed.emit(None)
            return
        result = collect_context_library_status(self.base_dir, self.config, **(
            {'config_file': self.config_file} if self.config.get('engine') == 'rpgmaker_mv' else {}))
        self.signals.completed.emit(result)
