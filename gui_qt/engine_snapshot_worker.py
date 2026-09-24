"""Background worker for engine/snapshot/reuse service tasks (#424 P6, #512)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QThread, Signal


@dataclass(frozen=True)
class EngineSnapshotTaskResult:
    """Result of one snapshot/reuse task."""

    ok: bool
    payload: Any = None
    error: str = ""
    error_code: str = ""


class EngineSnapshotTaskWorker(QThread):
    """Run one snapshot/reuse loader or shared-service action in the worker thread."""

    completed = Signal(object)

    def __init__(self, task: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self._task = task

    def run(self) -> None:
        try:
            payload = self._task()
        except Exception as exc:  # noqa: BLE001 - GUI boundary
            result = EngineSnapshotTaskResult(
                ok=False,
                error=str(exc),
                error_code=type(exc).__name__,
            )
        else:
            result = EngineSnapshotTaskResult(ok=True, payload=payload)
        if not self.isInterruptionRequested():
            self.completed.emit(result)
