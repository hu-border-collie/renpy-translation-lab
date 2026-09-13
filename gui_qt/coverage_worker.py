"""Background worker for importing a completed coverage review (#424 P6)."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from .coverage_actions import import_coverage_review


class CoverageReviewImportWorker(QThread):
    """Validate/install one review JSON without blocking the GUI thread."""

    completed = Signal(object)  # CoverageReviewImportResult

    def __init__(self, review_path: str, parent=None) -> None:
        super().__init__(parent)
        self._review_path = str(review_path or "")
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        if self._cancel_requested or self.isInterruptionRequested():
            return
        result = import_coverage_review(self._review_path)
        if self._cancel_requested or self.isInterruptionRequested():
            return
        self.completed.emit(result)
