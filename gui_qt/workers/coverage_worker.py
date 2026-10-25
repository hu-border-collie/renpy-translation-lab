"""Background worker for importing a completed coverage review (#424 P6)."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gui_qt.workbench.coverage_actions import import_coverage_review


class CoverageReviewImportWorker(QThread):
    """Validate/install one review JSON without blocking the GUI thread."""

    completed = Signal(object)  # CoverageReviewImportResult

    def __init__(self, review_path: str, parent=None, *, config=None) -> None:
        super().__init__(parent)
        self._review_path = str(review_path or "")
        self._cancel_requested = False
        self._config = config

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        if self._cancel_requested or self.isInterruptionRequested():
            return
        if self._config is None:
            result = import_coverage_review(self._review_path)
        else:
            import translator_runtime as runtime
            with runtime.runtime_config_scope(self._config):
                result = import_coverage_review(self._review_path)
        if self._cancel_requested or self.isInterruptionRequested():
            return
        self.completed.emit(result)


class CoverageExportWorker(QThread):
    """Export the same read-only coverage package as the CLI."""

    completed = Signal(object)

    def __init__(self, output_dir: str, parent=None, *, config=None):
        super().__init__(parent)
        self.output_dir, self.config = output_dir, config

    def run(self):
        try:
            import gemini_translate_batch as batch
            import translator_runtime as runtime
            with runtime.runtime_config_scope(self.config, reload_translator_settings=self.config is None,
                                              persist_corrected_game_root=False):
                batch.export_live_coverage_package(self.output_dir)
            result = {'ok': True, 'output_dir': self.output_dir}
        except (Exception, SystemExit) as exc:
            result = {'ok': False, 'error': str(exc)}
        if not self.isInterruptionRequested():
            self.completed.emit(result)
