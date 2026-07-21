"""Background worker for registry refresh so the GUI stays responsive."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .games_registry_actions import (
    RegistryActionResult,
    ingest_registry_project,
    refresh_registry_projects,
)


class RegistryRefreshWorker(QThread):
    """Run registry refresh off the UI thread with optional cancellation."""

    progress = Signal(int, int, str)
    completed = Signal(object)

    def __init__(
        self,
        *,
        workspace_root: Path,
        project_id: str | None = None,
        refresh_everything: bool = False,
        mode: str,
        parent=None,
    ):
        super().__init__(parent)
        self._workspace_root = workspace_root
        self._project_id = project_id
        self._refresh_everything = refresh_everything
        self._mode = mode
        self._cancel_requested = False

    def request_stop(self) -> None:
        self._cancel_requested = True

    def _should_cancel(self) -> bool:
        return self._cancel_requested

    def _emit_progress(self, current: int, total: int, name: str) -> None:
        self.progress.emit(current, total, name)

    def run(self) -> None:
        try:
            result = refresh_registry_projects(
                self._workspace_root,
                project_id=self._project_id,
                refresh_everything=self._refresh_everything,
                mode=self._mode,
                on_progress=self._emit_progress,
                should_cancel=self._should_cancel,
            )
        except Exception as exc:
            result = RegistryActionResult(False, f"刷新失败：{exc}")
        if not isinstance(result, RegistryActionResult):
            result = RegistryActionResult(False, "刷新失败：未知结果。")
        self.completed.emit(result)


class RegistryIngestWorker(QThread):
    """Copy a game directory/zip into Game_* layout off the UI thread."""

    progress = Signal(int, int, str)
    completed = Signal(object)

    def __init__(
        self,
        *,
        workspace_root: Path,
        source: Path,
        game_name: str,
        mode: str = "lite",
        parent=None,
    ):
        super().__init__(parent)
        self._workspace_root = workspace_root
        self._source = source
        self._game_name = game_name
        self._mode = mode
        self._cancel_requested = False

    def request_stop(self) -> None:
        self._cancel_requested = True

    def _should_cancel(self) -> bool:
        return self._cancel_requested

    def _emit_progress(self, current: int, total: int, name: str) -> None:
        self.progress.emit(current, total, name)

    def run(self) -> None:
        try:
            result = ingest_registry_project(
                self._workspace_root,
                source=self._source,
                game_name=self._game_name,
                mode=self._mode,
                on_progress=self._emit_progress,
                should_cancel=self._should_cancel,
            )
        except Exception as exc:
            result = RegistryActionResult(False, f"导入失败：{exc}")
        if not isinstance(result, RegistryActionResult):
            result = RegistryActionResult(False, "导入失败：未知结果。")
        self.completed.emit(result)