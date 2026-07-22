"""Background worker for recommended Ren'Py SDK install (no network until started)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from renpy_sdk_install import SdkInstallResult, install_recommended_sdk


class SdkInstallWorker(QThread):
    """Run ``install_recommended_sdk`` off the GUI thread with cancel support."""

    progress = Signal(str, int, int)  # phase, current, total
    completed = Signal(object)  # SdkInstallResult

    def __init__(
        self,
        target_dir: Path | str,
        *,
        workspace_root: Path | str | None = None,
        game_root: Path | str | None = None,
        persist_config: bool = True,
        config_path: Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._target_dir = Path(target_dir)
        self._workspace_root = workspace_root
        self._game_root = game_root
        self._persist_config = bool(persist_config)
        self._config_path = config_path
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        def _progress(phase: str, current: int, total: int) -> None:
            self.progress.emit(phase, current, total)

        try:
            result = install_recommended_sdk(
                self._target_dir,
                workspace_root=self._workspace_root,
                game_root=self._game_root,
                persist_config=self._persist_config,
                config_path=self._config_path,
                should_cancel=lambda: self._cancel or self.isInterruptionRequested(),
                progress=_progress,
            )
        except Exception as exc:  # pragma: no cover - defensive
            result = SdkInstallResult(ok=False, message=f"SDK 安装异常：{exc}")
        self.completed.emit(result)
