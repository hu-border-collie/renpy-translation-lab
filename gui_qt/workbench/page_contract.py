"""Small page boundary for the workbench migration (#176 P0).

Pages own their widgets and render the session supplied by ``MainWindow``.
They report user intent through callbacks; workflow state, CLI execution, and
writeback safety remain in the application coordinator.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..work_modes import WorkMode
from ..workbench_session import WorkbenchModeSession


@dataclass(frozen=True)
class WorkbenchPageActions:
    """Coordinator callbacks a page may expose through its own controls."""

    start: Callable[[], None] | None = None
    resume: Callable[[], None] | None = None
    stop: Callable[[], None] | None = None
    writeback: Callable[[], None] | None = None
    prebuild: Callable[[str], None] | None = None
    open_settings: Callable[[], None] | None = None


class WorkbenchPage(Protocol):
    """Contract for one persistent page in the workbench stack."""

    supported_modes: tuple[WorkMode, ...]

    def set_action_callbacks(self, actions: WorkbenchPageActions) -> None:
        """Receive coordinator-owned actions for page-local controls."""

    def activate(self, mode: WorkMode, session: WorkbenchModeSession) -> None:
        """Render the active mode using its sole runtime session state."""

    def set_task_running(self, running: bool) -> None:
        """Reflect the global runner lock in page-local controls."""

    def reset_project(self) -> None:
        """Clear view-only state after the coordinator changes game_root."""
