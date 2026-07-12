"""Navigation and layout coordination for persistent workbench pages."""
from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtWidgets import QSizePolicy, QStackedWidget, QWidget

from ..work_modes import WorkMode, WorkbenchNavItem, workbench_nav_for_work_mode
from ..workbench_session import WorkbenchModeSession


class WorkbenchPageCoordinator:
    """Own stack selection, page activation, and current-page sizing policy."""

    def __init__(
        self,
        stack: QStackedWidget,
        pages: Mapping[WorkbenchNavItem, QWidget],
    ) -> None:
        self._stack = stack
        self._pages = dict(pages)

    def activate(
        self,
        mode: WorkMode,
        session: WorkbenchModeSession,
        *,
        running: bool,
    ) -> WorkbenchNavItem:
        nav_item = workbench_nav_for_work_mode(mode)
        page = self._pages[nav_item]
        self._stack.setCurrentWidget(page)
        page.activate(mode, session)
        page.set_task_running(running)
        self.resize(nav_item)
        return nav_item

    def resize(self, nav_item: WorkbenchNavItem) -> None:
        """Pin the stack to the active page instead of its tallest sibling."""
        page = self._pages[nav_item]
        preferred_height = getattr(page, "preferred_height", None)
        if callable(preferred_height):
            height = preferred_height(self._stack.width())
        else:
            height = page.sizeHint().height()
        self._stack.setMinimumHeight(0)
        self._stack.setMaximumHeight(max(int(height), 48))
        self._stack.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )