"""Navigation and layout coordination for persistent workbench pages."""
from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy, QStackedWidget

from ..work_modes import WorkMode, WorkbenchNavItem, workbench_nav_for_work_mode
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPage


class WorkbenchPageCoordinator:
    """Own stack selection, page activation, and current-page sizing policy."""

    def __init__(
        self,
        stack: QStackedWidget,
        pages: Mapping[WorkbenchNavItem, WorkbenchPage],
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
        page_stack = getattr(page, "page_stack", None)
        current = page_stack.currentWidget() if page_stack is not None else None
        gate_widgets = {
            getattr(page, "empty_state", None),
            getattr(page, "project_gate_state", None),
        }
        parent_layout = self._stack.parentWidget().layout()
        if current is not None and current in gate_widgets:
            # A project gate is the whole page, not a short task-control row.
            # Let it consume the remaining shell height so its centered CTA is
            # never clipped by a content-page height estimate (#298/#316).
            self._stack.setMinimumHeight(0)
            self._stack.setMaximumHeight(16_777_215)
            self._stack.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            if parent_layout is not None:
                parent_layout.setAlignment(self._stack, Qt.AlignmentFlag(0))
                parent_layout.setStretchFactor(self._stack, 1)
            return
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
        if parent_layout is not None:
            parent_layout.setAlignment(self._stack, Qt.AlignmentFlag.AlignTop)
            parent_layout.setStretchFactor(self._stack, 0)
