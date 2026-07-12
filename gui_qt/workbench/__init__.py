"""Shared contracts and coordination for persistent workbench pages."""

from .coordinator import WorkbenchPageCoordinator
from .page_contract import WorkbenchPage, WorkbenchPageActions

__all__ = (
    "WorkbenchPage",
    "WorkbenchPageActions",
    "WorkbenchPageCoordinator",
)