"""Frozen Settings page boundary for the #202 Phase B coordinator.

Pages own their widgets, field mapping, local validation, and local actions.
The coordinator owns page identity, lazy materialization, in-page navigation,
the load/collect/validate/reset/focus contract, the dirty baseline, and
leave-guard copy. ``MainWindow`` uniquely owns the single save transaction and
Qt dialogs; coordinator helpers never write configuration themselves. This
module deliberately has no Qt dependency so the contract can be tested
independently from ``MainWindow``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class SettingsIssue:
    """One page-local validation issue, addressed by page and field key."""

    page_key: str
    field_key: str
    message: str
    severity: Literal["error", "warning"] = "error"


@dataclass(frozen=True)
class SettingsPageActions:
    """Coordinator-owned callbacks injected into a page.

    Pages must use these callbacks instead of reaching into ``MainWindow``
    private attributes, so they stay independently constructible.
    """

    save: Callable[[], None] | None = None
    reload: Callable[[], None] | None = None
    navigate: Callable[[str], None] | None = None
    run_immediate: Callable[[str, Mapping[str, object]], bool] | None = None
    show_status: Callable[[str], None] | None = None


@runtime_checkable
class SettingsPage(Protocol):
    """Minimal page adapter contract frozen by #202 Phase A."""

    page_key: str
    nav_label: str
    config_keys: frozenset[str]
    immediate_action_ids: frozenset[str]

    def load(self, snapshot: Mapping[str, object]) -> None:
        """Push owned values from a flat config snapshot into page widgets."""

    def collect(self) -> Mapping[str, object]:
        """Return only the config keys owned by this page; never write disk."""

    def validate(self) -> Sequence[SettingsIssue]:
        """Return page-local issues; no dialogs and no disk writes."""

    def reset(self) -> None:
        """Discard unsaved page edits back to the last loaded/saved baseline."""

    def focus_issue(self, issue: SettingsIssue) -> bool:
        """Focus/scroll/decorate the issue field; return whether handled."""

    def set_task_running(self, running: bool) -> None:
        """Reflect the global task lock without owning it."""
