"""Compatibility adapter for Settings pages that still live in MainWindow.

The adapter lets the coordinator take over page identity, navigation, and lazy
build before each page is migrated to a real ``SettingsPage`` object. It
delegates field work to the host, so unmigrated pages keep their current
behavior and the old single save transaction stays authoritative.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .page_contract import SettingsIssue
from .registry import SettingsPageSpec


class LegacySettingsHost(Protocol):
    """Host hooks implemented by ``MainWindow`` for unmigrated pages."""

    def _legacy_settings_page_load(
        self, page_key: str, snapshot: Mapping[str, object]
    ) -> None: ...

    def _legacy_settings_page_collect(
        self, page_key: str
    ) -> Mapping[str, object]: ...

    def _legacy_settings_page_validate(
        self, page_key: str
    ) -> Sequence[SettingsIssue]: ...

    def _legacy_settings_page_reset(self, page_key: str) -> None: ...

    def _legacy_settings_page_focus(
        self, page_key: str, field_key: str
    ) -> bool: ...

    def _legacy_settings_page_set_task_running(
        self, page_key: str, running: bool
    ) -> None: ...


class LegacySettingsPageAdapter:
    """Expose an existing MainWindow-built page through the Settings contract."""

    def __init__(
        self,
        host: LegacySettingsHost,
        spec: SettingsPageSpec,
        widget: object,
    ) -> None:
        self._host = host
        self._spec = spec
        self.widget = widget

    @property
    def page_key(self) -> str:
        return self._spec.key

    @property
    def nav_label(self) -> str:
        return self._spec.nav_label

    @property
    def config_keys(self) -> frozenset[str]:
        return self._spec.config_keys

    @property
    def immediate_action_ids(self) -> frozenset[str]:
        return self._spec.immediate_action_ids

    def load(self, snapshot: Mapping[str, object]) -> None:
        self._host._legacy_settings_page_load(self.page_key, snapshot)

    def collect(self) -> Mapping[str, object]:
        return self._host._legacy_settings_page_collect(self.page_key)

    def validate(self) -> Sequence[SettingsIssue]:
        return self._host._legacy_settings_page_validate(self.page_key)

    def reset(self) -> None:
        self._host._legacy_settings_page_reset(self.page_key)

    def focus_issue(self, issue: SettingsIssue) -> bool:
        return bool(
            self._host._legacy_settings_page_focus(
                self.page_key, issue.field_key
            )
        )

    def set_task_running(self, running: bool) -> None:
        self._host._legacy_settings_page_set_task_running(self.page_key, running)
