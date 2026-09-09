"""Settings page contract, registry, coordinator, and migrated pages (#202)."""

from .coordinator import SettingsCoordinator, snapshot_subset
from .leave_guard import SettingsLeaveGuardPrompt, settings_leave_guard_prompt
from .save_apply import (
    SettingsSaveApplyResult,
    SettingsSaveExtras,
    apply_collected_settings,
)
from .page_contract import SettingsIssue, SettingsPage, SettingsPageActions
from .registry import (
    ADVANCED_CONFIG_KEYS,
    CONFIG_SNAPSHOT_KEYS_BY_PAGE,
    CONTEXT_CONFIG_KEYS,
    PROJECT_CONFIG_KEYS,
    SETTINGS_CONFIG_PAGE_KEYS,
    SETTINGS_LAZY_ATTR_TO_PAGE,
    SETTINGS_PAGE_SPECS,
    WORKSPACE_MANAGED_KEYS,
    SettingsPageRegistry,
    SettingsPageSpec,
    build_default_registry,
)

__all__ = (
    "ADVANCED_CONFIG_KEYS",
    "CONFIG_SNAPSHOT_KEYS_BY_PAGE",
    "CONTEXT_CONFIG_KEYS",
    "PROJECT_CONFIG_KEYS",
    "SETTINGS_CONFIG_PAGE_KEYS",
    "SETTINGS_LAZY_ATTR_TO_PAGE",
    "SETTINGS_PAGE_SPECS",
    "SettingsCoordinator",
    "SettingsIssue",
    "SettingsLeaveGuardPrompt",
    "SettingsPage",
    "SettingsPageActions",
    "SettingsPageRegistry",
    "SettingsPageSpec",
    "SettingsSaveApplyResult",
    "SettingsSaveExtras",
    "WORKSPACE_MANAGED_KEYS",
    "apply_collected_settings",
    "build_default_registry",
    "settings_leave_guard_prompt",
    "snapshot_subset",
)
