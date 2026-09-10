"""Settings page identity and field-ownership registry (#202 Phase B).

The registry is the single source for page keys, nav labels, builder names,
lazy-attribute mapping, and which flat config keys each page owns. It has no
Qt dependency and can be tested independently from ``MainWindow``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..settings_schema import ADVANCED_SETTING_FIELDS, CONTEXT_PRIMARY_SETTING_KEYS

WORKSPACE_MANAGED_KEYS: frozenset[str] = frozenset({"game_root"})

_PROJECT_CATEGORIES: frozenset[str] = frozenset({"项目与资源", "准备流程"})

PROJECT_CONFIG_KEYS: frozenset[str] = frozenset(
    field.key
    for field in ADVANCED_SETTING_FIELDS
    if field.category in _PROJECT_CATEGORIES
    and field.key not in WORKSPACE_MANAGED_KEYS
)

CONTEXT_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "rag_enabled",
        "source_index_enabled",
        "bootstrap_on_build",
        "sync_source_index_enabled",
        "sync_project_analysis_inject_enabled",
        "context_storage_location",
    }
) | CONTEXT_PRIMARY_SETTING_KEYS

ADVANCED_CONFIG_KEYS: frozenset[str] = frozenset(
    field.key
    for field in ADVANCED_SETTING_FIELDS
    if field.key not in WORKSPACE_MANAGED_KEYS
    and field.key not in PROJECT_CONFIG_KEYS
    and field.key not in CONTEXT_PRIMARY_SETTING_KEYS
)


@dataclass(frozen=True)
class SettingsPageSpec:
    """One Settings page identity plus its config-key ownership."""

    key: str
    nav_label: str
    builder_name: str
    config_page: bool = False
    config_keys: frozenset[str] = frozenset()
    immediate_action_ids: frozenset[str] = frozenset()


SETTINGS_PAGE_SPEC_OBJECTS: tuple[SettingsPageSpec, ...] = (
    SettingsPageSpec(
        "workspace",
        "项目列表",
        "_create_workspace_settings_page",
        immediate_action_ids=frozenset(
            {"switch_project", "refresh_registry", "import_projects"}
        ),
    ),
    SettingsPageSpec(
        "project",
        "项目",
        "_create_project_settings_page",
        config_page=True,
        config_keys=PROJECT_CONFIG_KEYS,
    ),
    SettingsPageSpec(
        "api_keys",
        "密钥",
        "_create_api_keys_settings_page",
        config_page=True,
        immediate_action_ids=frozenset(
            {"manage_gemini_keys", "manage_litellm_keys"}
        ),
    ),
    SettingsPageSpec(
        "models",
        "模型",
        "_create_models_settings_page",
        config_page=True,
        config_keys=frozenset(
            {
                "sync_model",
                "sync_embedding_model",
                "batch_model",
                "batch_embedding_model",
                "batch_thinking_level",
            }
        ),
    ),
    SettingsPageSpec(
        "profiles",
        "模型与 Provider",
        "_create_profiles_settings_page",
        config_page=True,
        config_keys=frozenset({"model_routing"}),
        immediate_action_ids=frozenset({"probe_profile"}),
    ),
    SettingsPageSpec(
        "litellm",
        "LiteLLM",
        "_create_litellm_settings_page",
        config_page=True,
        config_keys=frozenset(
            {
                "sync_backend",
                "litellm_model",
                "custom_litellm_providers",
            }
        ),
        immediate_action_ids=frozenset(
            {
                "refresh_providers",
                "refresh_models",
                "test_connection",
                "install_litellm",
                "manage_provider_keys",
            }
        ),
    ),
    SettingsPageSpec(
        "extensions",
        "扩展",
        "_create_extensions_settings_page",
        immediate_action_ids=frozenset({"install_relation_analyzer"}),
    ),
    SettingsPageSpec(
        "context",
        "上下文",
        "_create_context_settings_page",
        config_page=True,
        config_keys=CONTEXT_CONFIG_KEYS,
    ),
    SettingsPageSpec(
        "appearance",
        "外观",
        "_create_appearance_settings_page",
        config_page=True,
        config_keys=frozenset({"theme"}),
        immediate_action_ids=frozenset({"preview_theme", "download_fonts"}),
    ),
    SettingsPageSpec(
        "shortcuts",
        "快捷键",
        "_create_shortcuts_settings_page",
    ),
    SettingsPageSpec(
        "advanced",
        "高级",
        "_create_advanced_settings_page",
        config_page=True,
        config_keys=ADVANCED_CONFIG_KEYS,
    ),
)

# Legacy ``(key, label, builder)`` shape derived from the spec objects above.
# Keep this derived so page identity has one source of truth.
SETTINGS_PAGE_SPECS: tuple[tuple[str, str, str], ...] = tuple(
    (spec.key, spec.nav_label, spec.builder_name)
    for spec in SETTINGS_PAGE_SPEC_OBJECTS
)

SETTINGS_CONFIG_PAGE_KEYS: frozenset[str] = frozenset(
    spec.key for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.config_page
)

# Kept as a derived view for existing app helpers. The registry methods are the
# source of truth; do not hand-maintain a second key list.
CONFIG_SNAPSHOT_KEYS_BY_PAGE: dict[str, frozenset[str]] = {
    spec.key: spec.config_keys
    for spec in SETTINGS_PAGE_SPEC_OBJECTS
    if spec.config_keys
}

# Attribute -> settings section for lazy materialization (tests + direct access).
SETTINGS_LAZY_ATTR_TO_PAGE: dict[str, str] = {
    "_games_registry_panel": "workspace",
    "api_status_label": "api_keys",
    "api_btn": "api_keys",
    "litellm_keys_provider_combo": "api_keys",
    "litellm_keys_status_label": "api_keys",
    "litellm_keys_manage_btn": "api_keys",
    "settings_project_root_value": "project",
    "settings_go_workspace_btn": "project",
    "rag_enabled_cb": "context",
    "source_index_enabled_cb": "context",
    "bootstrap_on_build_cb": "context",
    "sync_source_index_enabled_cb": "context",
    "sync_inject_published_brief_cb": "context",
    "context_storage_game_cb": "context",
    "sync_model_combo": "models",
    "sync_embedding_combo": "models",
    "batch_model_combo": "models",
    "batch_embedding_combo": "models",
    "batch_thinking_combo": "models",
    "sync_backend_combo": "litellm",
    "litellm_provider_combo": "litellm",
    "litellm_refresh_providers_btn": "litellm",
    "litellm_clear_provider_btn": "litellm",
    "litellm_provider_catalog_status_label": "litellm",
    "litellm_model_combo": "litellm",
    "litellm_refresh_models_btn": "litellm",
    "litellm_catalog_status_label": "litellm",
    "sync_backend_hint": "litellm",
    "litellm_version_label": "litellm",
    "litellm_check_version_btn": "litellm",
    "install_litellm_btn": "litellm",
    "litellm_install_progress": "litellm",
    "litellm_credentials_box": "litellm",
    "litellm_provider_label": "litellm",
    "litellm_manage_keys_btn": "litellm",
    "litellm_credential_status_label": "litellm",
    "litellm_test_connection_btn": "litellm",
    "litellm_connection_status_label": "litellm",
    "custom_provider_table": "litellm",
    "custom_provider_add_btn": "litellm",
    "custom_provider_edit_btn": "litellm",
    "custom_provider_delete_btn": "litellm",
    "custom_provider_status_label": "litellm",
    "relation_analyzer_status_label": "extensions",
    "relation_analyzer_failure_label": "extensions",
    "relation_analyzer_install_btn": "extensions",
    "relation_analyzer_docs_btn": "extensions",
    "relation_analyzer_install_progress": "extensions",
    "theme_combo": "appearance",
    "font_install_status_label": "appearance",
    "download_fonts_btn": "appearance",
    "font_install_progress": "appearance",
}


class SettingsPageRegistry:
    """Ordered registry enforcing unique page keys and config-key ownership."""

    def __init__(self, specs: Iterable[SettingsPageSpec] = ()) -> None:
        self._specs: dict[str, SettingsPageSpec] = {}
        self._config_key_owners: dict[str, str] = {}
        self._lazy_attr_to_page: Mapping[str, str] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: SettingsPageSpec) -> None:
        if not isinstance(spec, SettingsPageSpec):
            raise TypeError("spec must be a SettingsPageSpec")
        if not spec.key:
            raise ValueError("settings page key must not be empty")
        if spec.key in self._specs:
            raise ValueError(f"duplicate settings page key: {spec.key!r}")
        if not spec.nav_label:
            raise ValueError(f"settings page {spec.key!r} needs a nav label")
        if not spec.builder_name:
            raise ValueError(f"settings page {spec.key!r} needs a builder name")
        for config_key in spec.config_keys:
            owner = self._config_key_owners.get(config_key)
            if owner is not None and owner != spec.key:
                raise ValueError(
                    f"config key {config_key!r} already owned by {owner!r}; "
                    f"cannot also own it in {spec.key!r}"
                )
            self._config_key_owners[config_key] = spec.key
        self._specs[spec.key] = spec

    def set_lazy_attr_map(self, mapping: Mapping[str, str]) -> None:
        for attr, page_key in mapping.items():
            if page_key not in self._specs:
                raise ValueError(
                    f"lazy attribute {attr!r} points at unknown page {page_key!r}"
                )
        self._lazy_attr_to_page = dict(mapping)

    def validate(self) -> None:
        """Raise ValueError when page identity or key ownership is ambiguous."""

        labels: dict[str, str] = {}
        for spec in self._specs.values():
            owner = labels.get(spec.nav_label)
            if owner is not None:
                raise ValueError(
                    f"duplicate settings nav label {spec.nav_label!r}: "
                    f"{owner!r} and {spec.key!r}"
                )
            labels[spec.nav_label] = spec.key
        for config_key, owner in self._config_key_owners.items():
            if owner not in self._specs:
                raise ValueError(
                    f"config key {config_key!r} points at unknown page {owner!r}"
                )
        self.set_lazy_attr_map(self._lazy_attr_to_page)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> tuple[SettingsPageSpec, ...]:
        return tuple(self._specs.values())

    def get(self, key: str) -> SettingsPageSpec:
        try:
            return self._specs[key]
        except KeyError as exc:
            raise KeyError(f"unknown settings page: {key!r}") from exc

    def has(self, key: str) -> bool:
        return key in self._specs

    def config_page_keys(self) -> frozenset[str]:
        return frozenset(
            spec.key for spec in self._specs.values() if spec.config_page
        )

    def config_keys_for(self, page_key: str) -> frozenset[str]:
        return self.get(page_key).config_keys

    def config_keys_for_pages(self, pages: Iterable[str]) -> frozenset[str]:
        keys: set[str] = set()
        for page_key in pages:
            keys.update(self.config_keys_for(page_key))
        return frozenset(keys)

    def owner_of(self, config_key: str) -> str | None:
        return self._config_key_owners.get(config_key)

    def page_for_lazy_attr(self, attr: str) -> str | None:
        return self._lazy_attr_to_page.get(attr)


def build_default_registry() -> SettingsPageRegistry:
    registry = SettingsPageRegistry(SETTINGS_PAGE_SPEC_OBJECTS)
    registry.set_lazy_attr_map(SETTINGS_LAZY_ATTR_TO_PAGE)
    registry.validate()
    return registry
