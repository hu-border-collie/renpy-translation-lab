"""Apply collected Settings values onto a translator_config object.

Qt-free so the unique save transaction can be tested without MainWindow.
The host still writes files through ProjectState / config_store.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gemini_model_catalog import write_model_catalog_extras
from project_context_settings import resolve_batch_context_flags

from ..litellm_settings import write_sync_backend_models
from ..settings_schema import (
    ADVANCED_SETTING_FIELDS,
    apply_advanced_settings,
    read_advanced_settings,
    resolve_project_analysis_flags_for_save,
    validate_advanced_settings,
)
from ..theme_helpers import write_gui_theme_to_config
from ..user_copy import CUSTOM_LITELLM_PROVIDER_COPY, MODEL_ROUTING_RUNTIME_COPY
from .registry import WORKSPACE_MANAGED_KEYS

_CONTEXT_FLAG_KEYS = (
    "rag_enabled",
    "source_index_enabled",
    "bootstrap_on_build",
    "sync_source_index_enabled",
    "sync_project_analysis_inject_enabled",
)


@dataclass(frozen=True)
class SettingsSaveExtras:
    """Host-only flags that are not page-owned collect() keys."""

    game_root: str | None = None
    custom_providers_modified: bool = False
    custom_providers_load_error: str = ""
    batch_thinking_user_changed: bool = False


@dataclass
class SettingsSaveApplyResult:
    """Outcome of applying collected values onto a config object (no disk I/O)."""

    project_context_flags: dict[str, Any] = field(default_factory=dict)
    advanced_errors: dict[str, str] = field(default_factory=dict)
    block_page: str | None = None
    block_title: str | None = None
    block_message: str | None = None
    block_warning: bool = False

    @property
    def ok(self) -> bool:
        return not self.advanced_errors and self.block_page is None


def _config_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _ensure_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    section = config.get(key)
    if not isinstance(section, dict):
        section = {}
        config[key] = section
    return section


def _supports_batch_thinking(model_name: Any) -> bool:
    return _config_string(model_name).startswith("gemini-3")


def _should_save_batch_thinking_level(
    batch_config: Mapping[str, Any],
    batch_model: str,
    thinking_level: str,
    user_changed: bool,
) -> bool:
    return (
        bool(thinking_level)
        or (_supports_batch_thinking(batch_model) and user_changed)
        or "thinking_level" in batch_config
    )


def _sync_models_for_save(existing_models: Any, selected_model: str) -> list[str] | None:
    existing: list[str] = []
    if isinstance(existing_models, list):
        for model in existing_models:
            cleaned = _config_string(model)
            if cleaned and cleaned not in existing:
                existing.append(cleaned)
    else:
        cleaned = _config_string(existing_models)
        if cleaned:
            existing.append(cleaned)
    if not selected_model:
        return existing or None
    return [selected_model, *[model for model in existing if model != selected_model]]


def _custom_provider_entries(value: object) -> list[dict[str, Any]]:
    if not value:
        return []
    entries: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        return [dict(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    for item in value:
        if isinstance(item, Mapping):
            entries.append(dict(item))
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            mapping = _mapping_from_pairs(item)
            if mapping is not None:
                entries.append(mapping)
    return entries


def _mapping_from_pairs(item: Sequence[object]) -> dict[str, Any] | None:
    if all(isinstance(pair, Sequence) and len(pair) == 2 for pair in item):
        try:
            return dict(item)
        except (TypeError, ValueError):
            return None
    return None


def apply_collected_settings(
    config: dict[str, Any],
    collected: Mapping[str, object],
    *,
    original_config: Mapping[str, Any],
    extras: SettingsSaveExtras | None = None,
) -> SettingsSaveApplyResult:
    """Mutate ``config`` from a flat collect() snapshot. Does not write disk.

    Only keys present in ``collected`` are applied. Host extras cover flags
    that are not page-owned collect() keys. Callers still own the two-file
    write and rollback.
    """

    routing_from_page = "model_routing" in collected
    routing_removed = routing_from_page and collected.get("model_routing") is None
    if routing_removed:
        # Explicit user removal: fall back to legacy config instead of
        # blocking every save on a broken hand-edited section.
        config.pop("model_routing", None)
    effective_routing = (
        None
        if routing_removed
        else (
            collected.get("model_routing")
            if routing_from_page
            else config.get("model_routing")
        )
    )
    if effective_routing is not None:
        from model_routing_config import validate_model_routing_section

        if not isinstance(effective_routing, Mapping):
            return SettingsSaveApplyResult(
                block_page="profiles" if routing_from_page else "models",
                block_title=MODEL_ROUTING_RUNTIME_COPY["invalid_title"],
                block_message=MODEL_ROUTING_RUNTIME_COPY["invalid_message"],
            )
        if validate_model_routing_section(dict(effective_routing)):
            return SettingsSaveApplyResult(
                block_page="profiles" if routing_from_page else "models",
                block_title=MODEL_ROUTING_RUNTIME_COPY["invalid_title"],
                block_message=MODEL_ROUTING_RUNTIME_COPY["invalid_message"],
            )
        if routing_from_page and not routing_removed:
            # Preserve the page's unknown fields: apply the edited section as-is.
            config["model_routing"] = copy.deepcopy(dict(effective_routing))
    extras = extras or SettingsSaveExtras()
    project_context_flags: dict[str, Any] = {}
    for key in _CONTEXT_FLAG_KEYS:
        if key in collected:
            project_context_flags[key] = bool(collected[key])

    if extras.custom_providers_modified and extras.custom_providers_load_error:
        return SettingsSaveApplyResult(
            project_context_flags=project_context_flags,
            block_page="litellm",
            block_title=CUSTOM_LITELLM_PROVIDER_COPY["load_error_title"],
            block_message=CUSTOM_LITELLM_PROVIDER_COPY["load_error_save_blocked"].format(
                error=extras.custom_providers_load_error
            ),
            block_warning=True,
        )

    sync_backend = _config_string(collected.get("sync_backend")).lower()
    litellm_model = _config_string(collected.get("litellm_model"))
    if sync_backend == "litellm" and not litellm_model:
        return SettingsSaveApplyResult(
            project_context_flags=project_context_flags,
            block_page="litellm",
            block_title="请配置 LiteLLM 模型",
            block_message="启用 LiteLLM 前，请填写带 provider 前缀的模型名称。",
        )

    if collected.get("context_storage_location") in {"game", "tool"}:
        context_storage_config = _ensure_section(config, "context_storage")
        context_storage_config["location"] = collected["context_storage_location"]
        context_storage_config["game_dir_name"] = (
            _config_string(
                context_storage_config.get(
                    "game_dir_name",
                    context_storage_config.get(
                        "directory_name",
                        context_storage_config.get("directory", "translation_context"),
                    ),
                )
            )
            or "translation_context"
        )

    writes_sync_models = any(
        key in collected for key in ("sync_backend", "sync_model", "litellm_model")
    )
    writes_custom_providers = (
        "custom_litellm_providers" in collected or extras.custom_providers_modified
    )
    if writes_sync_models or writes_custom_providers:
        sync_config = _ensure_section(config, "sync")
        if writes_sync_models:
            if sync_backend not in {"gemini", "litellm"}:
                sync_backend = "gemini"
            sync_model = write_sync_backend_models(
                sync_config,
                sync_backend,
                _config_string(collected.get("sync_model")),
                litellm_model,
            )
            if "models" in sync_config:
                sync_models = _sync_models_for_save(
                    sync_config.get("models"), str(sync_model)
                )
                if sync_models:
                    sync_config["models"] = sync_models
                else:
                    sync_config.pop("models", None)
        if writes_custom_providers:
            custom_entries = _custom_provider_entries(
                collected.get("custom_litellm_providers")
            )
            if custom_entries:
                sync_config["custom_litellm_providers"] = custom_entries
            elif extras.custom_providers_modified:
                sync_config.pop("custom_litellm_providers", None)

    if "batch_model" in collected:
        batch_config = _ensure_section(config, "batch")
        batch_config["model"] = _config_string(collected.get("batch_model"))
    if "sync_embedding_model" in collected:
        sync_rag_config = _ensure_section(_ensure_section(config, "sync"), "rag")
        sync_rag_config["embedding_model"] = _config_string(
            collected.get("sync_embedding_model")
        )
    if "batch_embedding_model" in collected:
        batch_rag_config = _ensure_section(_ensure_section(config, "batch"), "rag")
        batch_rag_config["embedding_model"] = _config_string(
            collected.get("batch_embedding_model")
        )
    if (
        "batch_thinking_level" in collected
        or extras.batch_thinking_user_changed
    ):
        batch_config = _ensure_section(config, "batch")
        thinking_level = _config_string(collected.get("batch_thinking_level"))
        batch_model = _config_string(batch_config.get("model"))
        if _should_save_batch_thinking_level(
            batch_config,
            batch_model,
            thinking_level,
            extras.batch_thinking_user_changed,
        ):
            batch_config["thinking_level"] = thinking_level

    if "theme" in collected:
        write_gui_theme_to_config(config, collected.get("theme"))

    advanced_values = {
        field.key: collected[field.key]
        for field in ADVANCED_SETTING_FIELDS
        if field.key not in WORKSPACE_MANAGED_KEYS and field.key in collected
    }
    complete_advanced_values: dict[str, Any] = {}
    if advanced_values:
        complete_advanced_values = read_advanced_settings(config)
        complete_advanced_values.update(advanced_values)
        if extras.game_root:
            complete_advanced_values["game_root"] = extras.game_root
        errors = validate_advanced_settings(
            complete_advanced_values,
            translator_config=config,
        )
        if errors:
            return SettingsSaveApplyResult(
                project_context_flags=project_context_flags,
                advanced_errors=dict(errors),
            )
        apply_advanced_settings(config, complete_advanced_values)
        original_project_analysis = original_config.get("batch")
        if not isinstance(original_project_analysis, Mapping):
            original_project_analysis = {}
        original_project_analysis = original_project_analysis.get("project_analysis")
        if not isinstance(original_project_analysis, Mapping):
            original_project_analysis = {}
        saved_project_analysis = _ensure_section(
            _ensure_section(config, "batch"), "project_analysis"
        )
        for setting_key in ("enabled", "inject_published_brief"):
            if setting_key in original_project_analysis:
                saved_project_analysis[setting_key] = original_project_analysis[
                    setting_key
                ]
            else:
                saved_project_analysis.pop(setting_key, None)
        write_model_catalog_extras(
            config,
            translation_models=list(
                complete_advanced_values.get("catalog_gemini_models") or []
            ),
            embedding_models=list(
                complete_advanced_values.get("catalog_gemini_embedding_models") or []
            ),
        )

    saved_context_flags = resolve_batch_context_flags(
        config,
        extras.game_root,
    )
    project_context_flags.update(
        resolve_project_analysis_flags_for_save(
            saved_context_flags,
            advanced_values,
            complete_advanced_values,
        )
    )
    return SettingsSaveApplyResult(project_context_flags=project_context_flags)
