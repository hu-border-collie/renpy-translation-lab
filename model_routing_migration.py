"""Deterministic, credential-free legacy-to-v1 migration planning (#348 P1).

This module prepares and checks a candidate using legacy routing semantics.
The production runtime consumes installed v1 configurations.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Any, Mapping

import model_profile as routing
from embedding_runtime import parse_embedding_runtime_settings
from gemini_model_catalog import DEFAULT_GEMINI_TRANSLATION_MODEL
from litellm_provider_config import custom_provider_registry
from model_routing_config import legacy_fields_present, _find_sensitive_keys
from model_routing_reader import checked_section, read_routing_plan, read_embedding_settings


@dataclass(frozen=True)
class MigrationPreview:
    """Detached candidate and metadata; config must not be logged/exported as a report."""

    config: dict[str, Any]
    status: str
    mapped_fields: tuple[str, ...]
    profile_ids: tuple[str, ...]
    provider_ids: tuple[str, ...]

    def public_report(self) -> dict[str, Any]:
        """Expose IDs and field mappings, never arbitrary config values."""
        return {
            "status": self.status, "source_schema": 1 if self.status == "already_current" else 0,
            "target_schema": 1, "mapped_fields": list(self.mapped_fields),
            "profile_ids": list(self.profile_ids), "provider_ids": list(self.provider_ids),
            "preserved_fields": ["all original fields (legacy and unknown)"],
            "warnings": ["Only the specified file is changed; installed v1 configurations take effect on the next runtime load"],
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise routing.ModelRoutingConfigError(f"{name} must be an object")
    return value


def _signature(plan: routing.ModelRoutingPlan) -> dict[str, Any]:
    result = {}
    for stage, route in plan.routes.items():
        profile = plan.profiles[route.profile_id].to_manifest_dict()
        for field in ("id", "label", "embedding_profile_id"):
            profile.pop(field)
        result[stage] = {
            "strategy": route.strategy.value, "profile": profile,
            "capabilities": plan.capabilities[route.profile_id].to_manifest_dict(),
        }
    return result


def preview_migration(
    config: Mapping[str, Any], *, game_config: Mapping[str, Any] | None = None,
) -> MigrationPreview:
    """Build a v1 candidate; refuse any unrepresentable or invalid legacy input.

    Old fields are retained byte-semantically as JSON values for rollback and
    the staged P1 runtime. A repeated call validates v1 and performs no migration.
    No key file, environment value, keyring or provider endpoint is accessed.
    """
    if not isinstance(config, dict):
        raise routing.ModelRoutingConfigError("Configuration must be an object")
    sensitive = _find_sensitive_keys(config, "$")
    rotation = config.get("rotation")
    key_rotation = rotation.get("api_key") if isinstance(rotation, dict) else None
    if isinstance(key_rotation, dict) and set(key_rotation) <= {"enabled"} and isinstance(key_rotation.get("enabled"), bool):
        # This exact legacy object is a boolean policy, never a credential value.
        sensitive = [issue for issue in sensitive if issue.path != "$.rotation.api_key"]
    if sensitive:
        raise routing.ModelRoutingConfigError("Embedded credential fields prevent migration")
    candidate = copy.deepcopy(config)
    if "model_routing" in config:
        section = checked_section(config)
        return MigrationPreview(candidate, "already_current", (),
                                tuple(section["profiles"]), tuple(section["providers"]))
    sync = _mapping(config.get("sync"), "sync")
    batch = _mapping(config.get("batch"), "batch")
    for key in ("project_analysis", "final_review"):
        stage_config = _mapping(batch.get(key), f"batch.{key}")
        if "model" in stage_config and not isinstance(stage_config["model"], str):
            raise routing.ModelRoutingConfigError(f"batch.{key}.model must be a string")
    for name, section_cfg in (("sync", sync), ("batch", batch)):
        for key in ("model", "backend"):
            if key in section_cfg and not isinstance(section_cfg[key], str):
                raise routing.ModelRoutingConfigError(f"{name}.{key} must be a string")
    pool = sync.get("models", [])
    if not isinstance(pool, list) or any(not isinstance(m, str) or not m.strip() for m in pool):
        raise routing.ModelRoutingConfigError("sync.models must be a list of non-empty model ids")
    selected = str(sync.get("model") or "").strip()
    if pool and (not selected or pool[0].strip() != selected):
        raise routing.ModelRoutingConfigError("Ambiguous legacy model selection: set sync.model to the first sync.models entry")
    rotation = _mapping(config.get("rotation"), "rotation")
    rotation_model = _mapping(rotation.get("model"), "rotation.model")
    if rotation_model.get("enabled") not in (None, False) and not pool:
        raise routing.ModelRoutingConfigError("Implicit rotation pool requires an explicit sync.models snapshot before migration")
    batch_selected = str(batch.get("model") or (game_config or {}).get("batch_model") or DEFAULT_GEMINI_TRANSLATION_MODEL).strip()
    if not selected and batch_selected != DEFAULT_GEMINI_TRANSLATION_MODEL:
        raise routing.ModelRoutingConfigError("Ambiguous legacy defaults: set sync.model explicitly before migration")
    custom = custom_provider_registry(sync.get("custom_litellm_providers"), allow_import=False)
    legacy = routing.resolve_routing_plan(config, custom_providers=custom, game_config=game_config)
    # Validate without credential availability probes or network access.
    errors = routing.validate_routing_plan(legacy, custom_providers=custom, environ={})
    if errors:
        raise routing.ModelRoutingConfigError("Legacy routing is invalid: " + ", ".join(sorted({e.code for e in errors})))
    profiles: dict[str, Any] = {}
    providers: dict[str, Any] = {}
    ids = {"primary": "legacy-sync", "batch": "legacy-batch",
           "project_analysis_model": "legacy-project-analysis",
           "final_review_model": "legacy-final-review"}
    for slot, profile in legacy.profiles.items():
        # Adapter prefix avoids collisions between Gemini-direct and custom IDs.
        provider_id = f"{profile.adapter}-{profile.provider}"
        metadata = custom.get(profile.provider) if profile.adapter == "litellm" else None
        providers[provider_id] = {
            "label": metadata.label if metadata else profile.provider,
            "adapter": profile.adapter, "provider": profile.provider,
            "credential_ref": profile.credential_ref.to_manifest_dict(),
            "base_url": profile.base_url, "models_url": metadata.models_url if metadata else "",
        }
        profiles[ids[slot]] = {
            "label": profile.label, "provider_id": provider_id, "model": profile.model,
            "models": list(profile.models), "params": dict(profile.params),
            "capability_overrides": dict(profile.capability_overrides),
        }
    # Keep dormant custom connections addressable, not only the selected provider.
    for upstream, metadata in custom.items():
        providers.setdefault(f"litellm-{upstream}", {
            "label": metadata.label, "adapter": "litellm", "provider": upstream,
            "base_url": metadata.base_url, "models_url": metadata.models_url,
            "credential_ref": {"kind": "keyring" if metadata.requires_key else "none",
                               "name": upstream, "env_name": metadata.api_key_env},
        })
    # RAG policy stays in place; freeze connection settings separately per path.
    for execution, cfg in (("sync", sync), ("batch", batch)):
        rag = _mapping(cfg.get("rag"), f"{execution}.rag")
        settings = parse_embedding_runtime_settings(rag)
        embedding_id = f"legacy-{execution}-embedding"
        provider_id = f"embedding-{execution}"
        providers[provider_id] = {
            "label": f"{execution} embedding", "provider": settings.provider,
            "adapter": "gemini" if settings.backend == "gemini" else "litellm",
            "base_url": settings.endpoint,
            "credential_ref": (
                {"kind": "api_keys_json", "name": "api_keys"} if settings.backend == "gemini"
                else {"kind": "env", "name": settings.api_key_env} if settings.api_key_env
                else {"kind": "none"}
            ),
        }
        profiles[embedding_id] = {
            "label": f"{execution} embedding", "provider_id": provider_id,
            "model": settings.model, "models": [], "purpose": "embedding",
            "capability_overrides": {"embedding": True, "sync_generation": False, "remote_batch": False},
            # Includes adapter semantics: openai_compatible must not turn into generation routing.
            "params": {key: value for key, value in asdict(settings).items()
                       if key not in ("provider", "model", "endpoint", "api_key_env")},
        }
        profiles[f"legacy-{execution}"]["embedding_profile_id"] = embedding_id
    for slot in ("project_analysis_model", "final_review_model"):
        if slot in ids and ids[slot] in profiles:
            scope = "sync" if slot == "project_analysis_model" else "batch"
            profiles[ids[slot]]["embedding_profile_id"] = f"legacy-{scope}-embedding"
    section = {
        "schema_version": 1, "providers": providers, "profiles": profiles,
        "defaults": {"primary_profile_id": "legacy-batch", "execution_strategy": "gemini_batch"},
        "legacy_entrypoints": {"sync_profile_id": "legacy-sync", "batch_profile_id": "legacy-batch"},
        "routes": {
            "project_analysis": {"profile_id": ids[legacy.routes["project_analysis"].profile_id], "strategy": "sync"},
            "final_review": {"profile_id": ids[legacy.routes["final_review"].profile_id], "strategy": "gemini_batch"},
        },
    }
    candidate["model_routing"] = section
    checked_section(candidate)
    for execution in ("sync", "gemini_batch"):
        old_plan = routing.resolve_routing_plan(config, custom_providers=custom,
                                               game_config=game_config, execution=execution)
        new_plan = read_routing_plan(candidate, legacy_execution=execution)
        if _signature(old_plan) != _signature(new_plan):
            raise routing.ModelRoutingConfigError("Migration changes effective routing")
        if read_embedding_settings(config, execution=execution) != read_embedding_settings(candidate, execution=execution):
            raise routing.ModelRoutingConfigError("Migration changes effective embedding settings")
    return MigrationPreview(candidate, "ready", tuple(".".join(p) for p in legacy_fields_present(config)),
                            tuple(profiles), tuple(providers))
