"""Offline compatibility reader for #348 P1; production wiring belongs to P2.

The new section is an all-or-nothing source. No field falls back to legacy
configuration when it exists, and this module never resolves credentials.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import model_profile as routing
from embedding_runtime import EmbeddingRuntimeSettings, parse_embedding_runtime_settings
from litellm_provider_config import CustomLiteLLMProvider, custom_provider_registry
from model_routing_config import validate_model_routing_section


def checked_section(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached validated section, refusing malformed/newer schemas."""
    section = config.get("model_routing")
    issues = validate_model_routing_section(section)
    if issues:
        # Do not echo values or unknown field names, which may contain secrets.
        raise routing.ModelRoutingConfigError(
            "Invalid model_routing: " + ", ".join(sorted({i.code for i in issues}))
        )
    assert isinstance(section, dict)
    return copy.deepcopy(section)


def section_custom_providers(section: Mapping[str, Any]) -> dict[str, CustomLiteLLMProvider]:
    """Reconstruct custom connection metadata without consulting legacy fields."""
    result = {}
    for provider in section["providers"].values():
        if provider["adapter"] != "litellm" or not provider.get("base_url"):
            continue
        ref = provider["credential_ref"]
        upstream = provider["provider"]
        result[upstream] = CustomLiteLLMProvider(
            id=upstream, label=provider["label"], base_url=provider["base_url"],
            models_url=provider.get("models_url", ""),
            api_key_env=ref.get("env_name", ""), requires_key=ref["kind"] != "none",
        )
    return result


def read_routing_plan(
    config: Mapping[str, Any], *, legacy_execution: str | None = None,
    game_config: Mapping[str, Any] | None = None,
) -> routing.ModelRoutingPlan:
    """Read legacy or v1 routing without activating it in the runtime.

    ``legacy_execution`` selects a migrated old command's base profile; it
    changes translation/keyword/revision only. Explicit stage routes win.
    Missing compatibility pointers fail closed instead of reading old fields.
    """
    if "model_routing" not in config:
        sync = config.get("sync") or {}
        custom = custom_provider_registry(sync.get("custom_litellm_providers"), allow_import=False)
        return routing.resolve_routing_plan(
            config, execution=legacy_execution or "gemini_batch",
            custom_providers=custom, game_config=game_config,
        )
    section = checked_section(config)
    profiles = {}
    for profile_id, raw in section["profiles"].items():
        provider = section["providers"][raw["provider_id"]]
        profiles[profile_id] = routing.ModelProfile(
            id=profile_id, label=raw["label"], adapter=provider["adapter"],
            provider=provider["provider"], model=raw["model"],
            credential_ref=routing.CredentialRef.from_manifest_dict(provider["credential_ref"]),
            models=tuple(raw.get("models") or [raw["model"]]),
            base_url=provider.get("base_url", ""),
            capability_overrides=raw.get("capability_overrides", {}),
            params=raw.get("params", {}), embedding_profile_id=raw.get("embedding_profile_id", ""),
        )
    primary = section["defaults"]["primary_profile_id"]
    strategy = section["defaults"]["execution_strategy"]
    base = primary
    if legacy_execution is not None:
        if legacy_execution not in ("sync", "gemini_batch"):
            raise routing.ModelRoutingConfigError("Unsupported legacy execution")
        key = "sync_profile_id" if legacy_execution == "sync" else "batch_profile_id"
        base = section.get("legacy_entrypoints", {}).get(key)
        if not isinstance(base, str) or base not in profiles:
            raise routing.ModelRoutingConfigError("Missing legacy entrypoint profile")
        strategy = legacy_execution
    routes = {}
    for stage in ("translation", "keyword", "revision", "project_analysis", "final_review"):
        raw_route = section.get("routes", {}).get(stage, {})
        default_profile = base if stage in ("translation", "keyword", "revision") else primary
        default_strategy = strategy if stage in ("translation", "keyword", "revision") else section["defaults"]["execution_strategy"]
        routes[stage] = routing.TaskRoute(
            stage=stage, profile_id=raw_route.get("profile_id", default_profile),
            strategy=routing.ExecutionStrategy(raw_route.get("strategy", default_strategy)),
            source="stage_config" if raw_route else "inherited",
        )
    # A/B is an old sync-only entrypoint, not a new configurable task stage.
    ab_profile = section.get("legacy_entrypoints", {}).get("sync_profile_id", primary)
    routes["ab_experiment"] = routing.TaskRoute(
        stage="ab_experiment", profile_id=ab_profile,
        strategy=routing.ExecutionStrategy.SYNC, source="inherited",
    )
    custom = section_custom_providers(section)
    return routing.ModelRoutingPlan(
        schema_version=routing.MODEL_PROFILE_SCHEMA_VERSION,
        primary_profile_id=primary, profiles=profiles, routes=routes,
        capabilities={key: routing.resolve_capabilities(value, custom_providers=custom)
                      for key, value in profiles.items()},
        created_at=routing.utc_now_iso(),
    )


def read_embedding_settings(config: Mapping[str, Any], *, execution: str) -> EmbeddingRuntimeSettings:
    """Read one path's embedding connection entirely from legacy or v1 fields."""
    if execution not in ("sync", "gemini_batch"):
        raise routing.ModelRoutingConfigError("Unsupported embedding execution")
    if "model_routing" not in config:
        scope = "sync" if execution == "sync" else "batch"
        return parse_embedding_runtime_settings((config.get(scope) or {}).get("rag"))
    section = checked_section(config)
    key = "sync_profile_id" if execution == "sync" else "batch_profile_id"
    base = section.get("legacy_entrypoints", {}).get(key, section["defaults"]["primary_profile_id"])
    embedding_id = section["profiles"][base].get("embedding_profile_id")
    if not embedding_id:
        raise routing.ModelRoutingConfigError("Missing embedding profile")
    raw = section["profiles"][embedding_id]
    if raw.get("purpose") != "embedding":
        raise routing.ModelRoutingConfigError("Embedding profile purpose must be explicit")
    provider = section["providers"][raw["provider_id"]]
    params = raw.get("params", {})
    backend = params.get("backend")
    if backend not in ("gemini", "openai_compatible"):
        raise routing.ModelRoutingConfigError("Missing or unsupported embedding backend")
    expected_adapter = "gemini" if backend == "gemini" else "litellm"
    if provider["adapter"] != expected_adapter:
        raise routing.ModelRoutingConfigError("Embedding backend and adapter disagree")
    ref = provider["credential_ref"]
    if backend == "openai_compatible" and ref["kind"] not in ("env", "none"):
        raise routing.ModelRoutingConfigError("Unsupported embedding credential reference")
    return parse_embedding_runtime_settings({
        "embedding_backend": backend, "embedding_provider": provider["provider"],
        "embedding_model": raw["model"], "embedding_endpoint": provider.get("base_url", ""),
        "embedding_api_key_env": ref.get("name", "") if ref["kind"] == "env" else "",
        "output_dimensionality": params.get("output_dimension"),
        "embedding_timeout_seconds": params.get("timeout_seconds"),
        "query_task_type": params.get("native_query_task_type"),
        "document_task_type": params.get("native_document_task_type"),
    })
