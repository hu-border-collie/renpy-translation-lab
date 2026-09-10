"""Compatibility reader and production snapshots for versioned model routing.

The new section is an all-or-nothing source. No field falls back to legacy
configuration when it exists, and this module never resolves credentials.
"""
from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
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
    """Build generation connections only; embedding endpoints have their own reader.

    Upstream names are shared across purposes, so an embedding connection must
    not shadow a built-in or custom generation provider with the same name.
    """
    generation_providers = {
        profile["provider_id"]
        for profile in section["profiles"].values()
        if isinstance(profile, Mapping) and profile.get("purpose") != "embedding"
    }
    result = {}
    for provider_id, provider in section["providers"].items():
        if provider_id not in generation_providers:
            continue
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


_PRIMARY_PROFILE_OVERRIDE: ContextVar[str | None] = ContextVar(
    "model_routing_primary_profile_override",
    default=None,
)


@contextmanager
def primary_profile_override(profile_id: str | None) -> Iterator[None]:
    """Pin translation generation to one ModelProfile for this call stack.

    CLI flags and the GUI selector use this so an explicit user choice wins
    over ``defaults.primary_profile_id`` without rewriting the config file.
    The value is scoped, so concurrent callers and later loads are unaffected.
    """

    token = _PRIMARY_PROFILE_OVERRIDE.set(str(profile_id or "").strip() or None)
    try:
        yield
    finally:
        _PRIMARY_PROFILE_OVERRIDE.reset(token)


def active_primary_profile_override() -> str | None:
    """Return the scoped ModelProfile override, if one is active."""
    return _PRIMARY_PROFILE_OVERRIDE.get()


def apply_primary_profile_override(
    config: Mapping[str, Any],
    *,
    execution: str,
    strict: bool = True,
) -> dict[str, Any]:
    """Return a detached section with the scoped profile pinned to translation.

    When *strict* is false an incompatible execution strategy leaves the
    section untouched instead of failing, which lets the compatibility
    projection render scopes a profile cannot serve.
    """

    section = checked_section(config)
    profile_id = active_primary_profile_override()
    if not profile_id:
        return section
    profiles = section.get("profiles") or {}
    if profile_id not in profiles:
        raise routing.ModelRoutingConfigError(f"Unknown ModelProfile: {profile_id}")
    raw = profiles[profile_id]
    if str(raw.get("purpose") or "generation") == "embedding":
        raise routing.ModelRoutingConfigError(
            "Embedding profiles cannot run generation tasks"
        )
    strategy = routing.ExecutionStrategy(execution)
    if strategy is routing.ExecutionStrategy.GEMINI_BATCH:
        provider = (section.get("providers") or {}).get(raw.get("provider_id")) or {}
        if provider.get("adapter") != "gemini":
            if strict:
                raise routing.ModelRoutingConfigError(
                    "Selected ModelProfile cannot run the gemini_batch strategy"
                )
            return section
    section = copy.deepcopy(section)
    defaults = section.setdefault("defaults", {})
    defaults["primary_profile_id"] = profile_id
    # Keep the aggregate root valid: the schema requires the default strategy
    # to be executable by the default profile.
    defaults["execution_strategy"] = strategy.value
    routes = dict(section.get("routes") or {})
    routes["translation"] = {
        "profile_id": profile_id,
        "strategy": strategy.value,
    }
    section["routes"] = routes
    return section


def _profiles_from_section(
    section: Mapping[str, Any],
) -> dict[str, routing.ModelProfile]:
    """Build the detached ModelProfile map shared by readers and selectors."""

    profiles: dict[str, routing.ModelProfile] = {}
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
    return profiles


def profile_strategy_choices(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return credential-free ModelProfile/ExecutionStrategy selector data.

    The result exposes only public profile metadata and machine capability
    reasons; it never resolves or echoes credentials and is safe to render in
    the GUI or print in a machine command.
    """

    section = checked_section(config)
    profiles = _profiles_from_section(section)
    custom = section_custom_providers(section)
    defaults = section.get("defaults") or {}
    entries = []
    for profile_id in sorted(profiles):
        profile = profiles[profile_id]
        raw = section["profiles"][profile_id]
        if str(raw.get("purpose") or "generation") == "embedding":
            continue
        capabilities = routing.resolve_capabilities(profile, custom_providers=custom)
        strategies: list[str] = []
        unsupported: dict[str, str] = {}
        if capabilities.sync_generation.supported:
            strategies.append("sync")
        else:
            unsupported["sync"] = "missing_sync_generation"
        if profile.adapter == routing.ADAPTER_GEMINI and capabilities.remote_batch.supported:
            strategies.append("gemini_batch")
        else:
            unsupported["gemini_batch"] = (
                "missing_gemini_adapter"
                if profile.adapter != routing.ADAPTER_GEMINI
                else "missing_remote_batch"
            )
        entries.append({
            "id": profile_id,
            "label": profile.label,
            "model": profile.model,
            "adapter": profile.adapter,
            "purpose": str(raw.get("purpose") or "generation"),
            "strategies": tuple(strategies),
            "unsupported": dict(unsupported),
            "is_default": profile_id == defaults.get("primary_profile_id"),
        })
    return {
        "defaults": {
            "primary_profile_id": str(defaults.get("primary_profile_id") or ""),
            "execution_strategy": str(defaults.get("execution_strategy") or ""),
        },
        "profiles": tuple(entries),
    }


def read_routing_plan(
    config: Mapping[str, Any], *, legacy_execution: str | None = None,
    game_config: Mapping[str, Any] | None = None,
) -> routing.ModelRoutingPlan:
    """Read legacy or v1 routing without I/O or mutating runtime settings.

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
    profiles = _profiles_from_section(section)
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


def read_embedding_settings(config: Mapping[str, Any], *, execution: str, profile_id: str | None = None) -> EmbeddingRuntimeSettings:
    """Read one path's embedding connection entirely from legacy or v1 fields."""
    if execution not in ("sync", "gemini_batch"):
        raise routing.ModelRoutingConfigError("Unsupported embedding execution")
    if "model_routing" not in config:
        scope = "sync" if execution == "sync" else "batch"
        return parse_embedding_runtime_settings((config.get(scope) or {}).get("rag"))
    section = checked_section(config)
    key = "sync_profile_id" if execution == "sync" else "batch_profile_id"
    base = section.get("legacy_entrypoints", {}).get(key, section["defaults"]["primary_profile_id"])
    base = profile_id or base
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


def resolve_runtime_plan(
    config,
    *,
    execution,
    stage_overrides=None,
    created_at="",
    config_origins=(),
    strict_override=True,
):
    """Freeze a v1 plan for a legacy entrypoint without consulting old model fields.

    Explicit model overrides retain the selected connection and credential reference;
    changing provider requires selecting a profile instead of a model string.
    """
    strategy = routing.ExecutionStrategy(execution)
    section = apply_primary_profile_override(
        config,
        execution=execution,
        strict=strict_override,
    )
    plan = read_routing_plan(
        {"model_routing": section},
        legacy_execution=strategy.value,
    )
    # Existing entrypoints cannot execute arbitrary stage strategies yet.
    for stage, expected in (("project_analysis", "sync"), ("final_review", "gemini_batch")):
        if plan.routes[stage].strategy.value != expected:
            raise routing.ModelRoutingConfigError("Unsupported legacy stage execution strategy")
    generation_ids = {route.profile_id for route in plan.routes.values()}
    for profile in plan.profiles.values():
        if profile.id in generation_ids and profile.params:
            raise routing.ModelRoutingConfigError("Profile params are not supported by legacy entrypoints; retain execution parameters in sync/batch")
        if profile.credential_ref.kind == "api_keys_json" and profile.credential_ref.name != "api_keys":
            raise routing.ModelRoutingConfigError("Unsupported Gemini credential slot")
        if profile.adapter == "gemini" and profile.credential_ref.kind != "api_keys_json":
            raise routing.ModelRoutingConfigError("Gemini legacy entrypoints require api_keys_json credentials")
    profiles, routes, capabilities = dict(plan.profiles), dict(plan.routes), dict(plan.capabilities)
    for stage, model in (stage_overrides or {}).items():
        if not str(model or "").strip():
            continue
        route = routes[stage]
        profile = profiles[route.profile_id]
        model = str(model).strip()
        if model == profile.model:
            continue
        if profile.adapter == "litellm":
            if model.split("/", 1)[0] != profile.provider or "/" not in model:
                raise routing.ModelRoutingConfigError("Model override must retain the selected provider")
        elif routing.is_provider_prefixed_model_id(model):
            raise routing.ModelRoutingConfigError("Gemini profile requires a Gemini model")
        profile_id = f"{stage}_override"
        profiles[profile_id] = replace(profile, id=profile_id, model=model, models=(model,))
        routes[stage] = replace(route, profile_id=profile_id, source=routing.ROUTE_SOURCE_EXPLICIT)
        capabilities[profile_id] = routing.resolve_capabilities(
            profiles[profile_id], custom_providers=section_custom_providers(checked_section(config)),
        )
    return replace(plan, profiles=profiles, routes=routes, capabilities=capabilities,
                   created_at=created_at or plan.created_at, config_origins=tuple(config_origins))


def runtime_settings_view(config):
    """Project v1 connections into legacy loader fields in memory only.

    Execution policy stays in sync/batch. Model and embedding connection fields
    are replaced as a unit, so obsolete retained fields cannot affect requests.
    The original mapping is never mutated or persisted.
    """
    if "model_routing" not in config:
        return config
    checked_section(config)
    result = copy.deepcopy(config)
    for scope, execution in (("sync", "sync"), ("batch", "gemini_batch")):
        plan = resolve_runtime_plan(
            config,
            execution=execution,
            strict_override=False,
        )
        route = plan.routes["translation"]
        profile = plan.profiles[route.profile_id]
        if not isinstance(result.get(scope), dict):
            result[scope] = {}
        target = result[scope]
        target["model"] = profile.model
        if scope == "sync":
            target["backend"] = profile.adapter
            target["models"] = [profile.model, *(model for model in profile.models if model != profile.model)]
            target["custom_litellm_providers"] = []
        for stage in ("project_analysis", "final_review"):
            stage_route = plan.routes[stage]
            if not isinstance(result.get("batch"), dict):
                result["batch"] = {}
            if not isinstance(result["batch"].get(stage), dict):
                result["batch"][stage] = {}
            result["batch"][stage]["model"] = plan.profiles[stage_route.profile_id].model
        embedding = read_embedding_settings(config, execution=execution, profile_id=profile.id)
        if not isinstance(target.get("rag"), dict):
            target["rag"] = {}
        rag = target["rag"]
        for key in tuple(rag):
            if key.startswith("embedding_") or key in ("output_dimensionality", "query_task_type", "document_task_type"):
                del rag[key]
        rag.update(embedding_backend=embedding.backend, embedding_provider=embedding.provider,
                   embedding_model=embedding.model, embedding_endpoint=embedding.endpoint,
                   embedding_api_key_env=embedding.api_key_env,
                   embedding_timeout_seconds=embedding.timeout_seconds,
                   output_dimensionality=embedding.output_dimension,
                   query_task_type=embedding.native_query_task_type,
                   document_task_type=embedding.native_document_task_type)
    return result


def require_entrypoint_strategy(plan, *, execution, stages):
    """Refuse routes that the selected legacy command cannot execute."""
    for stage in stages or ():
        expected = {"project_analysis": "sync", "final_review": "gemini_batch"}.get(stage, routing.ExecutionStrategy(execution).value)
        if stage in plan.routes and plan.routes[stage].strategy.value != expected:
            raise routing.ModelRoutingConfigError("Selected command cannot execute the configured stage strategy")
