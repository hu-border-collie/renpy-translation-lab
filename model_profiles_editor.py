"""Qt-free Model Profiles editor for the versioned ``model_routing`` section.

The editor is the shared core for the Settings Model Profiles page (and any
future CLI command): every mutation returns a detached deep copy, preserves
unknown or not-yet-rendered fields, and never reads, stores, or returns a
credential value. Credentials stay references such as ``api_keys_json`` slots,
keyring names, or environment variable names.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

import model_profile as routing
import model_routing_reader as reader
from model_routing_config import validate_model_routing_section

STAGE_ORDER: tuple[str, ...] = (
    routing.STAGE_TRANSLATION,
    routing.STAGE_KEYWORD,
    routing.STAGE_REVISION,
    routing.STAGE_PROJECT_ANALYSIS,
    routing.STAGE_FINAL_REVIEW,
)

STRATEGY_ORDER: tuple[str, ...] = (
    routing.ExecutionStrategy.SYNC.value,
    routing.ExecutionStrategy.GEMINI_BATCH.value,
)

ADAPTERS: tuple[str, ...] = (routing.ADAPTER_GEMINI, routing.ADAPTER_LITELLM)

CREDENTIAL_KINDS: tuple[str, ...] = (
    routing.CREDENTIAL_KIND_API_KEYS_JSON,
    routing.CREDENTIAL_KIND_KEYRING,
    routing.CREDENTIAL_KIND_ENV,
    routing.CREDENTIAL_KIND_NONE,
)

CAPABILITY_FLAG_KEYS: tuple[str, ...] = (
    "sync_generation",
    "reasoning_request",
    "reasoning_response",
    "usage_stats",
    "remote_batch",
    "embedding",
)

CAPABILITY_INT_KEYS: tuple[str, ...] = (
    "context_limit_tokens",
    "context_budget_tokens",
)

_SLUG_PATTERN = re.compile(r"[^a-z0-9_-]+")


class ModelProfilesEditorError(ValueError):
    """Expected editor refusal with a stable machine code."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(str(message))
        self.code = str(code)
        self.details = dict(details or {})


def empty_section() -> dict[str, Any]:
    """Return a minimal schema-v1 section with no profiles or providers."""
    return {
        "schema_version": 1,
        "providers": {},
        "profiles": {},
        "defaults": {"primary_profile_id": "", "execution_strategy": "sync"},
        "routes": {},
    }


def _copy_section(section: Mapping[str, Any] | None) -> dict[str, Any]:
    if section is None:
        return empty_section()
    if not isinstance(section, Mapping):
        raise ModelProfilesEditorError(
            "INVALID_SECTION",
            "model_routing must be an object",
        )
    result = copy.deepcopy(dict(section))
    for key, expected in (("providers", dict), ("profiles", dict), ("defaults", dict)):
        value = result.get(key)
        if not isinstance(value, expected):
            result[key] = expected()
    routes = result.get("routes")
    if routes is not None and not isinstance(routes, dict):
        result["routes"] = {}
    return result


def slugify(value: str, *, fallback: str = "profile") -> str:
    """Return a schema-safe lowercase id fragment."""
    slug = _SLUG_PATTERN.sub("-", str(value or "").strip().lower()).strip("-_")
    return slug or fallback


def unique_id(existing: Mapping[str, Any], base: str) -> str:
    """Return ``base`` or the first free ``base-2``/``base-3`` suffix."""
    candidate = base
    index = 2
    while candidate in existing:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def profile_ids(section: Mapping[str, Any] | None) -> tuple[str, ...]:
    return tuple(str(key) for key in (_copy_section(section).get("profiles") or {}))


def provider_ids(section: Mapping[str, Any] | None) -> tuple[str, ...]:
    return tuple(str(key) for key in (_copy_section(section).get("providers") or {}))


def _profile_entry(section: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    profiles = section.get("profiles") or {}
    raw = profiles.get(profile_id)
    if not isinstance(raw, Mapping):
        raise ModelProfilesEditorError(
            "UNKNOWN_PROFILE",
            f"Unknown ModelProfile: {profile_id}",
            details={"profile_id": str(profile_id)},
        )
    return dict(raw)


def _provider_entry(section: Mapping[str, Any], provider_id: str) -> dict[str, Any]:
    providers = section.get("providers") or {}
    raw = providers.get(provider_id)
    if not isinstance(raw, Mapping):
        raise ModelProfilesEditorError(
            "UNKNOWN_PROVIDER",
            f"Unknown Provider: {provider_id}",
            details={"provider_id": str(provider_id)},
        )
    return dict(raw)


def _profile_references(section: Mapping[str, Any], profile_id: str) -> list[str]:
    references: list[str] = []
    defaults = section.get("defaults") or {}
    if str(defaults.get("primary_profile_id") or "") == profile_id:
        references.append("defaults.primary_profile_id")
    for stage, raw_route in (section.get("routes") or {}).items():
        if isinstance(raw_route, Mapping) and str(raw_route.get("profile_id") or "") == profile_id:
            references.append(f"routes.{stage}.profile_id")
    for other_id, raw in (section.get("profiles") or {}).items():
        if isinstance(raw, Mapping) and str(raw.get("embedding_profile_id") or "") == profile_id:
            references.append(f"profiles.{other_id}.embedding_profile_id")
    return references


def _strategy_supported(section: Mapping[str, Any], profile_id: str, strategy: str) -> str:
    """Return "" when supported, else a machine reason code."""
    raw = _profile_entry(section, profile_id)
    provider = (section.get("providers") or {}).get(
        str(raw.get("provider_id") or "")
    )
    if not isinstance(provider, Mapping):
        return "missing_provider"
    if strategy == routing.ExecutionStrategy.SYNC.value:
        return ""
    if strategy != routing.ExecutionStrategy.GEMINI_BATCH.value:
        return "unsupported_strategy"
    if provider.get("adapter") != routing.ADAPTER_GEMINI:
        return "missing_gemini_adapter"
    try:
        capabilities = _capabilities_for(section, profile_id)
    except ModelProfilesEditorError:
        return "missing_remote_batch"
    return "" if capabilities.get("remote_batch") else "missing_remote_batch"


def model_profile_for(
    section: Mapping[str, Any],
    profile_id: str,
) -> routing.ModelProfile:
    """Build one detached ``ModelProfile`` for adapters, probes and diagnostics."""
    raw = _profile_entry(section, profile_id)
    provider = (section.get("providers") or {}).get(str(raw.get("provider_id") or ""))
    if not isinstance(provider, Mapping):
        raise ModelProfilesEditorError(
            "UNKNOWN_PROVIDER",
            "ModelProfile references an unknown provider",
        )
    return routing.ModelProfile(
        id=str(profile_id),
        label=str(raw.get("label") or profile_id),
        adapter=str(provider.get("adapter") or routing.ADAPTER_GEMINI),
        provider=str(provider.get("provider") or ""),
        model=str(raw.get("model") or ""),
        credential_ref=routing.CredentialRef.from_manifest_dict(
            provider.get("credential_ref") or {}
        ),
        models=tuple(str(item) for item in raw.get("models") or (str(raw.get("model") or ""),)),
        base_url=str(provider.get("base_url") or ""),
        capability_overrides=dict(raw.get("capability_overrides") or {}),
        params=dict(raw.get("params") or {}),
        embedding_profile_id=str(raw.get("embedding_profile_id") or ""),
    )


def _capabilities_for(section: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    """Best-effort capability summary for one draft profile."""
    profile = model_profile_for(section, profile_id)
    try:
        custom = reader.section_custom_providers(section)  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError):
        custom = {}
    capabilities = routing.resolve_capabilities(profile, custom_providers=custom)
    return {
        "sync_generation": bool(capabilities.sync_generation.supported),
        "structured_output_mode": capabilities.structured_output.mode,
        "structured_output_source": capabilities.structured_output.source,
        "reasoning_request": bool(capabilities.reasoning_request.supported),
        "reasoning_response": bool(capabilities.reasoning_response.supported),
        "usage_stats": bool(capabilities.usage_stats.supported),
        "remote_batch": bool(capabilities.remote_batch.supported),
        "embedding": bool(capabilities.embedding.supported),
        "context_limit_tokens": capabilities.context_limit_tokens,
        "context_budget_tokens": capabilities.context_budget_tokens,
        "context_source": capabilities.context_source,
        # Per-capability provenance: adapter_default / config_override / probed.
        "sources": {
            "sync_generation": capabilities.sync_generation.source,
            "reasoning_request": capabilities.reasoning_request.source,
            "reasoning_response": capabilities.reasoning_response.source,
            "usage_stats": capabilities.usage_stats.source,
            "remote_batch": capabilities.remote_batch.source,
            "embedding": capabilities.embedding.source,
        },
    }


def strategy_choices(section: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    """Return {profile_id: supported strategies} for draft-safe selection."""
    data = _copy_section(section)
    choices: dict[str, tuple[str, ...]] = {}
    for profile_id in (data.get("profiles") or {}):
        supported = [
            strategy
            for strategy in STRATEGY_ORDER
            if not _strategy_supported(data, profile_id, strategy)
        ]
        choices[str(profile_id)] = tuple(supported)
    return choices


def resolved_routes(section: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return each stage's effective profile/strategy with its origin."""
    data = _copy_section(section)
    defaults = data.get("defaults") or {}
    primary = str(defaults.get("primary_profile_id") or "")
    strategy = str(defaults.get("execution_strategy") or routing.ExecutionStrategy.SYNC.value)
    routes = data.get("routes") or {}
    result: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        raw = routes.get(stage)
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        result.append(
            {
                "stage": stage,
                "profile_id": str(raw.get("profile_id") or primary),
                "strategy": str(raw.get("strategy") or strategy),
                "explicit": bool(raw),
            }
        )
    return result


def editor_view(section: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the public, credential-free projection rendered by the page."""
    data = _copy_section(section)
    profiles: list[dict[str, Any]] = []
    for profile_id, raw in (data.get("profiles") or {}).items():
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        provider_id = str(raw.get("provider_id") or "")
        provider = (data.get("providers") or {}).get(provider_id)
        provider = dict(provider) if isinstance(provider, Mapping) else {}
        try:
            capabilities = _capabilities_for(data, str(profile_id))
        except ModelProfilesEditorError:
            capabilities = {}
        profiles.append(
            {
                "id": str(profile_id),
                "label": str(raw.get("label") or profile_id),
                "purpose": str(raw.get("purpose") or "generation"),
                "provider_id": provider_id,
                "provider_label": str(provider.get("label") or provider_id),
                "adapter": str(provider.get("adapter") or ""),
                "model": str(raw.get("model") or ""),
                "models": tuple(str(item) for item in raw.get("models") or ()),
                "embedding_profile_id": str(raw.get("embedding_profile_id") or ""),
                "capability_overrides": dict(raw.get("capability_overrides") or {}),
                "params": dict(raw.get("params") or {}),
                "referenced_by": tuple(_profile_references(data, str(profile_id))),
                "strategies": strategy_choices(data).get(str(profile_id), ()),
                "capabilities": capabilities,
            }
        )
    providers: list[dict[str, Any]] = []
    for provider_id, raw in (data.get("providers") or {}).items():
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        credential = raw.get("credential_ref")
        credential = dict(credential) if isinstance(credential, Mapping) else {}
        providers.append(
            {
                "id": str(provider_id),
                "label": str(raw.get("label") or provider_id),
                "adapter": str(raw.get("adapter") or ""),
                "provider": str(raw.get("provider") or ""),
                "base_url": str(raw.get("base_url") or ""),
                "models_url": str(raw.get("models_url") or ""),
                "credential_ref": {
                    "kind": str(credential.get("kind") or routing.CREDENTIAL_KIND_NONE),
                    "name": str(credential.get("name") or ""),
                    "env_name": str(credential.get("env_name") or ""),
                },
                "used_by": tuple(
                    str(profile_id)
                    for profile_id, profile in (data.get("profiles") or {}).items()
                    if isinstance(profile, Mapping)
                    and str(profile.get("provider_id") or "") == provider_id
                ),
            }
        )
    return {
        "schema_version": data.get("schema_version", 1),
        "defaults": dict(data.get("defaults") or {}),
        "providers": tuple(providers),
        "profiles": tuple(profiles),
        "routes": tuple(resolved_routes(data)),
    }


def section_issues(section: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """Return validator issues as plain dicts for pages and CLI output."""
    data = _copy_section(section)
    return tuple(
        {"path": issue.path, "code": issue.code, "message": issue.message}
        for issue in validate_model_routing_section(data)
    )


# -- provider mutations -------------------------------------------------


def add_provider(
    section: Mapping[str, Any] | None,
    *,
    label: str,
    adapter: str,
    provider: str,
    base_url: str = "",
    models_url: str = "",
    credential_kind: str = routing.CREDENTIAL_KIND_NONE,
    credential_name: str = "",
    credential_env_name: str = "",
) -> dict[str, Any]:
    data = _copy_section(section)
    if adapter not in ADAPTERS:
        raise ModelProfilesEditorError("INVALID_ADAPTER", f"Unsupported adapter: {adapter}")
    if credential_kind not in CREDENTIAL_KINDS:
        raise ModelProfilesEditorError(
            "INVALID_CREDENTIAL_KIND",
            f"Unsupported credential kind: {credential_kind}",
        )
    provider_id = unique_id(data["providers"], slugify(label, fallback=adapter))
    data["providers"][provider_id] = {
        "label": str(label or provider_id),
        "adapter": adapter,
        "provider": str(provider),
        "base_url": str(base_url or ""),
        "models_url": str(models_url or ""),
        "credential_ref": {
            "kind": credential_kind,
            "name": str(credential_name or ""),
            "env_name": str(credential_env_name or ""),
        },
    }
    return data


def update_provider(
    section: Mapping[str, Any] | None,
    provider_id: str,
    *,
    label: str | None = None,
    adapter: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    models_url: str | None = None,
    credential_kind: str | None = None,
    credential_name: str | None = None,
    credential_env_name: str | None = None,
) -> dict[str, Any]:
    data = _copy_section(section)
    entry = _provider_entry(data, provider_id)
    if adapter is not None:
        if adapter not in ADAPTERS:
            raise ModelProfilesEditorError("INVALID_ADAPTER", f"Unsupported adapter: {adapter}")
        entry["adapter"] = adapter
    if credential_kind is not None:
        if credential_kind not in CREDENTIAL_KINDS:
            raise ModelProfilesEditorError(
                "INVALID_CREDENTIAL_KIND",
                f"Unsupported credential kind: {credential_kind}",
            )
        ref = entry.get("credential_ref")
        ref = dict(ref) if isinstance(ref, Mapping) else {}
        ref["kind"] = credential_kind
        entry["credential_ref"] = ref
    for key, value in (
        ("label", label),
        ("provider", provider),
        ("base_url", base_url),
        ("models_url", models_url),
    ):
        if value is not None:
            entry[key] = str(value)
    ref = entry.get("credential_ref")
    ref = dict(ref) if isinstance(ref, Mapping) else {}
    if credential_name is not None:
        ref["name"] = str(credential_name)
    if credential_env_name is not None:
        ref["env_name"] = str(credential_env_name)
    entry["credential_ref"] = ref
    data["providers"][provider_id] = entry
    return data


def delete_provider(section: Mapping[str, Any] | None, provider_id: str) -> dict[str, Any]:
    data = _copy_section(section)
    _provider_entry(data, provider_id)
    used_by = [
        str(profile_id)
        for profile_id, profile in (data.get("profiles") or {}).items()
        if isinstance(profile, Mapping)
        and str(profile.get("provider_id") or "") == provider_id
    ]
    if used_by:
        raise ModelProfilesEditorError(
            "PROVIDER_IN_USE",
            "Provider is still referenced by ModelProfiles",
            details={"provider_id": provider_id, "profile_ids": used_by},
        )
    del data["providers"][provider_id]
    return data


# -- profile mutations --------------------------------------------------


def add_profile(
    section: Mapping[str, Any] | None,
    *,
    label: str,
    provider_id: str,
    model: str = "",
    models: Sequence[str] = (),
    embedding_profile_id: str = "",
    params: Mapping[str, Any] | None = None,
    capability_overrides: Mapping[str, Any] | None = None,
    purpose: str = "generation",
) -> dict[str, Any]:
    data = _copy_section(section)
    _provider_entry(data, provider_id)
    if purpose not in {"generation", "embedding"}:
        raise ModelProfilesEditorError("INVALID_PURPOSE", f"Unsupported purpose: {purpose}")
    profile_id = unique_id(data["profiles"], slugify(label, fallback="profile"))
    models_tuple = tuple(str(item).strip() for item in models if str(item).strip())
    data["profiles"][profile_id] = {
        "label": str(label or profile_id),
        "provider_id": str(provider_id),
        "model": str(model or ""),
        "models": [item for item in models_tuple if item != str(model or "")],
        "capability_overrides": dict(capability_overrides or {}),
        "params": dict(params or {}),
        "embedding_profile_id": str(embedding_profile_id or ""),
        "purpose": purpose,
    }
    return data


def copy_profile(
    section: Mapping[str, Any] | None,
    profile_id: str,
    *,
    label: str = "",
) -> dict[str, Any]:
    data = _copy_section(section)
    entry = _profile_entry(data, profile_id)
    new_label = str(label or f"{entry.get('label') or profile_id} 副本")
    new_id = unique_id(data["profiles"], slugify(new_label, fallback=profile_id))
    entry["label"] = new_label
    data["profiles"][new_id] = entry
    return data


def update_profile(
    section: Mapping[str, Any] | None,
    profile_id: str,
    *,
    label: str | None = None,
    provider_id: str | None = None,
    model: str | None = None,
    models: Sequence[str] | None = None,
    embedding_profile_id: str | None = None,
    params: Mapping[str, Any] | None = None,
    capability_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = _copy_section(section)
    entry = _profile_entry(data, profile_id)
    if provider_id is not None:
        _provider_entry(data, provider_id)
        entry["provider_id"] = str(provider_id)
    if label is not None:
        entry["label"] = str(label)
    if model is not None:
        entry["model"] = str(model)
    if models is not None:
        entry["models"] = [str(item).strip() for item in models if str(item).strip()]
    if embedding_profile_id is not None:
        entry["embedding_profile_id"] = str(embedding_profile_id)
    if params is not None:
        entry["params"] = dict(params)
    if capability_overrides is not None:
        unknown = sorted(set(capability_overrides) - routing.CAPABILITY_OVERRIDE_KEYS)
        if unknown:
            raise ModelProfilesEditorError(
                "UNKNOWN_CAPABILITY_OVERRIDE",
                "Unsupported capability override keys",
                details={"keys": unknown},
            )
        entry["capability_overrides"] = dict(capability_overrides)
    data["profiles"][profile_id] = entry
    return data


def delete_profile(section: Mapping[str, Any] | None, profile_id: str) -> dict[str, Any]:
    data = _copy_section(section)
    _profile_entry(data, profile_id)
    references = _profile_references(data, profile_id)
    if references:
        raise ModelProfilesEditorError(
            "PROFILE_IN_USE",
            "ModelProfile still has routing or embedding references",
            details={"profile_id": profile_id, "references": references},
        )
    del data["profiles"][profile_id]
    return data


# -- defaults and routes ------------------------------------------------


def set_defaults(
    section: Mapping[str, Any] | None,
    *,
    primary_profile_id: str,
    execution_strategy: str,
) -> dict[str, Any]:
    data = _copy_section(section)
    _profile_entry(data, primary_profile_id)
    if execution_strategy not in STRATEGY_ORDER:
        raise ModelProfilesEditorError(
            "INVALID_STRATEGY",
            f"Unsupported execution strategy: {execution_strategy}",
        )
    reason = _strategy_supported(data, primary_profile_id, execution_strategy)
    if reason:
        raise ModelProfilesEditorError(
            "STRATEGY_NOT_SUPPORTED",
            "Selected ModelProfile cannot run the requested strategy",
            details={
                "profile_id": primary_profile_id,
                "strategy": execution_strategy,
                "reason": reason,
            },
        )
    defaults = data.get("defaults")
    defaults = dict(defaults) if isinstance(defaults, Mapping) else {}
    defaults["primary_profile_id"] = str(primary_profile_id)
    defaults["execution_strategy"] = str(execution_strategy)
    data["defaults"] = defaults
    return data


def set_route(
    section: Mapping[str, Any] | None,
    stage: str,
    *,
    enabled: bool,
    profile_id: str = "",
    strategy: str = "",
) -> dict[str, Any]:
    data = _copy_section(section)
    if stage not in STAGE_ORDER:
        raise ModelProfilesEditorError("UNKNOWN_STAGE", f"Unsupported stage: {stage}")
    routes = data.get("routes")
    routes = dict(routes) if isinstance(routes, Mapping) else {}
    if not enabled:
        routes.pop(stage, None)
        data["routes"] = routes
        return data
    _profile_entry(data, profile_id)
    if strategy not in STRATEGY_ORDER:
        raise ModelProfilesEditorError(
            "INVALID_STRATEGY",
            f"Unsupported execution strategy: {strategy}",
        )
    reason = _strategy_supported(data, profile_id, strategy)
    if reason:
        raise ModelProfilesEditorError(
            "STRATEGY_NOT_SUPPORTED",
            "Selected ModelProfile cannot run the requested strategy",
            details={"profile_id": profile_id, "strategy": strategy, "reason": reason},
        )
    routes[stage] = {"profile_id": str(profile_id), "strategy": str(strategy)}
    data["routes"] = routes
    return data
