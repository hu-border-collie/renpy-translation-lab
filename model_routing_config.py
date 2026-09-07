"""Versioned product configuration contract for model routing.

This module defines and validates the ``translator_config.json``
``model_routing`` section introduced by issue #348.  It is deliberately pure:
it does not read or write files, resolve credentials, import provider SDKs, or
change the legacy runtime. The P1 offline migrator and compatibility reader
consume this independently tested contract; production activation is P2.

Unknown keys are accepted so a read-modify-write cycle can preserve fields
written by a newer version.  Known fields still fail closed when malformed,
and credential values are forbidden anywhere inside the section.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import TypeGuard
from urllib.parse import urlsplit


MODEL_ROUTING_SECTION = "model_routing"
MODEL_ROUTING_CONFIG_SCHEMA_VERSION = 1

ADAPTER_GEMINI = "gemini"
ADAPTER_LITELLM = "litellm"
KNOWN_ADAPTERS = frozenset({ADAPTER_GEMINI, ADAPTER_LITELLM})

STRATEGY_SYNC = "sync"
STRATEGY_GEMINI_BATCH = "gemini_batch"
KNOWN_EXECUTION_STRATEGIES = frozenset({
    STRATEGY_SYNC,
    STRATEGY_GEMINI_BATCH,
})

CONFIGURABLE_TASK_STAGES = (
    "translation",
    "keyword",
    "revision",
    "project_analysis",
    "final_review",
)

CREDENTIAL_KIND_API_KEYS_JSON = "api_keys_json"
CREDENTIAL_KIND_KEYRING = "keyring"
CREDENTIAL_KIND_ENV = "env"
CREDENTIAL_KIND_NONE = "none"
KNOWN_CREDENTIAL_KINDS = frozenset({
    CREDENTIAL_KIND_API_KEYS_JSON,
    CREDENTIAL_KIND_KEYRING,
    CREDENTIAL_KIND_ENV,
    CREDENTIAL_KIND_NONE,
})

_CONFIG_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_PROFILE_IDS = frozenset({"primary", "batch"})
_RESERVED_PROFILE_SUFFIXES = ("_model", "_override")
_SENSITIVE_NORMALIZED_KEYS = frozenset({
    "apikey",
    "apikeys",
    "accesstoken",
    "authorization",
    "bearertoken",
    "clientsecret",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "token",
    "xapikey",
})


@dataclass(frozen=True)
class ConfigContractIssue:
    """One machine-readable refusal produced by contract validation."""

    path: str
    code: str
    message: str


@dataclass(frozen=True)
class LegacyFieldMapping:
    """One frozen P0 mapping that the P1 migrator must implement."""

    source_path: tuple[str, ...]
    target: str
    rule: str


LEGACY_FIELD_MAPPINGS = (
    LegacyFieldMapping(
        ("sync", "backend"),
        "model_routing.providers + profiles",
        "select gemini or litellm adapter without changing the effective backend",
    ),
    LegacyFieldMapping(
        ("sync", "model"),
        "model_routing.profiles.<sync-profile>.model",
        "preserve the effective Sync model",
    ),
    LegacyFieldMapping(
        ("sync", "models"),
        "model_routing.profiles.<sync-profile>.models",
        "preserve explicit same-profile rotation order; non-empty sync.models wins over rotation.model",
    ),
    LegacyFieldMapping(
        ("rotation", "model"),
        "model_routing.profiles.<sync-profile>.models",
        "require an explicit sync.models snapshot for enabled rotation; never expand an implicit catalog pool",
    ),
    LegacyFieldMapping(
        ("sync", "custom_litellm_providers"),
        "model_routing.providers",
        "convert connection metadata and credential references, never key values",
    ),
    LegacyFieldMapping(
        ("batch", "model"),
        "model_routing.profiles.<batch-profile>.model",
        "preserve the effective Gemini Batch model",
    ),
    LegacyFieldMapping(
        ("batch", "project_analysis", "model"),
        "model_routing.routes.project_analysis",
        "always emit an explicit sync route; empty/omitted model uses the Sync profile",
    ),
    LegacyFieldMapping(
        ("batch", "final_review", "model"),
        "model_routing.routes.final_review",
        "always emit an explicit gemini_batch route; empty/omitted model uses the Batch profile",
    ),
) + tuple(
    LegacyFieldMapping(
        (scope, "rag", field), target, rule,
    )
    for scope in ("sync", "batch")
    for field, target, rule in (
        ("embedding_backend", "embedding profile params.backend", "preserve backend and adapter"),
        ("embedding_provider", "embedding provider.provider", "preserve upstream provider"),
        ("embedding_endpoint", "embedding provider.base_url", "preserve connection endpoint"),
        ("embedding_api_key_env", "embedding provider.credential_ref", "environment reference only; never migrate embedding_api_key values"),
        ("embedding_model", "embedding profile.model", "preserve embedding model"),
        ("output_dimensionality", "embedding profile params.output_dimension", "preserve request dimension"),
        ("embedding_timeout_seconds", "embedding profile params.timeout_seconds", "preserve request timeout"),
        ("query_task_type", "embedding profile params.native_query_task_type", "preserve query task type"),
        ("document_task_type", "embedding profile params.native_document_task_type", "preserve document task type"),
    )
)



def _issue(path: str, code: str, message: str) -> ConfigContractIssue:
    return ConfigContractIssue(path=path, code=code, message=message)


def _is_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


def _valid_id(value: object) -> bool:
    return isinstance(value, str) and bool(_CONFIG_ID_PATTERN.fullmatch(value))


def _reserved_profile_id(profile_id: str) -> bool:
    if profile_id in _RESERVED_PROFILE_IDS:
        return True
    return any(
        profile_id.endswith(suffix)
        and profile_id[: -len(suffix)] in (*CONFIGURABLE_TASK_STAGES, "ab_experiment")
        for suffix in _RESERVED_PROFILE_SUFFIXES
    )


def _find_sensitive_keys(value: object, path: str) -> list[ConfigContractIssue]:
    issues: list[ConfigContractIssue] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            normalized_key = re.sub(r"[^a-z0-9]", "", key.casefold())
            if normalized_key in _SENSITIVE_NORMALIZED_KEYS:
                issues.append(_issue(
                    child_path,
                    "credential_value_forbidden",
                    "Credential values must not be stored in model_routing; use credential_ref.",
                ))
            issues.extend(_find_sensitive_keys(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            issues.extend(_find_sensitive_keys(child, f"{path}[{index}]"))
    return issues


def _validate_credential_ref(
    raw: object,
    path: str,
) -> list[ConfigContractIssue]:
    if not _is_mapping(raw):
        return [_issue(path, "invalid_credential_ref", "credential_ref must be an object.")]
    issues: list[ConfigContractIssue] = []
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in KNOWN_CREDENTIAL_KINDS:
        issues.append(_issue(
            f"{path}.kind",
            "unsupported_credential_kind",
            f"credential_ref.kind must be one of {sorted(KNOWN_CREDENTIAL_KINDS)}.",
        ))
    for key in ("name", "env_name"):
        value = raw.get(key, "")
        if not isinstance(value, str):
            issues.append(_issue(
                f"{path}.{key}",
                "invalid_credential_reference",
                f"credential_ref.{key} must be a string reference.",
            ))
    env_name = raw.get("env_name", "")
    if kind == CREDENTIAL_KIND_ENV:
        name = raw.get("name")
        if not isinstance(name, str) or not _ENV_NAME_PATTERN.fullmatch(name):
            issues.append(_issue(f"{path}.name", "invalid_environment_name", "env credential name must be an environment variable name."))
    if isinstance(env_name, str) and env_name and not _ENV_NAME_PATTERN.fullmatch(env_name):
        issues.append(_issue(
            f"{path}.env_name",
            "invalid_environment_name",
            "credential_ref.env_name must be a valid environment variable name.",
        ))
    if (
        isinstance(kind, str) and kind in {
            CREDENTIAL_KIND_API_KEYS_JSON,
            CREDENTIAL_KIND_KEYRING,
            CREDENTIAL_KIND_ENV,
        }
        and not str(raw.get("name") or "").strip()
    ):
        issues.append(_issue(
            f"{path}.name",
            "missing_credential_reference",
            "This credential reference kind requires a non-empty lookup name.",
        ))
    return issues


def _validate_url(value: object, path: str) -> list[ConfigContractIssue]:
    if value in (None, ""):
        return []
    if not isinstance(value, str):
        return [_issue(path, "invalid_provider_url", "Provider URL must be a string.")]
    try:
        parsed = urlsplit(value.strip())
        parsed.port
    except ValueError:
        return [_issue(path, "invalid_provider_url", "Provider URL is malformed.")]
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return [_issue(
            path,
            "invalid_provider_url",
            "Provider URL must be a clean http(s) URL without credentials, query, or fragment.",
        )]
    return []


def _validate_providers(raw: object) -> tuple[list[ConfigContractIssue], dict[str, str]]:
    path = f"{MODEL_ROUTING_SECTION}.providers"
    if not _is_mapping(raw) or not raw:
        return ([_issue(path, "missing_providers", "providers must be a non-empty object.")], {})
    issues: list[ConfigContractIssue] = []
    adapters: dict[str, str] = {}
    for provider_id, provider in raw.items():
        provider_path = f"{path}.{provider_id}"
        if not _valid_id(provider_id):
            issues.append(_issue(
                provider_path,
                "invalid_provider_id",
                "Provider ids must match ^[a-z0-9_-]+$.",
            ))
        if not _is_mapping(provider):
            issues.append(_issue(
                provider_path,
                "invalid_provider",
                "Each provider must be an object.",
            ))
            continue
        label = provider.get("label")
        if not isinstance(label, str) or not label.strip():
            issues.append(_issue(
                f"{provider_path}.label",
                "missing_provider_label",
                "provider.label must be a non-empty user-visible name.",
            ))
        adapter = provider.get("adapter")
        if not isinstance(adapter, str) or adapter not in KNOWN_ADAPTERS:
            issues.append(_issue(
                f"{provider_path}.adapter",
                "unsupported_adapter",
                f"adapter must be one of {sorted(KNOWN_ADAPTERS)}.",
            ))
        else:
            adapters[str(provider_id)] = str(adapter)
        upstream = provider.get("provider")
        if not isinstance(upstream, str) or not upstream.strip():
            issues.append(_issue(
                f"{provider_path}.provider",
                "missing_provider_name",
                "provider must name the upstream provider/model prefix.",
            ))
        issues.extend(_validate_credential_ref(
            provider.get("credential_ref"),
            f"{provider_path}.credential_ref",
        ))
        issues.extend(_validate_url(provider.get("base_url", ""), f"{provider_path}.base_url"))
        issues.extend(_validate_url(provider.get("models_url", ""), f"{provider_path}.models_url"))
    return issues, adapters


def _validate_profiles(
    raw: object,
    provider_adapters: Mapping[str, str],
) -> tuple[list[ConfigContractIssue], dict[str, str]]:
    path = f"{MODEL_ROUTING_SECTION}.profiles"
    if not _is_mapping(raw) or not raw:
        return ([_issue(path, "missing_profiles", "profiles must be a non-empty object.")], {})
    issues: list[ConfigContractIssue] = []
    profile_providers: dict[str, str] = {}
    for profile_id, profile in raw.items():
        profile_path = f"{path}.{profile_id}"
        if not _valid_id(profile_id):
            issues.append(_issue(
                profile_path,
                "invalid_profile_id",
                "Profile ids must match ^[a-z0-9_-]+$.",
            ))
        elif _reserved_profile_id(str(profile_id)):
            issues.append(_issue(
                profile_path,
                "reserved_profile_id",
                "Profile id collides with a legacy resolver slot.",
            ))
        if not _is_mapping(profile):
            issues.append(_issue(
                profile_path,
                "invalid_profile",
                "Each profile must be an object.",
            ))
            continue
        label = profile.get("label")
        if not isinstance(label, str) or not label.strip():
            issues.append(_issue(
                f"{profile_path}.label",
                "missing_profile_label",
                "profile.label must be a non-empty user-visible name.",
            ))
        provider_id = profile.get("provider_id")
        if not isinstance(provider_id, str) or provider_id not in provider_adapters:
            issues.append(_issue(
                f"{profile_path}.provider_id",
                "unknown_provider",
                "profile.provider_id must reference a configured provider.",
            ))
        else:
            profile_providers[str(profile_id)] = provider_id
        model = profile.get("model")
        if not isinstance(model, str) or not model.strip():
            issues.append(_issue(
                f"{profile_path}.model",
                "missing_model",
                "profile.model must be a non-empty string.",
            ))
        models = profile.get("models", [])
        if (
            not isinstance(models, list)
            or any(not isinstance(item, str) or not item.strip() for item in models)
        ):
            issues.append(_issue(
                f"{profile_path}.models",
                "invalid_model_rotation",
                "profile.models must be a list of non-empty model ids.",
            ))
        elif models and isinstance(model, str) and model not in models:
            issues.append(_issue(
                f"{profile_path}.models",
                "primary_model_missing_from_rotation",
                "A non-empty profile.models list must contain profile.model.",
            ))
        for object_key in ("capability_overrides", "params"):
            if not _is_mapping(profile.get(object_key, {})):
                issues.append(_issue(
                    f"{profile_path}.{object_key}",
                    "invalid_profile_object",
                    f"profile.{object_key} must be an object.",
                ))
    return issues, profile_providers


def _validate_strategy_for_profile(
    strategy: object,
    profile_id: object,
    profile_providers: Mapping[str, str],
    provider_adapters: Mapping[str, str],
    path: str,
) -> list[ConfigContractIssue]:
    if not isinstance(strategy, str) or strategy not in KNOWN_EXECUTION_STRATEGIES:
        return [_issue(
            path,
            "unsupported_execution_strategy",
            f"strategy must be one of {sorted(KNOWN_EXECUTION_STRATEGIES)}.",
        )]
    if (
        strategy == STRATEGY_GEMINI_BATCH
        and isinstance(profile_id, str)
        and profile_id in profile_providers
        and provider_adapters.get(profile_providers.get(profile_id, "")) != ADAPTER_GEMINI
    ):
        return [_issue(
            path,
            "strategy_profile_mismatch",
            "gemini_batch requires a profile backed by the gemini adapter.",
        )]
    return []


def validate_model_routing_section(section: object) -> tuple[ConfigContractIssue, ...]:
    """Validate one schema-v1 ``model_routing`` section without mutating it.

    Unknown fields are intentionally ignored after recursively checking that
    they do not contain credential-shaped values.  This is the preservation
    contract required by #202/#348; a later writer must merge known edits into
    the original object rather than reconstructing it from known fields.
    """
    path = MODEL_ROUTING_SECTION
    if not _is_mapping(section):
        return (_issue(path, "invalid_section", "model_routing must be an object."),)

    issues = _find_sensitive_keys(section, path)
    if (type(section.get("schema_version")) is not int
            or section.get("schema_version") != MODEL_ROUTING_CONFIG_SCHEMA_VERSION):
        issues.append(_issue(
            f"{path}.schema_version",
            "unsupported_schema_version",
            f"schema_version must be {MODEL_ROUTING_CONFIG_SCHEMA_VERSION}.",
        ))

    provider_issues, provider_adapters = _validate_providers(section.get("providers"))
    issues.extend(provider_issues)
    profile_issues, profile_providers = _validate_profiles(
        section.get("profiles"),
        provider_adapters,
    )
    issues.extend(profile_issues)

    defaults = section.get("defaults")
    primary_profile_id: object = None
    default_strategy: object = None
    if not _is_mapping(defaults):
        issues.append(_issue(
            f"{path}.defaults",
            "missing_defaults",
            "defaults must be an object.",
        ))
    else:
        primary_profile_id = defaults.get("primary_profile_id")
        default_strategy = defaults.get("execution_strategy")
        if not isinstance(primary_profile_id, str) or primary_profile_id not in profile_providers:
            issues.append(_issue(
                f"{path}.defaults.primary_profile_id",
                "unknown_primary_profile",
                "primary_profile_id must reference a configured profile.",
            ))
        issues.extend(_validate_strategy_for_profile(
            default_strategy,
            primary_profile_id,
            profile_providers,
            provider_adapters,
            f"{path}.defaults.execution_strategy",
        ))

    profiles = section.get("profiles")
    entrypoints = section.get("legacy_entrypoints", {})
    if not _is_mapping(entrypoints):
        issues.append(_issue(f"{path}.legacy_entrypoints", "invalid_legacy_entrypoints", "legacy_entrypoints must be an object."))
    else:
        for key, adapter in (("sync_profile_id", None), ("batch_profile_id", ADAPTER_GEMINI)):
            if key not in entrypoints:
                continue
            value = entrypoints[key]
            if not isinstance(value, str) or value not in profile_providers:
                issues.append(_issue(f"{path}.legacy_entrypoints.{key}", "unknown_legacy_profile", "Legacy entrypoint must reference a configured profile."))
            elif adapter and provider_adapters[profile_providers[value]] != adapter:
                issues.append(_issue(f"{path}.legacy_entrypoints.{key}", "strategy_profile_mismatch", "Legacy Batch requires a Gemini profile."))
    if _is_mapping(profiles):
        for profile_id, profile in profiles.items():
            if not _is_mapping(profile):
                continue
            embedding_profile_id = profile.get("embedding_profile_id", "")
            if not isinstance(embedding_profile_id, str) or (
                embedding_profile_id and embedding_profile_id not in profile_providers
            ):
                issues.append(_issue(
                    f"{path}.profiles.{profile_id}.embedding_profile_id",
                    "unknown_embedding_profile",
                    "embedding_profile_id must reference a configured profile.",
                ))

    routes = section.get("routes", {})
    if not _is_mapping(routes):
        issues.append(_issue(f"{path}.routes", "invalid_routes", "routes must be an object."))
    else:
        for stage, route in routes.items():
            route_path = f"{path}.routes.{stage}"
            if stage not in CONFIGURABLE_TASK_STAGES:
                issues.append(_issue(
                    route_path,
                    "unsupported_task_stage",
                    f"stage must be one of {list(CONFIGURABLE_TASK_STAGES)}.",
                ))
            if not _is_mapping(route):
                issues.append(_issue(route_path, "invalid_route", "route must be an object."))
                continue
            if "profile_id" not in route and "strategy" not in route:
                issues.append(_issue(
                    route_path,
                    "empty_route_override",
                    "A route override must set profile_id, strategy, or both.",
                ))
                continue
            route_profile_id = route.get("profile_id", primary_profile_id)
            route_strategy = route.get("strategy", default_strategy)
            if not isinstance(route_profile_id, str) or route_profile_id not in profile_providers:
                issues.append(_issue(
                    f"{route_path}.profile_id",
                    "unknown_route_profile",
                    "route.profile_id must reference a configured profile.",
                ))
            issues.extend(_validate_strategy_for_profile(
                route_strategy,
                route_profile_id,
                profile_providers,
                provider_adapters,
                f"{route_path}.strategy",
            ))
    return tuple(issues)


def validate_translator_config_model_routing(
    config: object,
) -> tuple[ConfigContractIssue, ...]:
    """Validate only the new section; a legacy-only config remains accepted."""
    if not _is_mapping(config):
        return (_issue("$", "invalid_config", "translator config must be an object."),)
    if MODEL_ROUTING_SECTION not in config:
        return ()
    return validate_model_routing_section(config[MODEL_ROUTING_SECTION])


def legacy_fields_present(config: object) -> tuple[tuple[str, ...], ...]:
    """Return mapped legacy source paths present in *config*, without mutation."""
    if not _is_mapping(config):
        return ()
    present: list[tuple[str, ...]] = []
    for mapping in LEGACY_FIELD_MAPPINGS:
        current: object = config
        for part in mapping.source_path:
            if not _is_mapping(current) or part not in current:
                break
            current = current[part]
        else:
            present.append(mapping.source_path)
    return tuple(present)
