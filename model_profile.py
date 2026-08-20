"""Provider-neutral model routing contract: profiles, capabilities, strategies.

This module is the single place that answers "which model, through which
adapter, with which execution strategy, for which stage" (issue #345).  It is
intentionally pure: it reads the legacy ``sync.*`` / ``batch.*`` config shape
read-only, never imports optional SDKs at module scope, and never touches
credential *values* — profiles hold :class:`CredentialRef` references only.

Resolution semantics (approved with issue #345, option B):

* A stage's own config key (``batch.project_analysis.model``,
  ``batch.final_review.model``) or an explicit caller override (A/B
  ``--model``, manifest model) wins over the primary profile.
* Stages without their own configuration inherit the run's base profile —
  the primary sync profile for ``sync`` runs, the Gemini batch profile for
  ``gemini_batch`` runs.

The production entry points (``run_sync_request``, ``call_gemini_sdk``, ...)
freeze one :class:`ModelRoutingPlan` at run start and pass the stage
:class:`TaskRoute` into every sync call. ``sync.model`` is only the
inherited primary; an explicit stage route always wins.

Profile ids are **resolver slot ids** (``primary``, ``batch``,
``<stage>_model``, ``<stage>_override``) — an internal compatibility layer
over the legacy config keys, not the user-visible profile namespace of the
productized config (#344/#348). User profile ids must match
``USER_PROFILE_ID_PATTERN`` and stay clear of the reserved slots (see
:data:`RESERVED_PROFILE_SLOT_IDS`, :func:`is_profile_slot_id`); in the
productized config ``primary`` becomes a role pointer
(``ModelRoutingPlan.primary_profile_id``) rather than a profile's own id.
"""
from __future__ import annotations

import os
import re
from collections.abc import Collection
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping

from cli_contract import EXIT_INVALID_STATE, MachineContractError
from gemini_model_catalog import DEFAULT_GEMINI_TRANSLATION_MODEL
from litellm_provider_config import (
    CustomLiteLLMProvider,
    StructuredOutputCapability,
    provider_from_model,
    provider_display_label,
    structured_output_capability,
)

MODEL_PROFILE_SCHEMA_VERSION = 1

ADAPTER_GEMINI = "gemini"
ADAPTER_LITELLM = "litellm"
KNOWN_ADAPTERS = frozenset({ADAPTER_GEMINI, ADAPTER_LITELLM})

SYNC_BACKEND_GEMINI = "gemini"
SYNC_BACKEND_LITELLM = "litellm"
KNOWN_SYNC_BACKENDS = frozenset({SYNC_BACKEND_GEMINI, SYNC_BACKEND_LITELLM})

# Task stages that may appear in a routing plan.
STAGE_TRANSLATION = "translation"
STAGE_KEYWORD = "keyword"
STAGE_REVISION = "revision"
STAGE_PROJECT_ANALYSIS = "project_analysis"
STAGE_FINAL_REVIEW = "final_review"
STAGE_AB_EXPERIMENT = "ab_experiment"
KNOWN_STAGES = frozenset({
    STAGE_TRANSLATION,
    STAGE_KEYWORD,
    STAGE_REVISION,
    STAGE_PROJECT_ANALYSIS,
    STAGE_FINAL_REVIEW,
    STAGE_AB_EXPERIMENT,
})

# How a route was decided. "stage_config" = the stage's own config key,
# "explicit" = a caller-supplied override (CLI flag / manifest value),
# "inherited" = the stage has no own config and inherits the run's base
# profile (the primary sync profile for sync runs, the batch profile for
# gemini_batch runs — ``TaskRoute.profile_id`` says which), "builtin_default"
# = nothing configured anywhere; tool default.
ROUTE_SOURCE_STAGE_CONFIG = "stage_config"
ROUTE_SOURCE_EXPLICIT = "explicit"
ROUTE_SOURCE_INHERITED = "inherited"
ROUTE_SOURCE_BUILTIN_DEFAULT = "builtin_default"
KNOWN_ROUTE_SOURCES = frozenset({
    ROUTE_SOURCE_STAGE_CONFIG,
    ROUTE_SOURCE_EXPLICIT,
    ROUTE_SOURCE_INHERITED,
    ROUTE_SOURCE_BUILTIN_DEFAULT,
})

# Capability provenance. Adapter defaults come from this module's builtin
# tables, probe results are reserved for #340/#341 wiring, and config
# overrides arrive with the productized config format (#344).
CAPABILITY_SOURCE_ADAPTER_DEFAULT = "adapter_default"
CAPABILITY_SOURCE_PROBED = "probed"
CAPABILITY_SOURCE_CONFIG_OVERRIDE = "config_override"
CAPABILITY_SOURCES = frozenset({
    CAPABILITY_SOURCE_ADAPTER_DEFAULT,
    CAPABILITY_SOURCE_PROBED,
    CAPABILITY_SOURCE_CONFIG_OVERRIDE,
})

# How an adapter's structured-output mode was chosen (``StructuredOutputSpec
# .basis``). This is deliberately a separate vocabulary from
# ``CAPABILITY_SOURCES``: basis answers "which adapter table picked this
# mode", provenance answers "who vouches for this capability value".
STRUCTURED_OUTPUT_BASIS_BUILTIN = "builtin"
STRUCTURED_OUTPUT_BASIS_CUSTOM_OPENAI_COMPATIBLE = "custom_openai_compatible"
STRUCTURED_OUTPUT_BASIS_CONSERVATIVE_DEFAULT = "conservative_default"

# Credential reference kinds. References only; no key material ever enters
# this module.
CREDENTIAL_KIND_API_KEYS_JSON = "api_keys_json"
CREDENTIAL_KIND_KEYRING = "keyring"
CREDENTIAL_KIND_ENV = "env"
CREDENTIAL_KIND_NONE = "none"

# Machine-decidable error code names (stable CLI contract; registered in
# docs/quickstart_agent.md when the fail-fast wiring lands).
MODEL_PROFILE_INVALID = "MODEL_PROFILE_INVALID"
MODEL_ROUTE_CAPABILITY_MISSING = "MODEL_ROUTE_CAPABILITY_MISSING"
MODEL_PROFILE_CREDENTIAL_REF_MISSING = "MODEL_PROFILE_CREDENTIAL_REF_MISSING"

# Per-code next actions for the machine refusal contract. Reuses the
# existing ``inspect_configuration_and_artifacts`` token for credential
# problems because the config file itself may be perfectly valid — the
# missing credential lives in the environment or keyring.
_SUGGESTED_ACTION_BY_CODE = {
    MODEL_PROFILE_INVALID: "fix_translator_config",
    MODEL_ROUTE_CAPABILITY_MISSING: "choose_supported_strategy_or_profile",
    MODEL_PROFILE_CREDENTIAL_REF_MISSING: "inspect_configuration_and_artifacts",
}


class ModelRoutingConfigError(ValueError):
    """Invalid routing configuration before a plan can be constructed."""


# Resolver slot ids are internal compatibility slots over the legacy
# ``sync.*`` / ``batch.*`` keys — they are NOT the user-visible profile ids
# of the productized config (#344/#348). User profile ids must match
# ``USER_PROFILE_ID_PATTERN`` and must not collide with a reserved slot.
PROFILE_SLOT_PRIMARY = "primary"
PROFILE_SLOT_BATCH = "batch"
RESERVED_PROFILE_SLOT_IDS = frozenset({PROFILE_SLOT_PRIMARY, PROFILE_SLOT_BATCH})
RESERVED_PROFILE_SLOT_SUFFIXES = ("_model", "_override")
USER_PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")


def is_profile_slot_id(profile_id: str) -> bool:
    """True when *profile_id* belongs to the resolver's reserved slot space."""
    text = str(profile_id or "")
    if text in RESERVED_PROFILE_SLOT_IDS:
        return True
    return any(
        text.endswith(suffix) and text[: -len(suffix)] in KNOWN_STAGES
        for suffix in RESERVED_PROFILE_SLOT_SUFFIXES
    )


def user_profile_id_error(profile_id: str) -> str:
    """Return why *profile_id* cannot be a user profile id, or "" if valid."""
    text = str(profile_id or "")
    if not USER_PROFILE_ID_PATTERN.match(text):
        return (
            f"Profile id {text!r} must match ^[a-z0-9_-]+$ "
            "(same rule as custom provider ids)."
        )
    if is_profile_slot_id(text):
        return (
            f"Profile id {text!r} collides with a reserved resolver slot "
            "(primary/batch/<stage>_model/<stage>_override); pick another id."
        )
    return ""

# Capability keys accepted in ``ModelProfile.capability_overrides``.
CAPABILITY_OVERRIDE_KEYS = frozenset({
    "sync_generation",
    "structured_output",
    "reasoning_request",
    "reasoning_response",
    "usage_stats",
    "remote_batch",
    "embedding",
    "context_limit_tokens",
    "context_budget_tokens",
})


class ExecutionStrategy(str, Enum):
    """How a stage's requests are transported and scheduled.

    Strategies own transport, scheduling, and task lifecycle only; prompts,
    context assembly, and translation schemas stay with the stage callers.
    """

    SYNC = "sync"
    GEMINI_BATCH = "gemini_batch"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class CredentialRef:
    """Where an adapter looks up credentials — a reference, never a value.

    ``name`` is the lookup key (API key slot id, keyring username, or env
    var name); ``env_name`` is the optional secondary environment lookup used
    by custom OpenAI-compatible providers.
    """

    kind: str
    name: str = ""
    env_name: str = ""

    def to_manifest_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "name": self.name, "env_name": self.env_name}

    @classmethod
    def from_manifest_dict(cls, payload: Mapping[str, Any]) -> "CredentialRef":
        return cls(
            kind=str(payload.get("kind") or CREDENTIAL_KIND_NONE),
            name=str(payload.get("name") or ""),
            env_name=str(payload.get("env_name") or ""),
        )


@dataclass(frozen=True)
class CapabilityFlag:
    """One boolean capability plus its explainable provenance."""

    supported: bool
    source: str = CAPABILITY_SOURCE_ADAPTER_DEFAULT

    def to_manifest_dict(self) -> dict[str, Any]:
        return {"supported": bool(self.supported), "source": str(self.source)}


@dataclass(frozen=True)
class StructuredOutputSpec:
    """Structured-output capability with provenance separated from basis.

    ``source`` uses the shared :data:`CAPABILITY_SOURCES` vocabulary (who
    vouches for this value); ``basis`` keeps the adapter-table detail of how
    the mode was chosen (``builtin`` / ``custom_openai_compatible`` /
    ``conservative_default``). Keeping them apart lets a #340 probe write
    ``source=probed`` without erasing which adapter table picked the mode.
    """

    mode: str
    source: str = CAPABILITY_SOURCE_ADAPTER_DEFAULT
    basis: str = ""

    @classmethod
    def from_capability(cls, cap: StructuredOutputCapability) -> "StructuredOutputSpec":
        source = (
            CAPABILITY_SOURCE_CONFIG_OVERRIDE
            if cap.source == CAPABILITY_SOURCE_CONFIG_OVERRIDE
            else CAPABILITY_SOURCE_ADAPTER_DEFAULT
        )
        return cls(mode=cap.mode, source=source, basis=cap.source)

    def to_manifest_dict(self) -> dict[str, str]:
        return {"mode": self.mode, "source": self.source, "basis": self.basis}

    @classmethod
    def from_manifest_dict(cls, payload: Mapping[str, Any]) -> "StructuredOutputSpec":
        return cls(
            mode=str(payload.get("mode") or "prompt_only_json"),
            source=str(payload.get("source") or CAPABILITY_SOURCE_ADAPTER_DEFAULT),
            basis=str(payload.get("basis") or ""),
        )


@dataclass(frozen=True)
class ModelCapabilities:
    """What a profile's provider/model actually supports.

    Sources must be one of :data:`CAPABILITY_SOURCES`; model-name string
    branching is not allowed here.  Conservative adapter defaults err on the
    safe side so unsupported combinations fail validation instead of
    producing degraded requests.

    Context numbers v1 convention: ``context_limit_tokens`` and
    ``context_budget_tokens`` share the single ``context_source`` (both
    values come from the same origin); mixed provenance is not expressible
    in v1 and #340 probe results must not be written into these two fields —
    probes feed per-capability ``source=probed`` flags and ``params``
    instead. Unknown values are ``None``, never sentinel strings.
    """

    sync_generation: CapabilityFlag = CapabilityFlag(False)
    structured_output: StructuredOutputSpec = StructuredOutputSpec(
        "prompt_only_json",
    )
    reasoning_request: CapabilityFlag = CapabilityFlag(False)
    reasoning_response: CapabilityFlag = CapabilityFlag(False)
    usage_stats: CapabilityFlag = CapabilityFlag(False)
    remote_batch: CapabilityFlag = CapabilityFlag(False)
    embedding: CapabilityFlag = CapabilityFlag(False)
    context_limit_tokens: int | None = None
    context_budget_tokens: int | None = None
    context_source: str = ""

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "sync_generation": self.sync_generation.to_manifest_dict(),
            "structured_output": self.structured_output.to_manifest_dict(),
            "reasoning_request": self.reasoning_request.to_manifest_dict(),
            "reasoning_response": self.reasoning_response.to_manifest_dict(),
            "usage_stats": self.usage_stats.to_manifest_dict(),
            "remote_batch": self.remote_batch.to_manifest_dict(),
            "embedding": self.embedding.to_manifest_dict(),
            "context_limit_tokens": self.context_limit_tokens,
            "context_budget_tokens": self.context_budget_tokens,
            "context_source": self.context_source,
        }

    @classmethod
    def from_manifest_dict(cls, payload: Mapping[str, Any]) -> "ModelCapabilities":
        def flag(key: str) -> CapabilityFlag:
            raw = payload.get(key) or {}
            return CapabilityFlag(
                supported=bool(raw.get("supported")),
                source=str(raw.get("source") or CAPABILITY_SOURCE_ADAPTER_DEFAULT),
            )

        limit = payload.get("context_limit_tokens")
        budget = payload.get("context_budget_tokens")
        return cls(
            sync_generation=flag("sync_generation"),
            structured_output=StructuredOutputSpec.from_manifest_dict(
                payload.get("structured_output") or {},
            ),
            reasoning_request=flag("reasoning_request"),
            reasoning_response=flag("reasoning_response"),
            usage_stats=flag("usage_stats"),
            remote_batch=flag("remote_batch"),
            embedding=flag("embedding"),
            context_limit_tokens=int(limit) if limit is not None else None,
            context_budget_tokens=int(budget) if budget is not None else None,
            context_source=str(payload.get("context_source") or ""),
        )


@dataclass(frozen=True)
class ModelProfile:
    """One addressable model configuration that tasks can route to.

    ``models`` is the explicit candidate pool for model rotation inside this
    profile (provider, credentials, and capabilities stay fixed); it holds
    exactly ``(model,)`` when no rotation list is configured.
    ``capability_overrides`` and ``params`` must stay non-sensitive: no API
    keys, only flags and generation parameters.
    """

    id: str
    label: str
    adapter: str
    provider: str
    model: str
    credential_ref: CredentialRef
    models: tuple[str, ...] = ()
    base_url: str = ""
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    capability_overrides: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    embedding_profile_id: str = ""

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "adapter": self.adapter,
            "provider": self.provider,
            "model": self.model,
            "models": list(self.models or (self.model,)),
            "credential_ref": self.credential_ref.to_manifest_dict(),
            "base_url": self.base_url,
            "extra_headers": dict(self.extra_headers or {}),
            "capability_overrides": dict(self.capability_overrides or {}),
            "params": dict(self.params or {}),
            "embedding_profile_id": self.embedding_profile_id,
        }

    @classmethod
    def from_manifest_dict(cls, payload: Mapping[str, Any]) -> "ModelProfile":
        return cls(
            id=str(payload.get("id") or ""),
            label=str(payload.get("label") or ""),
            adapter=str(payload.get("adapter") or ""),
            provider=str(payload.get("provider") or ""),
            model=str(payload.get("model") or ""),
            credential_ref=CredentialRef.from_manifest_dict(
                payload.get("credential_ref") or {},
            ),
            models=tuple(str(item) for item in payload.get("models") or ()),
            base_url=str(payload.get("base_url") or ""),
            extra_headers=dict(payload.get("extra_headers") or {}),
            capability_overrides=dict(payload.get("capability_overrides") or {}),
            params=dict(payload.get("params") or {}),
            embedding_profile_id=str(payload.get("embedding_profile_id") or ""),
        )


@dataclass(frozen=True)
class TaskRoute:
    """Which profile and strategy one stage uses, plus why."""

    stage: str
    profile_id: str
    strategy: ExecutionStrategy
    source: str

    def to_manifest_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "profile_id": self.profile_id,
            "strategy": self.strategy.value,
            "source": self.source,
        }

    @classmethod
    def from_manifest_dict(cls, payload: Mapping[str, Any]) -> "TaskRoute":
        return cls(
            stage=str(payload.get("stage") or ""),
            profile_id=str(payload.get("profile_id") or ""),
            strategy=ExecutionStrategy(str(payload.get("strategy") or "sync")),
            source=str(payload.get("source") or ""),
        )


@dataclass(frozen=True)
class ConfigOrigin:
    """Which config file contributed to a routing plan (path + fingerprint).

    Fingerprints hash non-sensitive content only. Populated by the P2 wiring
    when a plan is first written into a manifest; plans built purely in
    memory may carry none.
    """

    kind: str  # e.g. "translator_config" | "game_config"
    path: str
    fingerprint: str = ""

    def to_manifest_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": self.path,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_manifest_dict(cls, payload: Mapping[str, Any]) -> "ConfigOrigin":
        return cls(
            kind=str(payload.get("kind") or ""),
            path=str(payload.get("path") or ""),
            fingerprint=str(payload.get("fingerprint") or ""),
        )


@dataclass(frozen=True)
class ModelRoutingPlan:
    """Immutable per-run snapshot of profiles, routes, and capabilities.

    Built once at run start; later config changes must not affect a run that
    already holds a plan. The manifest snapshot carries references only, so
    answering "which provider/model/strategy/capabilities did each stage
    actually use" never leaks credentials. ``primary_profile_id`` is a role
    pointer (which profile unconfigured stages inherit), not a user-facing
    profile identity.
    """

    schema_version: int
    primary_profile_id: str
    profiles: Mapping[str, ModelProfile]
    routes: Mapping[str, TaskRoute]
    capabilities: Mapping[str, ModelCapabilities]
    created_at: str = ""
    config_origins: tuple[ConfigOrigin, ...] = ()

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "primary_profile_id": self.primary_profile_id,
            "profiles": {
                profile_id: profile.to_manifest_dict()
                for profile_id, profile in sorted(self.profiles.items())
            },
            "routes": {
                stage: route.to_manifest_dict()
                for stage, route in sorted(self.routes.items())
            },
            "capabilities": {
                profile_id: caps.to_manifest_dict()
                for profile_id, caps in sorted(self.capabilities.items())
            },
            "config_origins": [
                origin.to_manifest_dict() for origin in self.config_origins
            ],
        }

    @classmethod
    def from_manifest_dict(cls, payload: Mapping[str, Any]) -> "ModelRoutingPlan":
        return cls(
            schema_version=int(payload.get("schema_version") or 0),
            primary_profile_id=str(payload.get("primary_profile_id") or ""),
            profiles={
                str(profile_id): ModelProfile.from_manifest_dict(profile)
                for profile_id, profile in (payload.get("profiles") or {}).items()
            },
            routes={
                str(stage): TaskRoute.from_manifest_dict(route)
                for stage, route in (payload.get("routes") or {}).items()
            },
            capabilities={
                str(profile_id): ModelCapabilities.from_manifest_dict(caps)
                for profile_id, caps in (payload.get("capabilities") or {}).items()
            },
            created_at=str(payload.get("created_at") or ""),
            config_origins=tuple(
                ConfigOrigin.from_manifest_dict(origin)
                for origin in payload.get("config_origins") or ()
            ),
        )


@dataclass(frozen=True)
class RoutingValidationIssue:
    """One machine-decidable routing problem found before a run starts."""

    code: str
    message: str
    stage: str = ""
    profile_id: str = ""
    missing_capabilities: tuple[str, ...] = ()

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "profile_id": self.profile_id,
            "missing_capabilities": list(self.missing_capabilities),
        }


def routing_validation_error(
    issues: tuple[RoutingValidationIssue, ...],
) -> MachineContractError:
    """Convert validation issues into the stable CLI refusal contract."""
    primary = issues[0]
    return MachineContractError(
        primary.message,
        code_name=primary.code,
        suggested_action=_SUGGESTED_ACTION_BY_CODE.get(
            primary.code, "inspect_configuration_and_artifacts",
        ),
        semantic_exit_code=EXIT_INVALID_STATE,
        retryable=False,
        details={
            "issues": [issue.to_manifest_dict() for issue in issues],
            "stage": primary.stage,
            "profile_id": primary.profile_id,
            "missing_capabilities": list(primary.missing_capabilities),
        },
    )


def routing_resolution_issue(
    exc: Exception,
    *,
    stage: str = "",
) -> RoutingValidationIssue:
    """Convert a resolver configuration exception into the stable issue shape."""
    return RoutingValidationIssue(
        MODEL_PROFILE_INVALID,
        f"Model routing configuration is invalid: {exc}",
        stage=stage,
    )


def routing_resolution_error(
    exc: Exception,
    *,
    stage: str = "",
) -> MachineContractError:
    """Convert a resolver configuration exception into the CLI refusal."""
    return routing_validation_error((routing_resolution_issue(exc, stage=stage),))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean_str(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _gemini_capabilities(overrides: Mapping[str, Any]) -> ModelCapabilities:
    """Adapter defaults for the direct google-genai client."""
    caps = ModelCapabilities(
        sync_generation=CapabilityFlag(True),
        structured_output=StructuredOutputSpec(
            mode="strict_json_schema",
            source=CAPABILITY_SOURCE_ADAPTER_DEFAULT,
            basis=STRUCTURED_OUTPUT_BASIS_BUILTIN,
        ),
        reasoning_request=CapabilityFlag(True),
        reasoning_response=CapabilityFlag(True),
        usage_stats=CapabilityFlag(True),
        remote_batch=CapabilityFlag(True),
        embedding=CapabilityFlag(True),
    )
    return _apply_capability_overrides(caps, overrides)


def _litellm_capabilities(
    provider: str,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None,
    overrides: Mapping[str, Any],
) -> ModelCapabilities:
    """Adapter defaults for the LiteLLM sync backend.

    ``thinking_config`` is intentionally dropped by the LiteLLM adapter (see
    ``LiteLLMSyncBackend._build_request_context``), so reasoning stays
    conservatively unsupported; remote batch and embeddings are non-goals of
    the first phase (#341 will revisit embeddings).
    """
    caps = ModelCapabilities(
        sync_generation=CapabilityFlag(True),
        structured_output=StructuredOutputSpec.from_capability(
            structured_output_capability(provider, custom_providers),
        ),
        reasoning_request=CapabilityFlag(False),
        reasoning_response=CapabilityFlag(False),
        usage_stats=CapabilityFlag(True),
        remote_batch=CapabilityFlag(False),
        embedding=CapabilityFlag(False),
    )
    return _apply_capability_overrides(caps, overrides)


def _apply_capability_overrides(
    caps: ModelCapabilities,
    overrides: Mapping[str, Any],
) -> ModelCapabilities:
    if not overrides:
        return caps
    data: dict[str, Any] = {
        "sync_generation": caps.sync_generation,
        "structured_output": caps.structured_output,
        "reasoning_request": caps.reasoning_request,
        "reasoning_response": caps.reasoning_response,
        "usage_stats": caps.usage_stats,
        "remote_batch": caps.remote_batch,
        "embedding": caps.embedding,
        "context_limit_tokens": caps.context_limit_tokens,
        "context_budget_tokens": caps.context_budget_tokens,
        "context_source": caps.context_source,
    }
    for key, value in (overrides or {}).items():
        if key == "structured_output" and isinstance(value, Mapping):
            data["structured_output"] = StructuredOutputSpec(
                mode=str(value.get("mode") or caps.structured_output.mode),
                source=CAPABILITY_SOURCE_CONFIG_OVERRIDE,
                # Keep the adapter-table basis the override replaced so the
                # manifest still explains what the default would have been.
                basis=caps.structured_output.basis,
            )
        elif key in {"context_limit_tokens", "context_budget_tokens"}:
            data[key] = int(value) if value is not None else None
            data["context_source"] = CAPABILITY_SOURCE_CONFIG_OVERRIDE
        elif key in data and isinstance(data[key], CapabilityFlag):
            data[key] = CapabilityFlag(
                bool(value),
                CAPABILITY_SOURCE_CONFIG_OVERRIDE,
            )
    return ModelCapabilities(**data)


def resolve_capabilities(
    profile: ModelProfile,
    *,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
) -> ModelCapabilities:
    """Resolve one profile's capabilities from adapter defaults + overrides."""
    if profile.adapter == ADAPTER_GEMINI:
        return _gemini_capabilities(profile.capability_overrides)
    if profile.adapter == ADAPTER_LITELLM:
        return _litellm_capabilities(
            profile.provider,
            custom_providers,
            profile.capability_overrides,
        )
    return _apply_capability_overrides(
        ModelCapabilities(
            structured_output=StructuredOutputSpec(
                mode="prompt_only_json",
                source=CAPABILITY_SOURCE_ADAPTER_DEFAULT,
                basis=STRUCTURED_OUTPUT_BASIS_CONSERVATIVE_DEFAULT,
            ),
        ),
        profile.capability_overrides,
    )


def _sync_profile(
    profile_id: str,
    model: str,
    sync_backend: str,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None,
    *,
    label: str = "",
    models: tuple[str, ...] = (),
) -> ModelProfile:
    """Build a profile for the sync execution path.

    The adapter mirrors the runtime branch exactly: ``sync.backend`` decides
    gemini-direct vs LiteLLM, so a provider-prefixed model under the gemini
    backend stays a gemini-adapter profile and is rejected by validation
    instead of being rerouted behind the config's back.
    """
    if sync_backend == SYNC_BACKEND_LITELLM:
        provider = provider_from_model(model)
        custom = (custom_providers or {}).get(provider)
        if custom is not None and getattr(custom, "requires_key", None) is None:
            custom = None
        if custom is not None:
            credential = CredentialRef(
                CREDENTIAL_KIND_NONE if not custom.requires_key
                else CREDENTIAL_KIND_KEYRING,
                name=custom.id,
                env_name=custom.api_key_env,
            )
        else:
            credential = CredentialRef(CREDENTIAL_KIND_KEYRING, provider)
        return ModelProfile(
            id=profile_id,
            label=label or provider_display_label(provider, custom_providers) or model,
            adapter=ADAPTER_LITELLM,
            provider=provider,
            model=model,
            credential_ref=credential,
            models=models or (model,),
            base_url=custom.base_url if custom is not None else "",
        )
    return ModelProfile(
        id=profile_id,
        label=label or "Gemini",
        adapter=ADAPTER_GEMINI,
        provider="gemini",
        model=model,
        credential_ref=CredentialRef(CREDENTIAL_KIND_API_KEYS_JSON, "api_keys"),
        models=models or (model,),
    )


def _batch_profile(
    model: str,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None,
    *,
    profile_id: str = "batch",
    label: str = "Gemini Batch",
) -> ModelProfile:
    """Build the Gemini batch transport profile.

    A provider-prefixed batch model is kept as a litellm-adapter profile on
    purpose so validation rejects it with a machine-decidable
    missing-``remote_batch`` error instead of shipping it to a Gemini-only
    transport.
    """
    provider = provider_from_model(model)
    if provider:
        return ModelProfile(
            id=profile_id,
            label=label or provider_display_label(provider, custom_providers) or model,
            adapter=ADAPTER_LITELLM,
            provider=provider,
            model=model,
            credential_ref=CredentialRef(CREDENTIAL_KIND_KEYRING, provider),
            models=(model,),
        )
    return ModelProfile(
        id=profile_id,
        label=label,
        adapter=ADAPTER_GEMINI,
        provider="gemini",
        model=model,
        credential_ref=CredentialRef(CREDENTIAL_KIND_API_KEYS_JSON, "api_keys"),
        models=(model,),
    )


def build_profile_registry(
    translator_config: Mapping[str, Any],
    *,
    game_config: Mapping[str, Any] | None = None,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
) -> dict[str, ModelProfile]:
    """Build every profile addressable by the legacy config, read-only.

    Always present: ``primary`` (sync execution) and ``batch`` (Gemini batch
    execution). Stage-specific profiles appear only when their config key is
    set.
    """
    sync_cfg = _mapping(translator_config.get("sync"))
    batch_cfg = _mapping(translator_config.get("batch"))
    game_cfg = _mapping(game_config)

    sync_backend = _clean_str(sync_cfg.get("backend")).lower() or SYNC_BACKEND_GEMINI
    sync_model = _clean_str(sync_cfg.get("model"))
    batch_model = (
        _clean_str(batch_cfg.get("model"))
        or _clean_str(game_cfg.get("batch_model"))
        or DEFAULT_GEMINI_TRANSLATION_MODEL
    )
    primary_model = sync_model or batch_model

    raw_models = sync_cfg.get("models")
    models = (
        tuple(_clean_str(item) for item in raw_models if _clean_str(item))
        if isinstance(raw_models, (list, tuple))
        else ()
    )

    registry: dict[str, ModelProfile] = {
        "primary": _sync_profile(
            "primary",
            primary_model,
            sync_backend,
            custom_providers,
            label="Primary sync model",
            models=models,
        ),
        "batch": _batch_profile(batch_model, custom_providers),
    }

    project_analysis_model = _clean_str(
        _mapping(batch_cfg.get("project_analysis")).get("model"),
    )
    if project_analysis_model:
        registry["project_analysis_model"] = _sync_profile(
            "project_analysis_model",
            project_analysis_model,
            sync_backend,
            custom_providers,
            label="Project analysis model",
        )

    final_review_model = _clean_str(
        _mapping(batch_cfg.get("final_review")).get("model"),
    )
    if final_review_model:
        # Final review is batch-executed, so its dedicated model always runs
        # on the Gemini batch transport.
        registry["final_review_model"] = _batch_profile(
            final_review_model,
            custom_providers,
            profile_id="final_review_model",
            label="Final review model",
        )
    return registry


def resolve_routing_plan(
    translator_config: Mapping[str, Any],
    *,
    game_config: Mapping[str, Any] | None = None,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
    execution: str | ExecutionStrategy = ExecutionStrategy.SYNC,
    stage_overrides: Mapping[str, str] | None = None,
    created_at: str = "",
) -> ModelRoutingPlan:
    """Resolve the immutable routing snapshot for one run.

    ``execution`` selects the strategy for stages that support both ways
    (translation / keyword / revision); final review is always Gemini batch
    and project analysis is always sync, matching today's CLI surface.
    ``stage_overrides`` maps a stage to an explicit model string (A/B
    ``--model``, manifest model); explicit overrides win over the primary
    profile (issue #345 option B semantics).
    """
    sync_cfg = _mapping(translator_config.get("sync"))
    batch_cfg = _mapping(translator_config.get("batch"))
    game_cfg = _mapping(game_config)

    sync_backend = _clean_str(sync_cfg.get("backend")).lower() or SYNC_BACKEND_GEMINI
    if sync_backend not in KNOWN_SYNC_BACKENDS:
        raise ModelRoutingConfigError(
            f"Unsupported sync backend: {sync_backend}. "
            "Choose 'gemini' or 'litellm'."
        )

    strategy = (
        execution
        if isinstance(execution, ExecutionStrategy)
        else ExecutionStrategy(str(execution))
    )

    profiles = build_profile_registry(
        translator_config,
        game_config=game_config,
        custom_providers=custom_providers,
    )
    gemini_batch_override_stages = {STAGE_FINAL_REVIEW}
    if strategy is ExecutionStrategy.GEMINI_BATCH:
        gemini_batch_override_stages.update(
            {STAGE_TRANSLATION, STAGE_KEYWORD, STAGE_REVISION},
        )
    overrides = dict(stage_overrides or {})
    for stage in list(overrides):
        model = _clean_str(overrides[stage])
        if not model:
            overrides.pop(stage)
            continue
        if stage in gemini_batch_override_stages:
            profiles[f"{stage}_override"] = _batch_profile(
                model,
                custom_providers,
                profile_id=f"{stage}_override",
                label=f"{stage} explicit model",
            )
        else:
            profiles[f"{stage}_override"] = _sync_profile(
                f"{stage}_override",
                model,
                sync_backend,
                custom_providers,
                label=f"{stage} explicit model",
            )

    sync_model = _clean_str(sync_cfg.get("model"))
    batch_model = (
        _clean_str(batch_cfg.get("model"))
        or _clean_str(game_cfg.get("batch_model"))
    )

    def translation_source() -> str:
        if strategy is ExecutionStrategy.GEMINI_BATCH:
            if batch_model:
                return ROUTE_SOURCE_STAGE_CONFIG
            return ROUTE_SOURCE_BUILTIN_DEFAULT
        if sync_model:
            return ROUTE_SOURCE_STAGE_CONFIG
        if batch_model:
            return ROUTE_SOURCE_INHERITED
        return ROUTE_SOURCE_BUILTIN_DEFAULT

    routes: dict[str, TaskRoute] = {}

    def route(
        stage: str,
        profile_id: str,
        stage_strategy: ExecutionStrategy,
        source: str,
    ) -> None:
        routes[stage] = TaskRoute(
            stage=stage,
            profile_id=profile_id,
            strategy=stage_strategy,
            source=source,
        )

    if strategy is ExecutionStrategy.GEMINI_BATCH:
        route(STAGE_TRANSLATION, "batch", strategy, translation_source())
        route(STAGE_KEYWORD, "batch", strategy, ROUTE_SOURCE_INHERITED)
        route(STAGE_REVISION, "batch", strategy, ROUTE_SOURCE_INHERITED)
    else:
        route(STAGE_TRANSLATION, "primary", strategy, translation_source())
        route(STAGE_KEYWORD, "primary", strategy, ROUTE_SOURCE_INHERITED)
        route(STAGE_REVISION, "primary", strategy, ROUTE_SOURCE_INHERITED)

    for stage in (STAGE_TRANSLATION, STAGE_KEYWORD, STAGE_REVISION):
        if stage in overrides:
            route(
                stage,
                f"{stage}_override",
                routes[stage].strategy,
                ROUTE_SOURCE_EXPLICIT,
            )

    if STAGE_PROJECT_ANALYSIS in overrides:
        route(
            STAGE_PROJECT_ANALYSIS,
            "project_analysis_override",
            ExecutionStrategy.SYNC,
            ROUTE_SOURCE_EXPLICIT,
        )
    elif "project_analysis_model" in profiles:
        route(
            STAGE_PROJECT_ANALYSIS,
            "project_analysis_model",
            ExecutionStrategy.SYNC,
            ROUTE_SOURCE_STAGE_CONFIG,
        )
    else:
        route(
            STAGE_PROJECT_ANALYSIS,
            "primary",
            ExecutionStrategy.SYNC,
            ROUTE_SOURCE_INHERITED,
        )

    if STAGE_FINAL_REVIEW in overrides:
        route(
            STAGE_FINAL_REVIEW,
            "final_review_override",
            ExecutionStrategy.GEMINI_BATCH,
            ROUTE_SOURCE_EXPLICIT,
        )
    elif "final_review_model" in profiles:
        route(
            STAGE_FINAL_REVIEW,
            "final_review_model",
            ExecutionStrategy.GEMINI_BATCH,
            ROUTE_SOURCE_STAGE_CONFIG,
        )
    else:
        route(
            STAGE_FINAL_REVIEW,
            "batch",
            ExecutionStrategy.GEMINI_BATCH,
            ROUTE_SOURCE_INHERITED,
        )

    if STAGE_AB_EXPERIMENT in overrides:
        route(
            STAGE_AB_EXPERIMENT,
            "ab_experiment_override",
            ExecutionStrategy.SYNC,
            ROUTE_SOURCE_EXPLICIT,
        )
    else:
        route(
            STAGE_AB_EXPERIMENT,
            "primary",
            ExecutionStrategy.SYNC,
            ROUTE_SOURCE_INHERITED,
        )

    capabilities = {
        profile_id: resolve_capabilities(profile, custom_providers=custom_providers)
        for profile_id, profile in profiles.items()
    }
    return ModelRoutingPlan(
        schema_version=MODEL_PROFILE_SCHEMA_VERSION,
        primary_profile_id="primary",
        profiles=profiles,
        routes=routes,
        capabilities=capabilities,
        created_at=created_at or utc_now_iso(),
    )


def validate_routing_plan(
    plan: ModelRoutingPlan,
    *,
    stages: Collection[str] | None = None,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
    environ: Mapping[str, str] | None = None,
    keyring_has_credential: Callable[[str], bool] | None = None,
) -> tuple[RoutingValidationIssue, ...]:
    """Return every machine-decidable routing problem.

    An empty tuple means the plan may start. When ``stages`` is supplied, only
    those routes and the profiles they reference are checked, so an unrelated
    optional stage cannot block the current task. Credential availability is
    checked honestly: environment references are verified against
    ``environ`` (default: the real process environment); keyring references
    are only flagged when a ``keyring_has_credential`` probe is supplied and
    reports the slot empty.

    Built-in LiteLLM profiles (``CredentialRef(KEYRING, provider)`` with an
    empty ``env_name``) are not probed even when a probe is supplied. Those
    providers also read native environment variables such as ``OPENAI_API_KEY``
    that this module does not model, so an empty keyring slot is not enough to
    call the reference missing. Custom-provider and named-env refs are
    machine-decidable here.
    """
    issues: list[RoutingValidationIssue] = []
    env_lookup = os.environ if environ is None else environ
    selected_stages = (
        frozenset(str(stage) for stage in stages)
        if stages is not None
        else frozenset(plan.routes)
    )
    selected_profile_ids: set[str] = (
        set(plan.profiles) if stages is None else set()
    )

    for stage in sorted(selected_stages):
        if stage not in KNOWN_STAGES:
            issues.append(RoutingValidationIssue(
                MODEL_PROFILE_INVALID,
                f"Unknown task stage in routing plan: {stage}",
                stage=stage,
            ))
            continue
        task_route = plan.routes.get(stage)
        if task_route is None:
            issues.append(RoutingValidationIssue(
                MODEL_PROFILE_INVALID,
                f"Routing plan has no route for required stage {stage}.",
                stage=stage,
            ))
            continue
        selected_profile_ids.add(task_route.profile_id)
        profile = plan.profiles.get(task_route.profile_id)
        if profile is None:
            issues.append(RoutingValidationIssue(
                MODEL_PROFILE_INVALID,
                f"Route for stage {stage} references unknown profile "
                f"{task_route.profile_id}.",
                stage=stage,
                profile_id=task_route.profile_id,
            ))
            continue
        caps = plan.capabilities.get(task_route.profile_id)
        if caps is None:
            issues.append(RoutingValidationIssue(
                MODEL_PROFILE_INVALID,
                f"Profile {task_route.profile_id} has no capability snapshot.",
                stage=stage,
                profile_id=task_route.profile_id,
            ))
            continue

        if (
            task_route.strategy is ExecutionStrategy.SYNC
            and not caps.sync_generation.supported
        ):
            issues.append(RoutingValidationIssue(
                MODEL_ROUTE_CAPABILITY_MISSING,
                f"Profile {task_route.profile_id} cannot serve stage {stage}: "
                "sync generation is not supported.",
                stage=stage,
                profile_id=task_route.profile_id,
                missing_capabilities=("sync_generation",),
            ))
        if task_route.strategy is ExecutionStrategy.GEMINI_BATCH:
            missing: list[str] = []
            if profile.adapter != ADAPTER_GEMINI:
                missing.append("gemini_adapter")
            if not caps.remote_batch.supported:
                missing.append("remote_batch")
            if missing:
                issues.append(RoutingValidationIssue(
                    MODEL_ROUTE_CAPABILITY_MISSING,
                    f"Profile {task_route.profile_id} cannot serve stage "
                    f"{stage} with the gemini_batch strategy.",
                    stage=stage,
                    profile_id=task_route.profile_id,
                    missing_capabilities=tuple(missing),
                ))

    for profile_id in sorted(selected_profile_ids):
        profile = plan.profiles.get(profile_id)
        if profile is None:
            # The route-level issue above already describes this failure.
            continue
        if profile.adapter not in KNOWN_ADAPTERS:
            issues.append(RoutingValidationIssue(
                MODEL_PROFILE_INVALID,
                f"Profile {profile_id} uses unknown adapter {profile.adapter}.",
                profile_id=profile_id,
            ))
            continue
        if profile.adapter == ADAPTER_LITELLM and not profile.provider:
            issues.append(RoutingValidationIssue(
                MODEL_PROFILE_INVALID,
                f"Profile {profile_id} has a model without a provider prefix; "
                "LiteLLM needs '<provider>/<model>'.",
                profile_id=profile_id,
            ))
        if profile.adapter == ADAPTER_GEMINI and "/" in profile.model:
            issues.append(RoutingValidationIssue(
                MODEL_PROFILE_INVALID,
                f"Profile {profile_id} uses provider-prefixed model "
                f"{profile.model} on the direct Gemini adapter; configure "
                "sync.backend 'litellm' or use a plain Gemini model id.",
                profile_id=profile_id,
            ))
        unknown_overrides = sorted(
            str(key) for key in profile.capability_overrides
            if key not in CAPABILITY_OVERRIDE_KEYS
        )
        if unknown_overrides:
            issues.append(RoutingValidationIssue(
                MODEL_PROFILE_INVALID,
                f"Profile {profile_id} has unknown capability overrides: "
                + ", ".join(unknown_overrides),
                profile_id=profile_id,
            ))

        ref = profile.credential_ref
        custom = (custom_providers or {}).get(profile.provider)
        requires_key = custom.requires_key if custom is not None else True
        if not requires_key:
            continue
        if ref.kind == CREDENTIAL_KIND_ENV:
            if ref.name and ref.name not in env_lookup:
                issues.append(RoutingValidationIssue(
                    MODEL_PROFILE_CREDENTIAL_REF_MISSING,
                    f"Profile {profile_id} references environment variable "
                    f"{ref.name}, which is not set.",
                    profile_id=profile_id,
                ))
        elif ref.kind == CREDENTIAL_KIND_KEYRING:
            env_missing = not (ref.env_name and ref.env_name in env_lookup)
            # Built-in providers also have native env vars this module does
            # not model; only custom/env_name refs are machine-decidable.
            keyring_missing = (
                (custom is not None or bool(ref.env_name))
                and keyring_has_credential is not None
                and ref.name
                and not keyring_has_credential(ref.name)
            )
            if env_missing and keyring_missing:
                detail = f"keyring slot {ref.name or '(unnamed)'} is empty"
                if ref.env_name:
                    detail += f" and {ref.env_name} is not set"
                issues.append(RoutingValidationIssue(
                    MODEL_PROFILE_CREDENTIAL_REF_MISSING,
                    f"Profile {profile_id} has no usable credential: {detail}.",
                    profile_id=profile_id,
                ))

    return tuple(issues)


_MANIFEST_MODE_TO_STAGE = {
    "translation": STAGE_TRANSLATION,
    "keyword_extraction": STAGE_KEYWORD,
    "revision": STAGE_REVISION,
    "final_review": STAGE_FINAL_REVIEW,
}


def stage_for_manifest_mode(mode: str) -> str:
    """Map a package/manifest mode string to a routing stage."""
    text = str(mode or "").strip()
    if text in _MANIFEST_MODE_TO_STAGE:
        return _MANIFEST_MODE_TO_STAGE[text]
    if text in KNOWN_STAGES:
        return text
    return STAGE_TRANSLATION


def profile_for_route(plan: ModelRoutingPlan, route: TaskRoute) -> ModelProfile:
    """Return the frozen profile a route points at."""
    try:
        return plan.profiles[route.profile_id]
    except KeyError as exc:
        raise ValueError(
            f"Route for stage {route.stage} references unknown profile "
            f"{route.profile_id}."
        ) from exc


def override_gemini_batch_stage(
    plan: ModelRoutingPlan,
    stage: str,
    model: str,
    *,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
) -> ModelRoutingPlan:
    """Replace one Gemini-batch stage on a frozen plan.

    Other stages, profiles, and ``config_origins`` stay as they were. The
    override is always a Gemini-batch profile, even when ``sync.backend`` is
    LiteLLM, so ``submit --model gemini-...`` cannot be rebuilt as a sync
    LiteLLM route.
    """
    cleaned = _clean_str(model)
    if not cleaned:
        raise ValueError("Gemini batch stage override requires a model id.")
    if stage not in KNOWN_STAGES:
        raise ValueError(f"Unknown task stage in routing plan: {stage}")
    profile_id = f"{stage}_override"
    profile = _batch_profile(
        cleaned,
        custom_providers,
        profile_id=profile_id,
        label=f"{stage} explicit model",
    )
    profiles = dict(plan.profiles)
    profiles[profile_id] = profile
    capabilities = dict(plan.capabilities)
    capabilities[profile_id] = resolve_capabilities(
        profile, custom_providers=custom_providers,
    )
    routes = dict(plan.routes)
    routes[stage] = TaskRoute(
        stage=stage,
        profile_id=profile_id,
        strategy=ExecutionStrategy.GEMINI_BATCH,
        source=ROUTE_SOURCE_EXPLICIT,
    )
    return replace(
        plan,
        profiles=profiles,
        routes=routes,
        capabilities=capabilities,
    )


def resolve_routing_plan_from_runtime(
    *,
    sync_backend: str,
    sync_model: str = "",
    batch_model: str = "",
    project_analysis_model: str = "",
    final_review_model: str = "",
    sync_models: tuple[str, ...] | list[str] = (),
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
    execution: str | ExecutionStrategy = ExecutionStrategy.SYNC,
    stage_overrides: Mapping[str, str] | None = None,
    created_at: str = "",
    config_origins: tuple[ConfigOrigin, ...] = (),
) -> ModelRoutingPlan:
    """Build a plan from already-loaded runtime settings.

    Callers snapshot ``SYNC_MODEL`` / ``BATCH_MODEL`` / stage models into
    this helper *once* at run start. Later mutations of those globals must
    not be re-read.
    """
    translator_config: dict[str, Any] = {
        "sync": {"backend": _clean_str(sync_backend) or SYNC_BACKEND_GEMINI},
        "batch": {},
    }
    if _clean_str(sync_model):
        translator_config["sync"]["model"] = _clean_str(sync_model)
    models = tuple(
        _clean_str(item) for item in (sync_models or ()) if _clean_str(item)
    )
    if models:
        translator_config["sync"]["models"] = list(models)
    if _clean_str(batch_model):
        translator_config["batch"]["model"] = _clean_str(batch_model)
    if _clean_str(project_analysis_model):
        translator_config["batch"]["project_analysis"] = {
            "model": _clean_str(project_analysis_model),
        }
    if _clean_str(final_review_model):
        translator_config["batch"]["final_review"] = {
            "model": _clean_str(final_review_model),
        }
    plan = resolve_routing_plan(
        translator_config,
        custom_providers=custom_providers,
        execution=execution,
        stage_overrides=stage_overrides,
        created_at=created_at,
    )
    if config_origins:
        plan = replace(plan, config_origins=tuple(config_origins))
    return plan


def build_sync_backend(
    profile: ModelProfile,
    *,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
    diagnostic_api_key: str | None = None,
    client: Any = None,
    serialize_response: Callable[..., Any] | None = None,
    extract_text: Callable[..., Any] | None = None,
    extract_finish_reason: Callable[..., Any] | None = None,
    extract_usage: Callable[..., Any] | None = None,
):
    """Construct the sync backend for one profile.

    Deferred imports keep this module free of heavy SDK imports. Production
    sync callers must go through this function; they may pass a pre-built
    Gemini ``client`` (for API-key rotation) and usage extractors that match
    the existing request contract. ``diagnostic_api_key`` is an ephemeral GUI
    or doctor probe override and must never be persisted in a profile or
    manifest.
    """
    if profile.adapter == ADAPTER_GEMINI:
        import model_usage_ledger
        import translator_runtime as runtime
        from sync_model_backend import GeminiSyncBackend

        if client is None:
            runtime.configure_genai()
            client = runtime.create_genai_client()
        return GeminiSyncBackend(
            client,
            serialize_response=serialize_response or runtime.serialize_unknown,
            extract_text=extract_text or runtime.extract_text_from_response_payload,
            extract_finish_reason=(
                extract_finish_reason or runtime.extract_finish_reason
            ),
            extract_usage=extract_usage or model_usage_ledger.extract_provider_usage,
        )
    if profile.adapter == ADAPTER_LITELLM:
        from litellm_sync_backend import LiteLLMSyncBackend

        return LiteLLMSyncBackend(
            api_key=str(diagnostic_api_key or "").strip() or None,
            custom_providers=dict(custom_providers or {}),
        )
    raise ValueError(
        f"Profile {profile.id} uses adapter {profile.adapter}, which has no "
        "sync backend."
    )
