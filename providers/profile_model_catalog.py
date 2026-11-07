"""Read-only model discovery shared by ModelProfile CLI and GUI entry points.

Gemini uses its official models.list endpoint and the profile's credential
reference. LiteLLM reuses existing native catalog endpoints and model-ID parsers
without importing its SDK. OpenAI-compatible discovery retains its connection contract.
Catalogs are suggestions, never allowlists or a reason to change a saved model.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from providers import openai_compatible_model_catalog as compatible
from providers.openai_compatible_model_catalog import (
    CatalogResponse as CatalogResponse,
    CatalogTransport as CatalogTransport,
)
from providers.model_profile import ADAPTER_GEMINI, ADAPTER_LITELLM, ADAPTER_OPENAI_COMPATIBLE
from providers import gemini_model_catalog, litellm_provider_config
from providers.openai_compatible_connection import (
    OpenAICompatibleConnection,
    redacted_endpoint as redacted_endpoint,
)
from providers.sync_model_backend import SyncBackendError, normalize_sync_timeout_seconds

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
SUPPORTED_CATALOG_ADAPTERS = frozenset({ADAPTER_GEMINI, ADAPTER_LITELLM, ADAPTER_OPENAI_COMPATIBLE})
MAX_GEMINI_CATALOG_PAGES = 10
CredentialLoader = Callable[[Mapping[str, object]], str]


@dataclass(frozen=True)
class GeminiCatalogConnection:
    """Official Gemini catalog identity; only credential references are stored."""

    credential_ref: Mapping[str, object]
    purpose: str = "generation"
    provider: str = "gemini"
    base_url: str = GEMINI_API_BASE

    def provider_models_url(self) -> str:
        return f"{GEMINI_API_BASE}/models"


@dataclass(frozen=True)
class LiteLLMCatalogConnection:
    """Discover a LiteLLM provider through its existing native catalog contract."""

    endpoint: litellm_provider_config.NativeCatalogEndpoint
    credential_ref: Mapping[str, object]
    extra_headers: Mapping[str, str]
    base_url: str

    @property
    def provider(self) -> str:
        return self.endpoint.provider

    def provider_models_url(self) -> str:
        return self.endpoint.url


def _profile_entries(section: Mapping[str, Any], profile_id: str) -> tuple[Mapping, Mapping]:
    if not isinstance(section, Mapping):
        return {}, {}
    profiles, providers = section.get("profiles"), section.get("providers")
    profile = profiles.get(profile_id) if isinstance(profiles, Mapping) else None
    if not isinstance(profile, Mapping) or not isinstance(profile.get("provider_id"), str):
        return {}, {}
    provider = providers.get(profile["provider_id"]) if isinstance(providers, Mapping) else None
    return profile, provider if isinstance(provider, Mapping) else {}


def connection_identity(section: Mapping[str, Any], profile_id: str) -> str:
    """Bind suggestions and late catalog results to a connection and purpose.

    The identity contains configuration and credential references only; reading
    it never resolves keys or imports a provider SDK.
    """
    profile, provider = _profile_entries(section, profile_id)
    provider_id = profile.get("provider_id")
    return json.dumps(
        [provider_id, profile.get("purpose", "generation"), {
            key: provider.get(key) for key in (
                "adapter", "provider", "base_url", "models_url", "credential_ref", "extra_headers",
            )
        }],
        sort_keys=True, ensure_ascii=True,
    )


def suggested_models(
    section: Mapping[str, Any], profile_id: str, config: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Offer configured and built-in IDs offline without selecting a model."""
    profiles = section.get("profiles")
    profiles = profiles if isinstance(profiles, Mapping) else {}
    profile, provider = _profile_entries(section, profile_id)
    provider_id = profile.get("provider_id")
    if not isinstance(provider_id, str):
        return ()
    purpose = profile.get("purpose", "generation")
    configured = []
    for item in (profile, *profiles.values()):
        if isinstance(item, Mapping) and item.get("provider_id") == provider_id and item.get("purpose", "generation") == purpose:
            configured.extend(gemini_model_catalog.normalize_model_names(item.get("model")))
            configured.extend(gemini_model_catalog.normalize_model_names(item.get("models")))
    builtin: list[str] | tuple[str, ...] = ()
    if provider.get("adapter") == ADAPTER_GEMINI:
        resolve = (
            gemini_model_catalog.resolve_gemini_embedding_models if purpose == "embedding"
            else gemini_model_catalog.resolve_gemini_translation_models
        )
        builtin = resolve(config)
    elif provider.get("adapter") == ADAPTER_LITELLM and purpose == "generation":
        builtin = litellm_provider_config.DEFAULT_MODELS.get(str(provider.get("provider")), ())
    return tuple(gemini_model_catalog.merge_model_lists(configured, builtin))


def adapter_for_profile(section: Mapping[str, Any], profile_id: str) -> str:
    """Read an adapter without loading credentials or making any requests."""

    profiles = section.get("profiles")
    providers = section.get("providers")
    if not isinstance(profiles, Mapping) or not isinstance(providers, Mapping):
        return ""
    profile = profiles.get(profile_id)
    if not isinstance(profile, Mapping):
        return ""
    provider_id = profile.get("provider_id")
    if not isinstance(provider_id, str):
        return ""
    provider = providers.get(provider_id)
    return str(provider.get("adapter") or "") if isinstance(provider, Mapping) else ""


def connection_for_profile(
    section: Mapping[str, Any], profile_id: str,
) -> OpenAICompatibleConnection | GeminiCatalogConnection | LiteLLMCatalogConnection:
    """Resolve a catalog connection with the same provider identity as the profile."""

    adapter = adapter_for_profile(section, profile_id)
    if adapter == ADAPTER_OPENAI_COMPATIBLE:
        return compatible.connection_for_profile(section, profile_id)
    if adapter == ADAPTER_LITELLM:
        profile = section["profiles"][profile_id]
        provider = section["providers"][profile["provider_id"]]
        if profile.get("purpose", "generation") != "generation":
            raise SyncBackendError("unsupported_capability")
        upstream = str(provider.get("provider") or "")
        endpoint = litellm_provider_config.native_catalog_endpoint(upstream)
        base_url = str(provider.get("base_url") or "")
        models_url = str(provider.get("models_url") or "")
        if base_url or models_url:
            url = OpenAICompatibleConnection(
                provider=upstream, base_url=base_url, models_url=models_url,
            ).provider_models_url()
            endpoint = (
                replace(endpoint, url=url) if endpoint is not None
                else litellm_provider_config.NativeCatalogEndpoint(
                    provider=upstream, url=url, label=upstream, source=upstream,
                    require_key=(provider.get("credential_ref") or {}).get("kind", "none") != "none",
                )
            )
        if endpoint is None:
            raise SyncBackendError("unsupported_capability")
        return LiteLLMCatalogConnection(
            endpoint=endpoint,
            credential_ref=dict(provider.get("credential_ref") or {}),
            extra_headers=dict(provider.get("extra_headers") or {}),
            base_url=base_url or endpoint.url.rsplit("/", 1)[0],
        )
    if adapter != ADAPTER_GEMINI:
        raise SyncBackendError("unsupported_capability")
    profile = section["profiles"][profile_id]
    provider = section["providers"][profile["provider_id"]]
    return GeminiCatalogConnection(
        credential_ref=dict(provider.get("credential_ref") or {}),
        purpose=str(profile.get("purpose") or "generation"),
    )


def _gemini_transport(request: urllib.request.Request, timeout: float) -> CatalogResponse:
    # Do not forward the Gemini key to a redirect target.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return CatalogResponse(status=response.status, body=response.read())
    except urllib.error.HTTPError as exc:
        # Neither the provider error body nor the request headers are diagnostics.
        exc.close()
        return CatalogResponse(status=exc.code, body=b"")


def _catalog_page(
    request: urllib.request.Request, timeout: float, transport: CatalogTransport,
) -> Mapping[str, Any]:
    try:
        response = transport(request, timeout)
    except (socket.timeout, TimeoutError) as exc:
        raise SyncBackendError("timeout") from exc
    except urllib.error.URLError as exc:
        category = (
            "timeout" if isinstance(exc.reason, (socket.timeout, TimeoutError)) else "provider_error"
        )
        raise SyncBackendError(category) from exc
    except OSError as exc:
        raise SyncBackendError("provider_error") from exc
    if not 200 <= response.status < 300:
        category = {
            400: "authentication", 401: "authentication", 403: "authentication",
            408: "timeout", 429: "rate_limit",
            500: "service_unavailable", 502: "service_unavailable",
            503: "service_unavailable", 504: "service_unavailable",
        }.get(response.status, "provider_error")
        raise SyncBackendError(category, request_metadata={"status_code": response.status})
    try:
        payload = json.loads(response.body)
    except (UnicodeError, ValueError) as exc:
        raise SyncBackendError("invalid_response") from exc
    if not isinstance(payload, Mapping):
        raise SyncBackendError("invalid_response")
    return payload


def _fetch_litellm_models(
    connection: LiteLLMCatalogConnection, timeout: float,
    transport: CatalogTransport | None, credential_loader: CredentialLoader | None,
) -> tuple[str, ...]:
    ref = dict(connection.credential_ref)
    kind = ref.get("kind", "none")
    if kind not in {"none", "env", "keyring"}:
        raise SyncBackendError("unsupported_capability")
    key = ""
    if kind != "none":
        if credential_loader is None:
            from providers.model_capability_probe import default_credential_loader

            credential_loader = default_credential_loader
        try:
            key = str(credential_loader({
                "adapter": ADAPTER_LITELLM, "provider": connection.provider,
                "credential_ref": ref,
            }) or "").strip()
        except Exception as exc:
            raise SyncBackendError("authentication") from exc
    if (kind != "none" or connection.endpoint.require_key) and not key:
        raise SyncBackendError("authentication")
    # Reuse header validation, then apply the native authentication style
    # (Anthropic uses x-api-key; local unauthenticated catalogs use neither).
    headers = OpenAICompatibleConnection(
        provider=connection.provider, base_url=connection.base_url,
        api_key=key, extra_headers=connection.extra_headers,
    ).request_headers(json_content=False)
    headers.pop("Authorization", None)
    headers.update(litellm_provider_config.build_native_catalog_headers(connection.endpoint, key))
    request = urllib.request.Request(connection.provider_models_url(), headers=headers, method="GET")
    payload = _catalog_page(request, timeout, transport or _gemini_transport)
    models = litellm_provider_config.models_from_native_catalog_payload(connection.endpoint, payload)
    if not models:
        raise SyncBackendError("invalid_response")
    return models


def fetch_models(
    connection: OpenAICompatibleConnection | GeminiCatalogConnection | LiteLLMCatalogConnection,
    *,
    timeout_seconds: int | float = compatible.DEFAULT_CATALOG_TIMEOUT_SECONDS,
    transport: CatalogTransport | None = None,
    credential_loader: CredentialLoader | None = None,
) -> tuple[str, ...]:
    """Fetch model IDs without generation, retries, config writes or key rotation.

    Gemini pagination shares one timeout budget and has a ten-page bound. The
    supported method filters generation and embedding profiles independently.
    Error text never includes provider response bodies or credential values.
    """

    if isinstance(connection, OpenAICompatibleConnection):
        return compatible.fetch_models(
            connection, timeout_seconds=timeout_seconds, transport=transport,
        )
    if isinstance(connection, LiteLLMCatalogConnection):
        return _fetch_litellm_models(
            connection, normalize_sync_timeout_seconds(timeout_seconds), transport, credential_loader,
        )
    if connection.purpose not in {"generation", "embedding"}:
        raise SyncBackendError("unsupported_capability")
    credential_kind = connection.credential_ref.get("kind")
    if credential_kind not in {"api_keys_json", "env"}:
        category = (
            "authentication" if credential_kind in {None, "none"} else "unsupported_capability"
        )
        raise SyncBackendError(category)
    if (
        credential_kind == "api_keys_json"
        and connection.credential_ref.get("name") not in {None, "", "api_keys"}
    ):
        raise SyncBackendError("unsupported_capability")
    if credential_loader is None:
        from providers.model_capability_probe import default_credential_loader

        credential_loader = default_credential_loader
    try:
        api_key = str(credential_loader({
            "adapter": ADAPTER_GEMINI, "provider": "gemini",
            "credential_ref": dict(connection.credential_ref),
        }) or "").strip()
    except Exception as exc:
        raise SyncBackendError("authentication") from exc
    if not api_key:
        raise SyncBackendError("authentication")
    method = "embedContent" if connection.purpose == "embedding" else "generateContent"
    deadline = time.monotonic() + normalize_sync_timeout_seconds(timeout_seconds)
    transport = transport or _gemini_transport
    page_token = ""
    seen_tokens: set[str] = set()
    models: dict[str, str] = {}
    for _ in range(MAX_GEMINI_CATALOG_PAGES):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SyncBackendError("timeout")
        params = {"pageSize": "1000"}
        if page_token:
            params["pageToken"] = page_token
        url = f"{connection.provider_models_url()}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url, headers={"x-goog-api-key": api_key, "Accept": "application/json"}, method="GET",
        )
        payload = _catalog_page(request, remaining, transport)
        if time.monotonic() >= deadline:
            raise SyncBackendError("timeout")
        if not isinstance(payload.get("models"), list) or len(payload["models"]) > 1000:
            raise SyncBackendError("invalid_response")
        for item in payload["models"]:
            if not isinstance(item, Mapping):
                continue
            methods = item.get("supportedGenerationMethods")
            name = item.get("name")
            if (
                not isinstance(methods, list) or method not in methods or not isinstance(name, str)
            ):
                continue
            model_id = name.strip().removeprefix("models/")
            if model_id:
                models.setdefault(model_id.casefold(), model_id)
        next_token = payload.get("nextPageToken", "")
        if not isinstance(next_token, str) or next_token in seen_tokens:
            raise SyncBackendError("invalid_response")
        if not next_token:
            if not models:
                raise SyncBackendError("invalid_response")
            return tuple(sorted(models.values(), key=str.casefold))
        seen_tokens.add(next_token)
        page_token = next_token
    raise SyncBackendError("invalid_response", request_metadata={"reason": "catalog_page_limit"})
