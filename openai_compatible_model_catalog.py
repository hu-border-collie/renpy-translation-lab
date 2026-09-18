"""Read-only model catalog discovery for the direct OpenAI-compatible adapter.

The provider model list is a discovery aid, never an allowlist: callers may
always keep a manually typed model id when the catalog endpoint is missing,
unreachable, or incomplete.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping

from openai_compatible_connection import (
    OpenAICompatibleConnection,
    redacted_endpoint,  # re-exported for CLI/GUI diagnostics
)
from sync_model_backend import (
    SyncBackendError,
    normalize_sync_timeout_seconds,
)

DEFAULT_CATALOG_TIMEOUT_SECONDS = 15
MAX_CATALOG_MODELS = 10_000


@dataclass(frozen=True)
class CatalogResponse:
    """Transport-neutral model-list response."""

    status: int
    body: bytes


CatalogTransport = Callable[[urllib.request.Request, float], CatalogResponse]


def _default_transport(
    request: urllib.request.Request,
    timeout: float,
) -> CatalogResponse:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return CatalogResponse(
                status=int(getattr(response, "status", 200) or 200),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        return CatalogResponse(
            status=int(getattr(exc, "code", 0) or 0),
            body=body,
        )


def parse_models_payload(payload: object) -> tuple[str, ...]:
    """Parse an OpenAI-style ``{"data":[{"id":"..."}]}`` model list.

    The parser keeps provider ids untouched (slashes preserved), skips
    non-object / empty entries, de-duplicates case-insensitively and returns a
    stable sorted tuple.  It deliberately does not filter by modality: the
    catalog is a discovery aid and a provider may expose text models this
    module cannot classify.
    """

    if not isinstance(payload, Mapping):
        return ()
    raw_data = payload.get("data")
    if not isinstance(raw_data, list):
        return ()
    models: dict[str, str] = {}
    for item in raw_data[:MAX_CATALOG_MODELS]:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        models.setdefault(model_id.casefold(), model_id)
    return tuple(sorted(models.values(), key=str.casefold))


def connection_for_profile(
    section: Mapping[str, object],
    profile_id: str,
) -> OpenAICompatibleConnection:
    """Build the shared connection object for one direct-adapter profile.

    Raises :class:`SyncBackendError` with ``unsupported_capability`` when the
    profile is missing or does not use the direct adapter, so CLI/GUI callers
    can surface a stable category instead of a traceback.
    """

    import model_profiles_editor as editor
    from model_profile import ADAPTER_OPENAI_COMPATIBLE

    view = editor.editor_view(section)
    profile = next(
        (
            item
            for item in view["profiles"]
            if str(item.get("id") or "") == str(profile_id or "")
        ),
        None,
    )
    if profile is None:
        raise SyncBackendError(
            "unsupported_capability",
            request_metadata={"reason": "unknown_profile"},
        )
    provider = next(
        (
            item
            for item in view["providers"]
            if str(item.get("id") or "") == str(profile.get("provider_id") or "")
        ),
        None,
    )
    if provider is None or str(provider.get("adapter") or "") != ADAPTER_OPENAI_COMPATIBLE:
        raise SyncBackendError(
            "unsupported_capability",
            request_metadata={"reason": "adapter_not_openai_compatible"},
        )
    credential_ref = provider.get("credential_ref")
    credential_ref = (
        dict(credential_ref) if isinstance(credential_ref, Mapping) else {}
    )
    extra_headers = provider.get("extra_headers")
    extra_headers = (
        dict(extra_headers) if isinstance(extra_headers, Mapping) else {}
    )
    return OpenAICompatibleConnection(
        provider=str(provider.get("provider") or ""),
        base_url=str(provider.get("base_url") or ""),
        models_url=str(provider.get("models_url") or ""),
        credential_ref=credential_ref,
        extra_headers=extra_headers,
    )


def _classify_http_error(status: int) -> str:
    if status in {401, 403}:
        return "authentication"
    if status == 429:
        return "rate_limit"
    if status == 408:
        return "timeout"
    if status in {500, 502, 503, 504}:
        return "service_unavailable"
    return "provider_error"


def fetch_models(
    connection: OpenAICompatibleConnection,
    *,
    timeout_seconds: int | float = DEFAULT_CATALOG_TIMEOUT_SECONDS,
    transport: CatalogTransport | None = None,
) -> tuple[str, ...]:
    """Fetch and parse one provider model catalog.

    Credentials and extra headers follow the same connection contract as
    generation.  HTTP and transport failures raise :class:`SyncBackendError`
    with the shared safe category; the response body is never echoed.
    """

    transport = transport or _default_transport
    timeout = normalize_sync_timeout_seconds(timeout_seconds)
    url = connection.provider_models_url()
    headers = connection.request_headers(json_content=False)
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        response = transport(request, timeout)
    except (socket.timeout, TimeoutError) as exc:
        raise SyncBackendError(
            "timeout",
            request_metadata={"reason": "catalog_timeout"},
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        category = (
            "timeout"
            if isinstance(reason, (socket.timeout, TimeoutError))
            else "provider_error"
        )
        raise SyncBackendError(
            category,
            request_metadata={"reason": "catalog_transport_error"},
        ) from exc
    except OSError as exc:
        raise SyncBackendError(
            "provider_error",
            request_metadata={"reason": "catalog_transport_error"},
        ) from exc

    if response.status < 200 or response.status >= 300:
        raise SyncBackendError(
            _classify_http_error(response.status),
            request_metadata={
                "status_code": response.status,
                "reason": "catalog_http_error",
            },
        )
    try:
        payload = json.loads(response.body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise SyncBackendError(
            "invalid_response",
            request_metadata={"reason": "catalog_not_json"},
        ) from exc
    models = parse_models_payload(payload)
    if not models:
        raise SyncBackendError(
            "invalid_response",
            request_metadata={"reason": "catalog_empty"},
        )
    return models
