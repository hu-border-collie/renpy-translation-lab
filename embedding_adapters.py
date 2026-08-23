# -*- coding: utf-8 -*-
"""Credential-safe provider adapters for the provider-neutral embedding core.

The adapters accept injected transports, which keeps their contract testable
offline. Provider SDK exceptions are reduced to closed error categories; raw
messages, response bodies, URLs, headers, and credentials never cross the
adapter boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import re
from urllib.parse import urlsplit, urlunsplit

from embedding_backend import (
    EmbeddingBackendError,
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingContractError,
    EmbeddingErrorCategory,
    EmbeddingIdentity,
    EmbeddingTaskType,
    EmbeddingUsage,
    canonical_json,
    validate_embedding_result,
    validate_safe_metadata,
)


_PROVIDER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_RETRYABLE_CATEGORIES = frozenset(
    {
        EmbeddingErrorCategory.RATE_LIMIT,
        EmbeddingErrorCategory.TIMEOUT,
        EmbeddingErrorCategory.UNAVAILABLE,
    }
)


def _required_provider_id(value: object) -> str:
    provider = str(value or "").strip().lower()
    if not provider or not _PROVIDER_ID_PATTERN.fullmatch(provider):
        raise EmbeddingContractError("provider must be a stable non-secret identifier")
    return provider


def _canonical_endpoint(value: str | None, *, default_endpoint: str) -> str:
    endpoint = str(value or "").strip()
    if not endpoint:
        return _required_provider_id(default_endpoint)
    parsed = urlsplit(endpoint)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise EmbeddingContractError("embedding endpoint must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise EmbeddingContractError("embedding endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise EmbeddingContractError("embedding endpoint must not contain query or fragment data")
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = parsed.port
    if port is not None and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        hostname = f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((scheme, hostname, path, "", ""))


def _reject_secret_url_values(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_secret_url_values(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_secret_url_values(item)
        return
    if not isinstance(value, str) or "://" not in value:
        return
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        return
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise EmbeddingContractError(
            "provider identity configuration must not contain secret-bearing URLs"
        )


def build_provider_identity(
    *,
    backend: str,
    provider: str,
    endpoint: str | None = None,
    default_endpoint: str,
    configuration: Mapping[str, object] | None = None,
) -> str:
    """Return a stable opaque identity for one endpoint/configuration.

    Endpoint URLs are validated and hashed, never persisted verbatim. Optional
    configuration is recursively checked for credential-shaped keys before it
    participates in the hash. API keys and request headers are intentionally
    not accepted by this function.
    """

    backend_id = _required_provider_id(backend)
    provider_id = _required_provider_id(provider)
    safe_configuration = validate_safe_metadata(
        configuration or {},
        field_name="provider_identity.configuration",
    )
    _reject_secret_url_values(safe_configuration)
    payload = {
        "schema_version": 1,
        "backend": backend_id,
        "provider": provider_id,
        "endpoint": _canonical_endpoint(endpoint, default_endpoint=default_endpoint),
        "configuration": safe_configuration,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{provider_id}@sha256:{digest}"


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _status_code(exc: BaseException) -> int | None:
    for name in ("status_code", "status", "http_status"):
        value = getattr(exc, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        numeric = getattr(value, "value", None)
        if isinstance(numeric, int) and not isinstance(numeric, bool):
            return numeric
    return None


def classify_provider_exception(exc: BaseException) -> EmbeddingErrorCategory:
    """Classify an exception without copying provider-controlled text."""

    if isinstance(exc, TimeoutError):
        return EmbeddingErrorCategory.TIMEOUT
    name = type(exc).__name__.lower()
    if "cancel" in name:
        return EmbeddingErrorCategory.CANCELLED
    if "timeout" in name or "deadline" in name:
        return EmbeddingErrorCategory.TIMEOUT
    status = _status_code(exc)
    if status == 401:
        return EmbeddingErrorCategory.AUTHENTICATION
    if status == 403:
        return EmbeddingErrorCategory.PERMISSION
    if status == 429:
        return EmbeddingErrorCategory.RATE_LIMIT
    if status in {400, 404, 409, 413, 422}:
        return EmbeddingErrorCategory.INVALID_REQUEST
    if status is not None and 500 <= status <= 599:
        return EmbeddingErrorCategory.UNAVAILABLE
    if any(marker in name for marker in ("connection", "unavailable", "serviceerror")):
        return EmbeddingErrorCategory.UNAVAILABLE
    return EmbeddingErrorCategory.PROVIDER_ERROR


def _safe_backend_error(exc: BaseException) -> EmbeddingBackendError:
    if isinstance(exc, EmbeddingBackendError):
        return exc
    category = classify_provider_exception(exc)
    return EmbeddingBackendError(category, retryable=category in _RETRYABLE_CATEGORIES)


def _usage(response: object) -> EmbeddingUsage:
    usage = _field(response, "usage") or _field(response, "usage_metadata")
    input_tokens = _field(usage, "prompt_tokens")
    if input_tokens is None:
        input_tokens = _field(usage, "input_tokens")
    if input_tokens is None:
        input_tokens = _field(usage, "prompt_token_count")
    total_tokens = _field(usage, "total_tokens")
    if total_tokens is None:
        total_tokens = _field(usage, "total_token_count")
    return EmbeddingUsage(input_tokens=input_tokens, total_tokens=total_tokens)


def _gemini_usage(response: object, embeddings: Sequence[object]) -> EmbeddingUsage:
    usage = _usage(response)
    if usage.input_tokens is not None or usage.total_tokens is not None:
        return usage
    token_counts = [_field(_field(item, "statistics"), "token_count") for item in embeddings]
    input_tokens = None
    if token_counts and all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0
        for count in token_counts
    ):
        input_tokens = sum(token_counts)
    metadata: dict[str, object] = {}
    billable_characters = _field(_field(response, "metadata"), "billable_character_count")
    if isinstance(billable_characters, int) and not isinstance(billable_characters, bool):
        metadata["billable_character_count"] = billable_characters
    truncated_count = sum(
        1 for item in embeddings if _field(_field(item, "statistics"), "truncated") is True
    )
    if truncated_count:
        metadata["truncated_input_count"] = truncated_count
    return EmbeddingUsage(
        input_tokens=input_tokens,
        total_tokens=input_tokens,
        metadata=metadata,
    )


class GeminiEmbeddingAdapter:
    """Google GenAI embedding adapter with explicit retrieval task mapping."""

    backend = "google_genai"

    def __init__(
        self,
        *,
        client: object,
        model: str,
        output_dimension: int,
        provider: str = "google_ai",
        endpoint: str | None = None,
        identity_configuration: Mapping[str, object] | None = None,
        config_factory: Callable[..., object] | None = None,
    ) -> None:
        self._client = client
        self._model = str(model or "").strip()
        self._output_dimension = output_dimension
        self._provider_identity = build_provider_identity(
            backend=self.backend,
            provider=provider,
            endpoint=endpoint,
            default_endpoint="google-ai-api",
            configuration=identity_configuration,
        )
        self._config_factory = config_factory or self._default_config_factory
        # Validate model/dimension at construction without inventing a task.
        self.identity(EmbeddingTaskType.DOCUMENT)

    @staticmethod
    def _default_config_factory(**kwargs: object) -> object:
        from google.genai import types

        timeout_ms = max(1, round(float(kwargs.pop("timeout_seconds")) * 1000))
        return types.EmbedContentConfig(
            **kwargs,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

    def identity(self, task_type: EmbeddingTaskType) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            backend=self.backend,
            provider=self._provider_identity,
            model=self._model,
            task_type=task_type,
            output_dimension=self._output_dimension,
        )

    def embed(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResult:
        expected = self.identity(request.identity.task_type)
        if request.identity != expected:
            raise EmbeddingBackendError(EmbeddingErrorCategory.INVALID_REQUEST)
        native_task = {
            EmbeddingTaskType.DOCUMENT: "RETRIEVAL_DOCUMENT",
            EmbeddingTaskType.QUERY: "RETRIEVAL_QUERY",
        }[request.identity.task_type]
        try:
            config = self._config_factory(
                task_type=native_task,
                output_dimensionality=request.identity.output_dimension,
                timeout_seconds=request.timeout_seconds,
            )
            response = self._client.models.embed_content(
                model=request.identity.model,
                contents=list(request.inputs),
                config=config,
            )
            embeddings = _field(response, "embeddings", ())
            if isinstance(embeddings, (str, bytes)) or not isinstance(embeddings, Sequence):
                raise EmbeddingContractError("provider embeddings must be a sequence")
            vectors = tuple(_field(item, "values", ()) for item in embeddings)
            result = EmbeddingBatchResult(
                identity=request.identity,
                request_fingerprint=request.fingerprint,
                vectors=vectors,
                usage=_gemini_usage(response, embeddings),
                metadata={"adapter": self.backend},
            )
            return validate_embedding_result(request, result)
        except EmbeddingBackendError:
            raise
        except EmbeddingContractError:
            raise EmbeddingBackendError(EmbeddingErrorCategory.INVALID_RESPONSE) from None
        except Exception as exc:
            raise _safe_backend_error(exc) from None


class OpenAICompatibleEmbeddingAdapter:
    """Adapter for LiteLLM or an OpenAI-compatible ``embeddings.create`` callable."""

    backend = "openai_compatible"

    def __init__(
        self,
        *,
        transport: Callable[..., object],
        model: str,
        output_dimension: int,
        provider: str,
        endpoint: str | None = None,
        identity_configuration: Mapping[str, object] | None = None,
        transport_kind: str = "litellm",
        api_key: str | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not callable(transport):
            raise EmbeddingContractError("transport must be callable")
        if transport_kind not in {"litellm", "openai_client"}:
            raise EmbeddingContractError("transport_kind must be litellm or openai_client")
        self._transport = transport
        self._transport_kind = transport_kind
        self._model = str(model or "").strip()
        self._output_dimension = output_dimension
        self._endpoint = str(endpoint or "").strip() or None
        self._api_key = api_key
        self._request_headers = dict(request_headers or {})
        self._provider_identity = build_provider_identity(
            backend=self.backend,
            provider=provider,
            endpoint=endpoint,
            default_endpoint=f"{str(provider or '').strip().lower()}-default",
            configuration=identity_configuration,
        )
        self.identity(EmbeddingTaskType.DOCUMENT)

    def identity(self, task_type: EmbeddingTaskType) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            backend=self.backend,
            provider=self._provider_identity,
            model=self._model,
            task_type=task_type,
            output_dimension=self._output_dimension,
        )

    def _transport_kwargs(self, request: EmbeddingBatchRequest) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": request.identity.model,
            "input": list(request.inputs),
            "dimensions": request.identity.output_dimension,
            "timeout": request.timeout_seconds,
        }
        if self._transport_kind == "litellm":
            if self._endpoint:
                kwargs["api_base"] = self._endpoint
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            if self._request_headers:
                kwargs["extra_headers"] = dict(self._request_headers)
        return kwargs

    def embed(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResult:
        expected = self.identity(request.identity.task_type)
        if request.identity != expected:
            raise EmbeddingBackendError(EmbeddingErrorCategory.INVALID_REQUEST)
        try:
            response = self._transport(**self._transport_kwargs(request))
            data = _field(response, "data", ())
            if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
                raise EmbeddingContractError("provider data must be a sequence")
            indexed: list[tuple[int, object]] = []
            for offset, item in enumerate(data):
                index = _field(item, "index", offset)
                if isinstance(index, bool) or not isinstance(index, int):
                    raise EmbeddingContractError("provider embedding index must be an integer")
                indexed.append((index, _field(item, "embedding", ())))
            if sorted(index for index, _vector in indexed) != list(range(len(indexed))):
                raise EmbeddingContractError("provider embedding indices are invalid")
            vectors = tuple(vector for _index, vector in sorted(indexed))
            result = EmbeddingBatchResult(
                identity=request.identity,
                request_fingerprint=request.fingerprint,
                vectors=vectors,
                usage=_usage(response),
                metadata={"adapter": self.backend},
            )
            return validate_embedding_result(request, result)
        except EmbeddingBackendError:
            raise
        except EmbeddingContractError:
            raise EmbeddingBackendError(EmbeddingErrorCategory.INVALID_RESPONSE) from None
        except Exception as exc:
            raise _safe_backend_error(exc) from None
