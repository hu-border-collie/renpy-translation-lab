# -*- coding: utf-8 -*-
"""Production wiring for provider-neutral embedding adapters (issue #341).

This module selects Gemini or OpenAI-compatible/LiteLLM adapters from explicit
RAG configuration, embeds batches through the frozen core contract, and gates
store writes/queries on document identity. It does not guess an embedding model
from a generation model, and it never places credentials in identity, metadata,
or public diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from typing import Any

from embedding_adapters import (
    GeminiEmbeddingAdapter,
    OpenAICompatibleEmbeddingAdapter,
    build_provider_identity,
)
from embedding_backend import (
    EmbeddingBackendError,
    EmbeddingBatchRequest,
    EmbeddingContractError,
    EmbeddingErrorCategory,
    EmbeddingIdentity,
    EmbeddingTaskType,
    check_persisted_store_query_compatibility,
)
from rag_memory import EmbeddingStoreIdentityError


BACKEND_GEMINI = 'gemini'
BACKEND_OPENAI_COMPATIBLE = 'openai_compatible'
ADAPTER_BACKEND_GEMINI = 'google_genai'
ADAPTER_BACKEND_OPENAI = 'openai_compatible'
DEFAULT_GEMINI_PROVIDER = 'google_ai'
DEFAULT_GEMINI_ENDPOINT = 'google-ai-api'
DEFAULT_GEMINI_MODEL = 'gemini-embedding-001'
DEFAULT_OUTPUT_DIMENSION = 768
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_QUERY_TASK_TYPE = 'RETRIEVAL_QUERY'
DEFAULT_DOCUMENT_TASK_TYPE = 'RETRIEVAL_DOCUMENT'

_BACKEND_ALIASES = {
    'gemini': BACKEND_GEMINI,
    'google_genai': BACKEND_GEMINI,
    'google-ai': BACKEND_GEMINI,
    'google_ai': BACKEND_GEMINI,
    'openai_compatible': BACKEND_OPENAI_COMPATIBLE,
    'openai-compatible': BACKEND_OPENAI_COMPATIBLE,
    'litellm': BACKEND_OPENAI_COMPATIBLE,
}

_NATIVE_TASK_TYPES = {
    'RETRIEVAL_DOCUMENT': EmbeddingTaskType.DOCUMENT,
    'RETRIEVAL_QUERY': EmbeddingTaskType.QUERY,
    EmbeddingTaskType.DOCUMENT.value: EmbeddingTaskType.DOCUMENT,
    EmbeddingTaskType.QUERY.value: EmbeddingTaskType.QUERY,
    'DOCUMENT': EmbeddingTaskType.DOCUMENT,
    'QUERY': EmbeddingTaskType.QUERY,
}
_UNSUPPORTED_GEMINI_TASK_TYPES = frozenset(
    {
        'SEMANTIC_SIMILARITY',
        'CLASSIFICATION',
        'CLUSTERING',
        'QUESTION_ANSWERING',
        'FACT_VERIFICATION',
        'CODE_RETRIEVAL_QUERY',
    }
)
_GEMINI_BACKEND_ALIASES = frozenset(
    {
        BACKEND_GEMINI,
        'google_genai',
        'google-ai',
        'google_ai',
        'google-genai',
    }
)


class EmbeddingRuntimeError(RuntimeError):
    """Configuration or store-identity failure before provider I/O."""

    def __init__(self, message: str, *, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(message)


@dataclass(frozen=True)
class EmbeddingRuntimeSettings:
    """Explicit, credential-free embedding backend selection."""

    backend: str
    provider: str
    model: str
    output_dimension: int
    endpoint: str
    timeout_seconds: float
    api_key_env: str = ''
    native_query_task_type: str = DEFAULT_QUERY_TASK_TYPE
    native_document_task_type: str = DEFAULT_DOCUMENT_TASK_TYPE

    @property
    def adapter_backend(self) -> str:
        if self.backend == BACKEND_GEMINI:
            return ADAPTER_BACKEND_GEMINI
        return ADAPTER_BACKEND_OPENAI

    @property
    def default_endpoint(self) -> str:
        if self.backend == BACKEND_GEMINI:
            return DEFAULT_GEMINI_ENDPOINT
        return f'{self.provider}-default'

    def provider_identity(self) -> str:
        return build_provider_identity(
            backend=self.adapter_backend,
            provider=self.provider,
            endpoint=self.endpoint or None,
            default_endpoint=self.default_endpoint,
        )

    def identity(self, task_type: EmbeddingTaskType) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            backend=self.adapter_backend,
            provider=self.provider_identity(),
            model=self.model,
            task_type=task_type,
            output_dimension=self.output_dimension,
        )

    def document_identity(self) -> EmbeddingIdentity:
        return self.identity(EmbeddingTaskType.DOCUMENT)

    def query_identity(self) -> EmbeddingIdentity:
        return self.identity(EmbeddingTaskType.QUERY)

    def public_dict(self) -> dict[str, object]:
        """Credential-safe identity for doctor, GUI, and manifests."""

        identity = self.document_identity()
        return {
            'backend': self.backend,
            'adapter_backend': self.adapter_backend,
            'provider': identity.provider,
            'model': self.model,
            'query_task_type': self.native_query_task_type,
            'document_task_type': self.native_document_task_type,
            'output_dimension': self.output_dimension,
            'timeout_seconds': self.timeout_seconds,
            'fingerprint': identity.fingerprint,
        }


def semantic_task_type(value: object, default: EmbeddingTaskType | None = None) -> EmbeddingTaskType:
    """Map persisted Gemini-native or semantic task names to the core enum."""

    if isinstance(value, EmbeddingTaskType):
        return value
    if value is None or (isinstance(value, str) and not str(value).strip()):
        if default is not None:
            return default
        raise EmbeddingContractError('embedding task type is required')
    text = str(value).strip()
    mapped = _NATIVE_TASK_TYPES.get(text) or _NATIVE_TASK_TYPES.get(text.upper().replace('-', '_'))
    if mapped is not None:
        return mapped
    unsupported = text.upper().replace('-', '_')
    if unsupported in _UNSUPPORTED_GEMINI_TASK_TYPES:
        raise EmbeddingContractError(
            f'embedding task type {text!r} is not supported; use RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY'
        )
    raise EmbeddingContractError(
        'embedding task type must be document, query, RETRIEVAL_DOCUMENT, or RETRIEVAL_QUERY'
    )


def persist_task_type(value: object, default: str) -> str:
    """Return the canonical persisted Gemini-native task name."""

    semantic = semantic_task_type(value, default=semantic_task_type(default))
    if semantic is EmbeddingTaskType.DOCUMENT:
        return DEFAULT_DOCUMENT_TASK_TYPE
    return DEFAULT_QUERY_TASK_TYPE


def is_explicit_non_gemini_backend(rag_config: Mapping[str, Any] | None) -> bool:
    """True when the user selected a non-Gemini embedding backend."""

    rag = rag_config if isinstance(rag_config, Mapping) else {}
    requested = _optional_text(rag.get('embedding_backend')).lower().replace('-', '_')
    if not requested:
        return False
    return requested not in _GEMINI_BACKEND_ALIASES


def _optional_text(value: object) -> str:
    if value is None:
        return ''
    return str(value).strip()


def _positive_int(value: object, field_name: str, default: int) -> int:
    if value is None or value == '':
        return default
    if isinstance(value, bool):
        raise EmbeddingContractError(f'{field_name} must be a positive integer')
    if isinstance(value, int) and value > 0:
        return value
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise EmbeddingContractError(f'{field_name} must be a positive integer') from exc
    if number <= 0:
        raise EmbeddingContractError(f'{field_name} must be a positive integer')
    return number


def _positive_float(value: object, field_name: str, default: float) -> float:
    if value is None or value == '':
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EmbeddingContractError(f'{field_name} must be a positive finite number') from exc
    if number != number or number in (float('inf'), float('-inf')) or number <= 0:
        raise EmbeddingContractError(f'{field_name} must be a positive finite number')
    return number


def _normalize_backend(value: object) -> str:
    raw = _optional_text(value).lower().replace('-', '_')
    if not raw:
        return BACKEND_GEMINI
    normalized = _BACKEND_ALIASES.get(raw) or _BACKEND_ALIASES.get(raw.replace('_', '-'))
    if normalized is None:
        raise EmbeddingContractError(
            'embedding_backend must be gemini or openai_compatible; '
            'do not infer it from a generation model'
        )
    return normalized


def parse_embedding_runtime_settings(
    rag_config: Mapping[str, Any] | None,
    *,
    default_model: str = DEFAULT_GEMINI_MODEL,
) -> EmbeddingRuntimeSettings:
    """Parse explicit embedding backend settings from a rag config object.

    Generation-model fields are ignored. An OpenAI-compatible backend requires an
    explicit embedding model and provider; Gemini keeps the historical default
    model for backward compatibility.
    """

    rag = rag_config if isinstance(rag_config, Mapping) else {}
    backend = _normalize_backend(rag.get('embedding_backend'))
    model = _optional_text(rag.get('embedding_model'))
    if not model:
        if backend != BACKEND_GEMINI:
            raise EmbeddingContractError(
                'embedding_model must be set explicitly for a non-Gemini embedding backend'
            )
        model = default_model
    provider = _optional_text(rag.get('embedding_provider')).lower()
    if not provider:
        if backend == BACKEND_GEMINI:
            provider = DEFAULT_GEMINI_PROVIDER
        else:
            raise EmbeddingContractError(
                'embedding_provider must be set explicitly for an OpenAI-compatible embedding backend'
            )
    endpoint = _optional_text(rag.get('embedding_endpoint'))
    api_key_env = _optional_text(rag.get('embedding_api_key_env'))
    if api_key_env and backend == BACKEND_GEMINI:
        raise EmbeddingContractError(
            'embedding_api_key_env is only valid for an OpenAI-compatible embedding backend'
        )
    return EmbeddingRuntimeSettings(
        backend=backend,
        provider=provider,
        model=model,
        output_dimension=_positive_int(
            rag.get('output_dimensionality'),
            'output_dimensionality',
            DEFAULT_OUTPUT_DIMENSION,
        ),
        endpoint=endpoint,
        timeout_seconds=_positive_float(
            rag.get('embedding_timeout_seconds'),
            'embedding_timeout_seconds',
            DEFAULT_TIMEOUT_SECONDS,
        ),
        api_key_env=api_key_env,
        native_query_task_type=persist_task_type(
            rag.get('query_task_type'),
            DEFAULT_QUERY_TASK_TYPE,
        ),
        native_document_task_type=persist_task_type(
            rag.get('document_task_type'),
            DEFAULT_DOCUMENT_TASK_TYPE,
        ),
    )


def build_embedding_adapter(
    settings: EmbeddingRuntimeSettings,
    *,
    gemini_client: object | None = None,
    openai_transport: Callable[..., object] | None = None,
    api_key: str | None = None,
    request_headers: Mapping[str, str] | None = None,
    transport_kind: str = 'litellm',
    gemini_config_factory: Callable[..., object] | None = None,
) -> GeminiEmbeddingAdapter | OpenAICompatibleEmbeddingAdapter:
    """Construct an adapter from explicit settings and an injected transport."""

    if not isinstance(settings, EmbeddingRuntimeSettings):
        raise EmbeddingContractError('settings must be EmbeddingRuntimeSettings')
    if settings.backend == BACKEND_GEMINI:
        if gemini_client is None:
            raise EmbeddingRuntimeError(
                'Gemini embedding adapter requires an injected client',
                reason='missing_gemini_client',
            )
        return GeminiEmbeddingAdapter(
            client=gemini_client,
            model=settings.model,
            output_dimension=settings.output_dimension,
            provider=settings.provider,
            endpoint=settings.endpoint or None,
            config_factory=gemini_config_factory,
        )
    if openai_transport is None:
        raise EmbeddingRuntimeError(
            'OpenAI-compatible embedding adapter requires an injected transport',
            reason='missing_openai_transport',
        )
    return OpenAICompatibleEmbeddingAdapter(
        transport=openai_transport,
        model=settings.model,
        output_dimension=settings.output_dimension,
        provider=settings.provider,
        endpoint=settings.endpoint or None,
        transport_kind=transport_kind,
        api_key=api_key,
        request_headers=request_headers,
    )


def embed_texts(
    adapter: GeminiEmbeddingAdapter | OpenAICompatibleEmbeddingAdapter,
    contents: Sequence[str],
    task_type: object,
    *,
    timeout_seconds: float | None = None,
) -> list[list[float]]:
    """Embed one batch through the adapter contract and return raw vectors."""

    if not contents:
        return []
    texts: list[str] = []
    for index, item in enumerate(contents):
        text = '' if item is None else str(item)
        if not text:
            raise EmbeddingContractError(
                f'embedding inputs[{index}] must be a non-empty string'
            )
        texts.append(text)
    semantic = semantic_task_type(task_type)
    identity = adapter.identity(semantic)
    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else DEFAULT_TIMEOUT_SECONDS
    )
    request = EmbeddingBatchRequest(
        identity=identity,
        inputs=tuple(texts),
        timeout_seconds=timeout,
    )
    result = adapter.embed(request)
    return [list(vector) for vector in result.vectors]


def public_error_diagnostics(exc: BaseException) -> dict[str, object]:
    """Return a credential-safe retrieval failure payload."""

    if isinstance(exc, EmbeddingBackendError):
        return {
            'failure_reason': 'retrieval_error',
            'error_category': exc.category.value,
            'retryable': bool(exc.retryable),
        }
    if isinstance(exc, EmbeddingContractError):
        return {
            'failure_reason': 'invalid_embedding_request',
            'error_category': EmbeddingErrorCategory.INVALID_REQUEST.value,
            'retryable': False,
        }
    if isinstance(exc, EmbeddingRuntimeError):
        return {
            'failure_reason': exc.reason,
            'error_category': EmbeddingErrorCategory.INVALID_REQUEST.value,
            'retryable': False,
        }
    if isinstance(exc, EmbeddingStoreIdentityError):
        return {
            'failure_reason': 'rebuild_store',
            'error_category': EmbeddingErrorCategory.INVALID_REQUEST.value,
            'retryable': False,
            'action': 'rebuild_store',
        }
    return {
        'failure_reason': 'retrieval_error',
        'error_category': EmbeddingErrorCategory.PROVIDER_ERROR.value,
        'retryable': False,
    }


def store_compatibility_report(store: object, query_identity: EmbeddingIdentity):
    """Return the store's compatibility report, fail-closed if metadata is missing."""

    if hasattr(store, 'embedding_compatibility'):
        return store.embedding_compatibility(query_identity)
    metadata = getattr(store, 'metadata', None)
    payload = metadata.get('embedding_identity') if isinstance(metadata, Mapping) else None
    return check_persisted_store_query_compatibility(payload, query_identity)


def ensure_store_document_identity(
    store: object,
    identity: EmbeddingIdentity,
    *,
    rebuild: bool = False,
) -> dict[str, object]:
    """Write document identity, or explicitly rebuild an incompatible store.

    Query paths must pass ``rebuild=False``. Bootstrap may pass ``rebuild=True``
    after the operator has requested a store rebuild.
    """

    if not hasattr(store, 'set_embedding_identity'):
        raise EmbeddingRuntimeError(
            'store does not support embedding identity',
            reason='unsupported_store',
        )
    query_identity = EmbeddingIdentity(
        backend=identity.backend,
        provider=identity.provider,
        model=identity.model,
        task_type=EmbeddingTaskType.QUERY,
        output_dimension=identity.output_dimension,
    )
    report = store_compatibility_report(store, query_identity)
    payload = {
        'compatible': bool(report.compatible),
        'action': report.action,
        'codes': list(report.codes),
        'message': report.message,
        'rebuilt': False,
        'ready': False,
    }
    try:
        store.set_embedding_identity(identity)
    except EmbeddingStoreIdentityError:
        if not rebuild:
            payload['action'] = 'rebuild_store'
            return payload
        store.rebuild_document_identity(identity)
        payload['rebuilt'] = True
        payload['ready'] = True
        payload['compatible'] = True
        payload['action'] = 'rebuilt_store'
        payload['codes'] = ['compatible']
        payload['message'] = 'Embedding store was rebuilt with the selected document identity.'
        return payload
    payload['ready'] = True
    payload['compatible'] = True
    payload['action'] = 'none'
    payload['codes'] = ['compatible']
    payload['message'] = report.message if report.compatible else (
        'Embedding document identity was written to an empty store.'
    )
    return payload


def resolve_api_key_from_env(env_name: str) -> str:
    """Read an API key from a named environment variable. Never logs the value."""

    name = str(env_name or '').strip()
    if not name:
        return ''
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) else ''
