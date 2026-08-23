# -*- coding: utf-8 -*-
"""Provider-neutral embedding contracts (issue #341).

This module is deliberately pure: it imports no provider SDK, LiteLLM, GUI
package, or networking code.  Provider adapters translate the semantic
``document`` / ``query`` task into their native API shape and return values
through these validated batch contracts.

Persisted stores should serialize the document :class:`EmbeddingIdentity`.
At query time, :func:`check_store_query_compatibility` prevents vectors from
different embedding spaces from being compared and provides a stable rebuild
diagnostic when the identities do not match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable


EMBEDDING_IDENTITY_SCHEMA_VERSION = 1
EMBEDDING_REQUEST_SCHEMA_VERSION = 1
EMBEDDING_RESULT_SCHEMA_VERSION = 1


class EmbeddingTaskType(str, Enum):
    """Provider-independent intent for an embedding request."""

    DOCUMENT = 'document'
    QUERY = 'query'


class EmbeddingErrorCategory(str, Enum):
    """Stable adapter error categories for retry and diagnostics policy."""

    INVALID_REQUEST = 'invalid_request'
    AUTHENTICATION = 'authentication'
    PERMISSION = 'permission'
    RATE_LIMIT = 'rate_limit'
    TIMEOUT = 'timeout'
    UNAVAILABLE = 'unavailable'
    CANCELLED = 'cancelled'
    INVALID_RESPONSE = 'invalid_response'
    PROVIDER_ERROR = 'provider_error'


class EmbeddingContractError(ValueError):
    """Raised before provider I/O when a core contract is invalid."""


class EmbeddingBackendError(RuntimeError):
    """Provider adapter failure with a closed, credential-safe public string.

    Adapters classify raw SDK exceptions but must not attach provider bodies,
    URLs, request headers, or codes to this cross-layer error.  This keeps
    ``str(exc)`` safe for logs and future doctor/GUI diagnostics.
    """

    def __init__(
        self,
        category: EmbeddingErrorCategory,
        *,
        retryable: bool = False,
    ) -> None:
        if not isinstance(category, EmbeddingErrorCategory):
            raise EmbeddingContractError('category must be an EmbeddingErrorCategory')
        self.category = category
        self.retryable = bool(retryable)
        super().__init__(_SAFE_ERROR_MESSAGES[category])


_SAFE_ERROR_MESSAGES = {
    EmbeddingErrorCategory.INVALID_REQUEST: 'embedding request was rejected',
    EmbeddingErrorCategory.AUTHENTICATION: 'embedding authentication failed',
    EmbeddingErrorCategory.PERMISSION: 'embedding permission denied',
    EmbeddingErrorCategory.RATE_LIMIT: 'embedding request was rate limited',
    EmbeddingErrorCategory.TIMEOUT: 'embedding request timed out',
    EmbeddingErrorCategory.UNAVAILABLE: 'embedding service temporarily unavailable',
    EmbeddingErrorCategory.CANCELLED: 'embedding request was cancelled',
    EmbeddingErrorCategory.INVALID_RESPONSE: 'embedding provider returned an invalid response',
    EmbeddingErrorCategory.PROVIDER_ERROR: 'embedding provider request failed',
}


def canonical_json(value: object) -> str:
    """Return deterministic JSON suitable for persisted identities and hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, object]) -> str:
    encoded = canonical_json(payload).encode('utf-8')
    return 'sha256:' + hashlib.sha256(encoded).hexdigest()


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmbeddingContractError(f'{field_name} must be a non-empty string')
    return value.strip()


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EmbeddingContractError(f'{field_name} must be a positive integer')
    return value


def _non_negative_int_or_none(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EmbeddingContractError(f'{field_name} must be a non-negative integer or null')
    return value


_CREDENTIAL_KEY_MARKERS = (
    'apikey',
    'authorization',
    'authsecret',
    'accesstoken',
    'authtoken',
    'refreshtoken',
    'idtoken',
    'secret',
    'password',
    'credential',
    'bearer',
    'accesskey',
    'privatekey',
    'clientkey',
    'signingkey',
)


def _normalized_key(key: object) -> str:
    return re.sub(r'[-_ ]', '', str(key).lower())


def _credential_shaped_key(key: object) -> bool:
    normalized = _normalized_key(key)
    if normalized.endswith('token') and not normalized.endswith('tokens'):
        return True
    return any(marker in normalized for marker in _CREDENTIAL_KEY_MARKERS)


def _validate_metadata(value: object, path: str = 'metadata') -> object:
    """Validate and recursively freeze JSON metadata after credential checks."""

    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if _credential_shaped_key(key):
                raise EmbeddingContractError(
                    f'{path}.{key} is credential-shaped and must not enter embedding metadata'
                )
            normalized[key] = _validate_metadata(item, f'{path}.{key}')
        return MappingProxyType(normalized)
    if isinstance(value, (list, tuple)):
        return tuple(_validate_metadata(item, f'{path}[]') for item in value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EmbeddingContractError(f'{path} must not contain NaN or Infinity')
        return value
    raise EmbeddingContractError(f'{path} must contain only JSON-compatible values')


def _validate_metadata_mapping(value: object, path: str = 'metadata') -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EmbeddingContractError(f'{path} must be an object')
    normalized = _validate_metadata(value, path)
    if not isinstance(normalized, Mapping):  # pragma: no cover - narrowed by the input check
        raise EmbeddingContractError(f'{path} must be an object')
    return normalized


def _metadata_to_json(value: object) -> object:
    """Return a detached mutable JSON shape from recursively frozen metadata."""

    if isinstance(value, Mapping):
        return {str(key): _metadata_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_to_json(item) for item in value]
    return value


def _task_type(value: object, field_name: str = 'task_type') -> EmbeddingTaskType:
    if isinstance(value, EmbeddingTaskType):
        return value
    try:
        return EmbeddingTaskType(value)
    except (TypeError, ValueError) as exc:
        allowed = ', '.join(item.value for item in EmbeddingTaskType)
        raise EmbeddingContractError(f'{field_name} must be one of: {allowed}') from exc


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Identity of one vector space and semantic task.

    ``backend`` names the adapter implementation (for example
    ``google_genai`` or ``openai_compatible``); ``provider`` names the routed
    service/configuration.  Both are persisted so two endpoints exposing an
    identically named model are not assumed compatible.
    """

    backend: str
    provider: str
    model: str
    task_type: EmbeddingTaskType
    output_dimension: int

    def __post_init__(self) -> None:
        object.__setattr__(self, 'backend', _required_text(self.backend, 'backend'))
        object.__setattr__(self, 'provider', _required_text(self.provider, 'provider'))
        object.__setattr__(self, 'model', _required_text(self.model, 'model'))
        object.__setattr__(self, 'task_type', _task_type(self.task_type))
        object.__setattr__(
            self,
            'output_dimension',
            _positive_int(self.output_dimension, 'output_dimension'),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            'schema_version': EMBEDDING_IDENTITY_SCHEMA_VERSION,
            'backend': self.backend,
            'provider': self.provider,
            'model': self.model,
            'task_type': self.task_type.value,
            'output_dimension': self.output_dimension,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload['fingerprint'] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> 'EmbeddingIdentity':
        if not isinstance(payload, Mapping):
            raise EmbeddingContractError('embedding identity must be an object')
        version = payload.get('schema_version')
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != EMBEDDING_IDENTITY_SCHEMA_VERSION
        ):
            raise EmbeddingContractError(f'unsupported embedding identity schema_version: {version!r}')
        identity = cls(
            backend=payload.get('backend'),
            provider=payload.get('provider'),
            model=payload.get('model'),
            task_type=payload.get('task_type'),
            output_dimension=payload.get('output_dimension'),
        )
        claimed = payload.get('fingerprint')
        if not isinstance(claimed, str) or not claimed.strip():
            raise EmbeddingContractError('embedding identity fingerprint is required')
        if claimed != identity.fingerprint:
            raise EmbeddingContractError('embedding identity fingerprint does not match its fields')
        return identity


@dataclass(frozen=True)
class EmbeddingBatchRequest:
    """Validated, auditable batch request passed to an adapter."""

    identity: EmbeddingIdentity
    inputs: tuple[str, ...]
    timeout_seconds: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EmbeddingIdentity):
            raise EmbeddingContractError('identity must be an EmbeddingIdentity')
        if isinstance(self.inputs, (str, bytes)) or not isinstance(self.inputs, Sequence):
            raise EmbeddingContractError('inputs must be a non-empty sequence of strings')
        normalized_inputs = tuple(self.inputs)
        if not normalized_inputs:
            raise EmbeddingContractError('inputs must not be empty')
        for index, text in enumerate(normalized_inputs):
            _required_text(text, f'inputs[{index}]')
        timeout = self.timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise EmbeddingContractError('timeout_seconds must be a positive finite number')
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise EmbeddingContractError('timeout_seconds must be a positive finite number')
        object.__setattr__(self, 'inputs', normalized_inputs)
        object.__setattr__(self, 'timeout_seconds', timeout)
        object.__setattr__(self, 'metadata', _validate_metadata_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            'schema_version': EMBEDDING_REQUEST_SCHEMA_VERSION,
            'identity': self.identity.to_dict(),
            'inputs': list(self.inputs),
            'timeout_seconds': self.timeout_seconds,
            'metadata': _metadata_to_json(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class EmbeddingUsage:
    """Provider-neutral counts plus safe provider-specific usage metadata."""

    input_tokens: int | None = None
    total_tokens: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'input_tokens',
            _non_negative_int_or_none(self.input_tokens, 'input_tokens'),
        )
        object.__setattr__(
            self,
            'total_tokens',
            _non_negative_int_or_none(self.total_tokens, 'total_tokens'),
        )
        object.__setattr__(
            self,
            'metadata',
            _validate_metadata_mapping(self.metadata, 'usage.metadata'),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            'input_tokens': self.input_tokens,
            'total_tokens': self.total_tokens,
            'metadata': _metadata_to_json(self.metadata),
        }


def _validated_vector(vector: object, index: int, dimension: int) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise EmbeddingContractError(f'vectors[{index}] must be a numeric sequence')
    if len(vector) != dimension:
        raise EmbeddingContractError(
            f'vectors[{index}] dimension mismatch: expected {dimension}, got {len(vector)}'
        )
    values: list[float] = []
    for offset, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingContractError(f'vectors[{index}][{offset}] must be numeric')
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingContractError(f'vectors[{index}][{offset}] must be finite')
        values.append(number)
    return tuple(values)


@dataclass(frozen=True)
class EmbeddingBatchResult:
    """Validated adapter output; count is bound to its request fingerprint."""

    identity: EmbeddingIdentity
    request_fingerprint: str
    vectors: tuple[tuple[float, ...], ...]
    usage: EmbeddingUsage = field(default_factory=EmbeddingUsage)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EmbeddingIdentity):
            raise EmbeddingContractError('identity must be an EmbeddingIdentity')
        request_fp = _required_text(self.request_fingerprint, 'request_fingerprint')
        if isinstance(self.vectors, (str, bytes)) or not isinstance(self.vectors, Sequence):
            raise EmbeddingContractError('vectors must be a sequence')
        vectors = tuple(
            _validated_vector(vector, index, self.identity.output_dimension)
            for index, vector in enumerate(self.vectors)
        )
        if not isinstance(self.usage, EmbeddingUsage):
            raise EmbeddingContractError('usage must be EmbeddingUsage')
        object.__setattr__(self, 'request_fingerprint', request_fp)
        object.__setattr__(self, 'vectors', vectors)
        object.__setattr__(self, 'metadata', _validate_metadata_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            'schema_version': EMBEDDING_RESULT_SCHEMA_VERSION,
            'identity': self.identity.to_dict(),
            'request_fingerprint': self.request_fingerprint,
            'vectors': [list(vector) for vector in self.vectors],
            'usage': self.usage.to_dict(),
            'metadata': _metadata_to_json(self.metadata),
        }


def validate_embedding_result(
    request: EmbeddingBatchRequest,
    result: EmbeddingBatchResult,
) -> EmbeddingBatchResult:
    """Reject response identity, request binding, count, or dimension drift."""

    if not isinstance(request, EmbeddingBatchRequest):
        raise EmbeddingContractError('request must be an EmbeddingBatchRequest')
    if not isinstance(result, EmbeddingBatchResult):
        raise EmbeddingContractError('result must be an EmbeddingBatchResult')
    if result.identity != request.identity:
        raise EmbeddingContractError('result identity does not match request identity')
    if result.request_fingerprint != request.fingerprint:
        raise EmbeddingContractError('result request_fingerprint does not match request')
    if len(result.vectors) != len(request.inputs):
        raise EmbeddingContractError(
            f'embedding count mismatch: expected {len(request.inputs)}, got {len(result.vectors)}'
        )
    return result


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Minimal protocol implemented by Gemini/OpenAI-compatible adapters."""

    def embed(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResult:
        """Embed one validated batch or raise :class:`EmbeddingBackendError`."""


class CompatibilityCode(str, Enum):
    COMPATIBLE = 'compatible'
    STORE_TASK_NOT_DOCUMENT = 'store_task_not_document'
    QUERY_TASK_NOT_QUERY = 'query_task_not_query'
    BACKEND_MISMATCH = 'backend_mismatch'
    PROVIDER_MISMATCH = 'provider_mismatch'
    MODEL_MISMATCH = 'model_mismatch'
    DIMENSION_MISMATCH = 'dimension_mismatch'


@dataclass(frozen=True)
class CompatibilityMismatch:
    code: CompatibilityCode
    field: str
    store_value: object
    query_value: object

    def to_dict(self) -> dict[str, object]:
        return {
            'code': self.code.value,
            'field': self.field,
            'store_value': self.store_value,
            'query_value': self.query_value,
        }


@dataclass(frozen=True)
class EmbeddingCompatibilityReport:
    """Machine-readable compatibility decision and deterministic remediation."""

    compatible: bool
    mismatches: tuple[CompatibilityMismatch, ...]
    action: str
    message: str

    @property
    def codes(self) -> tuple[str, ...]:
        if self.compatible:
            return (CompatibilityCode.COMPATIBLE.value,)
        return tuple(item.code.value for item in self.mismatches)

    def to_dict(self) -> dict[str, object]:
        return {
            'compatible': self.compatible,
            'codes': list(self.codes),
            'mismatches': [item.to_dict() for item in self.mismatches],
            'action': self.action,
            'message': self.message,
        }


def check_store_query_compatibility(
    store_identity: EmbeddingIdentity,
    query_identity: EmbeddingIdentity,
) -> EmbeddingCompatibilityReport:
    """Check whether query vectors may be compared with persisted documents."""

    if not isinstance(store_identity, EmbeddingIdentity):
        raise EmbeddingContractError('store_identity must be an EmbeddingIdentity')
    if not isinstance(query_identity, EmbeddingIdentity):
        raise EmbeddingContractError('query_identity must be an EmbeddingIdentity')

    mismatches: list[CompatibilityMismatch] = []

    def add(code: CompatibilityCode, field_name: str, store: object, query: object) -> None:
        mismatches.append(CompatibilityMismatch(code, field_name, store, query))

    if store_identity.task_type is not EmbeddingTaskType.DOCUMENT:
        add(
            CompatibilityCode.STORE_TASK_NOT_DOCUMENT,
            'task_type',
            store_identity.task_type.value,
            query_identity.task_type.value,
        )
    if query_identity.task_type is not EmbeddingTaskType.QUERY:
        add(
            CompatibilityCode.QUERY_TASK_NOT_QUERY,
            'task_type',
            store_identity.task_type.value,
            query_identity.task_type.value,
        )
    comparisons = (
        (CompatibilityCode.BACKEND_MISMATCH, 'backend'),
        (CompatibilityCode.PROVIDER_MISMATCH, 'provider'),
        (CompatibilityCode.MODEL_MISMATCH, 'model'),
        (CompatibilityCode.DIMENSION_MISMATCH, 'output_dimension'),
    )
    for code, field_name in comparisons:
        store_value = getattr(store_identity, field_name)
        query_value = getattr(query_identity, field_name)
        if store_value != query_value:
            add(code, field_name, store_value, query_value)

    if not mismatches:
        return EmbeddingCompatibilityReport(
            compatible=True,
            mismatches=(),
            action='none',
            message='Embedding query identity is compatible with the document store.',
        )
    fields = ', '.join(item.field for item in mismatches)
    return EmbeddingCompatibilityReport(
        compatible=False,
        mismatches=tuple(mismatches),
        action='rebuild_store',
        message=(
            'Embedding identity mismatch; do not compare these vectors. '
            f'Rebuild the store with the selected query backend configuration ({fields}).'
        ),
    )
