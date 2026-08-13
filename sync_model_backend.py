"""Minimal backend boundary for synchronous model generation.

Gemini Batch intentionally does not use this module. It remains the default
translation path; this boundary is for explicitly selected synchronous calls.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, runtime_checkable

from gemini_model_catalog import filter_gemini_generation_config

SYNC_EXECUTION_MODE = "sync"
DEFAULT_SYNC_TIMEOUT_SECONDS = 120
MIN_SYNC_TIMEOUT_SECONDS = 5
MAX_SYNC_TIMEOUT_SECONDS = 600
DEFAULT_SYNC_RETRY_ATTEMPTS = 3
SYNC_ERROR_CATEGORIES = frozenset({
    "authentication",
    "rate_limit",
    "service_unavailable",
    "timeout",
    "invalid_response",
    "unsupported_capability",
    "missing_dependency",
    "provider_error",
})
_SYNC_SAFE_ERROR_MESSAGES = {
    "authentication": "authentication failed",
    "rate_limit": "rate limited or quota exhausted",
    "service_unavailable": "service temporarily unavailable",
    "timeout": "request timed out",
    "invalid_response": "provider returned an invalid response",
    "unsupported_capability": "provider does not support this capability",
    "missing_dependency": "optional provider dependency is unavailable",
    "provider_error": "provider request failed",
}


class SyncBackendError(RuntimeError):
    """Provider-neutral failure that never stores the raw provider message."""

    def __init__(
        self,
        category: str = "provider_error",
        *,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized = str(category or "provider_error").strip().lower()
        if normalized not in SYNC_ERROR_CATEGORIES:
            normalized = "provider_error"
        self.category = normalized
        self.request_metadata = dict(request_metadata or {})
        super().__init__(_SYNC_SAFE_ERROR_MESSAGES[normalized])


@dataclass(frozen=True)
class SyncRecoveryDecision:
    """Provider-neutral recovery action derived from a structured category."""

    category: str
    retry_same_request: bool = False
    split_request: bool = False
    backoff: bool = False
    rotate_credentials: bool = False


def sync_error_category(exc: BaseException) -> str:
    """Classify a synchronous request failure without overriding provider data.

    ``LiteLLMBackendError.category`` (and any future backend category) is the
    authoritative source. Status/type/text fallbacks exist for SDK exceptions
    that do not expose the shared category contract yet.
    """
    explicit = str(getattr(exc, "category", "") or "").strip().lower()
    if explicit in SYNC_ERROR_CATEGORIES:
        return explicit

    reason_code = str(getattr(exc, "reason_code", "") or "").strip().lower()
    if reason_code and (
        reason_code in {
            "empty_response_text",
            "invalid_json",
            "truncated_output",
            "reasoning_budget_exhausted",
            "reasoning_without_text_output",
        }
        or reason_code.startswith(("response_", "result_"))
    ):
        return "invalid_response"

    status = getattr(exc, "status_code", getattr(exc, "code", None))
    try:
        status = int(status)
    except (TypeError, ValueError, OverflowError):
        status = None
    if status in {401, 403}:
        return "authentication"
    if status == 408:
        return "timeout"
    if status == 429:
        return "rate_limit"
    if status in {500, 502, 503, 504}:
        return "service_unavailable"
    if status == 404:
        return "unsupported_capability"
    if isinstance(exc, TimeoutError):
        return "timeout"

    type_name = type(exc).__name__.lower()
    if any(marker in type_name for marker in ("authentication", "permissiondenied")):
        return "authentication"
    if any(marker in type_name for marker in ("ratelimit", "resourceexhausted")):
        return "rate_limit"
    if "timeout" in type_name:
        return "timeout"
    if any(
        marker in type_name
        for marker in ("serviceunavailable", "apiconnection", "connecterror", "readerror")
    ):
        return "service_unavailable"
    if "notfound" in type_name:
        return "unsupported_capability"

    # google-genai does not consistently expose typed categories/status across
    # versions. Keep this fallback narrow and use it only after structured data.
    text = str(exc).upper()
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return "rate_limit"
    if any(marker in text for marker in ("CONNECTTIMEOUT", "READTIMEOUT", "TIMED OUT")):
        return "timeout"
    if any(
        marker in text
        for marker in (
            "UNAVAILABLE",
            "UNEXPECTED_EOF_WHILE_READING",
            "REMOTEPROTOCOLERROR",
            " 502",
            " 503",
            " 504",
        )
    ):
        return "service_unavailable"
    if "UNAUTHENTICATED" in text or "PERMISSION_DENIED" in text:
        return "authentication"
    return "provider_error"


def sync_recovery_decision(exc: BaseException) -> SyncRecoveryDecision:
    """Return the only retry/split actions allowed for *exc*."""
    category = sync_error_category(exc)
    if category == "rate_limit":
        return SyncRecoveryDecision(
            category,
            retry_same_request=True,
            backoff=True,
            rotate_credentials=True,
        )
    if category in {"service_unavailable", "timeout"}:
        return SyncRecoveryDecision(
            category,
            retry_same_request=True,
            backoff=True,
        )
    if category == "invalid_response":
        return SyncRecoveryDecision(category, split_request=True)
    return SyncRecoveryDecision(category)


def sync_error_summary(exc: BaseException) -> str:
    """Return a log-safe summary that never includes provider exception text."""
    category = sync_error_category(exc)
    message = _SYNC_SAFE_ERROR_MESSAGES.get(
        category,
        _SYNC_SAFE_ERROR_MESSAGES["provider_error"],
    )
    return f"{message} [{category}]"


def normalize_sync_timeout_seconds(
    value: Any,
    default: int = DEFAULT_SYNC_TIMEOUT_SECONDS,
) -> int:
    """Return a finite per-request timeout constrained to safe sync bounds."""
    try:
        if isinstance(value, bool):
            raise TypeError("boolean is not a timeout")
        timeout = int(value)
    except (TypeError, ValueError, OverflowError):
        timeout = int(default)
    return max(MIN_SYNC_TIMEOUT_SECONDS, min(MAX_SYNC_TIMEOUT_SECONDS, timeout))


@dataclass(frozen=True)
class SyncGenerationRequest:
    model: str
    contents: Any
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyncGenerationResult:
    provider: str
    model: str
    execution_mode: str
    response_payload: Any
    response_text: str = ""
    parsed: Any = None
    finish_reason: str = ""
    usage_metadata: Mapping[str, Any] = field(default_factory=dict)
    output_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    request_metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SyncModelBackend(Protocol):
    provider: str

    def generate(self, request: SyncGenerationRequest) -> SyncGenerationResult: ...


class GeminiSyncBackend:
    """Adapter for the existing google-genai synchronous client."""
    provider = "gemini"

    def __init__(self, client: Any, *, serialize_response: Callable[[Any], Any],
                 extract_text: Callable[[Any], str],
                 extract_finish_reason: Callable[[Any], str],
                 extract_usage: Optional[Callable[[Any], Mapping[str, Any]]] = None) -> None:
        self._client = client
        self._serialize_response = serialize_response
        self._extract_text = extract_text
        self._extract_finish_reason = extract_finish_reason
        self._extract_usage = extract_usage

    def generate(self, request: SyncGenerationRequest) -> SyncGenerationResult:
        config = filter_gemini_generation_config(request.model, request.config)
        # Internal provider-neutral hint; Gemini consumes response_json_schema
        # directly and must not receive this LiteLLM adapter option.
        config.pop("structured_output_mode", None)
        config.pop("response_schema_name", None)
        timeout = config.pop("timeout", DEFAULT_SYNC_TIMEOUT_SECONDS)
        raw_http_options = config.get("http_options")
        http_options = (
            dict(raw_http_options)
            if isinstance(raw_http_options, Mapping)
            else {}
        )
        # google-genai HttpOptions uses milliseconds; the public sync request
        # contract and LiteLLM both use seconds. Apply the default here as a
        # final boundary even if a future caller forgets to set it.
        http_options["timeout"] = normalize_sync_timeout_seconds(timeout) * 1000
        config["http_options"] = http_options
        try:
            response = self._client.models.generate_content(
                model=request.model,
                contents=request.contents,
                config=config,
            )
        except Exception as exc:
            raise SyncBackendError(
                sync_error_category(exc),
                request_metadata={"provider": self.provider},
            ) from None
        try:
            payload = self._serialize_response(response)
            usage: Dict[str, Any] = {}
            if self._extract_usage is not None:
                usage = dict(self._extract_usage(payload) or {})
            response_text = self._extract_text(payload) or ""
            finish_reason = self._extract_finish_reason(payload) or ""
            parsed = getattr(response, "parsed", None)
        except Exception:
            raise SyncBackendError(
                "invalid_response",
                request_metadata={"provider": self.provider},
            ) from None
        return SyncGenerationResult(
            provider=self.provider, model=request.model,
            execution_mode=SYNC_EXECUTION_MODE, response_payload=payload,
            response_text=response_text,
            parsed=parsed,
            finish_reason=finish_reason,
            usage_metadata=usage,
            request_metadata={"provider": self.provider})
