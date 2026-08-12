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
        response = self._client.models.generate_content(
            model=request.model,
            contents=request.contents,
            config=config,
        )
        payload = self._serialize_response(response)
        usage: Dict[str, Any] = {}
        if self._extract_usage is not None:
            usage = dict(self._extract_usage(payload) or {})
        return SyncGenerationResult(
            provider=self.provider, model=request.model,
            execution_mode=SYNC_EXECUTION_MODE, response_payload=payload,
            response_text=self._extract_text(payload) or "",
            parsed=getattr(response, "parsed", None),
            finish_reason=self._extract_finish_reason(payload) or "",
            usage_metadata=usage)
