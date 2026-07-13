"""Minimal backend boundary for synchronous model generation.

Gemini Batch intentionally does not use this module. It remains the default
translation path; this boundary is for explicitly selected synchronous calls.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, runtime_checkable

SYNC_EXECUTION_MODE = "sync"

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
        response = self._client.models.generate_content(
            model=request.model, contents=request.contents, config=dict(request.config))
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
