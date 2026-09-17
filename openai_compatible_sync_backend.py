"""Direct OpenAI Chat Completions-compatible sync backend (issue #431 S1).

This backend speaks the OpenAI Chat Completions wire format over the standard
library HTTP client.  It deliberately does not import LiteLLM, and it never
stores provider response text in errors: callers receive the shared
``SyncBackendError`` category contract instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from sync_model_backend import (
    SYNC_EXECUTION_MODE,
    SyncBackendError,
    SyncGenerationRequest,
    SyncGenerationResult,
    normalize_sync_timeout_seconds,
)

DEFAULT_STRUCTURED_OUTPUT_MODE = "prompt_only_json"
STRUCTURED_OUTPUT_MODES = frozenset({
    "strict_json_schema",
    "json_object",
    "prompt_only_json",
})

_GENERATION_PARAM_KEYS = frozenset({
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
})

# Gemini-only / adapter-internal options that must never leak into a Chat
# Completions body.  ``safety_settings`` fails closed because dropping it would
# silently change user intent; ``thinking_config`` has no provider-neutral
# meaning and is reported as an ignored option.
_GEMINI_ONLY_KEYS = frozenset({
    "response_mime_type",
    "response_schema_name",
    "http_options",
})
_IGNORED_OPTION_KEYS = frozenset({
    "thinking_config",
})
_UNSUPPORTED_OPTION_KEYS = frozenset({
    "safety_settings",
})

_STRUCTURED_OUTPUT_MARKERS = (
    "response_format",
    "json_schema",
    "json schema",
    "structured output",
    "response schema",
)


@dataclass(frozen=True)
class HTTPResponse:
    """Transport-neutral response used by the default and injected transports."""

    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[urllib.request.Request, float], HTTPResponse]


def _default_transport(
    request: urllib.request.Request,
    timeout: float,
) -> HTTPResponse:
    """Send one HTTP request with ``urllib`` and normalize non-2xx responses."""

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HTTPResponse(
                status=int(getattr(response, "status", 200) or 200),
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        return HTTPResponse(
            status=int(getattr(exc, "code", 0) or 0),
            headers=dict(exc.headers.items()) if exc.headers else {},
            body=body,
        )


def _instruction_text(value: Any) -> str:
    """Flatten the shared instruction shape (string or Gemini parts) to text."""

    if not isinstance(value, Mapping):
        return str(value or "")
    parts = value.get("parts") or []
    return "\n".join(
        str(part.get("text") or "")
        for part in parts
        if isinstance(part, Mapping)
    )


def _build_messages(contents: Any, config: Mapping[str, Any]) -> list[dict[str, str]]:
    """Map the provider-neutral sync request into Chat Completions messages."""

    messages: list[dict[str, str]] = []
    if config.get("system_instruction"):
        messages.append({
            "role": "system",
            "content": _instruction_text(config["system_instruction"]),
        })
    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
        return messages
    if not isinstance(contents, list):
        raise SyncBackendError(
            "unsupported_capability",
            request_metadata={"reason": "contents_not_text"},
        )
    for entry in contents:
        if not isinstance(entry, Mapping):
            raise SyncBackendError(
                "unsupported_capability",
                request_metadata={"reason": "message_not_object"},
            )
        messages.append({
            "role": str(entry.get("role") or "user"),
            "content": (
                str(entry.get("content") or "")
                if "content" in entry
                else _instruction_text(entry)
            ),
        })
    return messages


def _schema_name(schema: Mapping[str, Any]) -> str:
    """Derive a stable schema name from the shared response envelope."""

    properties = schema.get("properties") if isinstance(schema, Mapping) else {}
    envelope_key = next(
        (
            key
            for key in ("translations", "revisions", "candidates")
            if isinstance(properties, Mapping) and key in properties
        ),
        "model",
    )
    return f"{envelope_key}_response"


def _text_from_content(content: Any) -> str:
    """Extract final text from a Chat Completions message content value."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _reasoning_text(message: Mapping[str, Any], choice: Mapping[str, Any]) -> str:
    """Return provider reasoning text without mixing it into final output."""

    for source in (message, choice):
        for key in ("reasoning_content", "reasoning"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _classify_http_error(
    status: int,
    body_text: str,
    *,
    structured_output_requested: bool,
) -> tuple[str, str]:
    """Map an HTTP failure to a shared category and stable reason code."""

    if status in {401, 403}:
        return "authentication", "http_authentication"
    if status == 429:
        return "rate_limit", "http_rate_limit"
    if status == 408:
        return "timeout", "http_timeout"
    if status in {500, 502, 503, 504}:
        return "service_unavailable", "http_service_unavailable"
    lowered = str(body_text or "").casefold()
    if structured_output_requested and any(
        marker in lowered for marker in _STRUCTURED_OUTPUT_MARKERS
    ):
        return "unsupported_capability", "structured_output_unsupported"
    if status in {400, 404, 422} and any(
        marker in lowered
        for marker in ("unsupported", "not support", "unknown parameter")
    ):
        return "unsupported_capability", "provider_capability_unsupported"
    return "provider_error", "http_provider_error"


class OpenAICompatibleSyncBackend:
    """Chat Completions transport for the direct ``openai_compatible`` adapter."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        credential_ref: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        structured_output_mode: str = DEFAULT_STRUCTURED_OUTPUT_MODE,
        api_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.provider = str(provider or "openai_compatible")
        self._base_url = str(base_url or "").strip()
        self._credential_ref = dict(credential_ref or {})
        self._extra_headers = {
            str(key): str(value)
            for key, value in dict(extra_headers or {}).items()
        }
        self._params = dict(params or {})
        self._structured_output_mode = str(
            structured_output_mode or DEFAULT_STRUCTURED_OUTPUT_MODE
        ).strip()
        self._api_key = str(api_key or "").strip() or None
        self._transport = transport or _default_transport

    # ------------------------------------------------------------------
    # Public backend contract
    # ------------------------------------------------------------------
    def generate(self, request: SyncGenerationRequest) -> SyncGenerationResult:
        payload, request_metadata = self._build_payload(request)
        headers = self._build_headers()
        timeout = normalize_sync_timeout_seconds(
            self._effective_config(request).get("timeout")
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            http_request = urllib.request.Request(
                self._chat_completions_url(),
                data=body,
                headers=headers,
                method="POST",
            )
        except (UnicodeEncodeError, ValueError) as exc:
            raise SyncBackendError(
                "unsupported_capability",
                request_metadata={
                    **request_metadata,
                    "reason": "invalid_request",
                },
            ) from exc
        try:
            response = self._transport(http_request, timeout)
        except (socket.timeout, TimeoutError) as exc:
            raise SyncBackendError(
                "timeout",
                request_metadata={**request_metadata, "reason": "transport_timeout"},
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            category = "timeout" if isinstance(reason, (socket.timeout, TimeoutError)) else "provider_error"
            raise SyncBackendError(
                category,
                request_metadata={
                    **request_metadata,
                    "reason": "transport_error",
                },
            ) from exc
        except OSError as exc:
            raise SyncBackendError(
                "provider_error",
                request_metadata={
                    **request_metadata,
                    "reason": "transport_error",
                },
            ) from exc

        body_text = response.body.decode("utf-8", errors="replace")
        if response.status < 200 or response.status >= 300:
            category, reason = _classify_http_error(
                response.status,
                body_text,
                structured_output_requested=(
                    "response_format" in payload
                ),
            )
            raise SyncBackendError(
                category,
                request_metadata={
                    **request_metadata,
                    "status_code": response.status,
                    "reason": reason,
                },
            )
        return self._build_result(
            request,
            body_text,
            request_metadata=request_metadata,
            status_code=response.status,
        )

    async def generate_async(
        self,
        request: SyncGenerationRequest,
    ) -> SyncGenerationResult:
        """Run the blocking transport off the event loop for GUI/probe callers."""

        return await asyncio.to_thread(self.generate, request)

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------
    def _effective_config(self, request: SyncGenerationRequest) -> dict[str, Any]:
        """Merge request defaults with model-level profile params.

        The plan / caller config supplies task defaults; ModelProfile params are
        the model-level explicit override and therefore win on collision.
        """

        config = dict(request.config or {})
        config.update(self._params)
        return config

    def _chat_completions_url(self) -> str:
        base = self._base_url.rstrip("/")
        if not base:
            raise SyncBackendError(
                "unsupported_capability",
                request_metadata={"reason": "missing_base_url"},
            )
        lowered = base.casefold()
        if lowered.endswith("/chat/completions"):
            return base
        if not lowered.startswith(("http://", "https://")):
            raise SyncBackendError(
                "unsupported_capability",
                request_metadata={"reason": "invalid_base_url"},
            )
        return f"{base}/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "renpy-translation-lab/openai-compatible-sync",
        }
        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        for key, value in self._extra_headers.items():
            if not key or not value:
                continue
            try:
                key.encode("ascii")
                value.encode("latin-1")
            except UnicodeEncodeError as exc:
                raise SyncBackendError(
                    "unsupported_capability",
                    request_metadata={
                        "provider": self.provider,
                        "reason": "invalid_extra_header",
                    },
                ) from exc
            headers[key] = value
        return headers

    def _resolve_api_key(self) -> str | None:
        if self._api_key:
            return self._api_key
        ref = self._credential_ref
        kind = str(ref.get("kind") or "none").strip().lower()
        if kind == "none":
            return None
        if kind == "env":
            name = str(ref.get("name") or ref.get("env_name") or "").strip()
            value = str(os.environ.get(name) or "").strip()
            if not value:
                raise SyncBackendError(
                    "authentication",
                    request_metadata={"reason": "credential_env_missing"},
                )
            return value
        if kind == "keyring":
            name = str(ref.get("name") or self.provider or "").strip()
            try:
                from litellm_provider_config import load_provider_api_key

                value = str(load_provider_api_key(name) or "").strip()
            except Exception:
                value = ""
            if not value:
                raise SyncBackendError(
                    "authentication",
                    request_metadata={"reason": "credential_keyring_missing"},
                )
            return value
        raise SyncBackendError(
            "unsupported_capability",
            request_metadata={"reason": "credential_kind_unsupported"},
        )

    def _build_payload(
        self,
        request: SyncGenerationRequest,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        config = self._effective_config(request)
        unsupported = sorted(_UNSUPPORTED_OPTION_KEYS & set(config))
        if unsupported:
            raise SyncBackendError(
                "unsupported_capability",
                request_metadata={
                    "provider": self.provider,
                    "reason": "gemini_only_option",
                    "option": unsupported[0],
                },
            )
        ignored = sorted(
            (_IGNORED_OPTION_KEYS | _GEMINI_ONLY_KEYS) & set(config)
        )
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _build_messages(request.contents, config),
            "stream": False,
        }
        for key in _GENERATION_PARAM_KEYS:
            if key in config and config[key] is not None:
                payload[key] = config[key]
        max_output_tokens = config.get("max_output_tokens")
        if max_output_tokens is not None:
            try:
                parsed_limit = int(max_output_tokens)
            except (TypeError, ValueError, OverflowError):
                raise SyncBackendError(
                    "unsupported_capability",
                    request_metadata={
                        "provider": self.provider,
                        "reason": "invalid_max_output_tokens",
                    },
                )
            if parsed_limit > 0:
                payload["max_tokens"] = parsed_limit
        mode = str(
            config.get("structured_output_mode")
            or self._structured_output_mode
            or DEFAULT_STRUCTURED_OUTPUT_MODE
        ).strip()
        if mode not in STRUCTURED_OUTPUT_MODES:
            raise SyncBackendError(
                "unsupported_capability",
                request_metadata={
                    "provider": self.provider,
                    "reason": "unsupported_structured_output_mode",
                },
            )
        schema = config.get("response_json_schema")
        schema = dict(schema) if isinstance(schema, Mapping) else {}
        if mode == "strict_json_schema":
            if not schema:
                raise SyncBackendError(
                    "unsupported_capability",
                    request_metadata={
                        "provider": self.provider,
                        "reason": "missing_response_schema",
                    },
                )
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(schema),
                    "schema": schema,
                    "strict": True,
                },
            }
        elif mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        metadata: dict[str, Any] = {
            "provider": self.provider,
            "structured_output_mode": mode,
            "request_url": self._chat_completions_url(),
        }
        if ignored:
            metadata["ignored_provider_options"] = ignored
        return payload, metadata

    # ------------------------------------------------------------------
    # Response construction
    # ------------------------------------------------------------------
    def _build_result(
        self,
        request: SyncGenerationRequest,
        body_text: str,
        *,
        request_metadata: Mapping[str, Any] | None,
        status_code: int,
    ) -> SyncGenerationResult:
        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise SyncBackendError(
                "invalid_response",
                request_metadata={
                    **dict(request_metadata or {}),
                    "status_code": status_code,
                    "reason": "response_not_json",
                },
            ) from exc
        if not isinstance(payload, Mapping):
            raise SyncBackendError(
                "invalid_response",
                request_metadata={
                    **dict(request_metadata or {}),
                    "status_code": status_code,
                    "reason": "response_not_object",
                },
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SyncBackendError(
                "invalid_response",
                request_metadata={
                    **dict(request_metadata or {}),
                    "status_code": status_code,
                    "reason": "choices_missing",
                },
            )
        choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message = choice.get("message")
        message = message if isinstance(message, Mapping) else {}
        response_text = _text_from_content(message.get("content"))
        if not response_text.strip():
            raise SyncBackendError(
                "invalid_response",
                request_metadata={
                    **dict(request_metadata or {}),
                    "status_code": status_code,
                    "reason": "empty_response_text",
                },
            )
        parsed: Any = None
        stripped = response_text.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            usage = payload.get("usage_metadata")
        reasoning = _reasoning_text(message, choice)
        response_id = str(payload.get("id") or "").strip()
        return SyncGenerationResult(
            provider=self.provider,
            model=request.model,
            execution_mode=SYNC_EXECUTION_MODE,
            response_payload=dict(payload),
            response_text=response_text,
            parsed=parsed,
            finish_reason=str(choice.get("finish_reason") or ""),
            usage_metadata=dict(usage) if isinstance(usage, Mapping) else {},
            output_diagnostics={
                "reasoning_present": bool(reasoning),
                "reasoning_chars": len(reasoning),
            },
            request_metadata={
                **dict(request_metadata or {}),
                "status_code": status_code,
                "response_id": response_id,
            },
        )
