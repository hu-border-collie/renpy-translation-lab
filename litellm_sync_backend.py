"""Optional LiteLLM implementation of the synchronous model backend."""

import os
from typing import Any, Callable, Dict, List, Mapping, Optional

from gemini_model_catalog import filter_gemini_generation_config
from litellm_provider_config import (
    CustomLiteLLMProvider,
    provider_from_model,
)
from sync_model_backend import SYNC_EXECUTION_MODE, SyncGenerationRequest, SyncGenerationResult


class LiteLLMBackendError(RuntimeError):
    def __init__(self, message: str, *, category: str = "provider_error") -> None:
        super().__init__(message)
        self.category = category


class LiteLLMUnavailableError(LiteLLMBackendError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="missing_dependency")


class LiteLLMCapabilityError(LiteLLMBackendError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="unsupported_capability")


def _serialize_response(response: Any) -> Mapping[str, Any]:
    payload: Dict[str, Any] | None = None
    if isinstance(response, Mapping):
        payload = dict(response)
    else:
        for method_name in ("model_dump", "to_dict"):
            method = getattr(response, method_name, None)
            if callable(method):
                serialized = method()
                if isinstance(serialized, Mapping):
                    payload = dict(serialized)
                    break
    if payload is None:
        raise LiteLLMBackendError(
            f"LiteLLM returned unsupported response type: {type(response).__name__}",
            category="invalid_response",
        )
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, Mapping) and "_hidden_params" not in payload:
        payload["_hidden_params"] = dict(hidden)
    return payload


def _instruction_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return str(value or "")
    parts = value.get("parts") or []
    return "\n".join(
        str(part.get("text") or "") for part in parts if isinstance(part, Mapping)
    )


def _messages(contents: Any, config: Mapping[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if config.get("system_instruction"):
        messages.append({
            "role": "system",
            "content": _instruction_text(config["system_instruction"]),
        })
    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
        return messages
    if not isinstance(contents, list):
        raise LiteLLMCapabilityError("LiteLLM contents must be text or a message list.")
    for entry in contents:
        if not isinstance(entry, Mapping):
            raise LiteLLMCapabilityError("LiteLLM message entries must be objects.")
        messages.append({
            "role": str(entry.get("role") or "user"),
            "content": (
                str(entry.get("content") or "")
                if "content" in entry
                else _instruction_text(entry)
            ),
        })
    return messages


def _error_category(exc: Exception) -> str:
    try:
        import litellm

        typed_categories = (
            ("rate_limit", (getattr(litellm, "RateLimitError", None),)),
            ("service_unavailable", (
                getattr(litellm, "ServiceUnavailableError", None),
                getattr(litellm, "Timeout", None),
                getattr(litellm, "APIConnectionError", None),
            )),
            ("authentication", (
                getattr(litellm, "AuthenticationError", None),
                getattr(litellm, "PermissionDeniedError", None),
            )),
        )
        for category, candidates in typed_categories:
            exception_types = tuple(
                candidate for candidate in candidates if isinstance(candidate, type)
            )
            if exception_types and isinstance(exc, exception_types):
                return category
    except ImportError:
        pass

    status = getattr(exc, "status_code", None)
    if status == 429:
        return "rate_limit"
    if status in {502, 503, 504}:
        return "service_unavailable"
    if status in {401, 403}:
        return "authentication"
    return "provider_error"

class LiteLLMSyncBackend:
    """Lazy optional adapter; importing this module does not import LiteLLM."""

    provider = "litellm"

    def __init__(
        self,
        completion: Optional[Callable[..., Any]] = None,
        api_key: Optional[str] = None,
        async_completion: Optional[Callable[..., Any]] = None,
        custom_providers: Optional[Mapping[str, CustomLiteLLMProvider]] = None,
    ) -> None:
        self._completion = completion
        self._async_completion = async_completion
        self._api_key = str(api_key or "").strip()
        self._custom_providers = (
            dict(custom_providers) if isinstance(custom_providers, Mapping) else {}
        )

    def _resolve_completion(self) -> Callable[..., Any]:
        if self._completion is not None:
            return self._completion
        try:
            from litellm import completion
        except ImportError as exc:
            raise LiteLLMUnavailableError(
                "LiteLLM backend was selected but litellm is not installed. "
                "Install the optional dependency or select Gemini Batch."
            ) from exc
        self._completion = completion
        return completion

    def generate(self, request: SyncGenerationRequest) -> SyncGenerationResult:
        kwargs = self._build_request_kwargs(request)
        try:
            response = self._resolve_completion()(**kwargs)
        except LiteLLMBackendError:
            raise
        except Exception as exc:
            raise LiteLLMBackendError(
                f"LiteLLM request failed: {exc}", category=_error_category(exc)
            ) from exc
        return self._build_result(request, response)

    def _resolve_async_completion(self) -> Callable[..., Any]:
        if self._async_completion is not None:
            return self._async_completion
        try:
            from litellm import acompletion
        except ImportError as exc:
            raise LiteLLMUnavailableError(
                "LiteLLM async completion is unavailable. "
                "Install the optional dependency or use Gemini Batch."
            ) from exc
        self._async_completion = acompletion
        return acompletion

    async def generate_async(self, request: SyncGenerationRequest) -> SyncGenerationResult:
        """Run a LiteLLM request through its async API so task cancellation reaches I/O."""
        kwargs = self._build_request_kwargs(request)
        try:
            response = await self._resolve_async_completion()(**kwargs)
        except LiteLLMBackendError:
            raise
        except Exception as exc:
            raise LiteLLMBackendError(
                f"LiteLLM request failed: {exc}", category=_error_category(exc)
            ) from exc
        return self._build_result(request, response)

    def _build_request_kwargs(self, request: SyncGenerationRequest) -> Dict[str, Any]:
        config = filter_gemini_generation_config(request.model, request.config)
        if config.get("safety_settings"):
            raise LiteLLMCapabilityError(
                "LiteLLM does not share Gemini safety_settings semantics; "
                "remove that setting or use Gemini."
            )
        provider = provider_from_model(request.model)
        custom = self._custom_providers.get(provider)
        model = request.model
        if custom is not None:
            # Custom OpenAI-compatible providers are not known to LiteLLM, so
            # the display id is rewritten to the official openai prefix and the
            # endpoint is passed per-request as api_base (never process-wide).
            prefix = f"{provider}/"
            suffix = (
                model[len(prefix):]
                if model.lower().startswith(prefix)
                else model
            )
            model = f"openai/{suffix}"
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": _messages(request.contents, config),
        }
        api_key = self._api_key
        if not api_key:
            try:
                from litellm_provider_config import load_provider_api_key

                api_key = load_provider_api_key(provider)
            except Exception:
                # keyring is optional; LiteLLM can still use environment variables.
                api_key = ""
        if not api_key and custom is not None and custom.api_key_env:
            # Explicit opt-in env fallback: without a keyring entry, send the
            # named environment variable's value instead of letting LiteLLM
            # silently fall back to OPENAI_API_KEY for a third-party endpoint.
            api_key = str(os.environ.get(custom.api_key_env) or "").strip()
        if api_key:
            kwargs["api_key"] = api_key
        if custom is not None:
            kwargs["api_base"] = custom.base_url
        if "timeout" in config:
            kwargs["timeout"] = config["timeout"]
        if "temperature" in config:
            kwargs["temperature"] = config["temperature"]
        if "max_output_tokens" in config:
            kwargs["max_tokens"] = config["max_output_tokens"]
        schema = config.get("response_json_schema")
        if schema:
            # DeepSeek rejects json_schema even though it supports JSON mode.
            # Custom OpenAI-compatible providers usually target OpenAI-compatible
            # gateways, many of which reject strict schemas; the capability
            # decision therefore uses the *original* id, before the rewrite.
            if provider in {"openai", "azure"}:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "translation_response",
                        "schema": schema,
                        "strict": True,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _build_result(
        self, request: SyncGenerationRequest, response: Any
    ) -> SyncGenerationResult:
        payload = _serialize_response(response)
        choices = payload.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], Mapping) else {}
        message = choice.get("message") or {}
        text = message.get("content") if isinstance(message, Mapping) else ""
        usage = payload.get("usage") or {}
        return SyncGenerationResult(
            provider=self.provider,
            model=request.model,
            execution_mode=SYNC_EXECUTION_MODE,
            response_payload=payload,
            response_text=str(text or ""),
            finish_reason=str(choice.get("finish_reason") or ""),
            usage_metadata=dict(usage) if isinstance(usage, Mapping) else {},
        )
