"""Optional LiteLLM implementation of the synchronous model backend."""

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from gemini_model_catalog import filter_gemini_generation_config
from litellm_provider_config import (
    CustomLiteLLMProvider,
    provider_from_model,
    structured_output_capability,
)
from sync_model_backend import (
    SYNC_EXECUTION_MODE,
    SYNC_ERROR_CATEGORIES,
    SyncGenerationRequest,
    SyncGenerationResult,
    normalize_sync_timeout_seconds,
    sync_error_category,
)


class LiteLLMBackendError(RuntimeError):
    """Backend failure whose message is whitelisted or explicitly internal.

    ``message`` must be one of the safe per-category messages unless the caller
    explicitly opts in with ``internal=True`` for intentional user-facing text.
    Provider-derived text is never accepted as the message; it may only travel
    in ``detail``, which is never rendered by :meth:`__str__` and exists for
    local logs.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str = "provider_error",
        request_metadata: Mapping[str, Any] | None = None,
        internal: bool = False,
        detail: str | None = None,
    ) -> None:
        normalized = str(category or "provider_error").strip().lower()
        if normalized not in SYNC_ERROR_CATEGORIES:
            normalized = "provider_error"
        safe_message = _SAFE_ERROR_MESSAGES.get(
            normalized,
            _SAFE_ERROR_MESSAGES["provider_error"],
        )
        if not internal and message != safe_message:
            # Whitelist enforcement: only known safe text may become the
            # user-visible message; anything else is replaced by the category
            # summary so a missed sanitization point cannot echo provider text.
            message = safe_message
        super().__init__(message)
        self.category = normalized
        self.request_metadata = dict(request_metadata or {})
        self.detail = str(detail or "")[:500]


class LiteLLMUnavailableError(LiteLLMBackendError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="missing_dependency", internal=True)


class LiteLLMCapabilityError(LiteLLMBackendError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="unsupported_capability", internal=True)


_SAFE_ERROR_MESSAGES = {
    "authentication": "LiteLLM authentication failed.",
    "rate_limit": "LiteLLM request was rate limited.",
    "service_unavailable": "LiteLLM service is temporarily unavailable.",
    "timeout": "LiteLLM request timed out.",
    "invalid_response": "LiteLLM returned an invalid response.",
    "unsupported_capability": "LiteLLM provider does not support this request.",
    "missing_dependency": "LiteLLM optional dependency is unavailable.",
    "provider_error": "LiteLLM provider request failed.",
}


def _safe_backend_error(
    exc: Exception,
    *,
    request_metadata: Mapping[str, Any] | None = None,
) -> LiteLLMBackendError:
    """Convert arbitrary provider failures without echoing provider text."""
    category = sync_error_category(exc)
    return LiteLLMBackendError(
        _SAFE_ERROR_MESSAGES.get(category, _SAFE_ERROR_MESSAGES["provider_error"]),
        category=category,
        request_metadata=request_metadata,
        detail=str(exc),
    )


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
            internal=True,
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


def _masked_key_identity(provider: str, key: str, index: int | None = None) -> str:
    """Return a log-safe provider/key identity without exposing the secret."""
    suffix = str(key or "")[-4:] if len(str(key or "")) >= 4 else ""
    ordinal = f"#{index + 1}" if index is not None else ""
    return f"{provider}{ordinal}:****{suffix}"


@dataclass(frozen=True)
class _ResolvedCredential:
    key: str
    identity: str
    source: str


class LiteLLMSyncBackend:
    """Lazy optional adapter; importing this module does not import LiteLLM."""

    provider = "litellm"

    def __init__(
        self,
        completion: Optional[Callable[..., Any]] = None,
        api_key: Optional[str] = None,
        async_completion: Optional[Callable[..., Any]] = None,
        custom_providers: Optional[Mapping[str, CustomLiteLLMProvider]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._completion = completion
        self._async_completion = async_completion
        self._api_key = str(api_key or "").strip()
        self._custom_providers = (
            dict(custom_providers) if isinstance(custom_providers, Mapping) else {}
        )
        self._sleep = sleep

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
        kwargs, credentials, metadata = self._build_request_context(request)
        response, credential, credential_attempts = self._run_with_credentials(
            self._resolve_completion(),
            kwargs,
            credentials,
            metadata,
        )
        return self._build_result(
            request,
            response,
            request_metadata=self._result_request_metadata(
                metadata,
                credential,
                credential_attempts,
            ),
        )

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
        kwargs, credentials, metadata = self._build_request_context(request)
        response, credential, credential_attempts = await self._run_with_credentials_async(
            self._resolve_async_completion(),
            kwargs,
            credentials,
            metadata,
        )
        return self._build_result(
            request,
            response,
            request_metadata=self._result_request_metadata(
                metadata,
                credential,
                credential_attempts,
            ),
        )

    def _resolve_credentials(
        self,
        provider: str,
        custom: CustomLiteLLMProvider | None,
    ) -> tuple[_ResolvedCredential, ...]:
        if self._api_key:
            return (
                _ResolvedCredential(
                    self._api_key,
                    _masked_key_identity(provider, self._api_key),
                    "explicit",
                ),
            )
        try:
            from litellm_provider_config import (
                load_provider_api_key,
                load_provider_key_store,
            )
        except Exception:
            load_provider_api_key = None
            load_provider_key_store = None
        try:
            active_key = (
                str(load_provider_api_key(provider) or "").strip()
                if load_provider_api_key is not None
                else ""
            )
        except Exception:
            active_key = ""
        if active_key:
            try:
                store = (
                    load_provider_key_store(provider).normalized()
                    if load_provider_key_store is not None
                    else None
                )
            except Exception:
                store = None
        else:
            store = None
        if store is not None and active_key in store.keys:
            ordered_indices = [
                store.keys.index(active_key),
                *(
                    index
                    for index in range(len(store.keys))
                    if index != store.keys.index(active_key)
                ),
            ]
            return tuple(
                _ResolvedCredential(
                    store.keys[index],
                    _masked_key_identity(provider, store.keys[index], index),
                    "keyring",
                )
                for index in ordered_indices
            )
        # Keep compatibility with older/injected credential readers that only
        # implement the single-active-key helper.
        if active_key:
            return (
                _ResolvedCredential(
                    active_key,
                    _masked_key_identity(provider, active_key),
                    "keyring",
                ),
            )
        if custom is not None and custom.api_key_env:
            env_key = str(os.environ.get(custom.api_key_env) or "").strip()
            if env_key:
                return (
                    _ResolvedCredential(
                        env_key,
                        _masked_key_identity(provider, env_key),
                        f"env:{custom.api_key_env}",
                    ),
                )
        return ()

    @staticmethod
    def _result_request_metadata(
        metadata: Mapping[str, Any],
        credential: _ResolvedCredential | None,
        credential_attempts: tuple[str, ...],
    ) -> Dict[str, Any]:
        result = dict(metadata)
        if credential is not None:
            result["credential_identity"] = credential.identity
            result["credential_source"] = credential.source
        if credential_attempts:
            result["credential_attempts"] = list(credential_attempts)
        return result

    def _run_with_credentials(
        self,
        completion: Callable[..., Any],
        kwargs: Mapping[str, Any],
        credentials: tuple[_ResolvedCredential, ...],
        metadata: Mapping[str, Any],
    ) -> tuple[Any, _ResolvedCredential | None, tuple[str, ...]]:
        attempts = credentials or (None,)
        attempted_identities: list[str] = []
        for index, credential in enumerate(attempts):
            request_kwargs = dict(kwargs)
            if credential is not None:
                request_kwargs["api_key"] = credential.key
                attempted_identities.append(credential.identity)
            try:
                response = completion(**request_kwargs)
                return response, credential, tuple(attempted_identities)
            except Exception as exc:
                category = sync_error_category(exc)
                if category == "rate_limit" and index + 1 < len(attempts):
                    self._sleep(min(index + 1, 2))
                    continue
                failure_metadata = self._result_request_metadata(
                    metadata,
                    credential,
                    tuple(attempted_identities),
                )
                raise _safe_backend_error(
                    exc,
                    request_metadata=failure_metadata,
                ) from exc
        raise LiteLLMBackendError(
            "LiteLLM request failed without a captured exception.",
            category="provider_error",
            internal=True,
        )

    async def _run_with_credentials_async(
        self,
        completion: Callable[..., Any],
        kwargs: Mapping[str, Any],
        credentials: tuple[_ResolvedCredential, ...],
        metadata: Mapping[str, Any],
    ) -> tuple[Any, _ResolvedCredential | None, tuple[str, ...]]:
        attempts = credentials or (None,)
        attempted_identities: list[str] = []
        for index, credential in enumerate(attempts):
            request_kwargs = dict(kwargs)
            if credential is not None:
                request_kwargs["api_key"] = credential.key
                attempted_identities.append(credential.identity)
            try:
                response = await completion(**request_kwargs)
                return response, credential, tuple(attempted_identities)
            except Exception as exc:
                category = sync_error_category(exc)
                if category == "rate_limit" and index + 1 < len(attempts):
                    import asyncio

                    await asyncio.sleep(min(index + 1, 2))
                    continue
                failure_metadata = self._result_request_metadata(
                    metadata,
                    credential,
                    tuple(attempted_identities),
                )
                raise _safe_backend_error(
                    exc,
                    request_metadata=failure_metadata,
                ) from exc
        raise LiteLLMBackendError(
            "LiteLLM request failed without a captured exception.",
            category="provider_error",
            internal=True,
        )

    def _build_request_context(
        self,
        request: SyncGenerationRequest,
    ) -> tuple[Dict[str, Any], tuple[_ResolvedCredential, ...], Dict[str, Any]]:
        provider = provider_from_model(request.model)
        custom = self._custom_providers.get(provider)
        credentials = self._resolve_credentials(provider, custom)
        kwargs = self._build_request_kwargs(
            request,
            credentials=credentials,
        )
        metadata: Dict[str, Any] = {
            "provider": provider,
            "credential_count": len(credentials),
        }
        if request.config.get("thinking_config"):
            # Gemini's thinking_config has no provider-neutral LiteLLM meaning.
            # It is intentionally not sent; preserve that capability decision
            # in safe request metadata instead of pretending it was honored.
            metadata["ignored_provider_options"] = ["thinking_config"]
        return kwargs, credentials, metadata

    def _build_request_kwargs(
        self,
        request: SyncGenerationRequest,
        *,
        credentials: tuple[_ResolvedCredential, ...] | None = None,
    ) -> Dict[str, Any]:
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
        resolved_credentials = (
            self._resolve_credentials(provider, custom)
            if credentials is None
            else credentials
        )
        if custom is not None and custom.requires_key and not resolved_credentials:
            # The model id has been rewritten to openai/<model>, so LiteLLM
            # would otherwise fall back to OPENAI_API_KEY and leak an unrelated
            # OpenAI key to the third-party api_base. Fail before dispatch
            # instead of risking that leak.
            raise LiteLLMBackendError(
                f"自定义 Provider {custom.label} 需要 API Key，"
                "但系统凭据与 api_key_env 环境变量均未提供。",
                category="authentication",
                internal=True,
            )
        if custom is not None:
            kwargs["api_base"] = custom.base_url
        kwargs["timeout"] = normalize_sync_timeout_seconds(config.get("timeout"))
        if "temperature" in config:
            kwargs["temperature"] = config["temperature"]
        if "max_output_tokens" in config:
            kwargs["max_tokens"] = config["max_output_tokens"]
        schema = config.get("response_json_schema")
        schema_properties = schema.get("properties") if isinstance(schema, Mapping) else {}
        envelope_key = next(
            (
                key
                for key in ("translations", "revisions", "candidates")
                if isinstance(schema_properties, Mapping) and key in schema_properties
            ),
            "model",
        )
        schema_name = f"{envelope_key}_response"
        if schema:
            capability = structured_output_capability(
                provider,
                self._custom_providers,
            )
            requested_mode = str(
                config.get("structured_output_mode") or capability.mode
            ).strip()
            if requested_mode not in {
                "strict_json_schema",
                "json_object",
                "prompt_only_json",
            }:
                raise LiteLLMCapabilityError(
                    f"Unsupported structured output mode: {requested_mode}"
                )
            if requested_mode == "strict_json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                }
            elif requested_mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _build_result(
        self,
        request: SyncGenerationRequest,
        response: Any,
        *,
        request_metadata: Mapping[str, Any] | None = None,
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
            request_metadata=dict(request_metadata or {}),
        )
