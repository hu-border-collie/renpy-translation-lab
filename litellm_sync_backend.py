"""Optional LiteLLM implementation of the synchronous model backend."""

from typing import Any, Callable, Dict, List, Mapping, Optional

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
    if isinstance(response, Mapping):
        return dict(response)
    for method_name in ("model_dump", "to_dict"):
        method = getattr(response, method_name, None)
        if callable(method):
            payload = method()
            if isinstance(payload, Mapping):
                return dict(payload)
    raise LiteLLMBackendError(
        f"LiteLLM returned unsupported response type: {type(response).__name__}",
        category="invalid_response",
    )


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

    def __init__(self, completion: Optional[Callable[..., Any]] = None) -> None:
        self._completion = completion

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
        config = dict(request.config)
        if config.get("safety_settings"):
            raise LiteLLMCapabilityError(
                "LiteLLM does not share Gemini safety_settings semantics; "
                "remove that setting or use Gemini."
            )
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": _messages(request.contents, config),
        }
        if "temperature" in config:
            kwargs["temperature"] = config["temperature"]
        if "max_output_tokens" in config:
            kwargs["max_tokens"] = config["max_output_tokens"]
        schema = config.get("response_json_schema")
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation_response",
                    "schema": schema,
                    "strict": True,
                },
            }
        try:
            response = self._resolve_completion()(**kwargs)
        except LiteLLMBackendError:
            raise
        except Exception as exc:
            raise LiteLLMBackendError(
                f"LiteLLM request failed: {exc}", category=_error_category(exc)
            ) from exc
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
