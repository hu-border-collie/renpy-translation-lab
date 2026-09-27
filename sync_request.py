"""Shared synchronous request execution, independent of CLI entry modules.

Routing is frozen by the caller. Runtime dependencies are short-lived assembly
values, not a second configuration store; defaults read translator_runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
import time

import model_profile
import model_usage_ledger
from gemini_model_catalog import filter_gemini_generation_config
from sync_model_backend import (
    DEFAULT_SYNC_RETRY_ATTEMPTS,
    SyncGenerationRequest,
    normalize_sync_timeout_seconds,
    sync_recovery_decision,
)


@dataclass(frozen=True)
class SyncRequestRuntime:
    """Explicit per-run configuration and credential access for sync requests.

    Credential callbacks use the existing runtime key store and rotation policy.
    No credentials or routing state are persisted in this object.
    """

    timeout_seconds: float
    custom_providers: Mapping[str, Any]
    create_client: Callable[..., Any]
    credential_attempts: Callable[[], int]
    rotate_credentials: Callable[[], bool]


def create_runtime_client(*, api_key_index=None):
    """Build a Gemini SDK client using the existing runtime credential store."""
    import translator_runtime as runtime

    if api_key_index is None:
        return runtime.create_genai_client()
    try:
        index = int(api_key_index)
    except (TypeError, ValueError):
        raise SystemExit(f'Invalid API key index: {api_key_index}') from None
    keys = runtime.API_KEYS or []
    if not 0 <= index < len(keys):
        raise SystemExit(f'Invalid API key index: {api_key_index}')
    return runtime.create_genai_client(api_key=keys[index])


def runtime_dependencies(*, timeout_seconds=None, create_client=None) -> SyncRequestRuntime:
    """Read applied runtime settings once; optionally bind an entry's SDK factory."""
    import translator_runtime as runtime

    return SyncRequestRuntime(
        timeout_seconds=normalize_sync_timeout_seconds(
            runtime.SYNC_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        ),
        custom_providers=dict(runtime.CUSTOM_LITELLM_PROVIDERS),
        create_client=create_client or create_runtime_client,
        credential_attempts=runtime.api_key_rotation_attempts,
        rotate_credentials=runtime.rotate_api_key,
    )


def serialize_unknown(value):
    """Convert SDK response objects into the existing JSON-compatible payload."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): serialize_unknown(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_unknown(item) for item in value]
    for method_name in ('model_dump', 'dict'):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return serialize_unknown(method())
            except Exception:
                pass
    if hasattr(value, '__dict__'):
        return serialize_unknown(vars(value))
    return str(value)


def extract_text_from_response_payload(response_payload):
    """Read candidate text from a Gemini payload or nested response envelope."""
    payload = response_payload
    if not isinstance(payload, dict):
        return ''

    nested_response = payload.get('response')
    if isinstance(nested_response, dict):
        payload = nested_response

    candidates = payload.get('candidates')
    if isinstance(candidates, list):
        for candidate in candidates:
            content = candidate.get('content') if isinstance(candidate, dict) else None
            parts = content.get('parts') if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            texts = []
            for part in parts:
                if isinstance(part, dict) and part.get('text'):
                    texts.append(part['text'])
            if texts:
                return ''.join(texts)

    text = payload.get('text')
    return text if isinstance(text, str) else ''


def extract_finish_reason(response_payload):
    """Read the recorded Gemini candidate finish reason."""
    payload = response_payload if isinstance(response_payload, dict) else {}
    nested_response = payload.get('response')
    if isinstance(nested_response, dict):
        payload = nested_response

    candidates = payload.get('candidates')
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get('finishReason'):
                return str(candidate['finishReason'])
    return ''


def extract_usage_metadata(response_payload):
    """Read provider usage without changing token field names."""
    payload = response_payload if isinstance(response_payload, dict) else {}
    nested_response = payload.get('response')
    if isinstance(nested_response, dict):
        payload = nested_response
    usage = payload.get('usageMetadata')
    return usage if isinstance(usage, dict) else {}


def summarize_usage_metadata(usage_metadata):
    """Copy usage metadata while preserving the existing response contract."""
    if not isinstance(usage_metadata, dict):
        return {}
    return dict(usage_metadata)


def _sync_result_to_dict(result):
    response = {
        'response_payload': result.response_payload,
        'response_text': result.response_text,
        'finish_reason': result.finish_reason,
        'usage_metadata': dict(result.usage_metadata),
        'provider': result.provider,
        'model': result.model,
        'execution_mode': result.execution_mode,
    }
    request_metadata = dict(getattr(result, 'request_metadata', None) or {})
    if request_metadata:
        response['request_metadata'] = request_metadata
    output_diagnostics = dict(getattr(result, 'output_diagnostics', None) or {})
    if output_diagnostics:
        response['output_diagnostics'] = output_diagnostics
    return response


def _run_sync_backend_with_retry(
    backend,
    request,
    *,
    attempts=DEFAULT_SYNC_RETRY_ATTEMPTS,
):
    """Retry only transient structured categories on the same backend."""
    limit = max(1, int(attempts or 1))
    for attempt in range(1, limit + 1):
        try:
            return backend.generate(request)
        except Exception as exc:
            decision = sync_recovery_decision(exc)
            if not decision.retry_same_request or attempt >= limit:
                raise
            print(
                f'Sync request {decision.category}; retrying '
                f'({attempt}/{limit})...',
            )
            if decision.backoff:
                time.sleep(min(attempt, 2))
    raise RuntimeError('Sync request failed without a captured exception.')


def _require_task_route(route):
    if not isinstance(route, model_profile.TaskRoute):
        raise TypeError(
            'run_sync_request requires an explicit TaskRoute; '
            f'got {type(route).__name__}.'
        )
    return route


def run_sync_request(
    request_payload,
    route,
    plan=None,
    *,
    api_key_index=None,
    retry_attempts=None,
    allow_credential_rotation=True,
    timeout_seconds=None,
    runtime: SyncRequestRuntime | None = None,
):
    """Execute one sync request using a frozen TaskRoute.

    The model comes from ``plan.profiles[route.profile_id]``. ``SYNC_MODEL``
    is never consulted here; callers must freeze a :class:`ModelRoutingPlan`
    at run start and pass that snapshot.  Durable callers disable credential
    rotation because every Provider invocation must have its own persisted
    attempt boundary.
    """
    route = _require_task_route(route)
    if plan is None:
        raise TypeError(
            'run_sync_request requires the frozen ModelRoutingPlan from run start.'
        )
    runtime = runtime if runtime is not None else runtime_dependencies()
    profile = model_profile.profile_for_route(plan, route)
    effective_model = profile.model
    config = dict(request_payload.get('generation_config') or {})
    config['timeout'] = normalize_sync_timeout_seconds(
        runtime.timeout_seconds if timeout_seconds is None else timeout_seconds
    )
    system_instruction = request_payload.get('system_instruction')
    if system_instruction:
        config['system_instruction'] = system_instruction
    safety_settings = request_payload.get('safety_settings')
    if safety_settings:
        config['safety_settings'] = safety_settings
    config = filter_gemini_generation_config(effective_model, config)

    if profile.adapter in {
        model_profile.ADAPTER_LITELLM,
        model_profile.ADAPTER_OPENAI_COMPATIBLE,
    }:
        if api_key_index is not None:
            raise SystemExit('--api-key-index is only supported by the Gemini sync backend.')
        backend_kwargs = {}
        if profile.adapter == model_profile.ADAPTER_LITELLM:
            backend_kwargs['custom_providers'] = runtime.custom_providers
        # openai_compatible profiles carry base_url / credential_ref /
        # extra_headers on the frozen ModelProfile; the LiteLLM custom-provider
        # registry is deliberately not part of that contract.
        backend = model_profile.build_sync_backend(profile, **backend_kwargs)
        request = SyncGenerationRequest(
            model=effective_model,
            contents=request_payload.get('contents') or [],
            config=config,
        )
        result = _run_sync_backend_with_retry(
            backend,
            request,
            attempts=(
                DEFAULT_SYNC_RETRY_ATTEMPTS
                if retry_attempts is None
                else retry_attempts
            ),
        )
        response = _sync_result_to_dict(result)
        response['output_diagnostics'] = model_usage_ledger.response_budget_diagnostics(
            response_text=result.response_text,
            finish_reason=result.finish_reason,
            usage_metadata=result.usage_metadata,
            max_output_tokens=config.get('max_output_tokens'),
        )
        return response

    key_attempts = (
        runtime.credential_attempts()
        if (
            allow_credential_rotation
            and api_key_index is None
        )
        else 1
    )
    attempts = (
        max(DEFAULT_SYNC_RETRY_ATTEMPTS, key_attempts)
        if retry_attempts is None
        else max(1, int(retry_attempts))
    )
    last_error = None

    for attempt in range(1, attempts + 1):
        client = runtime.create_client(api_key_index=api_key_index)
        try:
            backend = model_profile.build_sync_backend(
                profile,
                client=client,
                serialize_response=serialize_unknown,
                extract_text=extract_text_from_response_payload,
                extract_finish_reason=extract_finish_reason,
                extract_usage=lambda payload: summarize_usage_metadata(
                    extract_usage_metadata(payload)
                ),
            )
            result = backend.generate(SyncGenerationRequest(
                model=effective_model,
                contents=request_payload.get('contents') or [],
                config=config,
            ))
            response = _sync_result_to_dict(result)
            response['output_diagnostics'] = model_usage_ledger.response_budget_diagnostics(
                response_text=result.response_text,
                finish_reason=result.finish_reason,
                usage_metadata=result.usage_metadata,
                max_output_tokens=config.get('max_output_tokens'),
            )
            return response

        except Exception as exc:
            last_error = exc
            decision = sync_recovery_decision(exc)
            if decision.retry_same_request and attempt < attempts:
                rotated = bool(
                    decision.rotate_credentials
                    and allow_credential_rotation
                    and api_key_index is None
                    and runtime.rotate_credentials()
                )
                label = decision.category.replace('_', ' ')
                key_action = 'next API key' if rotated else 'same API key'
                print(
                    f'Sync request hit {label}. Retrying with {key_action} '
                    f'({attempt}/{attempts})...'
                )
                if decision.backoff:
                    time.sleep(min(attempt, 2))
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError('Sync request failed without a captured exception.')
