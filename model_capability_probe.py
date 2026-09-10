"""Bounded per-capability probe for one ModelProfile (#348 P3).

The probe answers "which capabilities does this connection actually serve?"
with one acknowledged, tightly bounded generation request. Declared-only
capabilities (remote Batch, embedding) are reported with their capability
source instead of being fabricated as passed. Output never contains credential
values, provider exception bodies, or request text.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import model_profiles_editor as profiles_editor
import model_profile as routing
from model_routing_config import validate_model_routing_section

BILLABLE_ACK_TOKEN = "I_ACKNOWLEDGE_ONE_BILLABLE_PROVIDER_REQUEST"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120
DEFAULT_MAX_OUTPUT_TOKENS = 64
MAX_OUTPUT_TOKENS = 256

PROBE_PROMPT = (
    'Return only the compact JSON object '
    '{"translations":[{"id":"probe-1","translation":"你好"}]}. '
    "Keep the id exact. Do not add prose or markdown."
)


@dataclass(frozen=True)
class ProbeRequest:
    model: str
    prompt: str
    json_schema: Mapping[str, Any]
    max_output_tokens: int
    timeout_seconds: int
    adapter: str
    provider: str
    api_base: str = ""
    profile_id: str = ""
    section: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResponse:
    text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""


class ProbeTransportError(RuntimeError):
    """Injected transport failure with a stable safe category."""

    def __init__(self, category: str, message: str = "") -> None:
        super().__init__(message or category)
        self.category = str(category or "provider_error")


CredentialLoader = Callable[[Mapping[str, Any]], str]
Generator = Callable[[ProbeRequest], ProbeResponse]


def _capability(name: str, status: str, detail: str = "") -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _declared_capabilities(
    section: Mapping[str, Any],
    profile_id: str,
) -> dict[str, Any]:
    choices = profiles_editor.strategy_choices(section)
    view = profiles_editor.editor_view(section)
    profile = next(
        (item for item in view["profiles"] if item["id"] == profile_id),
        {},
    )
    capabilities = dict(profile.get("capabilities") or {})
    return {
        "strategies": tuple(choices.get(profile_id, ())),
        "capabilities": capabilities,
        "sources": dict(capabilities.get("sources") or {}),
        "profile": profile,
    }


def _resolve_gemini_api_key() -> str:
    import os

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "SA_GEMINI_API_KEY"):
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    try:
        import translator_runtime as runtime
    except Exception:
        return ""
    config_path = str(getattr(runtime, "CONFIG_FILE", "") or "")
    if not config_path:
        return ""
    try:
        from config_store import read_json_object
        from pathlib import Path

        payload = read_json_object(Path(config_path), "api keys config")
    except (OSError, ValueError):
        return ""
    keys = payload.get("api_keys")
    if isinstance(keys, list):
        for item in keys:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def default_credential_loader(provider_entry: Mapping[str, Any]) -> str:
    """Resolve a credential reference without importing provider SDKs."""
    adapter = str(provider_entry.get("adapter") or "")
    ref = provider_entry.get("credential_ref")
    ref = dict(ref) if isinstance(ref, Mapping) else {}
    kind = str(ref.get("kind") or routing.CREDENTIAL_KIND_NONE)
    if kind == routing.CREDENTIAL_KIND_NONE:
        return ""
    if kind == routing.CREDENTIAL_KIND_ENV:
        import os

        return str(os.environ.get(str(ref.get("name") or "")) or "").strip()
    if adapter == routing.ADAPTER_GEMINI:
        return _resolve_gemini_api_key()
    provider_id = str(ref.get("name") or provider_entry.get("provider") or "")
    try:
        from litellm_provider_config import load_provider_api_key

        value = str(load_provider_api_key(provider_id) or "").strip()
    except Exception:
        value = ""
    if value:
        return value
    # LiteLLM also reads each provider's native environment variable; probes
    # must not report "credential unavailable" when only that path is set.
    import os

    for env_name in (
        str(ref.get("env_name") or ""),
        f"{provider_id.upper().replace('-', '_')}_API_KEY",
    ):
        if not env_name:
            continue
        candidate = str(os.environ.get(env_name) or "").strip()
        if candidate:
            return candidate
    return ""


def _classify_exception(exc: Exception) -> str:
    if isinstance(exc, ProbeTransportError):
        return exc.category
    try:
        from sync_model_backend import sync_error_category

        category = str(sync_error_category(exc) or "").strip()
        if category:
            return category
    except Exception:
        pass
    return "provider_error"


def _usage_fields(usage: Mapping[str, Any]) -> tuple[bool, bool]:
    usage = dict(usage or {})
    has_usage = bool(usage)
    reasoning = usage.get("reasoning_tokens")
    return has_usage, reasoning is not None


def probe_profile(
    section: Mapping[str, Any] | None,
    profile_id: str,
    *,
    acknowledge: str = "",
    generate: Generator | None = None,
    credential_loader: CredentialLoader | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """Probe one profile and return a credential-free capability report.

    ``generate`` and ``credential_loader`` are injectable so the report logic
    is testable without network access. At most one billed request is issued.
    """
    data = profiles_editor._copy_section(section) if section is not None else None
    if data is None or validate_model_routing_section(data):
        raise profiles_editor.ModelProfilesEditorError(
            "MODEL_ROUTING_INVALID",
            "model_routing must be valid before probing",
        )
    profile = profiles_editor._profile_entry(data, profile_id)
    provider = profiles_editor._provider_entry(
        data,
        str(profile.get("provider_id") or ""),
    )
    declared = _declared_capabilities(data, profile_id)
    results: list[dict[str, str]] = []

    if acknowledge != BILLABLE_ACK_TOKEN:
        raise profiles_editor.ModelProfilesEditorError(
            "BILLABLE_ACK_REQUIRED",
            "Probing issues one billable provider request; pass the acknowledgement token.",
        )

    credential_loader = credential_loader or default_credential_loader
    api_key = ""
    credential_error = ""
    try:
        api_key = str(credential_loader(provider) or "").strip()
    except Exception as exc:  # credential lookup must not leak exception text
        credential_error = _classify_exception(exc)
    needs_key = str(
        (provider.get("credential_ref") or {}).get("kind")
        or routing.CREDENTIAL_KIND_NONE
    ) != routing.CREDENTIAL_KIND_NONE
    if needs_key and not api_key:
        for name in (
            "auth",
            "sync_generation",
            "structured_output",
            "reasoning",
            "usage",
        ):
            results.append(
                _capability(name, "skipped", credential_error or "credential_unavailable")
            )
        results.append(
            _capability(
                "remote_batch",
                "declared" if declared["capabilities"].get("remote_batch") else "unsupported",
                declared["sources"].get("remote_batch", ""),
            )
        )
        results.append(
            _capability(
                "embedding",
                "declared" if declared["capabilities"].get("embedding") else "unsupported",
                declared["sources"].get("embedding", ""),
            )
        )
        return {
            "profile_id": profile_id,
            "model": str(profile.get("model") or ""),
            "adapter": str(provider.get("adapter") or ""),
            "provider": str(provider.get("provider") or ""),
            "requests": 0,
            "status": "skipped",
            "capabilities": results,
        }

    generate = generate or default_generator
    request = ProbeRequest(
        model=str(profile.get("model") or ""),
        prompt=PROBE_PROMPT,
        json_schema=_translation_schema(),
        max_output_tokens=max(1, min(int(max_output_tokens), MAX_OUTPUT_TOKENS)),
        timeout_seconds=max(1, min(int(timeout_seconds), MAX_TIMEOUT_SECONDS)),
        adapter=str(provider.get("adapter") or ""),
        provider=str(provider.get("provider") or ""),
        api_base=str(provider.get("base_url") or ""),
        profile_id=str(profile_id),
        section=data,
    )
    response: ProbeResponse | None = None
    try:
        response = generate(request)
    except Exception as exc:
        category = _classify_exception(exc)
        if category == "authentication":
            results.append(_capability("auth", "fail", category))
        else:
            results.append(_capability("auth", "pass", ""))
        for name in ("sync_generation", "structured_output", "reasoning", "usage"):
            results.append(_capability(name, "fail", category))
        results.append(
            _capability(
                "remote_batch",
                "declared" if declared["capabilities"].get("remote_batch") else "unsupported",
                declared["sources"].get("remote_batch", ""),
            )
        )
        results.append(
            _capability(
                "embedding",
                "declared" if declared["capabilities"].get("embedding") else "unsupported",
                declared["sources"].get("embedding", ""),
            )
        )
        return {
            "profile_id": profile_id,
            "model": request.model,
            "adapter": request.adapter,
            "provider": request.provider,
            "requests": 1,
            "status": "failed",
            "capabilities": results,
        }

    text = str(getattr(response, "text", "") or "").strip()
    usage = dict(getattr(response, "usage", {}) or {})
    results.append(_capability("auth", "pass", ""))
    results.append(
        _capability("sync_generation", "pass" if text else "fail", "" if text else "empty_response")
    )
    structured_ok = False
    structured_detail = "invalid_json"
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            items = parsed.get("translations")
            if (
                isinstance(items, list)
                and items
                and isinstance(items[0], Mapping)
                and str(items[0].get("id") or "") == "probe-1"
                and str(items[0].get("translation") or "").strip()
            ):
                structured_ok = True
                structured_detail = ""
            else:
                structured_detail = "schema_mismatch"
        elif parsed is not None:
            structured_detail = "schema_mismatch"
    results.append(
        _capability(
            "structured_output",
            "pass" if structured_ok else "fail",
            structured_detail if not structured_ok else "",
        )
    )
    has_usage, has_reasoning = _usage_fields(usage)
    results.append(
        _capability("usage", "pass" if has_usage else "fail", "" if has_usage else "usage_missing")
    )
    results.append(
        _capability(
            "reasoning",
            "pass" if has_reasoning else "not_reported",
            "" if has_reasoning else "provider_did_not_report",
        )
    )
    results.append(
        _capability(
            "remote_batch",
            "declared" if declared["capabilities"].get("remote_batch") else "unsupported",
            declared["sources"].get("remote_batch", ""),
        )
    )
    results.append(
        _capability(
            "embedding",
            "declared" if declared["capabilities"].get("embedding") else "unsupported",
            declared["sources"].get("embedding", ""),
        )
    )
    overall = "passed"
    severe = {
        item["name"]: item["status"]
        for item in results
        if item["name"] in {"auth", "sync_generation", "structured_output", "usage"}
    }
    if "fail" in severe.values():
        overall = "failed"
    return {
        "profile_id": profile_id,
        "model": request.model,
        "adapter": request.adapter,
        "provider": request.provider,
        "requests": 1,
        "status": overall,
        "capabilities": results,
    }


def _translation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "translation": {"type": "string"},
                    },
                    "required": ["id", "translation"],
                },
            }
        },
        "required": ["translations"],
    }


def default_generator(request: ProbeRequest) -> ProbeResponse:
    """Run exactly one bounded request through the shared Sync backend factory.

    Production construction goes through ``model_profile.build_sync_backend``
    so probes cannot drift from the real adapter/credential contract.
    """
    import model_routing_reader as reader
    from sync_model_backend import SyncGenerationRequest

    profile = profiles_editor.model_profile_for(request.section, request.profile_id)
    provider_raw = (request.section.get("profiles") or {}).get(request.profile_id) or {}
    provider_id = str(provider_raw.get("provider_id") or "")
    provider_entry = (request.section.get("providers") or {}).get(provider_id)
    try:
        custom_providers = reader.section_custom_providers(request.section)
    except (KeyError, TypeError, ValueError):
        custom_providers = {}
    diagnostic_api_key = ""
    if isinstance(provider_entry, Mapping):
        try:
            diagnostic_api_key = default_credential_loader(provider_entry)
        except Exception:
            diagnostic_api_key = ""

    client = None
    if profile.adapter == routing.ADAPTER_GEMINI:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=diagnostic_api_key,
            http_options=types.HttpOptions(timeout=request.timeout_seconds * 1000),
        )

    backend = routing.build_sync_backend(
        profile,
        custom_providers=custom_providers,
        diagnostic_api_key=diagnostic_api_key,
        client=client,
    )
    config: dict[str, Any] = {
        "temperature": 0,
        "max_output_tokens": request.max_output_tokens,
        "timeout": request.timeout_seconds,
        "response_json_schema": _translation_schema(),
    }
    if request.adapter == routing.ADAPTER_GEMINI:
        config["response_mime_type"] = "application/json"
    result = backend.generate(
        SyncGenerationRequest(
            model=request.model,
            contents=request.prompt,
            config=config,
        )
    )
    return ProbeResponse(
        text=str(getattr(result, "response_text", "") or ""),
        usage=dict(getattr(result, "usage_metadata", {}) or {}),
        finish_reason=str(getattr(result, "finish_reason", "") or ""),
    )
