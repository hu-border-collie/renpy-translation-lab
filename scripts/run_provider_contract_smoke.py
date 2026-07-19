"""Run bounded, credential-optional contract smokes through production adapters."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from litellm_provider_config import DEFAULT_MODELS, SUPPORTED_PROVIDERS
from sync_model_backend import GeminiSyncBackend, SyncGenerationRequest


REQUEST_TIMEOUT_SECONDS = 30
MAX_OUTPUT_TOKENS = 64
MAX_REQUESTS_PER_PROVIDER = 1
ESTIMATED_MAX_COST_USD_PER_PROVIDER = 0.01
GEMINI_MODEL = "gemini-3.1-flash-lite"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    backend: str
    model: str
    secret_environment: str


_LITELLM_SECRET_ENVIRONMENTS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
}


def provider_specs() -> tuple[ProviderSpec, ...]:
    specs = [
        ProviderSpec("gemini", "gemini", GEMINI_MODEL, "GEMINI_API_KEY"),
    ]
    for provider, _label in SUPPORTED_PROVIDERS:
        if provider == "ollama":
            continue
        secret_environment = _LITELLM_SECRET_ENVIRONMENTS.get(provider)
        models = DEFAULT_MODELS.get(provider, ())
        if not secret_environment or not models:
            raise RuntimeError(
                f"External provider {provider!r} needs a smoke secret and default model."
            )
        specs.append(
            ProviderSpec(provider, "litellm", models[0], secret_environment)
        )
    return tuple(specs)


PROVIDER_SPECS = provider_specs()
PROVIDER_BY_NAME = {spec.name: spec for spec in PROVIDER_SPECS}


class ContractSmokeError(RuntimeError):
    """The provider response no longer satisfies the expected contract."""

    category = "invalid_response"


def api_key_for(
    spec: ProviderSpec,
    environment: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environment is None else environment
    value = environment.get("PROVIDER_API_KEY") or environment.get(
        spec.secret_environment
    )
    return str(value or "").strip()


def contract_request(spec: ProviderSpec) -> SyncGenerationRequest:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    config: dict[str, Any] = {
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_json_schema": schema,
    }
    if spec.backend == "gemini":
        config["response_mime_type"] = "application/json"
    else:
        config["timeout"] = REQUEST_TIMEOUT_SECONDS
    return SyncGenerationRequest(
        model=spec.model,
        contents=(
            'Return only the compact JSON object {"ok":true}. '
            "Do not add prose or markdown."
        ),
        config=config,
    )


def create_backend(spec: ProviderSpec, api_key: str) -> Any:
    if spec.backend == "litellm":
        from litellm_sync_backend import LiteLLMSyncBackend

        return LiteLLMSyncBackend(api_key=api_key)

    from google import genai
    from google.genai import types
    from gemini_translate_batch import (
        extract_finish_reason,
        extract_text_from_response_payload,
        extract_usage_metadata,
        serialize_unknown,
        summarize_usage_metadata,
    )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
    )
    return GeminiSyncBackend(
        client,
        serialize_response=serialize_unknown,
        extract_text=extract_text_from_response_payload,
        extract_finish_reason=extract_finish_reason,
        extract_usage=lambda payload: summarize_usage_metadata(
            extract_usage_metadata(payload)
        ),
    )


def validate_result(spec: ProviderSpec, result: Any) -> dict[str, Any]:
    if result.provider != spec.backend:
        raise ContractSmokeError(
            f"provider mismatch: expected {spec.backend!r}, got {result.provider!r}"
        )
    if result.model != spec.model:
        raise ContractSmokeError(
            f"model mismatch: expected {spec.model!r}, got {result.model!r}"
        )
    text = str(result.response_text or "").strip()
    if not text:
        raise ContractSmokeError("response text is empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractSmokeError("response text is not valid JSON") from exc
    if not isinstance(parsed, dict) or parsed.get("ok") is not True:
        raise ContractSmokeError("response JSON does not satisfy the smoke schema")
    return parsed


def classify_error(exc: Exception) -> str:
    category = str(getattr(exc, "category", "") or "").strip()
    if category:
        return category
    status = getattr(exc, "status_code", None)
    if status in {401, 403}:
        return "authentication"
    if status == 429:
        return "rate_limit"
    if status in {408, 502, 503, 504} or isinstance(exc, TimeoutError):
        return "service_unavailable"
    return "provider_error"


def run_provider(spec: ProviderSpec, api_key: str, backend: Any = None) -> None:
    backend = backend or create_backend(spec, api_key)
    result = backend.generate(contract_request(spec))
    validate_result(spec, result)
    usage = dict(result.usage_metadata or {})
    print(
        "PASS "
        f"provider={spec.name} backend={spec.backend} model={spec.model} "
        f"requests={MAX_REQUESTS_PER_PROVIDER} "
        f"max_output_tokens={MAX_OUTPUT_TOKENS} usage={json.dumps(usage, sort_keys=True)}"
    )


def run_selected(
    selected: list[ProviderSpec],
    environment: Mapping[str, str] | None = None,
) -> int:
    passed = 0
    skipped = 0
    failed = 0
    for spec in selected:
        api_key = api_key_for(spec, environment)
        if not api_key:
            skipped += 1
            print(
                f"SKIP provider={spec.name} missing_secret={spec.secret_environment}"
            )
            continue
        try:
            run_provider(spec, api_key)
            passed += 1
        except Exception as exc:
            failed += 1
            print(
                f"FAIL provider={spec.name} category={classify_error(exc)}: {exc}",
                file=sys.stderr,
            )
    print(
        f"SUMMARY passed={passed} skipped={skipped} failed={failed} "
        f"requests_max={len(selected) * MAX_REQUESTS_PER_PROVIDER} "
        "estimated_cost_max_usd="
        f"{len(selected) * ESTIMATED_MAX_COST_USD_PER_PROVIDER:.2f}"
    )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("all", *PROVIDER_BY_NAME),
        default="all",
        help="provider contract to run; missing credentials are successful skips",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = (
        list(PROVIDER_SPECS)
        if args.provider == "all"
        else [PROVIDER_BY_NAME[args.provider]]
    )
    return run_selected(selected)


if __name__ == "__main__":
    raise SystemExit(main())
