"""Pure helpers for the GUI's provider-aware LiteLLM settings page."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from litellm_provider_config import (
    CustomLiteLLMProvider,
    provider_from_model,
)


@dataclass(frozen=True)
class ProviderCredentialStatus:
    provider: str
    environment_names: tuple[str, ...]
    configured: bool | None
    message: str


@dataclass(frozen=True)
class SyncBackendModels:
    gemini_model: str
    litellm_model: str


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def read_sync_backend_models(
    sync_config: Mapping[str, object],
    backend: str,
    recommended_gemini_model: str,
) -> SyncBackendModels:
    active_model = ""
    configured_models = sync_config.get("models")
    if isinstance(configured_models, list):
        for model in configured_models:
            active_model = _clean(model)
            if active_model:
                break
    elif isinstance(configured_models, str):
        active_model = _clean(configured_models)
    if not active_model:
        active_model = _clean(sync_config.get("model"))

    gemini_model = _clean(sync_config.get("gemini_model"))
    if not gemini_model and backend == "gemini":
        gemini_model = active_model
    if not gemini_model:
        gemini_model = recommended_gemini_model

    litellm_model = _clean(sync_config.get("litellm_model"))
    if not litellm_model and backend == "litellm":
        litellm_model = active_model
    return SyncBackendModels(gemini_model, litellm_model)


def write_sync_backend_models(
    sync_config: dict[str, object],
    backend: str,
    gemini_model: str,
    litellm_model: str,
) -> str:
    gemini_model = gemini_model.strip()
    litellm_model = litellm_model.strip()
    sync_config["gemini_model"] = gemini_model
    if litellm_model:
        sync_config["litellm_model"] = litellm_model
    else:
        sync_config.pop("litellm_model", None)
    active_model = litellm_model if backend == "litellm" else gemini_model
    sync_config["backend"] = backend
    sync_config["model"] = active_model
    return active_model


_PROVIDER_ENVIRONMENT: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "azure": ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "huggingface": ("HUGGINGFACE_API_KEY",),
    "novita": ("NOVITA_API_KEY",),
    "nvidia_nim": ("NVIDIA_NIM_API_KEY", "NVIDIA_NIM_API_BASE"),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "vercel_ai_gateway": ("VERCEL_AI_GATEWAY_API_KEY",),
    "xai": ("XAI_API_KEY",),
}


def provider_credential_status(
    model: str,
    environment: Mapping[str, str],
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
) -> ProviderCredentialStatus:
    provider = provider_from_model(model)
    if not provider:
        return ProviderCredentialStatus(
            provider="",
            environment_names=(),
            configured=None,
            message="请先填写带 provider 前缀的模型，例如 openai/gpt-5。",
        )
    if provider == "ollama":
        return ProviderCredentialStatus(
            provider=provider,
            environment_names=(),
            configured=True,
            message="Ollama 通常不需要 API Key；请确保本地服务可访问。",
        )
    custom = custom_providers.get(provider) if isinstance(custom_providers, Mapping) else None
    if custom is not None:
        if custom.api_key_env:
            names = (custom.api_key_env,)
            configured = names[0] in environment
            state = "已检测到" if configured else "未检测到"
            return ProviderCredentialStatus(
                provider=provider,
                environment_names=names,
                configured=configured,
                message=(
                    f"{state}环境变量：{custom.api_key_env}。"
                    "密钥也可直接在「Provider 凭据」区保存到系统凭据管理器。"
                ),
            )
        return ProviderCredentialStatus(
            provider=provider,
            environment_names=(),
            configured=None,
            message="自定义 Provider 未配置 api_key_env；请在「Provider 凭据」区保存 API Key。",
        )
    available_environment_names = frozenset(environment)
    if provider == "vertex_ai":
        names = ("VERTEXAI_PROJECT", "VERTEXAI_LOCATION")
        configured = all(name in available_environment_names for name in names)
        state = "已检测到" if configured else "未完整检测到"
        return ProviderCredentialStatus(
            provider=provider,
            environment_names=names,
            configured=configured,
            message=f"{state} Vertex AI 项目环境变量；身份凭据仍由 Google ADC 管理。",
        )
    names = _PROVIDER_ENVIRONMENT.get(provider, ())
    if not names:
        return ProviderCredentialStatus(
            provider=provider,
            environment_names=(),
            configured=None,
            message="未内置该 provider 的凭据检测；请按 LiteLLM 与供应商文档配置环境变量。",
        )
    configured = all(name in available_environment_names for name in names)
    state = "已检测到" if configured else "未检测到完整"
    return ProviderCredentialStatus(
        provider=provider,
        environment_names=names,
        configured=configured,
        message=f"{state}环境变量：{', '.join(names)}。",
    )


def validate_custom_provider_form(
    provider_id: str,
    base_url: str,
    *,
    label: str = "",
    models_url: str = "",
    api_key_env: str = "",
    reserved: frozenset[str] | None = None,
) -> str:
    """Validate one custom-provider form row; return an error message or ''."""
    from litellm_provider_config import (
        validate_custom_provider_env_name,
        validate_custom_provider_id,
        validate_custom_provider_url,
    )

    try:
        validate_custom_provider_id(provider_id, reserved=reserved)
        validate_custom_provider_url(base_url, field_name="API Base URL")
        if models_url.strip():
            validate_custom_provider_url(models_url, field_name="模型列表 URL")
        validate_custom_provider_env_name(api_key_env)
    except ValueError as exc:
        return str(exc)
    return ""
