"""Pure helpers for the GUI's provider-aware LiteLLM settings page."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


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


def provider_from_model(model: str) -> str:
    text = str(model or "").strip()
    if "/" not in text:
        return ""
    return text.split("/", 1)[0].strip().lower()


def provider_credential_status(
    model: str,
    environment: Mapping[str, str],
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
    if provider == "vertex_ai":
        names = ("VERTEXAI_PROJECT", "VERTEXAI_LOCATION")
        configured = all(str(environment.get(name, "")).strip() for name in names)
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
    configured = all(str(environment.get(name, "")).strip() for name in names)
    state = "已检测到" if configured else "未检测到完整"
    return ProviderCredentialStatus(
        provider=provider,
        environment_names=names,
        configured=configured,
        message=f"{state}环境变量：{', '.join(names)}。",
    )
