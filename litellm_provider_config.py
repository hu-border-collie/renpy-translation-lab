"""Provider catalog and secure credential helpers for the optional LiteLLM backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


KEYRING_SERVICE = "renpy-translation-lab:litellm"
SUPPORTED_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("openrouter", "OpenRouter"),
    ("deepseek", "DeepSeek"),
    ("xai", "xAI"),
    ("ollama", "Ollama（本地）"),
)
DEFAULT_MODELS: dict[str, tuple[str, ...]] = {
    "openai": ("openai/gpt-5",),
    "anthropic": ("anthropic/claude-sonnet-4-5-20250929",),
    "openrouter": ("openrouter/openai/gpt-5",),
    "deepseek": ("deepseek/deepseek-chat",),
    "xai": ("xai/grok-2-latest",),
    "ollama": ("ollama/llama3",),
}
_TEXT_MODES = frozenset({"chat", "responses", "completion"})


class ProviderCredentialStoreError(RuntimeError):
    """The operating-system credential store could not be used."""


def provider_from_model(model: str) -> str:
    text = str(model or "").strip()
    if "/" not in text:
        return ""
    return text.split("/", 1)[0].strip().lower()


def _keyring(keyring_module: Any = None) -> Any:
    if keyring_module is not None:
        return keyring_module
    try:
        import keyring
    except ImportError as exc:
        raise ProviderCredentialStoreError(
            "安全凭据支持尚未安装，请安装 LiteLLM 可选依赖。"
        ) from exc
    return keyring


def load_provider_api_key(provider: str, keyring_module: Any = None) -> str:
    provider = str(provider or "").strip().lower()
    if not provider or provider == "ollama":
        return ""
    try:
        value = _keyring(keyring_module).get_password(KEYRING_SERVICE, provider)
    except Exception as exc:
        if isinstance(exc, ProviderCredentialStoreError):
            raise
        raise ProviderCredentialStoreError("无法读取系统凭据管理器。") from exc
    return value.strip() if isinstance(value, str) else ""


def store_provider_api_key(provider: str, api_key: str, keyring_module: Any = None) -> None:
    provider = str(provider or "").strip().lower()
    api_key = str(api_key or "").strip()
    if not provider or provider == "ollama":
        raise ValueError("该 provider 不需要保存 API Key。")
    if not api_key:
        raise ValueError("API Key 不能为空。")
    try:
        _keyring(keyring_module).set_password(KEYRING_SERVICE, provider, api_key)
    except Exception as exc:
        if isinstance(exc, ProviderCredentialStoreError):
            raise
        raise ProviderCredentialStoreError("无法写入系统凭据管理器。") from exc


def delete_provider_api_key(provider: str, keyring_module: Any = None) -> bool:
    provider = str(provider or "").strip().lower()
    if not provider or provider == "ollama":
        return False
    store = _keyring(keyring_module)
    try:
        if not store.get_password(KEYRING_SERVICE, provider):
            return False
        store.delete_password(KEYRING_SERVICE, provider)
    except Exception as exc:
        if isinstance(exc, ProviderCredentialStoreError):
            raise
        raise ProviderCredentialStoreError("无法删除系统凭据管理器中的密钥。") from exc
    return True


def models_for_provider(provider: str, litellm_module: Any = None) -> tuple[str, ...]:
    """Return text-generation models from LiteLLM's installed model catalog."""
    provider = str(provider or "").strip().lower()
    defaults = DEFAULT_MODELS.get(provider, ())
    if not provider:
        return defaults
    if litellm_module is None:
        import litellm as litellm_module

    by_provider = getattr(litellm_module, "models_by_provider", {})
    cost = getattr(litellm_module, "model_cost", {})
    raw_models = by_provider.get(provider, ()) if isinstance(by_provider, Mapping) else ()
    models: set[str] = set(defaults)
    for raw_model in raw_models:
        raw_model = str(raw_model or "").strip()
        metadata = cost.get(raw_model, {}) if isinstance(cost, Mapping) else {}
        if isinstance(metadata, Mapping) and metadata.get("mode") not in _TEXT_MODES:
            continue
        if not raw_model:
            continue
        model = raw_model if raw_model.startswith(f"{provider}/") else f"{provider}/{raw_model}"
        models.add(model)
    return tuple(sorted(models, key=str.casefold))
