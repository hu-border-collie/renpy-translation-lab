"""Provider catalog and secure credential helpers for the optional LiteLLM backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
import re
from typing import Any, Literal


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
# Skip non-chat OpenAI-style model ids returned by /v1/models.
_OPENAI_STYLE_SKIP_FRAGMENTS = (
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "moderation",
    "realtime",
    "transcribe",
    "audio",
    "image",
    "sora",
    "babbage",
    "davinci",
    "chatgpt-image",
)
LITELLM_CATALOG_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
LITELLM_PYPI_URL = "https://pypi.org/pypi/litellm/json"
# OpenRouter hosts far more models than LiteLLM's pricing table covers.
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

AuthStyle = Literal["none", "bearer", "x-api-key"]
CatalogPayloadStyle = Literal["openai", "openrouter", "ollama"]


@dataclass(frozen=True)
class NativeCatalogEndpoint:
    """Official (or local) model-list endpoint for a LiteLLM provider prefix."""

    provider: str
    url: str
    label: str
    source: str
    auth: AuthStyle = "bearer"
    require_key: bool = True
    payload_style: CatalogPayloadStyle = "openai"
    extra_headers: tuple[tuple[str, str], ...] = ()


# Prefer these over LiteLLM's pricing table when refreshing the GUI model list.
NATIVE_CATALOG_ENDPOINTS: dict[str, NativeCatalogEndpoint] = {
    "openai": NativeCatalogEndpoint(
        provider="openai",
        url="https://api.openai.com/v1/models",
        label="OpenAI",
        source="openai",
        auth="bearer",
        require_key=True,
        payload_style="openai",
    ),
    "anthropic": NativeCatalogEndpoint(
        provider="anthropic",
        url="https://api.anthropic.com/v1/models",
        label="Anthropic",
        source="anthropic",
        auth="x-api-key",
        require_key=True,
        payload_style="openai",
        extra_headers=(("anthropic-version", "2023-06-01"),),
    ),
    "openrouter": NativeCatalogEndpoint(
        provider="openrouter",
        url=OPENROUTER_MODELS_URL,
        label="OpenRouter",
        source="openrouter",
        auth="bearer",
        require_key=False,
        payload_style="openrouter",
    ),
    "deepseek": NativeCatalogEndpoint(
        provider="deepseek",
        url="https://api.deepseek.com/models",
        label="DeepSeek",
        source="deepseek",
        auth="bearer",
        require_key=True,
        payload_style="openai",
    ),
    "xai": NativeCatalogEndpoint(
        provider="xai",
        url="https://api.x.ai/v1/models",
        label="xAI",
        source="xai",
        auth="bearer",
        require_key=True,
        payload_style="openai",
    ),
    "ollama": NativeCatalogEndpoint(
        provider="ollama",
        url="http://127.0.0.1:11434/api/tags",
        label="Ollama",
        source="ollama",
        auth="none",
        require_key=False,
        payload_style="ollama",
    ),
}


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
    models: set[str] = set()
    for raw_model in raw_models:
        raw_model = str(raw_model or "").strip()
        metadata = cost.get(raw_model, {}) if isinstance(cost, Mapping) else {}
        mode = (
            str(metadata.get("mode") or "chat").strip().lower()
            if isinstance(metadata, Mapping)
            else "chat"
        )
        if mode not in _TEXT_MODES:
            continue
        if not raw_model:
            continue
        model = raw_model if raw_model.startswith(f"{provider}/") else f"{provider}/{raw_model}"
        models.add(model)
    return tuple(sorted(models or set(defaults), key=str.casefold))


def models_from_remote_catalog(
    provider: str,
    catalog: Mapping[str, object],
) -> tuple[str, ...]:
    """Return provider text models from LiteLLM's current upstream catalog."""
    provider = str(provider or "").strip().lower()
    if not provider:
        return ()
    models: set[str] = set()
    for raw_model, raw_metadata in catalog.items():
        model = str(raw_model or "").strip()
        if not model or not isinstance(raw_metadata, Mapping):
            continue
        catalog_provider = str(raw_metadata.get("litellm_provider") or "").strip().lower()
        model_provider = provider_from_model(model)
        if catalog_provider != provider and model_provider != provider:
            continue
        mode = str(raw_metadata.get("mode") or "chat").strip().lower()
        if mode not in _TEXT_MODES:
            continue
        models.add(model if model_provider == provider else f"{provider}/{model}")
    return tuple(sorted(models, key=str.casefold))


def models_from_openrouter_payload(payload: Mapping[str, object] | object) -> tuple[str, ...]:
    """Parse OpenRouter ``GET /api/v1/models`` into LiteLLM model ids.

    LiteLLM expects ``openrouter/<vendor>/<model>`` (OpenRouter ids are already
    ``vendor/model``). Alias/router ids that start with ``~`` are skipped.
    """
    if not isinstance(payload, Mapping):
        return ()
    raw_data = payload.get("data")
    if not isinstance(raw_data, list):
        return ()

    models: set[str] = set()
    for item in raw_data:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id or model_id.startswith("~"):
            continue

        architecture = item.get("architecture")
        if isinstance(architecture, Mapping):
            outputs = architecture.get("output_modalities")
            if isinstance(outputs, list) and outputs:
                normalized = {
                    str(part).strip().lower() for part in outputs if str(part).strip()
                }
                if "text" not in normalized:
                    continue

        if model_id.startswith("openrouter/"):
            models.add(model_id)
        else:
            models.add(f"openrouter/{model_id}")
    return tuple(sorted(models, key=str.casefold))


def _looks_like_non_text_openai_model(model_id: str) -> bool:
    lowered = str(model_id or "").strip().lower()
    if not lowered:
        return True
    return any(fragment in lowered for fragment in _OPENAI_STYLE_SKIP_FRAGMENTS)


def models_from_openai_compatible_payload(
    provider: str,
    payload: Mapping[str, object] | object,
) -> tuple[str, ...]:
    """Parse OpenAI-style ``{data:[{id:...}]}`` catalogs into LiteLLM model ids."""
    provider = str(provider or "").strip().lower()
    if not provider or not isinstance(payload, Mapping):
        return ()
    raw_data = payload.get("data")
    if not isinstance(raw_data, list):
        return ()

    models: set[str] = set()
    for item in raw_data:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id or _looks_like_non_text_openai_model(model_id):
            continue
        if model_id.startswith(f"{provider}/"):
            models.add(model_id)
        else:
            models.add(f"{provider}/{model_id}")
    return tuple(sorted(models, key=str.casefold))


def models_from_ollama_payload(payload: Mapping[str, object] | object) -> tuple[str, ...]:
    """Parse Ollama ``GET /api/tags`` into LiteLLM ``ollama/<name>`` ids."""
    if not isinstance(payload, Mapping):
        return ()
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return ()

    models: set[str] = set()
    for item in raw_models:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if not name:
            continue
        if name.startswith("ollama/"):
            models.add(name)
        else:
            models.add(f"ollama/{name}")
    return tuple(sorted(models, key=str.casefold))


def native_catalog_endpoint(provider: str) -> NativeCatalogEndpoint | None:
    return NATIVE_CATALOG_ENDPOINTS.get(str(provider or "").strip().lower())


def build_native_catalog_headers(
    endpoint: NativeCatalogEndpoint,
    api_key: str = "",
) -> dict[str, str]:
    headers = {
        "User-Agent": "renpy-translation-lab",
        "Accept": "application/json",
    }
    for key, value in endpoint.extra_headers:
        headers[str(key)] = str(value)
    key = str(api_key or "").strip()
    if endpoint.auth == "bearer" and key:
        headers["Authorization"] = f"Bearer {key}"
    elif endpoint.auth == "x-api-key" and key:
        headers["x-api-key"] = key
    return headers


def models_from_native_catalog_payload(
    endpoint: NativeCatalogEndpoint,
    payload: Mapping[str, object] | object,
) -> tuple[str, ...]:
    if endpoint.payload_style == "openrouter":
        return models_from_openrouter_payload(payload)
    if endpoint.payload_style == "ollama":
        return models_from_ollama_payload(payload)
    return models_from_openai_compatible_payload(endpoint.provider, payload)


def catalog_source_label(source: str) -> str:
    """Human-readable status line for a catalog ``source`` token."""
    token = str(source or "").strip().lower()
    if token == "online":
        return "目录来源：LiteLLM 官方在线目录。"
    if token == "local":
        return "目录来源：本机 LiteLLM 随包目录（联网失败，可能过时）。"
    endpoint = NATIVE_CATALOG_ENDPOINTS.get(token)
    if endpoint is not None:
        if token == "ollama":
            return f"目录来源：{endpoint.label} 本机已安装模型。"
        return f"目录来源：{endpoint.label} 官方模型列表。"
    return "目录来源：未知。"


def installed_litellm_version() -> str:
    try:
        return metadata.version("litellm")
    except metadata.PackageNotFoundError:
        return ""


def version_key(value: str) -> tuple[int, ...]:
    """Build a sufficient comparison key for stable LiteLLM release versions."""
    release = str(value or "").strip().split("+", 1)[0]
    numbers: list[int] = []
    for part in release.split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


_STABLE_VERSION_PATTERN = re.compile(r"^\d+(?:\.\d+)*$")
_PYTHON_SPECIFIER_PATTERN = re.compile(r"^(<=|>=|==|!=|<|>)(\d+(?:\.\d+)*)$")


def python_requirement_allows(
    requirement: str,
    python_version: tuple[int, ...],
) -> bool:
    """Evaluate the simple Requires-Python bounds used by LiteLLM releases."""
    requirement = str(requirement or "").strip()
    if not requirement:
        return True
    current = tuple(int(part) for part in python_version)
    for raw_specifier in requirement.split(","):
        specifier = raw_specifier.strip().replace(" ", "")
        match = _PYTHON_SPECIFIER_PATTERN.fullmatch(specifier)
        if match is None:
            return False
        operator, raw_version = match.groups()
        expected = tuple(int(part) for part in raw_version.split("."))
        width = max(len(current), len(expected))
        left = current + (0,) * (width - len(current))
        right = expected + (0,) * (width - len(expected))
        allowed = {
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
            "==": left == right,
            "!=": left != right,
        }[operator]
        if not allowed:
            return False
    return True


def latest_compatible_litellm_version(
    releases: Mapping[str, object],
    python_version: tuple[int, ...],
) -> str:
    """Return the latest stable, non-yanked release compatible with Python."""
    compatible: list[str] = []
    for raw_version, raw_files in releases.items():
        version = str(raw_version or "").strip()
        if not _STABLE_VERSION_PATTERN.fullmatch(version):
            continue
        if not isinstance(raw_files, list):
            continue
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping) or raw_file.get("yanked"):
                continue
            requires_python = str(raw_file.get("requires_python") or "")
            if python_requirement_allows(requires_python, python_version):
                compatible.append(version)
                break
    if not compatible:
        return ""
    return max(compatible, key=version_key)
