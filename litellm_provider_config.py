"""Provider catalog and secure credential helpers for the optional LiteLLM backend."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from importlib import metadata
import json
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
_COMMON_PROVIDER_ORDER = (
    "openai",
    "anthropic",
    "gemini",
    "openrouter",
    "deepseek",
    "xai",
    "azure",
    "vertex_ai",
    "ollama",
)
_COMMON_PROVIDER_INDEX = {
    provider: index for index, provider in enumerate(_COMMON_PROVIDER_ORDER)
}
_PROVIDER_LABELS = dict(SUPPORTED_PROVIDERS) | {
    "gemini": "Google Gemini",
    "azure": "Azure OpenAI",
    "vertex_ai": "Google Vertex AI",
}
_LABEL_TO_PROVIDER_ID = {
    str(label).strip().casefold(): provider
    for provider, label in _PROVIDER_LABELS.items()
}
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


def provider_display_label(provider: str) -> str:
    """Return a friendly label without restricting dynamic provider ids."""
    provider = str(provider or "").strip().lower()
    return _PROVIDER_LABELS.get(provider, provider)


def resolve_provider_id(value: str) -> str:
    """Map free-typed provider input to a LiteLLM provider id when possible.

    Accepts known ids, known display labels (e.g. ``Ollama（本地）`` → ``ollama``),
    and otherwise returns the lowercased free-text id for custom providers.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in _PROVIDER_LABELS or lowered in _COMMON_PROVIDER_INDEX:
        return lowered
    mapped = _LABEL_TO_PROVIDER_ID.get(text.casefold())
    if mapped:
        return mapped
    return lowered


def sort_provider_ids(providers: object) -> tuple[str, ...]:
    """Place common providers first and sort all remaining ids by name."""
    if isinstance(providers, (str, bytes, bytearray)) or not isinstance(
        providers, Collection
    ):
        return ()
    cleaned = {
        str(provider or "").strip().lower()
        for provider in providers
        if str(provider or "").strip()
    }
    return tuple(
        sorted(
            cleaned,
            key=lambda provider: (
                _COMMON_PROVIDER_INDEX.get(provider, len(_COMMON_PROVIDER_INDEX)),
                provider.casefold(),
            ),
        )
    )


def credential_provider_candidates(
    *extra_groups: object,
    include_ollama: bool = False,
) -> tuple[str, ...]:
    """Providers shown when managing LiteLLM keys.

    Always includes the built-in common set (optionally without Ollama), then any
    extra ids (e.g. online catalog cache, current selection).
    """
    providers: set[str] = {
        provider
        for provider in _COMMON_PROVIDER_ORDER
        if include_ollama or provider != "ollama"
    }
    for group in extra_groups:
        if isinstance(group, (str, bytes, bytearray)) or not isinstance(
            group, Collection
        ):
            continue
        for raw in group:
            provider = str(raw or "").strip().lower()
            if not provider:
                continue
            if provider == "ollama" and not include_ollama:
                continue
            providers.add(provider)
    return sort_provider_ids(providers)


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


@dataclass(frozen=True)
class ProviderApiKeyStore:
    """OS keyring payload for one LiteLLM provider (multi-key + active index)."""

    keys: tuple[str, ...] = ()
    active_index: int = 0

    def normalized(self) -> "ProviderApiKeyStore":
        cleaned = tuple(str(key).strip() for key in self.keys if str(key).strip())
        if not cleaned:
            return ProviderApiKeyStore()
        index = int(self.active_index)
        if index < 0 or index >= len(cleaned):
            index = 0
        return ProviderApiKeyStore(keys=cleaned, active_index=index)

    def active_key(self) -> str:
        store = self.normalized()
        if not store.keys:
            return ""
        return store.keys[store.active_index]


def _encode_provider_key_store(store: ProviderApiKeyStore) -> str:
    normalized = store.normalized()
    if not normalized.keys:
        return ""
    # Single-key legacy plaintext keeps older readers working until next multi-key save.
    if len(normalized.keys) == 1 and normalized.active_index == 0:
        return normalized.keys[0]
    payload = {
        "version": 1,
        "keys": list(normalized.keys),
        "active_index": normalized.active_index,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _decode_provider_key_store(raw: str | None) -> ProviderApiKeyStore:
    text = str(raw or "").strip()
    if not text:
        return ProviderApiKeyStore()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Treat as a single opaque secret that happens to start with '{'.
            return ProviderApiKeyStore(keys=(text,), active_index=0)
        if not isinstance(payload, dict):
            return ProviderApiKeyStore(keys=(text,), active_index=0)
        keys_raw = payload.get("keys", [])
        if not isinstance(keys_raw, list):
            # Unrecognized object shape: keep the raw secret opaque.
            return ProviderApiKeyStore(keys=(text,), active_index=0)
        keys = tuple(str(item).strip() for item in keys_raw if str(item).strip())
        if not keys:
            return ProviderApiKeyStore(keys=(text,), active_index=0)
        try:
            active_index = int(payload.get("active_index", 0) or 0)
        except (TypeError, ValueError):
            active_index = 0
        return ProviderApiKeyStore(keys=keys, active_index=active_index).normalized()
    return ProviderApiKeyStore(keys=(text,), active_index=0)


def load_provider_key_store(
    provider: str,
    keyring_module: Any = None,
) -> ProviderApiKeyStore:
    """Load all saved API keys for *provider* from the OS credential store."""
    provider = str(provider or "").strip().lower()
    if not provider or provider == "ollama":
        return ProviderApiKeyStore()
    try:
        value = _keyring(keyring_module).get_password(KEYRING_SERVICE, provider)
    except Exception as exc:
        if isinstance(exc, ProviderCredentialStoreError):
            raise
        raise ProviderCredentialStoreError("无法读取系统凭据管理器。") from exc
    return _decode_provider_key_store(value if isinstance(value, str) else "")


def store_provider_key_store(
    provider: str,
    store: ProviderApiKeyStore | Mapping[str, object] | Collection[str],
    keyring_module: Any = None,
    *,
    active_index: int | None = None,
) -> ProviderApiKeyStore:
    """Replace the saved key list for *provider* (empty list deletes the entry)."""
    provider = str(provider or "").strip().lower()
    if not provider or provider == "ollama":
        raise ValueError("该 provider 不需要保存 API Key。")

    if isinstance(store, ProviderApiKeyStore):
        payload = store
    elif isinstance(store, Mapping):
        keys_raw = store.get("keys", ())
        keys = tuple(str(item).strip() for item in keys_raw if str(item).strip())  # type: ignore[union-attr]
        try:
            idx = int(store.get("active_index", 0) or 0)
        except (TypeError, ValueError):
            idx = 0
        payload = ProviderApiKeyStore(keys=keys, active_index=idx)
    else:
        keys = tuple(str(item).strip() for item in store if str(item).strip())
        payload = ProviderApiKeyStore(keys=keys)
    # Explicit keyword wins for every input type (store / mapping / collection).
    if active_index is not None:
        payload = ProviderApiKeyStore(keys=payload.keys, active_index=int(active_index))
    payload = payload.normalized()

    ring = _keyring(keyring_module)
    try:
        existing = ring.get_password(KEYRING_SERVICE, provider)
        if not payload.keys:
            if existing:
                ring.delete_password(KEYRING_SERVICE, provider)
            return ProviderApiKeyStore()
        ring.set_password(KEYRING_SERVICE, provider, _encode_provider_key_store(payload))
    except Exception as exc:
        if isinstance(exc, ProviderCredentialStoreError):
            raise
        raise ProviderCredentialStoreError("无法写入系统凭据管理器。") from exc
    return payload


def load_provider_api_keys(provider: str, keyring_module: Any = None) -> tuple[str, ...]:
    """Return all saved keys for *provider* (order preserved)."""
    return load_provider_key_store(provider, keyring_module).keys


def load_provider_api_key(provider: str, keyring_module: Any = None) -> str:
    """Return the active saved key for *provider* (empty when none)."""
    return load_provider_key_store(provider, keyring_module).active_key()


def store_provider_api_key(provider: str, api_key: str, keyring_module: Any = None) -> None:
    """Replace *provider* credentials with a single active key (legacy helper)."""
    api_key = str(api_key or "").strip()
    if not api_key:
        raise ValueError("API Key 不能为空。")
    store_provider_key_store(
        provider,
        ProviderApiKeyStore(keys=(api_key,), active_index=0),
        keyring_module,
    )


def delete_provider_api_key(provider: str, keyring_module: Any = None) -> bool:
    """Delete every saved key for *provider*. Returns True when something was removed."""
    provider = str(provider or "").strip().lower()
    if not provider or provider == "ollama":
        return False
    existing = load_provider_key_store(provider, keyring_module)
    if not existing.keys:
        return False
    store_provider_key_store(provider, ProviderApiKeyStore(), keyring_module)
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


def providers_from_remote_catalog(
    catalog: Mapping[str, object],
) -> tuple[str, ...]:
    """Discover provider ids represented by LiteLLM's current online catalog.

    Native endpoints are included even when the pricing/context catalog has no
    current entry for them. The result is a discovery aid, not a guarantee
    that every provider exposes a native model-list endpoint.
    """
    providers = set(NATIVE_CATALOG_ENDPOINTS)
    for raw_model, raw_metadata in catalog.items():
        if not isinstance(raw_metadata, Mapping):
            continue
        provider = str(raw_metadata.get("litellm_provider") or "").strip().lower()
        if not provider:
            provider = provider_from_model(str(raw_model or ""))
        if provider:
            providers.add(provider)
    return sort_provider_ids(providers)


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
