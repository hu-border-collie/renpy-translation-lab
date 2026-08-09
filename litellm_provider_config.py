"""Provider catalog and secure credential helpers for the optional LiteLLM backend."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from importlib import metadata
import json
import re
from typing import Any, Literal
from urllib.parse import urlsplit


KEYRING_SERVICE = "renpy-translation-lab:litellm"
SUPPORTED_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("openrouter", "OpenRouter"),
    ("deepseek", "DeepSeek"),
    ("xai", "xAI"),
    ("ollama", "Ollama（本地）"),
)
SUPPORTED_PROVIDER_IDS: frozenset[str] = frozenset(
    provider for provider, _label in SUPPORTED_PROVIDERS
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
class CustomLiteLLMProvider:
    """One user-defined OpenAI-compatible provider from ``sync.custom_litellm_providers``.

    The id doubles as the display/selection model prefix (``opencode-go/<model>``)
    and as the keyring username.  Requests are rewritten to ``openai/<model>``
    plus ``api_base`` by the sync backend, so the id must never collide with a
    prefix LiteLLM already understands.
    """

    id: str
    label: str
    base_url: str
    models_url: str
    api_key_env: str = ""
    requires_key: bool = True


_CUSTOM_PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def reserved_litellm_provider_ids(litellm_module: Any = None) -> frozenset[str]:
    """Return provider prefixes that custom ids must not shadow.

    The static reserve covers built-in choices and common LiteLLM prefixes;
    callers that already hold a litellm module (tests, optional integrations)
    may pass it to merge in the installed provider table. Importing litellm is
    deliberately avoided here: it is a heavy optional dependency and this
    function runs on GUI startup and config validation.
    """
    reserved = set(_RESERVED_PROVIDER_IDS)
    if litellm_module is not None:
        by_provider = getattr(litellm_module, "models_by_provider", {})
        if isinstance(by_provider, Mapping):
            for provider in by_provider:
                provider = str(provider or "").strip().lower()
                if provider:
                    reserved.add(provider)
    return frozenset(reserved)


def validate_custom_provider_url(value: object, *, field_name: str) -> str:
    """Require a clean http(s) URL; return the trimmed value.

    Rejects embedded credentials (``user:pass@``) and query/fragment strings:
    credentials must never land in the plaintext config, and a query/fragment
    would produce a malformed ``api_base`` for the request rewrite.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name}不能为空。")
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name}必须是 http(s) URL。")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name}不能包含用户名或密码。")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name}不能包含查询参数或片段。")
    return text


def validate_custom_provider_id(
    value: object,
    *,
    reserved: frozenset[str] | None = None,
) -> str:
    """Validate a custom provider id (lowercase, ``[a-z0-9_-]+``, no shadowing)."""
    provider = str(value or "").strip().lower()
    if not provider:
        raise ValueError("自定义 Provider id 不能为空。")
    if not _CUSTOM_PROVIDER_ID_PATTERN.fullmatch(provider):
        raise ValueError("自定义 Provider id 只能包含小写字母、数字、- 和 _。")
    reserved_ids = reserved_litellm_provider_ids() if reserved is None else reserved
    if provider in reserved_ids:
        raise ValueError(
            f"自定义 Provider id 与 LiteLLM 已知前缀冲突：{provider}。"
            "请更换一个不会覆盖内置 provider 的 id。"
        )
    return provider


def validate_custom_provider_env_name(value: object) -> str:
    """Validate an optional environment-variable name used as key fallback."""
    text = str(value or "").strip()
    if not text:
        return ""
    if not _ENVIRONMENT_NAME_PATTERN.fullmatch(text):
        raise ValueError("api_key_env 必须是合法的环境变量名（字母/数字/下划线）。")
    return text


def custom_provider_from_mapping(
    entry: Mapping[str, object],
    *,
    index: int = 0,
    reserved: frozenset[str] | None = None,
) -> CustomLiteLLMProvider:
    """Parse and validate one ``custom_litellm_providers`` config entry."""
    if not isinstance(entry, Mapping):
        raise ValueError(f"custom_litellm_providers[{index}] 必须是对象。")
    provider_id = validate_custom_provider_id(entry.get("id"), reserved=reserved)
    label = str(entry.get("label") or "").strip() or provider_id
    base_url = validate_custom_provider_url(
        entry.get("base_url"),
        field_name="base_url",
    )
    raw_models_url = str(entry.get("models_url") or "").strip()
    if raw_models_url:
        models_url = validate_custom_provider_url(
            raw_models_url,
            field_name="models_url",
        )
    else:
        models_url = base_url.rstrip("/") + "/models"
    api_key_env = validate_custom_provider_env_name(entry.get("api_key_env"))
    requires_key = entry.get("requires_key", True)
    if not isinstance(requires_key, bool):
        raise ValueError(f"custom_litellm_providers[{index}].requires_key 必须是布尔值。")
    return CustomLiteLLMProvider(
        id=provider_id,
        label=label,
        base_url=base_url,
        models_url=models_url,
        api_key_env=api_key_env,
        requires_key=requires_key,
    )


def parse_custom_litellm_providers(
    raw: object,
    *,
    litellm_module: Any = None,
) -> tuple[CustomLiteLLMProvider, ...]:
    """Parse ``sync.custom_litellm_providers`` into validated provider entries.

    A missing value (``None``) is treated as an empty registry so older configs
    keep working; any other non-list value raises ValueError. Raises ValueError
    on structural errors, invalid ids/URLs, non-boolean ``requires_key`` or
    duplicate ids. When *litellm_module* is omitted but the optional litellm
    dependency is installed, its provider table is merged into the reserved-id
    check (the import result is cached per process).
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("custom_litellm_providers 必须是列表。")
    if litellm_module is None:
        litellm_module = _installed_litellm_module()
    reserved = reserved_litellm_provider_ids(litellm_module)
    providers: list[CustomLiteLLMProvider] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        provider = custom_provider_from_mapping(
            entry,
            index=index,
            reserved=reserved,
        )
        if provider.id in seen:
            raise ValueError(f"自定义 Provider id 重复：{provider.id}。")
        seen.add(provider.id)
        providers.append(provider)
    return tuple(providers)


def custom_provider_registry(
    raw: object,
    *,
    litellm_module: Any = None,
) -> dict[str, CustomLiteLLMProvider]:
    """Build the id → provider lookup used by backend/GUI/CLI paths."""
    return {
        provider.id: provider
        for provider in parse_custom_litellm_providers(raw, litellm_module=litellm_module)
    }


_INSTALLED_LITELLM_MODULE: Any = None
_INSTALLED_LITELLM_PROBED = False


def _installed_litellm_module() -> Any:
    """Return the installed litellm module, or None (probed once per process)."""
    global _INSTALLED_LITELLM_MODULE, _INSTALLED_LITELLM_PROBED
    if _INSTALLED_LITELLM_PROBED:
        return _INSTALLED_LITELLM_MODULE
    _INSTALLED_LITELLM_PROBED = True
    try:
        from importlib.util import find_spec

        if find_spec("litellm") is None:
            return None
        import litellm as litellm_module
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    _INSTALLED_LITELLM_MODULE = litellm_module
    return litellm_module


def _custom_registry_lookup(
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None,
    provider: str,
) -> CustomLiteLLMProvider | None:
    if not isinstance(custom_providers, Mapping):
        return None
    return custom_providers.get(provider)


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

# Static reserve of provider prefixes that must never be shadowed by a custom
# id.  When litellm is installed its own provider table is merged in too, so
# the runtime check also covers providers LiteLLM knows but this list does not.
_RESERVED_PROVIDER_IDS: frozenset[str] = frozenset(
    {
        *SUPPORTED_PROVIDER_IDS,
        *_COMMON_PROVIDER_ORDER,
        *NATIVE_CATALOG_ENDPOINTS,
        # Frequent LiteLLM prefixes that may appear in the online catalog.
        "aliyun",
        "baidu",
        "bedrock",
        "cerebras",
        "cohere",
        "databricks",
        "deepinfra",
        "dashscope",
        "ernie",
        "fireworks_ai",
        "glm",
        "groq",
        "huggingface",
        "kimi",
        "lmstudio",
        "localai",
        "minimax",
        "mistral",
        "moonshot",
        "novita",
        "nvidia_nim",
        "perplexity",
        "predibase",
        "qwen",
        "replicate",
        "sagemaker",
        "spark",
        "stepfun",
        "tencent",
        "text-generation-webui",
        "together_ai",
        "vllm",
        "volcengine",
        "watsonx",
        "yi",
        "zhipu",
    }
)


class ProviderCredentialStoreError(RuntimeError):
    """The operating-system credential store could not be used."""


def provider_from_model(model: str) -> str:
    text = str(model or "").strip()
    if "/" not in text:
        return ""
    return text.split("/", 1)[0].strip().lower()


def provider_display_label(
    provider: str,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
) -> str:
    """Return a friendly label without restricting dynamic provider ids."""
    provider = str(provider or "").strip().lower()
    label = _PROVIDER_LABELS.get(provider)
    if label:
        return label
    custom = _custom_registry_lookup(custom_providers, provider)
    if custom is not None:
        return custom.label
    return provider


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


def native_catalog_endpoint(
    provider: str,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
) -> NativeCatalogEndpoint | None:
    """Return the model-list endpoint for *provider*.

    Built-in providers use the official ``NATIVE_CATALOG_ENDPOINTS`` table;
    custom OpenAI-compatible providers synthesize an OpenAI-style endpoint from
    their configured ``models_url`` so catalog refresh, key gating and error
    copy are shared with built-in providers.
    """
    provider = str(provider or "").strip().lower()
    endpoint = NATIVE_CATALOG_ENDPOINTS.get(provider)
    if endpoint is not None:
        return endpoint
    custom = _custom_registry_lookup(custom_providers, provider)
    if custom is not None:
        return NativeCatalogEndpoint(
            provider=provider,
            url=custom.models_url,
            label=custom.label,
            source=provider,
            auth="bearer",
            require_key=custom.requires_key,
            payload_style="openai",
        )
    return None


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


def catalog_source_label(
    source: str,
    custom_providers: Mapping[str, CustomLiteLLMProvider] | None = None,
) -> str:
    """Human-readable status line for a catalog ``source`` token."""
    token = str(source or "").strip().lower()
    if token == "online":
        return "目录来源：LiteLLM 官方在线目录。"
    endpoint = NATIVE_CATALOG_ENDPOINTS.get(token)
    if endpoint is not None:
        if token == "ollama":
            return f"目录来源：{endpoint.label} 本机已安装模型。"
        return f"目录来源：{endpoint.label} 官方模型列表。"
    custom = _custom_registry_lookup(custom_providers, token)
    if custom is not None:
        return f"目录来源：{custom.label} 官方模型列表。"
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
