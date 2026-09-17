"""Built-in OpenAI-compatible connection presets (issue #431 S1).

Presets only fill connection defaults (base URL, credential reference and the
declared structured-output mode).  They are not model allowlists: every preset
still lets the user type a model id manually, and the provider catalog is never
consulted before a request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from model_profile import ADAPTER_OPENAI_COMPATIBLE


@dataclass(frozen=True)
class OpenAICompatiblePreset:
    """One built-in provider preset for the direct adapter."""

    id: str
    label: str
    provider: str
    base_url: str
    models_url: str = ""
    credential_kind: str = "keyring"
    credential_name: str = ""
    credential_env_name: str = ""
    requires_key: bool = True
    structured_output_mode: str = "prompt_only_json"
    default_model: str = ""
    notes: str = ""


_PRESETS: tuple[OpenAICompatiblePreset, ...] = (
    OpenAICompatiblePreset(
        id="openai",
        label="OpenAI",
        provider="openai",
        base_url="https://api.openai.com/v1",
        models_url="https://api.openai.com/v1/models",
        credential_name="openai",
        credential_env_name="OPENAI_API_KEY",
        structured_output_mode="strict_json_schema",
    ),
    OpenAICompatiblePreset(
        id="openrouter",
        label="OpenRouter",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        models_url="https://openrouter.ai/api/v1/models",
        credential_name="openrouter",
        credential_env_name="OPENROUTER_API_KEY",
        structured_output_mode="json_object",
    ),
    OpenAICompatiblePreset(
        id="deepseek",
        label="DeepSeek",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        models_url="https://api.deepseek.com/v1/models",
        credential_name="deepseek",
        credential_env_name="DEEPSEEK_API_KEY",
        structured_output_mode="json_object",
    ),
    OpenAICompatiblePreset(
        id="xai",
        label="xAI",
        provider="xai",
        base_url="https://api.x.ai/v1",
        models_url="https://api.x.ai/v1/models",
        credential_name="xai",
        credential_env_name="XAI_API_KEY",
        structured_output_mode="json_object",
    ),
    OpenAICompatiblePreset(
        id="ollama",
        label="Ollama",
        provider="ollama",
        base_url="http://localhost:11434/v1",
        models_url="http://localhost:11434/v1/models",
        credential_kind="none",
        requires_key=False,
        structured_output_mode="json_object",
    ),
    OpenAICompatiblePreset(
        id="custom",
        label="自定义 OpenAI-compatible",
        provider="custom",
        base_url="",
        credential_kind="keyring",
        structured_output_mode="prompt_only_json",
        notes="手动填写 base URL、模型 ID 与非敏感额外请求头。",
    ),
)

PRESETS: Mapping[str, OpenAICompatiblePreset] = {preset.id: preset for preset in _PRESETS}


def preset_ids() -> tuple[str, ...]:
    """Return preset ids in display order."""

    return tuple(preset.id for preset in _PRESETS)


def get_preset(preset_id: str) -> OpenAICompatiblePreset:
    """Return one preset or raise ``KeyError`` for an unknown id."""

    key = str(preset_id or "").strip()
    return PRESETS[key]


def preset_provider_payload(preset_id: str) -> dict:
    """Return a config-shaped provider dict for one preset.

    The payload is suitable for ``model_profiles_editor.add_provider`` and the
    GUI provider form; it contains references only, never credential values.
    """

    preset = get_preset(preset_id)
    return {
        "label": preset.label,
        "adapter": ADAPTER_OPENAI_COMPATIBLE,
        "provider": preset.provider,
        "base_url": preset.base_url,
        "models_url": preset.models_url,
        "credential_ref": {
            "kind": preset.credential_kind,
            "name": preset.credential_name,
            "env_name": preset.credential_env_name,
        },
        "extra_headers": {},
    }
