"""Shared parsing for settings with legacy runtime representations."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LegacyTranslationDefaults = Mapping[tuple[str, ...], int | list[dict[str, str]]]


def parse_config_bool(value: object) -> bool | None:
    """Parse a JSON boolean or a legacy string accepted by the runtime."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return None


def legacy_sync_limits(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read raw Sync limit overrides from api_keys.json in runtime precedence."""
    max_chars = config.get("sync_max_source_chars")
    if max_chars is None:
        max_chars = config.get("max_chars")
    return {
        "chunk_size": config.get("sync_chunk_size", config.get("batch_size")),
        "max_source_chars": max_chars,
        "max_output_tokens": config.get("sync_max_output_tokens"),
    }


def legacy_batch_limits(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read raw Batch limit overrides from api_keys.json in runtime precedence."""
    return {
        "chunk_size": config.get("batch_target_size", config.get("batch_size")),
        "max_source_chars": config.get("batch_target_chars", config.get("batch_max_source_chars")),
        "max_output_tokens": config.get("batch_max_output_tokens"),
        "context_before": config.get("batch_context_before"),
        "context_after": config.get("batch_context_after"),
    }


def normalize_batch_safety_settings(value: object) -> list[dict[str, str]]:
    """Resolve the Batch safety preset or individual settings from either config file."""
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        normalized = cleaned.lower().replace('-', '_')
        if normalized in {'relaxed_adult', 'adult', 'sexually_explicit_block_none'}:
            return [
                {
                    'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    'threshold': 'BLOCK_NONE',
                }
            ]
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    settings = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        category = str(entry.get('category') or entry.get('harm_category') or '').strip().upper()
        threshold = str(entry.get('threshold') or entry.get('block_threshold') or '').strip().upper()
        if not category or not threshold:
            continue
        if not category.startswith('HARM_CATEGORY_'):
            category = f'HARM_CATEGORY_{category}'
        if threshold in {'NONE', 'NO_BLOCK', 'BLOCKNONE'}:
            threshold = 'BLOCK_NONE'
        settings.append({'category': category, 'threshold': threshold})
    return settings


def legacy_translation_defaults(config: Mapping[str, Any]) -> LegacyTranslationDefaults:
    """Project inherited limits and Batch safety settings onto translator_config.json paths."""
    defaults: dict[tuple[str, ...], int | list[dict[str, str]]] = {}
    for section, values in (("sync", legacy_sync_limits(config)), ("batch", legacy_batch_limits(config))):
        for key, raw in values.items():
            try:
                limit = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if limit > 0:
                defaults[(section, key)] = limit
    safety_settings = normalize_batch_safety_settings(config.get("batch_safety_settings"))
    if safety_settings:
        defaults[("batch", "safety_settings")] = safety_settings
    return defaults
