"""Shared parsing for settings with legacy runtime representations."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LegacyTranslationLimits = Mapping[tuple[str, ...], int]


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


def legacy_translation_limits(config: Mapping[str, Any]) -> dict[tuple[str, ...], int]:
    """Project the effective positive limits onto translator_config.json paths."""
    limits: dict[tuple[str, ...], int] = {}
    for section, values in (("sync", legacy_sync_limits(config)), ("batch", legacy_batch_limits(config))):
        for key, raw in values.items():
            try:
                limit = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if limit > 0:
                limits[(section, key)] = limit
    return limits
