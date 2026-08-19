"""Opaque digests that a worker and the window can compute independently."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import PurePath

from project_context_settings import default_context_flags_from_config


def _canonical_value(value: object) -> object:
    """Normalize one snapshot value into sorted, JSON-serializable form."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, PurePath):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, default=repr),
        )
    return repr(value)


def canonical_digest(value: object) -> str:
    """Return a stable sha256 digest for a config snapshot or identity tuple.

    Dict key order and tuple/list distinctions never change the digest; any
    semantic difference in the underlying snapshot does.  Values without a
    canonical form fall back to ``repr`` so unexpected types stay comparable
    instead of raising from a worker thread.
    """
    payload = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_current_identity(result_identity: str, current_identity: str) -> bool:
    """Return whether a completed operation still matches the live UI inputs.

    An empty result identity is treated as unspecified so older callers and
    direct slot tests keep applying.  A non-empty mismatch is stale.
    """
    result = str(result_identity or "")
    if not result:
        return True
    return result == str(current_identity or "")


def context_library_config_digest(
    config: object = None,
    *,
    context_flags: Mapping[str, object] | None = None,
) -> str:
    """Digest the enablement flags a context-library scan used.

    Project switches live in ``project_context_settings.json``, not the raw
    translator config, so the digest is the effective flag set. Unrelated
    global keys (theme, models) must not change it.
    """
    flags = default_context_flags_from_config(
        config if isinstance(config, dict) else None
    )
    if context_flags is not None:
        flags.update(
            {str(key): bool(value) for key, value in dict(context_flags).items()}
        )
    return canonical_digest({"context_flags": flags})


def _selected_custom_provider_endpoint(
    custom_providers: Mapping[str, object] | None,
    provider: str,
) -> dict[str, object]:
    """Return rewrite/credential fields for ``provider``, or ``{}`` if builtin."""
    if not provider or not isinstance(custom_providers, Mapping):
        return {}
    selected: object | None = None
    for raw_key, item in custom_providers.items():
        if str(raw_key or "").strip().lower() == provider:
            selected = item
            break
    if selected is None:
        return {}
    if is_dataclass(selected) and not isinstance(selected, type):
        selected = asdict(selected)
    if not isinstance(selected, Mapping):
        return {}
    return {
        "base_url": str(selected.get("base_url") or "").strip(),
        "models_url": str(selected.get("models_url") or "").strip(),
        "requires_key": bool(selected.get("requires_key", True)),
        "api_key_env": str(selected.get("api_key_env") or "").strip(),
    }


def litellm_connection_identity(
    *,
    provider: str,
    model: str,
    custom_providers: Mapping[str, object] | None = None,
) -> str:
    """Digest the provider, model, and selected custom endpoint for one test."""
    provider_id = str(provider or "").strip().lower()
    return canonical_digest(
        {
            "provider": provider_id,
            "model": str(model or "").strip(),
            "custom_provider": _selected_custom_provider_endpoint(
                custom_providers,
                provider_id,
            ),
        }
    )
