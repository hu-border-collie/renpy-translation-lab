"""Stable operation identities for asynchronous GUI results.

Issue #297 P2 requires every async task to carry an identity (project,
config digest, provider/model) and completion callbacks to compare it before
touching UI state, so a stale result can only finish cleanup.  The helpers
here turn plain Python snapshots into one opaque digest string that both the
worker (at start) and the window (at apply time) can compute independently.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import PurePath


def _canonical_value(value: object) -> object:
    """Normalize one snapshot value into sorted, JSON-serializable form."""
    if value is None or isinstance(value, (bool, int, float, str)):
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


def context_library_config_digest(config: object) -> str:
    """Digest the translator-config snapshot used for a context-library scan."""
    return canonical_digest(config or {})


def litellm_connection_identity(
    *,
    provider: str,
    model: str,
    custom_providers: Mapping[str, object] | None = None,
) -> str:
    """Digest the provider/model/custom-provider state a connection test used."""
    return canonical_digest(
        {
            "provider": str(provider or "").strip().lower(),
            "model": str(model or "").strip(),
            "custom_providers": custom_providers or {},
        }
    )
