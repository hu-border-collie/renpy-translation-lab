"""Pure helpers for API key masking and list updates (no Qt dependency)."""
from __future__ import annotations


def mask_api_key(key: str) -> str:
    stripped = key.strip()
    if not stripped:
        return "(空)"
    suffix = stripped[-4:] if len(stripped) > 4 else "****"
    return f"********{suffix}"


def commit_pending_key(keys: list[str], pending_key: str) -> tuple[list[str], str | None]:
    """Merge a non-empty input value into the key list before saving.

    Returns ``(updated_keys, error_message)``. ``error_message`` is set when the
    pending value duplicates an existing key (compared after stripping).
    """
    pending = pending_key.strip()
    if not pending:
        return list(keys), None
    if any(pending == existing.strip() for existing in keys):
        return list(keys), "duplicate"
    return [*keys, pending], None