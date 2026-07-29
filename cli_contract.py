"""Versioned machine-readable result contract shared by CLI frontends.

The contract is deliberately independent from Batch implementation details so
GUI and future CLI entry points can consume the same success/error envelope
without parsing human-readable console output.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any, TextIO


CLI_SCHEMA_VERSION = 1


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_compatible(item) for item in value), key=str)
    return str(value)


def success_envelope(
    command: str,
    *,
    status: str = "completed",
    result: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    warnings: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build a stable successful command result envelope."""

    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "command": str(command),
        "ok": True,
        "status": str(status),
        "result": _json_compatible(dict(result or {})),
        "artifacts": _json_compatible(dict(artifacts or {})),
        "warnings": _json_compatible(list(warnings or [])),
        "error": None,
    }


def error_envelope(
    command: str,
    *,
    code: str,
    message: str,
    retryable: bool = False,
    suggested_action: str = "",
    details: Mapping[str, Any] | None = None,
    warnings: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build a stable failed command result envelope."""

    error = {
        "code": str(code),
        "message": str(message),
        "retryable": bool(retryable),
        "suggested_action": str(suggested_action),
        "details": _json_compatible(dict(details or {})),
    }
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "command": str(command),
        "ok": False,
        "status": "failed",
        "result": {},
        "artifacts": {},
        "warnings": _json_compatible(list(warnings or [])),
        "error": error,
    }


def write_json_envelope(envelope: Mapping[str, Any], stream: TextIO) -> None:
    """Write exactly one JSON document followed by a newline."""

    json.dump(
        _json_compatible(dict(envelope)),
        stream,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    stream.write("\n")
    stream.flush()
