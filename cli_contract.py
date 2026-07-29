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

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NEEDS_ACTION = 3
EXIT_BLOCKED = 4
EXIT_INVALID_STATE = 5
EXIT_RETRYABLE = 6


def classify_error(message: str, *, exception_type: str = "") -> dict[str, Any]:
    """Classify existing CLI failures without changing their human-readable text."""

    normalized = str(message or "").strip().lower()
    stale_markers = (
        "changed after the last check",
        "stale check",
        "check fingerprint",
        "source snapshot",
        "manifest or results changed",
    )
    retryable_markers = (
        "rate limit",
        "resource exhausted",
        "quota",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
    )
    precondition_markers = (
        "not succeeded yet",
        "missing",
        "not found",
        "does not exist",
        "download first",
        "no job",
        "api key",
        "configuration",
        "config ",
    )
    blocked_markers = (
        "not safe",
        "safety level",
        "blocked",
        "refused",
        "already applied",
        "disabled",
        "cost limit",
        "max cost",
    )

    if any(marker in normalized for marker in stale_markers):
        return {
            "code": "STALE_STATE",
            "retryable": False,
            "suggested_action": "run_check_again",
            "exit_code": EXIT_INVALID_STATE,
        }
    if any(marker in normalized for marker in retryable_markers):
        return {
            "code": "REMOTE_RETRYABLE",
            "retryable": True,
            "suggested_action": "retry_later",
            "exit_code": EXIT_RETRYABLE,
        }
    if any(marker in normalized for marker in precondition_markers):
        return {
            "code": "PRECONDITION_FAILED",
            "retryable": False,
            "suggested_action": "inspect_configuration_and_artifacts",
            "exit_code": EXIT_INVALID_STATE,
        }
    if any(marker in normalized for marker in blocked_markers):
        return {
            "code": "COMMAND_BLOCKED",
            "retryable": False,
            "suggested_action": "inspect_diagnostics",
            "exit_code": EXIT_BLOCKED,
        }
    return {
        "code": "COMMAND_REFUSED" if exception_type == "SystemExit" else "INTERNAL_ERROR",
        "retryable": False,
        "suggested_action": "inspect_diagnostics",
        "exit_code": EXIT_BLOCKED if exception_type == "SystemExit" else 1,
    }


def strict_exit_code(envelope: Mapping[str, Any]) -> int:
    """Return the documented semantic exit code for a machine envelope."""

    if not envelope.get("ok"):
        error = envelope.get("error")
        if isinstance(error, Mapping):
            details = error.get("details")
            if isinstance(details, Mapping):
                semantic_code = details.get("semantic_exit_code")
                if isinstance(semantic_code, int):
                    return semantic_code
        return EXIT_BLOCKED

    command = str(envelope.get("command") or "")
    status = str(envelope.get("status") or "").strip().lower()
    if command == "check":
        if status == "safe":
            return EXIT_OK
        if status == "warn":
            return EXIT_NEEDS_ACTION
        if status == "block":
            return EXIT_BLOCKED
        return EXIT_INVALID_STATE
    if command == "doctor" and status == "blocked":
        return EXIT_BLOCKED
    if command in {"submit", "status"} and status in {
        "failed",
        "cancelled",
        "canceled",
        "expired",
        "job_state_failed",
        "job_state_cancelled",
        "job_state_canceled",
        "job_state_expired",
    }:
        return EXIT_BLOCKED
    return EXIT_OK


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
