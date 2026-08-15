"""Versioned machine-readable result contract shared by CLI frontends.

The contract is deliberately independent from Batch implementation details so
GUI and future CLI entry points can consume the same success/error envelope
without parsing human-readable console output.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from os import PathLike
from typing import Any, Iterator, TextIO

CLI_SCHEMA_VERSION = 1
_MISSING = object()
_MACHINE_OUTPUT_ACTIVE: ContextVar[bool] = ContextVar(
    "cli_machine_output_active",
    default=False,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NEEDS_ACTION = 3
EXIT_BLOCKED = 4
EXIT_INVALID_STATE = 5
EXIT_RETRYABLE = 6


def machine_output_active() -> bool:
    """Return whether the current call is executing a machine-output command."""

    return _MACHINE_OUTPUT_ACTIVE.get()


@contextmanager
def machine_output_context() -> Iterator[None]:
    """Mark nested core work as part of a machine-output CLI invocation."""

    token = _MACHINE_OUTPUT_ACTIVE.set(True)
    try:
        yield
    finally:
        _MACHINE_OUTPUT_ACTIVE.reset(token)


def parse_result_envelope(text: str) -> dict[str, Any]:
    """Parse and validate one versioned CLI result envelope."""

    try:
        document = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("CLI output is not a valid JSON result envelope.") from exc
    if not isinstance(document, Mapping):
        raise ValueError("CLI result envelope must be a JSON object.")
    if document.get("schema_version") != CLI_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported CLI result schema version: {document.get('schema_version')!r}."
        )
    if not isinstance(document.get("command"), str) or not document["command"]:
        raise ValueError("CLI result envelope is missing command.")
    if not isinstance(document.get("ok"), bool):
        raise ValueError("CLI result envelope is missing boolean ok.")
    if not isinstance(document.get("status"), str):
        raise ValueError("CLI result envelope is missing status.")
    for field in ("result", "artifacts"):
        if not isinstance(document.get(field), Mapping):
            raise ValueError(f"CLI result envelope field {field!r} must be an object.")
    if not isinstance(document.get("warnings"), list):
        raise ValueError("CLI result envelope field 'warnings' must be an array.")
    error = document.get("error")
    if document["ok"]:
        if error is not None:
            raise ValueError("Successful CLI result envelope must not contain an error.")
    elif not isinstance(error, Mapping):
        raise ValueError("Failed CLI result envelope must contain an error object.")
    return dict(document)


class MachineContractError(SystemExit):
    """Structured refusal raised by opt-in machine invocation guards."""

    def __init__(
        self,
        message: str,
        *,
        code_name: str,
        suggested_action: str,
        semantic_exit_code: int = EXIT_INVALID_STATE,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(str(message))
        self.code_name = str(code_name)
        self.suggested_action = str(suggested_action)
        self.semantic_exit_code = int(semantic_exit_code)
        self.retryable = bool(retryable)
        self.details = dict(details or {})


def classify_error(message: str, *, exception_type: str = "") -> dict[str, Any]:
    """Classify existing CLI failures without changing their human-readable text."""

    normalized = f"{exception_type} {message or ''}".strip().lower()
    stale_markers = (
        "changed after the last check",
        "stale check",
        "check fingerprint",
        "source snapshot",
        "manifest or results changed",
        "has no valid check summary",
        "older check contract",
    )
    retryable_markers = (
        "429",
        "rate limit",
        "resource exhausted",
        "resource_exhausted",
        "resourceexhausted",
        "quota exceeded",
        "503",
        "timed out",
        "timeout",
        "connecttimeout",
        "readtimeout",
        "connecterror",
        "readerror",
        "remoteprotocolerror",
        "unexpected_eof_while_reading",
        "temporarily unavailable",
        "service unavailable",
        "unavailable",
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
        if status in {"safe", "ready"}:
            return EXIT_OK
        if status in {"warn", "ready_with_warnings"}:
            return EXIT_NEEDS_ACTION
        if status in {"block", "blocked"}:
            return EXIT_BLOCKED
        return EXIT_INVALID_STATE
    if command == "doctor" and status == "blocked":
        return EXIT_BLOCKED
    if command == "reconcile-project-snapshots":
        if status == "ready":
            return EXIT_OK
        if status == "attention":
            return EXIT_NEEDS_ACTION
        return EXIT_INVALID_STATE
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


def project_fields(
    document: Mapping[str, Any],
    field_paths: Sequence[str],
) -> dict[str, Any]:
    """Project a JSON object using dot-separated mapping paths.

    Missing paths are omitted so callers can request optional fields across
    workflow states without treating their absence as a command failure.
    """

    source = _json_compatible(dict(document))
    projected: dict[str, Any] = {}
    for raw_path in field_paths:
        parts = field_path_parts(raw_path)
        value: Any = source
        for part in parts:
            if not isinstance(value, Mapping) or part not in value:
                value = _MISSING
                break
            value = value[part]
        if value is _MISSING:
            continue
        target = projected
        for part in parts[:-1]:
            existing = target.get(part)
            if not isinstance(existing, dict):
                existing = {}
                target[part] = existing
            target = existing
        target[parts[-1]] = value
    return projected


def field_path_parts(raw_path: Any) -> list[str]:
    """Validate and split one dot-separated machine-output field path."""

    parts = [part.strip() for part in str(raw_path).split(".")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid field path: {raw_path!r}")
    return parts


def write_json_envelope(
    envelope: Mapping[str, Any],
    stream: TextIO,
    *,
    compact: bool = False,
) -> None:
    """Write exactly one JSON document followed by a newline."""

    json.dump(
        _json_compatible(dict(envelope)),
        stream,
        ensure_ascii=False,
        indent=None if compact else 2,
        sort_keys=True,
        separators=(",", ":") if compact else None,
    )
    stream.write("\n")
    stream.flush()
