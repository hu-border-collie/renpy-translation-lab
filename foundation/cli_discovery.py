"""Machine discovery derived from the live argparse command definitions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any


DISCOVERY_SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse.Action:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise ValueError("parser has no subcommands")


def _command_summaries(action: argparse.Action) -> dict[str, str]:
    return {
        str(choice.dest): str(choice.help or "")
        for choice in getattr(action, "_choices_actions", [])
    }


def _value_type(action: argparse.Action) -> str:
    if action.nargs == 0 and isinstance(action.const, bool):
        return "boolean"
    if action.type is not None:
        return str(getattr(action.type, "__name__", action.type))
    if action.choices:
        first = next(iter(action.choices), "")
        return type(first).__name__ if first != "" else "string"
    return "string"


def _argument_schema(action: argparse.Action) -> dict[str, Any]:
    positional = not action.option_strings
    repeatable = action.__class__.__name__ in {"_AppendAction", "_AppendConstAction"}
    nargs = 1 if action.nargs is None else action.nargs
    return {
        "name": str(action.dest),
        "kind": "positional" if positional else "option",
        "flags": list(action.option_strings),
        "value_type": _value_type(action),
        "required": bool(action.required),
        "repeatable": repeatable,
        "nargs": _json_value(nargs),
        "choices": (
            [_json_value(item) for item in action.choices]
            if action.choices is not None
            else None
        ),
        "default": _json_value(action.default),
        "help": str(action.help or ""),
    }


def command_schema(parser: argparse.ArgumentParser, command: str) -> dict[str, Any]:
    """Return a stable machine schema for one live argparse subcommand."""

    subparsers = _subparsers_action(parser)
    choices = getattr(subparsers, "choices", {})
    if command not in choices:
        raise KeyError(command)
    command_parser = choices[command]
    summaries = _command_summaries(subparsers)
    arguments = [
        _argument_schema(action)
        for action in command_parser._actions
        if action.dest != "help"
    ]
    return {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "type": "command_schema",
        "command": command,
        "description": summaries.get(command, ""),
        "arguments": arguments,
    }


def capabilities(
    parser: argparse.ArgumentParser,
    *,
    cli_version: str,
    machine_output_commands: Sequence[str],
    explicit_target_commands: Sequence[str],
    result_schema_version: int,
) -> dict[str, Any]:
    """Return the command index and supported Agent-facing contracts."""

    subparsers = _subparsers_action(parser)
    choices = getattr(subparsers, "choices", {})
    summaries = _command_summaries(subparsers)
    machine_commands = set(machine_output_commands)
    explicit_commands = set(explicit_target_commands)
    discovery_commands = {"capabilities", "schema"}
    commands = []
    for name in sorted(choices):
        action_dests = {
            action.dest
            for action in choices[name]._actions
            if action.dest != "help"
        }
        commands.append(
            {
                "name": name,
                "description": summaries.get(name, ""),
                "machine_output": name in machine_commands or name in discovery_commands,
                "supports_json": "output" in action_dests or name in discovery_commands,
                "supports_strict_exit_codes": "strict_exit_codes" in action_dests,
                "supports_non_interactive": "non_interactive" in action_dests,
                "supports_compact": "compact" in action_dests,
                "supports_fields": "fields" in action_dests,
                "supports_output_file": "output_file" in action_dests,
                "requires_explicit_target_in_agent_mode": name in explicit_commands,
            }
        )
    return {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "type": "capabilities",
        "cli_version": str(cli_version),
        "result_schema_version": int(result_schema_version),
        "command_count": len(commands),
        "commands": commands,
    }
