"""Model migration presentation for the local settings confirmation."""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import shlex
import sys
from typing import Any

from providers.model_routing_reader import read_routing_plan
from ..user_copy import MODEL_MIGRATION_COPY, MODEL_PROFILES_PAGE_COPY


def format_migration_preview(
    config_file: Path, candidate: Mapping[str, Any], *, execution: str,
) -> str:
    """Describe the validated candidate's effective routes without connection details."""
    plan = read_routing_plan(candidate, legacy_execution=execution)
    section = candidate["model_routing"]
    defaults = section["defaults"]
    entrypoints = section["legacy_entrypoints"]
    strategies = MODEL_PROFILES_PAGE_COPY["strategy_labels"]
    stages = MODEL_PROFILES_PAGE_COPY["route_stage_labels"]
    lines = [
        MODEL_MIGRATION_COPY["file"].format(path=config_file),
        "",
        MODEL_MIGRATION_COPY["default"].format(
            model=plan.profiles[defaults["primary_profile_id"]].model,
            strategy=strategies[defaults["execution_strategy"]],
        ),
        MODEL_MIGRATION_COPY["legacy_models"].format(
            sync_model=plan.profiles[entrypoints["sync_profile_id"]].model,
            batch_model=plan.profiles[entrypoints["batch_profile_id"]].model,
        ),
        "",
        MODEL_MIGRATION_COPY["routes_title"].format(strategy=strategies[execution]),
    ]
    for stage, label in stages.items():
        route = plan.routes[stage]
        lines.append(MODEL_MIGRATION_COPY["route"].format(
            stage=label, model=plan.profiles[route.profile_id].model,
            strategy=strategies[route.strategy.value],
        ))
    lines.extend(["", MODEL_MIGRATION_COPY["confirm_hint"]])
    return "\n".join(lines)


def format_rollback_command(
    tool_root: Path, config_file: Path, report_path: str,
) -> tuple[str, str]:
    """Render the CLI rollback for the desktop platform's shell."""
    args = [
        sys.executable, str(tool_root / "providers" / "model_config_migration.py"),
        "rollback", "--config", str(config_file), "--report", report_path, "--json",
    ]
    if os.name == "nt":
        # PowerShell also treats typographic apostrophes as quote delimiters.
        escaped_quotes = str.maketrans({quote: quote * 2 for quote in "'‘’‚‛"})
        quoted = ["'" + arg.translate(escaped_quotes) + "'" for arg in args]
        return "PowerShell", "& " + " ".join(quoted)
    return "Shell", shlex.join(args)
