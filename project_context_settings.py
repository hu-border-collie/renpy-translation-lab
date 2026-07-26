"""Per-project batch context flags (RAG / source index / Project Analysis).

Global defaults still live in translator_config.json under batch.rag / batch.source_index /
batch.project_analysis.
When a game work directory has project_context_settings.json, those values override the
global defaults for that project only.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from project_asset_paths import canonical_abs_path

PROJECT_CONTEXT_SETTINGS_FILENAME = "project_context_settings.json"
SCHEMA_VERSION = 1

# Keys stored in the project file (stable API).
PROJECT_CONTEXT_FLAG_KEYS = (
    "batch_rag_enabled",
    "batch_source_index_enabled",
    "batch_rag_bootstrap_on_build",
    "batch_project_analysis_enabled",
    "batch_project_analysis_inject_published_brief",
)


def project_context_settings_path(game_root: str | os.PathLike[str] | None) -> str:
    if not game_root:
        return ""
    root = canonical_abs_path(game_root)
    if not root:
        return ""
    return os.path.join(root, PROJECT_CONTEXT_SETTINGS_FILENAME)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def default_context_flags_from_config(config: dict[str, Any] | None) -> dict[str, bool]:
    """Read global defaults from translator_config-shaped dict."""
    if not isinstance(config, dict):
        config = {}
    batch = config.get("batch")
    if not isinstance(batch, dict):
        batch = {}
    rag = batch.get("rag")
    if not isinstance(rag, dict):
        rag = {}
    source_index = batch.get("source_index")
    if not isinstance(source_index, dict):
        source_index = {}
    project_analysis = batch.get("project_analysis")
    if not isinstance(project_analysis, dict):
        project_analysis = {}
    return {
        "rag_enabled": _coerce_bool(rag.get("enabled"), False),
        "source_index_enabled": _coerce_bool(source_index.get("enabled"), False),
        "bootstrap_on_build": _coerce_bool(rag.get("bootstrap_on_build"), True),
        "project_analysis_enabled": _coerce_bool(project_analysis.get("enabled"), False),
        "project_analysis_inject_enabled": _coerce_bool(
            project_analysis.get("inject_published_brief"), False
        ),
    }


def load_project_context_settings(
    game_root: str | os.PathLike[str] | None,
) -> dict[str, bool] | None:
    """Load project overrides, or None if missing/invalid."""
    path = project_context_settings_path(game_root)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle) or {}
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    aliases = {
        "batch_rag_enabled": ("rag_enabled", False),
        "batch_source_index_enabled": ("source_index_enabled", False),
        "batch_rag_bootstrap_on_build": ("bootstrap_on_build", True),
        "batch_project_analysis_enabled": ("project_analysis_enabled", False),
        "batch_project_analysis_inject_published_brief": (
            "project_analysis_inject_enabled",
            False,
        ),
    }
    flags: dict[str, bool] = {}
    for stored_key, (flag_key, default) in aliases.items():
        if stored_key in data:
            flags[flag_key] = _coerce_bool(data.get(stored_key), default)
        elif flag_key in data:
            flags[flag_key] = _coerce_bool(data.get(flag_key), default)
    return flags or None


def save_project_context_settings(
    game_root: str | os.PathLike[str] | None,
    flags: dict[str, Any],
) -> str:
    """Write project context flags. Returns the path written."""
    path = project_context_settings_path(game_root)
    if not path:
        raise ValueError("game_root is required to save project context settings")
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "batch_rag_enabled": _coerce_bool(flags.get("rag_enabled"), False),
        "batch_source_index_enabled": _coerce_bool(flags.get("source_index_enabled"), False),
        "batch_rag_bootstrap_on_build": _coerce_bool(flags.get("bootstrap_on_build"), True),
        "batch_project_analysis_enabled": _coerce_bool(
            flags.get("project_analysis_enabled"), False
        ),
        "batch_project_analysis_inject_published_brief": _coerce_bool(
            flags.get("project_analysis_inject_enabled"), False
        ),
    }
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{PROJECT_CONTEXT_SETTINGS_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return path


def resolve_batch_context_flags(
    config: dict[str, Any] | None,
    game_root: str | os.PathLike[str] | None = None,
) -> dict[str, bool]:
    """Global defaults from config, overridden by project file when present."""
    flags = default_context_flags_from_config(config)
    project_flags = load_project_context_settings(game_root)
    if project_flags:
        flags.update(project_flags)
    return flags


def apply_project_context_settings_to_config(
    config: dict[str, Any],
    game_root: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Mutate config so context enablement matches the selected project."""
    if not isinstance(config, dict):
        return {}
    flags = resolve_batch_context_flags(config, game_root)
    batch = config.get("batch")
    if not isinstance(batch, dict):
        batch = {}
        config["batch"] = batch
    rag = batch.get("rag")
    if not isinstance(rag, dict):
        rag = {}
        batch["rag"] = rag
    source_index = batch.get("source_index")
    if not isinstance(source_index, dict):
        source_index = {}
        batch["source_index"] = source_index
    project_analysis = batch.get("project_analysis")
    if not isinstance(project_analysis, dict):
        project_analysis = {}
        batch["project_analysis"] = project_analysis
    rag["enabled"] = flags["rag_enabled"]
    rag["bootstrap_on_build"] = flags["bootstrap_on_build"]
    source_index["enabled"] = flags["source_index_enabled"]
    project_analysis["enabled"] = flags["project_analysis_enabled"]
    project_analysis["inject_published_brief"] = flags["project_analysis_inject_enabled"]
    return config


def project_has_context_settings(game_root: str | os.PathLike[str] | None) -> bool:
    path = project_context_settings_path(game_root)
    return bool(path) and os.path.isfile(path)
