"""Project-scoped asset paths (glossary, macro setting) for translator_config.json."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_GLOSSARY_NAME = "glossary.json"
DEFAULT_MACRO_SETTING_NAME = "macro_setting.md"


def canonical_abs_path(path: str | os.PathLike[str]) -> str:
    """Return a stable absolute path (long path on Windows, not 8.3 short names)."""
    if not path:
        return ""
    abs_path = os.path.abspath(str(path))
    try:
        return str(Path(abs_path).resolve(strict=False))
    except OSError:
        return abs_path


def expected_project_asset_paths(game_root: str | os.PathLike[str]) -> dict[str, str]:
    root = canonical_abs_path(game_root)
    return {
        "glossary_file": os.path.join(root, DEFAULT_GLOSSARY_NAME),
        "macro_setting_file": os.path.join(root, DEFAULT_MACRO_SETTING_NAME),
    }


def resolve_project_asset_path(
    configured: str | os.PathLike[str] | None,
    *,
    game_root: str | os.PathLike[str] = "",
    tool_dir: str | os.PathLike[str] = "",
    default_name: str,
) -> str:
    """Resolve a project asset path, preferring the current work directory.

    Relative values (including bare names like ``glossary.json``) are always
    joined under ``game_root`` when it is set. They must not silently fall back
    to a same-named file under the tool install directory.

    Absolute paths are kept as-is so a deliberately shared glossary still works.
    """
    text = str(configured or "").strip()
    root = canonical_abs_path(game_root) if game_root else ""
    tool = canonical_abs_path(tool_dir) if tool_dir else ""

    if text and os.path.isabs(text):
        return canonical_abs_path(text)

    if root:
        if text:
            return canonical_abs_path(os.path.join(root, text))
        return os.path.join(root, default_name)

    if tool:
        if text:
            return canonical_abs_path(os.path.join(tool, text))
        return os.path.join(tool, default_name)

    return text


def resolve_glossary_path(
    configured: str | os.PathLike[str] | None,
    *,
    game_root: str | os.PathLike[str] = "",
    tool_dir: str | os.PathLike[str] = "",
) -> str:
    return resolve_project_asset_path(
        configured,
        game_root=game_root,
        tool_dir=tool_dir,
        default_name=DEFAULT_GLOSSARY_NAME,
    )


def resolve_macro_setting_path(
    configured: str | os.PathLike[str] | None,
    *,
    game_root: str | os.PathLike[str] = "",
    tool_dir: str | os.PathLike[str] = "",
) -> str:
    return resolve_project_asset_path(
        configured,
        game_root=game_root,
        tool_dir=tool_dir,
        default_name=DEFAULT_MACRO_SETTING_NAME,
    )


def sync_project_asset_paths_in_config(
    config: dict[str, Any],
    game_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Force glossary/macro_setting paths to the current work directory.

    Used when switching projects: assets are always re-homed to the new work.
    """
    if not isinstance(config, dict):
        config = {}
    expected = expected_project_asset_paths(game_root)
    config["glossary_file"] = expected["glossary_file"]
    batch = config.get("batch")
    if not isinstance(batch, dict):
        batch = {}
        config["batch"] = batch
    batch["macro_setting_file"] = expected["macro_setting_file"]
    return config


def normalize_relative_project_assets_in_config(
    config: dict[str, Any],
    game_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Rewrite relative glossary/macro paths under game_root; keep abs paths.

    Unlike :func:`sync_project_asset_paths_in_config`, this does not force
    absolute custom paths back onto the default work filenames. It only stops
    bare names like ``glossary.json`` from drifting to the tool directory.
    """
    if not isinstance(config, dict):
        config = {}

    root = str(game_root or "").strip() or str(config.get("game_root") or "").strip()
    if not root:
        return config

    expected = expected_project_asset_paths(root)

    glossary_configured = config.get("glossary_file")
    if glossary_configured is None:
        glossary_configured = config.get("glossary_path")
    glossary_text = str(glossary_configured or "").strip()
    if not glossary_text:
        config["glossary_file"] = expected["glossary_file"]
    elif not os.path.isabs(glossary_text):
        config["glossary_file"] = resolve_glossary_path(
            glossary_text, game_root=root
        )

    batch = config.get("batch")
    if not isinstance(batch, dict):
        batch = {}
        config["batch"] = batch
    macro_text = str(batch.get("macro_setting_file") or "").strip()
    if not macro_text:
        batch["macro_setting_file"] = expected["macro_setting_file"]
    elif not os.path.isabs(macro_text):
        batch["macro_setting_file"] = resolve_macro_setting_path(
            macro_text, game_root=root
        )
    return config


def paths_match_project(
    configured_path: str,
    expected_path: str,
) -> bool:
    if not configured_path or not expected_path:
        return False
    return os.path.normcase(canonical_abs_path(configured_path)) == os.path.normcase(
        canonical_abs_path(expected_path)
    )
