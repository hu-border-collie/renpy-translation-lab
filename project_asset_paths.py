"""Project-scoped asset paths (glossary, macro setting) for translator_config.json."""
from __future__ import annotations

import os
from typing import Any

DEFAULT_GLOSSARY_NAME = "glossary.json"
DEFAULT_MACRO_SETTING_NAME = "macro_setting.md"


def canonical_abs_path(path: str | os.PathLike[str]) -> str:
    if not path:
        return ""
    return os.path.abspath(str(path))


def expected_project_asset_paths(game_root: str | os.PathLike[str]) -> dict[str, str]:
    root = canonical_abs_path(game_root)
    return {
        "glossary_file": os.path.join(root, DEFAULT_GLOSSARY_NAME),
        "macro_setting_file": os.path.join(root, DEFAULT_MACRO_SETTING_NAME),
    }


def sync_project_asset_paths_in_config(
    config: dict[str, Any],
    game_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Point glossary/macro_setting paths at the current work directory."""
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


def paths_match_project(
    configured_path: str,
    expected_path: str,
) -> bool:
    if not configured_path or not expected_path:
        return False
    return os.path.normcase(canonical_abs_path(configured_path)) == os.path.normcase(
        canonical_abs_path(expected_path)
    )