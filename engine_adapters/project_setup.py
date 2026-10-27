"""Read-only MV preparation facts and the shared settings update contract."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from .project_config import (
    detect_project_material,
    read_mv_project_config,
    resolve_project_configuration,
)


def mv_project_setup_status(config: Mapping[str, Any], config_file: str | Path) -> dict:
    """Describe missing inputs and a local source suggestion without adopting it.

    A conventional original/www sibling is only a form suggestion. No support
    window, review, file filter or occurrence scope is inferred from artifacts.
    """
    config = resolve_project_configuration(config, config_file)
    base = Path(config_file).resolve().parent
    section = config.get("rpgmaker_mv")
    section = section if isinstance(section, Mapping) else {}

    def resolved(value):
        if not isinstance(value, str) or not value.strip():
            return ""
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else base / path).resolve())

    target = resolved(config.get("game_root"))
    source = resolved(section.get("source_root"))
    window = resolved(section.get("support_window_file"))
    missing = []
    if not source or not Path(source).is_dir():
        missing.append("source_root")
    if not window or not Path(window).is_file():
        missing.append("support_window_file")
    suggestion = ""
    if target:
        content = Path(target)
        package = content.parent if content.name.lower() == "www" else content
        if package.name.lower() == "work":
            material = detect_project_material(package.parent / "original")
            if material and material["engine"] == "rpgmaker_mv":
                suggestion = material["content_root"]
    return {
        "target_root": target,
        "source_root": source,
        "source_suggestion": suggestion,
        "support_window_file": window,
        "missing": missing,
        "scope": "selected"
        if (
            config.get("include_files")
            or config.get("include_prefixes")
            or section.get("occurrence_scope_file")
        )
        else "full_project",
    }


def build_mv_setup_config(
    config: Mapping[str, Any],
    config_file: str | Path,
    *,
    target_root: str,
    source_root: str,
    support_window_file: str,
) -> dict:
    """Build a validated settings edit, preserving routes, scope and credentials.

    This validates paths/window syntax only. The following doctor run performs
    the authoritative material and coverage checks; saving never grants a
    review or writeback permission and never calls a provider.
    """
    result = copy.deepcopy(dict(config))
    result["engine"] = "rpgmaker_mv"
    result = resolve_project_configuration(result, config_file)
    result["game_root"] = target_root
    section = result.setdefault("rpgmaker_mv", {})
    if not isinstance(section, dict):
        raise ValueError("rpgmaker_mv must be an object.")
    section.update(source_root=source_root, support_window_file=support_window_file)
    result = resolve_project_configuration(result, config_file)
    settings = read_mv_project_config(result, config_file)
    if not settings.source_root.is_dir() or not settings.target_root.is_dir():
        raise ValueError("MV source and target directories must exist.")
    if (
        settings.source_root == settings.target_root
        or settings.source_root in settings.target_root.parents
        or settings.target_root in settings.source_root.parents
    ):
        raise ValueError("MV source and target must be independent, non-nested directories.")
    result["rpgmaker_mv"]["source_root"] = str(settings.source_root)
    result["rpgmaker_mv"]["support_window_file"] = str(settings.window_file)
    prepare = result.setdefault("prepare", {})
    if isinstance(prepare, dict):
        prepare["enabled"] = False
    return result
