"""Lightweight path helpers for the GUI shell.

Kept separate from ``translator_runtime`` so importing the GUI does not pull
in the full translation runtime dependency chain.
"""
from __future__ import annotations

import os
from pathlib import Path


def canonical_abs_path(path: str | Path) -> str:
    """Return a stable absolute path (long path on Windows, not 8.3 short names)."""
    if not path:
        return ""
    abs_path = os.path.abspath(str(path))
    try:
        return str(Path(abs_path).resolve(strict=False))
    except OSError:
        return abs_path


def resolve_effective_game_root(game_root: str | Path) -> str:
    """Prefer nested work/ when game_root points at a project-root layout."""
    normalized = canonical_abs_path(game_root)
    if os.path.basename(normalized).lower() == "work":
        return normalized

    nested_work = os.path.join(normalized, "work")
    original_game = os.path.join(normalized, "original", "game")
    if os.path.isdir(nested_work) and os.path.isdir(original_game):
        return canonical_abs_path(nested_work)
    return normalized


def normalize_context_storage_location(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"game", "game_dir", "game_directory"}:
            return "game"
        if normalized in {"tool", "project", "repo", "repository", "internal"}:
            return "tool"
    return "tool"