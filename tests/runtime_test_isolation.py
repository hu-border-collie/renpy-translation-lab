"""Keep CLI/GUI tests off the developer's translator_config.json.

A local config with LiteLLM + a real game_root otherwise leaks into full-suite
runs: routing preflight rejects unprefixed models, and workbench empty-state
tests pick up live manifests.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

_ISOLATED_CONFIG_PATH: Path | None = None
_PROJECT_STATE_PATCHED = False


def isolated_translator_config_path() -> Path:
    """Return the empty temp translator_config.json used by test isolation."""
    global _ISOLATED_CONFIG_PATH
    if _ISOLATED_CONFIG_PATH is None:
        config_dir = Path(
            tempfile.mkdtemp(prefix="renpy-translation-lab-test-config-")
        )
        config_path = config_dir / "translator_config.json"
        config_path.write_text("{}\n", encoding="utf-8")
        _ISOLATED_CONFIG_PATH = config_path
    return _ISOLATED_CONFIG_PATH


def isolate_developer_translator_config() -> Path:
    """Point runtime and GUI ProjectState at an empty temp translator_config."""
    config_path = isolated_translator_config_path()
    _patch_runtime_translator_config(config_path)
    _patch_project_state(config_path)
    return config_path


def _patch_runtime_translator_config(config_path: Path) -> None:
    import translator_runtime as runtime

    runtime.TRANSLATOR_CONFIG = str(config_path)
    try:
        import gemini_translate_batch as batch
    except ImportError:
        return
    batch.SYNC_BACKEND = "gemini"
    batch.SYNC_MODEL = ""


def _patch_project_state(config_path: Path) -> None:
    global _PROJECT_STATE_PATCHED
    if _PROJECT_STATE_PATCHED:
        return
    try:
        from gui_qt.project_state import ProjectState
    except ImportError:
        return

    original_init = ProjectState.__init__

    def isolated_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.config_path = config_path
        self._game_root = None
        self._workspace_root = None
        self._game_root_redirect_from = None

    ProjectState.__init__ = isolated_init  # type: ignore[method-assign]
    ProjectState._load_game_root_from_config = lambda self: None  # type: ignore[method-assign]
    ProjectState._load_workspace_root_from_config = lambda self: None  # type: ignore[method-assign]
    _PROJECT_STATE_PATCHED = True
