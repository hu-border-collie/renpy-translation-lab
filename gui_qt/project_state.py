"""Project state management for the GUI.

Responsible for:
- Resolving the location of the CLI scripts (root of the project).
- Managing the current game/work directory.
- Reading and writing configuration files (api_keys.json, translator_config.json)
  while preserving unknown fields.
- Providing paths for logs, manifests, etc.

This module contains **no** translation logic. It is only configuration and path helpers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ProjectState:
    """Holds the current project context for the GUI shell."""

    def __init__(self):
        # Resolve the directory that contains gemini_translate_batch.py (the tool root)
        # gui_qt/ -> parent is the root
        gui_qt_dir = Path(__file__).resolve().parent
        self.tool_root: Path = gui_qt_dir.parent

        self.batch_script: Path = self.tool_root / "gemini_translate_batch.py"
        if not self.batch_script.exists():
            # Fallback for some layouts
            self.batch_script = self.tool_root / "gemini_translate_batch.py"

        self.api_keys_path: Path = self._resolve_api_keys_path()
        self.config_path: Path = self.tool_root / "translator_config.json"

        self._game_root: Path | None = None
        self._load_game_root_from_config()

    # --- Path helpers ---

    def get_tool_root(self) -> Path:
        return self.tool_root

    def get_batch_script_path(self) -> Path:
        return self.batch_script

    def get_logs_dir(self) -> Path:
        return self.tool_root / "logs" / "batch_jobs"

    def get_latest_manifest_path(self) -> Path | None:
        latest_file = self.tool_root / "logs" / "batch_jobs" / "latest_manifest.txt"
        if latest_file.exists():
            try:
                content = latest_file.read_text(encoding="utf-8").strip()
                if content and Path(content).exists():
                    return Path(content)
            except Exception:
                pass
        return None

    def _resolve_api_keys_path(self) -> Path:
        root_api_keys = self.tool_root / "api_keys.json"
        if root_api_keys.exists():
            return root_api_keys
        return self.tool_root.parent / "data" / "api_keys.json"

    def _path_without_resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        return candidate.absolute()

    # --- JSON helpers ---

    def _read_json_object(self, path: Path, description: str) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise ValueError(f"Failed to read {description}: {path}") from exc
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{description} is not valid JSON: {path}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{description} must be a JSON object: {path}")
        return data

    def _write_json_object(self, path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing_mode = path.stat().st_mode & 0o777 if path.exists() else None
            target_mode = existing_mode
            if target_mode is None and path.name == "api_keys.json":
                target_mode = 0o600
            payload = json.dumps(data, ensure_ascii=False, indent=2)
            if target_mode is None:
                tmp.write_text(payload, encoding="utf-8")
            else:
                if tmp.exists():
                    tmp.unlink()
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, target_mode)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        fd = None
                        handle.write(payload)
                finally:
                    if fd is not None:
                        os.close(fd)
                os.chmod(tmp, target_mode)
            os.replace(tmp, path)
        except OSError as exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise ValueError(f"Failed to write JSON file: {path}") from exc

    # --- Game root (project directory) ---

    def get_game_root(self) -> Path | None:
        return self._game_root

    def set_game_root(self, path: str | Path) -> None:
        """Update current game root and persist it to translator_config.json."""
        p = self._path_without_resolve(path)
        if not p.exists():
            # Still allow setting even if not exist yet (user may create later)
            pass
        self._save_game_root_to_config(p)
        self._game_root = p

    def _load_game_root_from_config(self) -> None:
        if not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8-sig") or "{}")
            game_root = data.get("game_root")
            if isinstance(game_root, str) and game_root.strip():
                self._game_root = self._path_without_resolve(game_root)
        except Exception:
            # Non-fatal for GUI
            pass

    def _save_game_root_to_config(self, game_root: Path) -> None:
        """Update only the game_root key, preserve everything else."""
        data = self._read_json_object(self.config_path, "translator_config.json")
        data["game_root"] = str(game_root)
        self._write_json_object(self.config_path, data)

    # --- Config helpers (api_keys + translator_config) ---

    def load_api_keys(self) -> list[str]:
        if not self.api_keys_path.exists():
            return []
        try:
            data = json.loads(self.api_keys_path.read_text(encoding="utf-8-sig") or "{}")
            keys = data.get("api_keys", [])
            return [k for k in keys if isinstance(k, str)]
        except Exception:
            return []

    def _is_placeholder_api_key(self, value: str) -> bool:
        text = value.strip().lower()
        if not text:
            return True
        placeholder_markers = (
            "your-key",
            "your api key",
            "your-api-key",
            "your_gemini_api_key",
            "your-gemini-api-key",
            "paste-key",
            "paste-api-key",
            "replace-me",
        )
        return any(marker in text for marker in placeholder_markers)

    def _valid_api_keys(self, keys: list[str]) -> list[str]:
        return [
            key
            for key in keys
            if isinstance(key, str) and key.strip() and not self._is_placeholder_api_key(key)
        ]

    def get_api_key_status(self) -> tuple[int, str]:
        file_keys = self._valid_api_keys(self.load_api_keys())
        if file_keys:
            return len(file_keys), "file"
        env_keys = [
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("GEMINI_API_KEY_2"),
            os.environ.get("GEMINI_API_KEY_3"),
        ]
        configured_env_keys = self._valid_api_keys([key for key in env_keys if isinstance(key, str)])
        if configured_env_keys:
            return len(configured_env_keys), "environment"
        return 0, ""

    def save_api_keys(self, keys: list[str]) -> None:
        """Save API keys while preserving legacy/unknown fields."""
        data = self._read_json_object(self.api_keys_path, "api_keys.json")
        data["api_keys"] = keys
        self._write_json_object(self.api_keys_path, data)

    def load_translator_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            raw = self.config_path.read_text(encoding="utf-8-sig")
            if not raw.strip():
                return {}
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_translator_config(self, config: dict[str, Any]) -> None:
        """Write config while trying to preserve structure."""
        self._write_json_object(self.config_path, config)
