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

import translator_runtime as runtime


class ProjectState:
    """Holds the current project context for the GUI shell."""

    def __init__(self):
        # Resolve the directory that contains gemini_translate_batch.py (the tool root)
        # gui_qt/ -> parent is the root
        gui_qt_dir = Path(__file__).resolve().parent
        self.tool_root: Path = gui_qt_dir.parent

        self.batch_script: Path = self.tool_root / "gemini_translate_batch.py"
        self.sync_script: Path = self.tool_root / "gemini_translate.py"

        self.api_keys_path: Path = self._resolve_api_keys_path()
        self.config_path: Path = self.tool_root / "translator_config.json"

        self._game_root: Path | None = None
        self._game_root_redirect_from: Path | None = None
        self._load_game_root_from_config()

    # --- Path helpers ---

    def get_tool_root(self) -> Path:
        return self.tool_root

    def get_batch_script_path(self) -> Path:
        return self.batch_script

    def get_sync_script_path(self) -> Path:
        return self.sync_script

    def get_cli_root_dir(self) -> Path:
        if (self.tool_root / "api_keys.json").exists():
            return self.tool_root
        return self.tool_root.parent

    def get_logs_dir(self) -> Path:
        return self.get_cli_root_dir() / "logs" / "batch_jobs"

    def get_latest_manifest_path(self) -> Path | None:
        latest_file = self.get_logs_dir() / "latest_manifest.txt"
        if latest_file.exists():
            try:
                content = latest_file.read_text(encoding="utf-8").strip()
                if content and Path(content).exists():
                    return Path(content)
            except OSError:
                pass
        return None

    def get_latest_manifest_path_for_mode(
        self,
        game_root: Path,
        work_mode: Any,
    ) -> Path | None:
        from .work_modes import manifest_mode_for_work_mode
        expected_mode = manifest_mode_for_work_mode(work_mode)
        if expected_mode is None:
            return None
        normalized_game_root = self._normalized_path_text(game_root)

        # Prefer the CLI's current pointer so newly-created jobs are visible
        # immediately even when the broader history index is cached.
        latest = self.get_latest_manifest_path()
        if latest is not None:
            try:
                manifest = self.load_manifest_file(latest)
            except ValueError:
                return None
            if self._manifest_matches(manifest, expected_mode, normalized_game_root):
                return latest

        for _, path, actual_mode, base_dir in self._manifest_history_index():
            if not self._manifest_mode_matches(actual_mode, expected_mode):
                continue
            if base_dir == normalized_game_root:
                return path

        return None

    def invalidate_manifest_history_cache(self) -> None:
        self._manifest_history_cache_signature = None
        self._manifest_history_entries = None

    def _manifest_history_index(self) -> list[tuple[int, Path, str, str]]:
        logs_dir = self.get_logs_dir()
        signature = self._manifest_history_signature(logs_dir)
        cached_signature = getattr(self, "_manifest_history_cache_signature", None)
        cached_entries = getattr(self, "_manifest_history_entries", None)
        if cached_signature == signature and cached_entries is not None:
            return cached_entries

        entries: list[tuple[int, Path, str, str]] = []
        if logs_dir.exists():
            for root, _, files in os.walk(logs_dir):
                if "manifest.json" not in files:
                    continue
                path = Path(root) / "manifest.json"
                try:
                    stat = path.stat()
                    manifest = self.load_manifest_file(path)
                    actual_mode = manifest.get("mode", "translation")
                    actual_text = actual_mode.strip() if isinstance(actual_mode, str) else "translation"
                    base_dir = manifest.get("base_dir")
                    if not isinstance(base_dir, str) or not base_dir.strip():
                        continue
                    entries.append((
                        stat.st_mtime_ns,
                        path,
                        actual_text,
                        self._normalized_path_text(base_dir),
                    ))
                except (OSError, ValueError):
                    continue

        entries.sort(key=lambda entry: entry[0], reverse=True)
        self._manifest_history_cache_signature = signature
        self._manifest_history_entries = entries
        return entries

    def _manifest_history_signature(self, logs_dir: Path) -> tuple[str, int | None, int | None]:
        latest_file = logs_dir / "latest_manifest.txt"

        def mtime_ns(path: Path) -> int | None:
            try:
                return path.stat().st_mtime_ns
            except OSError:
                return None

        return (
            self._normalized_path_text(logs_dir),
            mtime_ns(logs_dir),
            mtime_ns(latest_file),
        )

    def _manifest_matches(
        self,
        manifest: dict[str, Any],
        expected_mode: str,
        normalized_game_root: str,
    ) -> bool:
        actual_mode = manifest.get("mode", "translation")
        actual_text = actual_mode.strip() if isinstance(actual_mode, str) else "translation"
        if not self._manifest_mode_matches(actual_text, expected_mode):
            return False
        base_dir = manifest.get("base_dir")
        return (
            isinstance(base_dir, str)
            and bool(base_dir.strip())
            and self._normalized_path_text(base_dir) == normalized_game_root
        )

    def _manifest_mode_matches(self, actual_text: str, expected_mode: str) -> bool:
        if expected_mode == "translation":
            return actual_text in {"", "translation"}
        return actual_text == expected_mode

    def load_manifest_file(self, manifest_path: str | Path) -> dict[str, Any]:
        manifest = self._read_json_object(Path(manifest_path), "batch manifest")
        manifest["_manifest_path"] = str(Path(manifest_path))
        return manifest

    def load_resume_manifest(
        self,
        manifest_path: str | Path,
        *,
        work_mode=None,
    ) -> dict[str, Any]:
        from .work_modes import WorkMode, normalize_work_mode
        from .workflow_factory import validate_resume_manifest

        manifest = self.load_manifest_file(manifest_path)
        game_root = self.get_game_root()
        validate_resume_manifest(
            normalize_work_mode(work_mode or WorkMode.BATCH_TRANSLATION),
            manifest,
            game_root=str(game_root) if game_root is not None else None,
            normalized_path_text=self._normalized_path_text,
        )
        return manifest

    def _normalized_path_text(self, path: str | Path) -> str:
        return os.path.normcase(runtime.canonical_abs_path(str(path)))

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

    def normalize_game_root(self, path: str | Path) -> tuple[Path, bool]:
        """Resolve project-root selections to nested work/ when that directory exists."""
        original = self._path_without_resolve(path)
        effective = Path(runtime.canonical_abs_path(runtime.resolve_effective_game_root(str(original))))
        adjusted = self._normalized_path_text(original) != self._normalized_path_text(effective)
        return effective, adjusted

    def set_game_root(self, path: str | Path) -> tuple[Path, bool]:
        """Update current game root and persist it to translator_config.json."""
        original = self._path_without_resolve(path)
        p, adjusted = self.normalize_game_root(path)
        if not p.exists():
            # Still allow setting even if not exist yet (user may create later)
            pass
        self._save_game_root_to_config(p)
        self._game_root = p
        if adjusted:
            self._game_root_redirect_from = original
        else:
            self._game_root_redirect_from = None
        return p, adjusted

    def take_game_root_redirect_from(self) -> Path | None:
        """Return and clear the last auto-redirect source path, if any."""
        source = self._game_root_redirect_from
        self._game_root_redirect_from = None
        return source

    def _load_game_root_from_config(self) -> None:
        if not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8-sig") or "{}")
            game_root = data.get("game_root")
            if isinstance(game_root, str) and game_root.strip():
                original = self._path_without_resolve(game_root)
                effective, adjusted = self.normalize_game_root(game_root)
                self._game_root = effective
                if adjusted:
                    self._game_root_redirect_from = original
                    self._save_game_root_to_config(effective)
        except Exception:
            # Non-fatal for GUI
            pass

    def _save_game_root_to_config(self, game_root: Path) -> None:
        """Update only the game_root key, preserve everything else."""
        data = self._read_json_object(self.config_path, "translator_config.json")
        data["game_root"] = runtime.canonical_abs_path(str(game_root))
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
