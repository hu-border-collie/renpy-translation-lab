"""Explicit product engine selection and audited MV material configuration.

Missing engine preserves Ren'Py behavior. MV's game_root is a controlled
content copy; source_root and the independently approved hash window are
resolved relative to the configuration file, never inferred from that copy.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .rpgmaker_mv import MvSupportWindow
from .rpgmaker_mv_fields import parse_display_fields


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON value: {value}')


def _load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def read_project_configuration(config_file: str | Path) -> dict:
    """Read product configuration without interpreting malformed JSON as Ren'Py."""
    path = Path(config_file)
    if not path.exists():
        return {}
    try:
        value = _load_json(path)
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError(f'Project configuration is unreadable: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError('Project configuration must be a JSON object.')
    return value


def project_engine(config: Mapping[str, Any]) -> str:
    value = config.get("engine", "renpy")
    if value not in ("renpy", "rpgmaker_mv"):
        raise ValueError("engine must be renpy or rpgmaker_mv.")
    return str(value)


def _path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an explicit nonempty path.")
    path = Path(value.strip()).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class MvProjectConfig:
    """Resolved material paths and the strict opt-in display field selection."""
    source_root: Path
    target_root: Path
    window_file: Path
    window: MvSupportWindow
    occurrence_scope_file: Path | None = None
    display_fields: tuple[str, ...] = ()
    reviewed_history_file: Path | None = None


def read_mv_project_config(config: Mapping[str, Any], config_file: str | Path) -> MvProjectConfig:
    """Resolve a schema-1 window and strict display opt-ins without approving bytes."""
    if project_engine(config) != "rpgmaker_mv":
        raise ValueError("MV configuration requires engine=rpgmaker_mv.")
    section = config.get("rpgmaker_mv")
    if not isinstance(section, Mapping):
        raise ValueError("rpgmaker_mv must be an object.")
    base = Path(config_file).resolve().parent
    source = _path(section.get("source_root"), base, "rpgmaker_mv.source_root")
    target = _path(config.get("game_root"), base, "game_root")
    window_file = _path(section.get("support_window_file"), base, "rpgmaker_mv.support_window_file")
    scope_value = section.get('occurrence_scope_file', '')
    if not isinstance(scope_value, str):
        raise ValueError('rpgmaker_mv.occurrence_scope_file must be a path string or empty.')
    scope_file = _path(scope_value, base, 'rpgmaker_mv.occurrence_scope_file') if scope_value.strip() else None
    fields_value = section.get('display_fields', [])
    if not isinstance(fields_value, list):
        raise ValueError('rpgmaker_mv.display_fields must be a list.')
    display_fields = parse_display_fields(fields_value)
    history_value = section.get('reviewed_history_file', '')
    if not isinstance(history_value, str):
        raise ValueError('rpgmaker_mv.reviewed_history_file must be a path string or empty.')
    history_file = _path(history_value, base, 'rpgmaker_mv.reviewed_history_file') if history_value.strip() else None
    try:
        payload = _load_json(window_file)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"MV support window cannot be read: {exc}") from exc
    if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
            or not isinstance(payload.get("material_sha256"), dict)
            or type(payload.get("message_styles", False)) is not bool):
        raise ValueError("Invalid MV support window schema; expected version 1, material_sha256 and boolean message_styles.")
    return MvProjectConfig(source, target, window_file, MvSupportWindow(
        payload["material_sha256"], message_styles=payload.get("message_styles", False)), scope_file, display_fields, history_file)


def require_engine_operation(config: Mapping[str, Any], operation: str) -> None:
    """Reject unsupported MV routes before preparation or provider dispatch."""
    if project_engine(config) != "rpgmaker_mv":
        return
    allowed = {"doctor", "translate-preflight", "coverage-status", "coverage-review-import",
               "sync-start", "sync-resume", "sync-status", "sync-cancel", "sync-derive",
               "sync-keywords", "export-keywords", "merge-keywords-to-glossary",
               "bootstrap-source-index", "bootstrap-rag",
               "check", "apply", "quality-report", "usage-report", "usage-import"}
    if operation not in allowed:
        raise ValueError(f"RPG Maker MV narrative mode does not support {operation}; use durable Sync (sync-start -> check -> apply).")


def mv_material_revision(config: Mapping[str, Any], config_file: str | Path) -> tuple:
    """Cheap GUI invalidation only; execution always hashes authoritative bytes.

    Include paths, file sets and stat revisions so a source/window edit cannot
    leave a late doctor/preflight callback attached to apparently current UI.
    No provider, JSON data parsing or game writes occur here.
    """
    if project_engine(config) != 'rpgmaker_mv':
        return ()
    try:
        settings = read_mv_project_config(config, config_file)
        paths = {settings.window_file,
                 settings.target_root / 'translation_context' / 'coverage_review.json'}
        if settings.occurrence_scope_file is not None:
            paths.add(settings.occurrence_scope_file)
        if settings.reviewed_history_file is not None:
            paths.add(settings.reviewed_history_file)
        for root in (settings.source_root, settings.target_root):
            paths.update((root / 'data').rglob('*.json'))
            paths.update(root / relative for relative in settings.window.material_sha256)
            paths.add(root / 'js' / 'rmmz_core.js')
        revisions = []
        for path in sorted(paths):
            try:
                stat = path.stat()
                revisions.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                revisions.append((str(path), None, None))
        return tuple(revisions)
    except (ValueError, OSError, TypeError) as exc:
        return ('invalid_mv_material', str(exc))
