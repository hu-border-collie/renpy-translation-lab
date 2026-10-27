"""Explicit product engine selection and audited MV material configuration.

Missing engine detects MV materials, otherwise preserving Ren'Py behavior. MV's game_root is a controlled
content copy; source_root and the independently approved hash window are
resolved relative to the configuration file, never inferred from that copy.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
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
    return resolve_project_configuration(value, path)


def detect_project_material(game_root: Any, config_file: str | Path | None = None) -> dict | None:
    """Recognize bounded on-disk engine markers, without approving compatibility.

    Inspect the selected root, its packaged ``www`` and a conventional
    original/work layout. No artifact tree, sibling project, translation
    scope or support window is selected.
    """
    if not isinstance(game_root, (str, Path)) or not str(game_root).strip():
        return None
    root = Path(game_root).expanduser()
    if not root.is_absolute() and config_file is not None:
        root = Path(config_file).resolve().parent / root
    candidates = [root, root / 'www']
    if (root / 'original').is_dir() and (root / 'work').is_dir():
        candidates.extend((root / 'work', root / 'work' / 'www'))
    for content in candidates:
        if not (content / 'data').is_dir():
            continue
        for engine, core in (('rpgmaker_mz', 'rmmz_core.js'), ('rpgmaker_mv', 'rpg_core.js')):
            if (content / 'js' / core).is_file():
                return {'engine': engine, 'content_root': str(content.resolve())}
    game = root / 'game'
    if game.is_dir() and any(any(game.glob(pattern)) for pattern in ('*.rpy', '*.rpyc', '*.rpa')):
        return {'engine': 'renpy', 'content_root': str(root.resolve())}
    return None


def project_engine(config: Mapping[str, Any], *, config_file: str | Path | None = None) -> str:
    """Honor explicit engines; infer MV when an older config omits the field."""
    if 'engine' not in config:
        material = detect_project_material(config.get('game_root'), config_file)
        if material and material['engine'] == 'rpgmaker_mv':
            return 'rpgmaker_mv'
    value = config.get('engine', 'renpy')
    if value not in ("renpy", "rpgmaker_mv"):
        raise ValueError("engine must be renpy or rpgmaker_mv.")
    return str(value)


def resolve_project_configuration(config: Mapping[str, Any], config_file: str | Path) -> dict:
    """Resolve MV engine/content roots in memory, preserving every scope field.

    Readers, including doctor, never persist this projection. Explicit Ren'Py
    selections are kept for mismatch diagnostics; choosing a directory in the
    GUI is a separate saved project-selection action.
    """
    result = copy.deepcopy(dict(config))
    if project_engine(config, config_file=config_file) == 'rpgmaker_mv':
        material = detect_project_material(config.get('game_root'), config_file)
        result['engine'] = 'rpgmaker_mv'
        if material and material['engine'] == 'rpgmaker_mv':
            original = Path(config['game_root']).expanduser()
            if not original.is_absolute():
                original = Path(config_file).resolve().parent / original
            if str(original.resolve()) != material['content_root']:
                from foundation.project_asset_paths import normalize_relative_project_assets_in_config
                normalize_relative_project_assets_in_config(result, str(original.resolve()))
            result['game_root'] = material['content_root']
    return result


def detect_project_engine_mismatch(
    config: Mapping[str, Any],
    config_file: str | Path,
    *,
    game_root: str | Path | None = None,
) -> dict | None:
    """Report RPG Maker markers under a Ren'Py selection, without approving MV.

    Inspect only the selected directory and its packaged ``www`` content root.
    A data directory plus a core file is diagnostic evidence, never an engine
    selection, support window, source-copy inference or configuration write.
    """
    if project_engine(config, config_file=config_file) != 'renpy':
        return None
    value = game_root if game_root is not None else config.get('game_root')
    material = detect_project_material(value, config_file)
    if not material or material['engine'] == 'renpy':
        return None
    detected, content_root = material['engine'], material['content_root']
    action = (
        'Choose the detected MV engine and configure its independent source and audited support window.'
        if detected == 'rpgmaker_mv' else 'RPG Maker MZ is unsupported; select a supported project.'
    )
    return {
        'code': 'engine.project_mismatch', 'severity': 'error',
        'message': f'Configured engine is renpy, but {detected} material was found at {content_root}. {action}',
        'suggested_action': 'configure_project_engine',
        'details': {'configured_engine': 'renpy', 'detected_engine': detected,
                    'content_root': content_root},
    }


class ProjectEngineMismatchError(ValueError):
    """Expose the same engine-selection diagnostic to CLI and GUI consumers."""

    def __init__(self, issue: dict):
        self.issue = issue
        super().__init__(issue['message'])


class MvProjectSetupRequiredError(ValueError):
    """A project execution must wait for explicit MV material configuration."""


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
    config = resolve_project_configuration(config, config_file)
    if project_engine(config) != "rpgmaker_mv":
        raise ValueError("MV configuration requires engine=rpgmaker_mv.")
    section = config.get("rpgmaker_mv")
    if not isinstance(section, Mapping):
        raise ValueError("rpgmaker_mv must be an object.")
    base = Path(config_file).resolve().parent
    source = _path(section.get("source_root"), base, "rpgmaker_mv.source_root")
    source_material = detect_project_material(source)
    if source_material and source_material['engine'] == 'rpgmaker_mv':
        source = Path(source_material['content_root'])
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


def require_engine_operation(
    config: Mapping[str, Any], operation: str, *, config_file: str | Path | None = None,
    game_root: str | Path | None = None,
) -> None:
    """Reject mismatched materials and unsupported MV routes before execution.

    Only project-material routes need this check. Doctor, job queries and
    independent package/index operations remain available without interpreting
    the selected project's game text.
    """
    if config_file is not None:
        config = resolve_project_configuration(config, config_file)
    if config_file is not None and operation in {
        'prepare', 'bootstrap-work', 'generate-template', 'legacy-sync',
        'build', 'build-keywords', 'build-revisions', 'translate-preflight',
        'sync-start', 'sync-resume', 'sync-derive', 'sync-keywords', 'sync-revisions',
        'bootstrap-rag', 'bootstrap-source-index', 'coverage-status', 'coverage-review-import',
        'export-revision-corpus', 'export-project-snapshot', 'work-export',
        'project-analysis-build-structure', 'project-analysis-generate',
    }:
        issue = detect_project_engine_mismatch(config, config_file, game_root=game_root)
        if issue is not None:
            raise ProjectEngineMismatchError(issue)
    if project_engine(config) != "rpgmaker_mv":
        return
    allowed = {"doctor", "translate-preflight", "coverage-status", "coverage-review-import",
               "sync-start", "sync-resume", "sync-status", "sync-cancel", "sync-derive",
               "sync-keywords", "export-keywords", "merge-keywords-to-glossary",
               "bootstrap-source-index", "bootstrap-rag",
               "check", "apply", "quality-report", "usage-report", "usage-import"}
    if operation not in allowed:
        raise ValueError(f"RPG Maker MV narrative mode does not support {operation}; use durable Sync (sync-start -> check -> apply).")
    if config_file is not None and operation in {
        'sync-start', 'sync-resume', 'sync-derive', 'sync-keywords', 'bootstrap-rag', 'bootstrap-source-index',
    }:
        from .project_setup import mv_project_setup_status
        missing = mv_project_setup_status(config, config_file)['missing']
        if missing:
            raise MvProjectSetupRequiredError('MV project preparation requires: ' + ', '.join(missing))


def mv_material_revision(config: Mapping[str, Any], config_file: str | Path) -> tuple:
    """Cheap GUI invalidation only; execution always hashes authoritative bytes.

    Include paths, file sets and stat revisions so a source/window edit cannot
    leave a late doctor/preflight callback attached to apparently current UI.
    No provider, JSON data parsing or game writes occur here.
    """
    config = resolve_project_configuration(config, config_file)
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
