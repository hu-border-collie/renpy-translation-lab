# -*- coding: utf-8 -*-
"""Merge reviewed keyword candidates into glossary.json."""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

GLOSSARY_SECTION_PRESERVE = 'preserve_terms'
GLOSSARY_SECTION_NON_TRANSLATABLE = 'non_translatable_exact'
GLOSSARY_SECTION_NORMALIZE = 'normalize_map'


@dataclass
class MergeAction:
    candidate: dict
    section: str
    source: str
    target: str
    status: str
    reason: str = ''
    existing_target: str = ''
    existing_section: str = ''


@dataclass
class MergeSummary:
    candidates_read: int = 0
    accepted: int = 0
    skipped_duplicate: int = 0
    skipped_low_confidence: int = 0
    skipped_empty: int = 0
    skipped_user: int = 0
    overwritten: int = 0
    preview_lines: list[str] = field(default_factory=list)
    backup_path: str = ''
    glossary_path: str = ''
    candidates_path: str = ''
    dry_run: bool = False
    wrote_glossary: bool = False


def _compact_text(text: object) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def _match_key(text: object) -> str:
    return _compact_text(text).casefold()


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _default_glossary_data() -> dict:
    return {
        GLOSSARY_SECTION_PRESERVE: [],
        GLOSSARY_SECTION_NON_TRANSLATABLE: [],
        GLOSSARY_SECTION_NORMALIZE: {},
    }


def normalize_glossary_data(data: object) -> dict:
    normalized = _default_glossary_data()
    if not isinstance(data, dict):
        return normalized

    preserve_terms = data.get(GLOSSARY_SECTION_PRESERVE)
    if isinstance(preserve_terms, list):
        normalized[GLOSSARY_SECTION_PRESERVE] = [
            _compact_text(item) for item in preserve_terms if _compact_text(item)
        ]

    non_translatable = data.get(GLOSSARY_SECTION_NON_TRANSLATABLE)
    if isinstance(non_translatable, list):
        normalized[GLOSSARY_SECTION_NON_TRANSLATABLE] = [
            _compact_text(item) for item in non_translatable if _compact_text(item)
        ]

    normalize_map = data.get(GLOSSARY_SECTION_NORMALIZE)
    if isinstance(normalize_map, dict):
        cleaned_map = {}
        for source, target in normalize_map.items():
            source_text = _compact_text(source)
            target_text = _compact_text(target)
            if source_text and target_text:
                cleaned_map[source_text] = target_text
        normalized[GLOSSARY_SECTION_NORMALIZE] = cleaned_map

    for key, value in data.items():
        if key in normalized:
            continue
        normalized[key] = value
    return normalized


def load_glossary_file(glossary_path: str) -> dict:
    if not glossary_path or not os.path.isfile(glossary_path):
        return _default_glossary_data()
    try:
        with open(glossary_path, 'r', encoding='utf-8-sig') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'Failed to load glossary file: {glossary_path} ({exc})') from exc
    return normalize_glossary_data(data)


def dump_glossary_file(glossary_path: str, data: dict) -> None:
    parent = os.path.dirname(glossary_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f'{glossary_path}.tmp-{os.getpid()}'
    try:
        with open(tmp_path, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        os.replace(tmp_path, glossary_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_keyword_candidates_jsonl(candidates_path: str) -> list[dict]:
    if not candidates_path or not os.path.isfile(candidates_path):
        raise SystemExit(f'Keyword candidates JSONL not found: {candidates_path}')

    candidates = []
    with open(candidates_path, 'r', encoding='utf-8-sig') as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f'Invalid JSON on line {line_number} of {candidates_path}: {exc}'
                ) from exc
            if isinstance(row, dict):
                candidates.append(row)
    return candidates


def resolve_keyword_candidates_path(target: str) -> str:
    if not target or not str(target).strip():
        raise SystemExit('Keyword candidates JSONL path is required.')

    candidate_path = os.path.abspath(str(target).strip())
    if candidate_path.lower().endswith('.jsonl'):
        if not os.path.isfile(candidate_path):
            raise SystemExit(f'Keyword candidates JSONL not found: {candidate_path}')
        return candidate_path

    manifest_path = candidate_path
    if os.path.isdir(candidate_path):
        manifest_path = os.path.join(candidate_path, 'manifest.json')
    elif not candidate_path.lower().endswith('manifest.json'):
        raise SystemExit(
            'Expected keyword_candidates.jsonl, a package directory, or manifest.json.'
        )

    if not os.path.isfile(manifest_path):
        raise SystemExit(f'Manifest not found: {manifest_path}')

    try:
        with open(manifest_path, 'r', encoding='utf-8-sig') as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'Failed to load manifest: {manifest_path} ({exc})') from exc

    export = manifest.get('keyword_export') if isinstance(manifest, dict) else None
    if not isinstance(export, dict):
        raise SystemExit('Manifest does not contain keyword_export metadata. Run export-keywords first.')

    jsonl_path = export.get('jsonl_path') or ''
    if not isinstance(jsonl_path, str) or not jsonl_path.strip():
        raise SystemExit('Manifest keyword_export.jsonl_path is missing.')

    resolved = os.path.abspath(jsonl_path.strip())
    if not os.path.isfile(resolved):
        raise SystemExit(f'Keyword candidates JSONL from manifest not found: {resolved}')
    return resolved


def classify_candidate_entry(candidate: dict) -> tuple[str, str, str] | None:
    source = _compact_text(candidate.get('source'))
    suggested_target = _compact_text(candidate.get('suggested_target'))
    if not source:
        return None
    if not suggested_target or _match_key(source) == _match_key(suggested_target):
        return (GLOSSARY_SECTION_PRESERVE, source, source)
    return (GLOSSARY_SECTION_NORMALIZE, source, suggested_target)


def _find_preserve_term(glossary: dict, source: str) -> str:
    for term in glossary.get(GLOSSARY_SECTION_PRESERVE) or []:
        if _match_key(term) == _match_key(source):
            return _compact_text(term)
    return ''


def _find_non_translatable(glossary: dict, source: str) -> str:
    for term in glossary.get(GLOSSARY_SECTION_NON_TRANSLATABLE) or []:
        if _match_key(term) == _match_key(source):
            return _compact_text(term)
    return ''


def _find_normalize_target(glossary: dict, source: str) -> tuple[str, str]:
    normalize_map = glossary.get(GLOSSARY_SECTION_NORMALIZE) or {}
    if not isinstance(normalize_map, dict):
        return '', ''
    for existing_source, existing_target in normalize_map.items():
        if _match_key(existing_source) == _match_key(source):
            return _compact_text(existing_source), _compact_text(existing_target)
    return '', ''


def _glossary_has_source(glossary: dict, source: str) -> tuple[str, str, str]:
    preserve = _find_preserve_term(glossary, source)
    if preserve:
        return GLOSSARY_SECTION_PRESERVE, preserve, preserve
    non_translatable = _find_non_translatable(glossary, source)
    if non_translatable:
        return GLOSSARY_SECTION_NON_TRANSLATABLE, non_translatable, non_translatable
    existing_source, existing_target = _find_normalize_target(glossary, source)
    if existing_source:
        return GLOSSARY_SECTION_NORMALIZE, existing_source, existing_target
    return '', '', ''


def plan_merge_action(
    candidate: dict,
    glossary: dict,
    *,
    min_confidence: float = 0.0,
    overwrite: bool = False,
) -> MergeAction | None:
    classified = classify_candidate_entry(candidate)
    if not classified:
        return MergeAction(
            candidate=candidate,
            section='',
            source='',
            target='',
            status='skip_empty',
            reason='missing source',
        )

    section, source, target = classified
    confidence = _coerce_confidence(candidate.get('confidence'))
    if confidence < min_confidence:
        return MergeAction(
            candidate=candidate,
            section=section,
            source=source,
            target=target,
            status='skip_low_confidence',
            reason=f'confidence {confidence:.2f} < {min_confidence:.2f}',
        )

    existing_section, existing_source, existing_target = _glossary_has_source(glossary, source)
    if existing_section:
        if _match_key(existing_target) == _match_key(target):
            return MergeAction(
                candidate=candidate,
                section=section,
                source=source,
                target=target,
                status='skip_duplicate',
                reason=f'already present in {existing_section}',
                existing_target=existing_target,
            )
        if not overwrite:
            return MergeAction(
                candidate=candidate,
                section=section,
                source=source,
                target=target,
                status='skip_duplicate',
                reason=(
                    f'conflicts with {existing_section} entry '
                    f'"{existing_source}" -> "{existing_target}"'
                ),
                existing_target=existing_target,
            )
        return MergeAction(
            candidate=candidate,
            section=section,
            source=source,
            target=target,
            status='overwrite',
            reason=f'overwrite {existing_section} entry',
            existing_target=existing_target,
            existing_section=existing_section,
        )

    return MergeAction(
        candidate=candidate,
        section=section,
        source=source,
        target=target,
        status='accept',
    )


def format_candidate_preview(candidate: dict, action: MergeAction) -> str:
    source = _compact_text(candidate.get('source'))
    suggested_target = _compact_text(candidate.get('suggested_target'))
    category = _compact_text(candidate.get('category')) or 'other'
    confidence = _coerce_confidence(candidate.get('confidence'))
    evidence = _compact_text(candidate.get('evidence'))
    lines = [
        f'source: {source}',
        f'suggested_target: {suggested_target}',
        f'category: {category}',
        f'confidence: {confidence:.2f}',
        f'glossary: {action.section} -> {action.target}',
    ]
    if evidence:
        lines.append(f'evidence: {evidence}')
    if action.existing_target:
        lines.append(f'existing: {action.existing_target}')
    if action.reason:
        lines.append(f'note: {action.reason}')
    return '\n'.join(lines)


def _preview_line_for_action(action: MergeAction) -> str:
    if action.status == 'accept':
        return f'+ {action.section}: {action.source} -> {action.target}'
    if action.status == 'overwrite':
        return (
            f'~ {action.section}: {action.source} '
            f'"{action.existing_target}" -> "{action.target}"'
        )
    return f'= skip ({action.status}): {action.source} ({action.reason})'


def _remove_from_section(glossary: dict, section: str, source: str) -> None:
    if section == GLOSSARY_SECTION_NORMALIZE:
        normalize_map = dict(glossary.get(GLOSSARY_SECTION_NORMALIZE) or {})
        for existing_source in list(normalize_map.keys()):
            if _match_key(existing_source) == _match_key(source):
                normalize_map.pop(existing_source, None)
        glossary[GLOSSARY_SECTION_NORMALIZE] = normalize_map
        return

    if section in (GLOSSARY_SECTION_PRESERVE, GLOSSARY_SECTION_NON_TRANSLATABLE):
        glossary[section] = [
            term for term in glossary.get(section) or []
            if _match_key(term) != _match_key(source)
        ]


def apply_merge_action(glossary: dict, action: MergeAction) -> None:
    if action.status not in {'accept', 'overwrite'}:
        return

    if action.status == 'overwrite' and action.existing_section:
        _remove_from_section(glossary, action.existing_section, action.source)

    if action.section == GLOSSARY_SECTION_PRESERVE:
        terms = list(glossary.get(GLOSSARY_SECTION_PRESERVE) or [])
        if not any(_match_key(term) == _match_key(action.source) for term in terms):
            terms.append(action.source)
        glossary[GLOSSARY_SECTION_PRESERVE] = terms
        return

    if action.section == GLOSSARY_SECTION_NORMALIZE:
        normalize_map = dict(glossary.get(GLOSSARY_SECTION_NORMALIZE) or {})
        normalize_map[action.source] = action.target
        glossary[GLOSSARY_SECTION_NORMALIZE] = normalize_map


def backup_glossary_file(glossary_path: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup_path = f'{glossary_path}.bak-{timestamp}'
    shutil.copy2(glossary_path, backup_path)
    return backup_path


def merge_keywords_to_glossary(
    candidates_path: str,
    glossary_path: str,
    *,
    dry_run: bool = False,
    min_confidence: float = 0.0,
    accept_confidence: float | None = None,
    overwrite: bool = False,
    interactive: bool = True,
    backup: bool = True,
    input_func: Callable[[str], str] = input,
) -> MergeSummary:
    if dry_run:
        interactive = False

    candidates = load_keyword_candidates_jsonl(candidates_path)
    glossary_before = load_glossary_file(glossary_path)
    glossary_after = json.loads(json.dumps(glossary_before, ensure_ascii=False))

    summary = MergeSummary(
        candidates_read=len(candidates),
        glossary_path=os.path.abspath(glossary_path),
        candidates_path=os.path.abspath(candidates_path),
        dry_run=dry_run,
    )

    quit_requested = False
    for candidate in candidates:
        if quit_requested:
            break

        action = plan_merge_action(
            candidate,
            glossary_after,
            min_confidence=min_confidence,
            overwrite=overwrite,
        )
        if action is None:
            continue

        if action.status == 'skip_empty':
            summary.skipped_empty += 1
            summary.preview_lines.append(_preview_line_for_action(action))
            continue
        if action.status == 'skip_low_confidence':
            summary.skipped_low_confidence += 1
            summary.preview_lines.append(_preview_line_for_action(action))
            continue
        if action.status == 'skip_duplicate':
            summary.skipped_duplicate += 1
            summary.preview_lines.append(_preview_line_for_action(action))
            continue

        should_apply = False
        auto_accept = (
            accept_confidence is not None
            and _coerce_confidence(candidate.get('confidence')) >= accept_confidence
        )
        if auto_accept:
            should_apply = True
        elif interactive:
            print(format_candidate_preview(candidate, action))
            while True:
                prompt = 'Add to glossary? [y/N/q]: '
                choice = _compact_text(input_func(prompt)).lower()
                if choice in {'y', 'yes'}:
                    should_apply = True
                    break
                if choice in {'n', 'no', ''}:
                    should_apply = False
                    break
                if choice in {'q', 'quit'}:
                    print('Stopping review; keeping already accepted entries.')
                    summary.preview_lines.append('= stopped review early by user')
                    quit_requested = True
                    break
                print('Please answer y, n, or q.')
            if quit_requested:
                break
        else:
            should_apply = True

        if not should_apply:
            summary.skipped_user += 1
            summary.preview_lines.append(
                f'= skip (user): {action.source} -> {action.target}'
            )
            continue

        apply_merge_action(glossary_after, action)
        if action.status == 'overwrite':
            summary.overwritten += 1
        summary.accepted += 1
        summary.preview_lines.append(_preview_line_for_action(action))

    print('Keyword glossary merge preview:')
    for line in summary.preview_lines:
        print(line)
    print(
        'Summary: '
        f'read={summary.candidates_read}, accepted={summary.accepted}, '
        f'duplicates={summary.skipped_duplicate}, '
        f'low_confidence={summary.skipped_low_confidence}, '
        f'user_skipped={summary.skipped_user}, overwritten={summary.overwritten}'
    )

    if dry_run:
        print('Dry run: glossary was not modified.')
        return summary

    if summary.accepted == 0:
        print('No glossary changes to write.')
        return summary

    if os.path.isfile(glossary_path) and backup:
        summary.backup_path = backup_glossary_file(glossary_path)
        print(f'Backup: {summary.backup_path}')

    dump_glossary_file(glossary_path, glossary_after)
    summary.wrote_glossary = True
    print(f'Updated glossary: {glossary_path}')
    return summary


def build_merge_keywords_cli_command(
    candidates_path: str,
    *,
    glossary_path: str = '',
    dry_run: bool = False,
    min_confidence: float = 0.0,
    accept_confidence: float | None = None,
    overwrite: bool = False,
    yes: bool = False,
    no_backup: bool = False,
) -> list[str]:
    command = ['merge-keywords-to-glossary', candidates_path]
    if glossary_path:
        command.extend(['--glossary', glossary_path])
    if dry_run:
        command.append('--dry-run')
    if min_confidence > 0:
        command.extend(['--min-confidence', str(min_confidence)])
    if accept_confidence is not None:
        command.extend(['--accept-confidence', str(accept_confidence)])
    if overwrite:
        command.append('--overwrite')
    if yes:
        command.append('--yes')
    if no_backup:
        command.append('--no-backup')
    return command