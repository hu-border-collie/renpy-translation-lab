"""Bind an explicit occurrence allowlist to the independently audited MV source.

This narrows requests only. Discovery, coverage, source/target snapshots and
writeback freshness continue to include every supplied material file.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .project_config import _reject_constant, _unique_object


@dataclass(frozen=True)
class MvOccurrenceScope:
    """Exact file-content digest and canonical item IDs authorized by one allowlist."""
    digest: str
    item_ids: frozenset[str]


def read_occurrence_scope(path: Path) -> tuple[str, dict]:
    """Read one strict schema-1 allowlist and bind its exact UTF-8 bytes."""
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f'MV occurrence scope cannot be read: {exc}') from exc
    if (not isinstance(payload, dict) or set(payload) != {'schema_version', 'source_files', 'occurrences'}
            or type(payload.get('schema_version')) is not int or payload['schema_version'] != 1
            or not isinstance(payload['source_files'], dict)
            or not isinstance(payload['occurrences'], list) or not payload['occurrences']):
        raise ValueError('Invalid MV occurrence scope schema; expected nonempty version 1 allowlist.')
    files = payload['source_files']
    if any(not isinstance(key, str) or not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
           for key, value in files.items()):
        raise ValueError('MV occurrence scope requires exact source file SHA-256 values.')
    seen, referenced = set(), set()
    for row in payload['occurrences']:
        if (not isinstance(row, dict) or set(row) != {'file_rel_path', 'json_path', 'source_sha256'}
                or not isinstance(row['file_rel_path'], str) or row['file_rel_path'] not in files
                or not isinstance(row['source_sha256'], str)
                or not re.fullmatch(r'[0-9a-f]{64}', row['source_sha256'])
                or not isinstance(row['json_path'], list) or not row['json_path']
                or any(not ((type(part) is int and part >= 0) or (type(part) is str and part))
                       for part in row['json_path'])):
            raise ValueError('Invalid MV occurrence scope position or source digest.')
        key = row['file_rel_path'], tuple(row['json_path'])
        if key in seen:
            raise ValueError('Duplicate MV occurrence scope position.')
        seen.add(key)
        referenced.add(row['file_rel_path'])
    if referenced != set(files):
        raise ValueError('MV occurrence scope source files must exactly match its positions.')
    return hashlib.sha256(raw).hexdigest(), payload


def bind_occurrence_scope(path: Path, project, occurrences) -> MvOccurrenceScope:
    """Reject stale, absent, mirror or unsupported positions before planning."""
    digest, payload = read_occurrence_scope(path)
    documents = project.document_by_path()
    for relative, expected in payload['source_files'].items():
        if relative not in documents or documents[relative].sha256 != expected:
            raise ValueError(f'MV occurrence scope source file changed or missing: {relative}.')
    by_position = {(row.unit.file_rel_path, tuple(row.locator.locator['json_path'])): row
                   for row in occurrences}
    selected = set()
    for record in payload['occurrences']:
        key = record['file_rel_path'], tuple(record['json_path'])
        row = by_position.get(key)
        if row is None or hashlib.sha256(row.unit.source_text.encode('utf-8')).hexdigest() != record['source_sha256']:
            raise ValueError('MV occurrence scope position is absent, unsupported, a mirror, or its source changed.')
        selected.add(row.unit.id)
    return MvOccurrenceScope(digest, frozenset(selected))
