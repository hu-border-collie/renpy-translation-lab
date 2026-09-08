"""Request-bound model text views; canonical sources are never mutated.

Absent metadata is the explicit legacy path. Present metadata must reproduce
exactly under the current rules, including request and item identity. Digests
are integrity checks, not signatures; the existing artifact trust boundary
and check -> apply gates remain authoritative.
"""

from collections import Counter
from dataclasses import replace
import hashlib
import json
import re

from engine_adapters import structure_rules

VERSION = 2
KEY = 'structure_protection'
INSTRUCTION = '\nCopy every __RTL_ placeholder exactly once into its translation. Never edit placeholders. Only placeholders containing _variable_ may move with language order; keep all other placeholders in their original order.'
MARKER = re.compile(r'__RTL_[A-Za-z0-9_]*?__')
LITERAL_MARKER = re.compile(r'__RTL_[A-Za-z0-9_]*')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def request_scope(request):
    """Recompute the model-context binding instead of trusting a stored scope."""
    payload = request.to_dict() if hasattr(request, 'to_dict') else request
    prompt = str(payload.get('user_prompt') or '')
    metadata = (payload.get('transport_metadata') or {}).get(KEY) or {}
    if isinstance(metadata, dict):
        replacements = {}
        for mapping in (metadata.get('items') or {}).values():
            if isinstance(mapping, dict):
                for entry in mapping.get('entries') or []:
                    replacements[entry['marker']] = json.dumps(entry['value'], ensure_ascii=False)[1:-1]
        prompt = MARKER.sub(lambda match: replacements.get(match.group(), match.group()), prompt)
    return digest({
        'prompt': prompt,
        'context': payload.get('context_assembly') or {},
        'system': str(payload.get('system_instruction') or '').removesuffix(INSTRUCTION),
        'chunk': payload.get('chunk_id') or '',
    })


def protect(source, *, engine, request_id, item_id, scope=None, literal_brackets=False):
    """Build a deterministic, collision-free view and its identity-bound map."""
    binding = dict(version=VERSION, engine=engine, request_id=request_id,
                   item_id=item_id, source_digest=digest(source), scope=scope or request_id, literal_brackets=bool(literal_brackets))
    namespace = digest({key: value for key, value in binding.items() if key != 'request_id'})[:24]
    while f'__RTL_{namespace}_' in source:
        namespace = digest(namespace)[:24]
    entries, parts, end = [], [], 0
    spans = list(structure_rules.spans(source, engine, literal_brackets=literal_brackets))
    for match in LITERAL_MARKER.finditer(source):
        if not any(start < match.end() and stop > match.start() for start, stop, _ in spans):
            spans.append((match.start(), match.end(), 'literal'))
    for index, (start, stop, kind) in enumerate(sorted(spans)):
        marker = f'__RTL_{namespace}_{kind}_{index}__'
        entries.append(dict(marker=marker, value=source[start:stop], kind=kind))
        parts.extend((source[end:start], marker))
        end = stop
    parts.append(source[end:])
    metadata = {**binding, 'entries': entries}
    metadata['digest'] = digest(metadata)
    return ''.join(parts), metadata


def model_units(units, *, engine, request_id, scope=None):
    """Copy translation units for prompt rendering, retaining canonical objects."""
    views, maps = [], {}
    for unit in units:
        view, mapping = protect(unit.text, engine=engine, request_id=request_id, item_id=unit.id, scope=scope,
                                literal_brackets=unit.metadata.get('tyrano_literal_brackets', False))
        views.append(replace(unit, text=view))
        maps[unit.id] = mapping
    return views, dict(version=VERSION, engine=engine, request_id=request_id, scope=scope or request_id, items=maps)


def restore(text, mapping, *, source, engine, request_id, item_id, scope=None, literal_brackets=False):
    """Restore only exact, once-present markers; never repair missing tokens."""
    _, expected = protect(source, engine=engine, request_id=request_id, item_id=item_id, scope=scope, literal_brackets=literal_brackets)
    if mapping != expected:
        raise ValueError('protection.mapping_mismatch')
    markers = {entry['marker']: entry['value'] for entry in mapping['entries']}
    for marker in markers:
        count = text.count(marker)
        if count == 0:
            raise ValueError('protection.missing_token')
        if count > 1:
            raise ValueError('protection.duplicate_token')
    # Literal marker-shaped source text is legal and is not our namespace.
    extras = set(MARKER.findall(text)) - set(markers) - set(MARKER.findall(source))
    if extras:
        raise ValueError('protection.extra_token')
    restored = MARKER.sub(lambda match: markers.get(match.group(), match.group()), text)
    validate_literal_markers(source, restored)
    structure_rules.validate(source, restored, engine, literal_brackets=literal_brackets)
    return restored


def validate_literal_markers(source, text):
    """Reject residual or invented reserved markers, preserving source literals."""
    if Counter(LITERAL_MARKER.findall(source)) != Counter(LITERAL_MARKER.findall(text)):
        raise ValueError('protection.modified_token')


def validate_parent(request, units):
    """Reject stale or cross-item maps before deriving a new child request."""
    metadata = request.transport_metadata[KEY]
    if not isinstance(metadata, dict) or metadata.get('version') != VERSION:
        raise ValueError('protection.stale_mapping')
    if metadata.get('request_id') != request.request_id or metadata.get('scope') != request_scope(request):
        raise ValueError('protection.mapping_mismatch')
    for unit in units:
        _, expected = protect(unit.text, engine=metadata['engine'],
                              request_id=request.request_id, item_id=unit.id,
                              scope=request_scope(request),
                              literal_brackets=unit.metadata.get('tyrano_literal_brackets', False))
        if metadata['items'].get(unit.id) != expected:
            raise ValueError('protection.mapping_mismatch')


def validate_report(report, request, units, *, canonical=False):
    """Restore a shared contract report and reject failed items for targeted retry.

    Raw provider envelopes remain owned by their caller. Canonical persisted
    results are checked without applying a map a second time.
    """
    import translation_core

    payload = request.to_dict() if hasattr(request, 'to_dict') else dict(request or {})
    transport = payload.get('transport_metadata') or {}
    if KEY not in transport:
        return report
    metadata = transport[KEY]
    units = translation_core.units_from_items(units)
    sources = {unit.id: unit for unit in units}
    request_id = payload.get('request_id')
    error = ''
    if not isinstance(metadata, dict) or metadata.get('version') != VERSION:
        error = 'protection.stale_mapping'
    elif metadata.get('request_id') != request_id or metadata.get('scope') != request_scope(payload) or set(metadata.get('items') or {}) != set(payload.get('expected_ids') or []):
        error = 'protection.mapping_mismatch'
    accepted = []
    for item in report.items:
        item_id = item['id']
        try:
            if error:
                raise ValueError(error)
            engine = metadata['engine']
            unit = sources[item_id]
            source = unit.text
            literal_brackets = unit.metadata.get('tyrano_literal_brackets', False)
            _, expected = protect(source, engine=engine, request_id=request_id, item_id=item_id, scope=request_scope(payload), literal_brackets=literal_brackets)
            if metadata['items'].get(item_id) != expected:
                raise ValueError('protection.mapping_mismatch')
            text = item['translation']
            if canonical:
                validate_literal_markers(source, text)
                structure_rules.validate(source, text, engine, literal_brackets=literal_brackets)
            else:
                text = restore(text, expected, source=source, engine=engine,
                               request_id=request_id, item_id=item_id, scope=request_scope(payload), literal_brackets=literal_brackets)
            accepted.append({**item, 'translation': text})
        except (ValueError, KeyError, TypeError) as exc:
            reason = str(exc)
            if not reason.startswith('protection.'):
                reason = 'protection.mapping_mismatch'
            report.issues.append(translation_core.ModelContractIssue(
                reason_code=reason, item_id=item_id))
    report.items = accepted
    report.valid_ids = [item['id'] for item in accepted]
    report.retry_ids = [item_id for item_id in report.expected_ids if item_id not in report.valid_ids]
    return report
