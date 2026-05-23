# -*- coding: utf-8 -*-
"""Shared translation pipeline primitives for sync and batch workflows.

This module deliberately keeps public CLI and manifest shapes out at the edge:
callers adapt their legacy task/chunk/result dictionaries into these internal
objects, then serialize back to the same dictionaries when writing manifests or
applying replacements.
"""

from dataclasses import dataclass, field
import json
import math
import re

import prompt_context


CORE_SCHEMA_VERSION = 1

MODE_TRANSLATION = 'translation'
MODE_KEYWORD_EXTRACTION = 'keyword_extraction'
MODE_REVISION = 'revision'

KEYWORD_CATEGORY_ORDER = [
    'term',
    'character',
    'place',
    'item',
    'ability',
    'concept',
    'relationship',
    'style',
    'other',
]
KEYWORD_CATEGORIES = set(KEYWORD_CATEGORY_ORDER)


@dataclass
class TranslationUnit:
    id: str
    mode: str = MODE_TRANSLATION
    text: str = ''
    source: str = ''
    current_translation: str = ''
    file_rel_path: str = ''
    file_path: str = ''
    line: int = 0
    line_number: int = 0
    start: int = 0
    end: int = 0
    prefix: str = ''
    quote: str = '"'
    speaker_id: str = ''
    speaker: str = ''
    progress_entry: str = ''
    metadata: dict = field(default_factory=dict)

    @property
    def source_text(self):
        return self.source or self.text

    @property
    def display_line_number(self):
        # ``line_number`` is 1-indexed. A zero value means "not supplied", so
        # display falls back to the internal 0-indexed ``line`` value.
        if self.line_number:
            return self.line_number
        return self.line + 1 if self.line >= 0 else 0


@dataclass
class ContextWindow:
    before: list = field(default_factory=list)
    after: list = field(default_factory=list)


@dataclass
class ContextBundle:
    glossary_hits: list = field(default_factory=list)
    history_hits: list = field(default_factory=list)
    story_hits: object = None
    rag_stats: dict = field(default_factory=dict)


@dataclass
class ModelResult:
    id: str = ''
    mode: str = MODE_TRANSLATION
    translation: str = ''
    should_update: bool = False
    revised_translation: str = ''
    reason: str = ''
    source: str = ''
    suggested_target: str = ''
    category: str = 'other'
    confidence: float = 0.0
    evidence: str = ''
    source_item_ids: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_legacy_dict(self):
        if self.mode == MODE_REVISION:
            return {
                'id': self.id,
                'should_update': self.should_update,
                'revised_translation': self.revised_translation,
                'reason': self.reason,
            }
        if self.mode == MODE_KEYWORD_EXTRACTION:
            return {
                'source': self.source,
                'suggested_target': self.suggested_target,
                'category': self.category,
                'confidence': self.confidence,
                'evidence': self.evidence,
                'source_item_ids': list(self.source_item_ids),
            }
        return {'id': self.id, 'translation': self.translation}


@dataclass
class WritebackAction:
    mode: str
    file_rel_path: str
    line: int
    start: int
    end: int
    replacement: str
    prefix: str = ''
    quote: str = '"'
    expected_text: str = ''
    item_id: str = ''
    chunk_key: str = ''


def compact_text(text):
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def _coerce_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_line(item):
    """Return the internal 0-indexed line; legacy line_number is 1-indexed."""
    if 'line' in item:
        return max(0, _coerce_int(item.get('line'), 0))
    if 'line_number' in item:
        return max(0, _coerce_int(item.get('line_number'), 1) - 1)
    return 0


def _coerce_line_number(item, line):
    if 'line_number' in item:
        return max(0, _coerce_int(item.get('line_number'), 0))
    return 0


def _metadata_for(item, known_keys):
    return {key: value for key, value in dict(item).items() if key not in known_keys}


_TRANSLATION_KEYS = {
    'id',
    'mode',
    'text',
    'source',
    'current_translation',
    'file_rel_path',
    'file_path',
    'line',
    'line_number',
    'start',
    'end',
    'prefix',
    'quote',
    'speaker_id',
    'speaker',
    'progress_entry',
}


def unit_from_translation_item(item, file_rel_path='', file_path='', mode=MODE_TRANSLATION):
    item = dict(item or {})
    line = _coerce_line(item)
    text = str(item.get('text') if item.get('text') is not None else item.get('source') or '')
    source = str(item.get('source') or text)
    speaker_id = str(item.get('speaker_id') or item.get('speaker') or '')
    return TranslationUnit(
        id=str(item.get('id') or ''),
        mode=mode or MODE_TRANSLATION,
        text=text,
        source=source,
        current_translation=str(item.get('current_translation') or ''),
        file_rel_path=str(item.get('file_rel_path') or file_rel_path or ''),
        file_path=str(item.get('file_path') or file_path or ''),
        line=line,
        line_number=_coerce_line_number(item, line),
        start=_coerce_int(item.get('start'), 0),
        end=_coerce_int(item.get('end'), 0),
        prefix=str(item.get('prefix') or ''),
        quote=str(item.get('quote') or '"'),
        speaker_id=speaker_id,
        speaker=str(item.get('speaker') or speaker_id),
        progress_entry=str(item.get('progress_entry') or ''),
        metadata=_metadata_for(item, _TRANSLATION_KEYS),
    )


def unit_from_sync_task(task, file_rel_path='', file_path=''):
    return unit_from_translation_item(task, file_rel_path=file_rel_path, file_path=file_path)


def unit_from_revision_item(item, file_rel_path='', file_path=''):
    item = dict(item or {})
    unit = unit_from_translation_item(
        item,
        file_rel_path=file_rel_path,
        file_path=file_path,
        mode=MODE_REVISION,
    )
    source = str(item.get('source') or '')
    if source:
        # Revision manifests may carry an empty text field; reviewers should
        # still see the original source string in that case.
        unit.source = source
        unit.text = str(item.get('text') or source)
    return unit


def unit_from_keyword_item(item, file_rel_path='', file_path=''):
    unit = unit_from_translation_item(
        item,
        file_rel_path=file_rel_path,
        file_path=file_path,
        mode=MODE_KEYWORD_EXTRACTION,
    )
    if isinstance(item, dict) and 'translation_line_number' in item:
        unit.metadata['translation_line_number'] = item.get('translation_line_number')
    return unit


def unit_from_manifest_item(item, mode=MODE_TRANSLATION, chunk=None):
    chunk = chunk or {}
    actual_mode = mode or chunk.get('mode') or MODE_TRANSLATION
    if actual_mode == MODE_REVISION:
        return unit_from_revision_item(
            item,
            file_rel_path=chunk.get('file_rel_path', ''),
            file_path=chunk.get('file_path', ''),
        )
    if actual_mode == MODE_KEYWORD_EXTRACTION:
        return unit_from_keyword_item(
            item,
            file_rel_path=chunk.get('file_rel_path', ''),
            file_path=chunk.get('file_path', ''),
        )
    return unit_from_translation_item(
        item,
        file_rel_path=chunk.get('file_rel_path', ''),
        file_path=chunk.get('file_path', ''),
    )


def units_from_items(items, mode=MODE_TRANSLATION, file_rel_path='', file_path=''):
    units = []
    for item in items or []:
        if isinstance(item, TranslationUnit):
            units.append(item)
        elif mode == MODE_REVISION:
            units.append(unit_from_revision_item(item, file_rel_path=file_rel_path, file_path=file_path))
        elif mode == MODE_KEYWORD_EXTRACTION:
            units.append(unit_from_keyword_item(item, file_rel_path=file_rel_path, file_path=file_path))
        else:
            units.append(unit_from_translation_item(item, file_rel_path=file_rel_path, file_path=file_path))
    return units


def unit_to_translation_item(unit):
    item = {
        'id': unit.id,
        'text': unit.text,
        'line': unit.line,
        'start': unit.start,
        'end': unit.end,
        'prefix': unit.prefix,
        'quote': unit.quote,
        'speaker_id': unit.speaker_id,
        'speaker': unit.speaker,
    }
    return item


def unit_to_revision_item(unit):
    item = {
        'id': unit.id,
        'text': unit.source_text,
        'source': unit.source_text,
        'current_translation': unit.current_translation,
        'file_rel_path': unit.file_rel_path,
        'line': unit.line,
        'line_number': unit.display_line_number,
        'start': unit.start,
        'end': unit.end,
        'prefix': unit.prefix,
        'quote': unit.quote,
    }
    if unit.speaker_id:
        item['speaker_id'] = unit.speaker_id
    return item


def unit_to_keyword_item(unit):
    item = {
        'id': unit.id,
        'text': unit.text,
        'file_rel_path': unit.file_rel_path,
        'line_number': unit.display_line_number,
    }
    translation_line_number = unit.metadata.get('translation_line_number')
    if translation_line_number is not None:
        item['translation_line_number'] = translation_line_number
    if unit.speaker_id:
        item['speaker_id'] = unit.speaker_id
    return item


def legacy_item_from_unit(unit, mode=None):
    actual_mode = mode or unit.mode
    if actual_mode == MODE_REVISION:
        return unit_to_revision_item(unit)
    if actual_mode == MODE_KEYWORD_EXTRACTION:
        return unit_to_keyword_item(unit)
    return unit_to_translation_item(unit)


def format_context_block(lines, empty_label='(none)'):
    if not lines:
        return empty_label
    rendered = []
    for line in lines:
        if isinstance(line, TranslationUnit):
            rendered.append(line.source_text)
        else:
            rendered.append(str(line))
    return '\n'.join(f'- {line}' for line in rendered if line) or empty_label


def format_revision_context_block(items, empty_label='(none)'):
    if not items:
        return empty_label
    lines = []
    for item in items:
        if isinstance(item, TranslationUnit):
            source = compact_text(item.source_text)
            current = compact_text(item.current_translation)
        else:
            source = compact_text((item or {}).get('source', ''))
            current = compact_text((item or {}).get('current_translation', ''))
        if source or current:
            lines.append(f'- {source} => {current}')
    return '\n'.join(lines) if lines else empty_label


def build_context_bundle(glossary_hits=None, history_hits=None, story_hits=None, rag_stats=None):
    return ContextBundle(
        glossary_hits=list(glossary_hits or []),
        history_hits=list(history_hits or []),
        story_hits=story_hits,
        rag_stats=dict(rag_stats or {}),
    )


def build_reference_blocks(
    context_bundle,
    history_char_limit=220,
    story_char_limit=1200,
    include_translation_memory=True,
    include_source_text=True,
    story_block_suffix='\n\n',
):
    context_bundle = context_bundle or ContextBundle()
    return prompt_context.build_reference_blocks(
        include_translation_memory=include_translation_memory,
        glossary_hits=context_bundle.glossary_hits,
        history_hits=context_bundle.history_hits,
        story_hits=context_bundle.story_hits,
        history_char_limit=history_char_limit,
        story_char_limit=story_char_limit,
        include_source_text=include_source_text,
        story_block_suffix=story_block_suffix,
    )


def build_translation_system_instruction(preserve_terms, macro_setting=''):
    glossary = ', '.join(str(term) for term in preserve_terms or [])
    return (
        'Setting:\n'
        f'{macro_setting or ""}\n\n'
        'Task:\n'
        'Translate only TARGET lines into Simplified Chinese. CONTEXT lines are reference only.\n'
        f'Keep these terms unchanged: {glossary}\n'
        "Keep names, Ren'Py tags, placeholders, variables, and format strings unchanged.\n"
        'Return JSON only. Preserve every id exactly. Item count must match. '
        'translation must contain only the translated Chinese text.'
    )


def build_translation_user_prompt(
    context_window,
    units,
    context_bundle=None,
    history_char_limit=220,
    story_char_limit=1200,
    include_translation_memory=True,
    include_source_text=True,
    story_block_suffix='\n\n',
):
    context_window = context_window or ContextWindow()
    units = units_from_items(units, MODE_TRANSLATION)
    target_payload = json.dumps(
        [{'id': unit.id, 'text': unit.text} for unit in units],
        ensure_ascii=False,
        separators=(',', ':'),
    )
    return ''.join(
        [
            build_reference_blocks(
                context_bundle,
                history_char_limit=history_char_limit,
                story_char_limit=story_char_limit,
                include_translation_memory=include_translation_memory,
                include_source_text=include_source_text,
                story_block_suffix=story_block_suffix,
            ),
            f'CONTEXT BEFORE:\n{format_context_block(context_window.before, "(none)")}\n\n',
            f'TARGET:\n{target_payload}\n\n',
            f'CONTEXT AFTER:\n{format_context_block(context_window.after, "(none)")}\n\n',
            'Return the result now.',
        ]
    )


def build_sync_translation_prompt(
    units,
    preserve_terms,
    context_bundle=None,
    history_char_limit=220,
    story_char_limit=1200,
    include_translation_memory=True,
):
    units = units_from_items(units, MODE_TRANSLATION)
    glossary = ', '.join(str(term) for term in preserve_terms or [])
    payload = json.dumps(
        [{'id': unit.id, 'text': unit.text} for unit in units],
        ensure_ascii=False,
    )
    reference_body = build_reference_blocks(
        context_bundle,
        history_char_limit=history_char_limit,
        story_char_limit=story_char_limit,
        include_translation_memory=include_translation_memory,
        include_source_text=False,
        story_block_suffix='\n',
    )
    reference_blocks = ''
    if reference_body:
        reference_blocks = (
            '\nReference blocks:\n'
            f'{reference_body}'
            'Use reference blocks only as style, terminology, and continuity reference; '
            'ignore them when unrelated.\n'
        )
    return (
        "You are translating a Ren'Py visual novel into Simplified Chinese (zh-CN).\n"
        'Rules:\n'
        f'1. Preserve these terms exactly (do not translate): {glossary}\n'
        '1.1 Keep all person names in English; do not translate names.\n'
        "2. Preserve Ren'Py tags like {i}, {/i}, {color=...}, [name], %s.\n"
        '3. Output plain Chinese text. No markdown, no Pinyin, no explanations.\n'
        '4. Return ONLY a JSON array matching the requested id/translation structure.\n'
        f'{reference_blocks}'
        f'Input JSON:\n{payload}'
    )


def build_revision_system_instruction(preserve_terms, macro_setting=''):
    glossary = ', '.join(str(term) for term in preserve_terms or [])
    return (
        'Setting:\n'
        f'{macro_setting or ""}\n\n'
        'Task:\n'
        "Review existing Simplified Chinese Ren'Py TL translations. "
        'For each TARGET item, decide whether the current translation should be revised. '
        "Preserve meaning, tone, Ren'Py tags, placeholders, variables, format strings, and locked terms. "
        f'Keep these terms unchanged: {glossary}\n'
        'If the current translation is already acceptable, set should_update=false and repeat it as revised_translation. '
        'If it needs a change, set should_update=true and provide only the revised Chinese translation. '
        'Return JSON only. Preserve every id exactly. Item count must match.'
    )


def build_revision_user_prompt(
    context_window,
    units,
    context_bundle=None,
    history_char_limit=220,
    story_char_limit=1200,
    include_source_text=True,
):
    context_window = context_window or ContextWindow()
    units = units_from_items(units, MODE_REVISION)
    target_payload = json.dumps(
        [
            {
                'id': unit.id,
                'file': unit.file_rel_path,
                'line': unit.display_line_number,
                'speaker_id': unit.speaker_id,
                'source': unit.source_text,
                'current_translation': unit.current_translation,
            }
            for unit in units
        ],
        ensure_ascii=False,
        separators=(',', ':'),
    )
    return ''.join(
        [
            build_reference_blocks(
                context_bundle,
                history_char_limit=history_char_limit,
                story_char_limit=story_char_limit,
                include_translation_memory=True,
                include_source_text=include_source_text,
            ),
            f'CONTEXT BEFORE:\n{format_revision_context_block(context_window.before, "(none)")}\n\n',
            f'TARGET:\n{target_payload}\n\n',
            f'CONTEXT AFTER:\n{format_revision_context_block(context_window.after, "(none)")}\n\n',
            'Return objects with id, should_update, revised_translation, and reason.',
        ]
    )


def build_keyword_glossary_block(preserve_terms=None, normalize_map=None, non_translatable_terms=None):
    lines = []
    for term in preserve_terms or []:
        if isinstance(term, str) and term.strip():
            lines.append(f'- Preserve: {term.strip()}')
    for source, target in (normalize_map or {}).items():
        if source:
            lines.append(f'- Existing mapping: {source} -> {target}')
    for term in sorted(non_translatable_terms or []):
        if isinstance(term, str) and term.strip():
            lines.append(f'- Non-translatable: {term.strip()}')
    return '\n'.join(lines) if lines else '(none)'


def build_keyword_system_instruction(
    preserve_terms=None,
    normalize_map=None,
    non_translatable_terms=None,
    macro_setting='',
    max_candidates_per_chunk=12,
):
    max_candidates = max(1, _coerce_int(max_candidates_per_chunk, 12))
    return (
        'Setting:\n'
        f'{macro_setting or ""}\n\n'
        'Existing glossary entries:\n'
        f'{build_keyword_glossary_block(preserve_terms, normalize_map, non_translatable_terms)}\n\n'
        'Task:\n'
        "Extract glossary or story-memory keyword candidates from Ren'Py TL source text. "
        'Do not translate full lines. Return only high-value terms, names, places, items, concepts, abilities, '
        'relationship labels, or recurring phrasing that a human may want to add to glossary.json or story_graph.json.\n'
        f'Return at most {max_candidates} candidates for this chunk. '
        'Avoid generic words, common function words, UI filler, and candidates already covered by existing glossary entries. '
        'Set source_item_ids to the input id values that support the candidate. '
        'Use concise evidence that cites the relevant input id or phrase.\n'
        'Also write a compact chunk_summary in Chinese that summarizes only the visible story events in this chunk. '
        'Use 1-3 sentences, avoid invented continuity, and leave chunk_summary empty if the lines do not contain usable story content. '
        'Set summary_evidence_item_ids to the input ids that support the summary. Return JSON only.'
    )


def build_keyword_user_prompt(units):
    units = units_from_items(units, MODE_KEYWORD_EXTRACTION)
    target_payload = json.dumps(
        [
            {
                'id': unit.id,
                'file': unit.file_rel_path,
                'line': unit.display_line_number,
                'speaker_id': unit.speaker_id,
                'text': unit.text,
            }
            for unit in units
        ],
        ensure_ascii=False,
        separators=(',', ':'),
    )
    return (
        'TARGET LINES:\n'
        f'{target_payload}\n\n'
        'Return a JSON object with candidates, chunk_summary, and summary_evidence_item_ids. '
        'Each candidate must include source, suggested_target, category, confidence, evidence, and source_item_ids.'
    )


def build_translation_schema(units):
    units = units_from_items(units, MODE_TRANSLATION)
    target_ids = [unit.id for unit in units]
    return {
        'type': 'array',
        'minItems': len(units),
        'maxItems': len(units),
        'items': {
            'type': 'object',
            'required': ['id', 'translation'],
            'additionalProperties': False,
            'properties': {
                'id': {'type': 'string', 'enum': target_ids},
                'translation': {'type': 'string'},
            },
        },
    }


def build_revision_schema(units):
    units = units_from_items(units, MODE_REVISION)
    target_ids = [unit.id for unit in units]
    return {
        'type': 'array',
        'minItems': len(units),
        'maxItems': len(units),
        'items': {
            'type': 'object',
            'required': ['id', 'should_update', 'revised_translation', 'reason'],
            'additionalProperties': False,
            'properties': {
                'id': {'type': 'string', 'enum': target_ids},
                'should_update': {'type': 'boolean'},
                'revised_translation': {'type': 'string'},
                'reason': {'type': 'string'},
            },
        },
    }


def build_keyword_schema(max_candidates_per_chunk=12):
    max_candidates = max(1, _coerce_int(max_candidates_per_chunk, 12))
    candidate_schema = {
        'type': 'array',
        'maxItems': max_candidates,
        'items': {
            'type': 'object',
            'required': ['source', 'suggested_target', 'category', 'confidence', 'evidence', 'source_item_ids'],
            'additionalProperties': False,
            'properties': {
                'source': {'type': 'string'},
                'suggested_target': {'type': 'string'},
                'category': {
                    'type': 'string',
                    'enum': KEYWORD_CATEGORY_ORDER,
                },
                'confidence': {'type': 'number'},
                'evidence': {'type': 'string'},
                'source_item_ids': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
            },
        },
    }
    return {
        'type': 'object',
        'required': ['candidates', 'chunk_summary', 'summary_evidence_item_ids'],
        'additionalProperties': False,
        'properties': {
            'candidates': candidate_schema,
            'chunk_summary': {'type': 'string'},
            'summary_evidence_item_ids': {
                'type': 'array',
                'items': {'type': 'string'},
            },
        },
    }


def build_response_json_schema(units=None, mode=MODE_TRANSLATION, max_candidates_per_chunk=12):
    if mode == MODE_REVISION:
        return build_revision_schema(units or [])
    if mode == MODE_KEYWORD_EXTRACTION:
        return build_keyword_schema(max_candidates_per_chunk)
    return build_translation_schema(units or [])


def normalize_translation_results(payload):
    data = payload
    if isinstance(data, dict):
        if isinstance(data.get('items'), list):
            data = data['items']
        elif isinstance(data.get('translations'), list):
            data = data['translations']

    if not isinstance(data, list):
        raise ValueError(f'Response JSON is not a list: {type(data)}')

    normalized = []
    for item in data:
        if not isinstance(item, dict):
            continue
        item_id = item.get('id')
        translation = item.get('translation')
        if item_id is None or translation is None:
            continue
        normalized.append(
            ModelResult(
                id=str(item_id),
                mode=MODE_TRANSLATION,
                translation=str(translation),
            ).to_legacy_dict()
        )
    return normalized


def coerce_revision_should_update(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'update', 'revise', 'change', 'changed'}
    return False


def normalize_revision_results(payload):
    data = payload
    if isinstance(data, dict):
        for key in ('revisions', 'items', 'results'):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f'Response JSON is not a revision list: {type(data)}')

    normalized = []
    for item in data:
        if not isinstance(item, dict):
            continue
        item_id = item.get('id')
        if item_id is None:
            continue
        revised = item.get('revised_translation')
        if revised is None:
            revised = item.get('translation')
        if revised is None:
            revised = item.get('revised')
        normalized.append(
            ModelResult(
                id=str(item_id),
                mode=MODE_REVISION,
                should_update=coerce_revision_should_update(item.get('should_update')),
                revised_translation=str(revised or ''),
                reason=compact_text(str(item.get('reason') or '')),
            ).to_legacy_dict()
        )
    return normalized


def coerce_keyword_confidence(value):
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(confidence):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def normalize_keyword_results(payload):
    data = payload
    if isinstance(data, dict):
        if isinstance(data.get('candidates'), list):
            data = data['candidates']
        elif isinstance(data.get('items'), list):
            data = data['items']
        elif isinstance(data.get('keywords'), list):
            data = data['keywords']
    if not isinstance(data, list):
        raise ValueError(f'Response JSON is not a candidate list: {type(data)}')

    normalized = []
    for item in data:
        if not isinstance(item, dict):
            continue
        source = compact_text(str(item.get('source') or ''))
        if not source:
            continue
        category = compact_text(str(item.get('category') or 'other')).lower()
        if category not in KEYWORD_CATEGORIES:
            category = 'other'
        raw_source_item_ids = item.get('source_item_ids')
        if not isinstance(raw_source_item_ids, list):
            raw_source_item_ids = []
        normalized.append(
            ModelResult(
                mode=MODE_KEYWORD_EXTRACTION,
                source=source,
                suggested_target=compact_text(str(item.get('suggested_target') or '')),
                category=category,
                confidence=coerce_keyword_confidence(item.get('confidence')),
                evidence=compact_text(str(item.get('evidence') or '')),
                source_item_ids=[str(value) for value in raw_source_item_ids if str(value).strip()],
            ).to_legacy_dict()
        )
    return normalized


def normalize_model_results(payload, mode=MODE_TRANSLATION):
    if mode == MODE_REVISION:
        return normalize_revision_results(payload)
    if mode == MODE_KEYWORD_EXTRACTION:
        return normalize_keyword_results(payload)
    return normalize_translation_results(payload)


def translation_writeback_action(unit, result, chunk_key=''):
    result = result or {}
    return WritebackAction(
        mode=MODE_TRANSLATION,
        file_rel_path=unit.file_rel_path,
        line=unit.line,
        start=unit.start,
        end=unit.end,
        replacement=str(result.get('translation') or ''),
        prefix=unit.prefix,
        quote=unit.quote,
        expected_text=unit.text,
        item_id=unit.id,
        chunk_key=chunk_key,
    )


def revision_writeback_action(unit, result, chunk_key=''):
    result = result or {}
    return WritebackAction(
        mode=MODE_REVISION,
        file_rel_path=unit.file_rel_path,
        line=unit.line,
        start=unit.start,
        end=unit.end,
        replacement=str(result.get('revised_translation') or ''),
        prefix=unit.prefix,
        quote=unit.quote,
        expected_text=unit.current_translation,
        item_id=unit.id,
        chunk_key=chunk_key,
    )


def keyword_writeback_action(unit, result, chunk_key=''):
    """Keyword extraction produces glossary candidates and never edits scripts."""
    return None


def build_writeback_action(unit, result, mode=None, chunk_key=''):
    actual_mode = mode or unit.mode
    if actual_mode == MODE_REVISION:
        return revision_writeback_action(unit, result, chunk_key=chunk_key)
    if actual_mode == MODE_KEYWORD_EXTRACTION:
        return keyword_writeback_action(unit, result, chunk_key=chunk_key)
    return translation_writeback_action(unit, result, chunk_key=chunk_key)


def writeback_tuple(action, include_expected=True):
    base = (
        action.start,
        action.end,
        action.replacement,
        action.prefix,
        action.quote,
    )
    if not include_expected:
        return base
    return base + (action.expected_text, action.item_id, action.chunk_key)
