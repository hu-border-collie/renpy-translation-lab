# -*- coding: utf-8 -*-
"""Shared translation pipeline primitives for sync and batch workflows.

This module deliberately keeps public CLI and manifest shapes out at the edge:
callers adapt their legacy task/chunk/result dictionaries into these internal
objects, then serialize back to the same dictionaries when writing manifests or
applying replacements.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import prompt_context


CORE_SCHEMA_VERSION = 2

MODE_TRANSLATION = 'translation'
MODE_KEYWORD_EXTRACTION = 'keyword_extraction'
MODE_REVISION = 'revision'

MODEL_RESPONSE_ENVELOPE_KEYS = {
    MODE_TRANSLATION: 'translations',
    MODE_REVISION: 'revisions',
    MODE_KEYWORD_EXTRACTION: 'candidates',
}

CONTRACT_EMPTY_RESPONSE_TEXT = 'empty_response_text'
CONTRACT_INVALID_JSON = 'invalid_json'
CONTRACT_ENVELOPE_MISSING = 'response_envelope_missing'
CONTRACT_ITEMS_NOT_ARRAY = 'response_items_not_array'
CONTRACT_ITEM_NOT_OBJECT = 'result_item_not_object'
CONTRACT_MISSING_ID = 'result_missing_id'
CONTRACT_UNKNOWN_ID = 'result_unknown_id'
CONTRACT_DUPLICATE_ID = 'result_duplicate_id'
CONTRACT_MISSING_FIELD = 'result_missing_field'
CONTRACT_INVALID_FIELD_TYPE = 'result_invalid_field_type'
CONTRACT_UNEXPECTED_FIELD = 'result_unexpected_field'
CONTRACT_EMPTY_TRANSLATION = 'result_empty_translation'
CONTRACT_MISSING_EXPECTED_ID = 'response_missing_expected_id'
CONTRACT_UNKNOWN_SOURCE_ID = 'result_unknown_source_id'

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
    speaker_name: str = ''
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
    source_hits: list = field(default_factory=list)
    project_brief_text: str = ''
    project_brief_diagnostics: str = ''
    project_local_labels: list = field(default_factory=list)
    project_local_routes: list = field(default_factory=list)
    project_local_diagnostics: str = ''


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


class ModelResponseContractError(ValueError):
    """A stable, provider-neutral parse or response-contract failure."""

    def __init__(self, reason_code, message):
        super().__init__(message)
        self.reason_code = str(reason_code or 'response_contract_error')


@dataclass(frozen=True)
class ModelContractIssue:
    """One rejected response fragment with a stable machine-readable reason."""

    reason_code: str
    item_id: str = ''
    result_index: int = -1
    field: str = ''
    message: str = ''

    def to_dict(self):
        payload = {'reason_code': self.reason_code}
        if self.item_id:
            payload['id'] = self.item_id
        if self.result_index >= 0:
            payload['result_index'] = self.result_index
        if self.field:
            payload['field'] = self.field
        if self.message:
            payload['message'] = self.message
        return payload


@dataclass
class ModelContractReport:
    """Normalized valid results plus completeness and targeted-retry diagnostics."""

    mode: str
    envelope_key: str
    items: list = field(default_factory=list)
    expected_ids: list = field(default_factory=list)
    valid_ids: list = field(default_factory=list)
    retry_ids: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)
    legacy_shape: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def complete(self):
        return not self.issues and not self.retry_ids

    @property
    def completeness(self):
        if self.mode == MODE_KEYWORD_EXTRACTION:
            return 1.0 if self.complete else 0.0
        if not self.expected_ids:
            return 1.0 if self.complete else 0.0
        return len(self.valid_ids) / len(self.expected_ids)

    def reason_counts(self):
        counts = {}
        for issue in self.issues:
            counts[issue.reason_code] = counts.get(issue.reason_code, 0) + 1
        return counts

    def diagnostic_counts(self):
        counts = {}
        for diagnostic in self.diagnostics:
            counts[diagnostic.reason_code] = (
                counts.get(diagnostic.reason_code, 0) + 1
            )
        return counts

    def to_envelope(self):
        if self.mode == MODE_KEYWORD_EXTRACTION:
            return {
                self.envelope_key: list(self.items),
                'chunk_summary': str(self.metadata.get('chunk_summary') or ''),
                'summary_evidence_item_ids': list(
                    self.metadata.get('summary_evidence_item_ids') or []
                ),
            }
        return {self.envelope_key: list(self.items)}

    def to_diagnostics(self):
        if self.mode == MODE_KEYWORD_EXTRACTION:
            expected_count = 1
            valid_count = int(self.complete)
        else:
            expected_count = len(self.expected_ids)
            valid_count = len(self.valid_ids)
        return {
            'mode': self.mode,
            'envelope_key': self.envelope_key,
            'legacy_shape': self.legacy_shape,
            'complete': self.complete,
            'expected_count': expected_count,
            'valid_count': valid_count,
            'missing_count': len(self.retry_ids),
            'completeness': self.completeness,
            'valid_ids': list(self.valid_ids),
            'source_item_count': len(self.expected_ids),
            'retry_ids': list(self.retry_ids),
            'reason_counts': self.reason_counts(),
            'diagnostic_counts': self.diagnostic_counts(),
            'issues': [issue.to_dict() for issue in self.issues],
            'diagnostics': [item.to_dict() for item in self.diagnostics],
        }


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


def build_identity_v2(file_rel_path, block_name, block_index, source_text, block_occurrence=1):
    clean_path = str(file_rel_path or '').replace('\\', '/').strip()
    clean_block = str(block_name or '_global').strip()
    try:
        occurrence = max(1, int(block_occurrence or 1))
    except (TypeError, ValueError):
        occurrence = 1
    if occurrence > 1:
        clean_block = f"{clean_block}#{occurrence}"
    clean_text = str(source_text or '')
    source_hash = hashlib.sha1(clean_text.encode('utf-8')).hexdigest()[:8]
    return f"{clean_path}:{clean_block}:{block_index}:{source_hash}"


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
    'speaker_name',
    'progress_entry',
}


def unit_from_translation_item(item, file_rel_path='', file_path='', mode=MODE_TRANSLATION):
    item = dict(item or {})
    line = _coerce_line(item)
    text = str(item.get('text') if item.get('text') is not None else item.get('source') or '')
    source = str(item.get('source') or text)
    speaker_id = str(item.get('speaker_id') or item.get('speaker') or '')
    speaker_name = str(
        item.get('speaker_name')
        or item.get('speaker_display_name')
        or item.get('character_name')
        or ''
    )
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
        speaker_name=speaker_name,
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
    if unit.speaker_name:
        item['speaker_name'] = unit.speaker_name
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
    if unit.speaker_name:
        item['speaker_name'] = unit.speaker_name
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
    if unit.speaker_name:
        item['speaker_name'] = unit.speaker_name
    return item


def legacy_item_from_unit(unit, mode=None):
    actual_mode = mode or unit.mode
    if actual_mode == MODE_REVISION:
        return unit_to_revision_item(unit)
    if actual_mode == MODE_KEYWORD_EXTRACTION:
        return unit_to_keyword_item(unit)
    return unit_to_translation_item(unit)


def _speaker_label(speaker_id='', speaker_name=''):
    speaker_id = str(speaker_id or '').strip()
    speaker_name = str(speaker_name or '').strip()
    if speaker_id and speaker_name and speaker_id != speaker_name:
        return f'{speaker_name} ({speaker_id})'
    return speaker_name or speaker_id


def _format_context_line(line):
    speaker_id = ''
    speaker_name = ''
    if isinstance(line, TranslationUnit):
        text = line.source_text
        speaker_id = line.speaker_id
        speaker_name = line.speaker_name
    elif isinstance(line, dict):
        text = line.get('text') or line.get('source') or ''
        speaker_id = line.get('speaker_id') or line.get('speaker') or ''
        speaker_name = (
            line.get('speaker_name')
            or line.get('speaker_display_name')
            or line.get('character_name')
            or ''
        )
    else:
        text = str(line)
    label = _speaker_label(speaker_id, speaker_name)
    if label and text:
        return f'{label}: {text}'
    return text


def format_context_block(lines, empty_label='(none)'):
    if not lines:
        return empty_label
    rendered = []
    for line in lines:
        rendered.append(_format_context_line(line))
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


def build_context_bundle(
    glossary_hits=None,
    history_hits=None,
    story_hits=None,
    rag_stats=None,
    source_hits=None,
    project_brief_text='',
    project_brief_diagnostics='',
    project_local_labels=None,
    project_local_routes=None,
    project_local_diagnostics='',
):
    return ContextBundle(
        glossary_hits=list(glossary_hits or []),
        history_hits=list(history_hits or []),
        story_hits=story_hits,
        rag_stats=dict(rag_stats or {}),
        source_hits=list(source_hits or []),
        project_brief_text=str(project_brief_text or ''),
        project_brief_diagnostics=str(project_brief_diagnostics or ''),
        project_local_labels=list(project_local_labels or []),
        project_local_routes=list(project_local_routes or []),
        project_local_diagnostics=str(project_local_diagnostics or ''),
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
        source_hits=context_bundle.source_hits,
        project_brief_text=context_bundle.project_brief_text,
        project_brief_diagnostics=context_bundle.project_brief_diagnostics,
        project_local_labels=context_bundle.project_local_labels,
        project_local_routes=context_bundle.project_local_routes,
        project_local_diagnostics=context_bundle.project_local_diagnostics,
        history_char_limit=history_char_limit,
        story_char_limit=story_char_limit,
        include_source_text=include_source_text,
        story_block_suffix=story_block_suffix,
    )


def translation_target_payload_item(unit):
    item = {'id': unit.id, 'text': unit.text}
    if unit.speaker_id:
        item['speaker_id'] = unit.speaker_id
    if unit.speaker_name:
        item['speaker_name'] = unit.speaker_name
    return item


def build_translation_system_instruction(preserve_terms, macro_setting=''):
    glossary = ', '.join(str(term) for term in preserve_terms or [])
    return (
        'Setting:\n'
        f'{macro_setting or ""}\n\n'
        'Task:\n'
        'Translate only TARGET lines into Simplified Chinese. CONTEXT lines are reference only.\n'
        f'Keep these terms unchanged: {glossary}\n'
        'If a TARGET contains one of those terms, copy that exact source substring verbatim into the translation, '
        'including honorifics, apostrophes, numbers, and spacing; do not localize, reorder, or partially translate it.\n'
        "Keep names, Ren'Py tags, placeholders, variables, and format strings unchanged.\n"
        "Every bracketed Ren'Py interpolation such as [Gil_name!t], [Main], or [Parker_last!t] "
        'must be copied exactly; never replace it with a literal visible name.\n'
        'Return one result for every TARGET id even when the text is hard to translate; never omit an item.\n'
        'When TARGET items include speaker_id or speaker_name, use them only to identify the speaker and voice.\n'
        'Return one JSON object with a translations array. Preserve every id exactly. Item count must match. '
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
        [translation_target_payload_item(unit) for unit in units],
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
    context_window=None,
    macro_setting='',
    normalize_map=None,
    non_translatable_terms=None,
):
    """Build the reference-only sync translation prompt for one batch.

    ``context_window`` (before/after) is injected as ``CONTEXT BEFORE/AFTER``
    and is strictly reference material: the model must only translate the
    ``Input JSON`` items. ``macro_setting`` is prepended as a ``Setting``
    block when non-empty. ``normalize_map`` and ``non_translatable_terms`` are
    the current batch's lexical glossary hits; when provided they are rendered
    as ``Existing glossary entries`` alongside ``preserve_terms``.
    """
    units = units_from_items(units, MODE_TRANSLATION)
    glossary = ', '.join(str(term) for term in preserve_terms or [])
    payload = json.dumps(
        [translation_target_payload_item(unit) for unit in units],
        ensure_ascii=False,
    )
    setting_block = ''
    if isinstance(macro_setting, str) and macro_setting.strip():
        setting_block = f"Setting:\n{macro_setting.strip()}\n\n"
    glossary_block = ''
    if normalize_map or non_translatable_terms:
        glossary_block = (
            'Existing glossary entries:\n'
            f'{build_keyword_glossary_block(preserve_terms, normalize_map, non_translatable_terms)}\n\n'
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
    context_note = ''
    context_before_block = ''
    context_after_block = ''
    if context_window is not None:
        context_note = (
            '\nCONTEXT BEFORE/AFTER lines are reference only. '
            'Use them to keep speaker voice, pronoun references, and continuity consistent, '
            'but never translate or return them.\n'
        )
        context_before_block = (
            '\nCONTEXT BEFORE:\n'
            f'{format_context_block(list(context_window.before or []), "(none)")}\n'
        )
        context_after_block = (
            '\nCONTEXT AFTER:\n'
            f'{format_context_block(list(context_window.after or []), "(none)")}\n'
        )
    return (
        f'{setting_block}'
        f'{glossary_block}'
        "You are translating a Ren'Py visual novel into Simplified Chinese (zh-CN).\n"
        'Rules:\n'
        f'1. Preserve these terms exactly (do not translate): {glossary}\n'
        '1.0 If an input contains a listed term, copy that exact source substring verbatim, '
        'including honorifics, apostrophes, numbers, and spacing.\n'
        '1.1 Keep all person names in English; do not translate names.\n'
        "2. Preserve Ren'Py tags like {i}, {/i}, {color=...}, [name], %s.\n"
        "2.0a Copy bracketed Ren'Py interpolations exactly, e.g. [Gil_name!t], [Main], [Parker_last!t]; "
        'never turn them into literal names.\n'
        '2.1 If an input item includes speaker_id or speaker_name, use it only to identify who is speaking and their voice.\n'
        '3. Output plain Chinese text. No markdown, no Pinyin, no explanations.\n'
        '4. Return ONLY one JSON object shaped as '
        '{"translations":[{"id":"...","translation":"..."}]}.\n'
        f'{reference_blocks}'
        f'{context_note}'
        f'{context_before_block}'
        f'Input JSON:\n{payload}'
        f'{context_after_block}'
    )


# --- Canonical translation contract (issue #346, decisions D3-D6) -------------
#
# The single semantic contract both execution strategies must send to the
# model. The legacy build_translation_* / build_sync_translation_prompt
# builders above remain as compatibility shims; new executor wiring must not
# call them.

CANONICAL_CHUNK_MAX_ITEMS = 60       # D4-A
CANONICAL_CHUNK_MAX_CHARS = 18000    # D4-A
CANONICAL_LOCAL_CONTEXT_BEFORE = 30  # D1 window limits (sync/batch already agree)
CANONICAL_LOCAL_CONTEXT_AFTER = 10   # D1
CANONICAL_HISTORY_CHAR_LIMIT = 220   # D5
CANONICAL_STORY_CHAR_LIMIT = 1200    # D5
CANONICAL_ANALYSIS_CHAR_LIMIT = 4000  # D5 (matches batch max_brief_chars)
CANONICAL_INCLUDE_SOURCE_TEXT = True  # D5
CANONICAL_STORY_BLOCK_SUFFIX = '\n\n'  # D5 (batch form)


def file_hash_key(text):
    """Short stable content key for file paths (legacy batch chunk-key form)."""
    return hashlib.sha1(str(text).encode('utf-8')).hexdigest()[:10]


def translation_text_char_count(task):
    text = task.get('text', '') if isinstance(task, dict) else ''
    return len(text) if isinstance(text, str) else len(str(text))


def iter_translation_chunk_ranges(tasks, max_items=CANONICAL_CHUNK_MAX_ITEMS, max_chars=CANONICAL_CHUNK_MAX_CHARS):
    """Yield ``(start, end)`` task ranges shared by sync and batch chunking.

    The first item of a range is always accepted so a single oversized task
    still forms its own chunk. Extracted from the former batch-only iterator
    so both execution strategies derive identical translation-unit grouping
    (issue #346, D4).
    """
    total = len(tasks)
    start = 0
    while start < total:
        end = start
        current_chars = 0
        while end < total and (end - start) < max_items:
            item_chars = translation_text_char_count(tasks[end])
            if end > start and current_chars + item_chars > max_chars:
                break
            current_chars += item_chars
            end += 1
        if end == start:
            end = start + 1
        yield start, end
        start = end


def build_canonical_translation_system_instruction(preserve_terms, macro_setting=''):
    glossary = ', '.join(str(term) for term in preserve_terms or [])
    return (
        'Setting:\n'
        f'{macro_setting or ""}\n\n'
        'Task:\n'
        'Translate only TARGET lines into Simplified Chinese. CONTEXT lines are reference only.\n'
        f'Keep these terms unchanged: {glossary}\n'
        'If a TARGET contains one of those terms, copy that exact source substring verbatim into the translation, '
        'including honorifics, apostrophes, numbers, and spacing; do not localize, reorder, or partially translate it.\n'
        "Keep names, Ren'Py tags, placeholders, variables, and format strings unchanged.\n"
        'Keep all person names in English; do not translate names.\n'
        "Every bracketed Ren'Py interpolation such as [Gil_name!t], [Main], or [Parker_last!t] "
        'must be copied exactly; never replace it with a literal visible name.\n'
        'Return one result for every TARGET id even when the text is hard to translate; never omit an item.\n'
        'When TARGET items include speaker_id or speaker_name, use them only to identify the speaker and voice.\n'
        'Output plain Chinese text. No markdown, no Pinyin, no explanations.\n'
        'Return one JSON object with a translations array. Preserve every id exactly. Item count must match. '
        'translation must contain only the translated Chinese text.'
    )


def build_canonical_translation_user_prompt(
    context_window,
    units,
    reference_blocks_text='',
    lexical_glossary_text='',
):
    """Build the canonical per-request user prompt (issue #346, D3/D5).

    ``lexical_glossary_text`` carries the rendered lexical glossary hits of the
    current chunk (issue #338 wording, D2: injected whenever hits exist, never
    gated on RAG being enabled). ``reference_blocks_text`` is the pre-rendered
    retrieval/analysis reference section; callers render it with
    :func:`build_reference_blocks` using the D5 policy values so the assembled
    context layer accounting stays byte-identical to the embedded text. The
    TARGET payload always uses compact JSON separators (D5).
    """
    context_window = context_window or ContextWindow()
    units = units_from_items(units, MODE_TRANSLATION)
    target_payload = json.dumps(
        [translation_target_payload_item(unit) for unit in units],
        ensure_ascii=False,
        separators=(',', ':'),
    )
    glossary_block = ''
    if isinstance(lexical_glossary_text, str) and lexical_glossary_text.strip():
        glossary_block = f'Existing glossary entries:\n{lexical_glossary_text.strip()}\n\n'
    reference_section = ''
    if isinstance(reference_blocks_text, str) and reference_blocks_text.strip():
        reference_section = reference_blocks_text
        if not reference_section.endswith('\n\n'):
            reference_section += '\n\n'
    return ''.join(
        [
            glossary_block,
            reference_section,
            f'CONTEXT BEFORE:\n{format_context_block(context_window.before, "(none)")}\n\n',
            f'TARGET:\n{target_payload}\n\n',
            f'CONTEXT AFTER:\n{format_context_block(context_window.after, "(none)")}\n\n',
            'Return the result now.',
        ]
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
        'If a TARGET contains one of those terms, copy that exact source substring verbatim into the revised translation, '
        'including honorifics, apostrophes, numbers, and spacing. '
        'If the current translation is already acceptable, set should_update=false and repeat it as revised_translation. '
        'If it needs a change, set should_update=true and provide only the revised Chinese translation. '
        'Return one JSON object with a revisions array. Preserve every id exactly. '
        'Item count must match.'
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
    payload_items = []
    for unit in units:
        item = {
            'id': unit.id,
            'file': unit.file_rel_path,
            'line': unit.display_line_number,
            'speaker_id': unit.speaker_id,
            'source': unit.source_text,
            'current_translation': unit.current_translation,
        }
        if unit.speaker_name:
            item['speaker_name'] = unit.speaker_name
        payload_items.append(item)
    target_payload = json.dumps(
        payload_items,
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
            'Return {"revisions":[...]} with id, should_update, revised_translation, and reason.',
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
        'Set source_item_ids to one or more input id values that support the candidate. '
        'Use concise evidence that cites the relevant input id or phrase.\n'
        'Also write a compact chunk_summary in Chinese that summarizes only the visible story events in this chunk. '
        'Use 1-3 sentences, avoid invented continuity, and leave chunk_summary empty if the lines do not contain usable story content. '
        'When chunk_summary is non-empty, set summary_evidence_item_ids to one or more input ids that support it; '
        'otherwise use an empty array. Return JSON only.'
    )


def build_keyword_user_prompt(units):
    units = units_from_items(units, MODE_KEYWORD_EXTRACTION)
    payload_items = []
    for unit in units:
        item = {
            'id': unit.id,
            'file': unit.file_rel_path,
            'line': unit.display_line_number,
            'speaker_id': unit.speaker_id,
            'text': unit.text,
        }
        if unit.speaker_name:
            item['speaker_name'] = unit.speaker_name
        payload_items.append(item)
    target_payload = json.dumps(
        payload_items,
        ensure_ascii=False,
        separators=(',', ':'),
    )
    return (
        'TARGET LINES:\n'
        f'{target_payload}\n\n'
        'Return a JSON object with candidates, chunk_summary, and summary_evidence_item_ids. '
        'Each candidate must include source, suggested_target, category, confidence, evidence, and source_item_ids; '
        'source_item_ids must contain at least one TARGET LINES id. A non-empty chunk_summary must cite at least one '
        'TARGET LINES id in summary_evidence_item_ids.'
    )


def build_translation_schema(units):
    units = units_from_items(units, MODE_TRANSLATION)
    return {
        'type': 'object',
        'required': ['translations'],
        'additionalProperties': False,
        'properties': {
            'translations': {
                'type': 'array',
                'minItems': len(units),
                'maxItems': len(units),
                'items': {
                    'type': 'object',
                    'required': ['id', 'translation'],
                    'additionalProperties': False,
                    'properties': {
                        'id': {'type': 'string'},
                        'translation': {'type': 'string'},
                    },
                },
            },
        },
    }


def build_revision_schema(units):
    units = units_from_items(units, MODE_REVISION)
    return {
        'type': 'object',
        'required': ['revisions'],
        'additionalProperties': False,
        'properties': {
            'revisions': {
                'type': 'array',
                'minItems': len(units),
                'maxItems': len(units),
                'items': {
                    'type': 'object',
                    'required': ['id', 'should_update', 'revised_translation', 'reason'],
                    'additionalProperties': False,
                    'properties': {
                        'id': {'type': 'string'},
                        'should_update': {'type': 'boolean'},
                        'revised_translation': {'type': 'string'},
                        'reason': {'type': 'string'},
                    },
                },
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
                'category': {'type': 'string'},
                'confidence': {'type': 'number'},
                'evidence': {'type': 'string'},
                'source_item_ids': {
                    'type': 'array',
                    'minItems': 1,
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


def _contract_expected_ids(units, mode):
    normalized = units_from_items(units or [], mode)
    return [str(unit.id) for unit in normalized if str(unit.id)]


def _contract_items(payload, mode, allow_legacy):
    envelope_key = MODEL_RESPONSE_ENVELOPE_KEYS.get(mode, 'items')
    legacy_shape = False
    data = None
    if isinstance(payload, dict):
        if envelope_key in payload:
            data = payload.get(envelope_key)
        elif allow_legacy:
            aliases = {
                MODE_TRANSLATION: ('items',),
                MODE_REVISION: ('items', 'results'),
                MODE_KEYWORD_EXTRACTION: ('items', 'keywords'),
            }.get(mode, ('items',))
            for alias in aliases:
                if alias in payload:
                    data = payload.get(alias)
                    legacy_shape = True
                    break
    elif isinstance(payload, list) and allow_legacy:
        data = payload
        legacy_shape = True
    return envelope_key, data, legacy_shape


def _issue(reason_code, *, item_id='', result_index=-1, field_name='', message=''):
    return ModelContractIssue(
        reason_code=reason_code,
        item_id=str(item_id or ''),
        result_index=result_index,
        field=field_name,
        message=message,
    )


def _validate_id_results(payload, mode, expected_ids, allow_legacy):
    envelope_key, data, legacy_shape = _contract_items(payload, mode, allow_legacy)
    report = ModelContractReport(
        mode=mode,
        envelope_key=envelope_key,
        expected_ids=list(expected_ids),
        legacy_shape=legacy_shape,
    )
    if data is None:
        report.issues.append(_issue(
            CONTRACT_ENVELOPE_MISSING,
            field_name=envelope_key,
            message=f'Response must contain a {envelope_key} array.',
        ))
        report.retry_ids = list(expected_ids)
        return report
    if not isinstance(data, list):
        report.issues.append(_issue(
            CONTRACT_ITEMS_NOT_ARRAY,
            field_name=envelope_key,
            message=f'{envelope_key} must be an array.',
        ))
        report.retry_ids = list(expected_ids)
        return report

    if mode == MODE_REVISION:
        required = ('id', 'should_update', 'revised_translation', 'reason')
        allowed = set(required)
    else:
        required = ('id', 'translation')
        allowed = set(required)
    expected_set = set(expected_ids)
    valid_by_id = {}
    invalid_ids = set()
    seen_ids = set()

    for index, raw_item in enumerate(data):
        if not isinstance(raw_item, dict):
            report.issues.append(_issue(
                CONTRACT_ITEM_NOT_OBJECT,
                result_index=index,
                message='Result item must be an object.',
            ))
            continue
        raw_id = raw_item.get('id')
        item_id = raw_id if isinstance(raw_id, str) else ''
        if not item_id:
            reason = CONTRACT_MISSING_ID if raw_id is None else CONTRACT_INVALID_FIELD_TYPE
            report.issues.append(_issue(
                reason,
                result_index=index,
                field_name='id',
                message='Result id must be a non-empty string.',
            ))
            continue
        if item_id not in expected_set:
            report.issues.append(_issue(
                CONTRACT_UNKNOWN_ID,
                item_id=item_id,
                result_index=index,
                field_name='id',
                message='Result id was not requested.',
            ))
            continue
        if item_id in seen_ids:
            report.issues.append(_issue(
                CONTRACT_DUPLICATE_ID,
                item_id=item_id,
                result_index=index,
                field_name='id',
                message='Each requested id must appear exactly once.',
            ))
            invalid_ids.add(item_id)
            valid_by_id.pop(item_id, None)
            continue
        seen_ids.add(item_id)

        item_valid = True
        for field_name in required:
            if field_name not in raw_item:
                report.issues.append(_issue(
                    CONTRACT_MISSING_FIELD,
                    item_id=item_id,
                    result_index=index,
                    field_name=field_name,
                    message=f'Missing required field: {field_name}.',
                ))
                item_valid = False
        unexpected = sorted(set(raw_item) - allowed)
        for field_name in unexpected:
            report.diagnostics.append(_issue(
                CONTRACT_UNEXPECTED_FIELD,
                item_id=item_id,
                result_index=index,
                field_name=field_name,
                message=f'Unexpected result field: {field_name}.',
            ))
        if not item_valid:
            invalid_ids.add(item_id)
            continue

        if mode == MODE_REVISION:
            field_types = {
                'should_update': bool,
                'revised_translation': str,
                'reason': str,
            }
            output = {
                'id': item_id,
                'should_update': raw_item['should_update'],
                'revised_translation': raw_item['revised_translation'],
                'reason': raw_item['reason'],
            }
            translated = raw_item['revised_translation']
        else:
            field_types = {'translation': str}
            output = {'id': item_id, 'translation': raw_item['translation']}
            translated = raw_item['translation']
        for field_name, expected_type in field_types.items():
            if not isinstance(raw_item[field_name], expected_type):
                report.issues.append(_issue(
                    CONTRACT_INVALID_FIELD_TYPE,
                    item_id=item_id,
                    result_index=index,
                    field_name=field_name,
                    message=f'{field_name} must be {expected_type.__name__}.',
                ))
                item_valid = False
        if isinstance(translated, str) and not translated.strip():
            report.issues.append(_issue(
                CONTRACT_EMPTY_TRANSLATION,
                item_id=item_id,
                result_index=index,
                field_name='revised_translation' if mode == MODE_REVISION else 'translation',
                message='Translation text must not be empty.',
            ))
            item_valid = False
        if item_valid:
            valid_by_id[item_id] = output
        else:
            invalid_ids.add(item_id)

    retry_ids = []
    for item_id in expected_ids:
        if item_id not in valid_by_id or item_id in invalid_ids:
            retry_ids.append(item_id)
            report.issues.append(_issue(
                CONTRACT_MISSING_EXPECTED_ID,
                item_id=item_id,
                field_name='id',
                message='Requested id has no valid result.',
            ))
    report.retry_ids = retry_ids
    report.valid_ids = [item_id for item_id in expected_ids if item_id in valid_by_id]
    if expected_ids:
        report.items = [valid_by_id[item_id] for item_id in report.valid_ids]
    else:
        report.valid_ids = list(valid_by_id)
        report.items = list(valid_by_id.values())
    return report


def _validate_string_list(
    value,
    *,
    expected_ids,
    report,
    field_name,
    result_index=-1,
):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        report.issues.append(_issue(
            CONTRACT_INVALID_FIELD_TYPE,
            result_index=result_index,
            field_name=field_name,
            message=f'{field_name} must be an array of strings.',
        ))
        return []
    cleaned = []
    for item in value:
        if not item.strip():
            report.issues.append(_issue(
                CONTRACT_INVALID_FIELD_TYPE,
                result_index=result_index,
                field_name=field_name,
                message=f'{field_name} must contain non-empty strings.',
            ))
            continue
        cleaned.append(item)
    expected_set = set(expected_ids)
    for item_id in cleaned:
        if item_id not in expected_set:
            report.issues.append(_issue(
                CONTRACT_UNKNOWN_SOURCE_ID,
                item_id=item_id,
                result_index=result_index,
                field_name=field_name,
                message='Evidence id was not present in the request.',
            ))
    return [item_id for item_id in cleaned if item_id in expected_set]


def _validate_keyword_response(payload, expected_ids, allow_legacy):
    envelope_key, data, legacy_shape = _contract_items(
        payload,
        MODE_KEYWORD_EXTRACTION,
        allow_legacy,
    )
    report = ModelContractReport(
        mode=MODE_KEYWORD_EXTRACTION,
        envelope_key=envelope_key,
        expected_ids=list(expected_ids),
        legacy_shape=legacy_shape,
    )
    if data is None:
        report.issues.append(_issue(
            CONTRACT_ENVELOPE_MISSING,
            field_name=envelope_key,
            message='Response must contain a candidates array.',
        ))
        report.retry_ids = list(expected_ids)
        return report
    if not isinstance(data, list):
        report.issues.append(_issue(
            CONTRACT_ITEMS_NOT_ARRAY,
            field_name=envelope_key,
            message='candidates must be an array.',
        ))
        report.retry_ids = list(expected_ids)
        return report

    if isinstance(payload, dict):
        summary = payload.get('chunk_summary')
        evidence_ids = payload.get('summary_evidence_item_ids')
    else:
        summary = ''
        evidence_ids = []
    if not legacy_shape:
        if not isinstance(summary, str):
            report.issues.append(_issue(
                CONTRACT_INVALID_FIELD_TYPE,
                field_name='chunk_summary',
                message='chunk_summary must be a string.',
            ))
            summary = ''
        summary_ids = _validate_string_list(
            evidence_ids,
            expected_ids=expected_ids,
            report=report,
            field_name='summary_evidence_item_ids',
        )
    else:
        summary = summary if isinstance(summary, str) else ''
        summary_ids = evidence_ids if isinstance(evidence_ids, list) else []
    normalized_summary = compact_text(summary)
    if not legacy_shape and normalized_summary and not summary_ids:
        report.issues.append(_issue(
            CONTRACT_MISSING_FIELD,
            field_name='summary_evidence_item_ids',
            message='A non-empty chunk_summary must cite at least one requested id.',
        ))
    report.metadata = {
        'chunk_summary': normalized_summary,
        'summary_evidence_item_ids': summary_ids,
    }

    required = {
        'source': str,
        'suggested_target': str,
        'category': str,
        'confidence': (int, float),
        'evidence': str,
        'source_item_ids': list,
    }
    valid_candidates = []
    for index, raw_item in enumerate(data):
        if not isinstance(raw_item, dict):
            report.issues.append(_issue(
                CONTRACT_ITEM_NOT_OBJECT,
                result_index=index,
                message='Keyword candidate must be an object.',
            ))
            continue
        candidate_input = dict(raw_item)
        if legacy_shape and 'source_item_ids' not in candidate_input:
            candidate_input['source_item_ids'] = []
        item_valid = True
        for field_name, expected_type in required.items():
            if field_name not in candidate_input:
                report.issues.append(_issue(
                    CONTRACT_MISSING_FIELD,
                    result_index=index,
                    field_name=field_name,
                    message=f'Missing required field: {field_name}.',
                ))
                item_valid = False
                continue
            if not isinstance(candidate_input[field_name], expected_type) or (
                field_name == 'confidence'
                and isinstance(candidate_input[field_name], bool)
            ):
                report.issues.append(_issue(
                    CONTRACT_INVALID_FIELD_TYPE,
                    result_index=index,
                    field_name=field_name,
                    message=f'Invalid field type: {field_name}.',
                ))
                item_valid = False
        if not item_valid:
            continue
        source = compact_text(candidate_input['source'])
        if not source:
            report.issues.append(_issue(
                CONTRACT_MISSING_FIELD,
                result_index=index,
                field_name='source',
                message='Candidate source must not be empty.',
            ))
            continue
        raw_source_ids = candidate_input['source_item_ids']
        if not raw_source_ids and not legacy_shape:
            report.issues.append(_issue(
                CONTRACT_MISSING_FIELD,
                result_index=index,
                field_name='source_item_ids',
                message='Candidate source_item_ids must cite at least one requested id.',
            ))
            continue
        if legacy_shape and not raw_source_ids:
            normalized = normalize_keyword_results(
                {'candidates': [candidate_input]}
            )
            if normalized:
                valid_candidates.append(normalized[0])
            continue
        source_ids = _validate_string_list(
            raw_source_ids,
            expected_ids=expected_ids,
            report=report,
            field_name='source_item_ids',
            result_index=index,
        )
        if len(source_ids) != len(raw_source_ids):
            continue
        candidate = candidate_input
        candidate['source_item_ids'] = source_ids
        normalized = normalize_keyword_results({'candidates': [candidate]})
        if normalized:
            valid_candidates.append(normalized[0])
    report.items = valid_candidates
    report.valid_ids = sorted({
        item_id
        for candidate in valid_candidates
        for item_id in candidate.get('source_item_ids', [])
    })
    if report.issues:
        implicated = {
            issue.item_id for issue in report.issues if issue.item_id in set(expected_ids)
        }
        report.retry_ids = [
            item_id for item_id in expected_ids if item_id in implicated
        ] or list(expected_ids)
    return report


def validate_model_response(payload, *, mode=MODE_TRANSLATION, expected_units=None, allow_legacy=True):
    """Validate one model response and return ordered valid items plus retry IDs.

    New requests use a named top-level object envelope. ``allow_legacy`` keeps
    historical bare-array artifacts readable, while every item is still checked
    for required fields, exact IDs, duplicates, types, and empty translations.
    """
    expected_ids = _contract_expected_ids(expected_units or [], mode)
    if mode == MODE_KEYWORD_EXTRACTION:
        return _validate_keyword_response(payload, expected_ids, allow_legacy)
    return _validate_id_results(payload, mode, expected_ids, allow_legacy)


def _salvage_partial_json_array(text):
    start = text.find('[')
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    index = start + 1
    items = []
    while index < len(text):
        while index < len(text) and text[index] in ' \r\n\t,':
            index += 1
        if index >= len(text) or text[index] == ']':
            return items
        try:
            item, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        items.append(item)
    return items


def parse_model_response_json(text, *, salvage_partial=True):
    """Parse provider text without echoing it and expose stable failure codes."""
    if not isinstance(text, str) or not text.strip():
        raise ModelResponseContractError(
            CONTRACT_EMPTY_RESPONSE_TEXT,
            'Model response text is empty.',
        )
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        embedded = []
        for start, char in enumerate(cleaned):
            if char not in '[{':
                continue
            try:
                payload, end = decoder.raw_decode(cleaned, start)
                embedded.append((end, -start, payload))
                if not cleaned[end:].strip():
                    break
            except json.JSONDecodeError:
                continue
        if embedded:
            _end, neg_start, payload = max(
                embedded,
                key=lambda item: (item[0], item[1]),
            )
            start = -neg_start
            previous_index = start - 1
            while previous_index >= 0 and cleaned[previous_index].isspace():
                previous_index -= 1
            salvaged = _salvage_partial_json_array(cleaned) if salvage_partial else []
            if (
                salvaged
                and previous_index >= 0
                and cleaned[previous_index] in '[,'
            ):
                return salvaged
            return payload
        if salvage_partial:
            salvaged = _salvage_partial_json_array(cleaned)
            if salvaged:
                return salvaged
        raise ModelResponseContractError(
            CONTRACT_INVALID_JSON,
            'Model response is not valid JSON.',
        ) from original_error


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
