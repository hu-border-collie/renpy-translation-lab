# -*- coding: utf-8 -*-
"""Deterministic, offline mechanical quality checks for translated strings.

This module owns the quality side of the check result contract introduced in
issue #313.  It is deliberately independent from the writeback safety gate:
rules here produce *findings* (warnings or configured blockers), while callers
combine those findings with structural writeback validation to compute the two
persisted gates:

* ``writeback_gate`` -- can the current results be written back safely?
* ``quality_gate``   -- do the results need human quality review?

Rule implementations must stay deterministic and offline.  Semantic / literary
review remains the responsibility of the final-review workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

QUALITY_FINDING_SCHEMA_VERSION = 1
QUALITY_RULE_SCHEMA_VERSION = 1

DISPOSITION_WARNING = 'warning'
DISPOSITION_BLOCKER = 'blocker'
DISPOSITION_OFF = 'off'
VALID_DISPOSITIONS = (DISPOSITION_WARNING, DISPOSITION_BLOCKER, DISPOSITION_OFF)

SEVERITY_INFO = 'info'
SEVERITY_LOW = 'low'
SEVERITY_MEDIUM = 'medium'
SEVERITY_HIGH = 'high'
VALID_SEVERITIES = (SEVERITY_INFO, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH)
SEVERITY_ORDER = {SEVERITY_INFO: 0, SEVERITY_LOW: 1, SEVERITY_MEDIUM: 2, SEVERITY_HIGH: 3}

# Fields every persisted quality finding must carry.  Final-review semantic
# findings are adapted into this same shape instead of inventing a second
# persistence contract for GUI lists, filters, and digest/staleness checks.
QUALITY_FINDING_FIELDS: tuple[str, ...] = (
    'finding_id',
    'schema_version',
    'reason_code',
    'rule_id',
    'severity',
    'disposition',
    'item_id',
    'file',
    'line',
    'source',
    'translation',
    'evidence',
    'suggestion',
    'rule_version',
)

GATE_READY = 'ready'
GATE_READY_WITH_WARNINGS = 'ready_with_warnings'
GATE_BLOCKED = 'blocked'
GATE_NEEDS_REVIEW = 'needs_review'
GATE_PASS = 'pass'
GATE_ALLOW = 'allow'
GATE_DENY = 'deny'

# Stable reason codes.  GUI filters and project configuration use these exact
# strings, so do not rename them without a schema migration.
REASON_WAIT_TAG_INSIDE_CJK = 'quality.renpy.wait_tag_inside_cjk'
REASON_UNCLOSED_DELIMITERS = 'quality.structure.unclosed_delimiters'
REASON_ENGLISH_SUFFIX_ADJACENT = 'quality.language.english_suffix_adjacent'
REASON_SUSPICIOUS_ENGLISH_RESIDUE = 'quality.language.suspicious_english_residue'
REASON_CJK_LATIN_SPACING = 'quality.typography.cjk_latin_spacing'
REASON_HALFWIDTH_PUNCTUATION = 'quality.typography.halfwidth_punctuation'
REASON_ASCII_ELLIPSIS = 'quality.typography.ascii_ellipsis'
REASON_GLOSSARY_TERM_NOT_APPLIED = 'quality.glossary.term_not_applied'
REASON_SPEAKER_LABEL_UNTRANSLATED = 'quality.speaker.label_untranslated'
REASON_INTERJECTION_UNTRANSLATED = 'quality.completeness.interjection_untranslated'
REASON_KNOWN_GARBLED_PHRASE = 'quality.garbled.known_bad_phrase'
# Collection diagnostic emitted by check when a validated writeback action
# cannot be mapped back to a manifest item for mechanical inspection.
REASON_UNMATCHED_QUALITY_SUBJECT = 'quality.collection.unmatched_subject'

# Final-review findings keep their LLM semantic fields, but are adapted into the
# common quality-finding data model with stable ``quality.llm.*`` reason codes.
# These codes deliberately sit in their own namespace so a literary conclusion
# can never be mistaken for a deterministic mechanical rule.
FINAL_REVIEW_REASON_PREFIX = 'quality.llm'
FINAL_REVIEW_REASON_OMISSION = 'quality.llm.omission'
FINAL_REVIEW_REASON_MISTRANSLATION = 'quality.llm.mistranslation'
FINAL_REVIEW_REASON_ADDITION = 'quality.llm.addition'
FINAL_REVIEW_REASON_FORMAT = 'quality.llm.format'
FINAL_REVIEW_REASON_TERMINOLOGY = 'quality.llm.terminology'
FINAL_REVIEW_REASON_ADDRESS = 'quality.llm.address'
FINAL_REVIEW_REASON_STYLE_DRIFT = 'quality.llm.style_drift'
FINAL_REVIEW_REASON_NEEDS_CONFIRMATION = 'quality.llm.needs_confirmation'

FINAL_REVIEW_FINDING_TYPE_TO_REASON: dict[str, str] = {
    'omission': FINAL_REVIEW_REASON_OMISSION,
    'mistranslation': FINAL_REVIEW_REASON_MISTRANSLATION,
    'addition': FINAL_REVIEW_REASON_ADDITION,
    'format': FINAL_REVIEW_REASON_FORMAT,
    'terminology': FINAL_REVIEW_REASON_TERMINOLOGY,
    'address': FINAL_REVIEW_REASON_ADDRESS,
    'style_drift': FINAL_REVIEW_REASON_STYLE_DRIFT,
    'needs_confirmation': FINAL_REVIEW_REASON_NEEDS_CONFIRMATION,
}
FINAL_REVIEW_REASON_TO_FINDING_TYPE = {
    reason: finding_type
    for finding_type, reason in FINAL_REVIEW_FINDING_TYPE_TO_REASON.items()
}

ALL_REASON_CODES: tuple[str, ...] = (
    REASON_WAIT_TAG_INSIDE_CJK,
    REASON_UNCLOSED_DELIMITERS,
    REASON_ENGLISH_SUFFIX_ADJACENT,
    REASON_SUSPICIOUS_ENGLISH_RESIDUE,
    REASON_CJK_LATIN_SPACING,
    REASON_HALFWIDTH_PUNCTUATION,
    REASON_ASCII_ELLIPSIS,
    REASON_GLOSSARY_TERM_NOT_APPLIED,
    REASON_SPEAKER_LABEL_UNTRANSLATED,
    REASON_INTERJECTION_UNTRANSLATED,
    REASON_KNOWN_GARBLED_PHRASE,
)

RULE_KEYS: dict[str, str] = {
    'renpy_wait_inside_cjk': REASON_WAIT_TAG_INSIDE_CJK,
    'unclosed_delimiters': REASON_UNCLOSED_DELIMITERS,
    'english_suffix_adjacent': REASON_ENGLISH_SUFFIX_ADJACENT,
    'suspicious_english_residue': REASON_SUSPICIOUS_ENGLISH_RESIDUE,
    'cjk_latin_spacing': REASON_CJK_LATIN_SPACING,
    'halfwidth_punctuation': REASON_HALFWIDTH_PUNCTUATION,
    'ascii_ellipsis': REASON_ASCII_ELLIPSIS,
    'glossary_term_not_applied': REASON_GLOSSARY_TERM_NOT_APPLIED,
    'speaker_label_untranslated': REASON_SPEAKER_LABEL_UNTRANSLATED,
    'interjection_untranslated': REASON_INTERJECTION_UNTRANSLATED,
    'known_garbled_phrase': REASON_KNOWN_GARBLED_PHRASE,
}
REASON_TO_RULE_KEY = {reason: key for key, reason in RULE_KEYS.items()}

DEFAULT_ALLOWED_LATIN_TOKENS: tuple[str, ...] = (
    "Ren'Py",
    'OK',
    'NG',
    'CG',
    'BGM',
    'SE',
    'HP',
    'MP',
    'TP',
    'XP',
    'LV',
    'AI',
    'ID',
    'VIP',
    'DLC',
    'RPG',
    'CPU',
    'USB',
    'TV',
    'PC',
    'URL',
)

DEFAULT_INTERJECTIONS: tuple[str, ...] = (
    'Hm',
    'Hmm',
    'Oh',
    'Ow',
    'Whoa',
    'Wow',
    'Hey',
    'Ah',
    'Eh',
    'Ugh',
    'Ouch',
    'Huh',
    'Eek',
    'Argh',
    'Gah',
)

DEFAULT_SPEAKER_OCCUPATION_HINTS: tuple[str, ...] = (
    'Knight',
    'Guard',
    'Bouncer',
    'Merchant',
    'Captain',
    'Doctor',
    'Nurse',
    'Priest',
    'Priestess',
    'Maid',
    'Butler',
    'Sister',
    'Brother',
    'Father',
    'Mother',
    'Chief',
    'Elder',
    'Witch',
    'Wizard',
    'Hunter',
    'Smith',
    'Teacher',
    'Soldier',
    'Commander',
    'Adventurer',
    'Guild',
    'Church',
    'City',
    'Sir',
    'Boss',
    'Master',
    'Miss',
    'Mr',
    'Mrs',
    'Ms',
)

_CJK_RE = r'\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
_CJK_CLASS = f'[{_CJK_RE}]'
_RENPY_TAG_RE = re.compile(r'\{[^{}\r\n]*\}')
_RENPY_FIELD_RE = re.compile(r'\[[^\[\]\r\n]*\]')
_WAIT_TAG_RE = re.compile(
    rf'({_CJK_CLASS})\s*(\{{w(?:=[^{{}}\r\n]*)?\}})\s*({_CJK_CLASS})'
)
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*")
_ENGLISH_SUFFIX_RE = re.compile(
    r"(?:'s|s|es|ed|ing|er|est|ly|ness|ment|tion|able|ible|ful|less)$"
)
_HALFWIDTH_PUNCT_RE = re.compile(r'[,.;:!?()]')
_HALFWIDTH_QUOTE_RE = re.compile(r'"')
_ASCII_ELLIPSIS_RE = re.compile(r'\.{3,}')
_INTERJECTION_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# Bare hints cover single-word speaker labels (Bouncer, Sir, Boss); prefixed
# hints cover multi-word labels such as ``Church Knight`` / ``City Guard``.
SPEAKER_HINT_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        {hint for hint in DEFAULT_SPEAKER_OCCUPATION_HINTS}
        | {f' {hint}' for hint in DEFAULT_SPEAKER_OCCUPATION_HINTS},
        key=len,
        reverse=True,
    )
)

DEFAULT_POLICY: dict[str, Any] = {
    'schema_version': QUALITY_RULE_SCHEMA_VERSION,
    'enabled': True,
    'rules': {key: DISPOSITION_WARNING for key in RULE_KEYS},
    'allowed_latin_tokens': list(DEFAULT_ALLOWED_LATIN_TOKENS),
    'garbled_phrases': [],
}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {'0', 'false', 'no', 'off'}
    return bool(value)


def _as_text(value: Any) -> str:
    return str(value or '').strip()


def _as_text_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = _as_text(value)
        if text and text not in result:
            result.append(text)
    return result


def _resolve_reason_code(key: str) -> str:
    text = key.strip()
    if text in ALL_REASON_CODES:
        return text
    return RULE_KEYS.get(text, '')


def normalize_policy(configured: Any) -> dict[str, Any]:
    """Normalize a ``batch.quality_gate`` project configuration object."""

    policy = {
        'schema_version': QUALITY_RULE_SCHEMA_VERSION,
        'enabled': True,
        'rules': {key: DISPOSITION_WARNING for key in RULE_KEYS},
        'allowed_latin_tokens': list(DEFAULT_ALLOWED_LATIN_TOKENS),
        'garbled_phrases': [],
    }
    if not isinstance(configured, Mapping):
        return policy

    policy['enabled'] = _as_bool(configured.get('enabled'), True)
    raw_rules = configured.get('rules')
    if isinstance(raw_rules, Mapping):
        for raw_key, raw_disposition in raw_rules.items():
            reason_code = _resolve_reason_code(str(raw_key))
            if not reason_code:
                continue
            disposition = _as_text(raw_disposition).lower()
            if disposition not in VALID_DISPOSITIONS:
                disposition = DISPOSITION_WARNING
            policy['rules'][REASON_TO_RULE_KEY[reason_code]] = disposition

    allowed = _as_text_list(configured.get('allowed_latin_tokens'))
    if allowed:
        merged = list(DEFAULT_ALLOWED_LATIN_TOKENS)
        for token in allowed:
            if token not in merged:
                merged.append(token)
        policy['allowed_latin_tokens'] = merged
    # An empty list means "no project additions" and intentionally keeps the
    # built-in conservative allowlist so obvious acronyms do not become noise.

    policy['garbled_phrases'] = _as_text_list(configured.get('garbled_phrases'))
    return policy


def policy_digest(policy: Mapping[str, Any] | None) -> str:
    """Return a stable digest for the effective quality policy."""

    payload = dict(policy or normalize_policy(None))
    payload = {
        'schema_version': payload.get('schema_version'),
        'enabled': payload.get('enabled'),
        'rules': payload.get('rules'),
        'allowed_latin_tokens': payload.get('allowed_latin_tokens'),
        'garbled_phrases': payload.get('garbled_phrases'),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def glossary_digest(glossary_map: Mapping[str, str] | None) -> str:
    """Stable digest of the glossary pairs actually consumed by quality rules."""

    entries = [
        [str(source), str(target)]
        for source, target in (glossary_map or {}).items()
        if str(source).strip() or str(target).strip()
    ]
    entries.sort(key=lambda pair: (pair[0], pair[1]))
    serialized = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def load_policy_from_config(translator_config: Any) -> dict[str, Any]:
    if not isinstance(translator_config, Mapping):
        return normalize_policy(None)
    batch = translator_config.get('batch')
    if not isinstance(batch, Mapping):
        return normalize_policy(None)
    return normalize_policy(batch.get('quality_gate'))


def effective_policy(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Prefer the policy snapshotted in a manifest, then runtime defaults."""

    if isinstance(manifest, Mapping):
        stored = manifest.get('quality_policy')
        if isinstance(stored, Mapping):
            return normalize_policy(stored)
    return normalize_policy(None)


def manifest_quality_policy_fields(
    source_manifest: Mapping[str, Any] | None = None,
    *,
    runtime_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(source_manifest, Mapping) and isinstance(source_manifest.get('quality_policy'), Mapping):
        policy = effective_policy(source_manifest)
    elif isinstance(runtime_policy, Mapping):
        policy = normalize_policy(runtime_policy)
    else:
        policy = normalize_policy(None)
    return {'quality_policy': policy}


def disposition_for(policy: Mapping[str, Any], reason_code: str) -> str:
    if not _as_bool(policy.get('enabled'), True):
        return DISPOSITION_OFF
    rules = policy.get('rules')
    if not isinstance(rules, Mapping):
        return DISPOSITION_WARNING
    key = REASON_TO_RULE_KEY.get(reason_code, reason_code)
    disposition = rules.get(key, rules.get(reason_code, DISPOSITION_WARNING))
    text = _as_text(disposition).lower()
    return text if text in VALID_DISPOSITIONS else DISPOSITION_WARNING


def allowed_latin_tokens(policy: Mapping[str, Any]) -> set[str]:
    tokens = policy.get('allowed_latin_tokens')
    if not isinstance(tokens, list):
        return set()
    return {
        text.casefold()
        for raw in tokens
        if (text := _as_text(raw))
    }


def _strip_markup(text: str) -> str:
    previous = None
    current = text or ''
    # Valid Ren'Py tags and fields are balanced; remove them before lexical rules.
    while previous != current:
        previous = current
        current = _RENPY_TAG_RE.sub('', current)
        current = _RENPY_FIELD_RE.sub('', current)
    return current


def _mask_markup(text: str) -> str:
    """Replace balanced Ren'Py tags/fields with spaces of the same length.

    The returned string keeps original character offsets, so evidence spans
    collected by lexical rules still point at the original translation.
    """

    previous = None
    current = text or ''
    while previous != current:
        previous = current
        current = _RENPY_TAG_RE.sub(lambda match: ' ' * len(match.group(0)), current)
        current = _RENPY_FIELD_RE.sub(lambda match: ' ' * len(match.group(0)), current)
    return current


def _line_number(subject: Mapping[str, Any]) -> int:
    value = subject.get('line_number')
    if isinstance(value, int) and value > 0:
        return value
    value = subject.get('line')
    if isinstance(value, int) and value >= 0:
        return value + 1
    return 0


def _make_finding(
    policy: Mapping[str, Any],
    reason_code: str,
    subject: Mapping[str, Any],
    *,
    evidence: str,
    suggestion: str = '',
    severity: str = 'medium',
) -> dict[str, Any]:
    disposition = disposition_for(policy, reason_code)
    if disposition == DISPOSITION_BLOCKER:
        severity = 'high'
    else:
        severity = str(severity or 'medium')
    finding_id = hashlib.sha256(
        json.dumps(
            [
                QUALITY_FINDING_SCHEMA_VERSION,
                str(subject.get('item_id') or ''),
                str(subject.get('file_rel_path') or subject.get('file') or ''),
                _line_number(subject),
                reason_code,
                str(evidence or ''),
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()[:20]
    return {
        'finding_id': finding_id,
        'schema_version': QUALITY_FINDING_SCHEMA_VERSION,
        'reason_code': reason_code,
        'rule_id': REASON_TO_RULE_KEY.get(reason_code, reason_code),
        'severity': severity,
        'disposition': disposition,
        'item_id': str(subject.get('item_id') or ''),
        'file': str(subject.get('file_rel_path') or subject.get('file') or ''),
        'line': _line_number(subject),
        'source': str(subject.get('source') or ''),
        'translation': str(subject.get('translation') or ''),
        'evidence': str(evidence or ''),
        'suggestion': str(suggestion or ''),
        'rule_version': QUALITY_RULE_SCHEMA_VERSION,
    }


def check_wait_tag_inside_cjk(subject: Mapping[str, Any]) -> list[dict[str, Any]]:
    translation = str(subject.get('translation') or '')
    matches = _WAIT_TAG_RE.finditer(translation)
    return [
        {'match': match.group(0), 'span': (match.start(), match.end())}
        for match in matches
    ]


def check_unclosed_delimiters(subject: Mapping[str, Any]) -> list[dict[str, Any]]:
    translation = str(subject.get('translation') or '')
    # ``[[`` / ``]]`` and ``{{`` / ``}}`` are Ren'Py escapes for literal
    # brackets and braces; remove those paired tokens before counting.
    normalized = (
        translation.replace('{{', '')
        .replace('}}', '')
        .replace('[[', '')
        .replace(']]', '')
    )
    evidence: list[dict[str, Any]] = []
    if normalized.count('{') != normalized.count('}'):
        evidence.append(
            {
                'delimiter': '{}',
                'opening': normalized.count('{'),
                'closing': normalized.count('}'),
            }
        )
    if normalized.count('[') != normalized.count(']'):
        evidence.append(
            {
                'delimiter': '[]',
                'opening': normalized.count('['),
                'closing': normalized.count(']'),
            }
        )
    return evidence


def check_english_suffix_adjacent(
    subject: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    translation = _mask_markup(str(subject.get('translation') or ''))
    allowed = allowed_latin_tokens(policy)
    evidence: list[dict[str, Any]] = []
    for match in _LATIN_TOKEN_RE.finditer(translation):
        token = match.group(0)
        if not token or match.start() == 0:
            continue
        if token.casefold() in allowed:
            continue
        if translation[match.start() - 1] not in '“”‘’「」『』（）：:、。！？；;,.!?()[]{} \t\r\n':
            # Direct adjacency is required for the suffix-specific rule; the
            # general spacing rule reports all other CJK/Latin boundaries.
            if re.search(_CJK_CLASS, translation[match.start() - 1]):
                if _ENGLISH_SUFFIX_RE.search(token) or token.casefold() in {'s', 'es'}:
                    evidence.append({'token': token, 'span': (match.start(), match.end())})
    return evidence


def check_cjk_latin_spacing(
    subject: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    translation = _mask_markup(str(subject.get('translation') or ''))
    allowed = allowed_latin_tokens(policy)
    evidence: list[dict[str, Any]] = []
    for match in _LATIN_TOKEN_RE.finditer(translation):
        token = match.group(0)
        if not token:
            continue
        token_key = token.casefold()
        if token_key in allowed:
            continue
        left = translation[match.start() - 1] if match.start() > 0 else ''
        right = translation[match.end()] if match.end() < len(translation) else ''
        if re.search(_CJK_CLASS, left) or re.search(_CJK_CLASS, right):
            evidence.append(
                {
                    'token': token,
                    'span': (match.start(), match.end()),
                    'side': 'left' if re.search(_CJK_CLASS, left) else 'right',
                }
            )
    return evidence


def check_halfwidth_punctuation(subject: Mapping[str, Any]) -> list[dict[str, Any]]:
    translation = str(subject.get('translation') or '')
    evidence: list[dict[str, Any]] = []
    for match in _HALFWIDTH_PUNCT_RE.finditer(translation):
        index = match.start()
        left = translation[index - 1] if index > 0 else ''
        right = translation[index + 1] if index + 1 < len(translation) else ''
        if re.search(_CJK_CLASS, left) or re.search(_CJK_CLASS, right):
            evidence.append({'punctuation': match.group(0), 'span': (match.start(), match.end())})
    for match in _HALFWIDTH_QUOTE_RE.finditer(translation):
        index = match.start()
        left = translation[index - 1] if index > 0 else ''
        right = translation[index + 1] if index + 1 < len(translation) else ''
        if re.search(_CJK_CLASS, left) or re.search(_CJK_CLASS, right):
            evidence.append({'punctuation': '"', 'span': (match.start(), match.end())})
    return evidence


def check_ascii_ellipsis(subject: Mapping[str, Any]) -> list[dict[str, Any]]:
    translation = str(subject.get('translation') or '')
    evidence: list[dict[str, Any]] = []
    for match in _ASCII_ELLIPSIS_RE.finditer(translation):
        left = translation[match.start() - 1] if match.start() > 0 else ''
        right = translation[match.end()] if match.end() < len(translation) else ''
        if re.search(_CJK_CLASS, left) or re.search(_CJK_CLASS, right):
            evidence.append({'token': match.group(0), 'span': (match.start(), match.end())})
    return evidence


def check_suspicious_english_residue(
    subject: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    translation = str(subject.get('translation') or '')
    if not re.search(_CJK_CLASS, translation):
        return []
    allowed = allowed_latin_tokens(policy)
    cleaned = _mask_markup(translation)
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for match in _LATIN_TOKEN_RE.finditer(cleaned):
        token = match.group(0)
        token_key = token.casefold()
        if token_key in allowed or len(token) < 2:
            continue
        # Numbers only are not visible English residue.
        if not any(ch.isalpha() for ch in token):
            continue
        span = (match.start(), match.end())
        if span in seen:
            continue
        seen.add(span)
        evidence.append({'token': token, 'span': span})
    return evidence


def load_glossary_map(
    glossary_path: str | os.PathLike[str],
    *,
    base_dir: str = "",
) -> dict[str, str]:
    """Load ``normalize_map`` / ``translations`` glossary pairs.

    Relative paths are resolved against *base_dir* so a project-local
    ``glossary.json`` does not silently resolve against the process CWD.
    """

    raw_path = str(glossary_path or "").strip()
    if not raw_path:
        return {}
    path = raw_path
    if not os.path.isabs(path) and base_dir:
        path = os.path.join(str(base_dir), path)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle) or {}
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(data, Mapping):
        return {}
    mapping = data.get("normalize_map")
    if not isinstance(mapping, Mapping):
        mapping = data.get("translations") or {}
    if not isinstance(mapping, Mapping):
        return {}
    result: dict[str, str] = {}
    for raw_source, raw_target in mapping.items():
        source = str(raw_source).strip()
        target = str(raw_target).strip()
        if source and target:
            result[source] = target
    return result


def _load_glossary_map(manifest: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(manifest, Mapping):
        return {}
    glossary_path = manifest.get("glossary_file")
    if not isinstance(glossary_path, str) or not glossary_path.strip():
        glossary_path = str(os.environ.get("GLOSSARY_FILE") or "")
    if not glossary_path:
        return {}
    base_dir = str(
        manifest.get("_package_dir")
        or manifest.get("base_dir")
        or ""
    )
    return load_glossary_map(glossary_path, base_dir=base_dir)


def _contains_glossary_term(text: str, term: str) -> bool:
    """Return whether *term* occurs as a word, not as a substring.

    Short Latin glossary entries such as ``art`` must not match inside
    ``heart`` or ``start``.
    """

    if not text or not term:
        return False
    if re.fullmatch(r"[A-Za-z0-9_'\-\s]+", term) and any(ch.isalpha() for ch in term):
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_'\-]){re.escape(term)}(?![A-Za-z0-9_'\-])",
                text,
            )
        )
    return term in text


def check_glossary_term_not_applied(
    subject: Mapping[str, Any],
    glossary_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    source_text = str(subject.get('source') or '')
    translation = str(subject.get('translation') or '')
    if not source_text or not translation or not glossary_map:
        return []
    evidence: list[dict[str, Any]] = []
    for source, target in glossary_map.items():
        if not _contains_glossary_term(source_text, source):
            continue
        # A mechanical rule cannot prove a free but correct translation is
        # missing the glossary target.  It can, however, flag the source term
        # still visible verbatim in the translated output.
        if _contains_glossary_term(translation, source) and not _contains_glossary_term(
            translation,
            target,
        ):
            evidence.append({'source': source, 'target': target})
    return evidence


def check_speaker_label_untranslated(
    subject: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    speaker_name = str(
        subject.get('speaker_name')
        or subject.get('speaker_display_name')
        or ''
    ).strip()
    translation = str(subject.get('translation') or '')
    source = str(subject.get('source') or '')
    if not speaker_name:
        return []
    if not re.search(_CJK_CLASS, translation) and re.search(_CJK_CLASS, source):
        # Whole translation is already covered by other quality/structural
        # rules; avoid duplicate speaker-specific noise.
        return []
    allowed = allowed_latin_tokens(policy)
    normalized_speaker = ' '.join(speaker_name.split())
    for hint in SPEAKER_HINT_SUFFIXES:
        if not normalized_speaker.endswith(hint):
            continue
        token = hint.strip()
        if token.casefold() in allowed:
            return []
        if re.search(rf'(?<![A-Za-z0-9\'\-]){re.escape(token)}(?![A-Za-z0-9\'\-])', translation):
            return [{'speaker_name': speaker_name, 'token': token}]
        translated_label = str(
            subject.get('speaker_name_translation')
            or subject.get('speaker_display_name_translation')
            or ''
        ).strip()
        if translated_label and re.search(_CJK_CLASS, translated_label):
            return []
        # The dialogue body is translated but the source occupation/identity
        # label itself is still English with no translated label evidence.
        if re.search(_CJK_CLASS, translation):
            return [
                {
                    'speaker_name': speaker_name,
                    'token': token,
                    'reason': 'speaker_label_has_no_translation_evidence',
                }
            ]
    return []


def check_interjection_untranslated(subject: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = str(subject.get('source') or '')
    translation = str(subject.get('translation') or '')
    if not source or not translation:
        return []
    source_tokens = [token.casefold() for token in _INTERJECTION_TOKEN_RE.findall(source)]
    if not source_tokens:
        return []
    interjections = {token.casefold() for token in DEFAULT_INTERJECTIONS}
    candidates = {
        token.casefold()
        for token in source_tokens
        if token in interjections
    }
    if not candidates:
        return []
    translated_tokens = {token.casefold() for token in _INTERJECTION_TOKEN_RE.findall(translation)}
    unchanged = sorted(candidates & translated_tokens)
    if unchanged and re.search(_CJK_CLASS, source) is None:
        return [{'tokens': unchanged}]
    return []


def check_known_garbled_phrase(
    subject: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    translation = str(subject.get('translation') or '')
    phrases = policy.get('garbled_phrases')
    if not isinstance(phrases, list) or not translation:
        return []
    return [{'phrase': phrase} for phrase in phrases if phrase and phrase in translation]


RULE_CHECKERS = {
    REASON_WAIT_TAG_INSIDE_CJK: lambda subject, policy, glossary_map: check_wait_tag_inside_cjk(subject),
    REASON_UNCLOSED_DELIMITERS: lambda subject, policy, glossary_map: check_unclosed_delimiters(subject),
    REASON_ENGLISH_SUFFIX_ADJACENT: (
        lambda subject, policy, glossary_map: check_english_suffix_adjacent(subject, policy)
    ),
    REASON_SUSPICIOUS_ENGLISH_RESIDUE: (
        lambda subject, policy, glossary_map: check_suspicious_english_residue(subject, policy)
    ),
    REASON_CJK_LATIN_SPACING: (
        lambda subject, policy, glossary_map: check_cjk_latin_spacing(subject, policy)
    ),
    REASON_HALFWIDTH_PUNCTUATION: (
        lambda subject, policy, glossary_map: check_halfwidth_punctuation(subject)
    ),
    REASON_ASCII_ELLIPSIS: lambda subject, policy, glossary_map: check_ascii_ellipsis(subject),
    REASON_GLOSSARY_TERM_NOT_APPLIED: (
        lambda subject, policy, glossary_map: check_glossary_term_not_applied(subject, glossary_map)
    ),
    REASON_SPEAKER_LABEL_UNTRANSLATED: (
        lambda subject, policy, glossary_map: check_speaker_label_untranslated(subject, policy)
    ),
    REASON_INTERJECTION_UNTRANSLATED: (
        lambda subject, policy, glossary_map: check_interjection_untranslated(subject)
    ),
    REASON_KNOWN_GARBLED_PHRASE: (
        lambda subject, policy, glossary_map: check_known_garbled_phrase(subject, policy)
    ),
}

SUGGESTIONS: dict[str, str] = {
    REASON_WAIT_TAG_INSIDE_CJK: 'Move the wait tag outside the Chinese word boundary.',
    REASON_UNCLOSED_DELIMITERS: 'Repair or remove the broken {}/[] token.',
    REASON_ENGLISH_SUFFIX_ADJACENT: 'Remove the English morphology suffix from the Chinese word.',
    REASON_SUSPICIOUS_ENGLISH_RESIDUE: 'Translate the visible English token or add it to the allowlist.',
    REASON_CJK_LATIN_SPACING: 'Insert the project-required spacing between CJK and Latin text.',
    REASON_HALFWIDTH_PUNCTUATION: 'Use full-width or Chinese quotation punctuation.',
    REASON_ASCII_ELLIPSIS: 'Use the project Chinese ellipsis instead of ASCII dots.',
    REASON_GLOSSARY_TERM_NOT_APPLIED: 'Apply the glossary target translation.',
    REASON_SPEAKER_LABEL_UNTRANSLATED: 'Translate or localize the speaker label.',
    REASON_INTERJECTION_UNTRANSLATED: 'Translate the short interjection or onomatopoeia.',
    REASON_KNOWN_GARBLED_PHRASE: 'Repair or retranslate the known garbled phrase.',
}


def check_subject(
    subject: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
    glossary_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run every enabled mechanical rule against one translated item."""

    effective = normalize_policy(policy)
    glossary = dict(glossary_map or {})
    findings: list[dict[str, Any]] = []
    if not _as_bool(effective.get('enabled'), True):
        return findings
    for reason_code, checker in RULE_CHECKERS.items():
        if disposition_for(effective, reason_code) == DISPOSITION_OFF:
            continue
        try:
            evidence = checker(subject, effective, glossary) or []
        except (TypeError, ValueError, IndexError):
            # A rule bug must never turn check into a traceback or writeback
            # blocker; the structural contract remains authoritative.
            continue
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            findings.append(
                _make_finding(
                    effective,
                    reason_code,
                    subject,
                    evidence=json.dumps(dict(item), ensure_ascii=False, sort_keys=True),
                    suggestion=SUGGESTIONS.get(reason_code, ''),
                )
            )
    findings.sort(key=lambda finding: (finding['file'], finding['line'], finding['reason_code']))
    return findings


def check_quality(
    subjects: Iterable[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
    glossary_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run all rules over a collection of translated items."""

    effective = policy if isinstance(policy, Mapping) else effective_policy(manifest)
    glossary = dict(glossary_map or {})
    if glossary_map is None and isinstance(manifest, Mapping):
        glossary = _load_glossary_map(manifest)
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for subject in subjects:
        if not isinstance(subject, Mapping):
            continue
        for finding in check_subject(subject, policy=effective, glossary_map=glossary):
            # ``finding_id`` already includes item/file/line/reason/evidence, so
            # multiple independent hits on the same line are preserved.
            key = str(finding.get('finding_id') or '')
            if not key or key in seen:
                continue
            seen.add(key)
            findings.append(finding)
    findings.sort(key=lambda finding: (finding['file'], finding['line'], finding['reason_code']))
    return findings


def make_unmatched_quality_subject_finding(
    collection_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the collection-diagnostic finding for unmappable actions."""

    evidence = json.dumps(
        dict(collection_stats),
        ensure_ascii=False,
        sort_keys=True,
    )
    finding_id = hashlib.sha256(
        f'{QUALITY_FINDING_SCHEMA_VERSION}:{evidence}'.encode('utf-8')
    ).hexdigest()[:20]
    return {
        'finding_id': finding_id,
        'schema_version': QUALITY_FINDING_SCHEMA_VERSION,
        'reason_code': REASON_UNMATCHED_QUALITY_SUBJECT,
        'rule_id': 'unmatched_subject',
        'severity': SEVERITY_MEDIUM,
        'disposition': DISPOSITION_WARNING,
        'item_id': '',
        'file': '',
        'line': 0,
        'source': '',
        'translation': '',
        'evidence': evidence,
        'suggestion': (
            'Inspect quality_action_items / quality_unmatched_items and rerun check.'
        ),
        'rule_version': QUALITY_RULE_SCHEMA_VERSION,
    }


def _count_disposition(findings: Iterable[Mapping[str, Any]], disposition: str) -> int:
    return sum(
        1
        for finding in findings
        if isinstance(finding, Mapping) and finding.get('disposition') == disposition
    )


def summarize_quality_gate(
    findings: Iterable[Mapping[str, Any]],
    *,
    acknowledged_ids: Iterable[Any] | None = None,
) -> dict[str, Any]:
    findings = [dict(item) for item in findings if isinstance(item, Mapping)]
    acknowledged = {str(value) for value in (acknowledged_ids or [])}
    warning_count = _count_disposition(findings, DISPOSITION_WARNING)
    blocker_count = _count_disposition(findings, DISPOSITION_BLOCKER)
    # Only warning dispositions can be acknowledged.  Acknowledging a blocker
    # must never consume the unacknowledged-warning budget or turn a blocked
    # batch into an apparently acknowledged state.
    acknowledged_count = sum(
        1
        for finding in findings
        if finding.get('disposition') == DISPOSITION_WARNING
        and str(finding.get('finding_id') or '') in acknowledged
    )
    has_warnings = warning_count > 0 or blocker_count > 0
    unacknowledged_warnings = max(0, warning_count - acknowledged_count)
    if blocker_count:
        decision = GATE_NEEDS_REVIEW
    elif unacknowledged_warnings:
        decision = GATE_NEEDS_REVIEW
    elif warning_count:
        decision = 'acknowledged'
    else:
        decision = GATE_PASS
    return {
        'decision': decision,
        'warning_count': warning_count,
        'blocker_count': blocker_count,
        'acknowledged_count': acknowledged_count,
        'has_warnings': has_warnings,
    }


def overall_check_status(writeback_gate: Mapping[str, Any], quality_gate: Mapping[str, Any]) -> str:
    can_apply = bool((writeback_gate or {}).get('can_apply'))
    has_warnings = bool((quality_gate or {}).get('has_warnings'))
    if not can_apply:
        return GATE_BLOCKED
    if has_warnings:
        return GATE_READY_WITH_WARNINGS
    return GATE_READY


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def warning_finding_ids(findings: Iterable[Mapping[str, Any]]) -> set[str]:
    """Return current warning finding IDs that acknowledgement may reference."""

    return {
        str((finding.get('finding_id') or '')).strip()
        for finding in findings
        if isinstance(finding, Mapping)
        and finding.get('disposition') == DISPOSITION_WARNING
        and str((finding.get('finding_id') or '')).strip()
    }


def select_quality_findings_path(manifest: Mapping[str, Any] | None) -> str:
    """Return the relative or absolute findings path recorded on a manifest."""

    payload = manifest if isinstance(manifest, Mapping) else {}
    for key in (
        'last_quality_findings_path',
        'last_revision_quality_findings_path',
        'quality_findings_path',
    ):
        report_path = payload.get(key)
        if isinstance(report_path, str) and report_path.strip():
            return report_path.strip()
    for summary_key in ('revision_apply_summary', 'last_revision_apply_summary'):
        apply_summary = payload.get(summary_key)
        if not isinstance(apply_summary, Mapping):
            continue
        report_path = apply_summary.get('quality_findings_path')
        if isinstance(report_path, str) and report_path.strip():
            return report_path.strip()
    revision_preview = payload.get('last_revision_preview')
    if isinstance(revision_preview, Mapping):
        report_path = revision_preview.get('quality_findings_path')
        if isinstance(report_path, str) and report_path.strip():
            return report_path.strip()
    last_summary = payload.get('last_check_summary')
    if isinstance(last_summary, Mapping):
        report_path = last_summary.get('quality_findings_path')
        if isinstance(report_path, str) and report_path.strip():
            return report_path.strip()
    return ''


def resolve_quality_findings_path(
    manifest: Mapping[str, Any] | None,
    *,
    package_dir: str = '',
    manifest_path: str = '',
) -> str:
    """Resolve the findings report CLI and GUI should both read.

    Preference order matches the GUI readers: explicit last-report fields,
    then revision apply/preview snapshots, then the last check summary, then
    ``quality_findings.jsonl`` next to the manifest.
    """

    selected = select_quality_findings_path(manifest)
    base = str(package_dir or '').strip()
    if not base:
        payload = manifest if isinstance(manifest, Mapping) else {}
        recorded = payload.get('_package_dir')
        if isinstance(recorded, str) and recorded.strip():
            base = recorded.strip()
        elif str(manifest_path or '').strip():
            base = os.path.dirname(str(manifest_path).strip())
    if not selected:
        selected = 'quality_findings.jsonl' if base else ''
    if not selected:
        return ''
    if os.path.isabs(selected):
        return selected
    if base:
        return os.path.join(base, selected)
    return selected


def prune_acknowledged_finding_ids(
    acknowledged_ids: Iterable[Any] | None,
    findings: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Keep only acknowledgement IDs that still match current warning findings."""

    current = warning_finding_ids(findings)
    kept = {
        str((finding_id or '')).strip()
        for finding_id in (acknowledged_ids or [])
        if str((finding_id or '')).strip() in current
    }
    return sorted(kept)


def update_manifest_quality_gate(
    manifest: Mapping[str, Any],
    quality_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Refresh the cached quality gate inside ``last_check_summary``.

    Acknowledging warnings is allowed to change ``quality_gate`` and the
    derived ``check_status`` / ``has_warnings`` fields, but it must never
    rewrite ``writeback_gate``.  Blockers and structural safety decisions are
    only produced by a real check, so that gate is left authoritative.
    """

    manifest = dict(manifest)
    last_summary = manifest.get('last_check_summary')
    if not isinstance(last_summary, dict):
        last_summary = {}
    else:
        last_summary = dict(last_summary)
    quality_gate = dict(quality_gate)
    quality_gate.setdefault('has_warnings', False)
    quality_gate.setdefault('acknowledged_count', 0)
    last_summary['quality_gate'] = quality_gate
    last_summary['has_warnings'] = bool(quality_gate.get('has_warnings'))
    if 'quality_blocker_count' in last_summary:
        last_summary['quality_blocker_count'] = int(
            quality_gate.get('blocker_count') or 0
        )
    writeback_gate = last_summary.get('writeback_gate')
    if isinstance(writeback_gate, dict):
        last_summary['can_apply'] = bool(writeback_gate.get('can_apply'))
        last_summary['check_status'] = overall_check_status(
            writeback_gate,
            quality_gate,
        )
    else:
        last_summary['can_apply'] = True
        last_summary['check_status'] = overall_check_status(
            {'can_apply': True},
            quality_gate,
        )
    manifest['last_check_summary'] = last_summary
    return dict(manifest)


def apply_manifest_quality_acknowledgement(
    manifest: Mapping[str, Any],
    findings: Iterable[Mapping[str, Any]],
    *,
    finding_ids: Iterable[Any] = (),
    all_findings: bool = False,
    unack: bool = False,
) -> dict[str, Any]:
    """Apply a quality acknowledgement update to a manifest.

    Returns the mutated manifest plus the ids that were selected, unmatched,
    and the freshly summarized quality gate.  Only warning dispositions can be
    acknowledged; blocker ids are ignored for acknowledgement purposes.
    """

    manifest = dict(manifest)
    findings = [dict(item) for item in findings if isinstance(item, Mapping)]
    warning_ids = warning_finding_ids(findings)
    requested = {
        str((finding_id or '')).strip()
        for finding_id in finding_ids
        if str((finding_id or '')).strip()
    }
    if all_findings:
        selected_ids = set(warning_ids)
        unmatched: list[str] = []
    else:
        selected_ids = requested & warning_ids
        unmatched = sorted(requested - warning_ids)
    current_ids = set(
        prune_acknowledged_finding_ids(
            manifest.get('quality_acknowledged_finding_ids'),
            findings,
        )
    )
    new_ids = current_ids - selected_ids if unack else current_ids | selected_ids
    new_ids &= warning_ids
    manifest['quality_acknowledged_finding_ids'] = sorted(new_ids)
    quality_gate = summarize_quality_gate(findings, acknowledged_ids=new_ids)
    manifest = update_manifest_quality_gate(manifest, quality_gate)
    return {
        'manifest': manifest,
        'quality_gate': quality_gate,
        'selected_ids': selected_ids,
        'unmatched': unmatched,
        'acknowledged_finding_ids': sorted(new_ids),
    }


def _coerce_line(value: Any) -> int:
    return max(0, _coerce_int(value, 0))


def _coerce_rule_version(value: Any, *, default: int) -> int:
    """Normalize rule versions without dropping intentionally unknown values."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return default


def normalize_finding(finding: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure a persisted finding carries the required contract fields.

    Extra fields (for example final-review provenance) are preserved so shared
    GUI consumers can display the same record regardless of producer.
    """

    if not isinstance(finding, Mapping):
        raise ValueError('quality finding must be a mapping')
    item = dict(finding)
    schema_version = _coerce_int(
        item.get('schema_version'),
        QUALITY_FINDING_SCHEMA_VERSION,
    )
    reason_code = str(item.get('reason_code') or '').strip()
    rule_id = str(item.get('rule_id') or '').strip() or REASON_TO_RULE_KEY.get(
        reason_code,
        reason_code,
    )
    severity = str(item.get('severity') or SEVERITY_MEDIUM).strip().lower()
    if severity not in VALID_SEVERITIES:
        severity = SEVERITY_MEDIUM
    disposition = str(item.get('disposition') or DISPOSITION_WARNING).strip().lower()
    if disposition not in VALID_DISPOSITIONS:
        disposition = DISPOSITION_WARNING

    item['schema_version'] = schema_version
    if not item.get('finding_id'):
        item['finding_id'] = hashlib.sha256(
            json.dumps(
                [
                    schema_version,
                    str(item.get('item_id') or ''),
                    str(item.get('file') or ''),
                    _coerce_line(item.get('line')),
                    reason_code,
                    str(item.get('evidence') or ''),
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode('utf-8')
        ).hexdigest()[:20]
    item['finding_id'] = str(item.get('finding_id') or '').strip()
    item.setdefault('reason_code', '')
    item['reason_code'] = reason_code
    item['rule_id'] = rule_id
    item['severity'] = severity
    item['disposition'] = disposition
    item['item_id'] = str(item.get('item_id') or '')
    item['file'] = str(item.get('file') or '')
    item['line'] = _coerce_line(item.get('line'))
    item['source'] = str(item.get('source') or '')
    item['translation'] = str(item.get('translation') or '')
    item['evidence'] = str(item.get('evidence') or '')
    item['suggestion'] = str(item.get('suggestion') or '')
    item['rule_version'] = _coerce_rule_version(
        item.get('rule_version'),
        default=QUALITY_RULE_SCHEMA_VERSION,
    )
    return item


def validate_finding(
    finding: Mapping[str, Any],
    *,
    require_known_reason_code: bool = False,
) -> list[str]:
    """Return structural contract violations for one finding (empty means valid).

    Validation is intentionally shape-only.  It does not judge whether a
    finding is correct; final-review semantic findings and mechanical findings
    must both satisfy the same persisted shape.
    """

    if not isinstance(finding, Mapping):
        return ['quality finding must be a mapping']
    item = dict(finding)
    errors: list[str] = []
    for field in QUALITY_FINDING_FIELDS:
        if item.get(field) is None:
            errors.append(f"missing required field: {field}")
    if not str(item.get('finding_id') or '').strip():
        errors.append('finding_id must be a non-empty string')
    schema_version = item.get('schema_version')
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 1
    ):
        errors.append('schema_version must be a positive integer')
    reason_code = item.get('reason_code')
    if not isinstance(reason_code, str) or not reason_code.strip():
        errors.append('reason_code must be a non-empty string')
    elif require_known_reason_code and reason_code not in {
        *ALL_REASON_CODES,
        REASON_UNMATCHED_QUALITY_SUBJECT,
        *FINAL_REVIEW_REASON_TO_FINDING_TYPE,
    }:
        errors.append(f'unknown reason_code: {reason_code}')
    if item.get('severity') not in VALID_SEVERITIES:
        errors.append(
            'severity must be one of: ' + ', '.join(VALID_SEVERITIES)
        )
    if item.get('disposition') not in VALID_DISPOSITIONS:
        errors.append(
            'disposition must be one of: ' + ', '.join(VALID_DISPOSITIONS)
        )
    line = item.get('line')
    if not isinstance(line, int) or isinstance(line, bool) or line < 0:
        errors.append('line must be a non-negative integer')
    rule_version = item.get('rule_version')
    if (
        not isinstance(rule_version, int)
        or isinstance(rule_version, bool)
        or rule_version < 0
    ):
        errors.append('rule_version must be a non-negative integer')
    return errors


def validate_findings(
    findings: Iterable[Mapping[str, Any]],
    *,
    require_known_reason_code: bool = False,
) -> dict[int, list[str]]:
    """Return row-indexed validation errors for an iterable of findings."""

    errors: dict[int, list[str]] = {}
    for index, finding in enumerate(findings):
        row_errors = validate_finding(
            finding,
            require_known_reason_code=require_known_reason_code,
        )
        if row_errors:
            errors[index] = row_errors
    return errors


def file_matches_filter(file_value: Any, candidate: Any) -> bool:
    """Path-aware file filter for findings and GUI file fragments.

    Exact file names and path prefixes match at component boundaries so
    ``script.rpy`` does not match ``xscript.rpy``.  Bare extension-less
    fragments such as ``script`` keep the legacy substring behaviour expected
    by the GUI file filter.
    """

    file_text = str(file_value or '').replace('\\', '/').casefold()
    candidate_text = str(candidate or '').replace('\\', '/').strip().casefold()
    if not file_text or not candidate_text:
        return False
    if file_text == candidate_text:
        return True
    if file_text.startswith(candidate_text.rstrip('/') + '/'):
        return True
    parts = file_text.split('/')
    if candidate_text in parts:
        return True
    # Bare, extension-less fragments are the GUI file-filter contract.
    return '/' not in candidate_text and '.' not in candidate_text and candidate_text in file_text


def filter_findings(
    findings: Iterable[Mapping[str, Any]],
    *,
    reason_codes: Iterable[str] | None = None,
    files: Iterable[str] | None = None,
    min_severity: str = '',
    dispositions: Iterable[str] | None = None,
    item_ids: Iterable[str] | None = None,
    text: str = '',
) -> list[dict[str, Any]]:
    """Filter normalized findings by the shared GUI/persistence dimensions."""

    selected_reasons = {
        str(value).strip()
        for value in (reason_codes or ())
        if str(value).strip()
    }
    selected_files = {
        str(value).strip().casefold()
        for value in (files or ())
        if str(value).strip()
    }
    selected_dispositions = {
        str(value).strip().lower()
        for value in (dispositions or ())
        if str(value).strip()
    }
    selected_item_ids = {
        str(value).strip()
        for value in (item_ids or ())
        if str(value).strip()
    }
    minimum = SEVERITY_ORDER.get(
        str(min_severity or '').strip().lower(),
        0,
    )
    needle = str(text or '').strip().casefold()
    result: list[dict[str, Any]] = []
    for raw in findings:
        if not isinstance(raw, Mapping):
            continue
        item = normalize_finding(raw)
        if selected_reasons and item.get('reason_code') not in selected_reasons:
            continue
        if selected_files and not any(
            file_matches_filter(item.get('file'), candidate)
            for candidate in selected_files
        ):
            continue
        if (
            SEVERITY_ORDER.get(str(item.get('severity') or ''), 2)
            < minimum
        ):
            continue
        if selected_dispositions and item.get('disposition') not in selected_dispositions:
            continue
        if selected_item_ids and item.get('item_id') not in selected_item_ids:
            continue
        if needle and needle not in ' '.join(
            (
                str(item.get('source') or ''),
                str(item.get('translation') or ''),
                str(item.get('evidence') or ''),
            )
        ).casefold():
            continue
        result.append(item)
    return result


def findings_digest(findings: Iterable[Mapping[str, Any]]) -> str:
    """Stable SHA-256 over the shared persisted finding fields."""

    rows = [
        {
            field: normalize_finding(finding).get(field)
            for field in QUALITY_FINDING_FIELDS
        }
        for finding in findings
        if isinstance(finding, Mapping)
    ]
    rows.sort(
        key=lambda row: (
            str(row.get('finding_id') or ''),
            str(row.get('file') or ''),
            int(row.get('line') or 0),
            str(row.get('reason_code') or ''),
        )
    )
    serialized = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def write_findings(
    path: str | os.PathLike[str],
    findings: Iterable[Mapping[str, Any]],
) -> str:
    """Atomically write normalized findings as JSONL and return the path."""

    from atomic_io import atomic_write_jsonl

    target = os.fspath(path)
    atomic_write_jsonl(
        target,
        [normalize_finding(finding) for finding in findings if isinstance(finding, Mapping)],
        ensure_ascii=False,
    )
    return target


def final_review_reason_code(finding_type: Any) -> str:
    """Map a final-review finding type to its stable common-schema code."""

    key = str(finding_type or '').strip().lower()
    return FINAL_REVIEW_FINDING_TYPE_TO_REASON.get(
        key,
        FINAL_REVIEW_REASON_NEEDS_CONFIRMATION,
    )


def adapt_final_review_finding(record: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a final-review finding to the common quality finding shape.

    Final-review records keep their LLM semantic fields and are never claimed
    to be deterministic mechanical findings.  The adapter only adds the shared
    location, severity, disposition, and provenance envelope used by GUI lists,
    filters, persistence, and digest checks.
    """

    if not isinstance(record, Mapping):
        raise ValueError('final-review finding must be a mapping')
    item = dict(record)
    finding_type = str(item.get('finding_type') or item.get('type') or '').strip().lower()
    if not finding_type:
        finding_type = 'needs_confirmation'
    reason_code = final_review_reason_code(finding_type)
    severity = str(item.get('severity') or SEVERITY_MEDIUM).strip().lower()
    if severity not in VALID_SEVERITIES:
        severity = SEVERITY_MEDIUM
    identity = str(
        item.get('identity_v2')
        or item.get('item_id')
        or item.get('id')
        or ''
    ).strip()
    file_rel_path = str(item.get('file_rel_path') or item.get('file') or '')
    source = str(item.get('source') or '')
    translation = str(
        item.get('current_translation')
        or item.get('translation')
        or ''
    )
    evidence = str(item.get('evidence') or '')
    reason = str(item.get('reason') or item.get('detail') or '')
    # ``evidence`` is the shared filter/detail field; the full semantic reason
    # is preserved verbatim in the ``reason`` provenance field below.
    evidence = evidence or reason
    suggestion = str(
        item.get('suggested_revision')
        or item.get('suggestion')
        or ''
    )
    finding_id = str(item.get('finding_id') or '').strip()
    if not finding_id:
        finding_id = hashlib.sha256(
            json.dumps(
                [identity, finding_type, source, translation, reason],
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode('utf-8')
        ).hexdigest()[:20]
    adapted = {
        'finding_id': finding_id,
        'schema_version': QUALITY_FINDING_SCHEMA_VERSION,
        'reason_code': reason_code,
        'rule_id': 'final_review',
        'severity': severity,
        'disposition': DISPOSITION_WARNING,
        'item_id': identity,
        'file': file_rel_path,
        'line': _coerce_line(item.get('line')),
        'source': source,
        'translation': translation,
        'evidence': evidence,
        'suggestion': suggestion,
        'rule_version': 0,
        # Semantic provenance stays intact and explicit.
        'provenance': 'final_review',
        'finding_type': finding_type,
        'reason': reason,
        'review_unit_id': str(item.get('review_unit_id') or ''),
        'review_unit_digest': str(item.get('review_unit_digest') or ''),
        'prompt_schema_version': str(item.get('prompt_schema_version') or ''),
        'selection_state': str(item.get('selection_state') or ''),
        'revision_state': str(item.get('revision_state') or ''),
    }
    return normalize_finding(adapted)


def adapt_final_review_findings(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt a collection of final-review findings for shared consumers."""

    return [
        adapt_final_review_finding(record)
        for record in records
        if isinstance(record, Mapping)
    ]


def load_findings(
    path: str | os.PathLike[str],
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Load a JSONL findings report.

    Default mode is lenient for backward compatibility: unreadable rows are
    skipped.  ``strict=True`` raises ``ValueError`` on malformed JSON or rows
    that do not satisfy :func:`validate_finding`.
    """

    target = os.fspath(path)
    if not target or not os.path.isfile(target):
        return []
    findings: list[dict[str, Any]] = []
    try:
        with open(target, 'r', encoding='utf-8-sig') as handle:
            for row_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    if strict:
                        raise ValueError(
                            f'quality findings row {row_number} is invalid JSON: {exc}'
                        ) from exc
                    continue
                if not isinstance(value, Mapping):
                    if strict:
                        raise ValueError(
                            f'quality findings row {row_number} must be an object'
                        )
                    continue
                if strict:
                    # Validate the raw persisted row before normalization so
                    # invalid enums/numbers cannot be silently coerced into
                    # defaults and then treated as contract-valid findings.
                    errors = validate_finding(value)
                    if errors:
                        raise ValueError(
                            f'quality findings row {row_number} is invalid: '
                            + '; '.join(errors)
                        )
                findings.append(normalize_finding(value))
    except OSError:
        if strict:
            raise
        return []
    return findings
