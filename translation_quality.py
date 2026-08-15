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
    acknowledged_count = sum(
        1
        for finding in findings
        if str(finding.get('finding_id') or '') in acknowledged
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


def normalize_finding(finding: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure a persisted finding carries the required contract fields."""

    item = dict(finding)
    item.setdefault('schema_version', QUALITY_FINDING_SCHEMA_VERSION)
    if not item.get('finding_id'):
        item['finding_id'] = hashlib.sha256(
            json.dumps(
                [
                    item.get('schema_version'),
                    str(item.get('item_id') or ''),
                    str(item.get('file') or ''),
                    int(item.get('line') or 0),
                    str(item.get('reason_code') or ''),
                    str(item.get('evidence') or ''),
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode('utf-8')
        ).hexdigest()[:20]
    item.setdefault('reason_code', '')
    item.setdefault('severity', 'medium')
    item.setdefault('disposition', DISPOSITION_WARNING)
    item.setdefault('item_id', '')
    item.setdefault('file', '')
    item.setdefault('line', 0)
    item.setdefault('source', '')
    item.setdefault('translation', '')
    item.setdefault('evidence', '')
    item.setdefault('suggestion', '')
    item.setdefault('rule_version', QUALITY_RULE_SCHEMA_VERSION)
    return item


def load_findings(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    findings: list[dict[str, Any]] = []
    try:
        with open(path, 'r', encoding='utf-8-sig') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, Mapping):
                    findings.append(normalize_finding(value))
    except OSError:
        return []
    return findings
