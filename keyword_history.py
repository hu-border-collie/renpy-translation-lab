"""Read-only evidence matching for historical keyword translations.

The keyword pipeline produces candidate terms from the current source view,
while the revision corpus contains old/current translation pairs that already
exist in the project.  This module only derives an explainable evidence
projection from those pairs.  It never updates a glossary or a Ren'Py file.

Matching is deliberately conservative: plain word boundaries are required,
Ren'Py interpolation-only matches are ignored, and case/plural variants are
reported as human-review conditions instead of being silently treated as an
exact match.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Sequence


HISTORY_EVIDENCE_SCHEMA_VERSION = 1
MAX_REPORTED_OCCURRENCES = 32

STATUS_CONSISTENT = "consistent"
STATUS_CONFLICT = "conflict"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNMATCHED = "unmatched"
STATUS_UNAVAILABLE = "unavailable"

_INTERPOLATION_TOKEN_RE = re.compile(
    r"\{[^{}\r\n]*\}|\[[^\[\]\r\n]*\]|%\([^)\r\n]+\)[#0+\- ]*\d*(?:\.\d+)?[a-zA-Z%]?"
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def _compact_text(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", _normalize_text(value)).strip()


def _match_key(value: Any) -> str:
    return _compact_text(value).casefold()


def is_actual_translation_row(row: Mapping[str, Any]) -> bool:
    """Return whether a corpus row has a non-empty, changed current translation."""

    source = _compact_text(row.get("source") or row.get("source_text"))
    current = _compact_text(row.get("current_translation") or row.get("translation"))
    return bool(source and current and _match_key(source) != _match_key(current))


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _interpolation_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in _INTERPOLATION_TOKEN_RE.finditer(text)]


def _span_is_inside_interpolation(
    span: tuple[int, int],
    interpolation_spans: Sequence[tuple[int, int]],
) -> bool:
    start, end = span
    return any(token_start <= start and end <= token_end for token_start, token_end in interpolation_spans)


def _space_flexible_pattern(value: str) -> str:
    """Escape a term while allowing equivalent whitespace in source text."""

    parts = re.split(r"(\s+)", value)
    return "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts if part)


def _boundary_pattern(term: str) -> str:
    pattern = _space_flexible_pattern(term)
    left = r"(?<![\w])" if term and (term[0].isalnum() or term[0] == "_") else ""
    right = r"(?![\w])" if term and (term[-1].isalnum() or term[-1] == "_") else ""
    return left + pattern + right


def _plural_variants(term: str) -> list[str]:
    """Return conservative English singular/plural alternatives for a term."""

    words = list(_WORD_RE.finditer(term))
    if not words or words[-1].end() != len(term):
        return []
    last = words[-1].group(0)
    if len(last) <= 2:
        return []
    prefix = term[:words[-1].start()]
    variants: list[str] = []
    if last.casefold().endswith("ies") and len(last) > 3:
        variants.append(prefix + last[:-3] + "y")
    elif last.casefold().endswith(("ses", "xes", "zes", "ches", "shes")):
        variants.append(prefix + last[:-2])
    elif last.casefold().endswith("s"):
        variants.append(prefix + last[:-1])
    else:
        suffix = "es" if last.casefold().endswith(("s", "x", "z", "ch", "sh")) else "s"
        variants.append(term + suffix)
    return [variant for variant in variants if _match_key(variant) != _match_key(term)]


def _find_term_matches(source_text: str, term: str) -> tuple[list[dict[str, Any]], bool]:
    """Find safe matches and report whether only interpolation matches existed."""

    source = _normalize_text(source_text)
    normalized_term = _compact_text(term)
    if not source or not normalized_term:
        return [], False

    has_interpolation = bool(_INTERPOLATION_TOKEN_RE.search(normalized_term))
    terms = [(normalized_term, "exact")]
    if not has_interpolation:
        terms.extend((variant, "plural_variant") for variant in _plural_variants(normalized_term))

    interpolation_spans = _interpolation_spans(source)
    matches: list[dict[str, Any]] = []
    interpolation_only = False
    seen_spans: set[tuple[int, int]] = set()
    for search_term, kind in terms:
        try:
            pattern = _boundary_pattern(search_term) if not has_interpolation else _space_flexible_pattern(search_term)
            found = list(re.finditer(pattern, source, flags=re.IGNORECASE))
        except re.error:
            found = []
        for match in found:
            span = match.span()
            if span in seen_spans:
                continue
            seen_spans.add(span)
            inside_interpolation = _span_is_inside_interpolation(span, interpolation_spans)
            if inside_interpolation and not has_interpolation:
                interpolation_only = True
                continue
            matched_text = match.group(0)
            match_kind = kind
            if kind == "exact" and _compact_text(matched_text) != _compact_text(normalized_term):
                match_kind = "case_variant"
            elif kind == "exact" and source[span[1]:].casefold().startswith(("'s", "’s")):
                match_kind = "possessive_variant"
            if has_interpolation:
                match_kind = "interpolation_exact"
            matches.append(
                {
                    "match_kind": match_kind,
                    "matched_text": matched_text,
                    "start": span[0],
                    "end": span[1],
                    "inside_interpolation": inside_interpolation,
                }
            )
    return matches, interpolation_only


def match_keyword_in_source(source_text: str, term: str) -> dict[str, Any]:
    """Return a deterministic, conservative match result for one source line."""

    matches, interpolation_only = _find_term_matches(source_text, term)
    return {
        "matches": matches,
        "interpolation_only": interpolation_only and not matches,
    }


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    locator = row.get("locator") if isinstance(row.get("locator"), Mapping) else {}
    return (
        _compact_text(row.get("file_rel_path")),
        _coerce_int(row.get("display_line") or row.get("line_number") or locator.get("line_number") or locator.get("line")),
        _coerce_int(locator.get("start") or row.get("start")),
        _compact_text(row.get("identity_v2") or row.get("occurrence_id") or row.get("id")),
    )


def _occurrence_from_row(row: Mapping[str, Any], match: Mapping[str, Any]) -> dict[str, Any]:
    locator = row.get("locator") if isinstance(row.get("locator"), Mapping) else {}
    identity = _compact_text(row.get("identity_v2") or row.get("occurrence_id") or row.get("id"))
    occurrence = {
        "occurrence_id": identity,
        "identity_v2": _compact_text(row.get("identity_v2") or identity),
        "file_rel_path": _compact_text(row.get("file_rel_path")),
        "line_number": _coerce_int(
            row.get("display_line") or row.get("line_number") or locator.get("line_number") or locator.get("line")
        ),
        "locator": dict(locator),
        "source": _compact_text(row.get("source") or row.get("source_text")),
        "current_translation": _compact_text(
            row.get("current_translation") or row.get("translation")
        ),
        "matched_text": _compact_text(match.get("matched_text")),
        "match_kind": _compact_text(match.get("match_kind")) or "exact",
        "match_start": _coerce_int(match.get("start")),
        "match_end": _coerce_int(match.get("end")),
    }
    speaker_id = _compact_text(row.get("speaker_id"))
    speaker_name = _compact_text(row.get("speaker_name"))
    if speaker_id:
        occurrence["speaker_id"] = speaker_id
    if speaker_name:
        occurrence["speaker_name"] = speaker_name
    return occurrence


def _unique_translations(occurrences: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for occurrence in occurrences:
        value = _compact_text(occurrence.get("current_translation"))
        key = _match_key(value)
        if not value or key in seen:
            continue
        seen.add(key)
        values.append(value)
    return values


def _target_is_visible_in_translation(target: str, translation: str) -> bool:
    target = _compact_text(target)
    translation = _compact_text(translation)
    if not target or not translation:
        return False
    if _match_key(target) == _match_key(translation):
        return True
    # Chinese text has no portable word-boundary primitive.  A substring hit
    # such as ``cart -> 车`` in ``购物车到了`` is not safe translation
    # alignment, so only an exact Chinese translation is allowed to support a
    # consistent result; all other Chinese hits remain human-review evidence.
    if any("\u4e00" <= char <= "\u9fff" for char in target):
        return False
    try:
        return re.search(_boundary_pattern(target), translation, flags=re.IGNORECASE) is not None
    except re.error:
        return False


def _reason(code: str) -> str:
    return {
        "no_history_occurrence": "没有找到已译历史 occurrence",
        "only_interpolation_match": "只在 Ren'Py 插值标记内匹配到，未作为普通术语计入",
        "interpolation_match_ignored": "另有匹配落在 Ren'Py 插值标记内，需人工确认",
        "multiple_historical_translations": "同一术语的历史 occurrence 存在多个不同现译",
        "case_variant": "匹配依赖大小写变体",
        "plural_variant": "匹配依赖单复数变体",
        "possessive_variant": "匹配到所有格/称呼变体，需要人工确认",
        "interpolation_exact": "术语包含 Ren'Py 插值标记，需要人工确认",
        "candidate_target_conflict": "候选译法与可直接对齐的首次历史译法冲突",
        "translation_alignment_unknown": "历史句有证据，但无法安全推断词级现译",
        "history_scan_unavailable": "历史 corpus 扫描不可用",
        "history_source_changed": "历史扫描期间源文件发生变化，应重新导出",
    }.get(code, code)


def _history_status(
    occurrences: Sequence[Mapping[str, Any]],
    *,
    candidate_source: str,
    candidate_target: str,
    interpolation_only: bool,
) -> tuple[str, list[str]]:
    if not occurrences:
        code = "only_interpolation_match" if interpolation_only else "no_history_occurrence"
        return STATUS_UNMATCHED, [code]

    reasons: list[str] = []
    if interpolation_only:
        reasons.append("interpolation_match_ignored")
    kinds = {str(item.get("match_kind") or "") for item in occurrences}
    if "case_variant" in kinds:
        reasons.append("case_variant")
    if "plural_variant" in kinds:
        reasons.append("plural_variant")
    if "possessive_variant" in kinds:
        reasons.append("possessive_variant")
    if "interpolation_exact" in kinds:
        reasons.append("interpolation_exact")

    translations = _unique_translations(occurrences)
    if len(translations) > 1:
        reasons.append("multiple_historical_translations")

    direct_source = _match_key(candidate_source) in {
        _match_key(occurrences[0].get("source")),
    }
    if direct_source and candidate_target:
        first_translation = str(occurrences[0].get("current_translation") or "")
        if _match_key(first_translation) != _match_key(candidate_target):
            reasons.append("candidate_target_conflict")
    elif candidate_target and not any(
        _target_is_visible_in_translation(candidate_target, item.get("current_translation", ""))
        for item in occurrences[:MAX_REPORTED_OCCURRENCES]
    ):
        reasons.append("translation_alignment_unknown")

    # Keep reason order stable and make the status explainable to both CLI and GUI.
    reasons = list(dict.fromkeys(reasons))
    if "candidate_target_conflict" in reasons:
        return STATUS_CONFLICT, reasons
    if "multiple_historical_translations" in reasons:
        return STATUS_AMBIGUOUS, reasons
    if reasons:
        return STATUS_AMBIGUOUS, reasons
    return STATUS_CONSISTENT, reasons


def build_keyword_history_evidence(
    candidate: Mapping[str, Any],
    corpus_items: Sequence[Mapping[str, Any]],
    *,
    source_changed_during_scan: bool = False,
) -> dict[str, Any]:
    """Build one stable evidence record from revision corpus rows.

    The first occurrence is selected only after sorting by relative file path,
    display line, source span, and identity.  No evidence record is an
    instruction to modify a glossary; non-consistent records are explicitly
    marked for human review.
    """

    source = _compact_text(candidate.get("source"))
    target = _compact_text(candidate.get("suggested_target"))
    matched_occurrences: list[dict[str, Any]] = []
    interpolation_only = False
    for row in sorted(corpus_items, key=_row_sort_key):
        row_source = _compact_text(row.get("source") or row.get("source_text"))
        match_result = match_keyword_in_source(row_source, source)
        if match_result["interpolation_only"]:
            interpolation_only = True
        for match in match_result["matches"]:
            matched_occurrences.append(_occurrence_from_row(row, match))

    # A row can contain a term more than once; preserve those spans for counts,
    # but keep the evidence payload bounded and deterministic.
    matched_occurrences.sort(
        key=lambda item: (
            item.get("file_rel_path", ""),
            _coerce_int(item.get("line_number")),
            _coerce_int(item.get("match_start")),
            item.get("occurrence_id", ""),
        )
    )
    status, reason_codes = _history_status(
        matched_occurrences,
        candidate_source=source,
        candidate_target=target,
        interpolation_only=interpolation_only,
    )
    if source_changed_during_scan:
        reason_codes.append("history_source_changed")
        status = STATUS_AMBIGUOUS if status == STATUS_CONSISTENT else status

    reported_occurrences = matched_occurrences[:MAX_REPORTED_OCCURRENCES]
    first = dict(reported_occurrences[0]) if reported_occurrences else None
    return {
        "schema_version": HISTORY_EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "review_required": status != STATUS_CONSISTENT,
        "candidate_source": source,
        "candidate_target": target,
        "match_count": len(matched_occurrences),
        "occurrence_count": len({item.get("occurrence_id") for item in matched_occurrences}),
        "first_occurrence": first,
        "translations": _unique_translations(matched_occurrences),
        "conflict_codes": list(dict.fromkeys(reason_codes)),
        "conflict_reasons": [_reason(code) for code in dict.fromkeys(reason_codes)],
        "occurrences": reported_occurrences,
    }


def unavailable_history_evidence(
    candidate: Mapping[str, Any],
    *,
    reason_code: str = "history_scan_unavailable",
) -> dict[str, Any]:
    """Return a fail-closed evidence record when the source scan is unavailable."""

    source = _compact_text(candidate.get("source"))
    target = _compact_text(candidate.get("suggested_target"))
    return {
        "schema_version": HISTORY_EVIDENCE_SCHEMA_VERSION,
        "status": STATUS_UNAVAILABLE,
        "review_required": True,
        "candidate_source": source,
        "candidate_target": target,
        "match_count": 0,
        "occurrence_count": 0,
        "first_occurrence": None,
        "translations": [],
        "conflict_codes": [reason_code],
        "conflict_reasons": [_reason(reason_code)],
        "occurrences": [],
    }


def attach_keyword_history_evidence(
    candidates: Sequence[Mapping[str, Any]],
    corpus_items: Sequence[Mapping[str, Any]],
    *,
    source_changed_during_scan: bool = False,
    unavailable_reason: str = "",
) -> list[dict[str, Any]]:
    """Return candidate copies enriched with read-only historical evidence."""

    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        if unavailable_reason:
            row["history_evidence"] = unavailable_history_evidence(
                candidate,
                reason_code=unavailable_reason,
            )
        else:
            row["history_evidence"] = build_keyword_history_evidence(
                candidate,
                corpus_items,
                source_changed_during_scan=source_changed_during_scan,
            )
        enriched.append(row)
    return enriched


def load_corpus_items(path: str) -> list[dict[str, Any]]:
    """Load revision-corpus rows through the #320 reader API."""

    from revision_corpus import load_corpus_items as load_revision_corpus_items

    return load_revision_corpus_items(path)
