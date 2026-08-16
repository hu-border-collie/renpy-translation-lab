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
STATUS_PRESERVE_EVIDENCE = "preserve_evidence"
STATUS_UNMATCHED = "unmatched"
STATUS_UNAVAILABLE = "unavailable"

_HISTORY_EVIDENCE_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "review_required",
        "candidate_source",
        "candidate_target",
        "match_count",
        "occurrence_count",
        "first_occurrence",
        "translations",
        "conflict_codes",
        "conflict_reasons",
        "occurrences",
    }
)
_HISTORY_OCCURRENCE_REQUIRED_FIELDS = frozenset(
    {
        "occurrence_id",
        "identity_v2",
        "file_rel_path",
        "line_number",
        "locator",
        "source",
        "current_translation",
        "matched_text",
        "match_kind",
        "match_start",
        "match_end",
    }
)

_INTERPOLATION_TOKEN_RE = re.compile(
    r"\{[^{}\r\n]*\}|\[[^\[\]\r\n]*\]|%\([^)\r\n]+\)[#0+\- ]*\d*(?:\.\d+)?[a-zA-Z%]?"
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]+")
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


def is_preserve_translation_row(row: Mapping[str, Any]) -> bool:
    """Return whether a corpus row keeps the source text unchanged.

    These rows are not ordinary translation evidence, but they are useful
    human-review evidence for preserve-term candidates (``AR -> AR``).
    They remain explicitly review-required and never become consistent.
    """

    source = _compact_text(row.get("source") or row.get("source_text"))
    current = _compact_text(row.get("current_translation") or row.get("translation"))
    return bool(source and current and _match_key(source) == _match_key(current))


def is_history_evidence_row(row: Mapping[str, Any]) -> bool:
    """Return whether a corpus row contributes to historical keyword evidence."""

    return is_actual_translation_row(row) or is_preserve_translation_row(row)


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


_SINGULAR_S_ENDING_EXCEPTIONS = frozenset(
    {
        # Common English words that end in "s" but are singular.  For these we
        # generate the regular "-es" plural instead of blindly stripping the s.
        "alias",
        "atlas",
        "bias",
        "boss",
        "bus",
        "campus",
        "canvas",
        "chaos",
        "class",
        "cosmos",
        "cross",
        "dress",
        "ethos",
        "focus",
        "gas",
        "glass",
        "grass",
        "iris",
        "kiss",
        "lens",
        "logos",
        "loss",
        "miss",
        "news",
        "oasis",
        "pathos",
        "press",
        "status",
        "this",
        "virus",
        "walrus",
        "yes",
    }
)


def _looks_like_regular_plural(word: str) -> bool:
    """Return whether a word looks like a regular English plural form."""

    lower = word.casefold()
    if len(lower) <= 2:
        return False
    if lower.endswith("ies") and len(lower) > 3:
        return True
    if lower.endswith(("ses", "xes", "zes", "ches", "shes")):
        return True
    if lower.endswith("s"):
        return lower not in _SINGULAR_S_ENDING_EXCEPTIONS and not lower.endswith(
            ("ss", "us", "is")
        )
    return False


def _plural_variants(term: str) -> list[str]:
    """Return conservative English singular/plural alternatives for a term."""

    words = list(_WORD_RE.finditer(term))
    if not words or words[-1].end() != len(term):
        return []
    last = words[-1].group(0)
    if len(last) <= 2:
        return []
    prefix = term[:words[-1].start()]
    lower = last.casefold()
    variants: list[str] = []

    # Plural -> singular: keep the existing conservative rules.
    if lower.endswith("ies") and len(last) > 3:
        variants.append(prefix + last[:-3] + "y")
    elif lower.endswith(("ses", "xes", "zes", "ches", "shes")):
        variants.append(prefix + last[:-2])
    elif _looks_like_regular_plural(last):
        variants.append(prefix + last[:-1])

    # Singular -> plural: add the missing direction.  Skip when the input
    # already looks like a plural so we do not produce forms like "storieses".
    if not _looks_like_regular_plural(last):
        if lower.endswith("y") and len(last) > 1 and lower[-2] not in "aeiou":
            variants.append(prefix + last[:-1] + "ies")
        elif lower.endswith(("s", "x", "z", "ch", "sh")):
            variants.append(term + "es")
        else:
            variants.append(term + "s")

    return [variant for variant in variants if _match_key(variant) != _match_key(term)]


def _search_specs_for_term(term: Any) -> tuple[str, list[dict[str, Any]]]:
    """Build compiled search specs for one candidate source term.

    Compiling once per candidate avoids rebuilding boundary/plural patterns for
    every corpus row in the export path.
    """

    normalized_term = _compact_text(term)
    if not normalized_term:
        return "", []

    has_interpolation = bool(_INTERPOLATION_TOKEN_RE.search(normalized_term))
    terms = [(normalized_term, "exact", has_interpolation)]
    if not has_interpolation:
        terms.extend(
            (variant, "plural_variant", False)
            for variant in _plural_variants(normalized_term)
        )

    specs: list[dict[str, Any]] = []
    for search_term, kind, is_interpolation in terms:
        try:
            pattern_text = (
                _space_flexible_pattern(search_term)
                if is_interpolation
                else _boundary_pattern(search_term)
            )
            pattern = re.compile(pattern_text, flags=re.IGNORECASE)
        except re.error:
            pattern = None
        probe_words = tuple(
            dict.fromkeys(
                match.group(0).casefold()
                for match in _ASCII_LETTER_RE.finditer(search_term)
            )
        )
        specs.append(
            {
                "search_term": search_term,
                "kind": kind,
                "pattern": pattern,
                "interpolation": is_interpolation,
                "probe_words": probe_words,
            }
        )
    return normalized_term, specs


def _matches_for_specs(
    source_text: Any,
    normalized_term: str,
    specs: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Run prepared specs against one source line.

    The second return value means "a plain-term match was skipped because it
    landed inside a Ren'Py interpolation token", regardless of whether other
    safe matches were found.
    """

    source = _normalize_text(source_text)
    if not source or not normalized_term or not specs:
        return [], False

    interpolation_spans = _interpolation_spans(source)
    matches: list[dict[str, Any]] = []
    skipped_interpolation = False
    seen_spans: set[tuple[int, int]] = set()
    for spec in specs:
        pattern = spec["pattern"]
        if pattern is None:
            continue
        for match in pattern.finditer(source):
            span = match.span()
            if span in seen_spans:
                continue
            seen_spans.add(span)
            inside_interpolation = _span_is_inside_interpolation(span, interpolation_spans)
            if inside_interpolation and not spec["interpolation"]:
                skipped_interpolation = True
                continue
            matched_text = match.group(0)
            match_kind = spec["kind"]
            if spec["kind"] == "exact" and _compact_text(matched_text) != _compact_text(normalized_term):
                match_kind = "case_variant"
            elif spec["kind"] == "exact" and source[span[1]:].casefold().startswith(("'s", "’s")):
                match_kind = "possessive_variant"
            if spec["interpolation"]:
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
    return matches, skipped_interpolation


def _find_term_matches(source_text: str, term: str) -> tuple[list[dict[str, Any]], bool]:
    """Find safe matches and report whether any interpolation-only skip happened."""

    source = _normalize_text(source_text)
    normalized_term, specs = _search_specs_for_term(term)
    if not source or not normalized_term:
        return [], False
    return _matches_for_specs(source, normalized_term, specs)


def match_keyword_in_source(source_text: str, term: str) -> dict[str, Any]:
    """Return a deterministic, conservative match result for one source line."""

    matches, skipped_interpolation = _find_term_matches(source_text, term)
    return {
        "matches": matches,
        "interpolation_only": skipped_interpolation and not matches,
        "interpolation_match_ignored": skipped_interpolation and bool(matches),
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


def _collect_matches_for_source(
    source: Any,
    corpus_items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool, bool]:
    """Collect safe matches for one candidate source across sorted corpus rows."""

    normalized_source, specs = _search_specs_for_term(source)
    matched_occurrences: list[dict[str, Any]] = []
    interpolation_only = False
    interpolation_match_ignored = False
    for row in sorted(corpus_items, key=_row_sort_key):
        row_source = _compact_text(row.get("source") or row.get("source_text"))
        if (
            is_preserve_translation_row(row)
            and _match_key(row_source) != _match_key(normalized_source)
        ):
            continue
        matches, skipped_interpolation = _matches_for_specs(
            row_source,
            normalized_source,
            specs,
        )
        if matches:
            if skipped_interpolation:
                interpolation_match_ignored = True
            matched_occurrences.extend(
                _occurrence_from_row(row, match)
                for match in matches
            )
        elif skipped_interpolation:
            interpolation_only = True

    matched_occurrences.sort(
        key=lambda item: (
            item.get("file_rel_path", ""),
            _coerce_int(item.get("line_number")),
            _coerce_int(item.get("match_start")),
            item.get("occurrence_id", ""),
        )
    )
    return matched_occurrences, interpolation_only, interpolation_match_ignored


def _candidate_probe_groups(
    specs_by_candidate: Sequence[tuple[str, list[dict[str, Any]]]],
) -> tuple[dict[str, list[int]], set[int]]:
    """Group candidates by an ASCII probe word for a cheaper row prefilter.

    A candidate is selected when any probe word appears in the row.  This is
    only a necessary condition for a regex match; the real boundary/plural
    matching still runs afterwards, so a probe collision can only cost time,
    never change evidence semantics.
    """

    probe_groups: dict[str, list[int]] = {}
    fallback_candidates: set[int] = set()
    for index, (_, specs) in enumerate(specs_by_candidate):
        if not specs:
            continue
        if any(not spec["probe_words"] for spec in specs):
            fallback_candidates.add(index)
            continue
        for spec in specs:
            probe = max(spec["probe_words"], key=lambda word: (len(word), word))
            probe_groups.setdefault(probe, []).append(index)
    return probe_groups, fallback_candidates


def _collect_matches_for_candidates(
    candidates: Sequence[Mapping[str, Any]],
    corpus_items: Sequence[Mapping[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[bool], list[bool]]:
    """Collect evidence matches for many candidates with one corpus pass.

    Instead of running ``candidates x corpus_rows`` regex scans, candidates are
    prefixed by a probe word from each search variant.  Only candidates whose
    probe occurs in a row are matched against that row.  Terms without an
    ASCII probe (for example Chinese-only candidates) remain on the old
    per-row path to preserve correctness.
    """

    specs_by_candidate: list[tuple[str, list[dict[str, Any]]]] = []
    for candidate in candidates:
        source = _compact_text(candidate.get("source"))
        normalized_source, specs = _search_specs_for_term(source)
        specs_by_candidate.append((normalized_source, specs))

    probe_groups, fallback_candidates = _candidate_probe_groups(
        specs_by_candidate
    )
    matched_by_candidate: list[list[dict[str, Any]]] = [
        [] for _ in candidates
    ]
    interpolation_only = [False] * len(candidates)
    interpolation_match_ignored = [False] * len(candidates)

    for row in sorted(corpus_items, key=_row_sort_key):
        row_is_preserve = is_preserve_translation_row(row)
        row_source = _compact_text(row.get("source") or row.get("source_text"))
        row_tokens = {
            match.group(0).casefold()
            for match in _ASCII_LETTER_RE.finditer(row_source)
        }
        selected = set(fallback_candidates)
        for token in row_tokens:
            selected.update(probe_groups.get(token, ()))

        for index in sorted(selected):
            normalized_source, specs = specs_by_candidate[index]
            if row_is_preserve and _match_key(row_source) != _match_key(normalized_source):
                continue
            matches, skipped_interpolation = _matches_for_specs(
                row_source,
                normalized_source,
                specs,
            )
            if matches:
                if skipped_interpolation:
                    interpolation_match_ignored[index] = True
                matched_by_candidate[index].extend(
                    _occurrence_from_row(row, match)
                    for match in matches
                )
            elif skipped_interpolation:
                interpolation_only[index] = True

    for index, matched_occurrences in enumerate(matched_by_candidate):
        matched_occurrences.sort(
            key=lambda item: (
                item.get("file_rel_path", ""),
                _coerce_int(item.get("line_number")),
                _coerce_int(item.get("match_start")),
                item.get("occurrence_id", ""),
            )
        )
    return matched_by_candidate, interpolation_only, interpolation_match_ignored


def _is_complete_history_occurrence(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not _HISTORY_OCCURRENCE_REQUIRED_FIELDS.issubset(value):
        return False

    for field in (
        "occurrence_id",
        "identity_v2",
        "file_rel_path",
        "source",
        "current_translation",
        "matched_text",
        "match_kind",
    ):
        if not _compact_text(value.get(field)):
            return False

    if not isinstance(value.get("locator"), Mapping):
        return False
    line_number = value.get("line_number")
    match_start = value.get("match_start")
    match_end = value.get("match_end")
    if (
        not isinstance(line_number, int)
        or isinstance(line_number, bool)
        or line_number < 1
        or not isinstance(match_start, int)
        or isinstance(match_start, bool)
        or match_start < 0
        or not isinstance(match_end, int)
        or isinstance(match_end, bool)
        or match_end <= match_start
    ):
        return False
    return True


def is_complete_consistent_history_evidence(value: object) -> bool:
    """Return whether evidence is complete enough for automatic glossary merge.

    Only a fully formed, schema-compatible ``consistent`` record is safe for
    confidence-based auto-accept.  Missing, malformed, or internally
    inconsistent fields fail closed so old or hand-written candidate files
    remain on the explicit review path.  Callers that also need to bind the
    evidence to the current candidate fields must additionally call
    :func:`history_evidence_matches_candidate`.
    """

    if not isinstance(value, Mapping):
        return False
    if not _HISTORY_EVIDENCE_REQUIRED_FIELDS.issubset(value):
        return False
    if value.get("schema_version") != HISTORY_EVIDENCE_SCHEMA_VERSION:
        return False
    if value.get("status") != STATUS_CONSISTENT:
        return False
    if value.get("review_required") is not False:
        return False

    candidate_source = value.get("candidate_source")
    candidate_target = value.get("candidate_target")
    if (
        not _compact_text(candidate_source)
        or not isinstance(candidate_target, str)
        or not _compact_text(candidate_target)
    ):
        return False

    match_count = value.get("match_count")
    occurrence_count = value.get("occurrence_count")
    occurrences = value.get("occurrences")
    if (
        not isinstance(match_count, int)
        or isinstance(match_count, bool)
        or match_count < 1
        or not isinstance(occurrence_count, int)
        or isinstance(occurrence_count, bool)
        or occurrence_count < 1
        or match_count < occurrence_count
        or not isinstance(occurrences, list)
        or not occurrences
        or match_count < len(occurrences)
    ):
        return False

    first_occurrence = value.get("first_occurrence")
    if (
        not _is_complete_history_occurrence(first_occurrence)
        or not all(_is_complete_history_occurrence(item) for item in occurrences)
        or first_occurrence not in occurrences
    ):
        return False

    first_id = _compact_text(first_occurrence.get("occurrence_id"))
    reported_ids = {
        _compact_text(item.get("occurrence_id"))
        for item in occurrences
        if isinstance(item, Mapping)
    }
    if first_id not in reported_ids or len(reported_ids) > occurrence_count:
        return False

    translations = value.get("translations")
    if (
        not isinstance(translations, list)
        or not translations
        or not all(isinstance(item, str) and _compact_text(item) for item in translations)
    ):
        return False
    translation_keys = {_match_key(item) for item in translations}
    if len(translation_keys) != len(translations) or len(translation_keys) != 1:
        return False
    if len(translations) > occurrence_count:
        return False
    unique_translation_key = next(iter(translation_keys))
    unique_translation = translations[0]
    first_translation = _match_key(first_occurrence.get("current_translation"))
    if first_translation != unique_translation_key:
        return False
    # Keep the same alignment rule as export-time status derivation: a direct
    # whole-source occurrence must match exactly, while a term embedded in a
    # longer translated sentence may align through the conservative
    # word-boundary rule (Chinese targets still require an exact match).
    if _match_key(first_occurrence.get("source")) == _match_key(candidate_source):
        target_matches_history = _match_key(candidate_target) == unique_translation_key
    else:
        target_matches_history = _target_is_visible_in_translation(
            candidate_target,
            unique_translation,
        )
    if not target_matches_history:
        return False
    reported_translation_keys = {
        _match_key(item.get("current_translation"))
        for item in occurrences
        if isinstance(item, Mapping)
    }
    if reported_translation_keys != {unique_translation_key}:
        return False

    # A consistent record must not carry hidden conflict signals.
    if value.get("conflict_codes") != [] or value.get("conflict_reasons") != []:
        return False
    return True


def history_evidence_matches_candidate(
    history_evidence: object,
    candidate: Mapping[str, Any],
) -> bool:
    """Return whether evidence snapshots the candidate's current source/target.

    Exported JSONL is editable.  A candidate whose ``suggested_target`` (or
    ``source``) changed after export still carries the old evidence snapshot,
    so the merge path must not trust that evidence for auto-accept decisions.
    """

    if not isinstance(history_evidence, Mapping):
        return False
    return (
        _match_key(history_evidence.get("candidate_source"))
        == _match_key(candidate.get("source"))
        and _match_key(history_evidence.get("candidate_target"))
        == _match_key(candidate.get("suggested_target"))
    )


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
        "preserve_evidence": "历史出现保留不译（原文=译文），需人工确认是否写入 preserve_terms",
        "history_scan_unavailable": "历史 corpus 扫描不可用",
        "history_source_changed": "历史扫描期间源文件发生变化，应重新导出",
    }.get(code, code)


def _history_status(
    occurrences: Sequence[Mapping[str, Any]],
    *,
    candidate_source: str,
    candidate_target: str,
    interpolation_only: bool,
    interpolation_match_ignored: bool = False,
) -> tuple[str, list[str]]:
    if not occurrences:
        code = "only_interpolation_match" if interpolation_only else "no_history_occurrence"
        return STATUS_UNMATCHED, [code]

    reasons: list[str] = []
    if interpolation_match_ignored:
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

    preserve_evidence = (
        _match_key(candidate_source) == _match_key(candidate_target)
        and any(
            _match_key(item.get("source")) == _match_key(item.get("current_translation"))
            for item in occurrences
        )
    )
    if preserve_evidence:
        reasons.append("preserve_evidence")

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
    if "preserve_evidence" in reasons:
        return STATUS_PRESERVE_EVIDENCE, reasons
    if reasons:
        return STATUS_AMBIGUOUS, reasons
    return STATUS_CONSISTENT, reasons


def _build_history_evidence(
    *,
    candidate_source: str,
    candidate_target: str,
    matched_occurrences: Sequence[Mapping[str, Any]],
    interpolation_only: bool,
    interpolation_match_ignored: bool,
    source_changed_during_scan: bool,
) -> dict[str, Any]:
    """Render one stable evidence record from already collected matches."""

    status, reason_codes = _history_status(
        matched_occurrences,
        candidate_source=candidate_source,
        candidate_target=candidate_target,
        interpolation_only=interpolation_only,
        interpolation_match_ignored=interpolation_match_ignored,
    )
    if source_changed_during_scan:
        reason_codes.append("history_source_changed")
        status = STATUS_AMBIGUOUS if status == STATUS_CONSISTENT else status

    reported_occurrences = list(matched_occurrences[:MAX_REPORTED_OCCURRENCES])
    first = dict(reported_occurrences[0]) if reported_occurrences else None
    return {
        "schema_version": HISTORY_EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "review_required": status != STATUS_CONSISTENT,
        "candidate_source": candidate_source,
        "candidate_target": candidate_target,
        "match_count": len(matched_occurrences),
        "occurrence_count": len({item.get("occurrence_id") for item in matched_occurrences}),
        "first_occurrence": first,
        "translations": _unique_translations(matched_occurrences),
        "conflict_codes": list(dict.fromkeys(reason_codes)),
        "conflict_reasons": [_reason(code) for code in dict.fromkeys(reason_codes)],
        "occurrences": reported_occurrences,
    }


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
    matched_occurrences, interpolation_only, interpolation_match_ignored = (
        _collect_matches_for_source(source, corpus_items)
    )
    return _build_history_evidence(
        candidate_source=source,
        candidate_target=target,
        matched_occurrences=matched_occurrences,
        interpolation_only=interpolation_only,
        interpolation_match_ignored=interpolation_match_ignored,
        source_changed_during_scan=source_changed_during_scan,
    )


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
    if unavailable_reason:
        for candidate in candidates:
            row = dict(candidate)
            row["history_evidence"] = unavailable_history_evidence(
                candidate,
                reason_code=unavailable_reason,
            )
            enriched.append(row)
        return enriched

    candidate_list = list(candidates)
    matched_by_candidate, interpolation_only, interpolation_match_ignored = (
        _collect_matches_for_candidates(candidate_list, corpus_items)
    )
    for index, candidate in enumerate(candidate_list):
        row = dict(candidate)
        source = _compact_text(candidate.get("source"))
        target = _compact_text(candidate.get("suggested_target"))
        row["history_evidence"] = _build_history_evidence(
            candidate_source=source,
            candidate_target=target,
            matched_occurrences=matched_by_candidate[index],
            interpolation_only=interpolation_only[index],
            interpolation_match_ignored=interpolation_match_ignored[index],
            source_changed_during_scan=source_changed_during_scan,
        )
        enriched.append(row)
    return enriched


def load_corpus_items(path: str) -> list[dict[str, Any]]:
    """Load revision-corpus rows through the #320 reader API."""

    from revision_corpus import load_corpus_items as load_revision_corpus_items

    return load_revision_corpus_items(path)
