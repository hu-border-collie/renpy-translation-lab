# -*- coding: utf-8 -*-
import json
import math
import os
import re


STORY_GRAPH_SCHEMA_VERSION = 1
STORY_HIT_CATEGORIES = ("characters", "relations", "terms", "scenes")


class _NormalizedStoryGraph(dict):
    pass


def _empty_graph():
    return _NormalizedStoryGraph({
        "characters": {},
        "relations": [],
        "terms": [],
        "scenes": [],
    })


def _normalize_rel_path(value):
    if not value:
        return ""
    text = str(value).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    text = text.lstrip("/")
    return text


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _has_string_content(value):
    return isinstance(value, str) and bool(_clean_text(value))


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_character_map(value):
    characters = {}
    if isinstance(value, dict):
        iterable = value.items()
    elif isinstance(value, list):
        iterable = []
        for item in value:
            if isinstance(item, dict):
                char_id = item.get("id") or item.get("key") or item.get("name")
                iterable.append((char_id, item))
    else:
        iterable = []

    for char_id, raw_data in iterable:
        char_id = _clean_text(char_id)
        if not char_id:
            continue
        data = dict(raw_data) if isinstance(raw_data, dict) else {}
        speaker_ids = [
            _clean_text(item)
            for item in _as_list(data.get("speaker_ids"))
            if _clean_text(item)
        ]
        data["speaker_ids"] = speaker_ids
        characters[char_id] = data
    return characters


def _normalize_terms(value):
    terms = []
    if isinstance(value, dict):
        iterable = [
            {"source": source, "target": target}
            for source, target in value.items()
        ]
    elif isinstance(value, list):
        iterable = value
    else:
        iterable = []

    for item in iterable:
        if isinstance(item, dict):
            source = _clean_text(item.get("source") or item.get("term"))
            target = _clean_text(item.get("target") or item.get("translation"))
            note = _clean_text(item.get("note"))
            aliases = [
                _clean_text(alias)
                for alias in _as_list(item.get("aliases"))
                if _clean_text(alias)
            ]
            if source or target or aliases:
                normalized = dict(item)
                normalized["source"] = source
                normalized["target"] = target
                normalized["note"] = note
                normalized["aliases"] = aliases
                terms.append(normalized)
        else:
            source = _clean_text(item)
            if source:
                terms.append({"source": source, "target": "", "note": "", "aliases": []})
    return terms


def _normalize_dict_list(value):
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _is_int_like(value):
    if isinstance(value, bool):
        return False
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return str(value).strip() == str(int(value))


def _is_numeric_like(value):
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _field_label(*parts):
    return ".".join(str(part) for part in parts if part != "")


def _validate_alias_field(value, label, warnings):
    if value is None:
        return
    if isinstance(value, str):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, str):
                warnings.append(f"{label}[{index}] should be a string")
        return
    warnings.append(f"{label} should be a string or list of strings")


def _validate_characters(value, warnings):
    if value is None:
        return
    if isinstance(value, dict):
        iterable = [(char_id, data, None) for char_id, data in value.items()]
    elif isinstance(value, list):
        iterable = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                warnings.append(f"characters[{index}] should be an object")
                continue
            char_id = item.get("id") or item.get("key") or item.get("name")
            iterable.append((char_id, item, f"characters[{index}]"))
    else:
        warnings.append("characters should be an object keyed by character id")
        return

    for char_id, data, source_label in iterable:
        clean_id = _clean_text(char_id)
        label = source_label or _field_label("characters", clean_id or "<empty>")
        if not clean_id:
            warnings.append(f"{label} is missing a usable id")
        if not isinstance(data, dict):
            warnings.append(f"{label} should be an object")
            continue
        _validate_alias_field(data.get("speaker_ids"), f"{label}.speaker_ids", warnings)
        _validate_alias_field(data.get("aliases"), f"{label}.aliases", warnings)


def _validate_relations(value, warnings):
    if value is None:
        return
    if not isinstance(value, list):
        warnings.append("relations should be a list")
        return
    for index, item in enumerate(value):
        label = f"relations[{index}]"
        if not isinstance(item, dict):
            warnings.append(f"{label} should be an object")
            continue
        if not _has_string_content(item.get("left")):
            warnings.append(f"{label}.left should be a non-empty string")
        if not _has_string_content(item.get("right")):
            warnings.append(f"{label}.right should be a non-empty string")
        if item.get("confidence") is not None:
            if not _is_numeric_like(item.get("confidence")):
                warnings.append(f"{label}.confidence should be a finite number from 0 to 1")
            else:
                confidence = float(item.get("confidence"))
                if confidence < 0 or confidence > 1:
                    warnings.append(f"{label}.confidence should be between 0 and 1")


def _validate_terms(value, warnings):
    if value is None:
        return
    if isinstance(value, dict):
        for source, target in value.items():
            if not _has_string_content(source):
                warnings.append("terms object keys should be non-empty strings")
            if target is not None and not isinstance(target, str):
                warnings.append(f"terms.{source} should be a string")
        return
    if not isinstance(value, list):
        warnings.append("terms should be a list or object")
        return
    for index, item in enumerate(value):
        label = f"terms[{index}]"
        if isinstance(item, str):
            continue
        if not isinstance(item, dict):
            warnings.append(f"{label} should be an object or string")
            continue
        aliases = item.get("aliases")
        _validate_alias_field(aliases, f"{label}.aliases", warnings)
        has_term_content = False
        for key in ("source", "term", "target", "translation"):
            if key not in item or item.get(key) is None:
                continue
            if isinstance(item.get(key), str):
                has_term_content = has_term_content or bool(_clean_text(item.get(key)))
            else:
                warnings.append(f"{label}.{key} should be a string")
        has_term_content = has_term_content or any(
            _has_string_content(alias) for alias in _as_list(aliases)
        )
        if not has_term_content:
            warnings.append(
                f"{label} should define source, term, target, translation, or aliases"
            )


def _validate_scenes(value, warnings):
    if value is None:
        return
    if not isinstance(value, list):
        warnings.append("scenes should be a list")
        return
    for index, item in enumerate(value):
        label = f"scenes[{index}]"
        if not isinstance(item, dict):
            warnings.append(f"{label} should be an object")
            continue
        for key in ("line_start", "line_end"):
            if item.get(key) is not None and not _is_int_like(item.get(key)):
                warnings.append(f"{label}.{key} should be an integer")
        line_start = _safe_int(item.get("line_start"))
        line_end = _safe_int(item.get("line_end"))
        if line_start is not None and line_start < 1:
            warnings.append(f"{label}.line_start should be >= 1")
        if line_end is not None and line_end < 1:
            warnings.append(f"{label}.line_end should be >= 1")
        if line_start is not None and line_end is not None and line_start > line_end:
            warnings.append(f"{label}.line_start should be <= line_end")
        if not _has_string_content(item.get("file_rel_path")):
            warnings.append(f"{label}.file_rel_path should be a non-empty string")
        _validate_alias_field(item.get("characters"), f"{label}.characters", warnings)
        if not (
            _normalize_rel_path(item.get("file_rel_path"))
            or _clean_text(item.get("summary"))
            or any(_clean_text(char) for char in _as_list(item.get("characters")))
        ):
            warnings.append(f"{label} should include file_rel_path, summary, or characters")


def validate_story_graph(raw_graph):
    """Return non-fatal schema warnings for a story graph JSON object."""
    if _is_normalized_story_graph(raw_graph):
        return []
    if not isinstance(raw_graph, dict):
        return ["root should be a JSON object"]

    warnings = []
    version = raw_graph.get("schema_version")
    if version is None:
        warnings.append(
            f"schema_version is required and should be {STORY_GRAPH_SCHEMA_VERSION}"
        )
    elif version != STORY_GRAPH_SCHEMA_VERSION:
        warnings.append(
            f"schema_version should be {STORY_GRAPH_SCHEMA_VERSION}"
        )

    _validate_characters(raw_graph.get("characters"), warnings)
    _validate_relations(raw_graph.get("relations"), warnings)
    _validate_terms(raw_graph.get("terms"), warnings)
    _validate_scenes(raw_graph.get("scenes"), warnings)
    return warnings


def _is_normalized_story_graph(value):
    return isinstance(value, _NormalizedStoryGraph)


def normalize_story_graph(raw_graph):
    if not isinstance(raw_graph, dict):
        return _empty_graph()
    if _is_normalized_story_graph(raw_graph):
        return raw_graph
    return _NormalizedStoryGraph({
        "characters": _normalize_character_map(raw_graph.get("characters")),
        "relations": _normalize_dict_list(raw_graph.get("relations")),
        "terms": _normalize_terms(raw_graph.get("terms")),
        "scenes": _normalize_dict_list(raw_graph.get("scenes")),
    })


def load_story_graph(path):
    if not path or not os.path.isfile(path):
        return _empty_graph()
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            raw_graph = json.load(handle) or {}
        for warning in validate_story_graph(raw_graph):
            print(f"Warning: Story graph {path}: {warning}")
        return normalize_story_graph(raw_graph)
    except Exception as exc:
        print(f"Warning: Failed to load story graph {path}: {exc}")
        return _empty_graph()


def _texts_from_items(items):
    texts = []
    for item in items or []:
        if isinstance(item, dict):
            text = _clean_text(item.get("text"))
        else:
            text = _clean_text(item)
        if text:
            texts.append(text)
    return texts


def _combined_search_text(target_items, context_past, context_future):
    parts = []
    parts.extend(_texts_from_items(context_past))
    parts.extend(_texts_from_items(target_items))
    parts.extend(_texts_from_items(context_future))
    return "\n".join(parts)


def _speaker_ids_from_items(target_items):
    speaker_ids = set()
    for item in target_items or []:
        if not isinstance(item, dict):
            continue
        for key in ("speaker", "speaker_id", "speaker_name", "character", "who"):
            value = _clean_text(item.get(key))
            if value:
                speaker_ids.add(value.lower())
    return speaker_ids


def _target_line_span(target_items):
    lines = []
    for item in target_items or []:
        if not isinstance(item, dict):
            continue
        line_start = _safe_int(item.get("line_start"))
        line_end = _safe_int(item.get("line_end"))
        line_number = _safe_int(item.get("line_number"))
        zero_based_line = _safe_int(item.get("line"))
        if line_start is not None:
            lines.append(line_start)
        if line_end is not None:
            lines.append(line_end)
        elif line_number is not None:
            lines.append(line_number)
        elif zero_based_line is not None:
            lines.append(zero_based_line + 1)
    if not lines:
        return None, None
    return min(lines), max(lines)


def _looks_word_like(text):
    return bool(re.fullmatch(r"[A-Za-z0-9_ -]+", text or ""))


def _text_contains_alias(search_text_lower, alias):
    alias = _clean_text(alias)
    if len(alias) <= 1:
        return False
    alias_lower = alias.lower()
    if _looks_word_like(alias):
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(alias_lower) + r"(?![A-Za-z0-9_])"
        return re.search(pattern, search_text_lower) is not None
    return alias_lower in search_text_lower


def _character_aliases(char_id, data):
    aliases = [char_id]
    if isinstance(data, dict):
        aliases.extend(_as_list(data.get("speaker_ids")))
        aliases.extend(_as_list(data.get("zh_name")))
        aliases.extend(_as_list(data.get("name")))
        aliases.extend(_as_list(data.get("aliases")))
    cleaned = []
    seen = set()
    for alias in aliases:
        text = _clean_text(alias)
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


def _term_aliases(term):
    aliases = []
    if isinstance(term, dict):
        aliases.extend(_as_list(term.get("source")))
        aliases.extend(_as_list(term.get("target")))
        aliases.extend(_as_list(term.get("aliases")))
    cleaned = []
    seen = set()
    for alias in aliases:
        text = _clean_text(alias)
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


def _scene_line_span(scene):
    line_start = _safe_int(scene.get("line_start"))
    line_end = _safe_int(scene.get("line_end"))
    if line_start is None and line_end is None:
        return None, None
    if line_start is None:
        line_start = line_end
    if line_end is None:
        line_end = line_start
    if line_start > line_end:
        line_start, line_end = line_end, line_start
    return line_start, line_end


def _line_overlap_score(scene_start, scene_end, target_start, target_end):
    if scene_start is None or target_start is None:
        return 0
    if scene_start <= target_end and target_start <= scene_end:
        overlap = min(scene_end, target_end) - max(scene_start, target_start) + 1
        return 100 + max(0, overlap)
    if scene_end < target_start:
        distance = target_start - scene_end
    else:
        distance = scene_start - target_end
    return max(0, 40 - min(distance, 40))


def _rank_matching_scenes(story_graph, file_rel_path, target_start, target_end, active_char_ids):
    normalized_file = _normalize_rel_path(file_rel_path)
    ranked = []
    for scene in story_graph.get("scenes", []):
        scene_file = _normalize_rel_path(scene.get("file_rel_path"))
        if normalized_file and scene_file and scene_file != normalized_file:
            continue
        score = 0
        if normalized_file and scene_file == normalized_file:
            score += 50
        scene_start, scene_end = _scene_line_span(scene)
        score += _line_overlap_score(scene_start, scene_end, target_start, target_end)
        scene_chars = {
            _clean_text(char)
            for char in _as_list(scene.get("characters"))
            if _clean_text(char)
        }
        score += 8 * len(scene_chars & active_char_ids)
        if score > 0:
            ranked.append((score, scene))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [scene for _, scene in ranked]


def _collect_active_characters(story_graph, search_text_lower, speaker_ids):
    active = set()
    characters = story_graph.get("characters", {})
    for char_id, data in characters.items():
        aliases = _character_aliases(char_id, data)
        if any(alias.lower() in speaker_ids for alias in aliases):
            active.add(char_id)
            continue
        if any(_text_contains_alias(search_text_lower, alias) for alias in aliases):
            active.add(char_id)
    return active


def retrieve_story_hits(
    story_graph,
    file_rel_path,
    target_items,
    context_past=None,
    context_future=None,
    top_k_relations=6,
    top_k_terms=12,
    include_scene_summary=True,
):
    story_graph = normalize_story_graph(story_graph)
    search_text = _combined_search_text(target_items, context_past, context_future)
    search_text_lower = search_text.lower()
    speaker_ids = _speaker_ids_from_items(target_items)
    target_start, target_end = _target_line_span(target_items)
    active_char_ids = _collect_active_characters(story_graph, search_text_lower, speaker_ids)

    matching_scenes = _rank_matching_scenes(
        story_graph,
        file_rel_path,
        target_start,
        target_end,
        active_char_ids,
    )
    for scene in matching_scenes[:3]:
        for char_id in _as_list(scene.get("characters")):
            char_id = _clean_text(char_id)
            if char_id:
                active_char_ids.add(char_id)

    characters = []
    for char_id in sorted(active_char_ids):
        data = story_graph.get("characters", {}).get(char_id, {})
        item = {"id": char_id}
        if isinstance(data, dict):
            for key in ("zh_name", "style", "note"):
                value = _clean_text(data.get(key))
                if value:
                    item[key] = value
        characters.append(item)

    term_hits = []
    max_terms = max(0, int(top_k_terms or 0))
    for term in story_graph.get("terms", []):
        if len(term_hits) >= max_terms:
            break
        aliases = _term_aliases(term)
        if any(_text_contains_alias(search_text_lower, alias) for alias in aliases):
            term_hits.append(term)

    relation_hits = []
    for relation in story_graph.get("relations", []):
        left = _clean_text(relation.get("left"))
        right = _clean_text(relation.get("right"))
        if not left or not right:
            continue
        left_active = left in active_char_ids
        right_active = right in active_char_ids
        if not (left_active and right_active):
            continue
        relation_hits.append(relation)

    def relation_confidence(item):
        try:
            value = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    relation_hits.sort(key=relation_confidence, reverse=True)

    scene_hits = matching_scenes[:3] if include_scene_summary else []
    return {
        "characters": characters,
        "relations": relation_hits[:max(0, int(top_k_relations or 0))],
        "terms": term_hits,
        "scenes": scene_hits,
    }


def _format_confidence(value):
    try:
        return f" confidence={float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _append_limited(lines, line, max_chars):
    line = _clean_text(line)
    if not line:
        return False
    candidate = "\n".join(lines + [line]) if lines else line
    if len(candidate) <= max_chars:
        lines.append(line)
        return True
    if not lines and max_chars > 3:
        lines.append(line[:max_chars - 3].rstrip() + "...")
    return False


def has_story_hits(story_hits):
    if not isinstance(story_hits, dict):
        return False
    for key in STORY_HIT_CATEGORIES:
        hits = story_hits.get(key)
        if isinstance(hits, list) and hits:
            return True
    return False


def story_hit_counts(story_hits):
    counts = {}
    if not isinstance(story_hits, dict):
        return {key: 0 for key in STORY_HIT_CATEGORIES}
    for key in STORY_HIT_CATEGORIES:
        hits = story_hits.get(key)
        counts[key] = len(hits) if isinstance(hits, list) else 0
    return counts


def format_story_hits_block(story_hits, max_chars, empty_label="(none)"):
    if not isinstance(story_hits, dict):
        return empty_label
    max_chars = max(1, int(max_chars or 1))
    lines = []

    characters = story_hits.get("characters") or []
    if characters:
        _append_limited(lines, "Characters:", max_chars)
        for item in characters:
            char_id = _clean_text(item.get("id"))
            zh_name = _clean_text(item.get("zh_name"))
            style = _clean_text(item.get("style"))
            note = _clean_text(item.get("note"))
            label = char_id
            if zh_name:
                label = f"{label} ({zh_name})"
            detail = style or note
            line = f"- {label}: {detail}" if detail else f"- {label}"
            if not _append_limited(lines, line, max_chars):
                break

    relations = story_hits.get("relations") or []
    if relations:
        _append_limited(lines, "Relations:", max_chars)
        for item in relations:
            left = _clean_text(item.get("left"))
            right = _clean_text(item.get("right"))
            relation_type = _clean_text(item.get("type"))
            note = _clean_text(item.get("note"))
            detail = relation_type or "related"
            confidence = _format_confidence(item.get("confidence"))
            line = f"- {left} -> {right} [{detail}{confidence}]"
            if note:
                line += f": {note}"
            if not _append_limited(lines, line, max_chars):
                break

    terms = story_hits.get("terms") or []
    if terms:
        _append_limited(lines, "Terms:", max_chars)
        for item in terms:
            source = _clean_text(item.get("source"))
            target = _clean_text(item.get("target"))
            note = _clean_text(item.get("note"))
            if source and target and source != target:
                line = f"- {source} -> {target}"
            elif source:
                line = f"- Term: {source}"
            elif target:
                line = f"- Term target: {target}"
            else:
                continue
            if note:
                line += f": {note}"
            if not _append_limited(lines, line, max_chars):
                break

    scenes = story_hits.get("scenes") or []
    if scenes:
        _append_limited(lines, "Scenes:", max_chars)
        for item in scenes:
            file_rel_path = _normalize_rel_path(item.get("file_rel_path"))
            line_start, line_end = _scene_line_span(item)
            summary = _clean_text(item.get("summary"))
            chars = [
                _clean_text(char)
                for char in _as_list(item.get("characters"))
                if _clean_text(char)
            ]
            location = file_rel_path or "scene"
            if line_start is not None and line_end is not None:
                location = f"{location}:{line_start}-{line_end}"
            line = f"- {location}"
            details = []
            if summary:
                details.append(summary)
            if chars:
                details.append("characters: " + ", ".join(chars))
            if details:
                line += ": " + "; ".join(details)
            if not _append_limited(lines, line, max_chars):
                break

    if not lines:
        return empty_label[:max_chars]
    return "\n".join(lines)[:max_chars]
